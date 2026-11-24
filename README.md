# 端侧推理 · 内存试跑（Flask）

把一份推理计划（输入张量、算子顺序、要保留的输出、缓冲容量）放进**一块定长
float32 缓冲**里模拟执行，观察每一步的分配、引用计数回收、碎片整理与占用峰值。

- 输入、中间值、输出全部驻留同一块缓冲区，搬移（compaction）也在其中完成。
- `add`、`relu`：为结果分配**独立的连续空间**。
- `slice`：**共享来源张量的整块存储**的视图，不额外占用元素。
- 引用计数：来源张量与视图都释放后，整块存储才回收；指定输出保留到执行结束。
- 分配器按“首次适配 + 空闲区合并”：若单块空闲不够但**总空闲量足够**，就把存活
  块向缓冲区头部紧凑搬移（视图随所属存储一起移动，相对偏移不变）后重试。
- 容量确实不够时返回**失败位置、所需空间、当时总/最大空闲量与仍在使用的值**。
- 成功时返回各输出值、缓冲占用峰值与逐步时间线。
- **低峰值试跑**（可选，最多 12 个算子）：输入仍按填写顺序装载，仅在所有符合
  数据依赖的算子顺序里用动态规划选出**缓冲峰值最低**的一条；选序只读取计划
  元数据（长度、依赖、输出），切片同源只计一份、输出保留到结束。容量足够时按
  选出的顺序执行一次，容量不足则跳过执行、只返回试算结果。

## 运行（Docker Compose）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

`MEMDRY_PORT` 可改宿主机端口：`MEMDRY_PORT=8080 docker compose up --build`。

服务用 gunicorn（2 worker × 4 线程）跑 `memdry.app:app`，容器只读、去权、带
healthcheck（`GET /healthz`）。镜像基于 `python:3.12-slim`，依赖只有 Flask 与
gunicorn（`requirements.txt` 已锁版本）。

> 本机验证环境未安装 Docker（`docker` 命令不存在），因此 compose 文件与 Dockerfile
> 只做了静态校验和同参数本地 gunicorn 试跑，详见“实测结果 / 未完成项”。

## 本机直接跑（等价路径，用于验证）

```bash
python3 -m pip install -r requirements.txt
python3 -m gunicorn --bind 127.0.0.1:8000 --workers 2 --threads 4 \
  --no-control-socket --worker-tmp-dir /tmp memdry.app:app
# 或开发模式：PORT=8000 python3 -m memdry.app
```

## 输入格式

页面左侧四个字段，对应 `POST /api/run` 的 JSON：

| 字段 | 含义 |
| --- | --- |
| `capacity` | 缓冲容量，按 float32 元素计（1 ≤ n ≤ 1,000,000） |
| `inputs` | 每行 `名称: v1, v2, ...`，逗号或空格分隔 |
| `ops` | 每行一个算子：`r = add(a, b)` / `r = relu(x)` / `v = slice(x, 起始, 结束)` |
| `outputs` | 逗号分隔的输出张量名，驻留到执行结束 |
| `low_peak` | 可选布尔（`true`/`false`，也接受 `1/on/yes`）。开启后启用低峰值选序，算子数须 ≤ 12 |

规则：`add` 要求两操作数等长；`slice` 必须是步长 1 的连续切片，下标语义与 Python
切片一致（支持负数、省略端点，结果不能为空）；只能引用输入或更早的算子结果。

```json
{
  "capacity": "24",
  "inputs": "X: 1, -2, 3, -4\nA: 10, -20, 30, -40, 50, -60\nY: -1, -2, 6, 7, -8, 9",
  "ops": "v = slice(A, 1, 4)\np = relu(X)\nq = relu(Y)",
  "outputs": "v, p, q"
}
```

## 页面上的四个示例

- **空闲区分散 → 整理**：容量 24，四个 relu 结果与输入交替释放，`w` 产生前只剩
  4+4 两段空闲；整理后最大连续空闲 4 → 8，分配成功。
- **保留切片视图**：`A` 的切片 `v` 在 `A` 释放后仍有效；随后 `q` 的结果分配触发
  整理，`A` 的整块存储连同视图从偏移 4 搬到偏移 0，输出值保持不变。
- **容量不足**：容量 10 时 `c = add(a, b)` 需要 4 个连续元素、空闲只有 2 个，
  页面给出失败步骤（第 3 步）、行号、所需/可用空间与仍占用的 `a, b`。
- **低峰值选序**：容量 21，原文顺序先算 8 元素的 `x = relu(A)`，峰值 24 会超容；
  开启低峰值后选序把 `y = add(B, C)` 链提前，`B、C` 先释放，峰值降到 20 即可跑完。
  把容量改成 15 再试：选序结果仍展示，但因 15 < 20 跳过实际执行。

## 低峰值选序怎么算

- 固定缓冲下，分配器在**总空闲足够**时总能整理成功，因此某次分配失败当且仅当
  “存活元素 + 本次申请长度 > 容量”；一条执行顺序所需的最小容量就是该式在全程
  的最大值，**只由张量长度和存活区间决定，与数值无关**。
- 存活规则与执行器完全一致：操作数在**最后一次使用**后回收（输出除外）；切片
  视图让来源的**整块存储**保活，同一块存储在任意时刻**只计一次**；指定输出保留
  到结束；输入始终按填写顺序装载，不参与重排。
- 算子数 `n ≤ 12` 时，在 `2^n` 个“已完成算子集合”上做精确动态规划
  （`best(S∪{i}) = min(max(live(S)+alloc(i), best(S∪{i})))`），从所有满足依赖的
  拓扑序里取峰值最低者；原顺序已最优时保持原顺序（同峰值取字典序最小）。
- 重排只改变存储的创建/回收时机，不改变任何计算，因此容量充足时输出的 float32
  结果与原顺序逐元素一致（由随机对照测试覆盖）。

## 接口

`POST /api/run` → 200 + 报告 JSON（容量不足也是 200，`ok=false` 并带 `failure`；
格式/语义错误返回 400 + `errors`）。未开启 `low_peak` 时响应结构与原来完全一致。

开启 `low_peak` 后额外返回 `mode: "low_peak"`、`executed` 与 `low_peak` 对象：

- `low_peak.order[]`：选出的执行顺序，每项含执行位 `position`、原文算子位
  `original_position`、可读 `label`、算子类型与行号；
- `low_peak.min_elements`：选序后**最少需要的元素数**；`original_peak` 为原顺序
  峰值；`input_peak` 为输入装载阶段峰值；`saved` 为节省量；`reordered` 表示是否
  发生重排；`capacity_sufficient` 表示当前容量是否够跑。
- 容量足够：`executed=true`，按选序执行一次，响应仍含完整的 `outputs/steps/
  layout/stats/peak` 等执行器字段。
- 容量不足：`executed=false`、`ok=false`，`steps/outputs/layout` 为空，仅返回
  选序与 `min_elements`（以及 `skip_reason`），不触碰缓冲执行。

主要字段：`ok`、`outputs[]`（值、角色 `input/result/view`、存储号与偏移、`slice_of`）、
`peak {elements, percent, at}`、`final_occupancy`、`stats {allocations, compactions,
moved_elements, free_total, largest_free, overflow_values}`、`steps[]`（每步的
`allocated / freed / view / compaction / occupancy`）、`layout[]`（当前存储与空闲区，
含视图相对区间）、`failure`、`warnings`。

## 代码结构

```
memdry/model.py    计划解析 + 静态校验（长度、切片边界、名称、上限）
memdry/memory.py   定长缓冲分配器（首次适配、合并、整理）+ 执行器（引用计数、视图、可选执行顺序）
memdry/schedule.py 低峰值选序：元数据存活分析 + 2^n 子集动态规划（≤ 12 算子）
memdry/app.py      Flask 路由：/、/healthz、POST /api/run（含 low_peak 分支）
templates/index.html, static/{app.js,style.css}   单页界面（含低峰值选序面板）
tests/             pytest：分配器、执行器、选序器、解析与 HTTP
```

## 测试

```bash
python3 -m pip install pytest
python3 -m pytest -q          # 46 passed
```

## 实测结果

本机 Python 3.11 + Flask 3.1.3 + gunicorn 27.0.1。

| 场景 | 结果 |
| --- | --- |
| `pytest` | 46 passed（新增选序器与低峰值 HTTP 用例 15 个） |
| 空闲区分散示例 | 峰值 22/24（91.67%），整理 1 次、搬移 10 元素，最大连续空闲 4→8，四个输出正确 |
| 保留切片 + 整理 | 峰值 22/24，整理 1 次、搬移 16 元素，视图从偏移 5 移到 1，值 `-20, 30, -40` 不变 |
| 容量不足 | `ok=false`，失败于第 3 步 `c = add(a, b)`，需要 4、空闲 2、最大连续 2，`live=[a, b]` |
| 视图共享/引用计数 | `v=slice(x,1,4)` 时 `x` 释放但存储保留；视图释放后才归还整块 |
| 随机对照（2977 个有效计划） | 与无缓冲参考实现的逐元素结果一致；布局无重叠、占用/空闲守恒 |
| 分配器随机压测（500 轮 × 80 次操作） | 搬移后内容不变、无重叠、失败仅当总空闲不足 |
| 低峰值选序穷举对照（120 个 ≤6 算子计划） | DP 的 `min_elements` 与全部合法拓扑序暴力枚举的最小值逐一相等 |
| 选序-执行器随机对照（3000 个计划，算子 ≤12） | 紧容量（=min）必成功且峰值等于 `min_elements`；少 1 元素必失败；重排后输出与原顺序 float32 逐元素一致；1708 例存在更优顺序 |
| 切片重排留存（83 个视图输出） | 切片算子被移动后角色仍为 `view`、来源整块存储保活、值不变 |
| 12 算子选序耗时 | 约 132 ms（本机 Python 3.11） |
| 原顺序超容、重排可跑完 | 容量 21：原顺序峰值 24 失败（第 4 步，需 8、空闲 5）；选序峰值 20 成功，输出正确 |
| 容量不足跳过执行 | 容量 15：`executed=false`、无 steps/outputs，仍返回选序与最少元素 20 |
| gunicorn 双 worker 试跑 | `/healthz` 200；页面与两个静态资源 200；普通/低峰值/跳过/超算子上限路径均正常 |
| 前端 jsdom 冒烟 | 真实接口驱动：选序面板、移动高亮、跳过态隐藏执行区块、关闭选项回到普通失败报告，均通过 |

## 未完成项与已知限制

- **未在真实 Docker 中构建运行**：当前环境没有 `docker`/`podman`。已按 compose
  参数（2 worker、`--no-control-socket`、只读根文件系统、tmpfs `/tmp`）在本地
  gunicorn 上跑通，并对 YAML 做了解析校验；首次 `docker compose up --build` 仍需
  在有 Docker 的机器上确认。compose/Dockerfile 入口未改动，新增代码随镜像一起打包。
- **未做浏览器端视觉核对**：环境没有 Chromium/Firefox。前端用 jsdom 驱动真实接口
  做了 DOM 级冒烟测试（含低峰值选序面板与跳过态），但没有渲染截图。
- **低峰值选序上限 12 个算子**：精确 DP 的状态数为 `2^n`，超过 12 个会返回 400
  提示关闭该选项或减少算子；普通试跑（不开启选序）仍支持最多 500 个算子。
- 模拟只覆盖 `add`、`relu`、步长 1 的连续切片；不支持广播、跨步/负步长切片、
  多输出算子、非 float32 类型。
- 加法按 `array('f')` 的 IEEE-754 binary32 做逐项运算；极少数情况下中间舍入与硬件
  FMA 的实现可能相差 1 ulp。溢出记作 `inf` 并在提示中计数。
- 分配策略是确定性的首次适配＋全量整理；不模拟真实硬件对齐、内存池或分页开销，
  峰值统计的是元素数而非字节。
- 并发/多用户未做隔离与限流；服务端每次请求独立计算，无状态。
