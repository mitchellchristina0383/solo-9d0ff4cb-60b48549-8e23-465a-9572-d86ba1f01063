"""Peak-minimising operator scheduler driven purely by plan metadata.

For a fixed buffer the allocator can always compact when the *total* free space
is large enough, so an allocation fails exactly when

    live elements + requested length > capacity

which makes the peak occupancy of a run the maximum of that expression.  The
required capacity of an execution order is therefore a pure function of tensor
lengths and lifetimes -- no values are read.

Reordering cannot change any computed value; it only changes when storages are
created and released.  The executor releases an operand after its *last* use
(unless it is a requested output), keeps a slice view's whole source storage
alive, counts a shared storage once, and holds outputs until the end.  This
module mirrors those rules.

Up to 12 ops are supported: the problem is solved exactly with a dynamic
program over the 2**n subsets of completed ops.  After a prefix set ``S`` the
set of live handles -- and therefore the live element count -- depends only on
``S``, not on the order that reached it, so the DP below is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .model import Plan, ValidationError

MAX_SCHEDULE_OPS = 12

@dataclass(frozen=True)
class Schedule:
    """Result of the low-peak ordering search."""

    order: tuple[int, ...]  # indices into plan.ops
    min_elements: int
    original_peak: int
    input_peak: int
    ops_count: int

    @property
    def reordered(self) -> bool:
        return self.order != tuple(range(self.ops_count))

    @property
    def saved(self) -> int:
        return max(0, self.original_peak - self.min_elements)


@dataclass(frozen=True)
class _TensorMeta:
    name: str
    length: int
    producer: int  # op index, or -1 for an input
    is_output: bool
    consumer_mask: int
    base: str  # name of the non-view root whose storage is shared


def _build_meta(plan: Plan) -> dict[str, _TensorMeta]:
    """Describe every tensor's producer, consumers and shared root storage."""
    ops_len = len(plan.ops)
    producer: dict[str, int] = {spec.name: -1 for spec in plan.inputs}
    length: dict[str, int] = {spec.name: spec.length for spec in plan.inputs}
    for index, op in enumerate(plan.ops):
        producer[op.name] = index
        length[op.name] = op.length

    base: dict[str, str] = {spec.name: spec.name for spec in plan.inputs}
    for op in plan.ops:
        base[op.name] = base[op.operands[0]] if op.kind == "slice" else op.name

    consumer_mask: dict[str, int] = {name: 0 for name in producer}
    for index, op in enumerate(plan.ops):
        for name in op.operands:
            consumer_mask[name] |= 1 << index

    outputs = set(plan.outputs)
    return {
        name: _TensorMeta(
            name=name,
            length=length[name],
            producer=producer[name],
            is_output=name in outputs,
            consumer_mask=consumer_mask[name],
            base=base[name],
        )
        for name in producer
    }


class _Liveness:
    """Live element count for any set of completed ops (bitmask).

    A shared storage is counted once while at least one handle that refers to
    it (the source or any slice view) is alive, which mirrors the executor's
    reference counting.  Counts for all masks are precomputed because the
    scheduler queries them for every DP state.
    """

    def __init__(self, plan: Plan):
        self.plan = plan
        self.meta = _build_meta(plan)
        self.full = (1 << len(plan.ops)) - 1
        # (storage root, its element cost, handles that map onto it)
        roots: dict[str, int] = {}
        for item in self.meta.values():
            roots.setdefault(item.base, self.meta[item.base].length)
        groups: dict[str, list[_TensorMeta]] = {root: [] for root in roots}
        for item in self.meta.values():
            groups[item.base].append(item)
        self.live_table = self._build_table(roots, groups)

    def handle_alive(self, item: _TensorMeta, mask: int) -> bool:
        if item.producer >= 0 and not (mask >> item.producer) & 1:
            return False  # not produced yet
        if item.is_output:
            return True
        # Alive while some consumer is still ahead of us.
        return bool(item.consumer_mask & ~mask)

    def _build_table(
        self, roots: dict[str, int], groups: dict[str, list[_TensorMeta]]
    ) -> list[int]:
        table = [0] * (self.full + 1)
        for mask in range(self.full + 1):
            total = 0
            for root, cost in roots.items():
                if any(self.handle_alive(item, mask) for item in groups[root]):
                    total += cost
            table[mask] = total
        return table

    def live(self, mask: int) -> int:
        return self.live_table[mask]

    def input_peak(self) -> int:
        """Peak while inputs are loaded in the order they were written."""
        occupied = 0
        peak = 0
        for spec in self.plan.inputs:
            occupied += spec.length
            peak = max(peak, occupied)
            if not self.handle_alive(self.meta[spec.name], 0):
                # Unused, non-output input: released right after loading.
                occupied -= spec.length
        return peak


def _dependencies(plan: Plan) -> list[int]:
    producer = {spec.name: -1 for spec in plan.inputs}
    for index, op in enumerate(plan.ops):
        producer[op.name] = index
    masks = []
    for op in plan.ops:
        mask = 0
        for name in op.operands:
            index = producer[name]
            if index >= 0:
                mask |= 1 << index
        masks.append(mask)
    return masks


def _alloc_lengths(plan: Plan) -> list[int]:
    return [0 if op.kind == "slice" else op.length for op in plan.ops]


def peak_for_order(plan: Plan, order: tuple[int, ...], live: _Liveness | None = None) -> int:
    """Peak occupancy of *order*, using only tensor lengths and lifetimes."""
    live = live or _Liveness(plan)
    allocs = _alloc_lengths(plan)
    mask = 0
    peak = live.input_peak()
    for index in order:
        peak = max(peak, live.live(mask) + allocs[index])
        mask |= 1 << index
    peak = max(peak, live.live(mask))
    return peak


def build_schedule(plan: Plan) -> Schedule:
    """Find a dependency-respecting order with the lowest buffer peak."""
    count = len(plan.ops)
    if count > MAX_SCHEDULE_OPS:
        raise ValidationError(
            f"低峰值选序最多支持 {MAX_SCHEDULE_OPS} 个算子（当前 {count} 个）"
        )

    live = _Liveness(plan)
    allocs = _alloc_lengths(plan)
    deps = _dependencies(plan)
    full = live.full
    input_peak = live.input_peak()

    @lru_cache(maxsize=None)
    def best(mask: int) -> int:
        """Minimal peak achievable over the ops that are still ahead of *mask*."""
        if mask == full:
            return live.live(mask)
        result = None
        current = live.live(mask)
        for index in range(count):
            bit = 1 << index
            if mask & bit or deps[index] & ~mask:
                continue
            candidate = max(current + allocs[index], best(mask | bit))
            if result is None or candidate < result:
                result = candidate
        assert result is not None, "plan ops are not topologically ordered"
        return result

    min_elements = max(input_peak, best(0))

    # Reconstruct the lexicographically smallest optimal order.  The original
    # order is the lexicographically smallest valid order, so it is kept
    # whenever it already reaches the minimum.
    order: list[int] = []
    mask = 0
    running = input_peak
    while mask != full:
        current = live.live(mask)
        for index in range(count):
            bit = 1 << index
            if mask & bit or deps[index] & ~mask:
                continue
            candidate = max(running, current + allocs[index])
            if max(candidate, best(mask | bit)) <= min_elements:
                order.append(index)
                running = candidate
                mask |= bit
                break
        else:  # pragma: no cover - defensive, the DP guarantees a choice
            raise AssertionError("no feasible next op during reconstruction")

    return Schedule(
        order=tuple(order),
        min_elements=min_elements,
        original_peak=peak_for_order(plan, tuple(range(count)), live),
        input_peak=input_peak,
        ops_count=count,
    )


def op_label(op) -> str:
    """Human-readable one-line form of an op, matching the input syntax."""
    if op.kind == "slice":
        start = "" if op.start is None else str(op.start)
        stop = "" if op.stop is None else str(op.stop)
        args = f"{op.operands[0]}, {start}, {stop}"
    else:
        args = ", ".join(op.operands)
    return f"{op.name} = {op.kind}({args})"


def describe_order(plan: Plan, order: tuple[int, ...]) -> list[dict]:
    """Order entries for the report, positions are 1-based execution slots."""
    return [
        {
            "position": position,
            "index": index,
            "original_position": index + 1,
            "label": op_label(plan.ops[index]),
            "kind": plan.ops[index].kind,
            "line": plan.ops[index].line,
        }
        for position, index in enumerate(order, 1)
    ]
