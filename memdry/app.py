"""Memory dry run: parse a plan, simulate one fixed float32 buffer, report."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .memory import execute_plan
from .model import ValidationError, parse_plan
from .schedule import (
    MAX_SCHEDULE_OPS,
    build_schedule,
    describe_order,
)

MAX_BODY_BYTES = 2 * 1024 * 1024
TRUE_TOKENS = {"1", "true", "on", "yes", "y"}


def _low_peak_enabled(payload: dict) -> bool:
    raw = payload.get("low_peak", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in TRUE_TOKENS
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config.update(MAX_CONTENT_LENGTH=MAX_BODY_BYTES)
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/api/run")
    def run_plan():
        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "ok": False,
                        "errors": [
                            "请求体必须是 JSON（Content-Type: application/json）"
                        ],
                    }
                ),
                400,
            )
        try:
            low_peak = _low_peak_enabled(payload)
            plan = parse_plan(payload)
            if low_peak and len(plan.ops) > MAX_SCHEDULE_OPS:
                raise ValidationError(
                    f"低峰值选序最多支持 {MAX_SCHEDULE_OPS} 个算子"
                    f"（当前计划有 {len(plan.ops)} 个），请减少算子或关闭该选项"
                )
        except ValidationError as exc:
            return jsonify({"ok": False, "errors": exc.messages}), 400

        if not low_peak:
            return jsonify(execute_plan(plan)), 200

        schedule = build_schedule(plan)
        order_info = {
            "selected": True,
            "max_ops": MAX_SCHEDULE_OPS,
            "ops_count": schedule.ops_count,
            "order": describe_order(plan, schedule.order),
            "original_order": describe_order(
                plan, tuple(range(schedule.ops_count))
            ),
            "min_elements": schedule.min_elements,
            "original_peak": schedule.original_peak,
            "input_peak": schedule.input_peak,
            "saved": schedule.saved,
            "reordered": schedule.reordered,
            "capacity": plan.capacity,
            "capacity_sufficient": plan.capacity >= schedule.min_elements,
        }

        if plan.capacity < schedule.min_elements:
            # Skip execution: the selected order still cannot fit, but the
            # metadata-only trial result (order + required elements) is shown.
            order_info["executed"] = False
            order_info["skip_reason"] = (
                f"缓冲容量 {plan.capacity} 个元素小于选序后最少需要的 "
                f"{schedule.min_elements} 个元素，已跳过实际执行"
            )
            return (
                jsonify(
                    {
                        "ok": False,
                        "executed": False,
                        "mode": "low_peak",
                        "capacity": plan.capacity,
                        "outputs": [],
                        "failure": None,
                        "steps": [],
                        "peak": {
                            "elements": schedule.min_elements,
                            "percent": round(
                                100.0 * schedule.min_elements / plan.capacity, 2
                            ),
                            "at": "低峰值选序试算（未执行）",
                        },
                        "final_occupancy": 0,
                        "final_percent": 0.0,
                        "stats": None,
                        "layout": [],
                        "warnings": list(plan.warnings),
                        "low_peak": order_info,
                    }
                ),
                200,
            )

        result = execute_plan(plan, schedule.order)
        result["executed"] = True
        result["mode"] = "low_peak"
        order_info["executed"] = True
        result["low_peak"] = order_info
        return jsonify(result), 200

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            jsonify(
                {
                    "ok": False,
                    "errors": [
                        f"请求体超过 {MAX_BODY_BYTES // 1024} KiB 上限，请缩短输入张量"
                    ],
                }
            ),
            413,
        )

    @app.errorhandler(Exception)
    def internal_error(exc):
        if isinstance(exc, ValidationError):
            return jsonify({"ok": False, "errors": exc.messages}), 400
        if isinstance(exc, HTTPException):
            if request.path.startswith("/api/"):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "errors": [
                                f"{exc.code} {exc.name}：{request.path} 不接受该请求"
                            ],
                        }
                    ),
                    exc.code,
                )
            return exc
        app.logger.exception("unexpected error")
        return (
            jsonify({"ok": False, "errors": ["服务器内部错误，请检查输入后重试"]}),
            500,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
