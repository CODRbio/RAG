"""
Debug API：热切换全局 Debug 模式，查看/清理 Debug 日志。
"""

from fastapi import APIRouter

from src.debug import get_debug_logger
from src.log.frontend_event_logger import log_frontend_event, log_frontend_events

router = APIRouter(tags=["debug"])


@router.post("/debug/toggle")
def toggle_debug(body: dict) -> dict:
    """热切换 Debug 模式（无需重启）。"""
    enabled = bool(body.get("enabled", False))
    get_debug_logger().toggle(enabled)
    return {"ok": True, "debug": enabled}


@router.get("/debug/status")
def debug_status() -> dict:
    """查看当前 Debug 模式状态。"""
    dl = get_debug_logger()
    return {
        "debug": dl.enabled,
        "log_dir": str(dl.log_dir),
    }


@router.post("/debug/cleanup")
def debug_cleanup(body: dict | None = None) -> dict:
    """清理过期的 Debug 日志文件。"""
    params = body or {}
    report = get_debug_logger().cleanup(
        max_age_days=int(params.get("max_age_days", 7)),
        max_total_mb=int(params.get("max_total_mb", 200)),
    )
    return {"ok": True, **report}


@router.post("/debug/frontend-log")
def frontend_log(body: dict) -> dict:
    """Write one frontend debug event into logs/frontend."""
    log_frontend_event(body or {})
    return {"ok": True}


@router.post("/debug/frontend-log/batch")
def frontend_log_batch(body: dict | None = None) -> dict:
    """Write frontend debug events in batch into logs/frontend."""
    payload = body or {}
    events = payload.get("events") or []
    if not isinstance(events, list):
        return {"ok": False, "written": 0, "error": "events must be a list"}
    written = log_frontend_events([e for e in events if isinstance(e, dict)])
    return {"ok": True, "written": written}
