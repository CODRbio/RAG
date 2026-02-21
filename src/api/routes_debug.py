"""
Debug API：热切换全局 Debug 模式，查看/清理 Debug 日志。
"""

from fastapi import APIRouter

from src.debug import get_debug_logger

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
