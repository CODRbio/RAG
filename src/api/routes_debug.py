"""
Debug API：热切换全局 Debug 模式，查看/清理 Debug 日志。
"""

from fastapi import APIRouter

from src.debug import get_debug_logger
from src.log.frontend_event_logger import log_frontend_event, log_frontend_events

from src.collaboration.memory.session_memory import load_session_memory, get_session_store

router = APIRouter(tags=["debug"])


@router.get("/debug/memory/{session_id}")
def debug_memory_state(session_id: str) -> dict:
    """查看特定会话的记忆内部状态（Buffer, Summary, Evidence Cache）。"""
    memory = load_session_memory(session_id)
    if not memory:
        return {"ok": False, "error": "session not found"}
        
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    evidence_cache = store.get_recent_evidence_cache(session_id)
    
    # 计算 Buffer 占用
    buffer_turns = memory.get_context_window()
    buffer_chars = sum(len(t.content or "") for t in buffer_turns)
    
    # 统计 Evidence Cache
    cache_stats = []
    pinned_count = 0
    total_hit_count = 0
    for item in evidence_cache:
        is_pinned = item.get("is_pinned", False)
        if is_pinned: pinned_count += 1
        
        turn_hits = sum(c.get("hit_count", 0) for c in item.get("chunks", []))
        total_hit_count += turn_hits
        
        cache_stats.append({
            "query": item.get("query"),
            "timestamp": item.get("timestamp"),
            "is_pinned": is_pinned,
            "chunk_count": len(item.get("chunks", [])),
            "hits": turn_hits
        })
        
    return {
        "ok": True,
        "session_id": session_id,
        "memory": {
            "buffer_turns_count": len(buffer_turns),
            "buffer_chars": buffer_chars,
            "summary_at_turn": memory.summary_at_turn,
            "rolling_summary_len": len(memory.rolling_summary),
            "rolling_summary_preview": memory.rolling_summary[:200] + "..." if len(memory.rolling_summary) > 200 else memory.rolling_summary,
        },
        "evidence_cache": {
            "total_items": len(evidence_cache),
            "pinned_items": pinned_count,
            "total_hits": total_hit_count,
            "items": cache_stats
        },
        "meta": meta
    }


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
