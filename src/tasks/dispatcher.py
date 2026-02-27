"""
Unified dispatcher: run Chat tasks from Redis queue with global slot limit.
Chat execution pushes events to Redis for GET /chat/stream/{task_id}.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict

from config.settings import settings
from src.log import get_logger
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int = 80) -> list:
    """Split text into chunks for streaming (same as routes_chat)."""
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def run_chat_task_sync(task_id: str, payload: Dict[str, Any]) -> None:
    """
    Run a single Chat task (sync): load ChatRequest from payload, call _run_chat,
    push meta/delta/done to Redis, update state and release slot.
    """
    from src.api.schemas import ChatRequest
    from src.api.routes_chat import _run_chat, _serialize_citation
    from src.collaboration.memory.session_memory import get_session_store

    q = get_task_queue()
    state = q.get_state(task_id)
    if not state or state.status != TaskStatus.queued:
        return
    session_id = state.session_id or ""
    user_id = state.user_id or ""
    payload = dict(state.payload)
    optional_user_id = payload.pop("_optional_user_id", None)

    state.status = TaskStatus.running
    state.started_at = time.time()
    state.queue_position = 0
    q.set_state(state)

    try:
        body = ChatRequest(**payload)
        session_id_out, response_text, citations, evidence_summary, parsed, dashboard_data, tool_trace_data, ref_map, agent_debug = _run_chat(
            body, optional_user_id
        )
        session_id = session_id_out or session_id
        store = get_session_store()
        current_stage = store.get_session_stage(session_id) or "explore"
        session_meta = store.get_session_meta(session_id) or {}
        canvas_id = (session_meta or {}).get("canvas_id") or ""

        citations_data = [_serialize_citation(c) for c in citations]
        intent_info = {
            "mode": parsed.intent_type.value,
            "intent_type": parsed.intent_type.value,
            "confidence": parsed.confidence,
            "from_command": parsed.from_command,
        }
        meta = {
            "session_id": session_id,
            "canvas_id": canvas_id,
            "citations": citations_data,
            "ref_map": ref_map or {},
            "evidence_summary": evidence_summary.model_dump() if evidence_summary else None,
            "intent": intent_info,
            "current_stage": current_stage,
        }
        q.push_event(task_id, "meta", meta)
        if dashboard_data:
            q.push_event(task_id, "dashboard", dashboard_data if isinstance(dashboard_data, dict) else {})
        if tool_trace_data:
            q.push_event(task_id, "tool_trace", tool_trace_data)
        if agent_debug:
            q.push_event(task_id, "agent_debug", agent_debug)
        for chunk in _chunk_text(response_text):
            q.push_event(task_id, "delta", {"delta": chunk})
        q.push_event(task_id, "done", {})

        state.status = TaskStatus.completed
        state.finished_at = time.time()
        q.set_state(state)
    except Exception as e:
        logger.exception("[dispatcher] chat task_id=%s failed: %s", task_id, e)
        state.status = TaskStatus.error
        state.finished_at = time.time()
        state.error_message = str(e)
        q.set_state(state)
        q.push_event(task_id, "error", {"message": str(e)})
    finally:
        q.release_slot(task_id, session_id)


async def run_unified_chat_worker_once() -> bool:
    """
    Try to run one Chat task from Redis queue if a slot is free and queue head can acquire.
    Returns True if a task was started.
    """
    try:
        q = get_task_queue()
    except Exception as e:
        logger.warning("[dispatcher] get_task_queue failed: %s", e)
        return False

    if q.active_count() >= settings.tasks.max_active_slots:
        return False

    pending = q.read_pending(count=1)
    if not pending:
        return False

    item = pending[0]
    stream_id = item.get("stream_id")
    task_id = item.get("task_id")
    if not stream_id or not task_id:
        return False

    state = q.get_state(task_id)
    if not state or state.status != TaskStatus.queued:
        return False

    session_id = state.session_id or ""
    if not q.acquire_slot(task_id, session_id):
        return False

    q.remove_from_queue_by_stream_id(stream_id)
    logger.info("[dispatcher] running chat task_id=%s session_id=%s", task_id, session_id)
    asyncio.create_task(asyncio.to_thread(run_chat_task_sync, task_id, state.payload))
    return True
