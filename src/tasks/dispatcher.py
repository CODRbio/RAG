"""
Unified dispatcher: run Chat tasks from Redis queue with global slot limit.
Chat execution pushes events to Redis for GET /chat/stream/{task_id}.
Scholar download+ingest runs as background task, state in same Redis task store.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Dict, Optional

from config.settings import settings
from src.log import get_logger
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState

logger = get_logger(__name__)
try:
    from src.observability import metrics as _obs_metrics
except Exception:
    _obs_metrics = None


def _mark_timeout_if_stale(task_id: str, q) -> bool:
    state = q.get_state(task_id)
    if not state or state.status != TaskStatus.queued:
        return False
    age = time.time() - float(state.created_at or 0)
    if age < settings.tasks.queue_timeout_seconds:
        return False
    state.status = TaskStatus.timeout
    state.finished_at = time.time()
    state.error_message = "queue timeout"
    q.set_state(state)
    q.push_event(task_id, "timeout", {"reason": "queue_timeout"})
    if _obs_metrics and hasattr(_obs_metrics, "task_queue_timeout_total"):
        _obs_metrics.task_queue_timeout_total.labels(kind=state.kind.value).inc()
    return True


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
    payload = dict(state.payload)
    optional_user_id = payload.pop("_optional_user_id", None)

    state.status = TaskStatus.running
    state.started_at = time.time()
    state.queue_position = 0
    q.set_state(state)

    try:
        body = ChatRequest(**payload)
        (session_id_out, response_text, citations, evidence_summary, parsed, dashboard_data, tool_trace_data, ref_map,
         agent_debug, _prompt_local_db, _local_db_msg) = _run_chat(body, optional_user_id)
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
            "prompt_local_db_choice": _prompt_local_db or False,
            "local_db_mismatch_message": _local_db_msg,
        }
        q.push_event(task_id, "meta", meta)
        # 如果需要用户选择是否使用本地库，单独推一个专属事件，前端更容易捕获
        if _prompt_local_db:
            q.push_event(task_id, "local_db_choice", {
                "prompt_local_db_choice": True,
                "message": _local_db_msg or "",
            })
        if dashboard_data:
            q.push_event(task_id, "dashboard", dashboard_data if isinstance(dashboard_data, dict) else {})
        if tool_trace_data:
            q.push_event(task_id, "tool_trace", tool_trace_data)
        if agent_debug:
            q.push_event(task_id, "agent_debug", agent_debug)
        for chunk in _chunk_text(response_text):
            q.push_event(task_id, "delta", {"delta": chunk})
            latest = q.get_state(task_id)
            if latest and latest.status == TaskStatus.cancelled:
                q.push_event(task_id, "cancelled", {"status": "cancelled"})
                return
        latest = q.get_state(task_id)
        if latest and latest.status == TaskStatus.cancelled:
            q.push_event(task_id, "cancelled", {"status": "cancelled"})
            return
        elapsed = time.time() - float(state.started_at or time.time())
        if elapsed > settings.tasks.run_timeout_seconds:
            state.status = TaskStatus.timeout
            state.error_message = "run timeout"
            q.push_event(task_id, "timeout", {"reason": "run_timeout"})
            if _obs_metrics and hasattr(_obs_metrics, "task_queue_timeout_total"):
                _obs_metrics.task_queue_timeout_total.labels(kind=state.kind.value).inc()
        else:
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

    pending = q.read_pending(count=min(100, settings.tasks.queue_max_len))
    if not pending:
        return False
    for item in pending:
        stream_id = item.get("stream_id")
        task_id = item.get("task_id")
        if not stream_id or not task_id:
            continue
        state = q.get_state(task_id)
        if not state or state.status != TaskStatus.queued:
            q.remove_from_queue_by_stream_id(stream_id)
            continue
        if _mark_timeout_if_stale(task_id, q):
            q.remove_from_queue_by_stream_id(stream_id)
            continue
        session_id = state.session_id or ""
        if not q.acquire_slot(task_id, session_id):
            continue
        q.remove_from_queue_by_stream_id(stream_id)
        logger.info("[dispatcher] running chat task_id=%s session_id=%s", task_id, session_id)
        asyncio.create_task(asyncio.to_thread(run_chat_task_sync, task_id, state.payload))
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Scholar: download PDF + trigger ingest (background, no queue slot)
# ─────────────────────────────────────────────────────────────────────────────


async def process_download_and_ingest(
    task_id: str,
    paper_info: Dict[str, Any],
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download one paper PDF then trigger ingest into the given collection.
    PDF stays in download_dir; state/events written to Redis for GET /scholar/task/{task_id}.
    """
    from src.retrieval.downloader.adapter import get_adapter

    q = get_task_queue()
    state = q.get_state(task_id)
    if not state:
        state = TaskState(
            task_id=task_id,
            kind=TaskKind.scholar,
            status=TaskStatus.running,
            payload={"paper_info": paper_info, "collection": collection or ""},
        )
        q.set_state(state)

    state.status = TaskStatus.running
    state.started_at = state.started_at or time.time()
    state.payload["progress"] = 10
    state.payload["stage"] = "DOWNLOADING"
    q.set_state(state)
    q.push_event(task_id, "progress", {"progress": 10, "stage": "DOWNLOADING"})

    adapter = get_adapter()
    try:
        dl_result = await adapter.download_paper(
            title=paper_info.get("title", ""),
            doi=paper_info.get("doi"),
            pdf_url=paper_info.get("pdf_url"),
            annas_md5=paper_info.get("annas_md5"),
            authors=paper_info.get("authors"),
            year=paper_info.get("year"),
        )
    except Exception as e:
        logger.exception("download_and_ingest download failed task_id=%s: %s", task_id, e)
        state.status = TaskStatus.error
        state.finished_at = time.time()
        state.error_message = f"下载失败: {e}"
        state.payload["progress"] = 0
        state.payload["stage"] = "FAILED"
        state.payload["download_result"] = {"success": False, "message": str(e)}
        q.set_state(state)
        q.push_event(task_id, "error", {"message": state.error_message})
        return {"success": False, "message": str(e)}

    if not dl_result.get("success"):
        state.status = TaskStatus.error
        state.finished_at = time.time()
        state.error_message = dl_result.get("message", "下载失败")
        state.payload["progress"] = 0
        state.payload["stage"] = "FAILED"
        state.payload["download_result"] = dl_result
        q.set_state(state)
        q.push_event(task_id, "error", {"message": state.error_message})
        return {**dl_result, "ingest_triggered": False}

    paper_id = dl_result["paper_id"]
    filepath = dl_result["filepath"]
    collection_name = collection or settings.collection.global_

    state.payload["progress"] = 50
    state.payload["stage"] = "INGESTING"
    state.payload["download_result"] = dl_result
    q.set_state(state)
    q.push_event(task_id, "progress", {"progress": 50, "stage": "INGESTING"})

    ingest_job_id = _trigger_ingest_pipeline(
        filepath=filepath,
        paper_id=paper_id,
        collection_name=collection_name,
        metadata={
            "source": "scholar_download",
            "download_method": dl_result.get("message", ""),
            "doi": paper_info.get("doi"),
            "authors": paper_info.get("authors"),
            "year": paper_info.get("year"),
        },
    )

    state.status = TaskStatus.completed
    state.finished_at = time.time()
    state.payload["progress"] = 100
    state.payload["stage"] = "DOWNLOAD_DONE_INGEST_PENDING"
    state.payload["ingest_job_id"] = ingest_job_id
    state.payload["ingest_poll_url"] = f"/ingest/jobs/{ingest_job_id}"
    state.payload["paper_id"] = paper_id
    state.payload["filepath"] = filepath
    state.payload["collection"] = collection_name
    q.set_state(state)
    q.push_event(
        task_id,
        "done",
        {
            "ingest_job_id": ingest_job_id,
            "paper_id": paper_id,
            "ingest_poll_url": f"/ingest/jobs/{ingest_job_id}",
        },
    )

    return {
        **dl_result,
        "collection": collection_name,
        "ingest_job_id": ingest_job_id,
        "ingest_triggered": True,
    }


def _trigger_ingest_pipeline(
    filepath: str,
    paper_id: str,
    collection_name: str,
    metadata: Dict[str, Any],
) -> str:
    """Start ingest job in a background thread; returns ingest job_id."""
    from src.api.routes_ingest import _run_ingest_job_safe
    from src.indexing.ingest_job_store import create_job

    job_id = f"scholar_{uuid.uuid4().hex[:8]}"
    ingest_cfg = {
        "file_paths": [filepath],
        "collection_name": collection_name,
        "content_hashes": {},
        "actual_skip": True,
        "enrich_tables": False,
        "enrich_figures": False,
        "llm_text_provider": None,
        "llm_text_model": None,
        "llm_text_concurrency": None,
        "llm_vision_provider": None,
        "llm_vision_model": None,
        "llm_vision_concurrency": None,
    }

    create_job(collection_name, {**ingest_cfg, "source": "scholar_download", "metadata": metadata}, total_files=1)

    thread = threading.Thread(
        target=_run_ingest_job_safe,
        args=(job_id, ingest_cfg),
        daemon=True,
        name=f"ingest-{job_id}",
    )
    thread.start()

    logger.info("Ingest triggered: paper_id=%s, collection=%s, job=%s", paper_id, collection_name, job_id)
    return job_id
