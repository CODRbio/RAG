"""
对话 API：POST /chat, POST /chat/stream, GET /sessions/{id}, DELETE /sessions/{id}
"""

import contextlib
import concurrent.futures
import json
import math
import logging
import re
import time
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from config.settings import settings
from src.api.routes_auth import get_current_user_id, get_optional_user_id
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

# ── Observability ──
try:
    from src.observability import metrics as _obs_metrics
except Exception:
    _obs_metrics = None  # type: ignore

from src.api.schemas import (
    ChatCitation,
    ChatRequest,
    ChatResponse,
    ChatSubmitResponse,
    ClarifyRequest,
    ClarifyResponse,
    ClarifyQuestion,
    DeepResearchRequest,
    DeepResearchStartRequest,
    DeepResearchStartAsyncResponse,
    DeepResearchStartStatusResponse,
    DeepResearchConfirmRequest,
    DeepResearchSubmitResponse,
    DeepResearchRestartPhaseRequest,
    DeepResearchRestartSectionRequest,
    DeepResearchRestartIncompleteSectionsRequest,
    DeepResearchRestartWithOutlineRequest,
    DeepResearchJobInfo,
    DeepResearchContextExtractResponse,
    EvidenceSummary,
    IntentDetectRequest,
    IntentDetectResponse,
    ChatSuggestionsRequest,
    ChatSuggestionsResponse,
    SessionInfo,
    SessionListItem,
    TurnItem,
)
from src.collaboration.intent import (
    ContextAnalysis,
    IntentParser,
    IntentType,
    ParsedIntent,
    analyze_chat_context,
    build_search_query_from_context,
    check_query_collection_scope,
    is_deep_research,
    resolve_intent_provider_name,
)
from src.collaboration.intent.commands import allocate_collection_quotas
from src.collaboration.graphic_abstract import (
    GRAPHIC_ABSTRACT_FAILURE_MD,
    render_graphic_abstract_markdown,
    resolve_graphic_abstract_model,
)
from src.collaboration.memory.session_memory import (
    SessionStore,
    get_session_store,
    load_session_memory,
)
from src.collaboration.memory.working_memory import get_or_generate_working_memory
from src.collaboration.memory.persistent_store import get_user_profile
from src.collaboration.workflow import run_workflow
from src.llm.llm_manager import get_manager, _timeout_for_model, _stream_and_collect
from src.llm.react_loop import react_loop
from src.llm.tools import (
    CORE_TOOLS,
    get_routed_skills,
    start_agent_chunk_collector,
    drain_agent_chunks,
    set_tool_collection,
    set_tool_step_top_k,
    set_agent_sonar_model,
)
from src.collaboration.citation.manager import (
    _dedupe_citations,
    chunk_to_citation,
    merge_citations_by_document,
    resolve_response_citations,
    sync_evidence_to_canvas,
)
from src.collaboration.canvas.models import Citation, CitationAnchor as CanvasCitationAnchor
from dataclasses import asdict
from src.retrieval.sonar_citations import parse_sonar_citations
from src.retrieval.structured_queries import web_queries_per_provider_from_1plus1plus1
from src.retrieval.service import (
    fuse_pools_with_gap_protection,
    get_retrieval_service,
    _hit_to_chunk as service_hit_to_chunk,
)
from src.retrieval.hybrid_retriever import _rerank_candidates
from src.retrieval.evidence import EvidenceChunk, EvidencePack
from src.generation.evidence_synthesizer import EvidenceSynthesizer, build_synthesis_system_prompt
from src.utils.context_limits import summarize_if_needed, cap_and_log
from src.utils.token_counter import count_tokens, get_context_window, needs_sliding_window
from src.collaboration.auto_complete import AutoCompleteService
from src.tasks import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus

router = APIRouter(tags=["chat"])

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
_DEEP_RESEARCH_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DEEP_RESEARCH_CANCEL_LOCK = threading.Lock()
_DEEP_RESEARCH_PAUSE_EVENTS: dict[str, threading.Event] = {}
_DEEP_RESEARCH_PAUSE_LOCK = threading.Lock()


def _is_admin_user(user_id: str) -> bool:
    profile = get_user_profile(user_id)
    return bool(profile and profile.get("role") == "admin")


def _ensure_session_access(session_id: str, current_user_id: str) -> dict[str, Any] | None:
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    is_admin = _is_admin_user(current_user_id)
    if meta is not None:
        owner_id = str(meta.get("user_id") or "").strip()
        if owner_id == current_user_id or is_admin:
            return meta
        raise HTTPException(status_code=403, detail="forbidden")
    try:
        from src.collaboration.research.job_store import get_latest_job_by_session

        latest_job = get_latest_job_by_session(session_id)
    except Exception:
        latest_job = None
    if latest_job:
        owner_id = str(latest_job.get("user_id") or "").strip()
        if owner_id == current_user_id or is_admin:
            return None
        raise HTTPException(status_code=403, detail="forbidden")
    raise HTTPException(status_code=404, detail="session not found")


def _get_intent_client(manager, body: ChatRequest | IntentDetectRequest):
    provider_name = resolve_intent_provider_name(
        intent_provider=getattr(body, "intent_provider", None),
        configured_intent_provider=getattr(settings.llm, "intent_provider", None),
        ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
        llm_provider=getattr(body, "llm_provider", None),
    )
    return manager.get_lite_client(provider_name)


_DEEP_RESEARCH_SUSPENDED: dict[str, dict[str, Any]] = {}
_DEEP_RESEARCH_SUSPENDED_LOCK = threading.Lock()


def _dr_request_cancel(job_id: str) -> None:
    with _DEEP_RESEARCH_CANCEL_LOCK:
        ev = _DEEP_RESEARCH_CANCEL_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _DEEP_RESEARCH_CANCEL_EVENTS[job_id] = ev
        ev.set()


def _dr_is_cancel_requested(job_id: str) -> bool:
    with _DEEP_RESEARCH_CANCEL_LOCK:
        ev = _DEEP_RESEARCH_CANCEL_EVENTS.get(job_id)
    return bool(ev and ev.is_set())


def _dr_clear_cancel_event(job_id: str) -> None:
    with _DEEP_RESEARCH_CANCEL_LOCK:
        _DEEP_RESEARCH_CANCEL_EVENTS.pop(job_id, None)


def _dr_request_pause(job_id: str) -> None:
    with _DEEP_RESEARCH_PAUSE_LOCK:
        ev = _DEEP_RESEARCH_PAUSE_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _DEEP_RESEARCH_PAUSE_EVENTS[job_id] = ev
        ev.set()


def _dr_is_pause_requested(job_id: str) -> bool:
    with _DEEP_RESEARCH_PAUSE_LOCK:
        ev = _DEEP_RESEARCH_PAUSE_EVENTS.get(job_id)
    return bool(ev and ev.is_set())


def _dr_clear_pause_event(job_id: str) -> None:
    with _DEEP_RESEARCH_PAUSE_LOCK:
        ev = _DEEP_RESEARCH_PAUSE_EVENTS.get(job_id)
        if ev is not None:
            ev.clear()


def _dr_wait_if_paused(job_id: str) -> None:
    from src.collaboration.research.job_store import append_event, get_job, update_job

    announced = False
    while _dr_is_pause_requested(job_id):
        if _dr_is_cancel_requested(job_id):
            raise RuntimeError("Deep Research cancelled by user")
        if not announced:
            job = get_job(job_id) or {}
            current_stage = str(job.get("current_stage") or "")
            update_job(
                job_id,
                status="paused",
                current_stage=current_stage,
                message="任务已暂停",
            )
            append_event(job_id, "paused", {"job_id": job_id, "message": "任务已暂停", "current_stage": current_stage})
            announced = True
        time.sleep(0.2)


def _dr_release_slot_eager(job_id: str) -> None:
    """Eagerly release the Redis task-queue slot for *job_id*.

    Called when a cancel or restart request is issued so that a subsequent new
    job for the same session can acquire the slot immediately (rather than
    waiting for the old job's thread to finish and call release_slot in its
    finally block — which causes a race condition where the new job stays
    pending indefinitely).

    The thread's own finally-block release_slot call is safe afterwards because
    Redis SREM on an already-removed member is a no-op.
    """
    try:
        from src.collaboration.research.job_store import get_job
        from src.tasks import get_task_queue
        job = get_job(job_id)
        if not job:
            return
        # get_job → to_dict() parses request_json → dict; session_id is a top-level field
        session_id = str(job.get("session_id") or "")
        get_task_queue().release_slot(job_id, session_id)
    except Exception as exc:
        _chat_logger.debug("[dr] eager slot release failed job_id=%s: %s", job_id, exc)


def _dr_store_suspended_runtime(job_id: str, runtime: dict[str, Any]) -> None:
    with _DEEP_RESEARCH_SUSPENDED_LOCK:
        current = _DEEP_RESEARCH_SUSPENDED.get(job_id) or {}
        merged = {**current, **runtime}
        merged.setdefault("resume_inflight", False)
        _DEEP_RESEARCH_SUSPENDED[job_id] = merged


def _dr_get_suspended_runtime(job_id: str) -> dict[str, Any] | None:
    with _DEEP_RESEARCH_SUSPENDED_LOCK:
        runtime = _DEEP_RESEARCH_SUSPENDED.get(job_id)
        return dict(runtime) if runtime else None


def _dr_pop_suspended_runtime(job_id: str) -> dict[str, Any] | None:
    with _DEEP_RESEARCH_SUSPENDED_LOCK:
        return _DEEP_RESEARCH_SUSPENDED.pop(job_id, None)


def _dr_mark_resume_inflight(job_id: str) -> bool:
    with _DEEP_RESEARCH_SUSPENDED_LOCK:
        runtime = _DEEP_RESEARCH_SUSPENDED.get(job_id)
        if not runtime:
            return False
        if bool(runtime.get("resume_inflight", False)):
            return False
        runtime["resume_inflight"] = True
        return True


def _dr_set_resume_idle(job_id: str) -> None:
    with _DEEP_RESEARCH_SUSPENDED_LOCK:
        runtime = _DEEP_RESEARCH_SUSPENDED.get(job_id)
        if runtime is not None:
            runtime["resume_inflight"] = False

# ── 智能查询路由：三层判断是否需要 RAG ──
#
# 第一层：正则强制 rag（零成本，关键词命中直接走检索，不调 LLM）
# 第二层：LLM 轻量分类（~300 tokens，带上下文判断 chat/rag）
# 第三层：严格解析 + 保守兜底（只有明确 chat 才跳过，其他一律 rag）

from src.log import get_logger as _get_chat_logger

_chat_logger = _get_chat_logger(__name__)


def _ensure_research_session_title(store: Any, session_id: str, topic: str) -> None:
    """Ensure a research session has a non-empty, readable title."""
    topic_text = (topic or "").strip()
    if not session_id or not topic_text:
        return
    try:
        meta = store.get_session_meta(session_id) or {}
        current_title = str(meta.get("title") or "").strip() if isinstance(meta, dict) else ""
        current_type = str(meta.get("session_type") or "").strip() if isinstance(meta, dict) else ""
        updates: Dict[str, Any] = {}
        if not current_title:
            updates["title"] = topic_text[:80]
        if current_type != "research":
            updates["session_type"] = "research"
        if updates:
            store.update_session_meta(session_id, updates)
    except Exception as exc:
        _chat_logger.debug("[deep-research] ensure session title failed sid=%s: %s", session_id[:12], exc)

# ── Start-phase job store ─────────────────────────────────────────────────────
# Scope+plan can take 10+ minutes; we run it in a background thread and return
# a job_id immediately so the client can poll instead of blocking on HTTP.
#
# State is kept in two layers:
#   1. _START_JOBS  – fast in-memory dict (lost on hot-reload / process restart)
#   2. _start_jobs_disk – DiskCache backed by SQLite (survives reloads, 2-hour TTL)
# The status endpoint reads from layer-1 first; on a miss it falls back to
# layer-2 so that a dev hot-reload doesn't immediately surface a 404 to the
# client while the background thread is still running in the old process.
_START_JOBS: Dict[str, Dict[str, Any]] = {}
_START_JOBS_LOCK = threading.Lock()
_MAX_START_JOBS = 50  # evict oldest entries once the dict exceeds this size

_start_jobs_disk: Any = None  # lazy-init DiskCache
_start_jobs_disk_lock = threading.Lock()


def _get_start_jobs_disk() -> Any:
    """Return (lazily initialised) DiskCache for start-phase job persistence."""
    global _start_jobs_disk
    if _start_jobs_disk is not None:
        return _start_jobs_disk
    with _start_jobs_disk_lock:
        if _start_jobs_disk is None:
            from src.utils.cache import DiskCache
            _db_path = str(Path(__file__).parent.parent.parent / "data" / "cache" / "start_jobs.db")
            _start_jobs_disk = DiskCache(db_path=_db_path, ttl_seconds=7200)  # 2-hour TTL
        return _start_jobs_disk


def _persist_start_job(job_id: str, state: Dict[str, Any]) -> None:
    """Write start-job state to the disk cache (best-effort, never raises)."""
    try:
        _get_start_jobs_disk().set(f"start:{job_id}", json.dumps(state, ensure_ascii=False, default=str))
    except Exception as _e:
        _chat_logger.debug("[start-job] disk-persist failed job_id=%s: %s", job_id[:12], _e)


def _load_start_job_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    """Read start-job state from disk cache; returns None if not found."""
    try:
        raw = _get_start_jobs_disk().get(f"start:{job_id}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _evict_old_start_jobs() -> None:
    """Remove oldest entries to keep _START_JOBS bounded."""
    with _START_JOBS_LOCK:
        if len(_START_JOBS) > _MAX_START_JOBS:
            excess = list(_START_JOBS.keys())[: len(_START_JOBS) - _MAX_START_JOBS]
            for k in excess:
                _START_JOBS.pop(k, None)


_START_JOB_HEARTBEAT_INTERVAL_S = 15  # DB heartbeat interval for start-phase liveness


def _run_start_job_bg(
    *,
    job_id: str,
    start_kwargs: Dict[str, Any],
    session_id: str,
    store: Any,
) -> None:
    """Background thread: execute scope+plan, write result to _START_JOBS, disk cache, and DB.

    A lightweight heartbeat mechanism periodically writes ``updated_at`` to
    the DB so that the frontend can distinguish "backend is still working"
    from "backend crashed silently".  The heartbeat also writes to the
    in-memory ``_START_JOBS`` dict so the fast-path poll endpoint sees a
    fresh ``heartbeat_ts``.
    """
    from src.collaboration.research.agent import start_deep_research
    from src.collaboration.research.job_store import update_job as _update_db_job

    # ── shared mutable state between main thread and heartbeat thread ──
    _hb_stop = threading.Event()
    _hb_last_stage = {"stage": "initializing", "percent": 0}

    def _progress_cb(stage: str, message: str, percent: int) -> None:
        _hb_last_stage["stage"] = message
        _hb_last_stage["percent"] = percent
        state = {
            "status": "running",
            "session_id": session_id,
            "result": None,
            "error": None,
            "current_stage": message,
            "progress": percent,
            "heartbeat_ts": time.time(),
        }
        with _START_JOBS_LOCK:
            if job_id in _START_JOBS:
                _START_JOBS[job_id].update({"current_stage": message, "progress": percent, "heartbeat_ts": time.time()})
        _persist_start_job(job_id, state)
        try:
            _update_db_job(job_id, current_stage=message, message=f"规划中: {message}")
        except Exception:
            pass

    def _heartbeat_loop() -> None:
        """Write a DB heartbeat every N seconds so the frontend can detect liveness."""
        while not _hb_stop.wait(_START_JOB_HEARTBEAT_INTERVAL_S):
            elapsed_s = time.perf_counter() - t0
            stage = _hb_last_stage["stage"]
            pct = _hb_last_stage["percent"]
            # Touch in-memory dict so poll endpoint returns a fresh heartbeat_ts
            with _START_JOBS_LOCK:
                entry = _START_JOBS.get(job_id)
                if entry and entry.get("status") == "running":
                    entry["heartbeat_ts"] = time.time()
            # Touch DB updated_at
            try:
                _update_db_job(
                    job_id,
                    message=f"规划中: {stage} ({int(elapsed_s)}s)",
                )
            except Exception:
                pass
            _chat_logger.debug(
                "[start-job] %s heartbeat | stage=%s pct=%d elapsed_s=%.0f",
                job_id[:12], stage, pct, elapsed_s,
            )

    t0 = time.perf_counter()
    _chat_logger.info(
        "[start-job] %s begin | session=%s",
        job_id[:12],
        session_id[:12],
    )
    _persist_start_job(job_id, {"status": "running", "session_id": session_id, "result": None, "error": None, "heartbeat_ts": time.time()})

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name=f"dr-hb-{job_id[:8]}")
    hb_thread.start()
    try:
        result = start_deep_research(
            **{
                **start_kwargs,
                "progress_callback": _progress_cb,
                "cancel_check": lambda: _dr_is_cancel_requested(job_id),
                "pause_waiter": lambda: _dr_wait_if_paused(job_id),
            }
        )
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        if result.get("canvas_id"):
            store.update_session_meta(session_id, {"canvas_id": result.get("canvas_id", "")})
            try:
                _update_db_job(job_id, canvas_id=result.get("canvas_id", ""))
            except Exception:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _chat_logger.info(
            "[start-job] %s done | session=%s elapsed_ms=%.0f sections=%d has_error=%s",
            job_id[:12],
            session_id[:12],
            elapsed_ms,
            len(result.get("outline") or []),
            bool(result.get("error")),
        )
        done_state: Dict[str, Any] = {
            "status": "done",
            "session_id": session_id,
            "result": result,
            "error": None,
        }
        with _START_JOBS_LOCK:
            _START_JOBS[job_id] = done_state
        _persist_start_job(job_id, done_state)
        try:
            _update_db_job(job_id, message="规划完成，等待确认")
        except Exception:
            pass
    except Exception as exc:
        is_cancelled = _dr_is_cancel_requested(job_id) or "cancelled" in str(exc).lower() or "canceled" in str(exc).lower()
        _chat_logger.error(
            "[start-job] %s failed | session=%s error=%s",
            job_id[:12],
            session_id[:12],
            exc,
            exc_info=True,
        )
        err_state: Dict[str, Any] = {
            "status": "error" if not is_cancelled else "cancelled",
            "session_id": session_id,
            "result": None,
            "error": str(exc),
        }
        with _START_JOBS_LOCK:
            _START_JOBS[job_id] = err_state
        _persist_start_job(job_id, err_state)
        try:
            _update_db_job(
                job_id,
                status="cancelled" if is_cancelled else "error",
                error_message="" if is_cancelled else str(exc)[:500],
                message="规划已取消" if is_cancelled else "规划失败",
            )
        except Exception:
            pass
    finally:
        _hb_stop.set()  # signal heartbeat thread to stop
        _dr_clear_pause_event(job_id)


# ── 第一层：正则强制走 rag 的关键词（只做"强制 rag"，不做"强制 chat"）──
_FORCE_RAG_RE = re.compile(
    r'(材料|文献|论文|资料|数据|根据|对比|对照|比较|验证|引用|出处|来源|原文'
    r'|最大|最小|最深|最高|最低|多少|几种|哪些|哪个|有多大|有多深|有多高'
    r'|图表|表格|实验|结果|结论|证据|研究|分析|综述|检索|查一下|帮我查|帮我找'
    r'|according\s+to|based\s+on\s+(the\s+)?(data|paper|material|literature|evidence))',
    re.IGNORECASE,
)

# ── 第二层：LLM 分类 prompt ──
_ROUTE_SYSTEM = _pm.render("chat_route_system.txt")

# ── 第三层解析：严格正则校验 LLM 输出 ──
_ROUTE_ANSWER_RE = re.compile(r'^(chat|rag)\b', re.IGNORECASE)


def _classify_query(message: str, history_turns: list, llm_client) -> bool:
    """
    三层查询路由：判断消息是否需要 RAG 检索。
    返回 True = 需要检索, False = 直接对话。

    第一层：正则关键词 → 强制 rag（零成本）
    第二层：LLM 分类（~300 tokens）
    第三层：严格解析 + 保守兜底
    """
    msg = message.strip()
    if not msg:
        return False

    # ── 第一层：关键词强制 rag ──
    if _FORCE_RAG_RE.search(msg):
        _chat_logger.info("query_route | msg=%r → rag (keyword hit)", msg[:80])
        return True

    # ── 第二层：LLM 分类 ──
    ctx_lines = []
    recent = history_turns[-6:] if len(history_turns) > 6 else history_turns
    for t in recent:
        role_label = "用户" if t.role == "user" else "助手"
        text = (t.content or "").strip().replace("\n", " ")
        if len(text) > 150:
            text = text[:150] + "…"
        ctx_lines.append(f"{role_label}: {text}")
    history_block = "\n".join(ctx_lines) if ctx_lines else "（首轮对话）"

    prompt = _pm.render("chat_route_classify.txt", history=history_block, message=msg)

    try:
        resp = llm_client.chat(
            [
                {"role": "system", "content": _ROUTE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        # 取 final_text；某些 thinking 模型答案可能在 reasoning_text 里
        raw_answer = (resp.get("final_text") or "").strip().lower()
        if not raw_answer:
            raw_answer = (resp.get("reasoning_text") or "").strip().lower()

        # ── 第三层：严格解析 ──
        # 在整个回复中搜索 chat/rag 关键词（兼容模型输出 "chat" / "答案：chat" 等格式）
        if re.search(r'\bchat\b', raw_answer):
            _chat_logger.info("query_route | msg=%r → chat (llm, raw=%r)", msg[:80], raw_answer[:50])
            return False
        elif re.search(r'\brag\b', raw_answer):
            _chat_logger.info("query_route | msg=%r → rag (llm, raw=%r)", msg[:80], raw_answer[:50])
            return True
        else:
            # 无法识别 → 保守走 rag
            _chat_logger.info(
                "query_route | msg=%r → rag (unrecognized, raw=%r)", msg[:80], raw_answer[:50],
            )
            return True
    except Exception as e:
        _chat_logger.warning("query_route failed (%s), fallback to rag", e)
        return True


def _build_system_with_context(context: str) -> str:
    return _pm.render("chat_rag_system.txt", context=context or "（本轮暂无检索结果）")


def _chunk_text(text: str, chunk_size: int = 20):
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _serialize_citation_anchor(anchor: CanvasCitationAnchor) -> dict:
    if isinstance(anchor, dict):
        return {
            "chunk_id": str(anchor.get("chunk_id") or ""),
            "page_num": anchor.get("page_num") if isinstance(anchor.get("page_num"), int) else None,
            "bbox": anchor.get("bbox") if isinstance(anchor.get("bbox"), list) else None,
            "snippet": anchor.get("snippet") if isinstance(anchor.get("snippet"), str) else None,
        }
    return {
        "chunk_id": anchor.chunk_id or "",
        "page_num": anchor.page_num,
        "bbox": anchor.bbox,
        "snippet": anchor.snippet,
    }


def _normalize_citation_anchor_dicts(raw_anchors: Any, fallback: dict | None = None) -> list[dict]:
    anchors_out: list[dict] = []
    seen: set[str] = set()
    anchor_items = raw_anchors if isinstance(raw_anchors, list) else []
    for raw in anchor_items:
        if not isinstance(raw, dict):
            continue
        chunk_id = str(raw.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        bbox = raw.get("bbox") if isinstance(raw.get("bbox"), list) else None
        snippet = raw.get("snippet")
        anchors_out.append({
            "chunk_id": chunk_id,
            "page_num": raw.get("page_num") if isinstance(raw.get("page_num"), int) else None,
            "bbox": bbox,
            "snippet": snippet if isinstance(snippet, str) and snippet.strip() else None,
        })

    if not anchors_out and fallback:
        chunk_id = str(fallback.get("chunk_id") or "").strip()
        if chunk_id:
            anchors_out.append({
                "chunk_id": chunk_id,
                "page_num": fallback.get("page_num") if isinstance(fallback.get("page_num"), int) else None,
                "bbox": fallback.get("bbox") if isinstance(fallback.get("bbox"), list) else None,
                "snippet": fallback.get("snippet") if isinstance(fallback.get("snippet"), str) else None,
            })
    return anchors_out


def _serialize_citation(c: Citation | str) -> dict:
    """将 Citation 对象或字符串序列化为字典。"""
    if isinstance(c, str):
        return {
            "cite_key": c,
            "chunk_id": None,
            "title": "",
            "authors": [],
            "year": None,
            "doc_id": None,
            "url": None,
            "pdf_url": None,
            "doi": None,
            "bbox": None,
            "page_num": None,
            "anchors": [],
            "provider": None,
        }
    anchors = _normalize_citation_anchor_dicts(
        [_serialize_citation_anchor(anchor) for anchor in getattr(c, "anchors", [])],
        fallback={
            "chunk_id": getattr(c, "chunk_id", None),
            "page_num": getattr(c, "page_num", None),
            "bbox": getattr(c, "bbox", None),
        },
    )
    primary = anchors[0] if anchors else {}
    return {
        "cite_key": c.cite_key or c.id,
        "chunk_id": primary.get("chunk_id"),
        "title": c.title or "",
        "authors": c.authors or [],
        "year": c.year,
        "doc_id": c.doc_id,
        "url": c.url,
        "pdf_url": getattr(c, "pdf_url", None),
        "doi": c.doi,
        "bbox": primary.get("bbox"),
        "page_num": primary.get("page_num"),
        "anchors": anchors,
        "provider": getattr(c, "provider", None),
    }


def _chat_citation_from_dict(raw: dict | str) -> ChatCitation:
    if isinstance(raw, str):
        raw = _serialize_citation(raw)
    anchors = _normalize_citation_anchor_dicts(
        raw.get("anchors"),
        fallback={
            "chunk_id": raw.get("chunk_id"),
            "page_num": raw.get("page_num"),
            "bbox": raw.get("bbox"),
        },
    )
    primary = anchors[0] if anchors else {}
    return ChatCitation(
        cite_key=str(raw.get("cite_key") or ""),
        chunk_id=primary.get("chunk_id"),
        title=str(raw.get("title") or ""),
        authors=raw.get("authors") if isinstance(raw.get("authors"), list) else [],
        year=raw.get("year") if isinstance(raw.get("year"), int) else None,
        doc_id=str(raw.get("doc_id") or "") or None,
        url=str(raw.get("url") or "") or None,
        pdf_url=str(raw.get("pdf_url") or "") or None,
        doi=str(raw.get("doi") or "") or None,
        bbox=primary.get("bbox"),
        page_num=primary.get("page_num"),
        anchors=anchors,
        provider=str(raw.get("provider") or "") or None,
    )


def _build_filters(body: ChatRequest) -> dict:
    """从 ChatRequest 中提取检索参数构建 filters 字典。"""
    filters = {}
    if body.web_providers:
        filters["web_providers"] = body.web_providers
    if body.web_source_configs:
        filters["web_source_configs"] = body.web_source_configs
    if body.serpapi_ratio is not None:
        filters["serpapi_ratio"] = body.serpapi_ratio
    if body.use_query_expansion is not None:
        filters["use_query_expansion"] = body.use_query_expansion
    
    pool_thresholds = getattr(body, "pool_score_thresholds", None)
    if pool_thresholds is not None:
        filters["pool_score_thresholds"] = pool_thresholds

    # Fused-pool score threshold (after merge); backward compat: fallback from local_threshold
    fused_threshold = getattr(body, "fused_pool_score_threshold", None)
    if fused_threshold is None and body.local_threshold is not None:
        fused_threshold = body.local_threshold
    if fused_threshold is not None:
        filters["fused_pool_score_threshold"] = fused_threshold
    if body.year_start is not None:
        filters["year_start"] = body.year_start
    if body.year_end is not None:
        filters["year_end"] = body.year_end
    if body.step_top_k is not None:
        filters["step_top_k"] = _chat_effective_step_top_k(body.step_top_k)
    if getattr(body, "write_top_k", None) is not None:
        filters["write_top_k"] = body.write_top_k
    if getattr(body, "graph_top_k", None) is not None:
        filters["graph_top_k"] = body.graph_top_k
    if body.reranker_mode:
        filters["reranker_mode"] = body.reranker_mode
    if getattr(body, "llm_provider", None):
        filters["llm_provider"] = body.llm_provider
    if getattr(body, "ultra_lite_provider", None):
        filters["ultra_lite_provider"] = body.ultra_lite_provider
    if body.model_override:
        filters["model_override"] = body.model_override
    # 配置优先级：UI/请求入参 > config > 代码默认（见 docs/configuration.md）
    if body.use_content_fetcher is not None:
        filters["use_content_fetcher"] = body.use_content_fetcher
    if getattr(body, "agent_sonar_model", None):
        filters["agent_sonar_model"] = body.agent_sonar_model
    if body.collection:
        filters["collection"] = body.collection
    _body_collections = getattr(body, "collections", None)
    if _body_collections:
        _valid_cols = [c.strip() for c in _body_collections if (c or "").strip()]
        if _valid_cols:
            filters["collections"] = _valid_cols
    # Main fusion rank-pool policy for chat requests (global local+web rerank).
    filters["rank_pool_multiplier"] = float(
        getattr(settings.search, "chat_rank_pool_multiplier", 3.0)
    )
    # Chat 主检索以 pool_only 模式返回原始候选池；
    # 唯一一次 BGE rerank 由 §5¾ _fuse_chat_main_gap_agent_candidates() 统一执行。
    filters["pool_only"] = True
    return filters


def _generate_chat_structured_queries(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Generate 1+1+1 for Chat Round 2. Delegates to shared structured_queries module."""
    from src.retrieval.structured_queries import generate_structured_queries_1plus1plus1
    return generate_structured_queries_1plus1plus1(
        query, evidence_context, llm_client, model_override=model_override
    )


def _chat_web_queries_from_1plus1plus1(
    structured: Dict[str, str],
    web_providers: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Build web_queries_per_provider from 1+1+1. Delegates to shared structured_queries module."""
    from src.retrieval.structured_queries import web_queries_per_provider_from_1plus1plus1
    return web_queries_per_provider_from_1plus1plus1(structured, web_providers)


# ── Chat 证据充分性评估（LLM 判断，借鉴 Deep Research evaluate_sufficiency）──
_CHAT_EVIDENCE_CONTEXT_CAP = 12000  # 送入评估的 evidence 最大字符，避免超长
_CHAT_EVIDENCE_SOFT_MAX_CHARS = 40_000
_CHAT_EVIDENCE_SUMMARIZE_TO_CHARS = 16_000
_CHAT_EVIDENCE_HARD_MAX_CHARS = 55_000
_CHAT_SYSTEM_HARD_MAX_CHARS = 70_000
_CHAT_PROMPT_SAFETY_MARGIN = 0.10
_CHAT_MIN_OUTPUT_TOKENS = 2048
_AGENT_MIN_OUTPUT_TOKENS = 4096
_CHAT_STEP_BUDGET_AMPLIFY = 1.2
_FOLLOWUP_FINAL_STAGE_TURNS = 2
_FOLLOWUP_CANDIDATE_POOL_MAX = 60
_FOLLOWUP_BROAD_RE = re.compile(
    r"\b(compare|comparison|versus|vs|difference|differences|latest|recent|progress|controversy|"
    r"summary|summarize|limitations?|limitation|open questions?)\b"
    r"|对比|比较|区别|差异|最新|进展|争议|总结|综述|局限|开放问题",
    re.IGNORECASE,
)


def _chat_effective_step_top_k(step_k: Optional[int]) -> Optional[int]:
    """Chat-only internal step budget expansion for candidate recall."""
    if step_k is None:
        return None
    try:
        base = int(step_k)
    except Exception:
        return None
    if base <= 0:
        return None
    return max(base, int(math.ceil(base * _CHAT_STEP_BUDGET_AMPLIFY)))


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    # Heuristic: role/header overhead + content tokens.
    total = 0
    for m in messages:
        total += 6
        total += count_tokens(_message_content_to_text(m.get("content")))
    return total


def _budget_chat_evidence_context(
    context: str,
    *,
    llm_client: Any,
    ultra_lite_provider: Optional[str],
    purpose: str,
) -> tuple[str, Dict[str, Any]]:
    raw = (context or "").strip()
    if not raw:
        return "", {
            "purpose": purpose,
            "raw_chars": 0,
            "after_soft_chars": 0,
            "final_chars": 0,
            "used_summary": False,
            "used_hard_cap": False,
        }

    used_summary = False
    after_soft = raw
    if len(raw) > _CHAT_EVIDENCE_SOFT_MAX_CHARS:
        after_soft = summarize_if_needed(
            raw,
            _CHAT_EVIDENCE_SUMMARIZE_TO_CHARS,
            llm_client=llm_client,
            ultra_lite_provider=ultra_lite_provider,
            purpose=f"{purpose}_soft_budget",
        )
        used_summary = after_soft != raw
    final = cap_and_log(
        after_soft,
        max_chars=_CHAT_EVIDENCE_HARD_MAX_CHARS,
        purpose=f"{purpose}_hard_cap",
    )
    diag = {
        "purpose": purpose,
        "raw_chars": len(raw),
        "after_soft_chars": len(after_soft),
        "final_chars": len(final),
        "used_summary": used_summary,
        "used_hard_cap": len(final) < len(after_soft),
        "soft_max_chars": _CHAT_EVIDENCE_SOFT_MAX_CHARS,
        "soft_target_chars": _CHAT_EVIDENCE_SUMMARIZE_TO_CHARS,
        "hard_max_chars": _CHAT_EVIDENCE_HARD_MAX_CHARS,
    }
    return final, diag


def _apply_pre_send_prompt_budget(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    min_output_tokens: int,
    mode: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pruned = list(messages)
    context_window = get_context_window(model)
    prompt_tokens_before = _estimate_prompt_tokens(pruned)
    trimmed_history = 0
    trimmed_non_history = 0
    used_system_cap = False

    # Keep system + latest user; trim oldest middle turns first.
    while len(pruned) > 2 and needs_sliding_window(
        _estimate_prompt_tokens(pruned),
        context_window,
        safety_margin=_CHAT_PROMPT_SAFETY_MARGIN,
        min_output_tokens=min_output_tokens,
    ):
        removed = pruned.pop(1)
        if removed.get("role") in ("user", "assistant"):
            trimmed_history += 1
        else:
            trimmed_non_history += 1

    # Final safety on system prompt length.
    if pruned and pruned[0].get("role") == "system":
        sys_content = _message_content_to_text(pruned[0].get("content"))
        capped = cap_and_log(
            sys_content,
            max_chars=_CHAT_SYSTEM_HARD_MAX_CHARS,
            purpose=f"chat_system_pre_send_{mode}",
        )
        if capped != sys_content:
            used_system_cap = True
            pruned[0] = {**pruned[0], "content": capped}

    # If still over budget, keep shrinking oldest non-system messages until fit.
    while len(pruned) > 2 and needs_sliding_window(
        _estimate_prompt_tokens(pruned),
        context_window,
        safety_margin=_CHAT_PROMPT_SAFETY_MARGIN,
        min_output_tokens=min_output_tokens,
    ):
        removed = pruned.pop(1)
        if removed.get("role") in ("user", "assistant"):
            trimmed_history += 1
        else:
            trimmed_non_history += 1

    prompt_tokens_after = _estimate_prompt_tokens(pruned)
    diag = {
        "mode": mode,
        "model": model or "default",
        "context_window": context_window,
        "prompt_tokens_before": prompt_tokens_before,
        "prompt_tokens_after": prompt_tokens_after,
        "min_output_tokens": min_output_tokens,
        "safety_margin": _CHAT_PROMPT_SAFETY_MARGIN,
        "history_trimmed": trimmed_history,
        "non_history_trimmed": trimmed_non_history,
        "used_system_hard_cap": used_system_cap,
        "message_count_before": len(messages),
        "message_count_after": len(pruned),
    }
    return pruned, diag


class _ChatSufficiencyResponse(BaseModel):
    """LLM 返回：当前证据是否足以一致、实质地支撑回答"""
    sufficient: bool = False
    coverage_score: float = 0.0
    substantive_support: bool = False
    specificity_ok: bool = False
    consistency_ok: bool = False
    reason: str = ""


def _evaluate_chat_evidence_sufficiency(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> dict:
    """
    用 LLM 判断检索证据是否足以一致、实质地支撑对 query 的回答。
    返回 {"ok": bool, "sufficient": bool, "coverage_score": float, "reason": str, ...}。
    若调用失败则返回 ok=False，由调用方做数量兜底。
    """
    if not (query or "").strip() or not (evidence_context or "").strip():
        return {
            "ok": False,
            "sufficient": False,
            "coverage_score": 0.0,
            "substantive_support": False,
            "specificity_ok": False,
            "consistency_ok": False,
            "reason": "no query or evidence",
        }
    ctx = evidence_context.strip()
    if len(ctx) > _CHAT_EVIDENCE_CONTEXT_CAP:
        ctx = ctx[:_CHAT_EVIDENCE_CONTEXT_CAP] + "\n\n[truncated]"
    prompt = _pm.render(
        "chat_evidence_sufficiency.txt",
        query=(query or "").strip()[:2000],
        evidence_context=ctx,
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are an evidence sufficiency evaluator. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            response_model=_ChatSufficiencyResponse,
        )
        parsed = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                # strip markdown code fence
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                parsed = _ChatSufficiencyResponse.model_validate_json(raw)
        if parsed is not None:
            return {
                "ok": True,
                "sufficient": parsed.sufficient,
                "coverage_score": float(parsed.coverage_score),
                "substantive_support": bool(parsed.substantive_support),
                "specificity_ok": bool(parsed.specificity_ok),
                "consistency_ok": bool(parsed.consistency_ok),
                "reason": (parsed.reason or "").strip(),
            }
    except Exception as e:
        _chat_logger.debug("chat evidence sufficiency eval failed: %s", e)
    return {
        "ok": False,
        "sufficient": False,
        "coverage_score": 0.0,
        "substantive_support": False,
        "specificity_ok": False,
        "consistency_ok": False,
        "reason": "",
    }


class _ChatGapQueryItem(BaseModel):
    """Per-gap structured queries (recall/precision/discovery) for per-engine search."""
    recall: str = ""
    precision: str = ""
    discovery: str = ""
    precision_zh: Optional[str] = None


class _ChatGapQueriesResponse(BaseModel):
    """LLM 返回：针对证据缺口的 1–3 组可检索 query（每组含 recall/precision/discovery，适配不同引擎）"""
    gap_queries: List[_ChatGapQueryItem] = Field(default_factory=list)


def _generate_chat_gap_queries(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    根据用户问题和已有证据，生成 1–3 组针对缺口的检索 query。
    每组为 {recall, precision, discovery, precision_zh?}，用于按引擎生成 web_queries_per_provider。
    """
    if not (query or "").strip():
        return []
    ctx = (evidence_context or "").strip()
    if len(ctx) > _CHAT_EVIDENCE_CONTEXT_CAP:
        ctx = ctx[:_CHAT_EVIDENCE_CONTEXT_CAP] + "\n\n[truncated]"
    prompt = _pm.render(
        "chat_gap_queries.txt",
        query=(query or "").strip()[:2000],
        evidence_context=ctx or "(no evidence yet)",
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "You output JSON only. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            response_model=_ChatGapQueriesResponse,
        )
        parsed = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                parsed = _ChatGapQueriesResponse.model_validate_json(raw)
        if parsed and getattr(parsed, "gap_queries", None):
            out: List[Dict[str, Any]] = []
            for item in parsed.gap_queries[:3]:
                d = item.model_dump() if hasattr(item, "model_dump") else {"recall": getattr(item, "recall", ""), "precision": getattr(item, "precision", ""), "discovery": getattr(item, "discovery", ""), "precision_zh": getattr(item, "precision_zh", None)}
                if (d.get("recall") or d.get("precision") or d.get("discovery") or "").strip():
                    out.append(d)
            return out
    except Exception as e:
        _chat_logger.debug("chat gap queries generation failed: %s", e)
    return []


def _chunk_to_hit(c: EvidenceChunk) -> Dict[str, Any]:
    """EvidenceChunk -> hit-like dict for fuse_pools_with_gap_protection."""
    meta: Dict[str, Any] = {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "title": c.doc_title,
        "authors": c.authors,
        "year": c.year,
        "url": c.url,
        "pdf_url": getattr(c, "pdf_url", None),
        "doi": c.doi,
        "provider": c.provider,
    }
    return {
        "content": c.text,
        "score": c.score,
        "chunk_id": c.chunk_id,
        "metadata": meta,
        "_source_type": c.source_type,
    }


def _fuse_chat_main_gap_agent_candidates(
    *,
    query: str,
    message: str,
    main_chunks: List[EvidenceChunk],
    gap_candidate_hits: List[Dict[str, Any]],
    agent_chunks: List[EvidenceChunk],
    write_k: int,
    filters: Dict[str, Any],
) -> tuple[List[EvidenceChunk], Dict[str, Any]]:
    """Fuse main/gap/agent candidate pools for final chat-visible evidence."""
    chat_gap_ratio = float(getattr(settings.search, "chat_gap_ratio", 0.2))
    chat_agent_ratio = float(getattr(settings.search, "chat_agent_ratio", 0.1))
    main_candidates = [_chunk_to_hit(c) for c in main_chunks]
    agent_candidates = [_chunk_to_hit(c) for c in agent_chunks]
    total = len(main_candidates) + len(gap_candidate_hits) + len(agent_candidates)
    if total <= 0:
        return [], {}
    final_top_k = min(max(int(write_k), 1), total)
    fusion_diag: Dict[str, Any] = {}
    fused_hits = fuse_pools_with_gap_protection(
        query=query or message,
        main_candidates=main_candidates,
        gap_candidates=gap_candidate_hits,
        top_k=final_top_k,
        agent_candidates=agent_candidates,
        gap_ratio=chat_gap_ratio,
        agent_ratio=chat_agent_ratio,
        gap_min_keep=math.ceil(final_top_k * chat_gap_ratio),
        agent_min_keep=math.ceil(final_top_k * chat_agent_ratio),
        rank_pool_multiplier=float(getattr(settings.search, "chat_rank_pool_multiplier", 3.0)),
        pool_score_thresholds=filters.get("pool_score_thresholds"),
        trace_ctx={"phase": "chat_agent_final_fusion", "section": "chat"},
        reranker_mode=(filters or {}).get("reranker_mode") or "bge_only",
        diag=fusion_diag,
    )
    final_chunks = [
        service_hit_to_chunk(h, h.get("_source_type", "dense"), query or message)
        for h in fused_hits
    ]
    return final_chunks, (fusion_diag.get("pool_fusion") or {})


def _chunk_to_cache_payload(c: EvidenceChunk) -> Dict[str, Any]:
    """EvidenceChunk -> compact JSON payload for session reuse cache."""
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "text": c.text,
        "score": c.score,
        "source_type": c.source_type,
        "doc_title": c.doc_title,
        "authors": c.authors,
        "year": c.year,
        "url": c.url,
        "pdf_url": getattr(c, "pdf_url", None),
        "doi": c.doi,
        "page_num": c.page_num,
        "section_title": c.section_title,
        "evidence_type": c.evidence_type,
        "bbox": c.bbox,
        "provider": c.provider,
    }


def _cache_payload_to_chunk(payload: Dict[str, Any]) -> Optional[EvidenceChunk]:
    """Deserialize one cached payload back into EvidenceChunk."""
    if not isinstance(payload, dict):
        return None
    chunk_id = str(payload.get("chunk_id") or "").strip()
    doc_id = str(payload.get("doc_id") or "").strip()
    text = str(payload.get("text") or "").strip()
    if not chunk_id or not doc_id or not text:
        return None
    source_type = str(payload.get("source_type") or "dense").strip().lower()
    if source_type not in ("dense", "sparse", "graph", "web"):
        source_type = "dense"
    try:
        score = float(payload.get("score") or 0.0)
    except Exception:
        score = 0.0
    authors = payload.get("authors")
    if not isinstance(authors, list):
        authors = None
    year_raw = payload.get("year")
    year = int(year_raw) if isinstance(year_raw, int) else None
    page_num_raw = payload.get("page_num")
    page_num = int(page_num_raw) if isinstance(page_num_raw, int) else None
    bbox = payload.get("bbox") if isinstance(payload.get("bbox"), list) else None
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=score,
        source_type=source_type,
        doc_title=(str(payload.get("doc_title") or "").strip() or None),
        authors=authors,
        year=year,
        url=(str(payload.get("url") or "").strip() or None),
        pdf_url=(str(payload.get("pdf_url") or "").strip() or None),
        doi=(str(payload.get("doi") or "").strip() or None),
        page_num=page_num,
        section_title=(str(payload.get("section_title") or "").strip() or None),
        evidence_type=(str(payload.get("evidence_type") or "").strip() or None),
        bbox=bbox,
        provider=(str(payload.get("provider") or "").strip() or None),
    )


def _cache_row_payloads(row: Dict[str, Any], pool_name: str = "final_chunks") -> List[Dict[str, Any]]:
    if not isinstance(row, dict):
        return []
    if pool_name == "candidate_pool":
        payloads = row.get("candidate_pool")
        if isinstance(payloads, list):
            return [p for p in payloads if isinstance(p, dict)]
        return []
    payloads = row.get(pool_name)
    if isinstance(payloads, list):
        return [p for p in payloads if isinstance(p, dict)]
    legacy = row.get("chunks")
    if isinstance(legacy, list):
        return [p for p in legacy if isinstance(p, dict)]
    return []


def _chunk_dedup_key(chunk: EvidenceChunk) -> str:
    return (chunk.chunk_id or chunk.doc_group_key or "").strip()


def _dedup_chunk_list(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    dedup: Dict[str, EvidenceChunk] = {}
    for chunk in chunks:
        key = _chunk_dedup_key(chunk)
        if key and key not in dedup:
            dedup[key] = chunk
    return list(dedup.values())


def _load_recent_cached_chunks(
    store: SessionStore,
    session_id: str,
    max_turns: int = 4,
    *,
    pool_name: str = "final_chunks",
) -> List[EvidenceChunk]:
    """Load and flatten recent cached evidence chunks from session preferences."""
    rows = store.get_recent_evidence_cache(session_id)
    if not rows:
        return []
    selected = rows[-max(1, int(max_turns)):]
    out: List[EvidenceChunk] = []
    for row in reversed(selected):
        for payload in _cache_row_payloads(row, pool_name=pool_name):
            if isinstance(payload, dict):
                c = _cache_payload_to_chunk(payload)
                if c is not None:
                    out.append(c)
    return _dedup_chunk_list(out)


def _is_broad_followup_query(*texts: Optional[str]) -> bool:
    for text in texts:
        if text and _FOLLOWUP_BROAD_RE.search(text):
            return True
    return False


def _rerank_followup_chunks(
    query: str,
    chunks: List[EvidenceChunk],
    *,
    top_k: int,
    phase: str,
    reranker_mode: str = "bge_only",
) -> List[EvidenceChunk]:
    deduped = _dedup_chunk_list(chunks)
    if not deduped:
        return []
    limit = min(max(1, int(top_k)), len(deduped))
    candidates = [_chunk_to_hit(c) for c in deduped]
    try:
        reranked = _rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=limit,
            reranker_mode=reranker_mode,
            trace_ctx={"phase": phase, "section": "chat"},
        )
        return [
            service_hit_to_chunk(hit, hit.get("_source_type", "dense"), query)
            for hit in reranked
        ]
    except Exception as exc:
        _chat_logger.debug("follow-up rerank failed phase=%s: %s", phase, exc)
        deduped.sort(key=lambda c: c.score, reverse=True)
        return deduped[:limit]


def _followup_stage1_expand_reason(
    *,
    query: str,
    message: str,
    is_deepen: bool,
    target_span: str,
    stage1_chunks: List[EvidenceChunk],
) -> str:
    if _is_broad_followup_query(query, message):
        return "broad_followup"
    if is_deepen:
        return "deepen_command"
    if len(stage1_chunks) < 3:
        return "few_stage1_chunks"
    if target_span:
        doc_counts: Dict[str, int] = {}
        for chunk in stage1_chunks[:4]:
            key = chunk.doc_group_key
            if key:
                doc_counts[key] = doc_counts.get(key, 0) + 1
        if not any(count >= 2 for count in doc_counts.values()):
            return "target_span_coverage_missing"
    return ""


def _followup_candidate_pool_cap(step_top_k: int) -> int:
    return min(max(1, int(step_top_k)) * 3, _FOLLOWUP_CANDIDATE_POOL_MAX)


def _build_deep_research_filters(body: Any) -> dict:
    """Build filters from deep-research start/confirm request objects."""
    filters: Dict[str, Any] = {}
    for key in (
        "web_providers",
        "web_source_configs",
        "serpapi_ratio",
        "local_top_k",
        "pool_score_thresholds",
        "fused_pool_score_threshold",
        "local_threshold",
        "year_start",
        "year_end",
        "step_top_k",
        "write_top_k",
        "graph_top_k",
        "reranker_mode",
        "agent_sonar_model",
        "llm_provider",
        "intent_provider",
        "ultra_lite_provider",
        "model_override",
        "collection",
        "use_content_fetcher",
        "gap_query_intent",
        "enable_graphic_abstract",
        "graphic_abstract_model",
    ):
        val = getattr(body, key, None)
        if val is not None and val != "":
            filters[key] = val
    # Multi-collection support
    _cols = getattr(body, "collections", None)
    if _cols:
        _valid = [c.strip() for c in _cols if (c or "").strip()]
        if _valid:
            filters["collections"] = _valid
    filters["rank_pool_multiplier"] = float(
        getattr(settings.search, "research_rank_pool_multiplier", 3.0)
    )
    return filters


def _infer_sources_from_citations(citations: List[Citation]) -> List[str]:
    has_web = any(bool(getattr(c, "url", None)) for c in citations)
    has_local = any(not bool(getattr(c, "url", None)) for c in citations)
    out: List[str] = []
    if has_local:
        out.append("local")
    if has_web:
        out.append("web")
    return out or ["deep_research"]


def _compute_citation_provider_stats(
    citations: List[Citation],
    chunks: list | None = None,
) -> dict[str, dict[str, int]]:
    """
    Compute per-provider counts at two levels:
    - citation_level: unique docs/URLs — same website only counts once
    - chunk_level: individual evidence fragments — one article with 5 chunks counts 5
    """
    cite_counts: dict[str, int] = {}
    for c in citations:
        p = getattr(c, "provider", None) or ("local" if not c.url else "web")
        cite_counts[p] = cite_counts.get(p, 0) + 1

    chunk_counts: dict[str, int] = {}
    if chunks:
        for c in chunks:
            p = getattr(c, "provider", None) or (
                "local" if getattr(c, "source_type", "") in ("dense", "graph") else "web"
            )
            chunk_counts[p] = chunk_counts.get(p, 0) + 1

    return {
        "chunk_level": chunk_counts if chunk_counts else {},
        "citation_level": cite_counts,
    }


def _generate_and_set_session_title(
    store: SessionStore,
    session_id: str,
    message: str,
    ultra_lite_provider: Optional[str] = None,
    max_chars: int = 80,
) -> None:
    """根据首条用户消息用 ultra-lite 模型生成会话标题并写入 store。"""
    if not (message or message.strip()):
        return
    try:
        manager = get_manager(str(_CONFIG_PATH))
        client = manager.get_ultra_lite_client(ultra_lite_provider)
        prompt = _pm.render("chat_title_from_message.txt", message=(message or "").strip()[:500])
        resp = client.chat(
            messages=[
                {"role": "system", "content": "Output only the title phrase in one line, no quotes or explanation."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.get("final_text") or "").strip()
        if text:
            title = text[:max_chars].strip()
            if title:
                store.update_session_meta(session_id, {"title": title})
    except Exception as e:
        _chat_logger.debug("session title generation failed: %s", e)


@contextlib.contextmanager
def _request_debug_level(body: ChatRequest):
    """当前端 agent_debug_mode 为 True 时，本请求期间将相关 logger 临时设为 DEBUG。"""
    if not getattr(body, "agent_debug_mode", False):
        yield
        return
    loggers_to_boost = ["src.retrieval", "src.api.routes_chat", "src.collaboration.research.agent"]
    saved = {n: logging.getLogger(n).level for n in loggers_to_boost}
    for n in loggers_to_boost:
        logging.getLogger(n).setLevel(logging.DEBUG)
    _chat_logger.info("[chat] agent_debug_mode=ON → DEBUG log level for this request")
    try:
        yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)


def _run_chat_impl(
    body: ChatRequest,
    optional_user_id: str | None = None,
    step_callback: Optional[Callable[[Optional[str], str], None]] = None,
    delta_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str], dict | None, bool, Optional[str]]:
    import time as _time
    from src.debug import get_debug_logger
    dbg = get_debug_logger()

    def _step(step_id: Optional[str], label: str) -> None:
        if step_callback:
            step_callback(step_id, label or "")

    def _stream_llm_text(chat_client, stream_messages, *, model_override=None) -> str:
        final_result: dict[str, Any] | None = None
        text_parts: list[str] = []
        for ev in chat_client.stream_chat(stream_messages, model=model_override, max_tokens=None):
            if ev.get("type") == "text_delta":
                delta = str(ev.get("delta") or "")
                if not delta:
                    continue
                text_parts.append(delta)
                if delta_callback:
                    delta_callback(delta)
            elif ev.get("type") == "completed":
                maybe_result = ev.get("response")
                if isinstance(maybe_result, dict):
                    final_result = maybe_result
        if final_result is not None:
            return str(final_result.get("final_text") or "")
        return "".join(text_parts)

    t_start = _time.perf_counter()

    session_id = body.session_id
    store = get_session_store()
    if not session_id:
        session_id = store.create_session(
            canvas_id=body.canvas_id or "",
            user_id=optional_user_id or "",
        )
    memory = load_session_memory(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="session not found")

    message = body.message.strip()

    # ── 0½. 本会话本地库偏好（前端传入用户选择后写入 session 变量，立即返回无需 LLM）──
    pref_local = getattr(body, "session_preference_local_db", None)
    if pref_local in ("no_local", "use"):
        store.update_session_meta(session_id, {"preferences": {"local_db": pref_local}})
        confirm = (
            "已设置：本会话将不使用本地知识库，仅使用网络检索。"
            if pref_local == "no_local"
            else "已设置：本会话继续使用当前本地知识库。"
        )
        memory.add_turn("user", message or "(设置偏好)")
        memory.add_turn("assistant", confirm, citations=[])
        _chat_logger.info("[chat] 0½ 写入 session 偏好 local_db=%s", pref_local)
        empty_ev = EvidenceSummary(query=message or "", total_chunks=0, sources_used=[], retrieval_time_ms=0)
        _parsed = ParsedIntent(intent_type=IntentType.CHAT, confidence=1.0, raw_input=message or "")
        _step(None, "")
        return session_id, confirm, [], empty_ev, _parsed, None, None, {}, None, False, None
    # 仅当会话尚无标题时用首条消息生成短标题（提交时已用问题作为标题的则保留）
    meta = store.get_session_meta(session_id)
    if store.get_turn_count(session_id) == 0 and message and not (meta and (meta.get("title") or "").strip()):
        _generate_and_set_session_title(
            store,
            session_id,
            message,
            ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
        )
    search_mode = body.search_mode or "local"
    if search_mode not in ("local", "web", "hybrid", "none"):
        search_mode = "local"

    current_stage = store.get_session_stage(session_id)
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    intent_client = _get_intent_client(manager, body)
    lite_client = manager.get_lite_client(body.llm_provider or None)
    history_for_intent = memory.get_context_window(n=6)

    agent_debug_data: dict | None = None

    # ── 0. 解析 agent_mode（在所有阶段之前完成，影响后续检索和生成逻辑）──
    # 优先级：agent_mode 字段 > use_agent 布尔兼容字段 > 默认 standard
    _raw_agent_mode = getattr(body, "agent_mode", None) or None
    if _raw_agent_mode in ("standard", "assist", "autonomous"):
        agent_mode = _raw_agent_mode
    else:
        _legacy_agent = body.use_agent if hasattr(body, "use_agent") and body.use_agent is not None else False
        agent_mode = "assist" if _legacy_agent else "standard"

    # ── 1. 请求概览 ──
    _chat_logger.info(
        "[chat] ▶ 新请求 | session=%s | msg=%r | provider=%s | model=%s | search_mode=%s"
        " | collection=%s | agent_mode=%s"
        " | local_top_k=%s | step_top_k=%s | reranker_mode=%s | web_providers=%s | serpapi_ratio=%s"
        " | year=%s~%s",
        session_id[:12], message[:60],
        body.llm_provider or "default", body.model_override or "default",
        search_mode, (body.collection or settings.collection.global_), agent_mode,
        body.local_top_k, body.step_top_k,
        body.reranker_mode,
        ",".join(body.web_providers) if body.web_providers else "none",
        body.serpapi_ratio,
        body.year_start, body.year_end,
    )

    # ── 2. 意图 + 上下文分析（单次 ultra-lite LLM 调用）──
    request_mode = (body.mode or "chat").strip().lower()
    ctx_analysis: ContextAnalysis | None = None
    _cmd_token: str = ""  # set when message starts with "/"

    if request_mode == "deep_research":
        parsed = ParsedIntent(
            intent_type=IntentType.DEEP_RESEARCH,
            confidence=1.0,
            params={"args": message},
            raw_input=message,
        )
        _chat_logger.info("[chat] ② 意图判断 → deep_research (前端 mode 指定)")
    elif message.startswith("/"):
        _cmd_token = message.split()[0].lower()
        parser = IntentParser(intent_client)
        parsed = parser.parse(message, current_stage=current_stage, history=history_for_intent)

        # 为 slash 命令合成 ContextAnalysis，使 followup_mode / topic_relevance 语义正确，
        # 避免命令始终回落到 "fresh"/"low" 默认值，并防止走老式 _classify_query 兜底。
        if _cmd_token == "/rewrite":
            # 重写已有回答：不做新检索，仅复用缓存 chunks 重新生成
            ctx_analysis = ContextAnalysis(
                action="chat",
                context_status="self_contained",
                rewritten_query=(parsed.params.get("args") or "").strip(),
                followup_mode="reuse_only",
                topic_relevance="high",
            )
        elif _cmd_token == "/deepen":
            _deepen_args = (parsed.params.get("args") or "").strip()
            if not _deepen_args:
                # 没有补充查询词，直接提示用户
                _step(None, "")
                _no_args_msg = "请在 /deepen 后面提供补充查询词，例如：`/deepen 深海热液口的温度范围`"
                memory.add_turn("user", message)
                memory.add_turn("assistant", _no_args_msg, citations=[])
                memory.update_rolling_summary(lite_client)
                return (
                    session_id, _no_args_msg, [],
                    EvidenceSummary(query=message, total_chunks=0, sources_used=[], retrieval_time_ms=0),
                    parsed, None, None, {}, None, False, None,
                )
            ctx_analysis = ContextAnalysis(
                action="rag",
                context_status="self_contained",
                rewritten_query=_deepen_args,
                followup_mode="reuse_and_search",
                topic_relevance="high",
            )
        elif _cmd_token in ("/search", "/explore"):
            # 显式检索命令：始终新鲜检索
            ctx_analysis = ContextAnalysis(
                action="rag",
                context_status="self_contained",
                followup_mode="fresh",
                topic_relevance="low",
            )
        else:
            # /outline, /draft, /edit, /status, /export, /set 等：纯对话，无检索
            ctx_analysis = ContextAnalysis(
                action="chat",
                context_status="self_contained",
                followup_mode="fresh",
                topic_relevance="low",
            )

        _chat_logger.info(
            "[chat] ② 意图判断 → %s (命令解析, confidence=%.2f) | cmd=%s → followup=%s action=%s",
            parsed.intent_type.value, parsed.confidence,
            _cmd_token, ctx_analysis.followup_mode, ctx_analysis.action,
        )
    else:
        # 统一上下文分析：一次 ultra-lite 调用完成 意图分类 + 指代检测 + query 重写/澄清
        _step("analyzing", "Analyzing query...")
        t_analyze = _time.perf_counter()
        ctx_analysis = analyze_chat_context(
            message=message,
            rolling_summary=memory.rolling_summary,
            history=history_for_intent,
            llm_client=intent_client,
        )
        analyze_ms = (_time.perf_counter() - t_analyze) * 1000

        # 关键词强制 rag（零成本兜底，覆盖 LLM 误判为 chat 的情况）
        if _FORCE_RAG_RE.search(message):
            ctx_analysis.action = "rag"

        if ctx_analysis.action == "deep_research":
            parsed = ParsedIntent(
                intent_type=IntentType.DEEP_RESEARCH,
                confidence=0.9,
                params={"args": message},
                raw_input=message,
            )
        else:
            parsed = ParsedIntent(
                intent_type=IntentType.CHAT,
                confidence=0.9,
                raw_input=message,
            )
        _chat_logger.info(
            "[chat] ② 上下文分析 → action=%s | context_status=%s | rewritten=%r | 耗时=%.0fms",
            ctx_analysis.action, ctx_analysis.context_status,
            (ctx_analysis.rewritten_query or "")[:60], analyze_ms,
        )
        _step(None, "")

    meta = store.get_session_meta(session_id)
    canvas_id = (meta or {}).get("canvas_id") or ""

    # ── Deep Research 分支 ──
    if is_deep_research(parsed):
        _step(None, "")
        _chat_logger.info("[work=research] 开始")
        _chat_logger.info("[chat] ── 进入 Deep Research 分支 ──")
        topic = (parsed.params.get("args") or message).strip() or "综述"
        meta_now = store.get_session_meta(session_id) or {}
        update_dr_meta = {"session_type": "research"}
        if not (meta_now.get("title") or "").strip():
            update_dr_meta["title"] = (topic or "Deep Research")[:80]
        store.update_session_meta(session_id, update_dr_meta)
        query = build_search_query_from_context(
            parsed, message, history_for_intent,
            llm_client=client, enforce_english_if_input_english=True,
            rolling_summary=memory.rolling_summary,
        )
        filters = _build_filters(body)
        max_sections = getattr(body, "max_sections", None) or 4
        svc = AutoCompleteService(llm_client=client, max_sections=max_sections, include_abstract=True)
        user_id = optional_user_id or body.user_id or ""
        use_agent_flag = getattr(body, "use_agent", None) or False
        result = svc.complete(
            topic=topic,
            canvas_id=canvas_id or None,
            session_id=session_id,
            search_mode=search_mode,
            filters=filters or None,
            user_id=user_id,
            clarification_answers=body.clarification_answers,
            output_language=body.output_language or "auto",
            step_models=body.step_models,
            use_agent=use_agent_flag,
        )
        response_text = result.markdown
        if result.canvas_id and not canvas_id:
            store.update_session_meta(session_id, {"canvas_id": result.canvas_id})
        store.update_session_stage(session_id, "refine")
        memory.add_turn("user", message)
        citations_data = [_serialize_citation(c) for c in result.citations] if result.citations else []
        memory.add_turn("assistant", response_text, citations=citations_data)
        memory.update_rolling_summary(lite_client)
        dr_pstats = _compute_citation_provider_stats(result.citations) if result.citations else None
        evidence_summary = EvidenceSummary(
            query=topic,
            total_chunks=len(result.citations),
            sources_used=_infer_sources_from_citations(result.citations),
            retrieval_time_ms=result.total_time_ms,
            provider_stats=dr_pstats,
        )
        dashboard_data = getattr(result, "dashboard", None)
        if dashboard_data is None and hasattr(result, "__dict__"):
            dashboard_data = getattr(result, "dashboard", None)
        elapsed = (_time.perf_counter() - t_start) * 1000
        _chat_logger.info(
            "[chat] ✔ Deep Research 完成 | citations=%d | time=%.0fms",
            len(result.citations) if result.citations else 0, elapsed,
        )
        return session_id, response_text, result.citations, evidence_summary, parsed, dashboard_data, None, {}, None, False, None

    _chat_logger.info("[work=chat] 第1轮")
    # ── 2½. 指代澄清短路（LLM 判定无法推断时直接追问用户）──
    if ctx_analysis and ctx_analysis.context_status == "needs_clarification" and ctx_analysis.clarification:
        _step(None, "")
        _chat_logger.info("[chat] ②½ 指代不明 → 返回澄清问题，跳过检索")
        memory.add_turn("user", message)
        memory.add_turn("assistant", ctx_analysis.clarification, citations=[])
        memory.update_rolling_summary(lite_client)
        empty_evidence = EvidenceSummary(
            query=message,
            total_chunks=0,
            sources_used=[],
            retrieval_time_ms=0,
        )
        return (
            session_id,
            ctx_analysis.clarification,
            [],
            empty_evidence,
            parsed,
            None,
            None,
            {},
            None,
        )

    # ── 3. 查询路由（基于上下文分析结果）──
    if ctx_analysis:
        query_needs_rag = ctx_analysis.action == "rag"
    else:
        # /command 或前端指定 mode 时，走原有 _classify_query 兜底
        query_needs_rag = _classify_query(message, history_for_intent, intent_client)
    followup_mode = (ctx_analysis.followup_mode if ctx_analysis else "fresh").strip().lower()
    if followup_mode not in ("fresh", "reuse_only", "reuse_and_search"):
        followup_mode = "fresh"
    topic_relevance = (ctx_analysis.topic_relevance if ctx_analysis else "medium").strip().lower()
    if topic_relevance not in ("low", "medium", "high"):
        topic_relevance = "medium"
    if followup_mode in ("reuse_only", "reuse_and_search") and topic_relevance == "low":
        topic_relevance = "medium"
    target_span = (ctx_analysis.target_span if ctx_analysis else "").strip()
    # 追问高相关场景优先复用已有证据；是否补搜交给后续 Agent/证据缺口判断。
    # 仅 fresh（低相关/新主题）走常规预检索。
    do_retrieval = (
        search_mode != "none"
        and query_needs_rag
        and followup_mode == "fresh"
        and agent_mode != "autonomous"
    )
    _agent_mode_before_route = agent_mode
    if not query_needs_rag and followup_mode != "reuse_and_search":
        agent_mode = "standard"

    _chat_logger.info(
        "[chat] ③ 查询路由 → %s | do_retrieval=%s | followup_mode=%s relevance=%s target=%r"
        " (search_mode=%s, agent_mode=%s)",
        "rag" if query_needs_rag else "chat",
        do_retrieval, followup_mode, topic_relevance, target_span[:40], search_mode, agent_mode,
    )
    if _agent_mode_before_route != agent_mode:
        _chat_logger.info(
            "[chat] ③ ⚠ agent_mode 被降级: %s → %s (原因: query_needs_rag=%s, ctx_action=%s)",
            _agent_mode_before_route, agent_mode, query_needs_rag,
            ctx_analysis.action if ctx_analysis else "N/A",
        )
    dbg.log_query_route(
        session_id,
        message=message[:200],
        decision="rag" if query_needs_rag else "chat",
        do_retrieval=do_retrieval,
        search_mode=search_mode,
        agent_mode=agent_mode,
        latency_ms=0,
    )

    # ── 3.5 读取会话证据缓存（用于 follow-up reuse）──
    _reuse_cache_turns = int(getattr(settings.search, "chat_followup_cache_turns", 4) or 4)
    cached_reuse_chunks = _load_recent_cached_chunks(
        store,
        session_id,
        max_turns=_reuse_cache_turns,
        pool_name="final_chunks",
    )
    cached_followup_stage1_chunks = _load_recent_cached_chunks(
        store,
        session_id,
        max_turns=min(_FOLLOWUP_FINAL_STAGE_TURNS, _reuse_cache_turns),
        pool_name="final_chunks",
    )
    cached_followup_candidate_chunks = _load_recent_cached_chunks(
        store,
        session_id,
        max_turns=_reuse_cache_turns,
        pool_name="candidate_pool",
    )
    reuse_enabled = (
        followup_mode in ("reuse_only", "reuse_and_search")
        and topic_relevance in ("medium", "high")
        and len(cached_reuse_chunks) > 0
    )
    # /rewrite 无缓存时：无法"重写已有内容"，直接告知用户而非回落到新检索
    if _cmd_token == "/rewrite" and not reuse_enabled:
        _step(None, "")
        _chat_logger.info("[chat] ③½ /rewrite: no cached context → inform user")
        _rewrite_fallback = (
            "没有找到可以重写的上下文——当前会话中还没有保留的检索证据。"
            "请先进行一次检索查询，再使用 /rewrite 整理结果。"
        )
        memory.add_turn("user", message)
        memory.add_turn("assistant", _rewrite_fallback, citations=[])
        memory.update_rolling_summary(lite_client)
        return (
            session_id, _rewrite_fallback, [],
            EvidenceSummary(query=message, total_chunks=0, sources_used=[], retrieval_time_ms=0),
            parsed, None, None, {}, None, False, None,
        )

    if followup_mode in ("reuse_only", "reuse_and_search") and not reuse_enabled:
        _chat_logger.info(
            "[chat] ③½ %s requested but cache unavailable/low relevance → fallback fresh",
            followup_mode,
        )
        followup_mode = "fresh"
        do_retrieval = (
            search_mode != "none"
            and query_needs_rag
            and agent_mode != "autonomous"
        )

    # ── 4. 检索 query 构建（检索或复用时）──
    if do_retrieval or reuse_enabled:
        # 若上下文分析已完成指代补全，使用 rewritten_query 作为输入（避免重复 LLM 调用）
        effective_msg = (
            ctx_analysis.rewritten_query
            if ctx_analysis and ctx_analysis.context_status == "resolved" and ctx_analysis.rewritten_query
            else message
        )
        t_query = _time.perf_counter()
        query = build_search_query_from_context(
            parsed, effective_msg, history_for_intent,
            llm_client=client, enforce_english_if_input_english=True,
            rolling_summary=memory.rolling_summary,
        )
        query_ms = (_time.perf_counter() - t_query) * 1000
        _chat_logger.info(
            "[chat] ④ Query 构建 | original=%r → effective=%r → query=%r | mode=%s | 耗时=%.0fms",
            message[:40], effective_msg[:40], query[:60], followup_mode, query_ms,
        )
        dbg.log_query_build(
            session_id,
            original_message=message[:500],
            rewritten_query=query[:500],
            rolling_summary=(memory.rolling_summary or "")[:500],
            latency_ms=round(query_ms),
        )
    else:
        query = message
        _chat_logger.info("[chat] ④ Query 构建 → 跳过 (无需检索/复用)")

    # ── 4½. 本会话「不用本地库」偏好（session 变量）──
    meta = store.get_session_meta(session_id) or {}
    session_preferences = meta.get("preferences") or {}
    effective_use_local = session_preferences.get("local_db") != "no_local"
    effective_search_mode = search_mode
    if not effective_use_local and search_mode in ("local", "hybrid"):
        effective_search_mode = "web" if search_mode == "hybrid" else "none"
        _chat_logger.info("[chat] ④½ 本会话已选不用本地库 → effective_search_mode=%s", effective_search_mode)
    if effective_search_mode == "none" and do_retrieval:
        do_retrieval = False

    # ── 4¾. 解析目标知识库列表，并执行范围检查（单库）或配额分配（多库）──
    # Resolve target_collections from body.collections > body.collection > default
    _raw_collections = body.collections if getattr(body, "collections", None) else None
    if _raw_collections:
        target_collections = [c.strip() for c in _raw_collections if (c or "").strip()]
    elif (body.collection or "").strip():
        target_collections = [(body.collection or "").strip()]
    else:
        target_collections = []
    # Derive single-collection shortcut (for backward-compat paths)
    target_collection = target_collections[0] if target_collections else None
    actual_collection = target_collection or settings.collection.global_
    # Multi-collection: use quota allocation; skip single-collection mismatch prompt
    _is_multi_collection = len(target_collections) > 1
    _collection_quotas: Dict[str, float] = {}

    if (
        do_retrieval
        and effective_use_local
        and effective_search_mode in ("local", "hybrid")
        and actual_collection
        and query
    ):
        if _is_multi_collection:
            # LLM allocates per-collection retrieval quota based on scope relevance
            _collection_quotas = allocate_collection_quotas(query, target_collections, lite_client)
            _chat_logger.info("[chat] ④¾ 多库配额分配 | collections=%s quotas=%s", target_collections, _collection_quotas)
        else:
            scope_result = check_query_collection_scope(actual_collection, query, lite_client)
            # 用户已在本会话中明确选择「仍使用当前库」时，不再提示，直接走检索
            if scope_result == "mismatch" and session_preferences.get("local_db") != "use":
                _step(None, "")
                mismatch_msg = (
                    f"当前问题与本地知识库（{actual_collection}）主题可能不符。"
                    "您可以选择：**本会话暂不使用本地库**（仅用网络检索），或**仍使用当前库**继续检索。"
                )
                _chat_logger.info("[chat] ④¾ 查询与本地库范围不符 → 提示用户选择")
                memory.add_turn("user", message)
                memory.add_turn("assistant", mismatch_msg, citations=[])
                memory.update_rolling_summary(lite_client)
                empty_evidence = EvidenceSummary(
                    query=query or message,
                    total_chunks=0,
                    sources_used=[],
                    retrieval_time_ms=0,
                )
                return (
                    session_id,
                    mismatch_msg,
                    [],
                    empty_evidence,
                    parsed,
                    None,
                    None,
                    {},
                    None,
                    True,
                    mismatch_msg,
                )

    # ── 4.9 Sonar 前置知识（可选）：sonar_strength 不为 off 时调用 Perplexity Sonar，注入 system prompt 并纳入引文池 ──
    preliminary_knowledge_block = ""
    sonar_chunks: list = []
    _sonar_strength = (getattr(body, "sonar_strength", None) or "").strip().lower() or None
    _use_sonar = _sonar_strength and _sonar_strength != "off"
    if not _use_sonar and getattr(body, "use_sonar_prelim", False):
        _use_sonar = True
        _sonar_strength = (getattr(body, "sonar_model", None) or "").strip() or "sonar-reasoning-pro"
    elif _use_sonar and not _sonar_strength:
        _sonar_strength = "sonar-reasoning-pro"
    if do_retrieval and _use_sonar:
        _step("pre_research", "Pre-research")
        _prelim_provider = None
        if manager.is_available("sonar"):
            _prelim_provider = "sonar"
        elif manager.is_available("perplexity"):
            _prelim_provider = "perplexity"
        if _prelim_provider:
            _prelim_model = _sonar_strength or "sonar-reasoning-pro"
            try:
                _ppl_client = manager.get_client(_prelim_provider)
                # 使用已补全的检索 query（context resolved 时为完整主题句），避免原始短消息导致前置知识偏离主题
                _prelim_query = (query or message or "").strip()
                _prelim_prompt = (
                    "Provide a brief, high-level overview answering the following question. "
                    "Outline key points and main sources. Keep it under 350 words. Respond in the same language as the question.\n\n"
                ) + _prelim_query
                # 流量感知超时：每 chunk 间隔不超过 idle 秒；deep-research 给更长窗口
                _idle_to = 150 if "deep-research" in (_prelim_model or "").lower() else 90
                _prelim_resp = _stream_and_collect(
                    _ppl_client,
                    [{"role": "user", "content": _prelim_prompt}],
                    model=_prelim_model,
                    idle_timeout_seconds=_idle_to,
                )
                _prelim_text = (_prelim_resp.get("final_text") or "").strip()
                if _prelim_text:
                    # Sonar reasoning-pro may wrap answer in thinking tags; strip for prelim block
                    _close = "</think>"
                    if _close in _prelim_text:
                        _idx = _prelim_text.find(_close)
                        _prelim_text = _prelim_text[_idx + len(_close) :].lstrip()
                    preliminary_knowledge_block = "Preliminary knowledge (Sonar):\n" + _prelim_text
                _raw = _prelim_resp.get("raw") or {}
                _citations = _raw.get("citations") or _prelim_resp.get("citations")
                _search_results = _raw.get("search_results") or _prelim_resp.get("search_results")
                sonar_chunks = parse_sonar_citations(
                    citations=_citations,
                    search_results=_search_results,
                    response_text=_prelim_text,
                    query=_prelim_query,
                )
                _chat_logger.info(
                    "[chat] ④.9 Sonar 前置知识 | provider=%s model=%s prelim_len=%d sonar_chunks=%d",
                    _prelim_provider, _prelim_model, len(preliminary_knowledge_block), len(sonar_chunks),
                )
            except Exception as _e:
                _chat_logger.debug("[chat] ④.9 Sonar 前置知识失败（静默降级）: %s", _e)
        _step(None, "")

    # Round 1 fallback: no Perplexity or sonar_strength=off → local LLM cognition (no search) for hybrid/web
    if (
        do_retrieval
        and effective_search_mode in ("hybrid", "web")
        and not preliminary_knowledge_block.strip()
    ):
        try:
            _cog_prompt = _pm.render("chat_local_cognition.txt", query=(query or message or "").strip())
            _cog_resp = lite_client.chat(
                [{"role": "user", "content": _cog_prompt}],
                model=body.model_override or None,
            )
            _cog_text = (_cog_resp.get("final_text") or "").strip()
            if _cog_text:
                preliminary_knowledge_block = "Preliminary knowledge (local):\n" + _cog_text
                _chat_logger.info(
                    "[chat] ④.9 本地认知(无搜索) | prelim_len=%d",
                    len(preliminary_knowledge_block),
                )
        except Exception as _e:
            _chat_logger.debug("[chat] ④.9 本地认知失败: %s", _e)
        _step(None, "")

    # 送入 LLM 的证据窗口上限：显式 write_top_k > step_top_k > fallback(15)
    _write_k_raw = getattr(body, "write_top_k", None) or body.step_top_k
    write_k = int(_write_k_raw) if _write_k_raw else 15
    if write_k <= 0:
        write_k = 15
    followup_stage2_top_k = _chat_effective_step_top_k(body.step_top_k or body.local_top_k or 20) or 20
    followup_candidate_pool_cap = _followup_candidate_pool_cap(followup_stage2_top_k)
    filters: Dict[str, Any] = {}
    chat_gap_candidates_hits: List[Dict[str, Any]] = []
    cache_candidate_pool_chunks: List[EvidenceChunk] = []
    retrieval = get_retrieval_service(collection=target_collection)

    # ── 5. 检索执行 ──
    if do_retrieval:
        _step("retrieval", "Searching (1+1+1)")
        t_retrieval = _time.perf_counter()
        filters = _build_filters(body)
        if filters.pop("fused_pool_score_threshold", None) is not None:
            _chat_logger.info(
                "[chat] fused_pool_score_threshold ignored in chat candidate-pool flow"
            )
        if filters.get("step_top_k") is None:
            _fallback_step = body.local_top_k if body.local_top_k is not None else 20
            filters["step_top_k"] = _chat_effective_step_top_k(_fallback_step)
        filters["trace_phase"] = "chat_main"
        filters["trace_section"] = "chat"
        filters["job_id"] = f"chat_{session_id}"
        # 强制 Chat 使用 bge_only reranker（速度优先）。UI 传入的 cascade/colbert
        # 在 hybrid 模式下仅用于写作阶段（Deep Research）。
        filters["reranker_mode"] = "bge_only"
        # Round 2: 1+1+1 结构化查询（仅 hybrid/web）；Sonar 不参与 1+1+1，仅 gap 补搜时用
        search_query = query or message
        _no_sonar = [
            p for p in (filters.get("web_providers") or [])
            if (p or "").strip().lower() != "sonar"
        ]
        if effective_search_mode in ("hybrid", "web"):
            structured = _generate_chat_structured_queries(
                search_query,  # 使用已补全的检索 query，避免短消息"请重新总结"等导致关键词漂移
                preliminary_knowledge_block,
                lite_client,
                model_override=body.model_override or None,
            )
            if structured:
                qpp = _chat_web_queries_from_1plus1plus1(structured, _no_sonar)
                if qpp:
                    filters["web_queries_per_provider"] = qpp
                    search_query = structured.get("recall") or search_query
                    _chat_logger.info(
                        "[chat] ⑤ Round2 1+1+1 | recall=%r | providers=%s",
                        (search_query or "")[:60],
                        list(qpp.keys()),
                    )
            else:
                _chat_logger.warning("[chat] ⑤ Round2 1+1+1 解析失败，使用单 query 检索")
        main_filters = dict(filters)
        main_filters["web_providers"] = _no_sonar

        if _is_multi_collection and _collection_quotas:
            # ── 5a. 多库并行检索 + 配额分配合并 ──
            total_step_k = filters.get("step_top_k") or 20
            active_quotas = {k: v for k, v in _collection_quotas.items() if v >= 0.05}
            if not active_quotas:
                active_quotas = _collection_quotas  # fallback: use all
            _retrieval_timeout_s = int(
                getattr(getattr(settings, "perf_retrieval", None), "timeout_seconds", 60) or 60
            )
            _web_soft_cap_s = int(
                getattr(getattr(settings, "perf_retrieval", None), "web_soft_wait_seconds", 500) or 500
            )
            _soft_wait_s = min(_retrieval_timeout_s * 5, _web_soft_cap_s)
            # Keep multi-collection wrapper timeout aligned with retrieval soft-wait.
            # Add 120s buffer for rerank/synthesis tail latency.
            _multi_col_wait_timeout_s = max(120, int(_soft_wait_s + 120))

            def _search_one_collection(col_name: str, ratio: float) -> EvidencePack:
                col_step_k = max(3, math.ceil(total_step_k * ratio))
                col_filters = dict(main_filters)
                col_filters["step_top_k"] = col_step_k
                col_filters["trace_phase"] = f"chat_main_col_{col_name}"
                svc = get_retrieval_service(collection=col_name)
                return svc.search(
                    query=search_query,
                    mode=effective_search_mode,
                    filters=col_filters or None,
                    top_k=body.local_top_k,
                    gap_candidates_hits=None,
                )

            col_packs: List[EvidencePack] = []
            if len(active_quotas) == 1:
                _single_col, _single_ratio = next(iter(active_quotas.items()))
                _pack = _search_one_collection(_single_col, _single_ratio)
                col_packs.append(_pack)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_quotas)) as _col_ex:
                    futures_map = {
                        _col_ex.submit(_search_one_collection, col, ratio): col
                        for col, ratio in active_quotas.items()
                    }
                    try:
                        for fut in concurrent.futures.as_completed(
                            futures_map,
                            timeout=_multi_col_wait_timeout_s,
                        ):
                            col_name = futures_map[fut]
                            try:
                                _pack = fut.result()
                                col_packs.append(_pack)
                            except Exception as _ce:
                                _chat_logger.warning("[chat] ⑤ 多库检索 col=%s 失败: %s", col_name, _ce)
                    except concurrent.futures.TimeoutError as _te:
                        raise

            # Merge all chunks: dedup by chunk_id, sort by relevance_score desc, truncate to write_top_k
            merged_chunks: List[EvidenceChunk] = []
            seen_chunk_ids: set = set()
            for _cp in col_packs:
                for _c in _cp.chunks:
                    if _c.chunk_id not in seen_chunk_ids:
                        merged_chunks.append(_c)
                        seen_chunk_ids.add(_c.chunk_id)
            merged_chunks.sort(key=lambda c: getattr(c, "relevance_score", 0.0) or 0.0, reverse=True)

            merged_sources = list(dict.fromkeys(s for _cp in col_packs for s in _cp.sources_used))
            merged_time = sum(_cp.retrieval_time_ms for _cp in col_packs)
            pack = EvidencePack(
                query=search_query,
                chunks=merged_chunks,
                total_candidates=sum(_cp.total_candidates for _cp in col_packs),
                retrieval_time_ms=merged_time,
                sources_used=merged_sources,
            )
            _chat_logger.info(
                "[chat] ⑤ 多库合并 | collections=%s | per_pack_chunks=%s | merged=%d",
                list(active_quotas.keys()),
                [len(cp.chunks) for cp in col_packs],
                len(merged_chunks),
            )
        else:
            # ── 5b. 单库检索（原有路径）──
            retrieval = get_retrieval_service(collection=target_collection)
            pack = retrieval.search(
                query=search_query,
                mode=effective_search_mode,
                filters=main_filters or None,
                top_k=body.local_top_k,
                gap_candidates_hits=None,
            )
        # Sonar pre-research chunks belong to main pool; merge into pack for all modes.
        if sonar_chunks:
            _existing = {c.chunk_id for c in pack.chunks}
            for c in sonar_chunks:
                if c.chunk_id not in _existing:
                    pack.chunks.append(c)
                    _existing.add(c.chunk_id)
            if "sonar" not in pack.sources_used:
                pack.sources_used.append("sonar")
        cache_candidate_pool_chunks = _dedup_chunk_list(list(pack.chunks))
        # write_top_k = 混合检索后的最终保留数（送入 LLM 的 evidence 上限）。UI 传 ragConfig.writeTopK，此处生效。
        # Chat 单轮 Q&A = 一个产出单元；无 write_top_k 时使用默认窗口。
        max_chunks_for_context = min(write_k, len(pack.chunks))
        synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
        context_str, synthesis_meta = synthesizer.synthesize(pack)
        context_str, _ctx_budget_diag = _budget_chat_evidence_context(
            context_str,
            llm_client=lite_client,
            ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
            purpose="chat_retrieval_context",
        )
        synth_dict = synthesis_meta.to_dict()
        retrieval_ms = (_time.perf_counter() - t_retrieval) * 1000
        evidence_summary = EvidenceSummary(
            query=pack.query,
            total_chunks=len(pack.chunks),
            sources_used=pack.sources_used,
            retrieval_time_ms=pack.retrieval_time_ms,
            year_range=synth_dict.get("year_range"),
            source_breakdown=synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown"),
            evidence_type_breakdown=synth_dict.get("evidence_type_breakdown"),
            cross_validated_count=synth_dict.get("cross_validated_count", 0),
            total_documents=synth_dict.get("total_documents", 0),
            diagnostics=pack.diagnostics,
        )
        if canvas_id:
            sync_evidence_to_canvas(canvas_id, pack)
        # 注意：citations 会在 LLM 生成后通过 resolve_response_citations() 获得
        citations: list[Citation] = []
        _diag = pack.diagnostics or {}
        _fusion_diag = _diag.get("pool_fusion") or {}
        _chat_logger.info(
            "[chat] ⑤ 检索完成 | mode=%s | top_k=%s | step_top_k=%s | write_top_k=%s | reranker_mode=%s"
            " | chunks=%d | context_max=%d | sources=%s | fusion(main=%s,gap=%s,out=%s)"
            " | ctx_budget(raw=%d,soft=%d,final=%d,summary=%s,hard_cap=%s)"
            " | soft_wait_ms=%s | 耗时=%.0fms",
            search_mode, body.local_top_k, filters.get("step_top_k"),
            getattr(body, "write_top_k", None),
            filters.get("reranker_mode", "bge_only"),
            len(pack.chunks), max_chunks_for_context, ",".join(pack.sources_used),
            _fusion_diag.get("main_in", "-"),
            _fusion_diag.get("gap_in", "-"),
            _fusion_diag.get("output_count", "-"),
            _ctx_budget_diag.get("raw_chars", 0),
            _ctx_budget_diag.get("after_soft_chars", 0),
            _ctx_budget_diag.get("final_chars", 0),
            _ctx_budget_diag.get("used_summary", False),
            _ctx_budget_diag.get("used_hard_cap", False),
            _diag.get("soft_wait_ms", "-"),
            retrieval_ms,
        )
        dbg.log_retrieval(
            session_id,
            query=query[:500],
            mode=search_mode,
            total_chunks=len(pack.chunks),
            sources_used=pack.sources_used,
            latency_ms=round(retrieval_ms),
            diagnostics=pack.diagnostics,
            source_breakdown=synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown"),
            year_range=synth_dict.get("year_range"),
            context_budget=_ctx_budget_diag,
        )
    else:
        context_str = ""
        pack = None
        evidence_summary = EvidenceSummary(
            query=query or message,
            total_chunks=0,
            sources_used=[],
            retrieval_time_ms=0,
        )
        citations: list[Citation] = []
        if query_needs_rag and reuse_enabled:
            _chat_logger.info("[chat] ⑤ 检索 → 跳过 fresh 主检索（follow-up reuse path）")
        else:
            _chat_logger.info("[chat] ⑤ 检索 → 跳过 (路由判定为 chat)")

    # ── 5.1 follow-up 证据复用（reuse_only / reuse_and_search）──
    followup_stage1_used = False
    followup_stage2_used = False
    followup_stage1_expand_reason = ""
    if reuse_enabled:
        reuse_diag: Dict[str, Any] = {
            "followup_mode": followup_mode,
            "topic_relevance": topic_relevance,
            "cached_chunk_count": len(cached_reuse_chunks),
            "fresh_chunk_count": len(pack.chunks) if pack else 0,
            "target_span": target_span,
            "stage1_used": False,
            "stage1_expand_reason": "",
            "stage1_chunk_count": 0,
            "stage2_chunk_count": 0,
            "candidate_pool_available": len(cached_followup_candidate_chunks) > 0,
            "fallback_invoked": False,
        }
        if followup_mode == "reuse_only":
            reuse_hits = [_chunk_to_hit(c) for c in cached_reuse_chunks]
            fusion_diag: Dict[str, Any] = {}
            fused_hits = fuse_pools_with_gap_protection(
                query=query or message,
                main_candidates=reuse_hits,
                gap_candidates=[],
                top_k=min(write_k, len(reuse_hits)),
                rank_pool_multiplier=float(getattr(settings.search, "chat_rank_pool_multiplier", 3.0)),
                reranker_mode="bge_only",
                diag=fusion_diag,
            )
            reused_chunks = [
                service_hit_to_chunk(h, h.get("_source_type", "dense"), query or message)
                for h in fused_hits
            ]
            reuse_diag["reuse_selected"] = len(reused_chunks)
            reuse_diag["fresh_chunk_count"] = 0
            reuse_diag["pool_fusion"] = fusion_diag.get("pool_fusion")
            pack = EvidencePack(
                query=query or message,
                chunks=reused_chunks,
                total_candidates=len(reuse_hits),
                retrieval_time_ms=0.0,
                sources_used=list(
                    dict.fromkeys(
                        (c.provider or ("local" if c.source_type in ("dense", "graph") else "web"))
                        for c in reused_chunks
                    )
                ),
                diagnostics={"followup_reuse": reuse_diag},
            )
            cache_candidate_pool_chunks = _dedup_chunk_list(list(cached_reuse_chunks))
            max_chunks_for_context = min(write_k, len(pack.chunks))
            synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
            context_str, synthesis_meta = synthesizer.synthesize(pack)
            context_str, _ctx_budget_diag = _budget_chat_evidence_context(
                context_str,
                llm_client=lite_client,
                ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
                purpose="chat_followup_reuse_only",
            )
            synth_dict = synthesis_meta.to_dict()
            evidence_summary = EvidenceSummary(
                query=pack.query,
                total_chunks=len(pack.chunks),
                sources_used=pack.sources_used,
                retrieval_time_ms=0.0,
                year_range=synth_dict.get("year_range"),
                source_breakdown=synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown"),
                evidence_type_breakdown=synth_dict.get("evidence_type_breakdown"),
                cross_validated_count=synth_dict.get("cross_validated_count", 0),
                total_documents=synth_dict.get("total_documents", 0),
                diagnostics=pack.diagnostics,
            )
            _chat_logger.info(
                "[chat] ⑤.1 followup reuse_only | cached=%d selected=%d write_k=%d",
                len(cached_reuse_chunks), len(pack.chunks), write_k,
            )
        elif followup_mode == "reuse_and_search":
            stage1_source_chunks = cached_followup_stage1_chunks or cached_reuse_chunks
            stage1_ranked_chunks = _rerank_followup_chunks(
                query=query or message,
                chunks=stage1_source_chunks,
                top_k=write_k,
                phase="chat_followup_stage1",
            )
            followup_stage1_expand_reason = _followup_stage1_expand_reason(
                query=query or message,
                message=message,
                is_deepen=_cmd_token == "/deepen",
                target_span=target_span,
                stage1_chunks=stage1_ranked_chunks,
            )
            reuse_diag["stage1_used"] = True
            reuse_diag["stage1_expand_reason"] = followup_stage1_expand_reason
            reuse_diag["stage1_chunk_count"] = len(stage1_ranked_chunks)
            cache_candidate_pool_chunks = _dedup_chunk_list(
                list(stage1_source_chunks) + list(cached_followup_candidate_chunks)
            )
            pack = EvidencePack(
                query=query or message,
                chunks=stage1_ranked_chunks,
                total_candidates=len(stage1_source_chunks),
                retrieval_time_ms=0.0,
                sources_used=list(
                    dict.fromkeys(
                        (c.provider or ("local" if c.source_type in ("dense", "graph") else "web"))
                        for c in stage1_ranked_chunks
                    )
                ),
                diagnostics={"followup_reuse": reuse_diag},
            )
            max_chunks_for_context = min(write_k, len(pack.chunks))
            synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
            context_str, synthesis_meta = synthesizer.synthesize(pack)
            context_str, _ctx_budget_diag = _budget_chat_evidence_context(
                context_str,
                llm_client=lite_client,
                ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
                purpose="chat_followup_stage1",
            )
            synth_dict = synthesis_meta.to_dict()
            evidence_summary = EvidenceSummary(
                query=pack.query,
                total_chunks=len(pack.chunks),
                sources_used=pack.sources_used,
                retrieval_time_ms=0.0,
                year_range=synth_dict.get("year_range"),
                source_breakdown=synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown"),
                evidence_type_breakdown=synth_dict.get("evidence_type_breakdown"),
                cross_validated_count=synth_dict.get("cross_validated_count", 0),
                total_documents=synth_dict.get("total_documents", 0),
                diagnostics=pack.diagnostics,
            )
            if followup_stage1_expand_reason:
                stage2_source_chunks = _dedup_chunk_list(
                    list(stage1_ranked_chunks) + list(cached_followup_candidate_chunks)
                )
                if cached_followup_candidate_chunks and stage2_source_chunks:
                    followup_stage2_used = True
                    stage2_ranked_chunks = _rerank_followup_chunks(
                        query=query or message,
                        chunks=stage2_source_chunks,
                        top_k=followup_stage2_top_k,
                        phase="chat_followup_stage2",
                    )
                    reuse_diag["stage2_chunk_count"] = len(stage2_ranked_chunks)
                    pack = EvidencePack(
                        query=query or message,
                        chunks=stage2_ranked_chunks,
                        total_candidates=len(stage2_source_chunks),
                        retrieval_time_ms=0.0,
                        sources_used=list(
                            dict.fromkeys(
                                (c.provider or ("local" if c.source_type in ("dense", "graph") else "web"))
                                for c in stage2_ranked_chunks
                            )
                        ),
                        diagnostics={"followup_reuse": reuse_diag},
                    )
                    max_chunks_for_context = min(write_k, len(pack.chunks))
                    synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
                    context_str, synthesis_meta = synthesizer.synthesize(pack)
                    context_str, _ctx_budget_diag = _budget_chat_evidence_context(
                        context_str,
                        llm_client=lite_client,
                        ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
                        purpose="chat_followup_stage2",
                    )
                    synth_dict = synthesis_meta.to_dict()
                    evidence_summary = EvidenceSummary(
                        query=pack.query,
                        total_chunks=len(pack.chunks),
                        sources_used=pack.sources_used,
                        retrieval_time_ms=0.0,
                        year_range=synth_dict.get("year_range"),
                        source_breakdown=synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown"),
                        evidence_type_breakdown=synth_dict.get("evidence_type_breakdown"),
                        cross_validated_count=synth_dict.get("cross_validated_count", 0),
                        total_documents=synth_dict.get("total_documents", 0),
                        diagnostics=pack.diagnostics,
                    )
                _chat_logger.info(
                    "[chat] ⑤.1 followup reuse_and_search | stage1=%d expand=%s candidate_pool=%d stage2=%d",
                    len(stage1_ranked_chunks),
                    followup_stage1_expand_reason,
                    len(cached_followup_candidate_chunks),
                    reuse_diag["stage2_chunk_count"],
                )
            else:
                followup_stage1_used = True
                _chat_logger.info(
                    "[chat] ⑤.1 followup reuse_and_search | stage1_direct=%d expand=no",
                    len(stage1_ranked_chunks),
                )

    # ── 5.5 证据充分性检查：LLM 判断是否有一致、实质的证据支撑（借鉴 DR），失败时用数量兜底 ──
    evidence_scarce = False
    _evidence_distinct_docs = 0
    _coverage_score: Optional[float] = None
    _sufficiency_reason: Optional[str] = None
    _broad_followup = _is_broad_followup_query(query or message, message)
    _targeted_followup = (
        followup_mode == "reuse_and_search"
        and (_cmd_token == "/deepen" or not _broad_followup)
    )
    _skip_full_sufficiency_for_stage1 = (
        followup_mode == "reuse_and_search"
        and followup_stage1_used
        and not followup_stage2_used
        and not followup_stage1_expand_reason
    )
    if pack:
        _doc_keys: set[str] = set()
        for c in pack.chunks:
            if getattr(c, "doi", None):
                _doc_keys.add(f"doi:{c.doi}")
            else:
                _doc_keys.add(c.doc_group_key)
        _evidence_distinct_docs = len(_doc_keys)
        # 数量兜底：无有效 context 或 LLM 失败时使用
        if _targeted_followup:
            _numeric_fallback = len(pack.chunks) < 2
        else:
            _numeric_fallback = len(pack.chunks) < 3 or _evidence_distinct_docs < 2
        if _skip_full_sufficiency_for_stage1:
            _chat_logger.info(
                "[chat] ⑤½ follow-up stage1 direct answer | chunks=%d | broad=%s",
                len(pack.chunks),
                _broad_followup,
            )
        elif query_needs_rag and context_str and context_str.strip():
            sufficiency = _evaluate_chat_evidence_sufficiency(
                message,
                context_str,
                lite_client,
                model_override=body.model_override or None,
            )
            _coverage_score = sufficiency.get("coverage_score")
            _sufficiency_reason = sufficiency.get("reason") or None
            _llm_ok = bool(sufficiency.get("ok"))
            if _llm_ok:
                _substantive_ok = bool(sufficiency.get("substantive_support"))
                _specificity_ok = bool(sufficiency.get("specificity_ok"))
                _consistency_ok = bool(sufficiency.get("consistency_ok"))
                _llm_insufficient = (
                    (sufficiency.get("sufficient") is False)
                    or (not _substantive_ok)
                    or (not _specificity_ok)
                    or (not _consistency_ok)
                )
                if _llm_insufficient:
                    evidence_scarce = True
                    _chat_logger.info(
                        "[chat] ⑤½ 证据不足(LLM) | coverage=%.2f | substantive=%s | specificity=%s | consistency=%s | reason=%r",
                        _coverage_score or 0.0,
                        _substantive_ok,
                        _specificity_ok,
                        _consistency_ok,
                        (_sufficiency_reason or "")[:80],
                    )
                else:
                    _chat_logger.info(
                        "[chat] ⑤½ 证据充分(LLM) | coverage=%.2f | substantive=%s | specificity=%s | consistency=%s",
                        _coverage_score or 0.0,
                        _substantive_ok,
                        _specificity_ok,
                        _consistency_ok,
                    )
            else:
                if _numeric_fallback:
                    evidence_scarce = True
                    _chat_logger.info(
                        "[chat] ⑤½ 证据不足(LLM失败→数量兜底) | chunks=%d | distinct_docs=%d",
                        len(pack.chunks), _evidence_distinct_docs,
                    )
                else:
                    _chat_logger.info(
                        "[chat] ⑤½ 证据评估(LLM失败) | 数量兜底判定为充分 | chunks=%d | distinct_docs=%d",
                        len(pack.chunks), _evidence_distinct_docs,
                    )
        else:
            # 未走 LLM 评估时用数量规则兜底
            if _numeric_fallback:
                evidence_scarce = True
                _chat_logger.info(
                    "[chat] ⑤½ 证据不足(数量兜底) | chunks=%d | distinct_docs=%d",
                    len(pack.chunks), _evidence_distinct_docs,
                )
        if evidence_scarce:
            evidence_summary.evidence_scarce = True
            if _coverage_score is not None:
                evidence_summary.coverage_score = _coverage_score
            if _sufficiency_reason:
                evidence_summary.sufficiency_reason = _sufficiency_reason
            if pack and pack.diagnostics and isinstance(pack.diagnostics.get("followup_reuse"), dict):
                pack.diagnostics["followup_reuse"]["fallback_invoked"] = True
                evidence_summary.diagnostics = pack.diagnostics
        if agent_mode == "standard" and evidence_scarce and query_needs_rag:
            agent_mode = "assist"
            _chat_logger.info(
                "[chat] ⑤½ 证据不足 → 自动升级 agent_mode: standard → assist (允许工具补搜)",
            )
    if do_retrieval:
        _step(None, "")

    # ── 5.6 证据不足时：生成 gap query、补搜、main+gap 一次融合（借鉴 DR）──
    if (
        pack
        and evidence_scarce
        and query_needs_rag
        and effective_search_mode != "none"
        and (do_retrieval or followup_mode == "reuse_and_search")
    ):
        _step("gap_fill", "Gap fill")
        if not filters:
            filters = _build_filters(body)
            if filters.pop("fused_pool_score_threshold", None) is not None:
                _chat_logger.info(
                    "[chat] fused_pool_score_threshold ignored in follow-up gap flow"
                )
            if filters.get("step_top_k") is None:
                filters["step_top_k"] = followup_stage2_top_k
            filters["reranker_mode"] = "bge_only"
        gap_queries = _generate_chat_gap_queries(
            message,
            context_str or "",
            lite_client,
            model_override=body.model_override or None,
        )
        gap_queries = [q for q in (gap_queries or []) if isinstance(q, dict) and (q.get("recall") or q.get("precision") or q.get("discovery"))][:3]
        if gap_queries:
            step_k = _chat_effective_step_top_k(body.step_top_k or body.local_top_k or 20) or 20
            supp_k = max(10, step_k)
            gap_candidates: List[Dict[str, Any]] = []
            _supp_total = 0
            web_providers = (filters or {}).get("web_providers") or []

            def _search_one_gap(gap_item: Dict[str, Any]) -> "EvidencePack":
                local_query = (gap_item.get("recall") or gap_item.get("precision") or gap_item.get("discovery") or "").strip()
                if not local_query:
                    return EvidencePack(query="", chunks=[], total_candidates=0, retrieval_time_ms=0, sources_used=[], diagnostics={})
                supp_filters = dict(filters or {})
                supp_filters["trace_phase"] = "chat_gap_supplement"
                supp_filters["pool_only"] = True
                qpp = web_queries_per_provider_from_1plus1plus1(gap_item, web_providers)
                if qpp:
                    supp_filters["web_queries_per_provider"] = qpp
                return retrieval.search(
                    query=local_query,
                    mode=effective_search_mode,
                    filters=supp_filters or None,
                    top_k=supp_k,
                )

            _chat_logger.info(
                "[chat] ⑤¾ gap 补搜开始（并行） | gap_queries=%d | queries=%s",
                len(gap_queries), [(g.get("recall") or g.get("precision") or "")[:60] for g in gap_queries],
            )
            _gap_parallel = max(1, int(getattr(settings.search, "chat_gap_parallel", 2)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(_gap_parallel, len(gap_queries))) as gap_ex:
                future_to_gq = {gap_ex.submit(_search_one_gap, gq): gq for gq in gap_queries}
                for fut in concurrent.futures.as_completed(future_to_gq):
                    gq = future_to_gq[fut]
                    try:
                        supp_pack = fut.result()
                        for c in supp_pack.chunks:
                            gap_candidates.append(_chunk_to_hit(c))
                        _supp_total += len(supp_pack.chunks)
                    except Exception as e:
                        _chat_logger.warning("[chat] gap supplement search failed for %r: %s", (gq.get("recall") or gq.get("precision") or "")[:50], e)
            if gap_candidates:
                chat_gap_candidates_hits = list(gap_candidates)
                cache_candidate_pool_chunks = _dedup_chunk_list(
                    list(cache_candidate_pool_chunks)
                    + [
                        service_hit_to_chunk(hit, hit.get("_source_type", "dense"), query or message)
                        for hit in gap_candidates
                    ]
                )
                # gap 候选暂存；不做中间 BGE rerank，延迟至下方统一融合步骤执行
                _chat_logger.info(
                    "[chat] ⑤¾ gap 候选收集完成 | gap_queries=%d | gap_candidates=%d（延迟至统一 BGE 融合）",
                    len(gap_queries), len(gap_candidates),
                )
                if pack is not None:
                    pack = EvidencePack(
                        query=pack.query,
                        chunks=pack.chunks,
                        total_candidates=pack.total_candidates + _supp_total,
                        retrieval_time_ms=pack.retrieval_time_ms,
                        sources_used=pack.sources_used,
                        diagnostics={
                            **(pack.diagnostics or {}),
                            "chat_gap_queries": gap_queries,
                            "chat_gap_supplement_chunks": len(gap_candidates),
                        },
                    )
        _step(None, "")

    # ── 5¾. 统一单次 BGE rerank：main + gap → pack.chunks / context_str ──
    # 整个 chat pipeline 唯一一次 BGE rerank（非 agent 路径此处即为最终排序；
    # agent 路径在此获得良好的初始上下文，agent 补充结果在下方第⑧步再做一次）。
    if do_retrieval and pack is not None:
        _main_pool_before_fusion = len(pack.chunks)
        _pre_agent_chunks, _pre_agent_fusion_diag = _fuse_chat_main_gap_agent_candidates(
            query=query or message,
            message=message,
            main_chunks=pack.chunks,
            gap_candidate_hits=chat_gap_candidates_hits,
            agent_chunks=[],
            write_k=write_k,
            filters=filters or {},
        )
        pack = EvidencePack(
            query=pack.query,
            chunks=_pre_agent_chunks,
            total_candidates=max(
                pack.total_candidates,
                _main_pool_before_fusion + len(chat_gap_candidates_hits),
            ),
            retrieval_time_ms=pack.retrieval_time_ms,
            sources_used=list(dict.fromkeys(
                (c.provider or ("local" if c.source_type in ("dense", "graph") else "web"))
                for c in _pre_agent_chunks
            )),
            diagnostics={**(pack.diagnostics or {}), "pool_fusion": _pre_agent_fusion_diag},
        )
        max_chunks_for_context = min(write_k, len(pack.chunks))
        synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
        context_str, synthesis_meta = synthesizer.synthesize(pack)
        context_str, _ctx_budget_diag = _budget_chat_evidence_context(
            context_str,
            llm_client=lite_client,
            ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
            purpose="chat_pre_agent_fusion_context",
        )
        synth_dict = synthesis_meta.to_dict()
        evidence_summary.query = pack.query
        evidence_summary.total_chunks = len(pack.chunks)
        evidence_summary.sources_used = pack.sources_used
        evidence_summary.year_range = synth_dict.get("year_range")
        evidence_summary.source_breakdown = synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown")
        evidence_summary.evidence_type_breakdown = synth_dict.get("evidence_type_breakdown")
        evidence_summary.cross_validated_count = synth_dict.get("cross_validated_count", 0)
        evidence_summary.total_documents = synth_dict.get("total_documents", 0)
        evidence_summary.diagnostics = pack.diagnostics
        if canvas_id:
            sync_evidence_to_canvas(canvas_id, pack)
        _chat_logger.info(
            "[chat] ⑤¾+ 单次 BGE 融合完成 | main=%d gap=%d → fused=%d"
            " | ctx_budget(raw=%d,soft=%d,final=%d,summary=%s,hard_cap=%s)",
            _main_pool_before_fusion, len(chat_gap_candidates_hits), len(_pre_agent_chunks),
            _ctx_budget_diag.get("raw_chars", 0),
            _ctx_budget_diag.get("after_soft_chars", 0),
            _ctx_budget_diag.get("final_chars", 0),
            _ctx_budget_diag.get("used_summary", False),
            _ctx_budget_diag.get("used_hard_cap", False),
        )

    # ── 6. System Prompt 组装 ──
    wf = run_workflow(
        current_stage,
        parsed.intent_type,
        topic="",
        context=context_str or "（本轮暂无检索结果）",
    )
    next_stage = wf["next_stage"]
    stage_prompt = wf.get("system_prompt") or ""

    # 如果是普通的 chat 意图，直接使用 RAG 问答 prompt，避免被工作流（综述大纲等）提示词污染
    is_normal_chat = parsed.intent_type.value == "chat"

    if not query_needs_rag:
        system_content = _pm.render("chat_direct_system.txt")
        prompt_mode = "chat_direct"
    elif is_normal_chat:
        if do_retrieval and context_str:
            system_content = build_synthesis_system_prompt(context_str)
            prompt_mode = "rag_synthesis"
        else:
            system_content = _build_system_with_context(context_str)
            prompt_mode = "rag_basic"
    elif stage_prompt:
        system_content = stage_prompt
        prompt_mode = "workflow_stage"
    elif do_retrieval and context_str:
        system_content = build_synthesis_system_prompt(context_str)
        prompt_mode = "rag_synthesis"
    else:
        system_content = _build_system_with_context(context_str)
        prompt_mode = "rag_basic"

    if preliminary_knowledge_block:
        system_content = system_content.rstrip() + "\n\n" + preliminary_knowledge_block + "\n"

    # 注入 rolling summary：早期被驱逐对话的压缩摘要，让 LLM 感知长会话历史背景
    _ROLLING_SUMMARY_INJECT_MAX = 3000  # 保守上限（~750 token），防止挤占主 context
    if memory.rolling_summary:
        _rs = memory.rolling_summary[:_ROLLING_SUMMARY_INJECT_MAX]
        system_content = system_content.rstrip() + (
            "\n\n【早期对话摘要】（本次会话已压缩的历史背景，供参考）\n" + _rs + "\n"
        )

    if canvas_id:
        wm = get_or_generate_working_memory(canvas_id, _CONFIG_PATH)
        if wm and wm.get("summary"):
            system_content = system_content.rstrip() + "\n\n【当前画布摘要】\n" + wm["summary"] + "\n"

    effective_user_id = optional_user_id or body.user_id
    if effective_user_id:
        profile = get_user_profile(effective_user_id)
        if profile and profile.get("preferences"):
            prefs_str = ", ".join(f"{k}={v}" for k, v in list(profile["preferences"].items())[:5])
            system_content = system_content.rstrip() + "\n\n【用户偏好】\n" + prefs_str + "\n"

    # Output language: match user question or respect explicit setting
    out_lang = (getattr(body, "output_language", None) or "auto").strip().lower()
    if out_lang == "en":
        system_content = system_content.rstrip() + "\n\nOutput language: Respond entirely in English.\n"
    elif out_lang == "zh":
        system_content = system_content.rstrip() + "\n\nOutput language: Respond entirely in Simplified Chinese (简体中文).\n"
    else:
        system_content = system_content.rstrip() + "\n\nOutput language: Respond in the exact same language as the user's question (e.g. if they ask in English, answer in English; if in Chinese, answer in Chinese).\n"

    _chat_logger.info(
        "[chat] ⑥ Prompt 组装 | mode=%s | stage=%s→%s | system_len=%d",
        prompt_mode, current_stage or "none", next_stage, len(system_content),
    )
    dbg.log_prompt_assembly(
        session_id,
        prompt_mode=prompt_mode,
        stage_transition=f"{current_stage or 'none'} → {next_stage}",
        system_content_len=len(system_content),
        system_content_preview=system_content[:2000],
        canvas_id=canvas_id or None,
    )

    store.update_session_stage(session_id, next_stage)

    # ── 7. 构建消息列表 ──
    # 加载全部活跃窗口（由 MAX_BUFFER_CHARS=40000 控制上界），
    # _apply_pre_send_prompt_budget 将按 token 预算做精确裁剪，无需在此固定 n=10
    history = memory.get_context_window()
    messages = [{"role": "system", "content": system_content}]
    for t in history:
        messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": message})

    # ── 8. LLM 生成 ──
    _followup_local_context_ready = (
        followup_mode == "reuse_and_search"
        and bool(context_str and context_str.strip())
        and not evidence_scarce
    )
    skip_assist_agent_for_sufficient_context = (
        agent_mode == "assist"
        and query_needs_rag
        and not evidence_scarce
        and (do_retrieval or _followup_local_context_ready)
    )
    use_agent = agent_mode == "autonomous" or (
        agent_mode == "assist" and not skip_assist_agent_for_sufficient_context
    )
    _chat_logger.info(
        "[chat] ⑦½ Agent 决策 | use_agent=%s | agent_mode=%s (用户请求=%s) | "
        "query_needs_rag=%s | search_mode=%s | evidence_scarce=%s | do_retrieval=%s",
        use_agent, agent_mode, _agent_mode_before_route,
        query_needs_rag, search_mode,
        evidence_scarce if pack else "N/A(未检索)",
        do_retrieval,
    )
    if skip_assist_agent_for_sufficient_context:
        _chat_logger.info(
            "[chat] ⑦½ Agent 跳过 | reason=sufficient_context | do_retrieval=%s | followup=%s | mode=assist→direct",
            do_retrieval, _followup_local_context_ready,
        )
    tool_trace = None

    # 根据模式注入不同的 Agent hint
    if agent_mode == "assist" and evidence_scarce:
        messages[0]["content"] = messages[0]["content"] + _pm.render(
            "chat_agent_evidence_scarce_hint.txt",
            chunk_count=len(pack.chunks) if pack else 0,
            distinct_docs=_evidence_distinct_docs,
            targeted_followup="yes" if _targeted_followup else "no",
            target_span=target_span or "当前追问目标",
        )
    elif agent_mode == "assist" and do_retrieval and context_str:
        messages[0]["content"] = messages[0]["content"] + _pm.render("chat_agent_hint.txt")
    elif agent_mode == "autonomous":
        messages[0]["content"] = messages[0]["content"] + _pm.render("chat_agent_autonomous_hint.txt")

    # 发送前进行消息预算控制：优先裁剪最旧历史，其次做系统提示硬上限保护。
    _pre_send_min_out = _AGENT_MIN_OUTPUT_TOKENS if use_agent else _CHAT_MIN_OUTPUT_TOKENS
    messages, _prompt_budget_diag = _apply_pre_send_prompt_budget(
        messages,
        model=body.model_override or None,
        min_output_tokens=_pre_send_min_out,
        mode="agent" if use_agent else "direct",
    )
    _chat_logger.info(
        "[chat] ⑦¼ Prompt预算 | mode=%s | tokens=%s→%s | messages=%d→%d | history_trimmed=%d | non_history_trimmed=%d | system_hard_cap=%s",
        "agent" if use_agent else "direct",
        _prompt_budget_diag.get("prompt_tokens_before", 0),
        _prompt_budget_diag.get("prompt_tokens_after", 0),
        _prompt_budget_diag.get("message_count_before", len(messages)),
        _prompt_budget_diag.get("message_count_after", len(messages)),
        _prompt_budget_diag.get("history_trimmed", 0),
        _prompt_budget_diag.get("non_history_trimmed", 0),
        _prompt_budget_diag.get("used_system_hard_cap", False),
    )

    gen_mode = agent_mode if use_agent else "direct"
    _chat_logger.info(
        "[chat] ⑦ LLM 生成 | mode=%s | provider=%s | model=%s | history_turns=%d | msg_count=%d",
        gen_mode, body.llm_provider or "default", body.model_override or "default",
        len(history), len(messages),
    )

    pre_agent_chunk_ids: set[str] = set()
    agent_chunk_ids: set[str] = set()
    if pack:
        pre_agent_chunk_ids = {c.chunk_id for c in pack.chunks}

    t_llm = _time.perf_counter()
    _regen_succeeded = True   # 默认：未走 regen 路径，答案是权威最终结果
    try:
        if use_agent:
            _step("agent", "Agent reasoning")
            start_agent_chunk_collector()
            # For multi-collection, pass the full list so agent tools can search across all
            set_tool_collection(target_collection, collections=target_collections if _is_multi_collection else None)
            set_tool_step_top_k(
                _chat_effective_step_top_k(body.step_top_k or body.local_top_k or 20)
            )
            set_agent_sonar_model(getattr(body, "agent_sonar_model", None) or "sonar-pro")
            routed_tools = get_routed_skills(
                message=message,
                current_stage=current_stage or "",
                search_mode=search_mode,
                allowed_web_providers=body.web_providers,
            )
            react_result = react_loop(
                messages=messages,
                tools=routed_tools,
                llm_client=client,
                max_iterations=getattr(body, "max_iterations", None) or 2,
                model=body.model_override or None,
                session_id=session_id,
                prompt_budget_min_output_tokens=_AGENT_MIN_OUTPUT_TOKENS,
                prompt_budget_safety_margin=_CHAT_PROMPT_SAFETY_MARGIN,
                max_tokens=None,
            )
            agent_extra_chunks = drain_agent_chunks()
            if agent_extra_chunks:
                cache_candidate_pool_chunks = _dedup_chunk_list(
                    list(cache_candidate_pool_chunks) + list(agent_extra_chunks)
                )
            agent_chunk_ids = {c.chunk_id for c in agent_extra_chunks}
            response_text = react_result.final_text.strip()
            tool_trace = react_result.tool_trace if react_result.tool_trace else None
            llm_ms = (_time.perf_counter() - t_llm) * 1000
            _chat_logger.info(
                "[chat] ⑧ Agent 完成 | iterations=%d | tools_called=%d | routed=%d/%d | 耗时=%.0fms",
                react_result.iterations, len(react_result.tool_trace),
                len(routed_tools), len(CORE_TOOLS), llm_ms,
            )

            # Agent evidence must re-enter final fused context before final answer generation.
            # pack.chunks 已是 main+gap BGE 融合结果；gap_candidate_hits 传空避免 gap 双重纳入。
            if agent_extra_chunks:
                if pack is None:
                    pack = EvidencePack(query=query or message, chunks=[])
                final_chunks, final_pool_fusion = _fuse_chat_main_gap_agent_candidates(
                    query=query or message,
                    message=message,
                    main_chunks=pack.chunks,
                    gap_candidate_hits=[],
                    agent_chunks=agent_extra_chunks,
                    write_k=write_k,
                    filters=filters or {},
                )
                base_diag = dict(pack.diagnostics or {})
                base_diag["pool_fusion"] = final_pool_fusion
                base_diag["agent_refusion"] = {
                    "agent_extra_chunks": len(agent_extra_chunks),
                    "gap_candidates": 0,  # gap 已融入 pack.chunks（pre-agent fusion）
                    "main_candidates": len(pack.chunks),
                    "output_count": len(final_chunks),
                }
                pack = EvidencePack(
                    query=pack.query,
                    chunks=final_chunks,
                    total_candidates=max(
                        pack.total_candidates,
                        len(pack.chunks) + len(chat_gap_candidates_hits) + len(agent_extra_chunks),
                    ),
                    retrieval_time_ms=pack.retrieval_time_ms,
                    sources_used=list(
                        dict.fromkeys(
                            (c.provider or ("local" if c.source_type in ("dense", "graph") else "web"))
                            for c in final_chunks
                        )
                    ),
                    diagnostics=base_diag,
                )
                max_chunks_for_context = min(write_k, len(pack.chunks))
                synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
                context_str, synthesis_meta = synthesizer.synthesize(pack)
                context_str, _ctx_budget_diag = _budget_chat_evidence_context(
                    context_str,
                    llm_client=lite_client,
                    ultra_lite_provider=getattr(body, "ultra_lite_provider", None),
                    purpose="chat_agent_refusion_context",
                )
                synth_dict = synthesis_meta.to_dict()
                evidence_summary.query = pack.query
                evidence_summary.total_chunks = len(pack.chunks)
                evidence_summary.sources_used = pack.sources_used
                evidence_summary.year_range = synth_dict.get("year_range")
                evidence_summary.source_breakdown = synth_dict.get("unique_source_breakdown") or synth_dict.get("source_breakdown")
                evidence_summary.evidence_type_breakdown = synth_dict.get("evidence_type_breakdown")
                evidence_summary.cross_validated_count = synth_dict.get("cross_validated_count", 0)
                evidence_summary.total_documents = synth_dict.get("total_documents", 0)
                evidence_summary.diagnostics = pack.diagnostics
                if canvas_id:
                    sync_evidence_to_canvas(canvas_id, pack)

                agent_notes = react_result.final_text.strip()
                regen_messages = [{"role": "system", "content": messages[0]["content"]}]
                for t in history:
                    regen_messages.append({"role": t.role, "content": t.content})
                regen_messages.append({"role": "assistant", "content": agent_notes})
                regen_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Please provide the final answer using the updated fused evidence context below. "
                            "If agent notes conflict with fused evidence, trust fused evidence.\n\n"
                            f"{context_str}"
                        ),
                    }
                )
                regen_messages, _ = _apply_pre_send_prompt_budget(
                    regen_messages,
                    model=body.model_override or None,
                    min_output_tokens=_CHAT_MIN_OUTPUT_TOKENS,
                    mode="direct",
                )
                _step("answering", "Writing answer")
                t_regen = _time.perf_counter()
                if delta_callback:
                    _regen_text = (_stream_llm_text(
                        client,
                        regen_messages,
                        model_override=body.model_override or None,
                    ) or "").strip()
                    _regen_succeeded = bool(_regen_text)
                    response_text = _regen_text or response_text
                else:
                    regen_resp = client.chat(regen_messages, model=body.model_override or None, max_tokens=None)
                    _regen_text = (regen_resp.get("final_text") or "").strip()
                    _regen_succeeded = bool(_regen_text)
                    response_text = _regen_text or response_text
                regen_ms = (_time.perf_counter() - t_regen) * 1000
                _chat_logger.info(
                    "[chat] ⑧b Agent 回流重生成 | fused_chunks=%d | agent_chunks=%d | regen_ms=%.0f",
                    len(pack.chunks), len(agent_extra_chunks), regen_ms,
                )
            else:
                if delta_callback and response_text:
                    for chunk in _chunk_text(response_text):
                        delta_callback(chunk)
                _chat_logger.info("[chat] ⑧b Agent 无新增检索块，复用 agent 最终回答")
            _step(None, "")
        else:
            _step("answering", "Writing answer")
            react_result = None
            if delta_callback:
                response_text = (_stream_llm_text(
                    client,
                    messages,
                    model_override=body.model_override or None,
                ) or "").strip()
                resp = {"meta": {"usage": None}}
            else:
                resp = client.chat(messages, model=body.model_override or None, max_tokens=None)
                response_text = (resp.get("final_text") or "").strip()
            llm_ms = (_time.perf_counter() - t_llm) * 1000
            usage = resp.get("meta", {}).get("usage") or resp.get("usage") or {}
            _chat_logger.info(
                "[chat] ⑧ LLM 完成 | response_len=%d | tokens=%s | 耗时=%.0fms",
                len(response_text),
                f"in={usage.get('prompt_tokens', '?')}/out={usage.get('completion_tokens', '?')}" if usage else "N/A",
                llm_ms,
            )
            dbg.log_llm_direct(
                session_id,
                response_len=len(response_text),
                tokens=dict(usage) if usage else None,
                latency_ms=round(llm_ms),
                message_count=len(messages),
                prompt_budget=_prompt_budget_diag,
            )
    except Exception as llm_err:
        _step(None, "")
        react_result = None
        llm_ms = (_time.perf_counter() - t_llm) * 1000
        _chat_logger.error("[chat] ⑧ LLM 失败 | error=%s | 耗时=%.0fms", llm_err, llm_ms)
        response_text = f"[LLM 调用失败] {type(llm_err).__name__}: {llm_err}\n\n请尝试切换其他模型。"

    # ── 9. 引文后处理：将 [ref_hash] 替换为正式 cite_key ──
    ref_map: dict[str, str] = {}
    all_chunks = (pack.chunks if pack else [])
    if all_chunks and _regen_succeeded:
        response_text, citations, ref_map = resolve_response_citations(
            response_text,
            all_chunks,
            include_unreferenced_documents=False,
        )
        if use_agent and len(all_chunks) > evidence_summary.total_chunks:
            evidence_summary.total_chunks = len(all_chunks)
            if "web" not in evidence_summary.sources_used:
                evidence_summary.sources_used.append("web")

        # 双层来源统计: chunk 级（每个信息块算一次）+ citation 级（同网站/文档只算一次）
        chunk_counts: dict[str, int] = {}
        for c in all_chunks:
            p = getattr(c, "provider", None) or ("local" if c.source_type in ("dense", "graph") else "web")
            chunk_counts[p] = chunk_counts.get(p, 0) + 1
        cite_counts: dict[str, int] = {}
        for c in citations:
            p = getattr(c, "provider", None) or ("local" if not c.url else "web")
            cite_counts[p] = cite_counts.get(p, 0) + 1
        evidence_summary.provider_stats = {
            "chunk_level": chunk_counts,
            "citation_level": cite_counts,
        }
        # source_breakdown 使用来源级（unique 文献/网页）统计
        evidence_summary.source_breakdown = cite_counts
        # sources_used 更新为具体 provider 列表（取代粗粒度的 dense/web）
        if cite_counts:
            evidence_summary.sources_used = list(cite_counts.keys())

        _chat_logger.info(
            "[chat] ⑨ 引文后处理 | cited_docs=%d | ref_map_size=%d | chunk_stats=%s | cite_stats=%s",
            len(citations), len(ref_map), chunk_counts, cite_counts,
        )

    # ── 9b. tools_contributed 判定 ──
    tools_contributed = False
    cited_from_agent_count = 0
    non_retrieval_tools_ok = 0
    if use_agent and react_result and react_result.tool_trace:
        _non_retrieval = {"run_code", "explore_graph", "canvas", "get_citations", "compare_papers", "summarize_quantitative"}
        for entry in react_result.tool_trace:
            if entry["tool"] in _non_retrieval and not entry.get("is_error"):
                non_retrieval_tools_ok += 1
        if agent_chunk_ids and citations:
            cited_chunk_ids = set()
            for c in all_chunks:
                if any(
                    (getattr(c, "doc_id", None) and getattr(c, "doc_id", None) == getattr(cit, "doc_id", None))
                    or (getattr(c, "url", None) and getattr(c, "url", None) == getattr(cit, "url", None))
                    for cit in citations
                ):
                    cited_chunk_ids.add(c.chunk_id)
            cited_from_agent_count = len(cited_chunk_ids & agent_chunk_ids)
        tools_contributed = cited_from_agent_count > 0 or non_retrieval_tools_ok > 0

        dbg.log_citation_resolve(
            session_id,
            cited_count=len(citations),
            ref_map_size=len(ref_map),
            pre_retrieval_chunks=len(pre_agent_chunk_ids),
            agent_added_chunks=len(agent_chunk_ids),
            cited_from_agent=cited_from_agent_count,
            non_retrieval_tools_ok=non_retrieval_tools_ok,
            tools_contributed=tools_contributed,
        )

    # ── 9c. 构建 agent_debug payload ──
    if use_agent and react_result:
        agent_debug_data = {
            "agent_stats": react_result.agent_stats,
            "tool_trace": react_result.tool_trace,
            "tools_contributed": tools_contributed,
            "pre_retrieval_chunks": len(pre_agent_chunk_ids),
            "agent_added_chunks": len(agent_chunk_ids),
            "cited_from_agent": cited_from_agent_count,
        }

    # ── 9d. 写入会话证据缓存（用于后续 follow-up reuse）──
    if pack and pack.chunks:
        try:
            _cache_turns = int(getattr(settings.search, "chat_followup_cache_turns", 4) or 4)
            _cache_chunks_per_turn = int(
                getattr(settings.search, "chat_followup_cache_chunks_per_turn", 36) or 36
            )
            _candidate_pool_chunks = _dedup_chunk_list(list(cache_candidate_pool_chunks))
            if _candidate_pool_chunks:
                _candidate_pool_chunks = _rerank_followup_chunks(
                    query=query or message,
                    chunks=_candidate_pool_chunks,
                    top_k=followup_candidate_pool_cap,
                    phase="chat_followup_cache_candidate_pool",
                )
            store.append_recent_evidence_cache(
                session_id=session_id,
                query=query or message,
                final_chunks=[_chunk_to_cache_payload(c) for c in pack.chunks],
                candidate_pool=[_chunk_to_cache_payload(c) for c in _candidate_pool_chunks],
                max_turns=_cache_turns,
                max_chunks_per_turn=_cache_chunks_per_turn,
                max_candidate_pool_per_turn=followup_candidate_pool_cap,
            )
        except Exception as cache_err:
            _chat_logger.debug("[chat] evidence cache write failed: %s", cache_err)

    # ── 9.5 Graphic Abstract ──
    if getattr(body, "enable_graphic_abstract", False) and response_text.strip() and _regen_succeeded:
        ga_model_raw = getattr(body, "graphic_abstract_model", None)
        ga_provider, ga_model = resolve_graphic_abstract_model(ga_model_raw)

        _step("graphic_abstract", f"Drawing Graphic Abstract ({ga_model})")
        _chat_logger.info(
            "[chat] 🎨 开始生成 Graphic Abstract, requested_model=%s, provider=%s, resolved_model=%s",
            ga_model_raw,
            ga_provider,
            ga_model,
        )
        try:
            # Predict the turn_id for the assistant message (current_count + 1)
            assistant_turn_id = (store.get_turn_count(session_id) or 0) + 1
            ga_result = render_graphic_abstract_markdown(
                response_text,
                model_raw=ga_model_raw,
                content_kind="chat",
                heading="### Graphic Abstract",
                session_id=session_id,
                turn_id=assistant_turn_id,
            )
            _chat_logger.info(
                "[chat] Graphic Abstract stored, backend=%s, key=%s, content_type=%s, url=%s",
                ga_result.storage_backend,
                ga_result.asset_key,
                ga_result.content_type,
                ga_result.image_url,
            )
            if delta_callback:
                delta_callback(ga_result.markdown)
            response_text += ga_result.markdown
        except Exception as e:
            _chat_logger.error("[chat] Graphic Abstract 生成失败: %s", e, exc_info=True)
            if delta_callback:
                delta_callback(GRAPHIC_ABSTRACT_FAILURE_MD)
            response_text += GRAPHIC_ABSTRACT_FAILURE_MD

    # ── 10. 写入 Memory ──
    memory.add_turn("user", message)
    citations_data = [_serialize_citation(c) for c in citations] if citations else []
    memory.add_turn("assistant", response_text, citations=citations_data)
    memory.update_rolling_summary(lite_client)

    # ── 11. 总结 ──
    total_ms = (_time.perf_counter() - t_start) * 1000
    _chat_logger.info(
        "[chat] ✔ 请求完成 | session=%s | route=%s | retrieval=%s | gen=%s "
        "| citations=%d | response_len=%d | 总耗时=%.0fms",
        session_id[:12],
        "rag" if query_needs_rag else "chat",
        f"{search_mode}({evidence_summary.total_chunks}chunks)" if do_retrieval else "skip",
        gen_mode,
        len(citations), len(response_text), total_ms,
    )
    dbg.log_turn_summary(
        session_id,
        route="rag" if query_needs_rag else "chat",
        gen_mode=gen_mode,
        citations=len(citations),
        response_len=len(response_text),
        total_ms=round(total_ms),
        tools_contributed=tools_contributed if use_agent else None,
        agent_stats=(react_result.agent_stats if react_result else None),
    )
    _step(None, "")
    return session_id, response_text, citations, evidence_summary, parsed, None, tool_trace, ref_map, agent_debug_data, False, None


def _run_chat(
    body: ChatRequest,
    optional_user_id: str | None = None,
    step_callback: Optional[Callable[[Optional[str], str], None]] = None,
    delta_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str], dict | None, bool, Optional[str]]:
    """包装 _run_chat_impl：当前端开启调试面板时临时提升本请求的日志级别为 DEBUG。"""
    with _request_debug_level(body):
        return _run_chat_impl(body, optional_user_id, step_callback, delta_callback)


def _citation_to_chat_citation(c: Citation) -> ChatCitation:
    """将 Citation 对象转换为 ChatCitation schema。"""
    return _chat_citation_from_dict(_serialize_citation(c))


@router.post("/chat", response_model=ChatResponse)
def chat_post(
    body: ChatRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> ChatResponse:
    (session_id, response_text, citations, evidence_summary, _parsed, _dashboard, _trace, _ref_map, _agent_debug,
     prompt_local_db_choice, local_db_mismatch_message) = _run_chat(body, optional_user_id)
    return ChatResponse(
        session_id=session_id,
        response=response_text,
        citations=[_citation_to_chat_citation(c) for c in citations],
        evidence_summary=evidence_summary,
        prompt_local_db_choice=prompt_local_db_choice or False,
        local_db_mismatch_message=local_db_mismatch_message,
    )


@router.post("/chat/submit", response_model=ChatSubmitResponse)
def chat_submit(
    body: ChatRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> ChatSubmitResponse:
    """异步提交 Chat 任务；返回 task_id，客户端通过 GET /chat/stream/{task_id} 订阅 SSE。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    payload = body.model_dump()
    store = get_session_store()
    if not body.session_id:
        created_session_id = store.create_session(
            canvas_id=body.canvas_id or "",
            user_id=optional_user_id or "",
        )
        payload["session_id"] = created_session_id
        # 提问时即用首条消息作为会话标题，历史里立即显示问题而非「未命名对话」
        first_msg = (body.message or "").strip()
        if first_msg:
            store.update_session_meta(created_session_id, {"title": first_msg[:80]})
    else:
        created_session_id = body.session_id
    payload["_optional_user_id"] = optional_user_id
    session_id = created_session_id or ""
    user_id = optional_user_id or ""
    try:
        task_id = q.submit(TaskKind.chat, session_id, user_id or "", payload)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue submit failed: {e}")
    if _obs_metrics and hasattr(_obs_metrics, "task_queue_submitted_total"):
        _obs_metrics.task_queue_submitted_total.labels(kind="chat").inc()
    return ChatSubmitResponse(task_id=task_id)


_SSE_HEARTBEAT_INTERVAL = 15  # 秒：每隔此时间发一次 SSE 注释保活，防止 proxy/浏览器因空闲超时断连


def _task_event_stream(task_id: str, q, after_id: str = "-"):
    """SSE 事件流生成器。

    关键顺序：先读取所有待推送事件，再检查是否终态。
    这样即使任务瞬间完成（如 mismatch 路径无 LLM 调用），
    meta/local_db_choice/delta/done 事件也能全部送达前端，
    不会被「已终态→直接 return」的提前退出所跳过。

    after_id: 断点续传起点（Redis stream ID）。前端重连时通过 Last-Event-ID
    header 传入，避免重放已收到的 delta 导致消息内容重复。

    心跳：任务运行期间每隔 _SSE_HEARTBEAT_INTERVAL 秒发一条 SSE 注释行
    (": heartbeat")，防止 Vite proxy / nginx 等因 idle timeout 断开连接。
    浏览器会忽略注释行，不触发任何事件。

    关键：在 while 循环开始前立即 yield 一个初始注释，强制 Starlette 立即
    发出 HTTP 响应头和第一个字节，避免浏览器/代理因等待首字节超时而报
    ERR_EMPTY_RESPONSE。
    """
    import time
    last_id = after_id
    last_sent_at = time.monotonic()
    if _obs_metrics:
        _obs_metrics.active_connections.inc()
    try:
        # 立即发送初始注释，让 HTTP 响应头和首字节立刻到达客户端
        yield ": stream-init\n\n"
        last_sent_at = time.monotonic()
        while True:
            # ① 先读取所有已推送但未发送的事件
            events = q.read_events(task_id, after_id=last_id)
            for ev in events:
                last_id = ev.get("id", last_id)
                typ = ev.get("type", "message")
                data = ev.get("data", {})
                # id: 字段让前端 EventSource / fetch 能追踪断点位置，用于重连续传
                yield f"id: {last_id}\nevent: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
                last_sent_at = time.monotonic()
                if typ in ("done", "error", "cancelled", "timeout"):
                    return

            # ② 再检查任务是否已进入终态（事件已读完则可以安全退出）
            state = q.get_state(task_id)
            if state and state.is_terminal():
                # 再读一次，防止「检查 state」和「读 events」之间有新事件写入
                events = q.read_events(task_id, after_id=last_id)
                for ev in events:
                    last_id = ev.get("id", last_id)
                    typ = ev.get("type", "message")
                    data = ev.get("data", {})
                    yield f"id: {last_id}\nevent: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
                    last_sent_at = time.monotonic()
                    if typ in ("done", "error", "cancelled", "timeout"):
                        return
                # 所有事件已发送完毕，补发终态通知
                yield f"event: {state.status.value}\ndata: {json.dumps({'status': state.status.value}, ensure_ascii=False)}\n\n"
                return

            # ③ 任务仍在运行：若已超过心跳间隔则发注释保活，防止 proxy idle timeout
            if time.monotonic() - last_sent_at >= _SSE_HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n"
                last_sent_at = time.monotonic()

            time.sleep(0.3)
    finally:
        if _obs_metrics:
            _obs_metrics.active_connections.dec()


@router.get("/chat/stream/{task_id}")
def chat_stream_by_task_id(task_id: str, request: Request):
    """SSE 订阅任务流式输出；事件由调度器在执行时推送。

    支持断点续传：前端重连时携带 Last-Event-ID header（标准 SSE 规范）或
    ?after_id= 查询参数，后端将只推送该 ID 之后的事件，避免 delta 重复叠加。
    """
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    after_id = (
        request.headers.get("Last-Event-ID")
        or request.query_params.get("after_id")
        or "-"
    )
    return StreamingResponse(
        _task_event_stream(task_id, q, after_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> StreamingResponse:
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    payload = body.model_dump()
    store = get_session_store()
    if not body.session_id:
        created_session_id = store.create_session(
            canvas_id=body.canvas_id or "",
            user_id=optional_user_id or "",
        )
        payload["session_id"] = created_session_id
        # 提问时即用首条消息作为会话标题，历史里立即显示问题而非「未命名对话」
        first_msg = (body.message or "").strip()
        if first_msg:
            store.update_session_meta(created_session_id, {"title": first_msg[:80]})
    else:
        created_session_id = body.session_id
    payload["_optional_user_id"] = optional_user_id
    session_id = created_session_id or ""
    user_id = optional_user_id or ""
    try:
        task_id = q.submit(TaskKind.chat, session_id, user_id or "", payload)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue submit failed: {e}")
    if _obs_metrics and hasattr(_obs_metrics, "task_queue_submitted_total"):
        _obs_metrics.task_queue_submitted_total.labels(kind="chat").inc()
    return StreamingResponse(
        _task_event_stream(task_id, q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/intent/detect", response_model=IntentDetectResponse)
def detect_intent(body: IntentDetectRequest) -> IntentDetectResponse:
    """
    意图检测 API（简化版）：Chat vs Deep Research 二分类。
    检索由前端 UI 开关决定，此处只判断执行模式。
    LLM 优先级：body.intent_provider > config.llm.intent_provider > body.llm_provider(lite) > body.ultra_lite_provider > config 默认。
    """
    manager = get_manager(str(_CONFIG_PATH))
    client = _get_intent_client(manager, body)
    parser = IntentParser(client)

    history = None
    if body.session_id:
        mem = load_session_memory(body.session_id)
        if mem:
            history = mem.get_context_window(n=6)

    parsed = parser.parse(
        body.message,
        current_stage=body.current_stage,
        history=history,
    )

    mode = parsed.intent_type.value  # "chat" or "deep_research"
    suggested_topic = ""
    if is_deep_research(parsed):
        suggested_topic = (parsed.params.get("args") or body.message).strip()

    return IntentDetectResponse(
        mode=mode,
        confidence=parsed.confidence,
        suggested_topic=suggested_topic,
        params=parsed.params or {},
        # 兼容旧字段
        intent_type=mode,
        needs_retrieval=True,  # 检索由 UI 决定，此字段不再有实际意义
        suggested_search_mode="hybrid",
    )


@router.post("/chat/suggestions", response_model=ChatSuggestionsResponse)
def chat_suggestions(body: ChatSuggestionsRequest) -> ChatSuggestionsResponse:
    """
    Chat 输入建议：根据前缀与会话历史返回候选（Google-style）。
    可选：后续可接入 ultra_lite 做排序或扩展。
    """
    prefix = (body.prefix or "").strip()
    limit = body.limit or 5
    suggestions: List[str] = []

    if body.session_id and prefix:
        store = get_session_store()
        turns = store.get_turns(body.session_id, limit=50, order_desc=True)
        seen: set = set()
        for t in turns:
            if t.role != "user" or not (t.content or "").strip():
                continue
            s = (t.content or "").strip()
            if s in seen:
                continue
            lower = s.lower()
            pre = prefix.lower()
            if lower.startswith(pre) or pre in lower:
                seen.add(s)
                suggestions.append(s)
                if len(suggestions) >= limit:
                    break

    return ChatSuggestionsResponse(suggestions=suggestions)


@router.post("/deep-research/clarify", response_model=ClarifyResponse)
def clarify_for_deep_research(
    body: ClarifyRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> ClarifyResponse:
    """
    Deep Research 澄清问题生成：基于 chat 历史和主题，
    至少生成 1 个关键澄清问题；若主题不明确则生成更多（最多 6 个）。
    前端在用户触发 Deep Research 时调用，显示澄清对话框。
    """
    t0 = time.perf_counter()
    _chat_logger.info(
        "[deep-research/clarify] begin | session=%s topic_chars=%d provider=%s model=%s",
        ((body.session_id or "").strip()[:12] or "(new)"),
        len((body.message or "").strip()),
        body.llm_provider or "(default)",
        body.model_override or "(default)",
    )
    manager = get_manager(str(_CONFIG_PATH))
    try:
        client = manager.get_client(body.llm_provider or None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid clarify LLM provider/model configuration: {e}")
    used_fallback = False
    fallback_reason = ""
    provider_used = body.llm_provider or manager.config.default
    model_used = body.model_override or ""
    try:
        cfg = getattr(client, "config", None)
        if not model_used and cfg is not None:
            model_used = str(getattr(cfg, "default_model", "") or "")
    except Exception:
        pass

    # Detect topic language for auto mode fallback
    _is_zh_topic = bool(re.search(r"[\u4e00-\u9fff]", body.message or ""))

    history_block = ""
    if body.session_id:
        mem = load_session_memory(body.session_id)
        if mem:
            turns = mem.get_context_window(n=8)
            lines = []
            for t in turns:
                role_label = "User" if t.role == "user" else "Assistant"
                text = (t.content or "").strip().replace("\n", " ")
                if len(text) > 300:
                    text = text[:300] + "..."
                lines.append(f"{role_label}: {text}")
            history_block = "\n".join(lines)

    # Determine output language: respect explicit output_language param, fallback to topic detection
    explicit_lang = (body.output_language or "auto").strip().lower()
    if explicit_lang == "zh":
        language_instruction = (
            "IMPORTANT: The user explicitly requests Chinese output. ALL output — including questions, "
            "suggested_topic, suggested_outline, and research_brief — MUST be in Chinese (中文)."
        )
    elif explicit_lang == "en":
        language_instruction = (
            "IMPORTANT: The user explicitly requests English output. ALL output — including questions, "
            "suggested_topic, suggested_outline, and research_brief — MUST be in English."
        )
    elif _is_zh_topic:
        language_instruction = (
            "IMPORTANT: The user's topic is in Chinese. ALL output — including questions, "
            "suggested_topic, suggested_outline, and research_brief — MUST be in Chinese (中文)."
        )
    else:
        language_instruction = (
            "IMPORTANT: The user's topic is in English. ALL output — including questions, "
            "suggested_topic, suggested_outline, and research_brief — MUST be in English."
        )

    store = get_session_store()
    session_id = (body.session_id or "").strip()
    if not session_id:
        session_id = store.create_session(
            session_type="research",
            user_id=optional_user_id or "",
        )

    # Set title + session_type immediately so the session shows a meaningful
    # name in the sidebar instead of "未命名对话".
    _topic_text = (body.message or "").strip()
    if _topic_text:
        _immediate_title = _topic_text[:80]
        try:
            store.update_session_meta(session_id, {"title": _immediate_title, "session_type": "research"})
        except Exception as _title_err:
            _chat_logger.debug("clarify: failed to set immediate title: %s", _title_err)

    # Preliminary knowledge via Perplexity/sonar for domain-aware clarification questions.
    # Use explicitly passed prelim_provider/model if available, otherwise fall back to global sonar-reasoning-pro.
    preliminary_knowledge_block = ""
    preliminary_knowledge_raw = ""
    _ui_prelim_provider = (body.prelim_provider or "").strip().lower()
    _ui_prelim_model = (body.prelim_model or "").strip() or None
    
    if _ui_prelim_provider in ("sonar", "perplexity") and manager.is_available(_ui_prelim_provider):
        prelim_provider = _ui_prelim_provider
        prelim_model = _ui_prelim_model or "sonar-reasoning-pro"
    else:
        prelim_provider = "perplexity" if manager.is_available("perplexity") else ("sonar" if manager.is_available("sonar") else None)
        prelim_model = "sonar-reasoning-pro"
    if prelim_provider and (body.message or "").strip():
        try:
            t_prelim = time.perf_counter()
            ppl_client = manager.get_client(prelim_provider)
            prelim_prompt = (
                "Provide a brief, high-level preliminary overview of the state of research for the following topic. "
                "Outline the main sub-fields, recent trends, and key controversies or open questions. "
                "Keep it under 300 words. Respond in English only.\n\nTopic: "
            ) + (body.message or "").strip()
            _idle_to = 150 if "deep-research" in (prelim_model or "").lower() else 90
            prelim_resp = _stream_and_collect(
                ppl_client,
                [{"role": "user", "content": prelim_prompt}],
                model=prelim_model,
                idle_timeout_seconds=_idle_to,
            )
            prelim_text = (prelim_resp.get("final_text") or "").strip()
            if prelim_text:
                preliminary_knowledge_raw = prelim_text
                preliminary_knowledge_block = "Preliminary knowledge about this topic (from web search):\n" + prelim_text
            _chat_logger.info(
                "[deep-research/clarify] preliminary done | provider=%s model=%s elapsed_ms=%.0f prelim_chars=%d",
                prelim_provider,
                prelim_model,
                (time.perf_counter() - t_prelim) * 1000.0,
                len(preliminary_knowledge_raw),
            )
        except Exception as _e:
            _chat_logger.debug("Perplexity preliminary knowledge failed: %s", _e)

    if preliminary_knowledge_raw:
        store.update_session_meta(
            session_id,
            {"preferences": {"preliminary_knowledge": preliminary_knowledge_raw}},
        )

    prompt = _pm.render(
        "chat_deep_research_clarify.txt",
        message=body.message,
        history_block=history_block or "(none)",
        language_instruction=language_instruction,
        preliminary_knowledge_block=preliminary_knowledge_block or "(none — use only the topic and history to generate questions)",
    )

    try:
        t_clarify_llm = time.perf_counter()
        _chat_logger.info(
            "[deep-research/clarify] calling main LLM | provider=%s model=%s prompt_chars=%d",
            provider_used,
            model_used or "(default)",
            len(prompt),
        )
        resp = client.chat(
            [
                {"role": "system", "content": _pm.render("chat_deep_research_system.txt")},
                {"role": "user", "content": prompt},
            ],
            model=body.model_override or None,

        )
        _chat_logger.info(
            "[deep-research/clarify] main LLM done | elapsed_ms=%.0f final_text_len=%d",
            (time.perf_counter() - t_clarify_llm) * 1000.0,
            len(resp.get("final_text") or ""),
        )
        text = (resp.get("final_text") or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
    except Exception as e:
        used_fallback = True
        fallback_reason = f"clarify_llm_failed: {str(e)[:240]}"
        # Use explicit output_language if set, otherwise fall back to topic detection
        _fallback_is_zh = (
            explicit_lang == "zh" or (explicit_lang == "auto" and _is_zh_topic)
        )
        _fallback_q_text = (
            "请确认本次研究最关键的目标与范围边界" if _fallback_is_zh
            else "Please confirm the key objectives and scope boundaries for this research"
        )
        data = {
            "suggested_topic": body.message,
            "suggested_outline": [],
            "questions": [
                {"id": "q1", "text": _fallback_q_text, "type": "text", "default": body.message},
            ],
        }

    questions = []
    for q in (data.get("questions") or [])[:6]:
        text = (q.get("text", "") or "").strip()
        if not text:
            continue
        qtype = q.get("type", "text")
        options = q.get("options", [])
        if qtype in ("choice", "multi_choice") and not isinstance(options, list):
            options = []
        questions.append(ClarifyQuestion(
            id=q.get("id", f"q{len(questions)+1}"),
            text=text,
            question_type=qtype,
            options=options or [],
            default=q.get("default", ""),
        ))

    if len(questions) == 0:
        used_fallback = True
        if not fallback_reason:
            fallback_reason = "empty_or_invalid_questions"
        questions.append(
            ClarifyQuestion(
                id="q1",
                text="请确认本次研究最关键的目标与范围边界",
                question_type="text",
                options=[],
                default=body.message,
            )
        )

    _chat_logger.info(
        "[deep-research/clarify] done | session=%s elapsed_ms=%.0f questions=%d used_fallback=%s prelim_chars=%d",
        session_id[:12],
        (time.perf_counter() - t0) * 1000.0,
        len(questions),
        used_fallback,
        len(preliminary_knowledge_raw),
    )
    return ClarifyResponse(
        questions=questions,
        session_id=session_id,
        suggested_topic=data.get("suggested_topic", body.message),
        suggested_outline=data.get("suggested_outline", []),
        research_brief=data.get("research_brief"),
        preliminary_knowledge=preliminary_knowledge_raw,
        used_fallback=used_fallback,
        fallback_reason=fallback_reason,
        llm_provider_used=provider_used,
        llm_model_used=model_used,
    )


def _extract_temp_context_from_file(file_name: str, raw_bytes: bytes) -> str:
    name = (file_name or "").lower()
    # 文本/Markdown 直接读取
    if name.endswith(".md") or name.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="ignore")
        return text.strip()[:12000]
    # PDF：优先使用 pypdf 做轻量抽取
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(BytesIO(raw_bytes))
            parts: List[str] = []
            for page in reader.pages[:40]:
                try:
                    parts.append((page.extract_text() or "").strip())
                except Exception:
                    continue
            text = "\n".join(x for x in parts if x)
            return text.strip()[:12000]
        except Exception:
            return ""
    return ""


@router.post("/deep-research/context-files", response_model=DeepResearchContextExtractResponse)
async def extract_deep_research_context_files(
    files: List[UploadFile] = File(...),
) -> DeepResearchContextExtractResponse:
    """提取用户上传文件为临时上下文（仅本次 Deep Research 使用，不入库）。"""
    documents: List[Dict[str, str]] = []
    for f in files:
        try:
            raw = await f.read()
        except Exception:
            continue
        if not raw:
            continue
        text = _extract_temp_context_from_file(f.filename or "file", raw)
        if not text:
            continue
        documents.append(
            {
                "name": (f.filename or "file").strip()[:200],
                "content": text,
            }
        )
    return DeepResearchContextExtractResponse(documents=documents)


@router.post("/deep-research/start", response_model=DeepResearchStartAsyncResponse)
def start_deep_research_endpoint(
    body: DeepResearchStartRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchStartAsyncResponse:
    """Phase 1: kick off scope+plan in background, return job_id immediately.
    Poll GET /deep-research/start/{job_id}/status for the result.
    Creates a persistent DB job entry with status='planning' so the job
    appears in sidebar immediately (like chat)."""
    from src.collaboration.research.job_store import create_job as _create_planning_job

    t0 = time.perf_counter()
    store = get_session_store()
    session_id = body.session_id or store.create_session(
        canvas_id=body.canvas_id or "",
        session_type="research",
        user_id=optional_user_id or "",
    )
    user_id = optional_user_id or ""
    meta = store.get_session_meta(session_id)
    preliminary_knowledge = ((meta or {}).get("preferences") or {}).get("preliminary_knowledge", "")

    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    filters = _build_deep_research_filters(body)
    job_id = str(uuid.uuid4())

    _chat_logger.info(
        "[deep-research/start] submit | session=%s topic=%r job=%s mode=%s local_top_k=%s step_top_k=%s providers=%s",
        session_id[:12],
        (body.topic or "")[:80],
        job_id[:12],
        body.search_mode,
        (filters or {}).get("local_top_k"),
        (filters or {}).get("step_top_k"),
        (filters or {}).get("web_providers"),
    )

    # Create persistent DB entry immediately so the job appears in sidebar
    try:
        _create_planning_job(
            topic=body.topic.strip(),
            session_id=session_id,
            canvas_id=body.canvas_id or "",
            job_id=job_id,
            status="planning",
            user_id=optional_user_id or "",
        )
    except Exception as _db_err:
        _chat_logger.warning("[deep-research/start] DB planning job creation failed: %s", _db_err)

    # Refine the session title: short topics used as-is, long ones summarised
    # via ultra-lite so the sidebar shows a concise readable title.
    _raw_topic = body.topic.strip()
    try:
        if len(_raw_topic) <= 50:
            _title = _raw_topic or "Deep Research"
            store.update_session_meta(session_id, {"title": _title, "session_type": "research"})
        else:
            store.update_session_meta(session_id, {"title": _raw_topic[:80], "session_type": "research"})
            _generate_and_set_session_title(store, session_id, _raw_topic, max_chars=50)
    except Exception as _title_err:
        _chat_logger.debug("[deep-research/start] title generation failed: %s", _title_err)

    with _START_JOBS_LOCK:
        _START_JOBS[job_id] = {"status": "running", "session_id": session_id, "result": None, "error": None}
    _evict_old_start_jobs()

    start_kwargs: Dict[str, Any] = dict(
        topic=body.topic.strip(),
        llm_client=client,
        canvas_id=body.canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=body.search_mode,
        filters=filters or None,
        max_sections=body.max_sections,
        clarification_answers=body.clarification_answers,
        preliminary_knowledge=preliminary_knowledge,
        output_language=body.output_language or "auto",
        model_override=body.model_override or None,
        step_models=body.step_models,
        step_model_strict=bool(body.step_model_strict),
    )

    t = threading.Thread(
        target=_run_start_job_bg,
        kwargs=dict(
            job_id=job_id,
            start_kwargs=start_kwargs,
            session_id=session_id,
            store=store,
        ),
        daemon=True,
        name=f"dr-start-{job_id[:8]}",
    )
    t.start()

    _chat_logger.info(
        "[deep-research/start] thread launched | job=%s session=%s elapsed_ms=%.0f",
        job_id[:12],
        session_id[:12],
        (time.perf_counter() - t0) * 1000.0,
    )
    return DeepResearchStartAsyncResponse(job_id=job_id, session_id=session_id)


@router.get("/deep-research/start/{job_id}/status", response_model=DeepResearchStartStatusResponse)
def get_start_phase_status(job_id: str) -> DeepResearchStartStatusResponse:
    """Poll start-phase job status. Returns outline/brief once status=='done'.

    Three-layer fallback: in-memory dict → DiskCache → DB planning job.
    This ensures the frontend never gets a 404 while the backend is alive.
    """
    with _START_JOBS_LOCK:
        job = _START_JOBS.get(job_id)
    if not job:
        job = _load_start_job_from_disk(job_id)
        if job:
            _chat_logger.info("[start-job] status served from disk cache job_id=%s status=%s", job_id[:12], job.get("status"))
    if not job:
        # Third layer: DB planning job (survives even full process restarts)
        from src.collaboration.research.job_store import get_job as _get_db_job
        db_job = _get_db_job(job_id)
        if db_job and db_job.get("status") in ("planning", "running", "pausing", "paused", "cancelled", "error"):
            db_status = db_job["status"]
            _chat_logger.info("[start-job] status served from DB job_id=%s status=%s", job_id[:12], db_status)
            return DeepResearchStartStatusResponse(
                job_id=job_id,
                status="running" if db_status in ("planning", "running", "pausing", "paused") else "error",
                session_id=db_job.get("session_id", ""),
                error=(
                    db_job.get("error_message")
                    if db_status == "error"
                    else ("任务已取消" if db_status == "cancelled" else None)
                ),
                canvas_id=db_job.get("canvas_id", ""),
                current_stage=db_job.get("message", "") or db_job.get("current_stage", ""),
                progress=0,
            )
        raise HTTPException(status_code=404, detail=f"Start job not found: {job_id}")

    raw_status = str(job.get("status", "running"))
    status = "running" if raw_status in {"paused", "pausing"} else ("error" if raw_status == "cancelled" else raw_status)
    session_id = job.get("session_id", "")
    error = job.get("error") or ("任务已取消" if raw_status == "cancelled" else None)
    result: Dict[str, Any] = job.get("result") or {}

    return DeepResearchStartStatusResponse(
        job_id=job_id,
        status=status,
        session_id=session_id,
        error=error,
        canvas_id=result.get("canvas_id", "") if result else "",
        brief=result.get("brief") or {} if result else {},
        outline=result.get("outline") or [] if result else [],
        initial_stats=result.get("initial_stats") or {} if result else {},
        current_stage=job.get("current_stage", ""),
        progress=int(job.get("progress", 0)),
    )


def _run_deep_research_job_safe(
    *,
    job_id: str,
    body: DeepResearchConfirmRequest,
    optional_user_id: str | None,
    restart_spec: dict[str, Any] | None = None,
) -> None:
    """后台执行 Deep Research 并持久化状态。"""
    from src.collaboration.research.agent import (
        prepare_deep_research_runtime,
        build_deep_research_result_from_state,
        reconstruct_state_from_checkpoint,
    )
    from src.collaboration.research.job_store import append_event, update_job, get_job, get_pending_review, load_checkpoint
    import time as _time

    t0 = _time.perf_counter()
    store = get_session_store()
    session_id = body.session_id or store.create_session(
        canvas_id=body.canvas_id or "",
        user_id=optional_user_id or "",
    )
    _ensure_research_session_title(store, session_id, body.topic)
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    user_id = optional_user_id or ""
    filters = _build_deep_research_filters(body)

    def _format_dr_step_label(event_type: str, payload: Dict[str, Any]) -> str:
        section = str((payload or {}).get("section") or "").strip()
        if event_type == "scope_done":
            return "Scoping"
        if event_type == "section_research_start":
            return f"Researching: {section}" if section else "Researching section"
        if event_type in ("research_main_1p1p1", "section_agent_supplement"):
            return f"Searching (1+1+1): {section}" if section else "Searching (1+1+1)"
        if event_type in ("evidence_insufficient", "quick_coverage_check"):
            return f"Checking coverage: {section}" if section else "Checking coverage"
        if event_type in ("tier1_sufficient", "tier2_sufficient", "tier3_start"):
            return f"Searching: {section}" if section else "Searching"
        if event_type in ("revise_started", "revise_queued"):
            return f"Revising: {section}" if section else "Revising"
        if event_type == "section_research_done":
            return f"Section done: {section}" if section else "Section done"
        if event_type == "section_verify_done":
            return f"Verifying: {section}" if section else "Verifying"
        if event_type == "evidence_optimization_start":
            return "Optimizing evidence"
        if event_type == "evidence_optimization_done":
            return "Evidence optimized"
        if section:
            return f"{event_type.replace('_', ' ').title()}: {section}"
        return event_type.replace("_", " ").title() if event_type else ""

    def _progress_cb(event_type: str, payload: Dict[str, Any]) -> None:
        append_event(job_id, "progress" if event_type != "warning" else "warning", {"type": event_type, **(payload or {})})
        stage = str(payload.get("section") or payload.get("type") or "")
        update_job(job_id, current_stage=stage, message=str(payload.get("message") or payload.get("section") or event_type))
        append_event(job_id, "step", {"step": event_type, "label": _format_dr_step_label(event_type, payload or {})})

    try:
        update_job(job_id, status="running", message="Deep Research 任务已启动", started_at=_time.time())
        append_event(job_id, "start", {"job_id": job_id, "topic": body.topic, "session_id": session_id})
        meta = store.get_session_meta(session_id)
        preliminary_knowledge = ((meta or {}).get("preferences") or {}).get("preliminary_knowledge", "")
        runtime_kwargs: Dict[str, Any] = {
            "topic": body.topic.strip(),
            "llm_client": client,
            "confirmed_outline": body.confirmed_outline,
            "confirmed_brief": body.confirmed_brief,
            "canvas_id": body.canvas_id,
            "session_id": session_id,
            "user_id": user_id,
            "search_mode": body.search_mode,
            "filters": filters or None,
            "model_override": body.model_override or None,
            "output_language": body.output_language or "auto",
            "step_models": body.step_models,
            "step_model_strict": bool(body.step_model_strict),
            "preliminary_knowledge": preliminary_knowledge,
            "user_context": body.user_context,
            "user_context_mode": body.user_context_mode or "supporting",
            "user_documents": body.user_documents,
            "progress_callback": _progress_cb,
            "cancel_check": lambda: _dr_is_cancel_requested(job_id),
            "pause_waiter": lambda: _dr_wait_if_paused(job_id),
            "review_waiter": lambda section_id: get_pending_review(job_id, section_id),
            "skip_draft_review": bool(body.skip_draft_review),
            "skip_refine_review": bool(body.skip_refine_review),
            "skip_claim_generation": bool(body.skip_claim_generation),
            "job_id": job_id,
            "depth": body.depth or "comprehensive",
            "max_sections": body.max_sections,
        }
        if restart_spec:
            source_job_id = str(restart_spec.get("source_job_id") or "").strip()
            mode = str(restart_spec.get("mode") or "").strip().lower()
            # plan 阶段重启允许无 checkpoint（用于中断/报错后从大纲阶段重新开始）
            if mode == "phase" and str(restart_spec.get("phase") or "").strip().lower() == "plan":
                runtime_kwargs["start_node"] = "plan"
                append_event(
                    job_id,
                    "restart_start",
                    {
                        "job_id": job_id,
                        "source_job_id": source_job_id,
                        "mode": mode,
                        "start_node": "plan",
                        "checkpoint_required": False,
                    },
                )
                checkpoint = None
            else:
                checkpoint = load_checkpoint(source_job_id)
                if not checkpoint:
                    # The direct source job may itself be a failed restart that never ran,
                    # so it has no checkpoint of its own.  Walk up the _restart chain until
                    # we find an ancestor that has a checkpoint (the original research job).
                    _walk_id = source_job_id
                    for _depth in range(10):
                        try:
                            _parent = get_job(_walk_id)
                            if not _parent:
                                break
                            _parent_req = _parent.get("request") or {}
                            _parent_restart = _parent_req.get("_restart") or {}
                            _grandparent_id = str(_parent_restart.get("source_job_id") or "").strip()
                            if not _grandparent_id or _grandparent_id == _walk_id:
                                break
                            _candidate = load_checkpoint(_grandparent_id)
                            if _candidate:
                                checkpoint = _candidate
                                _chat_logger.info(
                                    "[DR restart] resolved checkpoint via chain: %s -> %s (depth=%d)",
                                    source_job_id, _grandparent_id, _depth + 1,
                                )
                                break
                            _walk_id = _grandparent_id
                        except Exception:
                            break
                if not checkpoint:
                    raise RuntimeError(f"重启失败：未找到 checkpoint（source_job_id={source_job_id}）")

            if checkpoint:
                reconstructed = reconstruct_state_from_checkpoint(
                    checkpoint_data=checkpoint,
                    llm_client=client,
                    progress_callback=_progress_cb,
                    cancel_check=lambda: _dr_is_cancel_requested(job_id),
                    pause_waiter=lambda: _dr_wait_if_paused(job_id),
                    review_waiter=lambda section_id: get_pending_review(job_id, section_id),
                    model_override=body.model_override or None,
                )
                reconstructed["job_id"] = job_id
                reconstructed["session_id"] = session_id
                if body.canvas_id:
                    reconstructed["canvas_id"] = body.canvas_id
                if body.user_context is not None:
                    reconstructed["user_context"] = body.user_context
                if body.user_documents is not None:
                    reconstructed["user_documents"] = body.user_documents

                start_node = "research"
                if mode == "phase":
                    phase = str(restart_spec.get("phase") or "research").strip().lower()
                    if phase == "plan":
                        reconstructed["sections_completed"] = []
                        reconstructed["current_section"] = ""
                        reconstructed["markdown_parts"] = []
                        reconstructed["citations"] = []
                        reconstructed["iteration_count"] = 0
                        reconstructed["coverage_history"] = {}
                        reconstructed["review_handled_at"] = {}
                        reconstructed["review_seen_at"] = {}
                        reconstructed["review_gate_rounds"] = 0
                        reconstructed["review_gate_unchanged"] = 0
                        reconstructed["review_gate_last_snapshot"] = ""
                        reconstructed["verified_claims"] = ""
                        reconstructed["force_synthesize"] = False
                        reconstructed["cost_warned"] = False
                        reconstructed["graph_step_count"] = 0
                        reconstructed["last_cost_tick_step"] = 0
                        for sec in reconstructed["dashboard"].sections:
                            sec.status = "pending"
                            sec.coverage_score = 0.0
                            sec.source_count = 0
                            sec.gaps = []
                            sec.research_rounds = 0
                            sec.evidence_scarce = False
                            sec.completion_round_done = False
                        start_node = "plan"
                    elif phase == "research":
                        reconstructed["sections_completed"] = []
                        reconstructed["current_section"] = ""
                        reconstructed["markdown_parts"] = []
                        reconstructed["citations"] = []
                        reconstructed["verified_claims"] = ""
                        for sec in reconstructed["dashboard"].sections:
                            sec.status = "pending"
                            sec.source_count = 0
                            sec.gaps = []
                            sec.research_rounds = 0
                            sec.evidence_scarce = False
                            sec.completion_round_done = False
                        start_node = "research"
                    elif phase == "write":
                        reconstructed["sections_completed"] = []
                        reconstructed["current_section"] = ""
                        for sec in reconstructed["dashboard"].sections:
                            sec.status = "researching"
                        start_node = "write"
                    elif phase == "generate_claims":
                        reconstructed["sections_completed"] = []
                        reconstructed["current_section"] = ""
                        reconstructed["verified_claims"] = ""
                        for sec in reconstructed["dashboard"].sections:
                            sec.status = "researching"
                        start_node = "generate_claims"
                    elif phase == "verify":
                        reconstructed["sections_completed"] = []
                        reconstructed["verified_claims"] = ""
                        if not reconstructed.get("current_section"):
                            first_title = ""
                            if reconstructed["dashboard"].sections:
                                first_title = reconstructed["dashboard"].sections[0].title
                            reconstructed["current_section"] = first_title
                        for sec in reconstructed["dashboard"].sections:
                            if sec.title == reconstructed.get("current_section"):
                                sec.status = "writing"
                            elif sec.title in reconstructed.get("sections_completed", []):
                                sec.status = "done"
                            else:
                                sec.status = "pending"
                        start_node = "verify"
                    elif phase == "review_gate":
                        start_node = "review_gate"
                    elif phase == "synthesize":
                        start_node = "synthesize"
                    elif phase == "auto":
                        completed_set = set(reconstructed.get("sections_completed") or [])
                        all_titles = [s.title for s in reconstructed["dashboard"].sections]
                        if all(t in completed_set for t in all_titles):
                            start_node = "synthesize"
                        else:
                            reconstructed["current_section"] = ""
                            for sec in reconstructed["dashboard"].sections:
                                if sec.title not in completed_set:
                                    sec.status = "pending"
                                    sec.completion_round_done = False
                            start_node = "research"
                    else:
                        raise RuntimeError(f"不支持的重启阶段: {phase}")
                elif mode == "section":
                    target = str(restart_spec.get("section_title") or "").strip()
                    action = str(restart_spec.get("action") or "research").strip().lower()
                    if not target:
                        raise RuntimeError("重启失败：section_title 不能为空")
                    matched = None
                    for sec in reconstructed["dashboard"].sections:
                        if sec.title == target or sec.title.strip().lower() == target.lower():
                            matched = sec
                            target = sec.title
                            break
                    if matched is None:
                        raise RuntimeError(f"重启失败：章节不存在 ({target})")
                    reconstructed["current_section"] = target
                    reconstructed["sections_completed"] = [
                        s for s in (reconstructed.get("sections_completed") or []) if str(s).strip() != target
                    ]
                    matched.completion_round_done = False
                    if action == "research":
                        matched.status = "researching"
                        matched.source_count = 0
                        matched.gaps = []
                        matched.research_rounds = 0
                        matched.evidence_scarce = False
                        start_node = "research"
                    elif action == "write":
                        matched.status = "researching"
                        start_node = "write"
                    else:
                        raise RuntimeError(f"不支持的 section 重启动作: {action}")
                elif mode == "incomplete_sections":
                    targets_raw: list[str] = restart_spec.get("section_titles") or []
                    action = str(restart_spec.get("action") or "research").strip().lower()
                    if not targets_raw:
                        raise RuntimeError("重启失败：section_titles 不能为空")
                    if action not in {"research", "write"}:
                        raise RuntimeError(f"不支持的 incomplete_sections 重启动作: {action}")
                    # Normalise titles and match against dashboard
                    targets_lower = {str(t).strip().lower() for t in targets_raw}
                    matched_sections = []
                    for sec in reconstructed["dashboard"].sections:
                        if (sec.title.strip().lower() in targets_lower):
                            matched_sections.append(sec)
                    if not matched_sections:
                        raise RuntimeError("重启失败：未找到任何匹配章节")
                    matched_titles = {sec.title for sec in matched_sections}
                    # Remove all matched sections from sections_completed
                    reconstructed["sections_completed"] = [
                        s for s in (reconstructed.get("sections_completed") or [])
                        if str(s).strip() not in matched_titles
                    ]
                    for sec in matched_sections:
                        sec.completion_round_done = False
                        if action == "research":
                            sec.status = "researching"
                            sec.source_count = 0
                            sec.gaps = []
                            sec.research_rounds = 0
                            sec.evidence_scarce = False
                        else:
                            sec.status = "researching"
                    # Non-matched sections must be forced to "done" so that
                    # get_next_section() never picks them up, regardless of
                    # whatever stale status they carry from the checkpoint
                    # (e.g. "researching" from a previous crashed run).
                    _existing_completed: set = set(reconstructed.get("sections_completed") or [])
                    for sec in reconstructed["dashboard"].sections:
                        if sec.title not in matched_titles:
                            sec.status = "done"
                            _existing_completed.add(sec.title)
                    reconstructed["sections_completed"] = list(_existing_completed)
                    # Start from the first matched section (in outline order)
                    all_titles = [s.title for s in reconstructed["dashboard"].sections]
                    first_target = next(
                        (t for t in all_titles if t in matched_titles),
                        matched_sections[0].title,
                    )
                    reconstructed["current_section"] = first_target
                    start_node = "research" if action == "research" else "write"
                elif mode == "outline_update":
                    from src.collaboration.research.dashboard import ResearchDashboard, SectionStatus
                    new_outline_raw: list[str] = [str(t).strip() for t in (restart_spec.get("new_outline") or []) if str(t).strip()]
                    action = str(restart_spec.get("action") or "research").strip().lower()
                    if not new_outline_raw:
                        raise RuntimeError("重启失败：new_outline 不能为空")
                    if action not in {"research", "write"}:
                        raise RuntimeError(f"不支持的 outline_update 重启动作: {action}")
                    old_dashboard = reconstructed["dashboard"]
                    old_parts: list = list(reconstructed.get("markdown_parts") or [])
                    header = old_parts[0] if old_parts else f"# {reconstructed.get('topic', '')}\n"
                    new_sections: list = []
                    new_markdown_parts: list = [header]
                    for title in new_outline_raw:
                        old_sec = old_dashboard.get_section(title) if hasattr(old_dashboard, "get_section") else None
                        if old_sec and getattr(old_sec, "status", "") == "done":
                            new_sec = SectionStatus(
                                title=title,
                                status="done",
                                coverage_score=getattr(old_sec, "coverage_score", 0.0),
                                source_count=getattr(old_sec, "source_count", 0),
                                gaps=list(getattr(old_sec, "gaps", []) or []),
                                research_rounds=getattr(old_sec, "research_rounds", 0),
                                evidence_scarce=getattr(old_sec, "evidence_scarce", False),
                                verify_rewrite_count=getattr(old_sec, "verify_rewrite_count", 0),
                            )
                            new_sections.append(new_sec)
                            part_found = ""
                            for p in old_parts[1:]:
                                if title in str(p):
                                    part_found = p
                                    break
                            new_markdown_parts.append(part_found or f"\n## {title}\n\n")
                        else:
                            new_sections.append(SectionStatus(title=title, status="researching"))
                            new_markdown_parts.append(f"\n## {title}\n\n")
                    new_dashboard = ResearchDashboard(
                        brief=old_dashboard.brief,
                        sections=new_sections,
                        overall_confidence=getattr(old_dashboard, "overall_confidence", "low"),
                        total_sources=getattr(old_dashboard, "total_sources", 0),
                        total_iterations=getattr(old_dashboard, "total_iterations", 0),
                        coverage_gaps=list(getattr(old_dashboard, "coverage_gaps", []) or []),
                        conflict_notes=list(getattr(old_dashboard, "conflict_notes", []) or []),
                    )
                    reconstructed["dashboard"] = new_dashboard
                    reconstructed["markdown_parts"] = new_markdown_parts
                    reconstructed["sections_completed"] = [s.title for s in new_sections if s.status == "done"]
                    targets_raw = [s.title for s in new_sections if s.status != "done"]
                    if not targets_raw:
                        raise RuntimeError("重启失败：新大纲下没有需要补充的章节")
                    targets_lower = {str(t).strip().lower() for t in targets_raw}
                    matched_sections = [s for s in new_sections if s.title.strip().lower() in targets_lower]
                    matched_titles = {sec.title for sec in matched_sections}
                    reconstructed["sections_completed"] = [
                        s for s in (reconstructed.get("sections_completed") or [])
                        if str(s).strip() not in matched_titles
                    ]
                    for sec in matched_sections:
                        sec.completion_round_done = False
                        if action == "research":
                            sec.status = "researching"
                            sec.source_count = 0
                            sec.gaps = []
                            sec.research_rounds = 0
                            sec.evidence_scarce = False
                        else:
                            sec.status = "researching"
                    all_titles = [s.title for s in reconstructed["dashboard"].sections]
                    first_target = next(
                        (t for t in all_titles if t in matched_titles),
                        matched_sections[0].title if matched_sections else "",
                    )
                    reconstructed["current_section"] = first_target
                    start_node = "research" if action == "research" else "write"
                else:
                    raise RuntimeError("重启失败：mode 无效")

                runtime_kwargs["start_node"] = start_node
                runtime_kwargs["initial_state_override"] = reconstructed
                runtime_kwargs["confirmed_outline"] = [sec.title for sec in reconstructed["dashboard"].sections]
                runtime_kwargs["confirmed_brief"] = {
                    "topic": reconstructed["dashboard"].brief.topic,
                    "scope": reconstructed["dashboard"].brief.scope,
                }
                runtime_kwargs["topic"] = reconstructed.get("topic") or body.topic.strip()

                append_event(
                    job_id,
                    "restart_start",
                    {
                        "job_id": job_id,
                        "source_job_id": source_job_id,
                        "mode": mode,
                        "start_node": start_node,
                    },
                )

        runtime = prepare_deep_research_runtime(**runtime_kwargs)
        compiled = runtime["compiled"]
        config = runtime["config"]
        initial_state = runtime["initial_state"]
        compiled.invoke(initial_state, config=config)
        state_snapshot = compiled.get_state(config)
        if getattr(state_snapshot, "next", ()):
            _dr_store_suspended_runtime(
                job_id,
                {
                    "compiled": compiled,
                    "config": config,
                    "session_id": session_id,
                    "body": body,
                    "started_at_perf": t0,
                    "outline": runtime.get("outline") or [],
                    "topic": runtime.get("topic") or body.topic.strip(),
                },
            )
            update_job(
                job_id,
                status="waiting_review",
                session_id=session_id,
                current_stage="review_gate",
                message="等待人工审核",
            )
            append_event(
                job_id,
                "waiting_review",
                {
                    "job_id": job_id,
                    "session_id": session_id,
                    "next": list(getattr(state_snapshot, "next", ()) or ()),
                },
            )
            return

        final_state = getattr(state_snapshot, "values", {}) or {}
        result = build_deep_research_result_from_state(
            final_state,
            topic=str(runtime.get("topic") or body.topic.strip()),
            elapsed_ms=(_time.perf_counter() - t0) * 1000,
            fallback_outline=runtime.get("outline") or [],
        )
        _complete_deep_research_job(job_id=job_id, session_id=session_id, topic=body.topic, result=result)
        _dr_pop_suspended_runtime(job_id)
    except Exception as e:
        cancelled = _dr_is_cancel_requested(job_id) or "cancelled" in str(e).lower() or "canceled" in str(e).lower()
        status = "cancelled" if cancelled else "error"
        msg = "任务已取消" if cancelled else f"任务失败: {e}"
        update_job(
            job_id,
            status=status,
            message=msg,
            error_message="" if cancelled else str(e),
            finished_at=_time.time(),
            total_time_ms=(_time.perf_counter() - t0) * 1000,
        )
        append_event(job_id, status, {"message": msg, "error": "" if cancelled else str(e)})
        append_event(job_id, "step", {"step": None, "label": ""})
        try:
            from src.collaboration.research.job_store import purge_dr_checkpoints
            purged = purge_dr_checkpoints(job_id)
            if purged:
                _chat_logger.debug("Purged %d DR checkpoint(s) for %s job %s", purged, status, job_id)
        except Exception as _ce:
            _chat_logger.debug("purge_dr_checkpoints on %s failed (non-fatal): %s", status, _ce)
        _dr_pop_suspended_runtime(job_id)
    finally:
        _dr_clear_pause_event(job_id)
        _dr_clear_cancel_event(job_id)


def _complete_deep_research_job(*, job_id: str, session_id: str, topic: str, result: Dict[str, Any]) -> None:
    from src.collaboration.research.job_store import append_event, update_job
    import time as _time

    store = get_session_store()
    response_text = result.get("markdown", "")
    citations = result.get("citations") or []
    dashboard_data = result.get("dashboard") or {}
    canvas_id = result.get("canvas_id", "") or ""
    total_time_ms = float(result.get("total_time_ms", 0.0))

    if canvas_id:
        store.update_session_meta(session_id, {"canvas_id": canvas_id})
        try:
            from src.collaboration.canvas.canvas_manager import update_canvas
            update_canvas(canvas_id, stage="refine")
        except Exception:
            _chat_logger.debug("Failed to update canvas stage to refine", exc_info=True)
    store.update_session_stage(session_id, "refine")
    memory = load_session_memory(session_id)
    if memory:
        memory.add_turn("user", f"[Deep Research Confirmed] {topic}")
        memory.add_turn("assistant", response_text, citations=[_serialize_citation(c) for c in citations])

    update_job(
        job_id,
        status="done",
        session_id=session_id,
        canvas_id=canvas_id,
        current_stage="refine",
        message="Deep Research 完成",
        result_markdown=response_text,
        result_citations=json.dumps([_serialize_citation(c) for c in citations], ensure_ascii=False, default=str),
        result_dashboard=json.dumps(dashboard_data, ensure_ascii=False, default=str),
        total_time_ms=total_time_ms,
        finished_at=_time.time(),
    )
    for chunk in _chunk_text(response_text):
        append_event(job_id, "delta", {"delta": chunk})
    append_event(job_id, "step", {"step": None, "label": ""})
    try:
        from src.collaboration.research.job_store import purge_dr_checkpoints
        purged = purge_dr_checkpoints(job_id)
        if purged:
            _chat_logger.debug("Purged %d DR checkpoint(s) for completed job %s", purged, job_id)
    except Exception as _e:
        _chat_logger.debug("purge_dr_checkpoints on done failed (non-fatal): %s", _e)
    append_event(
        job_id,
        "done",
        {
            "session_id": session_id,
            "canvas_id": canvas_id,
            "total_time_ms": total_time_ms,
            "final_text": response_text,
            "citations": [_serialize_citation(c) for c in citations],
            "dashboard": dashboard_data,
        },
    )


def _resume_suspended_job(job_id: str) -> None:
    from langgraph.types import Command
    from src.collaboration.research.agent import build_deep_research_result_from_state
    from src.collaboration.research.job_store import append_event, get_job, update_job
    import time as _time

    runtime = _dr_get_suspended_runtime(job_id)
    if not runtime:
        return

    compiled = runtime.get("compiled")
    config = runtime.get("config")
    if compiled is None or not isinstance(config, dict):
        _dr_pop_suspended_runtime(job_id)
        update_job(job_id, status="error", message="任务恢复失败：缺少挂起上下文")
        try:
            from src.collaboration.research.job_store import purge_dr_checkpoints
            purge_dr_checkpoints(job_id)
        except Exception:
            pass
        return

    job = get_job(job_id) or {}
    topic = str(job.get("topic") or runtime.get("topic") or "")
    session_id = str(runtime.get("session_id") or job.get("session_id") or "")
    started_at_perf = float(runtime.get("started_at_perf") or _time.perf_counter())

    try:
        update_job(job_id, status="running", message="收到审核输入，恢复执行")
        append_event(job_id, "resume_start", {"job_id": job_id})
        compiled.invoke(Command(resume=True), config=config)
        state_snapshot = compiled.get_state(config)
        if getattr(state_snapshot, "next", ()):
            update_job(
                job_id,
                status="waiting_review",
                session_id=session_id,
                current_stage="review_gate",
                message="等待人工审核",
            )
            append_event(
                job_id,
                "waiting_review",
                {
                    "job_id": job_id,
                    "session_id": session_id,
                    "next": list(getattr(state_snapshot, "next", ()) or ()),
                },
            )
            return

        final_state = getattr(state_snapshot, "values", {}) or {}
        result = build_deep_research_result_from_state(
            final_state,
            topic=topic or "",
            elapsed_ms=(_time.perf_counter() - started_at_perf) * 1000,
            fallback_outline=runtime.get("outline") or [],
        )
        _complete_deep_research_job(job_id=job_id, session_id=session_id, topic=topic or "", result=result)
        _dr_pop_suspended_runtime(job_id)
    except Exception as e:
        cancelled = _dr_is_cancel_requested(job_id) or "cancelled" in str(e).lower() or "canceled" in str(e).lower()
        status = "cancelled" if cancelled else "error"
        msg = "任务已取消" if cancelled else f"任务恢复失败: {e}"
        update_job(
            job_id,
            status=status,
            message=msg,
            error_message="" if cancelled else str(e),
            finished_at=_time.time(),
            total_time_ms=(_time.perf_counter() - started_at_perf) * 1000,
        )
        append_event(job_id, status, {"message": msg, "error": "" if cancelled else str(e)})
        try:
            from src.collaboration.research.job_store import purge_dr_checkpoints
            purged = purge_dr_checkpoints(job_id)
            if purged:
                _chat_logger.debug("Purged %d DR checkpoint(s) for resumed job %s (%s)", purged, job_id, status)
        except Exception as _ce:
            _chat_logger.debug("purge_dr_checkpoints on resume %s failed (non-fatal): %s", status, _ce)
        _dr_pop_suspended_runtime(job_id)
    finally:
        _dr_set_resume_idle(job_id)
        _dr_clear_pause_event(job_id)
        _dr_clear_cancel_event(job_id)


@router.post("/deep-research/submit", response_model=DeepResearchSubmitResponse)
def submit_deep_research_endpoint(
    body: DeepResearchConfirmRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchSubmitResponse:
    """提交 Deep Research 后台任务（默认推荐前端使用此接口）。
    If planning_job_id is provided (from the start phase), reuse that DB
    entry instead of creating a new one — single ID throughout lifecycle."""
    from src.collaboration.research.job_store import create_job, get_job, update_job as _update_job
    import time as _t

    store = get_session_store()
    session_id = body.session_id or store.create_session(
        canvas_id=body.canvas_id or "",
        user_id=optional_user_id or "",
    )
    _ensure_research_session_title(store, session_id, body.topic)
    payload = body.model_dump()
    payload["session_id"] = session_id
    payload["_worker_user_id"] = optional_user_id

    planning_jid = (body.planning_job_id or "").strip()
    reused = False
    if planning_jid:
        existing = get_job(planning_jid)
        if existing and existing.get("status") == "planning":
            import json as _json
            _update_job(
                planning_jid,
                status="pending",
                session_id=session_id,
                canvas_id=body.canvas_id or "",
                request_json=_json.dumps(payload, ensure_ascii=False, default=str),
                message="已确认，等待执行",
                started_at=_t.time(),
            )
            job_id = planning_jid
            reused = True
            _chat_logger.info(
                "[deep-research/submit] reused planning job %s | session=%s",
                job_id[:12],
                session_id[:12],
            )

    if not reused:
        job = create_job(
            topic=body.topic.strip(),
            session_id=session_id,
            canvas_id=body.canvas_id or "",
            request_payload=payload,
            user_id=optional_user_id or "",
        )
        job_id = str(job.get("job_id") or "")

    return DeepResearchSubmitResponse(
        ok=True,
        job_id=job_id,
        session_id=session_id,
        canvas_id=body.canvas_id or "",
    )


@router.post("/deep-research/confirm")
def confirm_deep_research_endpoint(
    body: DeepResearchConfirmRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> StreamingResponse:
    """Phase 2: execute deep research using confirmed brief/outline and stream progress."""
    from src.collaboration.research.agent import execute_deep_research

    session_id = body.session_id or get_session_store().create_session(
        canvas_id=body.canvas_id or "",
        user_id=optional_user_id or "",
    )
    store = get_session_store()
    _ensure_research_session_title(store, session_id, body.topic)
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    user_id = optional_user_id or ""
    filters = _build_deep_research_filters(body)

    progress_events: List[Dict[str, Any]] = []

    def _progress_cb(event_type: str, payload: Dict[str, Any]) -> None:
        progress_events.append({"event": event_type, "data": payload})

    result = execute_deep_research(
        topic=body.topic.strip(),
        llm_client=client,
        confirmed_outline=body.confirmed_outline,
        confirmed_brief=body.confirmed_brief,
        canvas_id=body.canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=body.search_mode,
        filters=filters or None,
        model_override=body.model_override or None,
        output_language=body.output_language or "auto",
        step_models=body.step_models,
        step_model_strict=bool(body.step_model_strict),
        user_context=body.user_context,
        user_context_mode=body.user_context_mode or "supporting",
        user_documents=body.user_documents,
        progress_callback=_progress_cb,
        skip_draft_review=bool(body.skip_draft_review),
        skip_refine_review=bool(body.skip_refine_review),
        skip_claim_generation=bool(body.skip_claim_generation),
        depth=body.depth or "comprehensive",
    )
    response_text = result.get("markdown", "")
    citations = result.get("citations") or []
    if result.get("canvas_id"):
        store.update_session_meta(session_id, {"canvas_id": result.get("canvas_id", "")})
        try:
            from src.collaboration.canvas.canvas_manager import update_canvas
            update_canvas(result.get("canvas_id", ""), stage="refine")
        except Exception:
            _chat_logger.debug("Failed to update canvas stage to refine (stream confirm)", exc_info=True)
    store.update_session_stage(session_id, "refine")
    memory = load_session_memory(session_id)
    if memory:
        memory.add_turn("user", f"[Deep Research Confirmed] {body.topic}")
        memory.add_turn("assistant", response_text, citations=[_serialize_citation(c) for c in citations])
    sc_pstats = _compute_citation_provider_stats(citations) if citations else None
    evidence_summary = EvidenceSummary(
        query=body.topic,
        total_chunks=len(citations),
        sources_used=_infer_sources_from_citations(citations),
        retrieval_time_ms=float(result.get("total_time_ms", 0.0)),
        provider_stats=sc_pstats,
    )
    dashboard_data = result.get("dashboard") or {}

    def event_stream():
        meta = {
            "session_id": session_id,
            "canvas_id": result.get("canvas_id", "") or "",
            "citations": [_serialize_citation(c) for c in citations],
            "evidence_summary": evidence_summary.model_dump(),
            "intent": {"mode": "deep_research", "intent_type": "deep_research", "confidence": 1.0, "from_command": False},
            "current_stage": "refine",
        }
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
        for item in progress_events:
            ev = "warning" if item["event"] == "warning" else "progress"
            payload = {"type": item["event"], **(item.get("data") or {})}
            yield f"event: {ev}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        if dashboard_data:
            yield f"event: dashboard\ndata: {json.dumps(dashboard_data, ensure_ascii=False, default=str)}\n\n"
        for chunk in _chunk_text(response_text):
            yield f"event: delta\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({'final_text': response_text}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/deep-research/jobs", response_model=List[DeepResearchJobInfo])
def list_deep_research_jobs(limit: int = 20, status: str | None = None) -> List[DeepResearchJobInfo]:
    from src.collaboration.research.job_store import list_jobs

    jobs = list_jobs(limit=limit, status=status)
    out: List[DeepResearchJobInfo] = []
    for j in jobs:
        out.append(DeepResearchJobInfo(**j))
    return out


@router.get("/deep-research/jobs/{job_id}", response_model=DeepResearchJobInfo)
def get_deep_research_job(job_id: str) -> DeepResearchJobInfo:
    from src.collaboration.research.job_store import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return DeepResearchJobInfo(**job)


@router.get("/deep-research/jobs/{job_id}/events")
def get_deep_research_job_events(job_id: str, after_id: int = 0, limit: int = 200) -> dict:
    from src.collaboration.research.job_store import get_job, list_events

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {
        "job_id": job_id,
        "events": list_events(job_id, after_id=after_id, limit=limit),
    }


def _dr_sse(event: str, data: dict, event_id=None) -> str:
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"{id_line}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _dr_job_status_payload(job: dict) -> dict:
    """Extract status fields from a job dict for SSE heartbeat / job_status events.

    Includes timing information so the frontend can track phase duration
    and provide meaningful progress indicators (heartbeat rhythm).
    """
    dashboard = job.get("result_dashboard")
    if isinstance(dashboard, str):
        try:
            dashboard = json.loads(dashboard)
        except Exception:
            dashboard = {}
    now = time.time()
    created_at = float(job.get("created_at") or 0)
    started_at = job.get("started_at")
    started_at_f = float(started_at) if started_at is not None else None
    elapsed_since_create_ms = (now - created_at) * 1000 if created_at else 0
    elapsed_since_start_ms = (now - started_at_f) * 1000 if started_at_f else None
    return {
        "job_id": job.get("job_id", ""),
        "topic": job.get("topic", ""),
        "status": job.get("status", ""),
        "current_stage": job.get("current_stage", ""),
        "message": job.get("message", ""),
        "canvas_id": job.get("canvas_id", ""),
        "result_dashboard": dashboard or {},
        "created_at": created_at,
        "started_at": started_at_f,
        "heartbeat_ts": now,
        "elapsed_since_create_ms": round(elapsed_since_create_ms),
        "elapsed_since_start_ms": round(elapsed_since_start_ms) if elapsed_since_start_ms is not None else None,
    }


@router.get("/deep-research/jobs/{job_id}/stream")
def stream_deep_research_events(job_id: str, after_id: int = 0) -> StreamingResponse:
    """SSE stream for Deep Research job progress.

    Replaces the polling pattern of GET /events + GET /jobs/{id}.
    Emits all job events in real-time, plus periodic heartbeat events carrying
    the current job status, and a final job_status event when the job reaches
    a terminal state (done / error / cancelled).
    """
    import time as _time
    from src.collaboration.research.job_store import get_job, list_events

    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="job 不存在")

    def _ev_payload(ev: dict) -> dict:
        """Merge event_id into the data payload so the frontend can track resume position."""
        data = dict(ev["data"]) if isinstance(ev["data"], dict) else {"_raw": ev["data"]}
        data["_event_id"] = int(ev["event_id"])
        return data

    def event_stream():
        cursor = max(0, int(after_id))
        idle_ticks = 0
        while True:
            events = list_events(job_id, after_id=cursor, limit=500)
            if events:
                idle_ticks = 0
                for ev in events:
                    cursor = max(cursor, int(ev["event_id"]))
                    payload = _ev_payload(ev)
                    # Emit id: line so the browser / streamSSEResumable adapter can track
                    # the last seen event_id and resume from exactly this point on reconnect.
                    yield _dr_sse(ev["event"], payload, event_id=payload.get("_event_id"))
                continue

            job = get_job(job_id)
            if not job:
                yield _dr_sse("error", {"message": "job 不存在"})
                break

            status = str(job.get("status") or "")
            if status in {"done", "error", "cancelled"}:
                # Flush any final events that arrived before the status flip
                final_events = list_events(job_id, after_id=cursor, limit=500)
                for ev in final_events:
                    cursor = max(cursor, int(ev["event_id"]))
                    payload = _ev_payload(ev)
                    yield _dr_sse(ev["event"], payload, event_id=payload.get("_event_id"))
                # Terminal job_status has no persistent event_id; omit id: line intentionally
                yield _dr_sse("job_status", _dr_job_status_payload(job))
                break

            idle_ticks += 1
            # Heartbeat every 5 idle ticks (~10s) — frequent enough for meaningful
            # progress identification while still being lightweight.
            if idle_ticks % 5 == 0:
                yield _dr_sse("heartbeat", _dr_job_status_payload(job))
            _time.sleep(2)

    resp = StreamingResponse(event_stream(), media_type="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp


@router.post("/deep-research/jobs/{job_id}/cancel")
def cancel_deep_research_job(job_id: str, force: bool = False) -> dict:
    """Cancel a Deep Research job.

    When *force=true* the job is immediately written as "cancelled" in the DB and
    its Redis slot is released — bypassing the graceful cancellation handshake.
    Use this when the job is stuck in "cancelling" for a long time.
    """
    import time as _t
    from src.collaboration.research.job_store import get_job, update_job, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status in {"done", "error", "cancelled"}:
        return {"ok": True, "job_id": job_id, "status": current_status}
    if force or current_status in ("pending", "planning"):
        msg = "任务已强制取消" if force and current_status not in ("pending", "planning") else "任务已取消（未启动）"
        _dr_clear_pause_event(job_id)
        update_job(job_id, status="cancelled", message=msg, finished_at=_t.time())
        append_event(job_id, "cancelled", {"job_id": job_id, "message": msg})
        append_event(job_id, "step", {"step": None, "label": ""})
        try:
            from src.collaboration.research.job_store import purge_dr_checkpoints
            purged = purge_dr_checkpoints(job_id)
            if purged:
                _chat_logger.debug("Purged %d DR checkpoint(s) for force-cancelled job %s", purged, job_id)
        except Exception as _ce:
            _chat_logger.debug("purge_dr_checkpoints on force-cancel failed (non-fatal): %s", _ce)
        _dr_request_cancel(job_id)  # signal the thread too so it exits cleanly
        _dr_release_slot_eager(job_id)
        return {"ok": True, "job_id": job_id, "status": "cancelled"}
    _dr_request_cancel(job_id)
    _dr_clear_pause_event(job_id)
    _dr_release_slot_eager(job_id)  # unblock any queued job for the same session immediately
    update_job(job_id, status="cancelling", message="收到停止请求，正在终止任务...")
    append_event(job_id, "cancel_requested", {"job_id": job_id, "message": "已请求停止"})
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


@router.post("/deep-research/jobs/{job_id}/pause")
def pause_deep_research_job(job_id: str) -> dict:
    """Request cooperative pause for a Deep Research job."""
    from src.collaboration.research.job_store import get_job, update_job, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status in {"done", "error", "cancelled"}:
        return {"ok": False, "job_id": job_id, "status": current_status}
    if current_status in {"paused", "pausing"}:
        return {"ok": True, "job_id": job_id, "status": current_status}
    if current_status == "waiting_review":
        return {"ok": False, "job_id": job_id, "status": current_status, "message": "等待审核中的任务无需暂停"}
    _dr_request_pause(job_id)
    update_job(job_id, status="pausing", message="收到暂停请求，正在暂停任务...")
    append_event(job_id, "pause_requested", {"job_id": job_id, "message": "已请求暂停"})
    return {"ok": True, "job_id": job_id, "status": "pausing"}


@router.post("/deep-research/jobs/{job_id}/resume")
def resume_deep_research_job(job_id: str) -> dict:
    """Resume a paused Deep Research job."""
    from src.collaboration.research.job_store import get_job, update_job, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status not in {"paused", "pausing"}:
        return {"ok": False, "job_id": job_id, "status": current_status}
    _dr_clear_pause_event(job_id)
    next_status = "running"
    update_job(job_id, status=next_status, message="任务已恢复")
    append_event(job_id, "resumed", {"job_id": job_id, "message": "任务已恢复"})
    return {"ok": True, "job_id": job_id, "status": next_status}


@router.delete("/deep-research/jobs/{job_id}")
def delete_deep_research_job(job_id: str) -> dict:
    """删除已终态的后台调研任务及其关联数据（events、reviews、resume_queue 等），释放资源。"""
    from src.collaboration.research.job_store import get_job, delete_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status not in ("done", "error", "cancelled", "planning"):
        raise HTTPException(
            status_code=400,
            detail="只能删除已结束/已取消/规划中的任务，当前状态: " + current_status,
        )
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="job 不存在或无法删除")
    return {"ok": True, "job_id": job_id, "deleted": True}


@router.get("/deep-research/jobs/{job_id}/checkpoints")
def list_deep_research_checkpoints(job_id: str) -> dict:
    """List all saved checkpoints for a Deep Research job."""
    from src.collaboration.research.job_store import get_job, list_checkpoints

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    raw = list_checkpoints(job_id)
    items = []
    for cp in raw:
        state = cp.get("state") or {}
        completed = state.get("sections_completed") or []
        total_sections = len((state.get("dashboard") or {}).get("sections") or [])
        items.append({
            "phase": cp.get("phase", ""),
            "section_title": cp.get("section_title", ""),
            "created_at": cp.get("created_at", 0),
            "sections_completed": len(completed),
            "total_sections": total_sections,
            "resumable": cp.get("phase") in ("confirmed", "section_done", "research", "write", "verify", "synthesize", "crash"),
        })
    return {"job_id": job_id, "checkpoints": items}


@router.post("/deep-research/jobs/{job_id}/restart-phase", response_model=DeepResearchSubmitResponse)
def restart_deep_research_phase(
    job_id: str,
    body: DeepResearchRestartPhaseRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchSubmitResponse:
    from src.collaboration.research.job_store import get_job, create_job, update_job, append_event

    source_job = get_job(job_id)
    if not source_job:
        raise HTTPException(status_code=404, detail="job 不存在")
    source_status = str(source_job.get("status") or "")
    if source_status in {"pending", "running", "pausing", "paused", "cancelling", "waiting_review"}:
        _dr_request_cancel(job_id)
        update_job(job_id, status="cancelling", message="收到重启请求，正在终止旧任务...")
        append_event(
            job_id,
            "restart_requested",
            {
                "message": "收到重启请求，旧任务将被中止并创建新任务",
                "source_status": source_status,
            },
        )
    # Always release the session slot unconditionally so the new restart job can acquire it,
    # even if the source job was already cancelled/errored (slot may still be in Redis from
    # an earlier ancestor job in the same restart chain).
    _dr_release_slot_eager(job_id)
    phase = str(body.phase or "").strip().lower()
    if phase not in {"plan", "research", "generate_claims", "write", "verify", "review_gate", "synthesize"}:
        raise HTTPException(
            status_code=400,
            detail="phase must be plan|research|generate_claims|write|verify|review_gate|synthesize",
        )

    req_payload = source_job.get("request") or {}
    if not isinstance(req_payload, dict) or not req_payload:
        raise HTTPException(status_code=400, detail="原任务缺少 request payload，无法重启")

    payload = dict(req_payload)
    payload["_worker_user_id"] = optional_user_id
    payload["_restart"] = {
        "source_job_id": job_id,
        "mode": "phase",
        "phase": phase,
    }
    payload["session_id"] = source_job.get("session_id") or payload.get("session_id")
    job = create_job(
        topic=str(source_job.get("topic") or payload.get("topic") or "").strip(),
        session_id=str(source_job.get("session_id") or payload.get("session_id") or ""),
        canvas_id=str(source_job.get("canvas_id") or payload.get("canvas_id") or ""),
        request_payload=payload,
        user_id=str(source_job.get("user_id") or optional_user_id or ""),
    )
    new_job_id = str(job.get("job_id") or "")
    return DeepResearchSubmitResponse(
        ok=True,
        job_id=new_job_id,
        session_id=str(job.get("session_id") or ""),
        canvas_id=str(job.get("canvas_id") or ""),
    )


@router.post("/deep-research/jobs/{job_id}/restart-section", response_model=DeepResearchSubmitResponse)
def restart_deep_research_section(
    job_id: str,
    body: DeepResearchRestartSectionRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchSubmitResponse:
    from src.collaboration.research.job_store import get_job, create_job, update_job, append_event

    source_job = get_job(job_id)
    if not source_job:
        raise HTTPException(status_code=404, detail="job 不存在")
    source_status = str(source_job.get("status") or "")
    if source_status in {"pending", "running", "pausing", "paused", "cancelling", "waiting_review"}:
        _dr_request_cancel(job_id)
        update_job(job_id, status="cancelling", message="收到重启请求，正在终止旧任务...")
        append_event(
            job_id,
            "restart_requested",
            {
                "message": "收到重启请求，旧任务将被中止并创建新任务",
                "source_status": source_status,
            },
        )
    # Always release the session slot unconditionally so the new restart job can acquire it.
    _dr_release_slot_eager(job_id)
    section_title = str(body.section_title or "").strip()
    action = str(body.action or "research").strip().lower()
    if not section_title:
        raise HTTPException(status_code=400, detail="section_title 不能为空")
    if action not in {"research", "write"}:
        raise HTTPException(status_code=400, detail="action must be research|write")

    req_payload = source_job.get("request") or {}
    if not isinstance(req_payload, dict) or not req_payload:
        raise HTTPException(status_code=400, detail="原任务缺少 request payload，无法重启")

    payload = dict(req_payload)
    payload["_worker_user_id"] = optional_user_id
    payload["_restart"] = {
        "source_job_id": job_id,
        "mode": "section",
        "section_title": section_title,
        "action": action,
    }
    payload["session_id"] = source_job.get("session_id") or payload.get("session_id")
    job = create_job(
        topic=str(source_job.get("topic") or payload.get("topic") or "").strip(),
        session_id=str(source_job.get("session_id") or payload.get("session_id") or ""),
        canvas_id=str(source_job.get("canvas_id") or payload.get("canvas_id") or ""),
        request_payload=payload,
        user_id=str(source_job.get("user_id") or optional_user_id or ""),
    )
    new_job_id = str(job.get("job_id") or "")
    return DeepResearchSubmitResponse(
        ok=True,
        job_id=new_job_id,
        session_id=str(job.get("session_id") or ""),
        canvas_id=str(job.get("canvas_id") or ""),
    )


@router.post("/deep-research/jobs/{job_id}/restart-incomplete-sections", response_model=DeepResearchSubmitResponse)
def restart_incomplete_sections(
    job_id: str,
    body: DeepResearchRestartIncompleteSectionsRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchSubmitResponse:
    """Restart all incomplete (no-draft) sections in a single new job.

    The caller passes the list of section titles that still need to be written.
    The new job reuses the source job's canvas/outline and restarts only those
    sections from scratch, leaving already-completed sections untouched.
    """
    from src.collaboration.research.job_store import get_job, create_job, update_job, append_event

    source_job = get_job(job_id)
    if not source_job:
        raise HTTPException(status_code=404, detail="job 不存在")
    section_titles = [str(t).strip() for t in (body.section_titles or []) if str(t).strip()]
    if not section_titles:
        raise HTTPException(status_code=400, detail="section_titles 不能为空")
    action = str(body.action or "research").strip().lower()
    if action not in {"research", "write"}:
        raise HTTPException(status_code=400, detail="action must be research|write")

    source_status = str(source_job.get("status") or "")
    if source_status in {"pending", "running", "pausing", "paused", "cancelling", "waiting_review"}:
        _dr_request_cancel(job_id)
        update_job(job_id, status="cancelling", message="收到重启请求，正在终止旧任务...")
        append_event(job_id, "restart_requested", {
            "message": "收到重启请求（未完成章节），旧任务将被中止并创建新任务",
            "source_status": source_status,
        })
    # Always release the session slot unconditionally so the new restart job can acquire it.
    _dr_release_slot_eager(job_id)

    req_payload = source_job.get("request") or {}
    if not isinstance(req_payload, dict) or not req_payload:
        raise HTTPException(status_code=400, detail="原任务缺少 request payload，无法重启")

    payload = dict(req_payload)
    payload["_worker_user_id"] = optional_user_id
    payload["_restart"] = {
        "source_job_id": job_id,
        "mode": "incomplete_sections",
        "section_titles": section_titles,
        "action": action,
    }
    payload["session_id"] = source_job.get("session_id") or payload.get("session_id")
    job = create_job(
        topic=str(source_job.get("topic") or payload.get("topic") or "").strip(),
        session_id=str(source_job.get("session_id") or payload.get("session_id") or ""),
        canvas_id=str(source_job.get("canvas_id") or payload.get("canvas_id") or ""),
        request_payload=payload,
        user_id=str(source_job.get("user_id") or optional_user_id or ""),
    )
    new_job_id = str(job.get("job_id") or "")
    return DeepResearchSubmitResponse(
        ok=True,
        job_id=new_job_id,
        session_id=str(job.get("session_id") or ""),
        canvas_id=str(job.get("canvas_id") or ""),
    )


@router.post("/deep-research/jobs/{job_id}/restart-with-outline", response_model=DeepResearchSubmitResponse)
def restart_with_outline(
    job_id: str,
    body: DeepResearchRestartWithOutlineRequest,
    session_id_fallback: Optional[str] = None,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchSubmitResponse:
    """Restart with a new outline: keep existing drafts for matching sections, research+write new/changed ones.
    If job_id is not in DB (e.g. it was a start-phase-only id), pass session_id_fallback to use the latest job for that session."""
    from src.collaboration.research.job_store import get_job, create_job, append_event, get_latest_job_by_session

    source_job = get_job(job_id)
    if not source_job and session_id_fallback and session_id_fallback.strip():
        latest = get_latest_job_by_session(session_id_fallback.strip())
        if latest:
            job_id = str(latest.get("job_id") or "")
            source_job = latest
            _chat_logger.info("[restart-with-outline] resolved source job by session_id fallback: %s", job_id[:12])
    if not source_job:
        raise HTTPException(
            status_code=404,
            detail="job 不存在（可能为规划阶段任务或已删除）。请从侧边栏「后台调研」进入该画布对应任务后再试。",
        )
    new_outline = [str(t).strip() for t in (body.new_outline or []) if str(t).strip()]
    if not new_outline:
        raise HTTPException(status_code=400, detail="new_outline 不能为空")
    action = str(body.action or "research").strip().lower()
    if action not in {"research", "write"}:
        raise HTTPException(status_code=400, detail="action must be research|write")

    source_status = str(source_job.get("status") or "")
    if source_status in {"pending", "running", "pausing", "paused", "cancelling", "waiting_review"}:
        _dr_request_cancel(job_id)
        from src.collaboration.research.job_store import update_job
        update_job(job_id, status="cancelling", message="收到重启请求，正在终止旧任务...")
        append_event(job_id, "restart_requested", {
            "message": "收到重启请求（新大纲），旧任务将被中止并创建新任务",
            "source_status": source_status,
        })
    _dr_release_slot_eager(job_id)

    req_payload = source_job.get("request") or {}
    if not isinstance(req_payload, dict) or not req_payload:
        raise HTTPException(status_code=400, detail="原任务缺少 request payload，无法重启")

    payload = dict(req_payload)
    payload["_worker_user_id"] = optional_user_id
    payload["_restart"] = {
        "source_job_id": job_id,
        "mode": "outline_update",
        "new_outline": new_outline,
        "action": action,
    }
    payload["session_id"] = source_job.get("session_id") or payload.get("session_id")
    job = create_job(
        topic=str(source_job.get("topic") or payload.get("topic") or "").strip(),
        session_id=str(source_job.get("session_id") or payload.get("session_id") or ""),
        canvas_id=str(source_job.get("canvas_id") or payload.get("canvas_id") or ""),
        request_payload=payload,
        user_id=str(source_job.get("user_id") or optional_user_id or ""),
    )
    new_job_id = str(job.get("job_id") or "")
    return DeepResearchSubmitResponse(
        ok=True,
        job_id=new_job_id,
        session_id=str(job.get("session_id") or ""),
        canvas_id=str(job.get("canvas_id") or ""),
    )


class EvidenceOptimizeRequest(BaseModel):
    section_title: str = Field(..., min_length=1, description="要优化的章节标题")
    web_providers: Optional[List[str]] = Field(None, description="允许使用的搜索源（None=全部）")
    web_source_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="每个搜索源配置")
    use_content_fetcher: Optional[str] = Field(None, description="全文抓取模式: auto | force | off")


@router.post("/deep-research/jobs/{job_id}/optimize-evidence")
def optimize_section_evidence_endpoint(
    job_id: str,
    body: EvidenceOptimizeRequest,
    optional_user_id: Optional[str] = Depends(get_optional_user_id),
) -> dict:
    """Trigger full T1+T2+T3 evidence optimization for a specific section.

    Available when section status is 'written', 'done', or 'evidence_scarce'.
    Loads the job checkpoint, runs a targeted 3-tier search using the section's
    accumulated evidence + gaps as context, and updates the checkpoint.
    """
    from src.collaboration.research.job_store import get_job
    from src.collaboration.research.agent import optimize_section_evidence

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")

    filters: Dict[str, Any] = {}
    if body.web_providers is not None:
        filters["web_providers"] = body.web_providers
    if body.web_source_configs is not None:
        filters["web_source_configs"] = body.web_source_configs
    # 配置优先级：UI/请求入参 > config > 代码默认（见 docs/configuration.md）
    if body.use_content_fetcher is not None:
        filters["use_content_fetcher"] = body.use_content_fetcher

    manager = get_manager(str(_CONFIG_PATH))
    req_payload = job.get("request_payload") or {}
    llm_client = manager.get_client(req_payload.get("llm_provider") or None)
    search_mode = str(req_payload.get("search_mode") or "hybrid")

    result = optimize_section_evidence(
        job_id=job_id,
        section_title=body.section_title,
        llm_client=llm_client,
        search_mode=search_mode,
        filters=filters or None,
    )
    return result


@router.post("/deep-research/jobs/{job_id}/review")
def review_deep_research_section(job_id: str, body: dict) -> dict:
    from src.collaboration.research.job_store import get_job, submit_review, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    section_id = str(body.get("section_id") or "").strip()
    if not section_id:
        raise HTTPException(status_code=400, detail="section_id 不能为空")
    # Normalize section id against confirmed outline to avoid tiny title drift
    # (e.g. accidental extra spaces / case changes) causing review-gate mismatch.
    req_payload = job.get("request") or {}
    outline = req_payload.get("confirmed_outline") or []
    if isinstance(outline, list) and outline:
        normalized = section_id.strip().lower()
        for item in outline:
            title = str(item or "").strip()
            if title and title.lower() == normalized:
                section_id = title
                break
    action = str(body.get("action") or "approve").strip().lower()
    feedback = str(body.get("feedback") or "")
    if action not in {"approve", "revise"}:
        raise HTTPException(status_code=400, detail="action must be approve|revise")
    result = submit_review(job_id, section_id, action=action, feedback=feedback)
    append_event(job_id, "section_review", {"section_id": section_id, "action": action, "feedback": feedback})
    resume_submitted = False
    runtime = _dr_get_suspended_runtime(job_id)
    if runtime:
        compiled = runtime.get("compiled")
        config = runtime.get("config")
        try:
            state_snapshot = compiled.get_state(config) if compiled is not None and isinstance(config, dict) else None
        except Exception:
            state_snapshot = None
        if state_snapshot is not None and getattr(state_snapshot, "next", ()):
            if _dr_mark_resume_inflight(job_id):
                try:
                    from src.collaboration.research.job_store import enqueue_resume_request
                    from src.utils.task_runner import get_worker_instance_id
                    enqueue_resume_request(
                        job_id=job_id,
                        owner_instance=get_worker_instance_id(),
                        source="review",
                        message="收到人工审核结果，等待恢复执行",
                    )
                    resume_submitted = True
                except Exception:
                    _dr_set_resume_idle(job_id)
    return {"ok": True, "resume_submitted": resume_submitted, **result}


@router.get("/deep-research/jobs/{job_id}/reviews")
def list_deep_research_reviews(job_id: str) -> dict:
    from src.collaboration.research.job_store import get_job, list_reviews

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {"job_id": job_id, "reviews": list_reviews(job_id)}


@router.get("/deep-research/resume-queue")
def list_deep_research_resume_queue(
    limit: int = 50,
    status: str | None = None,
    owner_instance: str | None = None,
    job_id: str | None = None,
) -> dict:
    from src.collaboration.research.job_store import list_resume_requests

    rows = list_resume_requests(
        limit=limit,
        status=status,
        owner_instance=owner_instance,
        job_id=job_id,
    )
    return {"items": rows, "count": len(rows)}


@router.post("/deep-research/resume-queue/cleanup")
def cleanup_deep_research_resume_queue(body: dict) -> dict:
    from src.collaboration.research.job_store import cleanup_resume_requests
    import time as _time

    statuses = body.get("statuses")
    if statuses is not None and not isinstance(statuses, list):
        raise HTTPException(status_code=400, detail="statuses 必须是字符串数组")
    before_hours = body.get("before_hours", 72)
    before_ts = None
    if before_hours is not None:
        try:
            hours = float(before_hours)
            if hours < 0:
                raise ValueError("hours < 0")
            before_ts = _time.time() - hours * 3600
        except Exception:
            raise HTTPException(status_code=400, detail="before_hours 必须是非负数字或 null")
    deleted = cleanup_resume_requests(
        statuses=[str(s) for s in (statuses or [])] if statuses else None,
        before_ts=before_ts,
        owner_instance=str(body.get("owner_instance") or "") or None,
        job_id=str(body.get("job_id") or "") or None,
    )
    return {"ok": True, "deleted": deleted}


@router.post("/deep-research/resume-queue/{resume_id}/retry")
def retry_deep_research_resume_request(resume_id: int, body: dict | None = None) -> dict:
    from src.collaboration.research.job_store import retry_resume_request, append_event
    from src.utils.task_runner import get_worker_instance_id

    payload = body or {}
    owner_instance = str(payload.get("owner_instance") or get_worker_instance_id())
    message = str(payload.get("message") or "手动重试恢复请求")
    try:
        row = retry_resume_request(
            resume_id=resume_id,
            owner_instance=owner_instance,
            message=message,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not row:
        raise HTTPException(status_code=404, detail="resume request 不存在")
    append_event(
        str(row.get("job_id") or ""),
        "resume_retry_requested",
        {"resume_id": resume_id, "owner_instance": owner_instance, "message": message},
    )
    return {"ok": True, "item": row}


# ── Gap Supplement Endpoints ──

@router.post("/deep-research/jobs/{job_id}/gap-supplement")
def submit_gap_supplement_endpoint(job_id: str, body: dict) -> dict:
    from src.collaboration.research.job_store import get_job, submit_gap_supplement, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    section_id = str(body.get("section_id") or "").strip()
    if not section_id:
        raise HTTPException(status_code=400, detail="section_id 不能为空")
    gap_text = str(body.get("gap_text") or "")
    supplement_type = str(body.get("supplement_type") or "material").strip()
    content = body.get("content") or {}
    result = submit_gap_supplement(
        job_id=job_id,
        section_id=section_id,
        gap_text=gap_text,
        supplement_type=supplement_type,
        content=content,
    )
    append_event(job_id, "gap_supplement", {
        "section_id": section_id,
        "gap_text": gap_text,
        "supplement_type": supplement_type,
    })
    return {"ok": True, **result}


@router.get("/deep-research/jobs/{job_id}/gap-supplements")
def list_gap_supplements_endpoint(job_id: str, section_id: str | None = None) -> dict:
    from src.collaboration.research.job_store import get_job, list_gap_supplements

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    supplements = list_gap_supplements(job_id, section_id=section_id)
    return {"job_id": job_id, "supplements": supplements}


# ── Research Insights Ledger Endpoints ──

@router.get("/deep-research/jobs/{job_id}/insights")
def list_insights_endpoint(
    job_id: str,
    insight_type: str | None = None,
    status: str | None = None,
) -> dict:
    from src.collaboration.research.job_store import get_job, list_insights

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    insights = list_insights(job_id, insight_type=insight_type, status=status)
    return {"job_id": job_id, "insights": insights}


@router.post("/deep-research/jobs/{job_id}/insights/{insight_id}/status")
def update_insight_status_endpoint(job_id: str, insight_id: int, body: dict) -> dict:
    from src.collaboration.research.job_store import get_job, update_insight_status

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    new_status = str(body.get("status") or "").strip()
    if new_status not in {"open", "addressed", "deferred"}:
        raise HTTPException(status_code=400, detail="status must be open|addressed|deferred")
    update_insight_status(insight_id, new_status)
    return {"ok": True, "insight_id": insight_id, "status": new_status}


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id),
) -> SessionInfo:
    store = get_session_store()
    meta = _ensure_session_access(session_id, current_user_id)
    if meta is not None:
        store.touch_session(session_id)
    if meta is None:
        # 可能是仅存在于 Deep Research 任务中的 session（ChatSession 无记录），尝试从 job 恢复
        try:
            from src.collaboration.research.job_store import get_latest_job_by_session
            latest_job = get_latest_job_by_session(session_id)
            if latest_job and (
                _is_admin_user(current_user_id)
                or str(latest_job.get("user_id") or "").strip() == current_user_id
            ):
                dashboard = latest_job.get("result_dashboard") or {}
                dash = dashboard if isinstance(dashboard, dict) and dashboard else None
                return SessionInfo(
                    session_id=session_id,
                    canvas_id=latest_job.get("canvas_id") or "",
                    stage="explore",
                    turn_count=0,
                    turns=[],
                    research_dashboard=dash,
                )
        except Exception:
            pass
        raise HTTPException(status_code=404, detail="session not found")
    turns = store.get_turns(session_id)
    turn_items = []
    for t in turns:
        # 将存储的 citations 字典列表转换为 ChatCitation 对象
        sources = [
            _chat_citation_from_dict(c)
            for c in (t.citations or [])
        ]
        try:
            ts_iso = t.timestamp.isoformat() if getattr(t, "timestamp", None) else None
        except (AttributeError, TypeError):
            ts_iso = None
        if not ts_iso:
            import datetime as _dt
            ts_iso = _dt.datetime.now().isoformat()
        turn_items.append(TurnItem(role=t.role, content=t.content, sources=sources, timestamp=ts_iso))

    # 恢复最近一次 Deep Research dashboard（刷新页面后仍可看到章节列表）
    latest_dashboard = None
    try:
        from src.collaboration.research.job_store import get_latest_job_by_session
        latest_job = get_latest_job_by_session(session_id)
        if latest_job and (
            _is_admin_user(current_user_id)
            or str(latest_job.get("user_id") or "").strip() == current_user_id
        ):
            dashboard = latest_job.get("result_dashboard") or {}
            if isinstance(dashboard, dict) and dashboard:
                latest_dashboard = dashboard
    except Exception:
        latest_dashboard = None

    return SessionInfo(
        session_id=meta["session_id"],
        canvas_id=meta["canvas_id"] or "",
        stage=meta.get("stage") or "explore",
        turn_count=len(turns),
        turns=turn_items,
        research_dashboard=latest_dashboard,
    )


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user_id: str = Depends(get_current_user_id),
) -> dict:
    _ensure_session_access(session_id, current_user_id)
    store = get_session_store()
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True, "session_id": session_id}


@router.get("/sessions", response_model=List[SessionListItem])
def list_sessions(
    limit: int = 100,
    current_user_id: str = Depends(get_current_user_id),
) -> List[SessionListItem]:
    """获取所有会话列表。会合并 Deep Research 任务中存在的 session，确保有后台调研的会话出现在历史中。"""
    store = get_session_store()
    is_admin = _is_admin_user(current_user_id)
    sessions = store.list_all_sessions(limit=limit, user_id=None if is_admin else current_user_id)
    session_ids = {s["session_id"] for s in sessions}
    items: list[SessionListItem] = [
        SessionListItem(
            session_id=s["session_id"],
            title=s["title"],
            canvas_id=s["canvas_id"] or "",
            stage=s.get("stage", "explore"),
            turn_count=s["turn_count"],
            session_type=s.get("session_type") or "chat",
            created_at=s["created_at"],
            updated_at=s["updated_at"],
        )
        for s in sessions
    ]
    # 合并 DR 任务中的 session：若某 job 的 session_id 不在列表中，补一条记录（修复「历史中没有后台调研」的 bug）
    try:
        from src.collaboration.research.job_store import list_jobs

        jobs = list_jobs(limit=50)
        for j in jobs:
            owner_id = str(j.get("user_id") or "").strip()
            if not is_admin and owner_id != current_user_id:
                continue
            sid = (j.get("session_id") or "").strip()
            if not sid or sid in session_ids:
                continue
            session_ids.add(sid)
            created = j.get("created_at") or 0
            updated = j.get("updated_at") or created
            created_iso = _ts_to_iso(created) if created else ""
            updated_iso = _ts_to_iso(updated) if updated else ""
            items.append(
                SessionListItem(
                    session_id=sid,
                    title=(j.get("topic") or "未命名 Deep Research")[:80],
                    canvas_id=(j.get("canvas_id") or "").strip(),
                    stage="explore",
                    turn_count=0,
                    session_type="research",
                    created_at=created_iso,
                    updated_at=updated_iso,
                )
            )
        # 按 updated_at 倒序，保持 limit
        items.sort(key=lambda x: x.updated_at or "", reverse=True)
        items = items[:limit]
    except Exception:
        pass
    return items


def _ts_to_iso(ts: float) -> str:
    from datetime import datetime

    try:
        return datetime.fromtimestamp(ts).isoformat()
    except (ValueError, OSError, TypeError):
        return "1970-01-01T00:00:00"
