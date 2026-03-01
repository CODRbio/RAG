"""
对话 API：POST /chat, POST /chat/stream, GET /sessions/{id}, DELETE /sessions/{id}
"""

import contextlib
import json
import math
import logging
import re
import time
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from config.settings import settings
from src.api.routes_auth import get_optional_user_id
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
)
from src.collaboration.memory.session_memory import (
    SessionStore,
    get_session_store,
    load_session_memory,
)
from src.collaboration.memory.working_memory import get_or_generate_working_memory
from src.collaboration.memory.persistent_store import get_user_profile
from src.collaboration.workflow import run_workflow
from src.llm.llm_manager import get_manager
from src.llm.react_loop import react_loop
from src.llm.tools import CORE_TOOLS, get_routed_skills, start_agent_chunk_collector, drain_agent_chunks, set_tool_collection
from src.collaboration.citation.manager import (
    _dedupe_citations,
    chunk_to_citation,
    merge_citations_by_document,
    resolve_response_citations,
    sync_evidence_to_canvas,
)
from src.collaboration.canvas.models import Citation
from dataclasses import asdict
from src.retrieval.service import (
    fuse_pools_with_gap_protection,
    get_retrieval_service,
    _hit_to_chunk as service_hit_to_chunk,
)
from src.retrieval.evidence import EvidenceChunk, EvidencePack
from src.generation.evidence_synthesizer import EvidenceSynthesizer, build_synthesis_system_prompt
from src.collaboration.auto_complete import AutoCompleteService
from src.tasks import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus

router = APIRouter(tags=["chat"])

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
_DEEP_RESEARCH_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DEEP_RESEARCH_CANCEL_LOCK = threading.Lock()
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
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "[dr] eager slot release failed job_id=%s: %s", job_id, exc
        )


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
        result = start_deep_research(**{**start_kwargs, "progress_callback": _progress_cb})
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
        _chat_logger.error(
            "[start-job] %s failed | session=%s error=%s",
            job_id[:12],
            session_id[:12],
            exc,
            exc_info=True,
        )
        err_state: Dict[str, Any] = {
            "status": "error",
            "session_id": session_id,
            "result": None,
            "error": str(exc),
        }
        with _START_JOBS_LOCK:
            _START_JOBS[job_id] = err_state
        _persist_start_job(job_id, err_state)
        try:
            _update_db_job(job_id, status="error", error_message=str(exc)[:500], message="规划失败")
        except Exception:
            pass
    finally:
        _hb_stop.set()  # signal heartbeat thread to stop


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


def _serialize_citation(c: Citation | str) -> dict:
    """将 Citation 对象或字符串序列化为字典。"""
    if isinstance(c, str):
        return {"cite_key": c, "title": "", "authors": [], "year": None, "doc_id": None, "url": None, "provider": None}
    return {
        "cite_key": c.cite_key or c.id,
        "title": c.title or "",
        "authors": c.authors or [],
        "year": c.year,
        "doc_id": c.doc_id,
        "url": c.url,
        "doi": c.doi,
        "bbox": getattr(c, "bbox", None),
        "page_num": getattr(c, "page_num", None),
        "provider": getattr(c, "provider", None),
    }


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
    if body.use_query_optimizer is not None:
        filters["use_query_optimizer"] = body.use_query_optimizer
    if body.query_optimizer_max_queries is not None:
        filters["query_optimizer_max_queries"] = body.query_optimizer_max_queries
    if body.local_threshold is not None:
        filters["local_threshold"] = body.local_threshold
    if body.year_start is not None:
        filters["year_start"] = body.year_start
    if body.year_end is not None:
        filters["year_end"] = body.year_end
    if body.step_top_k is not None:
        filters["step_top_k"] = body.step_top_k
    if getattr(body, "write_top_k", None) is not None:
        filters["write_top_k"] = body.write_top_k
    if body.reranker_mode:
        filters["reranker_mode"] = body.reranker_mode
    if getattr(body, "llm_provider", None):
        filters["llm_provider"] = body.llm_provider
    if getattr(body, "ultra_lite_provider", None):
        filters["ultra_lite_provider"] = body.ultra_lite_provider
    if body.model_override:
        filters["model_override"] = body.model_override
    if body.use_content_fetcher is not None:
        filters["use_content_fetcher"] = body.use_content_fetcher
    if body.collection:
        filters["collection"] = body.collection
    return filters


# ── Chat 证据充分性评估（LLM 判断，借鉴 Deep Research evaluate_sufficiency）──
_CHAT_EVIDENCE_CONTEXT_CAP = 12000  # 送入评估的 evidence 最大字符，避免超长


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


class _ChatGapQueriesResponse(BaseModel):
    """LLM 返回：针对证据缺口的 1–3 条可检索 query"""
    gap_queries: List[str] = Field(default_factory=list)


def _generate_chat_gap_queries(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> List[str]:
    """
    根据用户问题和已有证据，生成 1–3 条针对缺口的检索 query。
    借鉴 Deep Research 的 gap 思路，返回非空泛、可直接检索的短 query 列表。
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
            return [str(q).strip() for q in parsed.gap_queries if str(q).strip()][:3]
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


def _build_deep_research_filters(body: Any) -> dict:
    """Build filters from deep-research start/confirm request objects."""
    filters: Dict[str, Any] = {}
    for key in (
        "web_providers",
        "web_source_configs",
        "serpapi_ratio",
        "use_query_optimizer",
        "query_optimizer_max_queries",
        "local_top_k",
        "local_threshold",
        "year_start",
        "year_end",
        "step_top_k",
        "write_top_k",
        "reranker_mode",
        "llm_provider",
        "ultra_lite_provider",
        "model_override",
        "collection",
        "use_content_fetcher",
        "gap_query_intent",
    ):
        val = getattr(body, key, None)
        if val is not None and val != "":
            filters[key] = val
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
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str], dict | None, bool, Optional[str]]:
    import time as _time
    from src.debug import get_debug_logger
    dbg = get_debug_logger()

    t_start = _time.perf_counter()

    session_id = body.session_id
    store = get_session_store()
    if not session_id:
        session_id = store.create_session(canvas_id=body.canvas_id or "")
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
        " | year=%s~%s | optimizer=%s",
        session_id[:12], message[:60],
        body.llm_provider or "default", body.model_override or "default",
        search_mode, (body.collection or settings.collection.global_), agent_mode,
        body.local_top_k, body.step_top_k,
        body.reranker_mode,
        ",".join(body.web_providers) if body.web_providers else "none",
        body.serpapi_ratio,
        body.year_start, body.year_end,
        body.use_query_optimizer,
    )

    # ── 2. 意图 + 上下文分析（单次 ultra-lite LLM 调用）──
    request_mode = (body.mode or "chat").strip().lower()
    ctx_analysis: ContextAnalysis | None = None

    if request_mode == "deep_research":
        parsed = ParsedIntent(
            intent_type=IntentType.DEEP_RESEARCH,
            confidence=1.0,
            params={"args": message},
            raw_input=message,
        )
        _chat_logger.info("[chat] ② 意图判断 → deep_research (前端 mode 指定)")
    elif message.startswith("/"):
        parser = IntentParser(lite_client)
        parsed = parser.parse(message, current_stage=current_stage, history=history_for_intent)
        _chat_logger.info(
            "[chat] ② 意图判断 → %s (命令解析, confidence=%.2f)",
            parsed.intent_type.value, parsed.confidence,
        )
    else:
        # 统一上下文分析：一次 ultra-lite 调用完成 意图分类 + 指代检测 + query 重写/澄清
        t_analyze = _time.perf_counter()
        ctx_analysis = analyze_chat_context(
            message=message,
            rolling_summary=memory.rolling_summary,
            history=history_for_intent,
            llm_client=lite_client,
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

    meta = store.get_session_meta(session_id)
    canvas_id = (meta or {}).get("canvas_id") or ""

    # ── Deep Research 分支 ──
    if is_deep_research(parsed):
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
        memory.update_rolling_summary(client)
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

    # ── 2½. 指代澄清短路（LLM 判定无法推断时直接追问用户）──
    if ctx_analysis and ctx_analysis.context_status == "needs_clarification" and ctx_analysis.clarification:
        _chat_logger.info("[chat] ②½ 指代不明 → 返回澄清问题，跳过检索")
        memory.add_turn("user", message)
        memory.add_turn("assistant", ctx_analysis.clarification, citations=[])
        memory.update_rolling_summary(client)
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
        query_needs_rag = _classify_query(message, history_for_intent, lite_client)
    do_retrieval = search_mode != "none" and query_needs_rag and agent_mode != "autonomous"
    _agent_mode_before_route = agent_mode
    if not query_needs_rag:
        agent_mode = "standard"

    _chat_logger.info(
        "[chat] ③ 查询路由 → %s | do_retrieval=%s (search_mode=%s, agent_mode=%s)",
        "rag" if query_needs_rag else "chat",
        do_retrieval, search_mode, agent_mode,
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

    # ── 4. 检索 query 构建（仅 rag 模式）──
    if do_retrieval:
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
            "[chat] ④ Query 构建 | original=%r → effective=%r → query=%r | 耗时=%.0fms",
            message[:40], effective_msg[:40], query[:60], query_ms,
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
        _chat_logger.info("[chat] ④ Query 构建 → 跳过 (不需要检索)")

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

    # ── 4¾. 查询与本地库范围检查：明显不符则提示换库/本会话不用本地库 ──
    target_collection = (body.collection or "").strip() or None
    actual_collection = target_collection or settings.collection.global_
    if (
        do_retrieval
        and effective_use_local
        and effective_search_mode in ("local", "hybrid")
        and actual_collection
        and query
    ):
        scope_result = check_query_collection_scope(actual_collection, query, lite_client)
        # 用户已在本会话中明确选择「仍使用当前库」时，不再提示，直接走检索
        if scope_result == "mismatch" and session_preferences.get("local_db") != "use":
            mismatch_msg = (
                f"当前问题与本地知识库（{actual_collection}）主题可能不符。"
                "您可以选择：**本会话暂不使用本地库**（仅用网络检索），或**仍使用当前库**继续检索。"
            )
            _chat_logger.info("[chat] ④¾ 查询与本地库范围不符 → 提示用户选择")
            memory.add_turn("user", message)
            memory.add_turn("assistant", mismatch_msg, citations=[])
            memory.update_rolling_summary(client)
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

    # ── 5. 检索执行 ──
    if do_retrieval:
        t_retrieval = _time.perf_counter()
        retrieval = get_retrieval_service(collection=target_collection)
        filters = _build_filters(body)
        # 强制 Chat 使用 bge_only reranker（速度优先）。UI 传入的 cascade/colbert
        # 在 hybrid 模式下仅用于写作阶段（Deep Research）。
        filters["reranker_mode"] = "bge_only"
        pack = retrieval.search(
            query=query or message,
            mode=effective_search_mode,
            filters=filters or None,
            top_k=body.local_top_k,
        )
        # write_top_k = per-output-unit evidence cap (Chat one Q&A = one unit). Fallback to step_top_k then all.
        write_k = getattr(body, "write_top_k", None) or body.step_top_k or len(pack.chunks)
        max_chunks_for_context = min(write_k, len(pack.chunks))
        synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
        context_str, synthesis_meta = synthesizer.synthesize(pack)
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
            " | soft_wait_ms=%s | 耗时=%.0fms",
            search_mode, body.local_top_k, body.step_top_k,
            getattr(body, "write_top_k", None),
            filters.get("reranker_mode", "bge_only"),
            len(pack.chunks), max_chunks_for_context, ",".join(pack.sources_used),
            _fusion_diag.get("main_in", "-"),
            _fusion_diag.get("gap_in", "-"),
            _fusion_diag.get("output_count", "-"),
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
        _chat_logger.info("[chat] ⑤ 检索 → 跳过 (路由判定为 chat)")

    # ── 5.5 证据充分性检查：LLM 判断是否有一致、实质的证据支撑（借鉴 DR），失败时用数量兜底 ──
    evidence_scarce = False
    _evidence_distinct_docs = 0
    _coverage_score: Optional[float] = None
    _sufficiency_reason: Optional[str] = None
    if do_retrieval and pack:
        _doc_keys: set[str] = set()
        for c in pack.chunks:
            if getattr(c, "doi", None):
                _doc_keys.add(f"doi:{c.doi}")
            else:
                _doc_keys.add(c.doc_group_key)
        _evidence_distinct_docs = len(_doc_keys)
        # 数量兜底：无有效 context 或 LLM 失败时使用
        _numeric_fallback = len(pack.chunks) < 3 or _evidence_distinct_docs < 2
        if query_needs_rag and context_str and context_str.strip():
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
        if agent_mode == "standard" and evidence_scarce and query_needs_rag:
            agent_mode = "assist"
            _chat_logger.info(
                "[chat] ⑤½ 证据不足 → 自动升级 agent_mode: standard → assist (允许工具补搜)",
            )

    # ── 5.6 证据不足时：生成 gap query、补搜、main+gap 一次融合（借鉴 DR）──
    if (
        do_retrieval
        and pack
        and evidence_scarce
        and query_needs_rag
        and effective_search_mode != "none"
    ):
        gap_queries = _generate_chat_gap_queries(
            message,
            context_str or "",
            lite_client,
            model_override=body.model_override or None,
        )
        gap_queries = [q for q in (gap_queries or []) if (q or "").strip()][:3]
        if gap_queries:
            step_k = body.step_top_k or body.local_top_k or 20
            supp_k = max(5, step_k // 2)
            main_candidates = [_chunk_to_hit(c) for c in pack.chunks]
            gap_candidates: List[Dict[str, Any]] = []
            _supp_total = 0
            for gq in gap_queries:
                try:
                    supp_pack = retrieval.search(
                        query=gq,
                        mode=effective_search_mode,
                        filters=filters or None,
                        top_k=supp_k,
                    )
                    for c in supp_pack.chunks:
                        gap_candidates.append(_chunk_to_hit(c))
                    _supp_total += len(supp_pack.chunks)
                except Exception as e:
                    _chat_logger.warning("[chat] gap supplement search failed for %r: %s", gq[:50], e)
            if gap_candidates:
                fusion_diag: Dict[str, Any] = {}
                fused = fuse_pools_with_gap_protection(
                    query=query or message,
                    main_candidates=main_candidates,
                    gap_candidates=gap_candidates,
                    top_k=step_k,
                    gap_min_keep=math.ceil(step_k * float(getattr(settings.search, "chat_gap_ratio", 0.2))),
                    gap_ratio=float(getattr(settings.search, "chat_gap_ratio", 0.2)),
                    rank_pool_multiplier=float(getattr(settings.search, "chat_rank_pool_multiplier", 3.0)),
                    reranker_mode=(filters or {}).get("reranker_mode") or "bge_only",
                    diag=fusion_diag,
                )
                new_chunks = [
                    service_hit_to_chunk(h, h.get("_source_type", "dense"), query or message)
                    for h in fused
                ]
                _pf = fusion_diag.get("pool_fusion") or {}
                pack = EvidencePack(
                    query=pack.query,
                    chunks=new_chunks,
                    total_candidates=pack.total_candidates + _supp_total,
                    retrieval_time_ms=pack.retrieval_time_ms,
                    sources_used=pack.sources_used,
                    diagnostics={
                        **(pack.diagnostics or {}),
                        "pool_fusion": _pf,
                        "chat_gap_queries": gap_queries,
                        "chat_gap_supplement_chunks": len(gap_candidates),
                    },
                )
                write_k = getattr(body, "write_top_k", None) or body.step_top_k or len(pack.chunks)
                max_chunks_for_context = min(write_k, len(pack.chunks))
                synthesizer = EvidenceSynthesizer(max_chunks=max_chunks_for_context)
                context_str, synthesis_meta = synthesizer.synthesize(pack)
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
                    "[chat] ⑤¾ gap 补搜完成 | gap_queries=%d | gap_candidates=%d | fused_out=%d | fusion=%s",
                    len(gap_queries), len(gap_candidates), len(new_chunks), _pf,
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
    history = memory.get_context_window(n=10)
    messages = [{"role": "system", "content": system_content}]
    for t in history:
        messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": message})

    # ── 8. LLM 生成 ──
    use_agent = agent_mode in ("assist", "autonomous")
    _chat_logger.info(
        "[chat] ⑦½ Agent 决策 | use_agent=%s | agent_mode=%s (用户请求=%s) | "
        "query_needs_rag=%s | search_mode=%s | evidence_scarce=%s | do_retrieval=%s",
        use_agent, agent_mode, _agent_mode_before_route,
        query_needs_rag, search_mode,
        evidence_scarce if do_retrieval and pack else "N/A(未检索)",
        do_retrieval,
    )
    tool_trace = None

    # 根据模式注入不同的 Agent hint
    if agent_mode == "assist" and evidence_scarce:
        messages[0]["content"] = messages[0]["content"] + _pm.render(
            "chat_agent_evidence_scarce_hint.txt",
            chunk_count=len(pack.chunks) if pack else 0,
            distinct_docs=_evidence_distinct_docs,
        )
    elif agent_mode == "assist" and do_retrieval and context_str:
        messages[0]["content"] = messages[0]["content"] + _pm.render("chat_agent_hint.txt")
    elif agent_mode == "autonomous":
        messages[0]["content"] = messages[0]["content"] + _pm.render("chat_agent_autonomous_hint.txt")

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
    try:
        if use_agent:
            start_agent_chunk_collector()
            set_tool_collection(target_collection)
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
                max_tokens=None,
            )
            agent_extra_chunks = drain_agent_chunks()
            agent_chunk_ids = {c.chunk_id for c in agent_extra_chunks}
            if agent_extra_chunks:
                if pack is None:
                    pack = EvidencePack(query=query or message, chunks=[])
                existing_ids = {c.chunk_id for c in pack.chunks}
                for c in agent_extra_chunks:
                    if c.chunk_id not in existing_ids:
                        pack.chunks.append(c)
                        existing_ids.add(c.chunk_id)
                _chat_logger.info(
                    "[chat] ⑧a Agent 工具证据合并 | extra_chunks=%d | total_chunks=%d",
                    len(agent_extra_chunks), len(pack.chunks),
                )
            response_text = react_result.final_text.strip()
            tool_trace = react_result.tool_trace if react_result.tool_trace else None
            llm_ms = (_time.perf_counter() - t_llm) * 1000
            _chat_logger.info(
                "[chat] ⑧ Agent 完成 | iterations=%d | tools_called=%d | routed=%d/%d | 耗时=%.0fms",
                react_result.iterations, len(react_result.tool_trace),
                len(routed_tools), len(CORE_TOOLS), llm_ms,
            )
        else:
            react_result = None
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
            )
    except Exception as llm_err:
        react_result = None
        llm_ms = (_time.perf_counter() - t_llm) * 1000
        _chat_logger.error("[chat] ⑧ LLM 失败 | error=%s | 耗时=%.0fms", llm_err, llm_ms)
        response_text = f"[LLM 调用失败] {type(llm_err).__name__}: {llm_err}\n\n请尝试切换其他模型。"

    # ── 9. 引文后处理：将 [ref_hash] 替换为正式 cite_key ──
    ref_map: dict[str, str] = {}
    all_chunks = (pack.chunks if pack else [])
    if all_chunks:
        response_text, citations, ref_map = resolve_response_citations(
            response_text, all_chunks,
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

        _chat_logger.info(
            "[chat] ⑨ 引文后处理 | cited_docs=%d | ref_map_size=%d | chunk_stats=%s | cite_stats=%s",
            len(citations), len(ref_map), chunk_counts, cite_counts,
        )

    # ── 9b. tools_contributed 判定 ──
    tools_contributed = False
    cited_from_agent_count = 0
    non_retrieval_tools_ok = 0
    if use_agent and react_result and react_result.tool_trace:
        _non_retrieval = {"run_code", "explore_graph", "canvas", "get_citations", "compare_papers"}
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

    # ── 10. 写入 Memory ──
    memory.add_turn("user", message)
    citations_data = [_serialize_citation(c) for c in citations] if citations else []
    memory.add_turn("assistant", response_text, citations=citations_data)
    memory.update_rolling_summary(client)

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

    return session_id, response_text, citations, evidence_summary, parsed, None, tool_trace, ref_map, agent_debug_data, False, None


def _run_chat(
    body: ChatRequest,
    optional_user_id: str | None = None,
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str], dict | None, bool, Optional[str]]:
    """包装 _run_chat_impl：当前端开启调试面板时临时提升本请求的日志级别为 DEBUG。"""
    with _request_debug_level(body):
        return _run_chat_impl(body, optional_user_id)


def _citation_to_chat_citation(c: Citation) -> ChatCitation:
    """将 Citation 对象转换为 ChatCitation schema。"""
    return ChatCitation(
        cite_key=c.cite_key or c.id,
        title=c.title or "",
        authors=c.authors or [],
        year=c.year,
        doc_id=c.doc_id,
        url=c.url,
        doi=c.doi,
        bbox=getattr(c, "bbox", None),
        page_num=getattr(c, "page_num", None),
        provider=getattr(c, "provider", None),
    )


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
        created_session_id = store.create_session(canvas_id=body.canvas_id or "")
        payload["session_id"] = created_session_id
        # 提问时即用首条消息作为会话标题，历史里立即显示问题而非「未命名对话」
        first_msg = (body.message or "").strip()
        if first_msg:
            store.update_session_meta(created_session_id, {"title": first_msg[:80]})
    else:
        created_session_id = body.session_id
    payload["_optional_user_id"] = optional_user_id
    session_id = created_session_id or ""
    user_id = optional_user_id or body.user_id or ""
    try:
        task_id = q.submit(TaskKind.chat, session_id, user_id or "", payload)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue submit failed: {e}")
    if _obs_metrics and hasattr(_obs_metrics, "task_queue_submitted_total"):
        _obs_metrics.task_queue_submitted_total.labels(kind="chat").inc()
    return ChatSubmitResponse(task_id=task_id)


def _task_event_stream(task_id: str, q):
    """SSE 事件流生成器。

    关键顺序：先读取所有待推送事件，再检查是否终态。
    这样即使任务瞬间完成（如 mismatch 路径无 LLM 调用），
    meta/local_db_choice/delta/done 事件也能全部送达前端，
    不会被「已终态→直接 return」的提前退出所跳过。
    """
    import time
    last_id = "-"
    if _obs_metrics:
        _obs_metrics.active_connections.inc()
    try:
        while True:
            # ① 先读取所有已推送但未发送的事件
            events = q.read_events(task_id, after_id=last_id)
            for ev in events:
                last_id = ev.get("id", last_id)
                typ = ev.get("type", "message")
                data = ev.get("data", {})
                yield f"event: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
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
                    yield f"event: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
                    if typ in ("done", "error", "cancelled", "timeout"):
                        return
                # 所有事件已发送完毕，补发终态通知
                yield f"event: {state.status.value}\ndata: {json.dumps({'status': state.status.value}, ensure_ascii=False)}\n\n"
                return

            time.sleep(0.3)
    finally:
        if _obs_metrics:
            _obs_metrics.active_connections.dec()


@router.get("/chat/stream/{task_id}")
def chat_stream_by_task_id(task_id: str):
    """SSE 订阅任务流式输出；事件由调度器在执行时推送。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    return StreamingResponse(_task_event_stream(task_id, q), media_type="text/event-stream")


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
        created_session_id = store.create_session(canvas_id=body.canvas_id or "")
        payload["session_id"] = created_session_id
        # 提问时即用首条消息作为会话标题，历史里立即显示问题而非「未命名对话」
        first_msg = (body.message or "").strip()
        if first_msg:
            store.update_session_meta(created_session_id, {"title": first_msg[:80]})
    else:
        created_session_id = body.session_id
    payload["_optional_user_id"] = optional_user_id
    session_id = created_session_id or ""
    user_id = optional_user_id or body.user_id or ""
    try:
        task_id = q.submit(TaskKind.chat, session_id, user_id or "", payload)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue submit failed: {e}")
    if _obs_metrics and hasattr(_obs_metrics, "task_queue_submitted_total"):
        _obs_metrics.task_queue_submitted_total.labels(kind="chat").inc()
    return StreamingResponse(_task_event_stream(task_id, q), media_type="text/event-stream")


@router.post("/intent/detect", response_model=IntentDetectResponse)
def detect_intent(body: IntentDetectRequest) -> IntentDetectResponse:
    """
    意图检测 API（简化版）：Chat vs Deep Research 二分类。
    检索由前端 UI 开关决定，此处只判断执行模式。
    LLM 优先使用 body.llm_provider（UI），无则使用 config 默认。
    """
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_lite_client(body.llm_provider or None)
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
        session_id = store.create_session(session_type="research")

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
            prelim_resp = ppl_client.chat(
                [{"role": "user", "content": prelim_prompt}],
                model=prelim_model,
                timeout_seconds=40,
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
    )
    user_id = optional_user_id or body.user_id or ""
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
        if db_job and db_job.get("status") in ("planning", "error"):
            db_status = db_job["status"]
            _chat_logger.info("[start-job] status served from DB job_id=%s status=%s", job_id[:12], db_status)
            return DeepResearchStartStatusResponse(
                job_id=job_id,
                status="running" if db_status == "planning" else "error",
                session_id=db_job.get("session_id", ""),
                error=db_job.get("error_message") if db_status == "error" else None,
                canvas_id=db_job.get("canvas_id", ""),
                current_stage=db_job.get("current_stage", ""),
                progress=0,
            )
        raise HTTPException(status_code=404, detail=f"Start job not found: {job_id}")

    status = job.get("status", "running")
    session_id = job.get("session_id", "")
    error = job.get("error")
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
    session_id = body.session_id or store.create_session(canvas_id=body.canvas_id or "")
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    user_id = optional_user_id or body.user_id or ""
    filters = _build_deep_research_filters(body)

    def _progress_cb(event_type: str, payload: Dict[str, Any]) -> None:
        append_event(job_id, "progress" if event_type != "warning" else "warning", {"type": event_type, **(payload or {})})
        stage = str(payload.get("section") or payload.get("type") or "")
        update_job(job_id, current_stage=stage, message=str(payload.get("message") or payload.get("section") or event_type))

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
        _dr_pop_suspended_runtime(job_id)
    finally:
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
    append_event(
        job_id,
        "done",
        {
            "session_id": session_id,
            "canvas_id": canvas_id,
            "total_time_ms": total_time_ms,
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
        _dr_pop_suspended_runtime(job_id)
    finally:
        _dr_set_resume_idle(job_id)
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
    session_id = body.session_id or store.create_session(canvas_id=body.canvas_id or "")
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

    session_id = body.session_id or get_session_store().create_session(canvas_id=body.canvas_id or "")
    store = get_session_store()
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    user_id = optional_user_id or body.user_id or ""
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
        yield "event: done\ndata: {}\n\n"

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


def _dr_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
                    yield _dr_sse(ev["event"], _ev_payload(ev))
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
                    yield _dr_sse(ev["event"], _ev_payload(ev))
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
        update_job(job_id, status="cancelled", message=msg, finished_at=_t.time())
        append_event(job_id, "cancelled", {"job_id": job_id, "message": msg})
        _dr_request_cancel(job_id)  # signal the thread too so it exits cleanly
        _dr_release_slot_eager(job_id)
        return {"ok": True, "job_id": job_id, "status": "cancelled"}
    _dr_request_cancel(job_id)
    _dr_release_slot_eager(job_id)  # unblock any queued job for the same session immediately
    update_job(job_id, status="cancelling", message="收到停止请求，正在终止任务...")
    append_event(job_id, "cancel_requested", {"job_id": job_id, "message": "已请求停止"})
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


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
    if source_status in {"pending", "running", "cancelling", "waiting_review"}:
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
    if source_status in {"pending", "running", "cancelling", "waiting_review"}:
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
    if source_status in {"pending", "running", "cancelling", "waiting_review"}:
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
    if source_status in {"pending", "running", "cancelling", "waiting_review"}:
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
def get_session(session_id: str) -> SessionInfo:
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    if meta is not None:
        store.touch_session(session_id)
    if meta is None:
        # 可能是仅存在于 Deep Research 任务中的 session（ChatSession 无记录），尝试从 job 恢复
        try:
            from src.collaboration.research.job_store import get_latest_job_by_session
            latest_job = get_latest_job_by_session(session_id)
            if latest_job:
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
            ChatCitation(
                cite_key=c.get("cite_key", ""),
                title=c.get("title", ""),
                authors=c.get("authors", []),
                year=c.get("year"),
                doc_id=c.get("doc_id"),
                url=c.get("url"),
                doi=c.get("doi"),
                bbox=c.get("bbox"),
                page_num=c.get("page_num"),
            )
            for c in (t.citations or [])
        ]
        turn_items.append(TurnItem(role=t.role, content=t.content, sources=sources))

    # 恢复最近一次 Deep Research dashboard（刷新页面后仍可看到章节列表）
    latest_dashboard = None
    try:
        from src.collaboration.research.job_store import get_latest_job_by_session
        latest_job = get_latest_job_by_session(session_id)
        if latest_job:
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
def delete_session(session_id: str) -> dict:
    store = get_session_store()
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True, "session_id": session_id}


@router.get("/sessions", response_model=List[SessionListItem])
def list_sessions(limit: int = 100) -> List[SessionListItem]:
    """获取所有会话列表。会合并 Deep Research 任务中存在的 session，确保有后台调研的会话出现在历史中。"""
    store = get_session_store()
    sessions = store.list_all_sessions(limit=limit)
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
