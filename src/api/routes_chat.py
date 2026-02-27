"""
对话 API：POST /chat, POST /chat/stream, GET /sessions/{id}, DELETE /sessions/{id}
"""

import json
import re
import threading
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
    ClarifyRequest,
    ClarifyResponse,
    ClarifyQuestion,
    DeepResearchRequest,
    DeepResearchStartRequest,
    DeepResearchStartResponse,
    DeepResearchConfirmRequest,
    DeepResearchSubmitResponse,
    DeepResearchRestartPhaseRequest,
    DeepResearchRestartSectionRequest,
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
    IntentParser,
    IntentType,
    ParsedIntent,
    build_search_query_from_context,
    is_deep_research,
)
from src.collaboration.memory.session_memory import (
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
from src.retrieval.service import get_retrieval_service
from src.generation.evidence_synthesizer import EvidenceSynthesizer, build_synthesis_system_prompt
from src.collaboration.auto_complete import AutoCompleteService

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


def _run_chat(
    body: ChatRequest,
    optional_user_id: str | None = None,
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str], dict | None]:
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

    # ── 2. 意图判断（Chat vs Deep Research）──
    request_mode = (body.mode or "chat").strip().lower()
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
        parsed = ParsedIntent(
            intent_type=IntentType.CHAT,
            confidence=1.0,
            raw_input=message,
        )

    meta = store.get_session_meta(session_id)
    canvas_id = (meta or {}).get("canvas_id") or ""

    # ── Deep Research 分支 ──
    if is_deep_research(parsed):
        _chat_logger.info("[chat] ── 进入 Deep Research 分支 ──")
        query = build_search_query_from_context(
            parsed, message, history_for_intent,
            llm_client=client, enforce_english_if_input_english=True,
            rolling_summary=memory.rolling_summary,
        )
        topic = (parsed.params.get("args") or message).strip() or "综述"
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
        return session_id, response_text, result.citations, evidence_summary, parsed, dashboard_data, None, {}, None

    # ── 3. Chat 分支：智能路由（正则 → LLM → 严格解析）──
    t_route = _time.perf_counter()
    query_needs_rag = _classify_query(message, history_for_intent, lite_client)
    # autonomous 模式跳过预检索，由 Agent 自主决定是否检索
    do_retrieval = search_mode != "none" and query_needs_rag and agent_mode != "autonomous"
    # 闲聊时不启动 Agent
    if not query_needs_rag:
        agent_mode = "standard"
    route_ms = (_time.perf_counter() - t_route) * 1000

    _chat_logger.info(
        "[chat] ③ 查询路由 → %s | do_retrieval=%s (search_mode=%s, agent_mode=%s) | 耗时=%.0fms",
        "rag" if query_needs_rag else "chat",
        do_retrieval, search_mode, agent_mode, route_ms,
    )
    dbg.log_query_route(
        session_id,
        message=message[:200],
        decision="rag" if query_needs_rag else "chat",
        do_retrieval=do_retrieval,
        search_mode=search_mode,
        agent_mode=agent_mode,
        latency_ms=round(route_ms),
    )

    # ── 4. 检索 query 构建（仅 rag 模式）──
    if do_retrieval:
        t_query = _time.perf_counter()
        query = build_search_query_from_context(
            parsed, message, history_for_intent,
            llm_client=client, enforce_english_if_input_english=True,
            rolling_summary=memory.rolling_summary,
        )
        query_ms = (_time.perf_counter() - t_query) * 1000
        _chat_logger.info(
            "[chat] ④ Query 构建 | original=%r → query=%r | 耗时=%.0fms",
            message[:40], query[:60], query_ms,
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

    # ── 5. 检索执行 ──
    target_collection = (body.collection or "").strip() or None
    if do_retrieval:
        t_retrieval = _time.perf_counter()
        retrieval = get_retrieval_service(collection=target_collection)
        filters = _build_filters(body)
        filters["reranker_mode"] = "bge_only"  # Chat 检索固定 bge_only，优先速度
        pack = retrieval.search(
            query=query or message,
            mode=search_mode,
            filters=filters or None,
            top_k=body.local_top_k,
        )
        synthesizer = EvidenceSynthesizer(max_chunks=len(pack.chunks))
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
        _chat_logger.info(
            "[chat] ⑤ 检索完成 | mode=%s | top_k=%s | step_top_k=%s | reranker_mode=%s | chunks=%d | sources=%s | 耗时=%.0fms",
            search_mode, body.local_top_k, body.step_top_k, body.reranker_mode,
            len(pack.chunks), ",".join(pack.sources_used), retrieval_ms,
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

    # ── 5.5 轻量证据充分性检查 ──
    evidence_scarce = False
    _evidence_distinct_docs = 0
    if do_retrieval and pack:
        _doc_keys: set[str] = set()
        for c in pack.chunks:
            if getattr(c, "doi", None):
                _doc_keys.add(f"doi:{c.doi}")
            else:
                _doc_keys.add(c.doc_group_key)
        _evidence_distinct_docs = len(_doc_keys)
        if len(pack.chunks) < 3 or _evidence_distinct_docs < 2:
            evidence_scarce = True
            evidence_summary.evidence_scarce = True
            _chat_logger.info(
                "[chat] ⑤½ 证据不足 | chunks=%d | distinct_docs=%d → 标记 evidence_scarce",
                len(pack.chunks), _evidence_distinct_docs,
            )
        if agent_mode == "standard" and evidence_scarce and query_needs_rag:
            agent_mode = "assist"
            _chat_logger.info(
                "[chat] ⑤½ 证据不足 → 自动升级 agent_mode: standard → assist (允许工具补搜)",
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
                max_iterations=8,
                model=body.model_override or None,
                session_id=session_id,
                max_tokens=None,
            )
            agent_extra_chunks = drain_agent_chunks()
            agent_chunk_ids = {c.chunk_id for c in agent_extra_chunks}
            if agent_extra_chunks:
                if pack is None:
                    from src.retrieval.evidence import EvidencePack
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
    memory.update_rolling_summary(client, interval=4)

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

    return session_id, response_text, citations, evidence_summary, parsed, None, tool_trace, ref_map, agent_debug_data


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
    session_id, response_text, citations, evidence_summary, _parsed, _dashboard, _trace, _ref_map, _agent_debug = _run_chat(body, optional_user_id)
    return ChatResponse(
        session_id=session_id,
        response=response_text,
        citations=[_citation_to_chat_citation(c) for c in citations],
        evidence_summary=evidence_summary,
    )


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> StreamingResponse:
    session_id, response_text, citations, evidence_summary, parsed, dashboard_data, tool_trace_data, ref_map, agent_debug_payload = _run_chat(body, optional_user_id)
    
    # 获取当前会话阶段
    store = get_session_store()
    current_stage = store.get_session_stage(session_id) or "explore"

    def event_stream():
        if _obs_metrics:
            _obs_metrics.active_connections.inc()
        try:
            # 序列化 citations 为完整的引用信息
            citations_data = [_serialize_citation(c) for c in citations]
            # 返回模式/意图信息
            intent_info = {
                "mode": parsed.intent_type.value,  # "chat" or "deep_research"
                "intent_type": parsed.intent_type.value,  # 兼容
                "confidence": parsed.confidence,
                "from_command": parsed.from_command,
            }
            # 获取 canvas_id
            session_meta = store.get_session_meta(session_id)
            canvas_id = (session_meta or {}).get("canvas_id") or ""
            meta = {
                "session_id": session_id,
                "canvas_id": canvas_id,  # 返回 canvas_id 供前端加载画布
                "citations": citations_data,
                "ref_map": ref_map or {},  # ref_hash → cite_key 映射
                "evidence_summary": evidence_summary.model_dump() if evidence_summary else None,
                "intent": intent_info,
                "current_stage": current_stage,  # 返回当前工作流阶段
            }
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            # Deep Research 进度仪表盘
            if dashboard_data:
                dash = dashboard_data if isinstance(dashboard_data, dict) else {}
                yield f"event: dashboard\ndata: {json.dumps(dash, ensure_ascii=False, default=str)}\n\n"
            # Agent 工具调用轨迹
            if tool_trace_data:
                yield f"event: tool_trace\ndata: {json.dumps(tool_trace_data, ensure_ascii=False, default=str)}\n\n"
            # Agent debug 详情（含 stats + tools_contributed）
            if agent_debug_payload:
                yield f"event: agent_debug\ndata: {json.dumps(agent_debug_payload, ensure_ascii=False, default=str)}\n\n"
            for chunk in _chunk_text(response_text):
                yield f"event: delta\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        finally:
            if _obs_metrics:
                _obs_metrics.active_connections.dec()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

    # Detect topic language to match clarification output language
    _is_zh_topic = bool(re.search(r"[\u4e00-\u9fff]", body.message or ""))

    history_block = ""
    if body.session_id:
        mem = load_session_memory(body.session_id)
        if mem:
            turns = mem.get_context_window(n=8)
            lines = []
            for t in turns:
                role_label = "用户" if t.role == "user" else "助手"
                text = (t.content or "").strip().replace("\n", " ")
                if len(text) > 300:
                    text = text[:300] + "..."
                lines.append(f"{role_label}: {text}")
            history_block = "\n".join(lines)

    language_instruction = (
        "IMPORTANT: The user's topic is in Chinese. ALL output — including questions, "
        "suggested_topic, suggested_outline, and research_brief — MUST be in Chinese."
    ) if _is_zh_topic else (
        "IMPORTANT: The user's topic is in English. ALL output — including questions, "
        "suggested_topic, suggested_outline, and research_brief — MUST be in English."
    )

    prompt = _pm.render(
        "chat_deep_research_clarify.txt",
        message=body.message,
        history_block=history_block or "(none)",
        language_instruction=language_instruction,
    )

    try:
        resp = client.chat(
            [
                {"role": "system", "content": _pm.render("chat_deep_research_system.txt")},
                {"role": "user", "content": prompt},
            ],
            model=body.model_override or None,

        )
        text = (resp.get("final_text") or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
    except Exception as e:
        used_fallback = True
        fallback_reason = f"clarify_llm_failed: {str(e)[:240]}"
        _fallback_q_text = (
            "请确认本次研究最关键的目标与范围边界" if _is_zh_topic
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

    return ClarifyResponse(
        questions=questions,
        suggested_topic=data.get("suggested_topic", body.message),
        suggested_outline=data.get("suggested_outline", []),
        research_brief=data.get("research_brief"),
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


@router.post("/deep-research/start", response_model=DeepResearchStartResponse)
def start_deep_research_endpoint(
    body: DeepResearchStartRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> DeepResearchStartResponse:
    """Phase 1: run scope + plan, then return editable brief/outline."""
    from src.collaboration.research.agent import start_deep_research

    store = get_session_store()
    session_id = body.session_id or store.create_session(canvas_id=body.canvas_id or "")
    user_id = optional_user_id or body.user_id or ""

    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)
    filters = _build_deep_research_filters(body)
    start_result = start_deep_research(
        topic=body.topic.strip(),
        llm_client=client,
        canvas_id=body.canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=body.search_mode,
        filters=filters or None,
        max_sections=body.max_sections,
        clarification_answers=body.clarification_answers,
        output_language=body.output_language or "auto",
        model_override=body.model_override or None,
        step_models=body.step_models,
        step_model_strict=bool(body.step_model_strict),
    )
    if start_result.get("canvas_id"):
        store.update_session_meta(session_id, {"canvas_id": start_result.get("canvas_id", "")})
    return DeepResearchStartResponse(
        session_id=session_id,
        canvas_id=start_result.get("canvas_id", "") or "",
        brief=start_result.get("brief") or {},
        outline=start_result.get("outline") or [],
        initial_stats=start_result.get("initial_stats") or {},
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
    from src.collaboration.research.job_store import append_event, update_job, get_pending_review, load_checkpoint
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
        update_job(job_id, status="running", message="Deep Research 任务已启动")
        append_event(job_id, "start", {"job_id": job_id, "topic": body.topic, "session_id": session_id})
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
    """提交 Deep Research 后台任务（默认推荐前端使用此接口）。"""
    from src.collaboration.research.job_store import create_job

    store = get_session_store()
    session_id = body.session_id or store.create_session(canvas_id=body.canvas_id or "")
    payload = body.model_dump()
    payload["session_id"] = session_id
    payload["_worker_user_id"] = optional_user_id
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
    """Extract status fields from a job dict for SSE heartbeat / job_status events."""
    dashboard = job.get("result_dashboard")
    if isinstance(dashboard, str):
        try:
            dashboard = json.loads(dashboard)
        except Exception:
            dashboard = {}
    return {
        "job_id": job.get("job_id", ""),
        "topic": job.get("topic", ""),
        "status": job.get("status", ""),
        "current_stage": job.get("current_stage", ""),
        "message": job.get("message", ""),
        "canvas_id": job.get("canvas_id", ""),
        "result_dashboard": dashboard or {},
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
            if idle_ticks % 5 == 0:
                yield _dr_sse("heartbeat", _dr_job_status_payload(job))
            _time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/deep-research/jobs/{job_id}/cancel")
def cancel_deep_research_job(job_id: str) -> dict:
    from src.collaboration.research.job_store import get_job, update_job, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status in {"done", "error", "cancelled"}:
        return {"ok": True, "job_id": job_id, "status": current_status}
    if current_status == "pending":
        import time as _t
        update_job(job_id, status="cancelled", message="任务已取消（未启动）", finished_at=_t.time())
        append_event(job_id, "cancelled", {"job_id": job_id, "message": "任务已取消（未启动）"})
        return {"ok": True, "job_id": job_id, "status": "cancelled"}
    _dr_request_cancel(job_id)
    update_job(job_id, status="cancelling", message="收到停止请求，正在终止任务...")
    append_event(job_id, "cancel_requested", {"job_id": job_id, "message": "已请求停止"})
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


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
