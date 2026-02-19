"""
对话 API：POST /chat, POST /chat/stream, GET /sessions/{id}, DELETE /sessions/{id}
"""

import json
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from config.settings import settings
from src.api.routes_auth import get_optional_user_id

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
from src.llm.tools import CORE_TOOLS, get_routed_skills
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
_DEEP_RESEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_DEEP_RESEARCH_CANCEL_EVENTS: dict[str, threading.Event] = {}
_DEEP_RESEARCH_CANCEL_LOCK = threading.Lock()
_DEEP_RESEARCH_SUSPENDED: dict[str, dict[str, Any]] = {}
_DEEP_RESEARCH_SUSPENDED_LOCK = threading.Lock()


def _dr_register_cancel_event(job_id: str) -> threading.Event:
    with _DEEP_RESEARCH_CANCEL_LOCK:
        ev = _DEEP_RESEARCH_CANCEL_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _DEEP_RESEARCH_CANCEL_EVENTS[job_id] = ev
        return ev


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
_ROUTE_SYSTEM = "你是一个查询路由分类器。只回答一个词：rag 或 chat。"

_ROUTE_PROMPT = """判断用户消息是否需要检索外部知识库。

## chat（不检索，LLM 直接回答）：
- 社交寒暄：问候、感谢、告别、确认（你好、谢谢、明白了）
- 情绪闲聊：哈哈、嗯嗯
- 用户明确要求用 LLM 自身知识："基于你的训练/知识""你觉得""不用查资料""用你自己的话"
- 纯编程/数学/逻辑推理（不需要领域知识库）
- 对上一轮回答的追问，且上下文已包含足够信息（如"详细说说第三点"）

## rag（需要检索知识库）：
- 涉及特定领域的事实、数据、数值
- 要求基于文献/材料做分析、对比、验证
- 涉及学术概念或专业术语
- 不确定时一律 rag

## 最近对话
{history}

## 当前消息
{message}

只回答 rag 或 chat"""

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

    prompt = _ROUTE_PROMPT.format(history=history_block, message=msg)

    try:
        resp = llm_client.chat(
            [
                {"role": "system", "content": _ROUTE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,  # thinking 模型的 reasoning 也消耗 tokens，需留足空间
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
    return (
        "你是一个基于检索增强的学术助手。请基于以下检索到的参考资料回答用户问题，"
        "保持多轮对话连贯。每条证据前有方括号引用标记（如 [a1b2c3d4]），"
        "请在行文中使用该标记引用对应证据。若无直接相关，可基于常识简要回答并说明。\n\n"
        "参考资料：\n"
        f"{context or '（本轮暂无检索结果）'}"
    )


def _chunk_text(text: str, chunk_size: int = 20):
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _serialize_citation(c: Citation | str) -> dict:
    """将 Citation 对象或字符串序列化为字典。"""
    if isinstance(c, str):
        return {"cite_key": c, "title": "", "authors": [], "year": None, "doc_id": None, "url": None}
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
    }


def _build_filters(body: ChatRequest) -> dict:
    """从 ChatRequest 中提取检索参数构建 filters 字典。"""
    filters = {}
    if body.web_providers:
        filters["web_providers"] = body.web_providers
    if body.web_source_configs:
        filters["web_source_configs"] = body.web_source_configs
    if body.use_query_expansion is not None:
        filters["use_query_expansion"] = body.use_query_expansion
    if body.use_query_optimizer is not None:
        filters["use_query_optimizer"] = body.use_query_optimizer
    if body.query_optimizer_max_queries is not None:
        filters["query_optimizer_max_queries"] = body.query_optimizer_max_queries
    if body.local_threshold is not None:
        filters["local_threshold"] = body.local_threshold
    if body.final_top_k is not None:
        filters["final_top_k"] = body.final_top_k
    if body.llm_provider:
        filters["llm_provider"] = body.llm_provider
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
        "use_query_optimizer",
        "query_optimizer_max_queries",
        "local_top_k",
        "local_threshold",
        "final_top_k",
        "llm_provider",
        "model_override",
        "collection",
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


def _run_chat(
    body: ChatRequest,
    optional_user_id: str | None = None,
) -> tuple[str, str, list[Citation], EvidenceSummary, ParsedIntent, dict | None, list | None, dict[str, str]]:
    import time as _time
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
    history_for_intent = memory.get_context_window(n=6)

    # ── 1. 请求概览 ──
    _chat_logger.info(
        "[chat] ▶ 新请求 | session=%s | msg=%r | provider=%s | model=%s | search_mode=%s | collection=%s | agent=%s",
        session_id[:12], message[:60],
        body.llm_provider or "default", body.model_override or "default",
        search_mode, (body.collection or settings.collection.global_), body.use_agent,
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
        parser = IntentParser(client)
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
        svc = AutoCompleteService(llm_client=client, max_sections=6, include_abstract=True)
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
        evidence_summary = EvidenceSummary(
            query=topic,
            total_chunks=len(result.citations),
            sources_used=_infer_sources_from_citations(result.citations),
            retrieval_time_ms=result.total_time_ms,
        )
        dashboard_data = getattr(result, "dashboard", None)
        if dashboard_data is None and hasattr(result, "__dict__"):
            dashboard_data = getattr(result, "dashboard", None)
        elapsed = (_time.perf_counter() - t_start) * 1000
        _chat_logger.info(
            "[chat] ✔ Deep Research 完成 | citations=%d | time=%.0fms",
            len(result.citations) if result.citations else 0, elapsed,
        )
        return session_id, response_text, result.citations, evidence_summary, parsed, dashboard_data, None, {}

    # ── 3. Chat 分支：智能路由（正则 → LLM → 严格解析）──
    t_route = _time.perf_counter()
    query_needs_rag = _classify_query(message, history_for_intent, client)
    do_retrieval = search_mode != "none" and query_needs_rag
    route_ms = (_time.perf_counter() - t_route) * 1000

    _chat_logger.info(
        "[chat] ③ 查询路由 → %s | do_retrieval=%s (search_mode=%s) | 耗时=%.0fms",
        "rag" if query_needs_rag else "chat",
        do_retrieval, search_mode, route_ms,
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
    else:
        query = message
        _chat_logger.info("[chat] ④ Query 构建 → 跳过 (不需要检索)")

    # ── 5. 检索执行 ──
    if do_retrieval:
        t_retrieval = _time.perf_counter()
        target_collection = (body.collection or "").strip() or None
        retrieval = get_retrieval_service(collection=target_collection)
        filters = _build_filters(body)
        pack = retrieval.search(
            query=query or message,
            mode=search_mode,
            filters=filters or None,
            top_k=body.local_top_k,
        )
        synthesizer = EvidenceSynthesizer()
        context_str, synthesis_meta = synthesizer.synthesize(pack)
        synth_dict = synthesis_meta.to_dict()
        retrieval_ms = (_time.perf_counter() - t_retrieval) * 1000
        evidence_summary = EvidenceSummary(
            query=pack.query,
            total_chunks=len(pack.chunks),
            sources_used=pack.sources_used,
            retrieval_time_ms=pack.retrieval_time_ms,
            year_range=synth_dict.get("year_range"),
            source_breakdown=synth_dict.get("source_breakdown"),
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
            "[chat] ⑤ 检索完成 | mode=%s | chunks=%d | sources=%s | 耗时=%.0fms",
            search_mode, len(pack.chunks),
            ",".join(pack.sources_used), retrieval_ms,
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

    # ── 6. System Prompt 组装 ──
    wf = run_workflow(
        current_stage,
        parsed.intent_type,
        topic="",
        context=context_str or "（本轮暂无检索结果）",
    )
    next_stage = wf["next_stage"]
    stage_prompt = wf.get("system_prompt") or ""
    if not query_needs_rag:
        system_content = (
            "你是一个友好的学术助手。当前用户的问题不需要检索外部知识库。\n"
            "请基于你自身的训练知识回答，保持专业、准确、简洁。\n"
            "如果问题超出你的知识范围，坦诚说明并建议用户让你检索知识库获取更准确的信息。"
        )
        prompt_mode = "chat_direct"
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

    store.update_session_stage(session_id, next_stage)

    # ── 7. 构建消息列表 ──
    history = memory.get_context_window(n=10)
    messages = [{"role": "system", "content": system_content}]
    for t in history:
        messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": message})

    # ── 8. LLM 生成 ──
    use_agent = body.use_agent if hasattr(body, "use_agent") and body.use_agent is not None else False
    use_agent = use_agent and query_needs_rag  # 闲聊时关闭 Agent
    tool_trace = None

    # Agent + 预检索结果：指示优先使用已有资料
    if use_agent and do_retrieval and context_str:
        agent_hint = (
            "\n\n【重要：检索结果已就绪】\n"
            "我已为你检索了相关参考资料（见上文）。请优先基于这些资料回答用户问题。\n"
            "只有当现有资料明显不足以回答时，才使用 search_local 或 search_web 工具补充检索。\n"
            "避免重复搜索已有资料中已覆盖的内容。"
        )
        messages[0]["content"] = messages[0]["content"] + agent_hint

    gen_mode = "agent" if use_agent else "direct"
    _chat_logger.info(
        "[chat] ⑦ LLM 生成 | mode=%s | provider=%s | model=%s | history_turns=%d | msg_count=%d",
        gen_mode, body.llm_provider or "default", body.model_override or "default",
        len(history), len(messages),
    )

    t_llm = _time.perf_counter()
    try:
        if use_agent:
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
                max_tokens=2000,
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
            resp = client.chat(messages, model=body.model_override or None, max_tokens=2000)
            response_text = (resp.get("final_text") or "").strip()
            llm_ms = (_time.perf_counter() - t_llm) * 1000
            usage = resp.get("meta", {}).get("usage") or resp.get("usage") or {}
            _chat_logger.info(
                "[chat] ⑧ LLM 完成 | response_len=%d | tokens=%s | 耗时=%.0fms",
                len(response_text),
                f"in={usage.get('prompt_tokens', '?')}/out={usage.get('completion_tokens', '?')}" if usage else "N/A",
                llm_ms,
            )
    except Exception as llm_err:
        llm_ms = (_time.perf_counter() - t_llm) * 1000
        _chat_logger.error("[chat] ⑧ LLM 失败 | error=%s | 耗时=%.0fms", llm_err, llm_ms)
        response_text = f"[LLM 调用失败] {type(llm_err).__name__}: {llm_err}\n\n请尝试切换其他模型。"

    # ── 9. 引文后处理：将 [ref_hash] 替换为正式 cite_key ──
    ref_map: dict[str, str] = {}
    if do_retrieval and pack and pack.chunks:
        response_text, citations, ref_map = resolve_response_citations(
            response_text, pack.chunks,
        )
        _chat_logger.info(
            "[chat] ⑨ 引文后处理 | cited_docs=%d | ref_map_size=%d",
            len(citations), len(ref_map),
        )

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

    return session_id, response_text, citations, evidence_summary, parsed, None, tool_trace, ref_map


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
    )


@router.post("/chat", response_model=ChatResponse)
def chat_post(
    body: ChatRequest,
    optional_user_id: str | None = Depends(get_optional_user_id),
) -> ChatResponse:
    session_id, response_text, citations, evidence_summary, _parsed, _dashboard, _trace, _ref_map = _run_chat(body, optional_user_id)
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
    session_id, response_text, citations, evidence_summary, parsed, dashboard_data, tool_trace_data, ref_map = _run_chat(body, optional_user_id)
    
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
    """
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client()
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

    # 获取 chat 历史上下文
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

    prompt = f"""用户想要进行深度研究。请先生成澄清问题：
- 至少 1 个问题（即使主题较明确，也要有 1 个关键确认问题）；
- 如果主题不明确或歧义较大，生成更多问题（最多 6 个）；
- 问题按优先级排序。
然后输出一个结构化研究简报。

用户的主题描述: "{body.message}"

对话历史上下文:
{history_block or "（无）"}

请生成问题帮助明确（优先级从高到低）：
1. 研究主题的精确范围
2. 重点关注的方向（如理论/应用/方法论）
3. 目标受众和写作风格（学术论文/科普/报告）
4. 篇幅和深度要求
5. 特别需要关注的子主题或排除的内容
6. 文献语言偏好

约束：
- questions 数量必须在 1 到 6 之间
- 问题应尽量具体、可回答，避免重复

返回 JSON 格式:
{{
  "suggested_topic": "建议的综述主题（一句话）",
  "suggested_outline": ["章节1", "章节2", ...],
  "research_brief": {{
    "scope": "研究范围描述",
    "success_criteria": ["完成标准1", "完成标准2"],
    "key_questions": ["核心问题1", "核心问题2"],
    "exclusions": ["排除内容"],
    "time_range": "文献时间范围",
    "source_priority": ["peer-reviewed papers", "reviews"]
  }},
  "questions": [
    {{"id": "q1", "text": "问题文本", "type": "text", "default": "建议答案"}},
    {{"id": "q2", "text": "问题文本", "type": "choice", "options": ["选项A", "选项B"], "default": "选项A"}}
  ]
}}

只返回 JSON，不要其他文字。"""

    try:
        resp = client.chat(
            [
                {"role": "system", "content": "你是一个学术研究规划助手，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=body.model_override or None,
            max_tokens=1024,
        )
        text = (resp.get("final_text") or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
    except Exception as e:
        # LLM 失败时至少返回 1 个关键问题
        used_fallback = True
        fallback_reason = f"clarify_llm_failed: {str(e)[:240]}"
        data = {
            "suggested_topic": body.message,
            "suggested_outline": [],
            "questions": [
                {"id": "q1", "text": "请确认本次研究最关键的目标与范围边界", "type": "text", "default": body.message},
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
) -> None:
    """后台执行 Deep Research 并持久化状态。"""
    from src.collaboration.research.agent import (
        prepare_deep_research_runtime,
        build_deep_research_result_from_state,
    )
    from src.collaboration.research.job_store import append_event, update_job, get_pending_review
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

        runtime = prepare_deep_research_runtime(
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
            cancel_check=lambda: _dr_is_cancel_requested(job_id),
            review_waiter=lambda section_id: get_pending_review(job_id, section_id),
            skip_draft_review=bool(body.skip_draft_review),
            skip_refine_review=bool(body.skip_refine_review),
            skip_claim_generation=bool(body.skip_claim_generation),
            job_id=job_id,
            depth=body.depth or "comprehensive",
        )
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
    job = create_job(
        topic=body.topic.strip(),
        session_id=session_id,
        canvas_id=body.canvas_id or "",
        request_payload=payload,
    )
    job_id = str(job.get("job_id") or "")
    _dr_register_cancel_event(job_id)
    _DEEP_RESEARCH_EXECUTOR.submit(
        _run_deep_research_job_safe,
        job_id=job_id,
        body=body,
        optional_user_id=optional_user_id,
    )
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
    evidence_summary = EvidenceSummary(
        query=body.topic,
        total_chunks=len(citations),
        sources_used=_infer_sources_from_citations(citations),
        retrieval_time_ms=float(result.get("total_time_ms", 0.0)),
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


@router.post("/deep-research/jobs/{job_id}/cancel")
def cancel_deep_research_job(job_id: str) -> dict:
    from src.collaboration.research.job_store import get_job, update_job, append_event

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    current_status = str(job.get("status") or "")
    if current_status in {"done", "error", "cancelled"}:
        return {"ok": True, "job_id": job_id, "status": current_status}
    _dr_request_cancel(job_id)
    update_job(job_id, status="cancelling", message="收到停止请求，正在终止任务...")
    append_event(job_id, "cancel_requested", {"job_id": job_id, "message": "已请求停止"})
    return {"ok": True, "job_id": job_id, "status": "cancelling"}


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
                    _DEEP_RESEARCH_EXECUTOR.submit(_resume_suspended_job, job_id)
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
    """获取所有会话列表"""
    store = get_session_store()
    sessions = store.list_all_sessions(limit=limit)
    return [
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
