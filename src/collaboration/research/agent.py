"""
递归研究 Agent — 基于 LangGraph 的 Deep Research 引擎。

替代原有线性 auto_complete 流水线，实现：
- Scoping → Plan → 递归研究循环 → 写作 → 验证 → 综合
- RE-TRAC 轨迹压缩 + ReCAP 仪表盘
- 动态分支探索 + 信息充分度评估
"""

from __future__ import annotations

import json
import re
import time
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.collaboration.research.dashboard import (
    ResearchBrief,
    ResearchDashboard,
    SectionStatus,
)
from src.collaboration.research.trajectory import (
    ResearchBranch,
    ResearchTrajectory,
    SearchAction,
    compress_trajectory,
)
from src.log import get_logger
from src.utils.prompt_manager import PromptManager

logger = get_logger(__name__)
_pm = PromptManager()
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "rag_config.json"


class _ScopingResponse(BaseModel):
    scope: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    key_questions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    time_range: str = ""
    source_priority: List[str] = Field(default_factory=list)


class _CoverageEvalResponse(BaseModel):
    coverage_score: float = 0.0
    gaps: List[str] = Field(default_factory=list)
    sufficient: bool = False


# ────────────────────────────────────────────────
# Depth Presets — "lite" vs "comprehensive"
# ────────────────────────────────────────────────
# Each preset defines the bounds for all loops in the graph, preventing the
# recursion-limit explosion that occurs when any loop runs unbounded.
#
#   lite          — fast but academically usable, ~5-15 min
#   comprehensive — thorough academic review, ~20-60 min
#
# Key thresholds (see architecture.md for full table):
#
# Iteration & rounds:
#   max_iterations_per_section  per-section iteration budget (× num_sections = global cap)
#   max_section_research_rounds per-section research→evaluate loop cap
#
# Coverage:
#   coverage_threshold          minimum coverage_score before moving to write
#
# Queries — split into recall (broad, synonym) + precision (specific, constrained):
#   recall_queries_per_section  broad synonym/variant queries per round
#   precision_queries_per_section  narrow method/time/object-constrained queries per round
#   (total queries = recall + precision + gap queries)
#
# Tiered search_top_k (chunks per query, varies by stage):
#   search_top_k_first          first-round broad sweep
#   search_top_k_gap            gap-fill / re-research rounds
#   search_top_k_write          write-node final evidence retrieval
#   search_top_k_write_max      hard cap for adaptive write retrieval window
#   verification_k              write-stage secondary evidence check for data-point citations
#
# Self-correction:
#   self_correction_trigger_coverage  if coverage is already high after early rounds, shrink gap retrieval
#   self_correction_min_round         round index to enable self-correction
#   search_top_k_gap_decay_factor     decay factor for search_top_k_gap
#   search_top_k_gap_min              lower bound after decay
#
# Early-stop by gain curve:
#   coverage_plateau_floor       coverage floor to enable plateau check
#   coverage_plateau_min_gain    minimal acceptable gain between rounds
#
# Verification — 3-tier (light / medium / severe):
#   verify_light_threshold      below this → just flag, no action
#   verify_medium_threshold     between light and severe → gap-fill only (no full re-research)
#   verify_severe_threshold     above this → full re-research with expanded queries
#
# Review gate:
#   review_gate_max_rounds      max polling rounds (with exponential backoff)
#   review_gate_base_sleep      initial sleep seconds (doubles each round, capped)
#   review_gate_max_sleep        max sleep seconds per round
#   review_gate_early_stop_unchanged  auto-stop after N consecutive unchanged polls
#
# LangGraph:
#   recursion_limit             compile-time recursion limit
#
# Cost monitor:
#   cost_warn_steps             warn for manual intervention when graph steps reach threshold
#   cost_force_summary_steps    force summarize mode when graph steps are extremely high
#   cost_tick_interval          emit progress heartbeat every N graph steps
# ────────────────────────────────────────────────

DEPTH_PRESETS: Dict[str, Dict[str, Any]] = {
    "lite": {
        # ── Iteration budget ──
        "max_iterations_per_section": 3,         # global = 3 × num_sections
        "max_section_research_rounds": 3,
        # ── Coverage ──
        "coverage_threshold": 0.60,
        # ── Queries (recall + precision) ──
        "recall_queries_per_section": 2,
        "precision_queries_per_section": 2,      # total = 4 + gaps
        # ── Tiered search_top_k ──
        "search_top_k_first": 18,                # broad first-round sweep
        "search_top_k_gap": 10,                  # targeted gap-fill
        "search_top_k_write": 10,                # precise evidence for writing
        "search_top_k_write_max": 40,            # hard cap for adaptive write retrieval
        "verification_k": 12,                    # secondary check context for data-point claims
        # ── Self-correction / adaptive cost control ──
        "self_correction_trigger_coverage": 0.75,
        "self_correction_min_round": 3,
        "search_top_k_gap_decay_factor": 0.60,
        "search_top_k_gap_min": 6,
        "coverage_plateau_floor": 0.70,
        "coverage_plateau_min_gain": 0.03,
        # ── Verification (3-tier) ──
        "verify_light_threshold": 0.20,          # < 20% → flag only
        "verify_medium_threshold": 0.40,         # 20-40% → gap-fill, no full re-research
        "verify_severe_threshold": 0.45,         # > 45% → full re-research
        # ── Review gate ──
        "review_gate_max_rounds": 80,            # ~5 min with backoff
        "review_gate_base_sleep": 2,
        "review_gate_max_sleep": 15,
        "review_gate_early_stop_unchanged": 8,   # stop after 8 unchanged polls
        # ── LangGraph ──
        "recursion_limit": 200,
        # ── Cost monitor (graph steps) ──
        "cost_warn_steps": 120,
        "cost_force_summary_steps": 180,
        "cost_tick_interval": 25,
    },
    "comprehensive": {
        # ── Iteration budget ──
        "max_iterations_per_section": 6,         # global = 6 × num_sections
        "max_section_research_rounds": 5,        # allows a 5th round for near-complete gaps
        # ── Coverage ──
        "coverage_threshold": 0.80,
        # ── Queries (recall + precision) ──
        "recall_queries_per_section": 4,
        "precision_queries_per_section": 4,      # total = 8 + gaps
        # ── Tiered search_top_k ──
        "search_top_k_first": 30,                # wide net
        "search_top_k_gap": 15,                  # focused supplement
        "search_top_k_write": 12,                # best-citation retrieval
        "search_top_k_write_max": 60,            # hard cap for adaptive write retrieval
        "verification_k": 16,                    # stronger secondary check context
        # ── Self-correction / adaptive cost control ──
        "self_correction_trigger_coverage": 0.78,
        "self_correction_min_round": 3,
        "search_top_k_gap_decay_factor": 0.70,
        "search_top_k_gap_min": 8,
        "coverage_plateau_floor": 0.78,
        "coverage_plateau_min_gain": 0.02,
        # ── Verification (3-tier) ──
        "verify_light_threshold": 0.15,          # < 15% → flag only
        "verify_medium_threshold": 0.30,         # 15-30% → gap-fill only
        "verify_severe_threshold": 0.35,         # > 35% → full re-research
        # ── Review gate ──
        "review_gate_max_rounds": 200,           # ~10 min with backoff
        "review_gate_base_sleep": 2,
        "review_gate_max_sleep": 20,
        "review_gate_early_stop_unchanged": 12,
        # ── LangGraph ──
        "recursion_limit": 500,
        # ── Cost monitor (graph steps) ──
        "cost_warn_steps": 300,
        "cost_force_summary_steps": 420,
        "cost_tick_interval": 30,
    },
}

DEFAULT_DEPTH = "comprehensive"


def get_depth_preset(depth: str) -> Dict[str, Any]:
    """Return depth preset dict; falls back to comprehensive for unknown values."""
    return dict(DEPTH_PRESETS.get(depth, DEPTH_PRESETS[DEFAULT_DEPTH]))


# ────────────────────────────────────────────────
# State 定义
# ────────────────────────────────────────────────

class DeepResearchState(TypedDict, total=False):
    """LangGraph Agent 的状态"""
    topic: str
    dashboard: ResearchDashboard
    trajectory: ResearchTrajectory
    canvas_id: str
    session_id: str
    user_id: str
    search_mode: str
    filters: Dict[str, Any]
    current_section: str
    sections_completed: List[str]
    markdown_parts: List[str]
    citations: List[Any]
    evidence_chunks: List[Any]  # 运行期累计的 EvidenceChunk（用于 hash->cite_key 后处理）
    evidence_chunk_empty_value: Any  # state 紧凑化时用于覆盖 text/raw_content 的值（默认 ""）
    citation_doc_key_map: Dict[str, str]  # doc_group_key -> cite_key（跨阶段保持稳定）
    citation_existing_keys: List[str]  # 已分配 cite_key（用于 numeric/hash/author_date 去重）
    iteration_count: int
    max_iterations: int
    llm_client: Any
    model_override: Optional[str]
    output_language: str
    clarification_answers: Dict[str, str]
    user_context: str
    user_context_mode: str
    user_documents: List[Dict[str, str]]
    step_models: Dict[str, Optional[str]]
    step_model_strict: bool
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]]
    cancel_check: Optional[Callable[[], bool]]
    review_waiter: Optional[Callable[[str], Optional[Dict[str, Any]]]]
    skip_draft_review: bool
    skip_refine_review: bool
    skip_claim_generation: bool
    verified_claims: str  # generated claims text, empty if skipped
    review_gate_next: str
    review_handled_at: Dict[str, float]
    # ── Depth preset (controls all loop bounds) ──
    depth: str                        # "lite" | "comprehensive"
    depth_preset: Dict[str, Any]      # resolved preset values
    review_gate_rounds: int           # counter for review_gate self-loop
    review_gate_unchanged: int        # consecutive unchanged poll counter (for early-stop)
    review_gate_last_snapshot: str    # last poll snapshot hash (for change detection)
    # ── Local Priority Revise ──
    revision_queue: List[str]         # sections queued for priority rework
    review_seen_at: Dict[str, float]  # last-seen review timestamp per section
    # ── Job reference (for insights / supplements persistence) ──
    job_id: str
    # ── Cost / stopping monitor ──
    graph_step_count: int
    cost_warned: bool
    force_synthesize: bool
    coverage_history: Dict[str, List[float]]
    last_cost_tick_step: int
    error: Optional[str]


def _emit_progress(state: DeepResearchState, event_type: str, payload: Dict[str, Any]) -> None:
    """Emit optional progress callbacks for SSE integration."""
    cb = state.get("progress_callback")
    if not cb:
        return
    try:
        cb(event_type, payload)
    except Exception:
        logger.debug("Progress callback failed", exc_info=True)


def _ensure_not_cancelled(state: DeepResearchState) -> None:
    """Cooperative cancellation checkpoint."""
    checker = state.get("cancel_check")
    if checker and checker():
        raise RuntimeError("Deep Research cancelled by user")


def _tick_cost_monitor(state: DeepResearchState, node_name: str) -> None:
    """Track graph step count and emit cost/recursion safety events."""
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    steps = int(state.get("graph_step_count", 0)) + 1
    state["graph_step_count"] = steps

    warn_steps = int(preset.get("cost_warn_steps", 300))
    force_steps = int(preset.get("cost_force_summary_steps", max(360, warn_steps + 60)))
    tick_interval = max(1, int(preset.get("cost_tick_interval", 25)))
    last_tick = int(state.get("last_cost_tick_step", 0))

    if steps - last_tick >= tick_interval:
        state["last_cost_tick_step"] = steps
        _emit_progress(
            state,
            "cost_monitor_tick",
            {
                "node": node_name,
                "steps": steps,
                "warn_steps": warn_steps,
                "force_steps": force_steps,
            },
        )

    if steps >= warn_steps and not bool(state.get("cost_warned", False)):
        state["cost_warned"] = True
        _emit_progress(
            state,
            "cost_monitor_warn",
            {
                "node": node_name,
                "steps": steps,
                "warn_steps": warn_steps,
                "message": "Graph step cost is high. Consider human intervention or narrowing scope.",
            },
        )

    if steps >= force_steps and not bool(state.get("force_synthesize", False)):
        state["force_synthesize"] = True
        # Clamp further exploration budget to avoid runaway loops.
        state["max_iterations"] = min(
            int(state.get("max_iterations", 30)),
            int(state.get("iteration_count", 0)),
        )
        _emit_progress(
            state,
            "cost_monitor_force_summary",
            {
                "node": node_name,
                "steps": steps,
                "force_steps": force_steps,
                "message": "Forced summary mode activated due to high graph step cost.",
            },
        )


def _get_retrieval_svc(state: DeepResearchState):
    """Extract collection from filters and get the correct RetrievalService."""
    from src.retrieval.service import get_retrieval_service

    collection = ((state.get("filters") or {}).get("collection") or "").strip() or None
    return get_retrieval_service(collection=collection)


def _parse_step_model_value(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not value:
        return None, None
    raw = value.strip()
    if not raw:
        return None, None
    if "::" in raw:
        provider, model = raw.split("::", 1)
        return provider.strip() or None, model.strip() or None
    return None, raw


def _resolve_step_client_and_model(state: DeepResearchState, step: str) -> Tuple[Any, Optional[str]]:
    """Resolve provider/model override for a specific step."""
    default_client = state["llm_client"]
    default_model = state.get("model_override")
    step_models = state.get("step_models") or {}
    strict = bool(state.get("step_model_strict", False))
    requested = (step_models.get(step) or "").strip()
    provider, model = _parse_step_model_value(step_models.get(step))
    if not provider and not model:
        return default_client, default_model
    if not provider:
        _emit_progress(
            state,
            "step_model_resolved",
            {
                "step": step,
                "requested": requested,
                "actual_provider": "default",
                "actual_model": model or default_model or "",
                "strict": strict,
            },
        )
        return default_client, model
    try:
        from src.llm.llm_manager import get_manager

        manager = get_manager(str(_CONFIG_PATH))
        client = manager.get_client(provider)
        _emit_progress(
            state,
            "step_model_resolved",
            {
                "step": step,
                "requested": requested,
                "actual_provider": provider,
                "actual_model": model or "",
                "strict": strict,
            },
        )
        return client, model
    except Exception as e:
        logger.warning("Failed to switch step model provider '%s': %s", provider, e)
        _emit_progress(
            state,
            "step_model_fallback",
            {
                "step": step,
                "requested": requested,
                "fallback_provider": "default",
                "fallback_model": model or default_model or "",
                "strict": strict,
                "message": f"Step '{step}' model fallback to default due to provider resolve failure.",
                "error": str(e)[:240],
            },
        )
        if strict:
            raise RuntimeError(f"Step '{step}' model provider '{provider}' resolve failed: {e}") from e
        return default_client, (model or default_model)


def _language_instruction(state: DeepResearchState) -> str:
    lang = (state.get("output_language") or "auto").strip().lower()
    if lang == "zh":
        return "\n\nIMPORTANT: Write the output in Chinese (中文)."
    if lang == "en":
        return "\n\nIMPORTANT: Write the output in English."
    return ""


def _build_user_context_block(state: DeepResearchState, max_chars: int = 2500) -> str:
    """Build optional user-supplied temporary context block."""
    chunks: List[str] = []
    user_context = (state.get("user_context") or "").strip()
    mode = (state.get("user_context_mode") or "supporting").strip().lower()
    if user_context:
        if mode == "direct_injection":
            chunks.append(
                "HIGH PRIORITY USER ASSERTIONS (treat as strong constraints/hypotheses and explicitly verify):\n"
                + user_context
            )
        else:
            chunks.append("User Notes:\n" + user_context)
    docs = state.get("user_documents") or []
    if docs:
        rendered: List[str] = []
        for d in docs[:6]:
            name = str((d or {}).get("name") or "document")
            content = str((d or {}).get("content") or "").strip()
            if not content:
                continue
            rendered.append(f"[{name}]\n{content[:1200]}")
        if rendered:
            chunks.append("User Uploaded Temporary Documents:\n" + "\n\n".join(rendered))
    if not chunks:
        return ""
    text = "\n\n".join(chunks).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return "\n\nAdditional temporary context:\n" + text


def _compute_effective_write_k(preset: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> int:
    """Compute adaptive write retrieval window with a safety cap."""
    preset_write_k = int(preset.get("search_top_k_write", 12))
    write_k_cap = int(preset.get("search_top_k_write_max", 60))
    if write_k_cap <= 0:
        write_k_cap = 60
    # Keep cap sane: never below the preset floor.
    write_k_cap = max(write_k_cap, preset_write_k)

    ui_top_k = 0
    ui_top_k_raw = (filters or {}).get("final_top_k")
    try:
        ui_top_k = int(ui_top_k_raw or 0)
    except (TypeError, ValueError):
        ui_top_k = 0

    effective_write_k = max(preset_write_k, int(ui_top_k * 1.5)) if ui_top_k > 0 else preset_write_k
    return min(effective_write_k, write_k_cap)


def _tokenize_for_overlap(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", (text or "").lower())
    return [t for t in raw if len(t) >= 2]


def _build_temp_chunks(state: DeepResearchState) -> List[Dict[str, str]]:
    docs = state.get("user_documents") or []
    out: List[Dict[str, str]] = []
    for d in docs[:10]:
        name = str((d or {}).get("name") or "temp")
        content = str((d or {}).get("content") or "").strip()
        if not content:
            continue
        blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
        if not blocks:
            blocks = [content]
        for bi, b in enumerate(blocks[:40]):
            chunk = b[:900].strip()
            if len(chunk) < 30:
                continue
            out.append({"name": name, "chunk_id": f"temp::{name}::{bi+1}", "text": chunk})
    return out


def _retrieve_temp_snippets(state: DeepResearchState, query: str, top_k: int = 4) -> List[Dict[str, str]]:
    chunks = _build_temp_chunks(state)
    if not chunks:
        return []
    q_tokens = set(_tokenize_for_overlap(query))
    if not q_tokens:
        return chunks[:top_k]
    scored: List[Tuple[float, Dict[str, str]]] = []
    for c in chunks:
        c_tokens = set(_tokenize_for_overlap(c["text"]))
        overlap = len(q_tokens.intersection(c_tokens))
        if overlap <= 0:
            continue
        score = overlap / max(len(q_tokens), 1)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [x[1] for x in scored[:top_k]]
    return chunks[: min(top_k, len(chunks))]


def _accumulate_evidence_chunks(state: DeepResearchState, chunks: List[Any]) -> None:
    """将检索到的 EvidenceChunk 去重后累积到 state，用于后续统一引用替换。"""
    if not chunks:
        return
    existing = state.setdefault("evidence_chunks", [])
    empty_value = state.get("evidence_chunk_empty_value", "")
    seen_ids = {str(getattr(c, "chunk_id", "")) for c in existing if getattr(c, "chunk_id", None)}
    for c in chunks:
        cid = str(getattr(c, "chunk_id", "") or "")
        if cid and cid in seen_ids:
            continue
        compact_chunk = copy.copy(c)
        if isinstance(compact_chunk, dict):
            if "text" in compact_chunk:
                compact_chunk["text"] = empty_value
            if "raw_content" in compact_chunk:
                compact_chunk["raw_content"] = empty_value
        else:
            if hasattr(compact_chunk, "text"):
                setattr(compact_chunk, "text", empty_value)
            if hasattr(compact_chunk, "raw_content"):
                setattr(compact_chunk, "raw_content", empty_value)
        existing.append(compact_chunk)
        if cid:
            seen_ids.add(cid)


def _resolve_text_citations(
    state: DeepResearchState,
    text: str,
    chunks: List[Any],
    include_unreferenced_documents: bool = False,
) -> tuple[str, List[Any]]:
    """
    使用共享的 doc_key->cite_key 映射做 hash 引文替换，确保跨阶段引用键稳定。
    """
    if not text or not chunks:
        return text, []
    from src.collaboration.citation.manager import resolve_response_citations

    doc_key_map = state.setdefault("citation_doc_key_map", {})
    existing_keys_list = state.setdefault("citation_existing_keys", [])
    existing_keys = set(existing_keys_list)
    resolved_text, citations, _ = resolve_response_citations(
        text,
        chunks,
        doc_key_to_cite_key=doc_key_map,
        existing_cite_keys=existing_keys,
        include_unreferenced_documents=include_unreferenced_documents,
    )
    state["citation_existing_keys"] = sorted(existing_keys)
    return resolved_text, citations


def _generate_section_queries(
    state: DeepResearchState,
    section: SectionStatus,
    max_queries: int = 8,
) -> List[str]:
    """Generate section queries using a recall + precision + gap strategy.

    Three query categories (budget from depth preset):
      1. Gap queries     — one per known gap (highest priority)
      2. Recall queries  — short, broad, synonym/variant phrasing (wide net)
      3. Precision queries — long, constrained with method/time/data types (deep evidence)

    This dual-category approach ensures both broad topic coverage and deep
    evidentiary specificity, which is critical for academic writing.
    """
    dashboard = state["dashboard"]
    topic = dashboard.brief.topic
    client, model_override = _resolve_step_client_and_model(state, "research")
    gaps = section.gaps or []
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))

    recall_budget = int(preset.get("recall_queries_per_section", 2))
    precision_budget = int(preset.get("precision_queries_per_section", 2))

    # ── Priority 1: gap-targeted queries (directly constructed, no LLM call) ──
    gap_queries: List[str] = []
    for gap in gaps[:max(max_queries // 2, 3)]:
        q = f"{topic} {gap}".strip()
        if q and q not in gap_queries:
            gap_queries.append(q)

    # ── Priority 2+3: LLM-generated recall + precision queries ──
    outline_block = "\n".join(f"- {s.title}" for s in dashboard.sections)
    other_sections = [s.title for s in dashboard.sections if s.title != section.title]
    gaps_block = "\n".join(f"- {g}" for g in gaps) if gaps else "(none)"
    avoid_overlap = ", ".join(other_sections[:4]) if other_sections else "(none)"
    temp_snippets = _retrieve_temp_snippets(state, f"{topic} {section.title}", top_k=3)
    temp_block = "\n\n".join(
        f"[{s['name']}] {s['text'][:350]}" for s in temp_snippets
    ) if temp_snippets else "(none)"

    prompt = _pm.render(
        "generate_queries.txt",
        topic=topic,
        scope=dashboard.brief.scope,
        outline_block=outline_block,
        section_title=section.title,
        gaps_block=gaps_block,
        temp_block=temp_block,
        user_context_block=_build_user_context_block(state, max_chars=1200),
        recall_budget=recall_budget,
        precision_budget=precision_budget,
        avoid_overlap=avoid_overlap,
    )

    recall_queries: List[str] = []
    precision_queries: List[str] = []
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "Output search queries ONLY in the specified format."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=500,
        )
        raw = (resp.get("final_text") or "").strip()
        # Parse into recall / precision buckets
        current_bucket: Optional[str] = None
        for line in raw.split("\n"):
            line = line.strip()
            if not line or len(line) <= 3:
                continue
            lower = line.lower().rstrip(":")
            if lower in ("recall", "category a", "recall queries"):
                current_bucket = "recall"
                continue
            if lower in ("precision", "category b", "precision queries"):
                current_bucket = "precision"
                continue
            # Skip header-like lines
            if line.startswith("Category") or line.startswith("##"):
                continue
            # Clean numbering
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(cleaned) <= 3:
                continue
            seen = set(gap_queries + recall_queries + precision_queries)
            if cleaned in seen:
                continue
            if current_bucket == "precision" and len(precision_queries) < precision_budget:
                precision_queries.append(cleaned)
            elif len(recall_queries) < recall_budget:
                recall_queries.append(cleaned)
            elif len(precision_queries) < precision_budget:
                precision_queries.append(cleaned)
    except Exception:
        # Fallback: simple section query
        recall_queries = [f"{topic} {section.title}".strip()]

    # Assemble: gap → recall → precision (priority order)
    result = gap_queries + recall_queries + precision_queries
    if not result:
        return [f"{topic} {section.title}".strip()]
    return result[:max_queries]


# ────────────────────────────────────────────────
# Local Priority Revise: scan for fresh review signals
# ────────────────────────────────────────────────

def _scan_fresh_revise_signals(state: DeepResearchState) -> None:
    """Check review_waiter for newly submitted 'revise' actions and enqueue them.

    This is called at the start of research_node so that mid-run revisions
    are picked up without waiting for the global review_gate.
    """
    waiter = state.get("review_waiter")
    if not waiter or bool(state.get("skip_draft_review", False)):
        return
    dashboard = state.get("dashboard")
    if not dashboard:
        return

    seen = state.setdefault("review_seen_at", {})
    queue = state.setdefault("revision_queue", [])

    for sec in dashboard.sections:
        if sec.title in queue:
            continue  # already queued
        review = waiter(sec.title)
        if not review:
            continue
        action = str(review.get("action") or "").strip().lower()
        if action != "revise":
            continue
        created_at = float(review.get("created_at") or 0.0)
        last_seen = float(seen.get(sec.title) or 0.0)
        if created_at > last_seen:
            queue.append(sec.title)
            seen[sec.title] = created_at
            _emit_progress(state, "revise_queued", {
                "section": sec.title,
                "message": f"章节 \"{sec.title}\" 已加入优先重写队列。",
                "feedback": str(review.get("feedback") or ""),
            })


def _consume_revision_queue(state: DeepResearchState) -> Optional[str]:
    """Pop the next section from the revision queue (if any).

    Returns the section title to rework, or None if queue is empty.
    """
    queue = state.get("revision_queue") or []
    if not queue:
        return None
    target = queue.pop(0)
    state["revision_queue"] = queue
    return target


# ────────────────────────────────────────────────
# Gap Supplements: load unconsumed supplements for a section
# ────────────────────────────────────────────────

def _load_section_supplements(state: DeepResearchState, section_id: str) -> str:
    """Load unconsumed gap supplements for a specific section and return as context block."""
    job_id = state.get("job_id") or ""
    if not job_id:
        return ""
    try:
        from src.collaboration.research.job_store import list_gap_supplements
        supplements = list_gap_supplements(job_id, section_id=section_id, status="pending")
        if not supplements:
            return ""
        parts: List[str] = []
        for sup in supplements:
            content = sup.get("content") or {}
            text = str(content.get("text") or content.get("info") or "").strip()
            gap = str(sup.get("gap_text") or "").strip()
            stype = str(sup.get("supplement_type") or "material")
            if text:
                label = f"[User Gap Supplement ({stype})]"
                if gap:
                    label += f" For gap: {gap}"
                parts.append(f"{label}\n{text[:1500]}")
        if not parts:
            return ""
        return "\n\nSection-scoped user supplements (HIGH PRIORITY -- incorporate this information):\n" + "\n\n".join(parts)
    except Exception:
        return ""


def _mark_section_supplements_consumed(state: DeepResearchState, section_id: str) -> None:
    """Mark all pending supplements for a section as consumed."""
    job_id = state.get("job_id") or ""
    if not job_id:
        return
    try:
        from src.collaboration.research.job_store import list_gap_supplements, mark_gap_supplement_consumed
        supplements = list_gap_supplements(job_id, section_id=section_id, status="pending")
        for sup in supplements:
            sid = sup.get("id")
            if sid:
                mark_gap_supplement_consumed(int(sid))
        if supplements:
            _emit_progress(state, "gap_supplement_consumed", {
                "section": section_id,
                "count": len(supplements),
                "message": f"已采纳 {len(supplements)} 条用户补充材料进入章节重写。",
            })
    except Exception:
        logger.debug("Failed to mark supplements consumed", exc_info=True)


# ────────────────────────────────────────────────
# Research Insights: record to persistent ledger
# ────────────────────────────────────────────────

def _record_insight(
    state: DeepResearchState,
    insight_type: str,
    text: str,
    section_id: str = "",
    source_context: str = "",
) -> None:
    """Append an insight to the persistent Research Insights Ledger."""
    job_id = state.get("job_id") or ""
    if not job_id or not text:
        return
    try:
        from src.collaboration.research.job_store import append_insight
        append_insight(
            job_id=job_id,
            insight_type=insight_type,
            text=text,
            section_id=section_id,
            source_context=source_context,
        )
    except Exception:
        logger.debug("Failed to record insight", exc_info=True)


# ────────────────────────────────────────────────
# Node 函数
# ────────────────────────────────────────────────

def scoping_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 1: 范围界定 — 生成 Research Brief"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "scope")
    client, model_override = _resolve_step_client_and_model(state, "scope")
    topic = state["topic"]
    clarification = state.get("clarification_answers") or {}
    clarification_lines = [f"- {k}: {v}" for k, v in clarification.items() if v]
    clarification_block = "\n".join(clarification_lines) if clarification_lines else "(none)"

    prompt = _pm.render(
        "scope_research.txt",
        topic=topic,
        clarification_block=clarification_block,
    )

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are a research planning expert. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=1000,
            response_model=_ScopingResponse,
        )
        parsed: Optional[_ScopingResponse] = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                parsed = _ScopingResponse.model_validate_json(raw)
        data = parsed.model_dump() if parsed is not None else {}
    except Exception as e:
        logger.warning(f"Scoping failed: {e}")
        data = {}

    brief = ResearchBrief(
        topic=topic,
        scope=data.get("scope", f"Comprehensive review of {topic}"),
        success_criteria=data.get("success_criteria", ["Cover major research directions", "Include recent advances"]),
        key_questions=data.get("key_questions", [topic]),
        exclusions=data.get("exclusions", []),
        time_range=data.get("time_range", ""),
        source_priority=data.get("source_priority", ["peer-reviewed"]),
    )

    dashboard = state.get("dashboard") or ResearchDashboard()
    dashboard.brief = brief
    state["dashboard"] = dashboard

    trajectory = state.get("trajectory") or ResearchTrajectory(topic=topic)
    state["trajectory"] = trajectory
    _emit_progress(state, "scope_done", {"topic": topic, "scope": brief.scope, "key_questions": brief.key_questions})

    return state


def plan_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 2: 规划 — 生成大纲并初始化 Dashboard"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "plan")
    client, model_override = _resolve_step_client_and_model(state, "plan")
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]

    # Initial retrieval for background context
    svc = _get_retrieval_svc(state)
    dr_filters = dict(state.get("filters") or {})
    dr_filters["use_query_optimizer"] = False
    pack = svc.search(
        query=dashboard.brief.topic,
        mode=state.get("search_mode", "hybrid"),
        top_k=15,
        filters=dr_filters,
    )
    _accumulate_evidence_chunks(state, pack.chunks)
    context = pack.to_context_string(max_chunks=15)

    # Track trajectory
    main_branch = trajectory.add_branch("main", "initial background survey")
    main_branch.status = "done"
    trajectory.add_search_action("main", SearchAction(
        query=dashboard.brief.topic,
        tool="search_hybrid",
        result_summary=f"Retrieved {len(pack.chunks)} relevant chunks",
        source_count=len(pack.chunks),
    ))
    dashboard.total_sources += len(pack.chunks)

    # Generate outline
    prompt = _pm.render(
        "plan_outline.txt",
        topic=dashboard.brief.topic,
        scope=dashboard.brief.scope,
        key_questions=", ".join(dashboard.brief.key_questions),
        context=context[:3000],
    )

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an expert at building academic review outlines."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=800,
        )
        raw = resp.get("final_text", "")
    except Exception:
        raw = f"1. Overview of {dashboard.brief.topic}\n2. Research progress\n3. Key findings\n4. Conclusions and outlook"

    # 解析大纲
    sections = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[\d\.\-\*]+\s*(.+)$", line)
        if m:
            sections.append(m.group(1).strip())

    if not sections:
        sections = [f"{dashboard.brief.topic} Review"]

    # 初始化 Dashboard 章节
    dashboard.sections = []
    for title in sections:
        dashboard.add_section(title)
        trajectory.add_branch(f"sec_{len(dashboard.sections)}", title)

    # 创建/获取 Canvas
    from src.collaboration.canvas.canvas_manager import create_canvas, get_canvas, update_canvas, upsert_outline
    from src.collaboration.canvas.models import OutlineSection
    canvas_id = state.get("canvas_id")
    if canvas_id:
        canvas = get_canvas(canvas_id)
        if canvas is None:
            canvas = create_canvas(
                session_id=state.get("session_id", ""),
                topic=dashboard.brief.topic,
                user_id=state.get("user_id", ""),
            )
            state["canvas_id"] = canvas.id
    else:
        canvas = create_canvas(
            session_id=state.get("session_id", ""),
            topic=dashboard.brief.topic,
            user_id=state.get("user_id", ""),
        )
        state["canvas_id"] = canvas.id

    # 将 confirmed outline 实时写入 Canvas（前端可立即看到 Outline）
    outline_sections: List[OutlineSection] = []
    for idx, title in enumerate(sections):
        outline_sections.append(
            OutlineSection(
                title=title,
                level=1,
                order=idx,
                status="todo",
            )
        )
    try:
        upsert_outline(state["canvas_id"], outline_sections)
        update_canvas(
            state["canvas_id"],
            stage="outline",
            skip_draft_review=bool(state.get("skip_draft_review", False)),
            skip_refine_review=bool(state.get("skip_refine_review", False)),
            # 将 brief 结构化保存，便于 Explore 阶段展示
            research_brief={
                "scope": dashboard.brief.scope,
                "success_criteria": dashboard.brief.success_criteria,
                "key_questions": dashboard.brief.key_questions,
                "exclusions": dashboard.brief.exclusions,
                "time_range": dashboard.brief.time_range,
                "source_priority": dashboard.brief.source_priority,
                "action_plan": "",
            },
        )
    except Exception as e:
        logger.warning("Failed to initialize canvas outline/brief: %s", e)

    # 同步引用
    from src.collaboration.citation.manager import sync_evidence_to_canvas
    sync_evidence_to_canvas(state["canvas_id"], pack)

    state["markdown_parts"] = [f"# {dashboard.brief.topic}\n"]
    state["sections_completed"] = []
    _emit_progress(
        state,
        "plan_done",
        {
            "outline": sections,
            "initial_sources": len(pack.chunks),
            "sources_used": pack.sources_used,
            "canvas_id": state.get("canvas_id", ""),
        },
    )

    return state


def research_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 3: 递归研究 — 对当前章节执行搜索 + 信息评估"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "research")
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]
    client = state["llm_client"]

    # ── Local Priority Revise: check for mid-run revise signals ──
    _scan_fresh_revise_signals(state)

    # ── Priority: consume revision queue before picking next section ──
    revise_target = _consume_revision_queue(state)
    if revise_target:
        section = dashboard.get_section(revise_target)
        if section:
            section.status = "researching"
            _emit_progress(state, "revise_started", {
                "section": section.title,
                "message": f"开始优先重写章节：{section.title}",
            })
        else:
            section = dashboard.get_next_section()
    else:
        section = dashboard.get_next_section()
    if section is None:
        return state

    state["current_section"] = section.title
    section.status = "researching"
    section.research_rounds += 1
    dashboard.total_iterations += 1
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    _emit_progress(state, "section_research_start", {"section": section.title, "round": section.research_rounds})

    # Build gaps-first focused queries (budget from depth preset: recall + precision + gap buffer)
    recall_q = int(preset.get("recall_queries_per_section", 2))
    precision_q = int(preset.get("precision_queries_per_section", 2))
    max_q = recall_q + precision_q + 2
    queries = _generate_section_queries(state, section, max_queries=max_q)
    temp_hits = _retrieve_temp_snippets(state, f"{dashboard.brief.topic} {section.title}", top_k=4)
    for hit in temp_hits:
        trajectory.add_finding(
            f"sec_{dashboard.sections.index(section) + 1}",
            f"[temp:{hit['name']}] {hit['text'][:180]}"
        )

    # Execute search — tiered top_k: first round (wide net) vs gap-fill (targeted)
    svc = _get_retrieval_svc(state)
    dr_filters = dict(state.get("filters") or {})
    dr_filters["use_query_optimizer"] = False
    if section.research_rounds <= 1:
        search_top_k = int(preset.get("search_top_k_first", 20))
    else:
        search_top_k = int(preset.get("search_top_k_gap", 12))
        # Self-correction: if early rounds already reached high coverage, shrink gap retrieval.
        trigger_cov = float(preset.get("self_correction_trigger_coverage", 0.75))
        min_round = int(preset.get("self_correction_min_round", 3))
        if section.research_rounds >= min_round and float(section.coverage_score or 0.0) >= trigger_cov:
            decay = float(preset.get("search_top_k_gap_decay_factor", 0.7))
            min_k = int(preset.get("search_top_k_gap_min", 6))
            decayed_k = max(min_k, int(search_top_k * max(0.3, min(decay, 1.0))))
            if decayed_k < search_top_k:
                _emit_progress(
                    state,
                    "search_self_correction",
                    {
                        "section": section.title,
                        "round": section.research_rounds,
                        "coverage": round(float(section.coverage_score or 0.0), 3),
                        "top_k_from": search_top_k,
                        "top_k_to": decayed_k,
                        "message": "High coverage detected; reducing gap retrieval top_k to save cost.",
                    },
                )
                search_top_k = decayed_k
                # Also trim query count in high-coverage rounds.
                queries = queries[: max(2, len(queries) // 2)]
    all_chunks = []
    all_sources: set[str] = set()

    for q in queries:
        _ensure_not_cancelled(state)
        pack = svc.search(
            query=q,
            mode=state.get("search_mode", "hybrid"),
            top_k=search_top_k,
            filters=dr_filters,
        )
        all_chunks.extend(pack.chunks)
        all_sources.update(pack.sources_used)
        section.source_count += len(pack.chunks)
        dashboard.total_sources += len(pack.chunks)

        # Track trajectory
        branch_id = f"sec_{dashboard.sections.index(section) + 1}"
        trajectory.add_search_action(branch_id, SearchAction(
            query=q,
            tool="search_hybrid",
            result_summary=f"Retrieved {len(pack.chunks)} chunks",
            source_count=len(pack.chunks),
        ))

    # Adaptive fallback for sparse evidence:
    # Trigger on (a) too few chunks overall, OR (b) too few independent
    # documents (corroboration principle — single-source evidence is fragile).
    def _count_distinct_docs(chunks):
        keys = set()
        for c in chunks:
            if getattr(c, "doi", None):
                keys.add(f"doi:{c.doi}")
            else:
                keys.add(c.doc_group_key)
        return len(keys)

    distinct_docs = _count_distinct_docs(all_chunks)
    needs_fallback = len(all_chunks) < 3 or distinct_docs < 3

    if needs_fallback:
        fallback_reason = (
            "Sparse evidence detected"
            if len(all_chunks) < 3
            else f"Corroboration risk: only {distinct_docs} independent source(s)"
        )
        _emit_progress(
            state,
            "warning",
            {
                "section": section.title,
                "message": f"{fallback_reason}; running adaptive fallback search.",
                "chunks_found": len(all_chunks),
                "distinct_docs": distinct_docs,
            },
        )
        fallback_query = f"{dashboard.brief.topic} {section.title}".strip()
        fallback_mode = state.get("search_mode", "hybrid")
        if fallback_mode == "local":
            fallback_mode = "hybrid"
        try:
            _ensure_not_cancelled(state)
            fallback_pack = svc.search(
                query=fallback_query,
                mode=fallback_mode,
                top_k=int(preset.get("search_top_k_first", 20)),
                filters=dr_filters,
            )
            all_chunks.extend(fallback_pack.chunks)
            all_sources.update(fallback_pack.sources_used)
            section.source_count += len(fallback_pack.chunks)
            dashboard.total_sources += len(fallback_pack.chunks)
            trajectory.add_search_action(branch_id, SearchAction(
                query=fallback_query,
                tool="search_adaptive_fallback",
                result_summary=f"Fallback retrieved {len(fallback_pack.chunks)} chunks",
                source_count=len(fallback_pack.chunks),
            ))
        except Exception as e:
            logger.warning("Adaptive fallback search failed for section '%s': %s", section.title, e)

    distinct_docs_post = _count_distinct_docs(all_chunks)
    if len(all_chunks) < 3 or distinct_docs_post < 3:
        section.evidence_scarce = True
        _emit_progress(
            state,
            "evidence_insufficient",
            {
                "section": section.title,
                "message": "Evidence remains insufficient after fallback search; section may be degraded in writing.",
                "chunks_found": len(all_chunks),
                "distinct_docs": distinct_docs_post,
                "gaps": section.gaps[:3],
            },
        )
    else:
        section.evidence_scarce = False

    # 累积证据块，供 write/synthesize 统一做 hash->cite_key 替换
    _accumulate_evidence_chunks(state, all_chunks)

    # Sync citations
    if all_chunks and state.get("canvas_id"):
        from src.retrieval.evidence import EvidencePack
        from src.collaboration.citation.manager import sync_evidence_to_canvas
        temp_pack = EvidencePack(
            query=section.title,
            chunks=all_chunks,
            total_candidates=len(all_chunks),
            retrieval_time_ms=0,
            sources_used=sorted(all_sources) or ["hybrid"],
        )
        sync_evidence_to_canvas(state["canvas_id"], temp_pack)

    # RE-TRAC: 检查是否需要压缩
    if trajectory.needs_compression():
        compress_trajectory(trajectory, client, model=state.get("model_override"))

    _emit_progress(
        state,
        "section_research_done",
        {
            "section": section.title,
            "queries": queries,
            "chunks_found": len(all_chunks),
            "sources_used": sorted(all_sources),
        },
    )

    return state


def evaluate_node(state: DeepResearchState) -> DeepResearchState:
    """Evaluate information sufficiency and decide if more search is needed."""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "evaluate")
    dashboard = state["dashboard"]
    client, model_override = _resolve_step_client_and_model(state, "evaluate")
    trajectory = state["trajectory"]

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    # Build current findings context
    branch_id = f"sec_{dashboard.sections.index(section) + 1}"
    branch = trajectory.get_branch(branch_id)
    findings = "\n".join(branch.key_findings[-10:]) if branch else ""

    prompt = _pm.render(
        "evaluate_sufficiency.txt",
        section_title=section.title,
        topic=dashboard.brief.topic,
        source_count=section.source_count,
        findings=findings if findings else "(no key findings yet)",
    )

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an information sufficiency evaluator. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=500,
            response_model=_CoverageEvalResponse,
        )
        parsed: Optional[_CoverageEvalResponse] = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                parsed = _CoverageEvalResponse.model_validate_json(raw)
        data = parsed.model_dump() if parsed is not None else {}
    except Exception:
        data = {
            "coverage_score": 0.3,
            "gaps": ["Evaluation failed; evidence may be insufficient and needs additional search."],
            "sufficient": False,
        }
        _emit_progress(
            state,
            "warning",
            {
                "section": section.title,
                "message": "Coverage evaluation failed; using conservative fallback and continuing evidence search.",
            },
        )

    section.coverage_score = float(data.get("coverage_score", 0.5))
    section.gaps = data.get("gaps", [])
    history = state.setdefault("coverage_history", {})
    sec_hist = history.setdefault(section.title, [])
    sec_hist.append(section.coverage_score)
    history[section.title] = sec_hist[-6:]
    coverage_gain = None
    if len(sec_hist) >= 2:
        coverage_gain = float(sec_hist[-1]) - float(sec_hist[-2])

    if section.gaps:
        dashboard.coverage_gaps.extend(section.gaps)
        # ── Record gaps to Research Insights Ledger ──
        for gap_text in section.gaps:
            _record_insight(
                state,
                insight_type="gap",
                text=gap_text,
                section_id=section.title,
                source_context="evaluate_node",
            )
    if section.coverage_score < 0.4:
        _emit_progress(
            state,
            "warning",
            {
                "section": section.title,
                "coverage": section.coverage_score,
                "message": "Low coverage detected. Consider refining keywords or expanding scope.",
                "gaps": section.gaps[:3],
            },
        )
    _emit_progress(
        state,
        "section_evaluate_done",
        {
            "section": section.title,
            "coverage": section.coverage_score,
            "gaps": section.gaps,
            "research_round": int(section.research_rounds),
            "graph_steps": int(state.get("graph_step_count", 0)),
            "coverage_gain": coverage_gain,
        },
    )

    return state


def generate_claims_node(state: DeepResearchState) -> DeepResearchState:
    """Extract 3-5 core claims from section evidence (with [ref_hash]) before writing."""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "generate_claims")
    dashboard = state["dashboard"]
    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    dr_filters = dict(state.get("filters") or {})
    write_top_k = _compute_effective_write_k(preset, dr_filters)
    svc = _get_retrieval_svc(state)
    dr_filters["use_query_optimizer"] = False
    pack = svc.search(
        query=f"{dashboard.brief.topic} {section.title}",
        mode=state.get("search_mode", "hybrid"),
        top_k=write_top_k,
        filters=dr_filters,
    )
    evidence_str = pack.to_context_string(max_chunks=write_top_k)

    client, model_override = _resolve_step_client_and_model(state, "write")
    user_content = _pm.render(
        "generate_claims.txt",
        section_title=section.title,
        evidence=evidence_str[:4000],
    )
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an expert at extracting concise, citation-backed claims from evidence. Preserve every [ref_hash] in each claim."},
                {"role": "user", "content": user_content},
            ],
            model=model_override,
            max_tokens=800,
        )
        verified_claims = (resp.get("final_text") or "").strip()
    except Exception as e:
        logger.warning("generate_claims_node LLM call failed: %s", e)
        verified_claims = ""

    state["verified_claims"] = verified_claims
    return state


def write_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 4: 写作 — 生成当前章节内容"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "write")
    dashboard = state["dashboard"]
    client, model_override = _resolve_step_client_and_model(state, "write")
    trajectory = state["trajectory"]

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    section.status = "writing"

    # 收集上下文
    context_parts = [dashboard.to_system_prompt()]
    if trajectory.compressed_summaries:
        context_parts.append("\n".join(trajectory.compressed_summaries[-2:]))
    context = "\n\n".join(context_parts)

    # Retrieve section context — adaptive write-stage top_k
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    dr_filters = dict(state.get("filters") or {})
    write_top_k = _compute_effective_write_k(preset, dr_filters)
    verification_k = int(preset.get("verification_k", max(10, write_top_k)))
    svc = _get_retrieval_svc(state)
    dr_filters["use_query_optimizer"] = False
    pack = svc.search(
        query=f"{dashboard.brief.topic} {section.title}",
        mode=state.get("search_mode", "hybrid"),
        top_k=write_top_k,
        filters=dr_filters,
    )
    evidence_str = pack.to_context_string(max_chunks=write_top_k)
    verify_pack = svc.search(
        query=f"{dashboard.brief.topic} {section.title} data evidence citation verification",
        mode=state.get("search_mode", "hybrid"),
        top_k=verification_k,
        filters=dr_filters,
    )
    verification_evidence_str = verify_pack.to_context_string(max_chunks=verification_k)
    _emit_progress(
        state,
        "write_verification_context",
        {
            "section": section.title,
            "write_top_k": write_top_k,
            "verification_k": verification_k,
            "primary_chunks": len(pack.chunks),
            "verification_chunks": len(verify_pack.chunks),
        },
    )
    temp_snippets = _retrieve_temp_snippets(state, f"{dashboard.brief.topic} {section.title}", top_k=5)
    temp_context = "\n\n".join(
        f"[temp:{s['name']}] {s['text'][:500]}" for s in temp_snippets
    )

    # ── Load section-scoped gap supplements (high priority user input) ──
    supplement_block = _load_section_supplements(state, section.title)

    cov_threshold = float(preset.get("coverage_threshold", 0.6))
    low_coverage = section.coverage_score < cov_threshold
    degraded_mode = bool(section.evidence_scarce) and section.source_count < 3

    if degraded_mode:
        # Hard degrade mode: avoid hallucinated long-form text when evidence is clearly insufficient.
        gaps_md = "\n".join(f"- {g}" for g in (section.gaps or [])[:5]) or "- Evidence retrieval was too sparse to support a reliable synthesis."
        section_text = (
            f"Evidence is currently insufficient to provide a fully supported section for **{section.title}**. "
            "This subsection is intentionally downgraded to avoid overconfident claims.\n\n"
            "Key unresolved gaps:\n"
            f"{gaps_md}\n\n"
            "Recommended next step: broaden data sources, refine terminology variants, and add domain-specific primary studies."
        )
        _emit_progress(
            state,
            "section_degraded",
            {
                "section": section.title,
                "message": "Section downgraded due to sparse evidence; generated a constrained summary instead of full prose.",
                "source_count": section.source_count,
                "coverage": section.coverage_score,
            },
        )
    else:
        caution_block = ""
        if low_coverage:
            caution_block = (
                "\nAdditional low-coverage constraints:\n"
                "- Current evidence coverage is LOW; avoid definitive language.\n"
                "- Use cautious wording (e.g., 'evidence suggests', 'limited data indicate').\n"
                "- If support is weak for a claim, mark it as [evidence limited].\n"
                "- Include a short 'Open Gaps' paragraph at the end listing unresolved evidence gaps.\n"
                "- Target length: 200-350 words (not 400-600) under low coverage.\n"
            )
        numeric_markers = (
            "computed_stats",
            "sample_size",
            "mean",
            "median",
            "std",
            "p-value",
            "p value",
            "confidence interval",
            "effect size",
        )
        numeric_context = f"{evidence_str}\n{verification_evidence_str}".lower()
        has_structured_numeric_data = any(m in numeric_context for m in numeric_markers)
        quantitative_block = ""
        if has_structured_numeric_data:
            quantitative_block = (
                "\nQuantitative strictness (MANDATORY):\n"
                "- Structured numeric evidence (e.g., computed_stats/table-like values) is present.\n"
                "- When any numeric comparison/difference is needed, you MUST call the `run_code` tool to run real Python/Pandas calculation.\n"
                "- Do not estimate, round by intuition, or fabricate any number; only report values that come from tool execution.\n"
                "- If calculation is not possible from available data, explicitly state the data limitation instead of guessing.\n"
            )
        claims_block = ""
        if state.get("verified_claims"):
            claims_block = (
                "\nPre-verified claims for this section (you MUST address each claim):\n"
                f"{state['verified_claims']}\n"
                "Expand each claim into well-supported prose. Do not omit any claim.\n"
            )
        triangulation_block = (
            "\nEvidence Triangulation (MANDATORY):\n"
            "- You MUST synthesize information across multiple independent sources for every major claim.\n"
            "- Do NOT rely on a single paper for any major claim.\n"
            "- If a finding is only supported by one source, explicitly qualify it "
            "(e.g., 'A single study [ref_hash] suggests...' or 'Preliminary evidence from [ref_hash] indicates...').\n"
            "- Actively look for converging evidence from different authors/studies to strengthen conclusions.\n"
        )
        prompt = _pm.render(
            "write_section.txt",
            section_title=section.title,
            language_instruction=_language_instruction(state),
            user_context_block=_build_user_context_block(state, max_chars=1800),
            triangulation_block=triangulation_block,
            caution_block=caution_block,
            quantitative_block=quantitative_block,
            claims_block=claims_block,
            evidence=evidence_str[:4000],
            verification_evidence=verification_evidence_str[:3500],
            temp_context=temp_context if temp_context else "(none)",
            supplement_block=supplement_block,
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic review writer."
                        " When numeric comparisons are needed, use tools to compute instead of guessing."
                        " You follow the principle that a single-source claim must never be presented as established fact;"
                        " always triangulate across multiple independent references.\n"
                        f"{context[:2000]}"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            if has_structured_numeric_data:
                from src.llm.react_loop import react_loop
                from src.llm.tools import get_tools_by_names

                react_result = react_loop(
                    messages=messages,
                    tools=get_tools_by_names(["run_code"]),
                    llm_client=client,
                    max_iterations=4,
                    model=model_override,
                    max_tokens=1500,
                )
                section_text = (react_result.final_text or "").strip()
                if not section_text:
                    resp = client.chat(
                        messages=messages,
                        model=model_override,
                        max_tokens=1500,
                    )
                    section_text = (resp.get("final_text") or "").strip()
            else:
                resp = client.chat(
                    messages=messages,
                    model=model_override,
                    max_tokens=1500,
                )
                section_text = (resp.get("final_text") or "").strip()
        except Exception as e:
            section_text = f"(Section '{section.title}' generation failed: {e})"

    # 对章节文本做 hash->cite_key 后处理，并在 state 内保持引用键稳定
    section_chunks = list(pack.chunks) + list(verify_pack.chunks)
    _accumulate_evidence_chunks(state, section_chunks)
    if section_text and section_chunks:
        section_text, section_citations = _resolve_text_citations(
            state,
            section_text,
            section_chunks,
            include_unreferenced_documents=False,
        )
        if section_citations:
            state["citations"] = section_citations

    # 添加到 markdown
    level = "##"
    state.setdefault("markdown_parts", []).append(f"\n{level} {section.title}\n\n{section_text}\n")

    # 实时写回 Canvas Draft（让前端 Drafting 面板在任务进行中可见）
    if state.get("canvas_id"):
        try:
            from src.collaboration.canvas.canvas_manager import get_canvas, upsert_draft, update_canvas
            from src.collaboration.canvas.models import DraftBlock
            canvas = get_canvas(state["canvas_id"])
            section_id = None
            if canvas:
                for s in canvas.outline:
                    if s.title == section.title:
                        section_id = s.id
                        break
            if not section_id:
                # 回退：用章节标题作为 section_id（避免草稿丢失）
                section_id = section.title
            upsert_draft(
                state["canvas_id"],
                DraftBlock(
                    section_id=section_id,
                    content_md=section_text,
                    version=1,
                    used_fragment_ids=[],
                    used_citation_ids=[],
                ),
            )
            update_canvas(state["canvas_id"], stage="drafting")
        except Exception as e:
            logger.warning("Failed to write section draft to canvas: %s", e)

    # ── Mark consumed supplements after successful write ──
    _mark_section_supplements_consumed(state, section.title)

    section.status = "reviewing"  # 写完进入审核
    dashboard.update_confidence()
    _emit_progress(
        state,
        "section_write_done",
        {
            "section": section.title,
            "section_id": section.title,  # 用 title 作为 review id，前后端对齐
            "word_count": len(section_text.split()),
            "coverage": section.coverage_score,
        },
    )

    # 手动介入：每章写完后发出审核事件（不阻塞后续章节）
    if not bool(state.get("skip_draft_review", False)):
        _emit_progress(
            state,
            "waiting_review",
            {
                "section": section.title,
                "section_id": section.title,
                "message": f"等待用户审核章节：{section.title}",
            },
        )

    return state


def verify_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 4b: 验证 — Chain of Verification 验证声明"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "verify")
    dashboard = state["dashboard"]
    client, model_override = _resolve_step_client_and_model(state, "verify")

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    # 找到刚写的章节文本
    section_text = ""
    for part in state.get("markdown_parts", []):
        if section.title in part:
            section_text = part
            break

    if not section_text:
        section.status = "done"
        state.setdefault("sections_completed", []).append(section.title)
        _emit_progress(state, "section_verify_done", {"section": section.title, "status": "done", "unsupported_claims": 0})
        return state

    # 执行 CoV
    from src.collaboration.research.verifier import verify_claims
    citations = state.get("citations", [])
    try:
        result = verify_claims(section_text, citations, client, model=model_override)
        section.coverage_score = max(section.coverage_score, {"high": 0.9, "medium": 0.7, "low": 0.4}.get(result.overall_confidence, 0.5))

        # ── 3-tier verification response ──
        # Instead of a single threshold, we handle three severity levels:
        #   light  (< light_threshold):   flag gaps for insight ledger, but don't disrupt flow
        #   medium (light..severe):        gap-fill — record gaps, status stays "reviewing" (not full re-research)
        #   severe (> severe_threshold):   full re-research with expanded queries
        preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
        light_thr = float(preset.get("verify_light_threshold", 0.20))
        medium_thr = float(preset.get("verify_medium_threshold", 0.30))
        severe_thr = float(preset.get("verify_severe_threshold", 0.40))

        if result.unsupported_claims > 0 and result.total_claims > 0:
            unsup_ratio = result.unsupported_claims / result.total_claims

            # Record supplementary queries as gaps regardless of tier
            if result.supplementary_queries:
                section.gaps = result.supplementary_queries[:4]
                dashboard.coverage_gaps.extend(result.supplementary_queries[:2])

            if unsup_ratio > severe_thr:
                # ── SEVERE: full re-research ──
                section.status = "researching"
                _emit_progress(
                    state,
                    "verify_severe",
                    {
                        "section": section.title,
                        "unsup_ratio": round(unsup_ratio, 2),
                        "message": f"Verification severe ({unsup_ratio:.0%} unsupported) — returning to research.",
                        "unsupported_claims": result.unsupported_claims,
                        "total_claims": result.total_claims,
                    },
                )
                return state

            elif unsup_ratio > light_thr:
                # ── MEDIUM: gap-fill only — record gaps but don't re-research ──
                # This prevents the "infinite verify→research loop" for moderately weak sections.
                _emit_progress(
                    state,
                    "verify_medium",
                    {
                        "section": section.title,
                        "unsup_ratio": round(unsup_ratio, 2),
                        "message": f"Verification medium ({unsup_ratio:.0%} unsupported) — gaps recorded, proceeding.",
                        "gaps": section.gaps[:3],
                    },
                )
                # Record as insights for the limitations section
                for gap in section.gaps[:3]:
                    _record_insight(state, "gap", gap, section.title, "verify_node_medium")
            else:
                # ── LIGHT: just flag ──
                _emit_progress(
                    state,
                    "verify_light",
                    {
                        "section": section.title,
                        "unsup_ratio": round(unsup_ratio, 2),
                        "message": f"Verification light ({unsup_ratio:.0%} unsupported) — minor gaps noted.",
                    },
                )

        conflict = getattr(result, "conflict_notes", None)
        if conflict:
            dashboard.conflict_notes.extend(conflict)
            # ── Record conflicts to Research Insights Ledger ──
            for note in conflict:
                _record_insight(
                    state,
                    insight_type="conflict",
                    text=note,
                    section_id=section.title,
                    source_context="verify_node",
                )
    except Exception as e:
        logger.warning(f"Verification failed for section '{section.title}': {e}")

    section.status = "done"
    state.setdefault("sections_completed", []).append(section.title)
    dashboard.update_confidence()
    _emit_progress(
        state,
        "section_verify_done",
        {
            "section": section.title,
            "status": "done",
            "coverage": section.coverage_score,
        },
    )

    return state


def review_gate_node(state: DeepResearchState) -> DeepResearchState:
    """最终审核门：所有章节审核通过后，才允许进入 synthesize。"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "review_gate")
    if bool(state.get("force_synthesize", False)):
        state["review_gate_next"] = "synthesize"
        return state
    if bool(state.get("skip_draft_review", False)):
        state["review_gate_next"] = "synthesize"
        return state

    dashboard = state.get("dashboard")
    if dashboard is None or not dashboard.sections:
        state["review_gate_next"] = "synthesize"
        return state

    waiter = state.get("review_waiter")
    if not waiter:
        _emit_progress(
            state,
            "warning",
            {"message": "未配置审核等待器，跳过最终审核门。"},
        )
        state["review_gate_next"] = "synthesize"
        return state

    handled = state.setdefault("review_handled_at", {})
    approved_count = 0
    pending_sections: List[str] = []
    revise_target: Optional[str] = None
    revise_feedback = ""

    for sec in dashboard.sections:
        review = waiter(sec.title)
        if not review:
            pending_sections.append(sec.title)
            continue
        action = str(review.get("action") or "").strip().lower()
        if action == "approve":
            approved_count += 1
            continue
        if action == "revise":
            created_at = float(review.get("created_at") or 0.0)
            last_handled = float(handled.get(sec.title) or 0.0)
            if created_at > last_handled:
                handled[sec.title] = created_at
                revise_target = sec.title
                revise_feedback = str(review.get("feedback") or "")
                break
            # 旧 revise 已处理过，等待用户提交新的审核结果
            pending_sections.append(sec.title)
            continue
        pending_sections.append(sec.title)

    if revise_target:
        section = dashboard.get_section(revise_target)
        if section:
            section.status = "researching"
        state["current_section"] = revise_target
        # ── Record revise feedback as a limitation insight ──
        if revise_feedback:
            _record_insight(
                state,
                insight_type="limitation",
                text=revise_feedback,
                section_id=revise_target,
                source_context="review_gate_node",
            )
        _emit_progress(
            state,
            "review_requeue",
            {
                "section": revise_target,
                "message": "根据审核意见回到该章节重写。",
                "feedback": revise_feedback,
            },
        )
        state["review_gate_next"] = "research"
        return state

    total = len(dashboard.sections)
    if approved_count >= total and not pending_sections:
        _emit_progress(
            state,
            "all_reviews_approved",
            {"approved": approved_count, "total": total},
        )
        state["review_gate_next"] = "synthesize"
        return state

    _emit_progress(
        state,
        "waiting_review_all",
        {
            "approved": approved_count,
            "total": total,
            "pending_sections": pending_sections,
            "message": "等待所有章节审核通过后进入最终整合。",
        },
    )
    interrupt(
        {
            "reason": "waiting_for_review",
            "pending_sections": pending_sections,
        }
    )
    return state


def synthesize_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 5: 全局综合 — 生成摘要 + 不足与展望 + 参考文献"""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "synthesize")
    client, model_override = _resolve_step_client_and_model(state, "synthesize")
    is_zh = (state.get("output_language") or "auto").lower() == "zh"
    dashboard = state.get("dashboard")

    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for raw in items:
            v = str(raw or "").strip()
            if not v:
                continue
            k = v.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
        return out

    def _split_references_block(markdown_text: str) -> Tuple[str, str]:
        """Split body and references, keeping references untouched for coherence pass."""
        if not markdown_text.strip():
            return "", ""
        ref_titles = ["## 参考文献", "## References"]
        for title in ref_titles:
            idx = markdown_text.rfind(f"\n{title}")
            if idx < 0 and markdown_text.startswith(title):
                idx = 0
            if idx >= 0:
                body = markdown_text[:idx].rstrip()
                refs = markdown_text[idx:].strip()
                return (body + "\n") if body else "", refs + "\n"
        return markdown_text.rstrip() + "\n", ""

    def _extract_bracket_tokens(markdown_text: str) -> List[str]:
        """Extract bracketed tokens like [cite_key] / [evidence limited]."""
        if not markdown_text:
            return []
        return [m.strip() for m in re.findall(r"\[([^\[\]\n]{1,120})\]", markdown_text) if m.strip()]

    def _is_protected_reference_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return False
        # Keep epistemic tags stable.
        if t == "evidence limited":
            return True
        # Numeric citation style, e.g., [1], [23]
        if re.fullmatch(r"\d{1,4}", t):
            return True
        # Common cite-key style, e.g., [3dd4798b...], [smith2021], [doi:...]
        if re.fullmatch(r"[a-z0-9_.:/-]{6,120}", t):
            return True
        return False

    def _citation_guard_ok(before_md: str, after_md: str) -> Tuple[bool, Dict[str, Any]]:
        """Guard against coherence pass dropping too many citation/evidence markers."""
        before_tokens = {t.lower() for t in _extract_bracket_tokens(before_md) if _is_protected_reference_token(t)}
        after_tokens = {t.lower() for t in _extract_bracket_tokens(after_md) if _is_protected_reference_token(t)}
        if not before_tokens:
            return True, {"before": 0, "after": len(after_tokens), "missing": 0, "reason": "no_protected_tokens"}

        missing = before_tokens - after_tokens
        before_n = len(before_tokens)
        after_n = len(after_tokens)
        missing_ratio = len(missing) / max(1, before_n)

        # Require at least 75% token retention and <=25% missing ratio.
        if after_n < max(1, int(before_n * 0.75)) or missing_ratio > 0.25:
            return False, {
                "before": before_n,
                "after": after_n,
                "missing": len(missing),
                "missing_examples": list(sorted(missing))[:8],
            }

        ev_before = before_md.lower().count("[evidence limited]")
        ev_after = after_md.lower().count("[evidence limited]")
        if ev_before > 0 and ev_after == 0:
            return False, {
                "before": before_n,
                "after": after_n,
                "missing": len(missing),
                "reason": "all_evidence_limited_tags_removed",
            }
        return True, {"before": before_n, "after": after_n, "missing": len(missing)}

    def _strip_redundant_heading(section_text: str, title_candidates: List[str]) -> str:
        text = (section_text or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        if not lines:
            return text
        first_raw = lines[0].strip()
        first = first_raw.lstrip("#").strip().lower()
        candidate_keys = [c.strip().lower() for c in title_candidates if str(c or "").strip()]
        if first in candidate_keys:
            lines = lines[1:]
        elif any(k and k in first for k in candidate_keys):
            lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _script_profile(text: str) -> Tuple[int, int]:
        cjk_n = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        latin_n = len(re.findall(r"[A-Za-z]", text or ""))
        return cjk_n, latin_n

    def _is_mostly_english(text: str) -> bool:
        cjk_n, latin_n = _script_profile(text)
        return latin_n >= 80 and latin_n > cjk_n * 1.8

    def _is_mostly_chinese(text: str) -> bool:
        cjk_n, latin_n = _script_profile(text)
        return cjk_n >= 80 and cjk_n > latin_n * 1.2

    def _coerce_to_target_language(text: str, section_label: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw
        lang = (state.get("output_language") or "auto").strip().lower()
        should_translate = False
        target_lang = ""
        if lang == "zh" and _is_mostly_english(raw):
            should_translate = True
            target_lang = "Chinese (中文)"
        elif lang == "en" and _is_mostly_chinese(raw):
            should_translate = True
            target_lang = "English"
        if not should_translate:
            return raw

        prompt = _pm.render(
            "translate_content.txt",
            target_lang=target_lang,
            section_label=section_label,
            raw=raw,
        )
        try:
            resp_lang = client.chat(
                messages=[
                    {"role": "system", "content": "You are a precise academic translator."},
                    {"role": "user", "content": prompt},
                ],
                model=model_override,
                max_tokens=1200,
            )
            translated = (resp_lang.get("final_text") or "").strip()
            return translated or raw
        except Exception:
            logger.debug("Language coercion failed for %s", section_label, exc_info=True)
            return raw

    def _language_consistency_ok(text: str) -> bool:
        lang = (state.get("output_language") or "auto").strip().lower()
        if lang not in ("zh", "en"):
            return True
        cjk_n, latin_n = _script_profile(text or "")
        if lang == "zh":
            # Allow English citation keys/terms, but body should still be Chinese-dominant.
            return cjk_n >= 80 and cjk_n >= int(latin_n * 0.55)
        # English target
        return latin_n >= 120 and latin_n >= int(cjk_n * 1.3)

    # ── Load Research Insights Ledger ──
    insights_by_type: Dict[str, List[str]] = {"gap": [], "conflict": [], "limitation": [], "future_direction": []}
    job_id = state.get("job_id") or ""
    if job_id:
        try:
            from src.collaboration.research.job_store import list_insights, bulk_mark_insights_addressed
            all_insights = list_insights(job_id, status="open")
            for ins in all_insights:
                itype = str(ins.get("insight_type") or "gap")
                text = str(ins.get("text") or "").strip()
                if text and itype in insights_by_type:
                    insights_by_type[itype].append(text)
        except Exception:
            logger.debug("Failed to load insights for synthesize", exc_info=True)

    # ── Aggregate open gaps from multiple channels for future agenda ──
    aggregated_open_gaps: List[str] = []
    aggregated_open_gaps.extend(insights_by_type.get("gap", []))
    if dashboard:
        aggregated_open_gaps.extend(getattr(dashboard, "coverage_gaps", []) or [])
        for sec in getattr(dashboard, "sections", []) or []:
            aggregated_open_gaps.extend(getattr(sec, "gaps", []) or [])
    aggregated_open_gaps = _dedupe_keep_order(aggregated_open_gaps)
    aggregated_conflict_notes: List[str] = []
    aggregated_conflict_notes.extend(insights_by_type.get("conflict", []))
    if dashboard:
        aggregated_conflict_notes.extend(getattr(dashboard, "conflict_notes", []) or [])
    aggregated_conflict_notes = _dedupe_keep_order(aggregated_conflict_notes)

    # 生成摘要
    full_md = "\n".join(state.get("markdown_parts", []))
    prompt = _pm.render(
        "generate_abstract.txt",
        language_instruction=_language_instruction(state),
        full_md=full_md[:5000],
    )

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an academic abstract writing expert."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=500,
        )
        abstract = (resp.get("final_text") or "").strip()
    except Exception:
        abstract = ""
    abstract = _strip_redundant_heading(abstract, ["摘要", "abstract"])
    abstract = _coerce_to_target_language(abstract, "abstract")

    # ── Generate "Limitations and Future Directions" section from insights ──
    limitations_section = ""
    scarce_sections = [s.title for s in (dashboard.sections if dashboard else []) if getattr(s, "evidence_scarce", False)]
    has_insights = any(len(v) > 0 for v in insights_by_type.values()) or bool(aggregated_conflict_notes)
    has_scarce_sections = len(scarce_sections) > 0
    if has_insights or has_scarce_sections:
        insight_block_parts: List[str] = []
        if insights_by_type["gap"]:
            insight_block_parts.append("Information Gaps:\n" + "\n".join(f"- {g}" for g in insights_by_type["gap"][:15]))
        if aggregated_conflict_notes:
            insight_block_parts.append(
                "Conflicts/Contradictions with Attribution Clues:\n"
                + "\n".join(f"- {c}" for c in aggregated_conflict_notes[:12])
            )
        if insights_by_type["limitation"]:
            insight_block_parts.append("Reviewer Noted Limitations:\n" + "\n".join(f"- {l}" for l in insights_by_type["limitation"][:10]))
        if has_scarce_sections:
            insight_block_parts.append(
                "Evidence-Scarce Sections (degraded due to sparse retrieval):\n"
                + "\n".join(f"- {s}" for s in scarce_sections[:20])
            )
        insight_block = "\n\n".join(insight_block_parts)

        lim_title = "不足与未来方向" if is_zh else "Limitations and Future Directions"
        lim_prompt = _pm.render(
            "limitations_section.txt",
            lim_title=lim_title,
            full_md=full_md[:3000],
            insight_block=insight_block[:2500],
            language_instruction=_language_instruction(state),
        )

        try:
            resp_lim = client.chat(
                messages=[
                    {"role": "system", "content": "You are an academic review writer specializing in critical analysis."},
                    {"role": "user", "content": lim_prompt},
                ],
                model=model_override,
                max_tokens=800,
            )
            limitations_section = (resp_lim.get("final_text") or "").strip()
        except Exception:
            # Fallback: simple bullet list
            fallback_items = insights_by_type["gap"][:5] + insights_by_type["conflict"][:3] + insights_by_type["limitation"][:3]
            if fallback_items:
                limitations_section = "\n".join(f"- {item}" for item in fallback_items)
        limitations_section = _strip_redundant_heading(
            limitations_section,
            ["不足与未来方向", "Limitations and Future Directions"],
        )
        limitations_section = _coerce_to_target_language(limitations_section, "limitations")

    # ── Structured "Open Gaps -> Future Directions" agenda ──
    open_gap_agenda = ""
    if aggregated_open_gaps:
        gap_lines = "\n".join(f"- {g}" for g in aggregated_open_gaps[:30])
        agenda_title = "开放问题与未来研究议程" if is_zh else "Open Gaps and Future Research Agenda"
        agenda_prompt = _pm.render(
            "open_gaps_agenda.txt",
            agenda_title=agenda_title,
            language_instruction=_language_instruction(state),
            full_md=full_md[:3500],
            gap_lines=gap_lines,
        )
        try:
            resp_agenda = client.chat(
                messages=[
                    {"role": "system", "content": "You are an expert in planning high-impact research roadmaps."},
                    {"role": "user", "content": agenda_prompt},
                ],
                model=model_override,
                max_tokens=1000,
            )
            open_gap_agenda = (resp_agenda.get("final_text") or "").strip()
        except Exception:
            # Deterministic fallback to ensure section always exists
            fallback_gaps = aggregated_open_gaps[:8]
            if fallback_gaps:
                bullet_title = "潜在方向" if is_zh else "Potential Directions"
                gap_label = "问题" if is_zh else "Gap"
                dir_label = "方向" if is_zh else "Direction"
                dir_text = (
                    "构建针对性数据集与实验流程，并执行可重复验证。"
                    if is_zh
                    else "build targeted datasets/protocols and run controlled verification."
                )
                open_gap_agenda = (
                    f"{bullet_title}:\n"
                    + "\n".join(f"- {gap_label}: {g}\n  - {dir_label}: {dir_text}" for g in fallback_gaps)
                )
        open_gap_agenda = _strip_redundant_heading(
            open_gap_agenda,
            ["开放问题与未来研究议程", "Open Gaps and Future Research Agenda"],
        )
        open_gap_agenda = _coerce_to_target_language(open_gap_agenda, "open gaps agenda")

    # 组装最终文档
    parts = state.get("markdown_parts", [])
    if abstract:
        # 在标题后插入摘要
        if len(parts) > 0:
            abstract_title = "摘要" if is_zh else "Abstract"
            parts.insert(1, f"\n## {abstract_title}\n\n{abstract}\n")

    # 添加不足与展望
    if limitations_section:
        lim_title = "不足与未来方向" if is_zh else "Limitations and Future Directions"
        parts.append(f"\n## {lim_title}\n\n{limitations_section}\n")
    if open_gap_agenda:
        gap_title = "开放问题与未来研究议程" if is_zh else "Open Gaps and Future Research Agenda"
        parts.append(f"\n## {gap_title}\n\n{open_gap_agenda}\n")

    # 添加参考文献
    if state.get("canvas_id"):
        try:
            from src.collaboration.citation.formatter import format_reference_list
            from src.collaboration.canvas.canvas_manager import get_canvas_citations
            citations = get_canvas_citations(state["canvas_id"])
            if citations:
                ref_text = format_reference_list(citations)
                ref_title = "参考文献" if is_zh else "References"
                parts.append(f"\n## {ref_title}\n\n{ref_text}\n")
                state["citations"] = citations
        except Exception:
            pass

    # ── Global coherence refinement pass (whole-document editorial integration) ──
    assembled = "\n".join(parts)
    body_md, refs_md = _split_references_block(assembled)
    final_markdown = assembled
    if body_md.strip():
        lang_hard_rule = ""
        if is_zh:
            lang_hard_rule = "Output must remain Chinese (中文). Keep citation tags unchanged."
        elif (state.get("output_language") or "auto").lower() == "en":
            lang_hard_rule = "Output must remain English. Keep citation tags unchanged."
        coherence_prompt = _pm.render(
            "coherence_refine.txt",
            language_instruction=_language_instruction(state),
            lang_hard_rule=lang_hard_rule if lang_hard_rule else "Respect the document's dominant language.",
            body_md=body_md,
        )
        try:
            resp_coherence = client.chat(
                messages=[
                    {"role": "system", "content": "You are an expert in scholarly synthesis and coherence editing."},
                    {"role": "user", "content": coherence_prompt},
                ],
                model=model_override,
                max_tokens=3500,
            )
            refined_body = (resp_coherence.get("final_text") or "").strip()
            if refined_body:
                candidate_markdown = refined_body + ("\n\n" + refs_md.strip() if refs_md.strip() else "")
                ok, guard_diag = _citation_guard_ok(assembled, candidate_markdown)
                lang_ok = _language_consistency_ok(refined_body)
                if ok and lang_ok:
                    final_markdown = candidate_markdown
                    _emit_progress(
                        state,
                        "global_refine_done",
                        {
                            "message": "已完成全文连贯性整合与跨章节一致性优化。",
                            "open_gaps": len(aggregated_open_gaps),
                            "citation_guard": guard_diag,
                        },
                    )
                else:
                    logger.warning(
                        "Global refine fallback (citation/lang guard): citation=%s, lang_ok=%s",
                        guard_diag,
                        lang_ok,
                    )
                    _emit_progress(
                        state,
                        "citation_guard_fallback",
                        {
                            "message": "检测到整合后语言/引用一致性风险，已回退到整合前版本。",
                            "guard": guard_diag,
                            "language_guard_ok": lang_ok,
                        },
                    )
        except Exception as e:
            logger.warning("Global coherence refine failed: %s", e)
            _emit_progress(
                state,
                "warning",
                {"message": "全文连贯性整合失败，已回退到合成稿。"},
            )

    # ── Final citation resolution pass (whole document) ──
    all_evidence_chunks = state.get("evidence_chunks", [])
    if all_evidence_chunks:
        resolved_markdown, resolved_citations = _resolve_text_citations(
            state,
            final_markdown,
            all_evidence_chunks,
            include_unreferenced_documents=False,
        )
        final_markdown = resolved_markdown
        if resolved_citations:
            state["citations"] = resolved_citations
            # 使用最终引用集重建参考文献，避免与正文 cite_key 不一致
            from src.collaboration.citation.formatter import format_reference_list
            body_md, _old_refs_md = _split_references_block(final_markdown)
            ref_title = "参考文献" if is_zh else "References"
            ref_text = format_reference_list(resolved_citations).strip()
            if ref_text:
                final_markdown = body_md.rstrip() + f"\n\n## {ref_title}\n\n{ref_text}\n"

    state["markdown_parts"] = [final_markdown]

    # ── Mark consumed insights as addressed ──
    if job_id and has_insights:
        try:
            from src.collaboration.research.job_store import bulk_mark_insights_addressed
            # Keep "gap" insights open for future research tracking.
            bulk_mark_insights_addressed(job_id, insight_type="conflict")
            bulk_mark_insights_addressed(job_id, insight_type="limitation")
            bulk_mark_insights_addressed(job_id, insight_type="future_direction")
        except Exception:
            logger.debug("Failed to mark insights addressed", exc_info=True)

    # ── Persist human-readable insights to Canvas model ──
    if state.get("canvas_id") and has_insights:
        try:
            from src.collaboration.canvas.canvas_manager import update_canvas
            all_insight_texts = []
            for itype, items in insights_by_type.items():
                for t in items[:10]:
                    all_insight_texts.append(f"[{itype}] {t}")
            update_canvas(state["canvas_id"], research_insights=all_insight_texts)
        except Exception:
            logger.debug("Failed to persist insights to canvas", exc_info=True)

    _emit_progress(
        state,
        "synthesize_done",
        {
            "sections_done": len(state.get("sections_completed", [])),
            "citations": len(state.get("citations", [])),
            "insights_consumed": sum(len(v) for v in insights_by_type.values()),
        },
    )
    return state


# ────────────────────────────────────────────────
# 路由函数
# ────────────────────────────────────────────────


def _write_or_claims(state: DeepResearchState) -> str:
    """When moving to writing: go to generate_claims unless skip_claim_generation or depth is lite."""
    if state.get("skip_claim_generation") or state.get("depth") == "lite":
        return "write"
    return "generate_claims"


def _should_continue_research(state: DeepResearchState) -> str:
    """决定评估后是继续搜索还是进入写作。

    Guard rails (all driven by depth preset):
    1. Global iteration cap (max_iterations)
    2. Per-section research round cap (max_section_research_rounds)
    3. Coverage threshold (coverage_threshold)
    """
    dashboard = state.get("dashboard")
    if dashboard is None:
        return _write_or_claims(state)

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return _write_or_claims(state)

    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))

    # Guard 0: forced summarize mode from cost monitor
    # Always short-circuit to write to minimize extra LLM hops under cost pressure.
    if bool(state.get("force_synthesize", False)):
        return "write"

    # Guard 1: global iteration cap (dynamically set in execute_deep_research)
    max_iter = state.get("max_iterations", 30)
    if state.get("iteration_count", 0) >= max_iter:
        logger.info("Global iteration cap reached (%d), moving to write", max_iter)
        return _write_or_claims(state)

    # Guard 2: per-section research round cap
    max_rounds = int(preset.get("max_section_research_rounds", 3))
    if section.research_rounds >= max_rounds:
        logger.info("Section '%s' hit per-section cap (%d rounds), moving to write", section.title, max_rounds)
        return _write_or_claims(state)

    # Guard 3: coverage threshold
    cov_threshold = float(preset.get("coverage_threshold", 0.6))
    if section.coverage_score >= cov_threshold or not section.gaps:
        return _write_or_claims(state)

    # Guard 4: diminishing-return early stop (coverage curve plateau)
    history = (state.get("coverage_history") or {}).get(section.title, [])
    if len(history) >= 3:
        gain_recent = float(history[-1]) - float(history[-2])
        gain_prev = float(history[-2]) - float(history[-3])
        plateau_floor = float(preset.get("coverage_plateau_floor", max(0.7, cov_threshold - 0.05)))
        min_gain = float(preset.get("coverage_plateau_min_gain", 0.02))
        if float(history[-1]) >= plateau_floor and gain_recent < min_gain and gain_prev < min_gain:
            _emit_progress(
                state,
                "coverage_plateau_early_stop",
                {
                    "section": section.title,
                    "coverage": round(float(history[-1]), 3),
                    "gain_recent": round(gain_recent, 4),
                    "gain_prev": round(gain_prev, 4),
                    "message": "Coverage gain curve flattened; early stop triggered for cost efficiency.",
                },
            )
            return _write_or_claims(state)

    # Still have gaps → continue research
    return "research"


def _after_verify(state: DeepResearchState) -> str:
    """验证后，检查是否还有章节待处理或需要补充研究"""
    if bool(state.get("force_synthesize", False)):
        return "synthesize"
    dashboard = state.get("dashboard")
    if dashboard is None:
        return "synthesize"

    # 如果当前章节被打回研究阶段
    section = dashboard.get_section(state.get("current_section", ""))
    if section and section.status == "researching":
        return "research"

    next_section = dashboard.get_next_section()
    if next_section is None:
        if not bool(state.get("skip_draft_review", False)):
            return "review_gate"
        return "synthesize"

    return "research"


def _after_review_gate(state: DeepResearchState) -> str:
    if bool(state.get("force_synthesize", False)):
        return "synthesize"
    return str(state.get("review_gate_next") or "review_gate")


# ────────────────────────────────────────────────
# 构建 LangGraph
# ────────────────────────────────────────────────

def build_research_graph(include_scope_plan: bool = True) -> StateGraph:
    """构建 Deep Research Agent 的 LangGraph

    Scope → Plan → Research → Evaluate → Write → Verify → (next section / Synthesize)
    """
    graph = StateGraph(DeepResearchState)

    # 添加节点
    graph.add_node("scope", scoping_node)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate_claims", generate_claims_node)
    graph.add_node("write", write_node)
    graph.add_node("verify", verify_node)
    graph.add_node("review_gate", review_gate_node)
    graph.add_node("synthesize", synthesize_node)

    # 添加边
    if include_scope_plan:
        graph.set_entry_point("scope")
        graph.add_edge("scope", "plan")
        graph.add_edge("plan", "research")
    else:
        graph.set_entry_point("research")
    graph.add_edge("research", "evaluate")
    graph.add_conditional_edges("evaluate", _should_continue_research, {
        "research": "research",
        "write": "write",
        "generate_claims": "generate_claims",
    })
    graph.add_edge("generate_claims", "write")
    graph.add_edge("write", "verify")
    graph.add_conditional_edges("verify", _after_verify, {
        "research": "research",
        "review_gate": "review_gate",
        "synthesize": "synthesize",
    })
    graph.add_conditional_edges("review_gate", _after_review_gate, {
        "review_gate": "review_gate",
        "research": "research",
        "synthesize": "synthesize",
    })
    graph.add_edge("synthesize", END)

    return graph


def _build_initial_state(
    *,
    topic: str,
    llm_client: Any,
    canvas_id: Optional[str] = None,
    session_id: str = "",
    user_id: str = "",
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    max_iterations: int = 30,
    output_language: str = "auto",
    clarification_answers: Optional[Dict[str, str]] = None,
    user_context: str = "",
    user_context_mode: str = "supporting",
    user_documents: Optional[List[Dict[str, str]]] = None,
    step_models: Optional[Dict[str, Optional[str]]] = None,
    step_model_strict: bool = False,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    review_waiter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    skip_draft_review: bool = False,
    skip_refine_review: bool = False,
    skip_claim_generation: bool = False,
    job_id: str = "",
    depth: str = DEFAULT_DEPTH,
) -> DeepResearchState:
    resolved_depth = depth if depth in DEPTH_PRESETS else DEFAULT_DEPTH
    preset = get_depth_preset(resolved_depth)
    # max_iterations is set as a placeholder here; the real value is computed
    # dynamically in execute_deep_research() after num_sections is known:
    #   max_iterations = max_iterations_per_section × num_sections
    return {
        "topic": topic,
        "dashboard": ResearchDashboard(),
        "trajectory": ResearchTrajectory(topic=topic),
        "canvas_id": canvas_id or "",
        "session_id": session_id,
        "user_id": user_id,
        "search_mode": search_mode,
        "filters": filters or {},
        "current_section": "",
        "sections_completed": [],
        "markdown_parts": [],
        "citations": [],
        "evidence_chunks": [],
        "evidence_chunk_empty_value": "",
        "citation_doc_key_map": {},
        "citation_existing_keys": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,  # placeholder; overridden in execute_deep_research()
        "llm_client": llm_client,
        "model_override": model_override,
        "output_language": output_language,
        "clarification_answers": clarification_answers or {},
        "user_context": user_context or "",
        "user_context_mode": user_context_mode or "supporting",
        "user_documents": user_documents or [],
        "step_models": step_models or {},
        "step_model_strict": bool(step_model_strict),
        "progress_callback": progress_callback,
        "cancel_check": cancel_check,
        "review_waiter": review_waiter,
        "skip_draft_review": bool(skip_draft_review),
        "skip_refine_review": bool(skip_refine_review),
        "skip_claim_generation": bool(skip_claim_generation),
        "verified_claims": "",
        "depth": resolved_depth,
        "depth_preset": preset,
        "review_gate_rounds": 0,
        "review_gate_unchanged": 0,
        "review_gate_last_snapshot": "",
        "review_gate_next": "review_gate",
        "review_handled_at": {},
        "revision_queue": [],
        "review_seen_at": {},
        "graph_step_count": 0,
        "cost_warned": False,
        "force_synthesize": False,
        "coverage_history": {},
        "last_cost_tick_step": 0,
        "job_id": job_id,
        "error": None,
    }


def start_deep_research(
    topic: str,
    llm_client: Any,
    canvas_id: Optional[str] = None,
    session_id: str = "",
    user_id: str = "",
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    clarification_answers: Optional[Dict[str, str]] = None,
    output_language: str = "auto",
    model_override: Optional[str] = None,
    step_models: Optional[Dict[str, Optional[str]]] = None,
    step_model_strict: bool = False,
    max_iterations: int = 30,
) -> Dict[str, Any]:
    """Phase 1 only: scope + plan. Return outline and brief for confirmation."""
    state = _build_initial_state(
        topic=topic,
        llm_client=llm_client,
        canvas_id=canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=search_mode,
        filters=filters,
        model_override=model_override,
        max_iterations=max_iterations,
        output_language=output_language,
        clarification_answers=clarification_answers,
        step_models=step_models,
        step_model_strict=step_model_strict,
    )
    try:
        state = scoping_node(state)
        state = plan_node(state)
        dashboard = state["dashboard"]
        return {
            "topic": topic,
            "session_id": state.get("session_id", ""),
            "canvas_id": state.get("canvas_id", ""),
            "brief": {
                "topic": dashboard.brief.topic,
                "scope": dashboard.brief.scope,
                "success_criteria": dashboard.brief.success_criteria,
                "key_questions": dashboard.brief.key_questions,
                "exclusions": dashboard.brief.exclusions,
                "time_range": dashboard.brief.time_range,
                "source_priority": dashboard.brief.source_priority,
            },
            "outline": [s.title for s in dashboard.sections],
            "initial_stats": {
                "total_sources": dashboard.total_sources,
                "total_iterations": dashboard.total_iterations,
            },
        }
    except Exception as e:
        logger.error("start_deep_research failed: %s", e)
        return {
            "topic": topic,
            "session_id": session_id,
            "canvas_id": canvas_id or "",
            "brief": {
                "topic": topic,
                "scope": f"Comprehensive review of {topic}",
                "success_criteria": [],
                "key_questions": [topic],
                "exclusions": [],
                "time_range": "",
                "source_priority": [],
            },
            "outline": [topic],
            "initial_stats": {"total_sources": 0, "total_iterations": 0},
            "error": str(e),
        }


def build_deep_research_result_from_state(
    state: Dict[str, Any],
    *,
    topic: str,
    elapsed_ms: float,
    fallback_outline: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build stable API result from a terminal graph state."""
    final_state: Dict[str, Any] = state if isinstance(state, dict) else {}
    final_dashboard = final_state.get("dashboard", ResearchDashboard())
    final_markdown = "\n".join(final_state.get("markdown_parts", []))
    final_citations = final_state.get("citations", [])
    all_chunks = final_state.get("evidence_chunks", [])
    if final_markdown and all_chunks:
        final_markdown, resolved_citations = _resolve_text_citations(
            final_state,
            final_markdown,
            all_chunks,
            include_unreferenced_documents=False,
        )
        if resolved_citations:
            final_citations = resolved_citations
    if isinstance(final_dashboard, ResearchDashboard):
        outline = [s.title for s in final_dashboard.sections]
        dashboard_dict = final_dashboard.to_dict()
    else:
        outline = list(fallback_outline or [])
        dashboard_dict = final_dashboard if isinstance(final_dashboard, dict) else {}
    return {
        "markdown": final_markdown,
        "canvas_id": final_state.get("canvas_id", ""),
        "outline": outline,
        "citations": final_citations,
        "dashboard": dashboard_dict,
        "total_time_ms": elapsed_ms,
        "topic": topic,
    }


def prepare_deep_research_runtime(
    topic: str,
    llm_client: Any,
    confirmed_outline: List[str],
    confirmed_brief: Optional[Dict[str, Any]] = None,
    canvas_id: Optional[str] = None,
    session_id: str = "",
    user_id: str = "",
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    max_iterations: int = 30,
    output_language: str = "auto",
    step_models: Optional[Dict[str, Optional[str]]] = None,
    step_model_strict: bool = False,
    user_context: str = "",
    user_context_mode: str = "supporting",
    user_documents: Optional[List[Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    review_waiter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    skip_draft_review: bool = False,
    skip_refine_review: bool = False,
    skip_claim_generation: bool = False,
    job_id: str = "",
    depth: str = DEFAULT_DEPTH,
) -> Dict[str, Any]:
    """Prepare compiled graph runtime for deep-research execution/resume."""
    t0 = time.perf_counter()
    resolved_topic = topic.strip()
    preset = get_depth_preset(depth)
    graph = build_research_graph(include_scope_plan=False)
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    compiled.recursion_limit = int(preset["recursion_limit"])
    initial_state = _build_initial_state(
        topic=resolved_topic,
        llm_client=llm_client,
        canvas_id=canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=search_mode,
        filters=filters,
        model_override=model_override,
        max_iterations=max_iterations,
        output_language=output_language,
        user_context=user_context,
        user_context_mode=user_context_mode,
        user_documents=user_documents,
        step_models=step_models,
        step_model_strict=step_model_strict,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        review_waiter=review_waiter,
        skip_draft_review=skip_draft_review,
        skip_refine_review=skip_refine_review,
        skip_claim_generation=skip_claim_generation,
        job_id=job_id,
        depth=depth,
    )

    _emit_progress(initial_state, "depth_resolved", {"depth": depth, "preset": preset})
    dashboard = initial_state["dashboard"]
    brief = confirmed_brief or {}
    dashboard.brief = ResearchBrief(
        topic=brief.get("topic", resolved_topic),
        scope=brief.get("scope", f"Comprehensive review of {resolved_topic}"),
        success_criteria=brief.get("success_criteria", []),
        key_questions=brief.get("key_questions", [resolved_topic]),
        exclusions=brief.get("exclusions", []),
        time_range=brief.get("time_range", ""),
        source_priority=brief.get("source_priority", ["peer-reviewed"]),
    )

    outline = [s.strip() for s in (confirmed_outline or []) if s and s.strip()]
    if not outline:
        outline = [resolved_topic]
    dashboard.sections = []
    for idx, title in enumerate(outline):
        dashboard.add_section(title)
        initial_state["trajectory"].add_branch(f"sec_{idx+1}", title)

    iter_per_sec = int(preset.get("max_iterations_per_section", 5))
    num_sections = len(outline)
    scaled_max_iter = iter_per_sec * num_sections
    initial_state["max_iterations"] = scaled_max_iter
    logger.info(
        "Depth=%s | sections=%d | max_iterations=%d (%d × %d)",
        depth, num_sections, scaled_max_iter, iter_per_sec, num_sections,
    )
    initial_state["markdown_parts"] = [f"# {dashboard.brief.topic}\n"]

    if initial_state.get("canvas_id"):
        try:
            from src.collaboration.canvas.canvas_manager import upsert_outline, update_canvas
            from src.collaboration.canvas.models import OutlineSection
            sections_payload = [
                OutlineSection(title=title, level=1, order=idx, status="todo")
                for idx, title in enumerate(outline)
            ]
            upsert_outline(initial_state["canvas_id"], sections_payload)
            update_canvas(
                initial_state["canvas_id"],
                stage="outline",
                skip_draft_review=bool(skip_draft_review),
                skip_refine_review=bool(skip_refine_review),
                research_brief={
                    "scope": dashboard.brief.scope,
                    "success_criteria": dashboard.brief.success_criteria,
                    "key_questions": dashboard.brief.key_questions,
                    "exclusions": dashboard.brief.exclusions,
                    "time_range": dashboard.brief.time_range,
                    "source_priority": dashboard.brief.source_priority,
                    "action_plan": "",
                },
            )
        except Exception as e:
            logger.warning("Failed to sync execute outline/brief to canvas: %s", e)

    _emit_progress(initial_state, "execute_started", {"outline": outline, "topic": dashboard.brief.topic})
    thread_id = job_id.strip() if isinstance(job_id, str) else ""
    if not thread_id:
        seed = f"{session_id}:{resolved_topic}:{int(time.time() * 1000)}"
        thread_id = f"dr-{abs(hash(seed))}"
    config = {"configurable": {"thread_id": thread_id}}
    return {
        "compiled": compiled,
        "config": config,
        "initial_state": initial_state,
        "outline": outline,
        "topic": dashboard.brief.topic,
        "started_at_perf": t0,
    }


def execute_deep_research(
    topic: str,
    llm_client: Any,
    confirmed_outline: List[str],
    confirmed_brief: Optional[Dict[str, Any]] = None,
    canvas_id: Optional[str] = None,
    session_id: str = "",
    user_id: str = "",
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    max_iterations: int = 30,
    output_language: str = "auto",
    step_models: Optional[Dict[str, Optional[str]]] = None,
    step_model_strict: bool = False,
    user_context: str = "",
    user_context_mode: str = "supporting",
    user_documents: Optional[List[Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    review_waiter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    skip_draft_review: bool = False,
    skip_refine_review: bool = False,
    skip_claim_generation: bool = False,
    job_id: str = "",
    depth: str = DEFAULT_DEPTH,
) -> Dict[str, Any]:
    """Phase 2: run research loop with confirmed brief/outline."""
    runtime = prepare_deep_research_runtime(
        topic=topic,
        llm_client=llm_client,
        confirmed_outline=confirmed_outline,
        confirmed_brief=confirmed_brief,
        canvas_id=canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=search_mode,
        filters=filters,
        model_override=model_override,
        max_iterations=max_iterations,
        output_language=output_language,
        step_models=step_models,
        step_model_strict=step_model_strict,
        user_context=user_context,
        user_context_mode=user_context_mode,
        user_documents=user_documents,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        review_waiter=review_waiter,
        skip_draft_review=skip_draft_review,
        skip_refine_review=skip_refine_review,
        skip_claim_generation=skip_claim_generation,
        job_id=job_id,
        depth=depth,
    )
    compiled = runtime["compiled"]
    config = runtime["config"]
    initial_state = runtime["initial_state"]

    # 将 job_id 注入 LangGraph RunnableConfig 的 tags/metadata，供 LangSmith 追踪使用
    if job_id:
        existing_tags = list(config.get("tags") or [])
        if f"job:{job_id}" not in existing_tags:
            existing_tags.append(f"job:{job_id}")
        existing_metadata = dict(config.get("metadata") or {})
        existing_metadata.update({"job_id": job_id, "topic": topic})
        config = {
            **config,
            "tags": existing_tags,
            "metadata": existing_metadata,
        }

    try:
        compiled.invoke(initial_state, config=config)
        state_snapshot = compiled.get_state(config)
    except Exception as e:
        logger.error(f"Deep Research execution failed: {e}")
        return {
            "markdown": f"# {topic}\n\nDeep Research execution failed: {e}",
            "canvas_id": canvas_id or "",
            "outline": runtime.get("outline") or [],
            "citations": [],
            "dashboard": {},
            "total_time_ms": (time.perf_counter() - float(runtime.get("started_at_perf") or time.perf_counter())) * 1000,
        }

    if getattr(state_snapshot, "next", ()):
        return {
            "status": "waiting_review",
            "markdown": "\n".join(initial_state.get("markdown_parts", [])),
            "canvas_id": initial_state.get("canvas_id", ""),
            "outline": runtime.get("outline") or [],
            "citations": initial_state.get("citations", []) or [],
            "dashboard": (initial_state.get("dashboard").to_dict() if initial_state.get("dashboard") else {}),
            "total_time_ms": (time.perf_counter() - float(runtime.get("started_at_perf") or time.perf_counter())) * 1000,
        }

    elapsed = (time.perf_counter() - float(runtime.get("started_at_perf") or time.perf_counter())) * 1000
    final_state = getattr(state_snapshot, "values", {}) or {}
    return build_deep_research_result_from_state(
        final_state,
        topic=str(runtime.get("topic") or topic),
        elapsed_ms=elapsed,
        fallback_outline=runtime.get("outline") or [],
    )


# ────────────────────────────────────────────────
# 入口函数
# ────────────────────────────────────────────────

def run_deep_research(
    topic: str,
    llm_client: Any,
    canvas_id: Optional[str] = None,
    session_id: str = "",
    user_id: str = "",
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    max_iterations: int = 30,
    clarification_answers: Optional[Dict[str, str]] = None,
    output_language: str = "auto",
    step_models: Optional[Dict[str, Optional[str]]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    depth: str = DEFAULT_DEPTH,
    step_model_strict: bool = False,
) -> Dict[str, Any]:
    """
    执行 Deep Research Agent。

    Args:
        depth: Research depth — "lite" (fast, ~3-10 min) or "comprehensive" (thorough, ~15-40 min).

    Returns:
        {
            "markdown": str,
            "canvas_id": str,
            "outline": list[str],
            "citations": list,
            "dashboard": dict,
            "total_time_ms": float,
        }
    """
    start_result = start_deep_research(
        topic=topic,
        llm_client=llm_client,
        canvas_id=canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=search_mode,
        filters=filters,
        clarification_answers=clarification_answers,
        output_language=output_language,
        model_override=model_override,
        step_models=step_models,
        max_iterations=max_iterations,
        step_model_strict=step_model_strict,
    )
    return execute_deep_research(
        topic=topic,
        llm_client=llm_client,
        confirmed_outline=start_result.get("outline", []),
        confirmed_brief=start_result.get("brief"),
        canvas_id=start_result.get("canvas_id") or canvas_id,
        session_id=session_id,
        user_id=user_id,
        search_mode=search_mode,
        filters=filters,
        model_override=model_override,
        max_iterations=max_iterations,
        output_language=output_language,
        step_models=step_models,
        progress_callback=progress_callback,
        depth=depth,
        step_model_strict=step_model_strict,
    )
