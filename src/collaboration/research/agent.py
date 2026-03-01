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

import requests

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from config.settings import settings
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
from src.retrieval.fulltext_compressor import compress_evidence_text_sync
from src.utils.prompt_manager import PromptManager
from src.utils.context_limits import (
    summarize_if_needed,
    cap_and_log,
    DR_USER_CONTEXT_MAX_CHARS,
    DR_SECTION_EVIDENCE_MAX_CHARS,
    FINAL_INTEGRATION_MAX_CHARS,
    PRELIMINARY_KNOWLEDGE_MAX_CHARS,
    PLAN_CONTEXT_MAX_CHARS,
    PLAN_CONTEXT_SUMMARIZE_TO,
)

logger = get_logger(__name__)
_pm = PromptManager()
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "rag_config.json"
_RUNTIME_LLM_CLIENTS: Dict[str, Any] = {}


def _register_runtime_llm_client(runtime_id: str, llm_client: Any) -> None:
    rid = str(runtime_id or "").strip()
    if not rid:
        return
    _RUNTIME_LLM_CLIENTS[rid] = llm_client


def _resolve_runtime_llm_client(state: "DeepResearchState") -> Any:
    """Read the live llm client from runtime registry (checkpoint-safe)."""
    rid = str(state.get("runtime_id") or state.get("job_id") or "").strip()
    if rid and rid in _RUNTIME_LLM_CLIENTS:
        return _RUNTIME_LLM_CLIENTS[rid]
    # Backward compatibility for in-process paths that still inject llm_client in state.
    legacy = state.get("llm_client")
    if legacy is not None:
        return legacy
    raise RuntimeError("Deep Research runtime missing llm client context")


# ── Runtime callback registry (checkpoint-safe) ───────────────────────────────
# progress_callback / cancel_check / review_waiter are Python callables that
# cannot be serialized by msgpack (LangGraph MemorySaver).  We keep them OUT of
# the LangGraph state and resolve them from this in-process registry instead.
_RUNTIME_CALLBACKS: Dict[str, Dict[str, Any]] = {}
_RUNTIME_QUERY_CACHES: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _register_runtime_callbacks(runtime_id: str, callbacks: Dict[str, Any]) -> None:
    rid = str(runtime_id or "").strip()
    if not rid:
        return
    existing = _RUNTIME_CALLBACKS.get(rid) or {}
    existing.update({k: v for k, v in callbacks.items() if v is not None})
    _RUNTIME_CALLBACKS[rid] = existing


def _resolve_runtime_callback(state: "DeepResearchState", key: str) -> Any:
    """Fetch a runtime callback from the registry (falls back to state for legacy paths)."""
    rid = str(state.get("runtime_id") or state.get("job_id") or "").strip()
    if rid and rid in _RUNTIME_CALLBACKS:
        return _RUNTIME_CALLBACKS[rid].get(key)
    return state.get(key)


def _resolve_runtime_query_cache(state: "DeepResearchState") -> Dict[str, Dict[str, Any]]:
    """Fetch per-runtime query cache; fallback to state for legacy/no-runtime paths."""
    rid = str(state.get("runtime_id") or state.get("job_id") or "").strip()
    if rid:
        cache = _RUNTIME_QUERY_CACHES.get(rid)
        if cache is None:
            cache = {}
            _RUNTIME_QUERY_CACHES[rid] = cache
        return cache
    # Legacy fallback (kept for backward compatibility only).
    return state.setdefault("query_chunk_cache", {})

# ── Engine-aware bilingual hints ──────────────────────────────────────────────
# Keep these prompt assets in src/prompts/ for centralized prompt management.
_BILINGUAL_HINT_ACADEMIC = _pm.load("bilingual_hint_academic.txt").strip()
_BILINGUAL_HINT_DISCOVERY = _pm.load("bilingual_hint_discovery.txt").strip()
_BILINGUAL_HINT_GAP = _pm.load("bilingual_hint_gap_queries.txt").strip()

# ── Engine-specific gap query routing ─────────────────────────────────────────
# Maps LLM JSON keys → category tags used by _execute_tiered_search routing.
_GAP_ENGINE_TAG_MAP = {
    "ncbi_pubmed": "gap_ncbi",
    "semantic_keywords": "gap_semantic",
    "tavily": "gap_tavily",
    "google_scholar": "gap_scholar",
    "google": "gap_google",
}
_GAP_ENGINE_ALL_KEYS = frozenset({
    "ncbi_pubmed", "semantic_keywords", "tavily",
    "google_scholar", "google",
    "semantic_umbrella", "semantic_ai4scholar_relevance",
    "semantic_ai4scholar_bulk", "meta",
})
_MAX_QUERY_CHARS = 2500

def _topic_is_chinese(topic: str) -> bool:
    """Return True if topic contains CJK characters (Chinese input detected)."""
    return bool(re.search(r"[\u4e00-\u9fff]", topic or ""))


def _normalize_topic_domain(value: Any) -> str:
    """Canonicalize topic_domain into stable routing labels."""
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"biomedical", "bio", "medicine", "medical", "life_sciences", "life_science", "biology"}:
        return "biomedical"
    if raw in {"cs_ai", "ai", "computer_science", "computerscience", "cs"}:
        return "cs_ai"
    return "general"


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+/]*", text or ""))


def _truncate_query_text(text: str, max_chars: int = _MAX_QUERY_CHARS) -> str:
    """Normalize whitespace and cap query length to avoid provider/parser issues."""
    q = re.sub(r"\s+", " ", (text or "").strip())
    if len(q) <= max_chars:
        return q
    clipped = q[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()


def _is_short_keyword_query(text: str) -> bool:
    q = (text or "").strip()
    if len(q) < 3:
        return False
    if "?" in q:
        return False
    # Avoid sentence-like punctuation for scholar/google keyword mode
    if any(p in q for p in [".", "!", "。", "；", ";"]):
        return False
    # English keyword phrase should be concise; Chinese phrase should not be too long
    if _topic_is_chinese(q):
        return len(q) <= 28
    return _token_count(q) <= 12


def _validate_queries_for_target(queries: List[str], target: str) -> Tuple[List[str], List[Dict[str, str]]]:
    valid: List[str] = []
    issues: List[Dict[str, str]] = []
    seen: set[str] = set()
    for q in queries:
        qq = (q or "").strip()
        if not qq:
            continue
        if qq in seen:
            continue
        seen.add(qq)
        if target == "scholar_google_keywords" and not _is_short_keyword_query(qq):
            issues.append({"query": qq, "reason": "not_short_keyword_phrase"})
            continue
        valid.append(qq)
    return valid, issues


def _repair_queries_with_llm(
    state: DeepResearchState,
    section: SectionStatus,
    original_queries: List[str],
    target: str,
    max_queries: int,
) -> List[str]:
    """Ask LLM to repair invalid queries without changing intent."""
    client, model_override = _resolve_step_lite_client(state, "research")
    topic = state["dashboard"].brief.topic
    payload = "\n".join(f"- {q}" for q in original_queries)
    if target == "scholar_google_keywords":
        prompt = _pm.render(
            "repair_queries_scholar_google.txt",
            topic=topic,
            section_title=section.title,
            payload=payload,
            max_queries=max_queries,
        )
    else:
        prompt = _pm.render(
            "repair_queries_generic.txt",
            topic=topic,
            section_title=section.title,
            payload=payload,
            max_queries=max_queries,
        )
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are a search query rewriting assistant. Return plain lines only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,

        )
        raw = (resp.get("final_text") or "").strip()
        repaired: List[str] = []
        seen: set[str] = set()
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(cleaned) <= 2:
                continue
            # Apply stopword filter so repaired queries are also keyword-clean.
            if target == "scholar_google_keywords" and not _topic_is_chinese(cleaned):
                cleaned = _extract_en_keywords(cleaned, max_terms=10)
            if len(cleaned) <= 2 or cleaned in seen:
                continue
            seen.add(cleaned)
            repaired.append(cleaned)
            if len(repaired) >= max_queries:
                break
        return repaired
    except Exception:
        return []


def _fallback_short_query(topic: str, section_title: str, max_terms: int = 8) -> str:
    """Best-effort fallback query when LLM output/repair is unavailable."""
    src = f"{topic} {section_title}".strip()
    kw = _extract_en_keywords(src, max_terms=max_terms)
    if kw:
        return kw
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", src)
    if zh_terms:
        return " ".join(zh_terms[:max_terms])
    return (section_title or topic or "research evidence").strip()


# ── Engine classification for per-provider query building ──
_KEYWORD_ENGINES = frozenset({"scholar", "google", "semantic", "ncbi"})
_NL_ENGINES = frozenset({"tavily"})

# ── Academic engines that return structured abstracts via API (no page fetch needed) ──
_API_STRUCTURED_PROVIDERS = frozenset({"ncbi", "semantic"})

# ── Stopwords for academic keyword query cleaning ──────────────────────────────
# These tokens add noise when sent to keyword-based search engines (Scholar,
# Semantic, NCBI, Google).  They are stripped by _extract_en_keywords and the
# downstream post-processing steps so that only content-bearing terms remain.
_SEARCH_STOPWORDS: frozenset = frozenset({
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions
    "of", "to", "in", "for", "on", "with", "by", "at", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "under", "over", "about", "against", "between", "among",
    "along", "within", "without", "toward", "towards", "upon", "across",
    "behind", "beyond", "per",
    # Pronouns
    "it", "its", "they", "them", "their", "we", "our", "he", "she",
    "his", "her", "who", "whom", "whose",
    # Interrogative / relative
    "which", "what", "where", "when", "how", "why",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet", "if", "then", "than",
    "because", "since", "whether", "although", "though",
    # Copulas / auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    # Common filler verbs — useless as standalone search tokens
    "enable", "enables", "enabled", "enabling",
    "show", "shows", "showed", "shown", "showing",
    "investigate", "investigates", "investigated", "investigating",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "study", "studies", "studied", "studying",
    "explore", "explores", "explored", "exploring",
    "reveal", "reveals", "revealed", "revealing",
    "suggest", "suggests", "suggested", "suggesting",
    "indicate", "indicates", "indicated", "indicating",
    "provide", "provides", "provided", "providing",
    "involve", "involves", "involved", "involving",
    "determine", "determines", "determined", "determining",
    "examine", "examines", "examined", "examining",
    "describe", "describes", "described", "describing",
    "discuss", "discusses", "discussed", "discussing",
    "report", "reports", "reported", "reporting",
    "identify", "identifies", "identified", "identifying",
    "analyze", "analyzes", "analysed", "analyzed", "analysing", "analyzing",
    "affect", "affects", "affected", "affecting",
    "relate", "relates", "related", "relating",
    "concern", "concerns", "concerned", "concerning",
    "consider", "considers", "considered", "considering",
    "require", "requires", "required", "requiring",
    "include", "includes", "included", "including",
    "occur", "occurs", "occurred", "occurring",
    "use", "uses", "used", "using",
    "make", "makes", "made", "making",
    "find", "finds", "found", "finding",
    "get", "gets", "getting", "got",
    "give", "gives", "gave", "given", "giving",
    "see", "sees", "saw", "seen", "seeing",
    "know", "knows", "knew", "known", "knowing",
    "take", "takes", "took", "taken", "taking",
    # Generic academic adjectives / adverbs — rarely add search value
    "based", "recent", "new", "novel", "key", "role", "via",
    "also", "however", "therefore", "moreover", "furthermore",
    "thus", "hence", "various", "several", "certain",
    "specific", "particular", "general", "overall",
    "mainly", "primarily", "especially", "especially", "specifically",
    "important", "significant", "potential", "possible", "possible",
    "different", "similar", "common", "such", "other", "both",
    "many", "much", "more", "most", "less", "least",
    "large", "small", "high", "low", "long", "short",
    "further", "previous", "current", "present", "known",
    # Stopword-like numbers and single chars are already excluded by len > 1
    # but include common abbreviation noise
    "et", "al", "eg", "ie", "vs",
})


def _extract_en_keywords(text: str, max_terms: int = 10) -> str:
    """Extract English keyword tokens from mixed text, filtering out stopwords."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+/]*", text or "")
    filtered = [t for t in tokens if t.lower() not in _SEARCH_STOPWORDS and len(t) > 1]
    return " ".join(filtered[:max_terms])


def _extract_zh_keywords(text: str, max_terms: int = 8) -> str:
    """Extract Chinese multi-char terms from mixed text."""
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
    return " ".join(zh_terms[:max_terms])


_ACADEMIC_ENGINES = frozenset({"ncbi", "semantic", "scholar"})
_GENERAL_KEYWORD_ENGINES = frozenset({"google"})


def _build_write_queries(
    topic: str,
    section_title: str,
    extra_kw_suffix: str = "",
) -> Dict[str, List[str]]:
    """Build per-provider queries for write / claims / verification stages.

    Engine-aware language routing:
    - Academic engines (ncbi, semantic, scholar): English-only keyword phrases.
      These indices are almost exclusively English; Chinese queries return near-zero results.
    - General keyword engines (google): English-only keyword phrases.
    - NL engines (tavily): For Chinese topics, both Chinese and English NL questions.

    No LLM call — pure extraction.
    """
    src = f"{topic} {section_title}"
    is_zh = _topic_is_chinese(topic)

    # ── English keyword query (for all keyword engines) ──
    en_kw = _extract_en_keywords(src, max_terms=10)
    if extra_kw_suffix:
        en_kw = f"{en_kw} {extra_kw_suffix}".strip()
    if not en_kw:
        en_kw = (section_title or topic or "research")[:80]

    # ── NL queries ──
    if is_zh:
        short_title = section_title[:40] if len(section_title) > 40 else section_title
        nl_queries = [
            f"关于{short_title}的最新研究进展",
            f"What are the key findings on {_extract_en_keywords(section_title, 6)}?",
        ]
    else:
        nl_queries = [
            f"What are the key findings on {_extract_en_keywords(section_title, 8)}?",
        ]

    result: Dict[str, List[str]] = {}
    # Academic + general keyword engines: English only
    for engine in (_ACADEMIC_ENGINES | _GENERAL_KEYWORD_ENGINES):
        result[engine] = [en_kw]
    # NL engines: bilingual for Chinese topics, English-only otherwise
    for engine in _NL_ENGINES:
        result[engine] = list(nl_queries)
    return result


def _log_query_pack(event: str, data: Dict[str, Any]) -> None:
    try:
        logger.info("%s %s", event, json.dumps(data, ensure_ascii=False))
    except Exception:
        logger.info("%s %s", event, data)


class _ScopingResponse(BaseModel):
    scope: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    key_questions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    time_range: str = ""
    source_priority: List[str] = Field(default_factory=list)
    topic_domain: str = "general"


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
#   max_section_research_rounds per-section research→evaluate loop cap
#   max_verify_rewrite_cycles   max times verify-SEVERE can send a section back to research
#                               global cap = (research_rounds + rewrite_cycles) × num_sections
#
# Coverage:
#   coverage_threshold          minimum coverage_score before moving to write
#
# Queries — split into recall (broad, synonym) + precision (specific, constrained):
#   recall_queries_per_section  broad synonym/variant queries per round
#   precision_queries_per_section  narrow method/time/object-constrained queries per round
#   (total queries = recall + precision + gap queries)
#
# Tiered search ceilings (per round type):
#   round1_max_tier             max tier allowed in Round 1 (Lite=2, Comprehensive=3)
#   gapfill_max_tier            max tier in gap-fill rounds 2..N-1 (Lite=2, Comprehensive=3)
#   last_round_max_tier         max tier in the last allowed round (always 3 — final escalation)
#   default_per_provider_top_k  fallback top_k when UI source_configs doesn't specify a provider
#
# Write/verification top_k (independent of research-round top_k, UI cannot override):
#   search_top_k_write          write-node final evidence retrieval
#   search_top_k_write_max      hard cap for adaptive write retrieval window
#   verification_k              write-stage secondary evidence check for data-point citations
#
# Tier-3 refined queries:
#   tier3_refined_queries       number of LLM-refined queries generated for T3
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
        "max_section_research_rounds": 3,
        "max_verify_rewrite_cycles": 1,          # global = (3+1) × num_sections
        # ── Coverage ──
        "coverage_threshold": 0.60,
        # ── Queries (recall + precision) ──
        "recall_queries_per_section": 2,
        "precision_queries_per_section": 2,      # total = 4 + gaps
        # ── Tiered search ceilings ──
        "round1_max_tier": 2,                    # Round 1: T1→T2 only (skip slow Playwright)
        "gapfill_max_tier": 2,                   # Gap-fill rounds 2..N-1: T1→T2 only
        "last_round_max_tier": 3,                # Last round: T1→T2→T3 (final escalation)
        "default_per_provider_top_k": 10,        # fallback when UI doesn't specify topK
        # ── Write/verification top_k ──
        "search_top_k_write": 10,
        "search_top_k_write_max": 40,
        "verification_k": 12,
        # ── Tier-3 refined queries ──
        "tier3_refined_queries": 1,
        # ── Coverage plateau ──
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
        "max_section_research_rounds": 5,        # allows a 5th round for near-complete gaps
        "max_verify_rewrite_cycles": 2,          # global = (5+2) × num_sections
        # ── Coverage ──
        "coverage_threshold": 0.80,
        # ── Queries (recall + precision) ──
        "recall_queries_per_section": 4,
        "precision_queries_per_section": 4,      # total = 8 + gaps
        # ── Tiered search ceilings ──
        "round1_max_tier": 3,                    # Round 1: T1→T2→T3 (full chain from start)
        "gapfill_max_tier": 3,                   # Every gap-fill round: T1→T2→T3
        "last_round_max_tier": 3,                # Last round: T1→T2→T3
        "default_per_provider_top_k": 15,        # fallback when UI doesn't specify topK
        # ── Write/verification top_k ──
        "search_top_k_write": 12,
        "search_top_k_write_max": 60,
        "verification_k": 16,
        # ── Tier-3 refined queries ──
        "tier3_refined_queries": 2,
        # ── Coverage plateau ──
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
    write_top_k: Optional[int]
    current_section: str
    sections_completed: List[str]
    markdown_parts: List[str]
    citations: List[Any]
    evidence_chunks: List[Any]  # 运行期累计的 EvidenceChunk（用于 hash->cite_key 后处理）
    section_evidence_pool: Dict[str, List[Any]]  # 按章节隔离的完整证据池（保留 text/raw_content）
    evidence_chunk_empty_value: Any  # state 紧凑化时用于覆盖 text/raw_content 的值（默认 ""）
    citation_doc_key_map: Dict[str, str]  # doc_group_key -> cite_key（跨阶段保持稳定）
    citation_existing_keys: List[str]  # 已分配 cite_key（用于 numeric/hash/author_date 去重）
    iteration_count: int
    max_iterations: int
    max_sections: int
    runtime_id: str
    llm_client: Any
    model_override: Optional[str]
    output_language: str
    clarification_answers: Dict[str, str]
    preliminary_knowledge: str
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
    executed_queries: Dict[str, List[str]]  # per-section executed query signatures (provider+query)
    gap_semantic_query_maps: Dict[str, Dict[str, str]]  # normalized semantic query -> {relevance_query, bulk_query}
    query_chunk_cache: Dict[str, Dict[str, Any]]  # legacy fallback cache for no-runtime paths
    last_cost_tick_step: int
    error: Optional[str]


def _emit_progress(state: DeepResearchState, event_type: str, payload: Dict[str, Any]) -> None:
    """Emit optional progress callbacks for SSE integration."""
    cb = _resolve_runtime_callback(state, "progress_callback")
    if not cb:
        return
    try:
        cb(event_type, payload)
    except Exception:
        logger.debug("Progress callback failed", exc_info=True)


def _build_dashboard_from_dict(data: Dict[str, Any], fallback_topic: str = "") -> ResearchDashboard:
    dashboard = ResearchDashboard()
    data = data if isinstance(data, dict) else {}
    sections = data.get("sections") if isinstance(data.get("sections"), list) else []
    dashboard.sections = []
    for raw in sections:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        if not title:
            continue
        dashboard.sections.append(
            SectionStatus(
                title=title,
                status=str(raw.get("status") or "pending"),
                coverage_score=float(raw.get("coverage_score") or 0.0),
                source_count=int(raw.get("source_count") or 0),
                gaps=[str(g) for g in (raw.get("gaps") or []) if str(g).strip()],
                research_rounds=int(raw.get("research_rounds") or 0),
                evidence_scarce=bool(raw.get("evidence_scarce", False)),
                verify_rewrite_count=int(raw.get("verify_rewrite_count") or 0),
            )
        )
    dashboard.overall_confidence = str(data.get("confidence") or "low")
    dashboard.total_sources = int(data.get("total_sources") or 0)
    dashboard.total_iterations = int(data.get("total_iterations") or 0)
    dashboard.coverage_gaps = [str(g) for g in (data.get("coverage_gaps") or []) if str(g).strip()]
    dashboard.conflict_notes = [str(c) for c in (data.get("conflict_notes") or []) if str(c).strip()]
    dashboard.brief = ResearchBrief(
        topic=str(data.get("topic") or fallback_topic),
        scope=str(data.get("scope") or ""),
    )
    return dashboard


def _build_trajectory_for_dashboard(topic: str, dashboard: ResearchDashboard) -> ResearchTrajectory:
    trajectory = ResearchTrajectory(topic=topic)
    for idx, sec in enumerate(dashboard.sections):
        trajectory.add_branch(f"sec_{idx+1}", sec.title)
    return trajectory


def _serializable_checkpoint_state(state: DeepResearchState) -> Dict[str, Any]:
    dashboard = state.get("dashboard")
    dashboard_data = dashboard.to_dict() if isinstance(dashboard, ResearchDashboard) else {}
    payload: Dict[str, Any] = {
        "topic": str(state.get("topic") or ""),
        "canvas_id": str(state.get("canvas_id") or ""),
        "session_id": str(state.get("session_id") or ""),
        "user_id": str(state.get("user_id") or ""),
        "runtime_id": str(state.get("runtime_id") or ""),
        "search_mode": str(state.get("search_mode") or "hybrid"),
        "filters": dict(state.get("filters") or {}),
        "current_section": str(state.get("current_section") or ""),
        "sections_completed": list(state.get("sections_completed") or []),
        "markdown_parts": list(state.get("markdown_parts") or []),
        "citations": list(state.get("citations") or []),
        "iteration_count": int(state.get("iteration_count") or 0),
        "max_iterations": int(state.get("max_iterations") or 0),
        "output_language": str(state.get("output_language") or "auto"),
        "clarification_answers": dict(state.get("clarification_answers") or {}),
        "preliminary_knowledge": str(state.get("preliminary_knowledge") or ""),
        "user_context": str(state.get("user_context") or ""),
        "user_context_mode": str(state.get("user_context_mode") or "supporting"),
        "user_documents": list(state.get("user_documents") or []),
        "step_models": dict(state.get("step_models") or {}),
        "step_model_strict": bool(state.get("step_model_strict")),
        "skip_draft_review": bool(state.get("skip_draft_review")),
        "skip_refine_review": bool(state.get("skip_refine_review")),
        "skip_claim_generation": bool(state.get("skip_claim_generation")),
        "verified_claims": str(state.get("verified_claims") or ""),
        "review_gate_next": str(state.get("review_gate_next") or "review_gate"),
        "review_handled_at": dict(state.get("review_handled_at") or {}),
        "depth": str(state.get("depth") or DEFAULT_DEPTH),
        "depth_preset": dict(state.get("depth_preset") or {}),
        "review_gate_rounds": int(state.get("review_gate_rounds") or 0),
        "review_gate_unchanged": int(state.get("review_gate_unchanged") or 0),
        "review_gate_last_snapshot": str(state.get("review_gate_last_snapshot") or ""),
        "revision_queue": list(state.get("revision_queue") or []),
        "review_seen_at": dict(state.get("review_seen_at") or {}),
        "job_id": str(state.get("job_id") or ""),
        "graph_step_count": int(state.get("graph_step_count") or 0),
        "cost_warned": bool(state.get("cost_warned")),
        "force_synthesize": bool(state.get("force_synthesize")),
        "coverage_history": dict(state.get("coverage_history") or {}),
        "last_cost_tick_step": int(state.get("last_cost_tick_step") or 0),
        "error": state.get("error"),
        "citation_doc_key_map": dict(state.get("citation_doc_key_map") or {}),
        "citation_existing_keys": list(state.get("citation_existing_keys") or []),
        "evidence_chunk_empty_value": state.get("evidence_chunk_empty_value", ""),
        "dashboard": dashboard_data,
    }
    return payload


def _save_phase_checkpoint(state: DeepResearchState, phase: str, section_title: str = "") -> None:
    job_id = str(state.get("job_id") or "").strip()
    if not job_id:
        return
    try:
        from src.collaboration.research.job_store import save_checkpoint
        save_checkpoint(
            job_id=job_id,
            phase=phase,
            section_title=section_title or "",
            state_dict=_serializable_checkpoint_state(state),
        )
    except Exception:
        logger.debug("save checkpoint failed phase=%s section=%s", phase, section_title, exc_info=True)


def _cleanup_shared_browser_if_no_active_jobs(current_job_id: str = "") -> None:
    """Close shared web-search browser only when no other DR jobs are active."""
    try:
        from sqlmodel import Session, select
        from src.db.engine import get_engine
        from src.db.models import DeepResearchJob

        with Session(get_engine()) as session:
            stmt = select(DeepResearchJob.job_id).where(
                DeepResearchJob.status.in_(["running", "cancelling"])
            )
            jid = str(current_job_id or "").strip()
            if jid:
                stmt = stmt.where(DeepResearchJob.job_id != jid)
            other_active = session.exec(stmt.limit(1)).first() is not None
        if other_active:
            logger.debug("Skip shared browser cleanup: other DR jobs still active")
            return

        from src.retrieval.google_search import cleanup_shared_browser_sync

        cleanup_shared_browser_sync()
    except Exception:
        logger.debug("Shared browser cleanup skipped due to error", exc_info=True)


def reconstruct_state_from_checkpoint(
    *,
    checkpoint_data: Dict[str, Any],
    llm_client: Any,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    review_waiter: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    model_override: Optional[str] = None,
) -> DeepResearchState:
    state_data = checkpoint_data.get("state") if isinstance(checkpoint_data, dict) else {}
    state_data = state_data if isinstance(state_data, dict) else {}
    topic = str(state_data.get("topic") or "")
    dashboard_data = state_data.get("dashboard") if isinstance(state_data.get("dashboard"), dict) else {}
    dashboard = _build_dashboard_from_dict(dashboard_data, fallback_topic=topic)
    trajectory = _build_trajectory_for_dashboard(topic or dashboard.brief.topic, dashboard)
    resolved_depth = str(state_data.get("depth") or DEFAULT_DEPTH)
    if resolved_depth not in DEPTH_PRESETS:
        resolved_depth = DEFAULT_DEPTH
    reconstructed: DeepResearchState = {
        "topic": topic or dashboard.brief.topic,
        "dashboard": dashboard,
        "trajectory": trajectory,
        "canvas_id": str(state_data.get("canvas_id") or ""),
        "session_id": str(state_data.get("session_id") or ""),
        "user_id": str(state_data.get("user_id") or ""),
        "search_mode": str(state_data.get("search_mode") or "hybrid"),
        "filters": dict(state_data.get("filters") or {}),
        "write_top_k": state_data.get("write_top_k"),
        "current_section": str(state_data.get("current_section") or ""),
        "sections_completed": list(state_data.get("sections_completed") or []),
        "markdown_parts": list(state_data.get("markdown_parts") or []),
        "citations": list(state_data.get("citations") or []),
        "evidence_chunks": [],
        "section_evidence_pool": dict(state_data.get("section_evidence_pool") or {}),
        "evidence_chunk_empty_value": state_data.get("evidence_chunk_empty_value", ""),
        "citation_doc_key_map": dict(state_data.get("citation_doc_key_map") or {}),
        "citation_existing_keys": list(state_data.get("citation_existing_keys") or []),
        "iteration_count": int(state_data.get("iteration_count") or 0),
        "max_iterations": int(state_data.get("max_iterations") or max(1, len(dashboard.sections) * 5)),
        "runtime_id": str(state_data.get("runtime_id") or state_data.get("job_id") or ""),
        "model_override": model_override if model_override is not None else state_data.get("model_override"),
        "output_language": str(state_data.get("output_language") or "auto"),
        "clarification_answers": dict(state_data.get("clarification_answers") or {}),
        "preliminary_knowledge": str(state_data.get("preliminary_knowledge") or ""),
        "user_context": str(state_data.get("user_context") or ""),
        "user_context_mode": str(state_data.get("user_context_mode") or "supporting"),
        "user_documents": list(state_data.get("user_documents") or []),
        "step_models": dict(state_data.get("step_models") or {}),
        "step_model_strict": bool(state_data.get("step_model_strict")),
        "progress_callback": progress_callback,
        "cancel_check": cancel_check,
        "review_waiter": review_waiter,
        "skip_draft_review": bool(state_data.get("skip_draft_review")),
        "skip_refine_review": bool(state_data.get("skip_refine_review")),
        "skip_claim_generation": bool(state_data.get("skip_claim_generation")),
        "verified_claims": str(state_data.get("verified_claims") or ""),
        "review_gate_next": str(state_data.get("review_gate_next") or "review_gate"),
        "review_handled_at": dict(state_data.get("review_handled_at") or {}),
        "depth": resolved_depth,
        "depth_preset": dict(state_data.get("depth_preset") or get_depth_preset(resolved_depth)),
        "review_gate_rounds": int(state_data.get("review_gate_rounds") or 0),
        "review_gate_unchanged": int(state_data.get("review_gate_unchanged") or 0),
        "review_gate_last_snapshot": str(state_data.get("review_gate_last_snapshot") or ""),
        "revision_queue": list(state_data.get("revision_queue") or []),
        "review_seen_at": dict(state_data.get("review_seen_at") or {}),
        "job_id": str(state_data.get("job_id") or ""),
        "graph_step_count": int(state_data.get("graph_step_count") or 0),
        "cost_warned": bool(state_data.get("cost_warned")),
        "force_synthesize": bool(state_data.get("force_synthesize")),
        "coverage_history": dict(state_data.get("coverage_history") or {}),
        "last_cost_tick_step": int(state_data.get("last_cost_tick_step") or 0),
        "error": state_data.get("error"),
    }
    return reconstructed


def _ensure_not_cancelled(state: DeepResearchState) -> None:
    """Cooperative cancellation checkpoint."""
    checker = _resolve_runtime_callback(state, "cancel_check")
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


def _normalize_max_sections(value: Any, default: int = 4) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return max(2, min(6, n))


def _parse_outline_sections(raw: str, topic: str) -> List[str]:
    """Parse outline sections from LLM output, handling multiple common formats.

    Supported formats:
      - Numbered list:       ``1. Title`` / ``1) Title`` / ``1、Title``
      - Bullet list:         ``- Title`` / ``* Title``
      - Markdown headers:    ``## Title`` / ``### 1. Title``
      - Mixed bold markers:  ``1. **Title**``
    """
    if not raw or not (raw or "").strip():
        logger.warning(
            "[plan_node] LLM returned empty outline text, using default sections."
        )
        return [
            f"Overview of {topic}",
            "Research progress",
            "Key findings",
            "Conclusions and outlook",
        ]

    _NUMBERED = re.compile(
        r"^(?:#{1,4}\s*)?[\d]+[\.\)、]\s*(.+)$"
    )
    _BULLET = re.compile(
        r"^[-\*]\s+(.+)$"
    )
    _MD_HEADER = re.compile(
        r"^#{1,4}\s+(.+)$"
    )

    sections: List[str] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        title: Optional[str] = None
        for pat in (_NUMBERED, _BULLET, _MD_HEADER):
            m = pat.match(line)
            if m:
                title = m.group(1).strip()
                break
        if title:
            title = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", title)
            title = title.strip(":： ")
            if title:
                sections.append(title)

    if not sections:
        logger.warning(
            "[plan_node] Failed to parse outline from LLM response, "
            "falling back to default. Raw response:\n%s", raw[:1000],
        )
        sections = [
            f"Overview of {topic}",
            "Research progress",
            "Key findings",
            "Conclusions and outlook",
        ]

    return sections


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
    default_client = _resolve_runtime_llm_client(state)
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


def _resolve_step_lite_client(state: DeepResearchState, step: str) -> Tuple[Any, Optional[str]]:
    """Like _resolve_step_client_and_model but auto-downgrades thinking providers.

    Used for lightweight tasks (query generation, JSON parsing, coverage checks)
    where extended thinking wastes tokens without improving quality.
    """
    client, model_override = _resolve_step_client_and_model(state, step)
    try:
        from src.llm.llm_manager import get_manager
        manager = get_manager(str(_CONFIG_PATH))
        provider_name = state.get("llm_provider") or ""
        step_models = state.get("step_models") or {}
        step_val = (step_models.get(step) or "").strip()
        if step_val and "::" in step_val:
            provider_name = step_val.split("::")[0].strip()
        if provider_name:
            lite_client = manager.get_lite_client(provider_name)
            return lite_client, model_override
    except Exception:
        pass
    return client, model_override


_DASHBOARD_TO_CANVAS_STATUS = {
    "pending": "todo",
    "researching": "todo",
    "writing": "drafting",
    "reviewing": "drafting",
    "done": "done",
}


def _sync_outline_status_to_canvas(state: DeepResearchState) -> None:
    """Push current dashboard section statuses to the Canvas outline model."""
    canvas_id = state.get("canvas_id")
    if not canvas_id:
        return
    dashboard = state.get("dashboard")
    if not isinstance(dashboard, ResearchDashboard):
        return
    try:
        from src.collaboration.canvas.canvas_manager import get_canvas, upsert_outline
        canvas = get_canvas(canvas_id)
        if canvas is None:
            return
        title_to_status = {
            s.title: _DASHBOARD_TO_CANVAS_STATUS.get(s.status, "todo")
            for s in dashboard.sections
        }
        changed = False
        for cs in canvas.outline:
            new_status = title_to_status.get(cs.title)
            if new_status and cs.status != new_status:
                cs.status = new_status
                changed = True
        if changed:
            upsert_outline(canvas_id, canvas.outline)
    except Exception:
        logger.debug("Failed to sync outline status to canvas", exc_info=True)


def _language_instruction(state: DeepResearchState) -> str:
    """Generate explicit language instruction based on output_language setting.

    Priority:
    1. If output_language is 'zh' or 'en', use that explicitly
    2. If output_language is 'auto', detect from topic language
    3. Default to English if detection fails
    """
    lang = (state.get("output_language") or "auto").strip().lower()
    if lang == "zh":
        return "\n\nIMPORTANT: Write the output in Chinese (中文)."
    if lang == "en":
        return "\n\nIMPORTANT: Write the output in English."
    # auto mode: detect from topic
    dashboard = state.get("dashboard")
    topic = dashboard.brief.topic if dashboard else ""
    is_zh = bool(re.search(r"[\u4e00-\u9fff]", topic or ""))
    if is_zh:
        return "\n\nIMPORTANT: Write the output in Chinese (中文)."
    return "\n\nIMPORTANT: Write the output in English."


def _build_user_context_block(state: DeepResearchState, max_chars: int = DR_USER_CONTEXT_MAX_CHARS) -> str:
    """Build optional user-supplied temporary context block. Over max_chars (default 40k) is summarized via ultra_lite."""
    chunks: List[str] = []
    prelim = (state.get("preliminary_knowledge") or "").strip()
    if prelim:
        chunks.append(
            "Preliminary Research Overview (background knowledge from web, treat as starting context):\n" + prelim
        )
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
    ultra_lite_provider = (state.get("filters") or {}).get("ultra_lite_provider")
    text = summarize_if_needed(
        text, max_chars,
        ultra_lite_provider=ultra_lite_provider,
        purpose="dr_user_context",
    )
    return "\n\nAdditional temporary context:\n" + text


def _compute_effective_write_k(preset: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> int:
    """Compute per-section write evidence window with a safety cap.

    write_top_k = per-output-unit evidence cap (one DR section = one unit).
    If explicitly provided, use it (capped). Otherwise derive from step_top_k * 1.5 or preset.
    """
    preset_write_k = int(preset.get("search_top_k_write", 12))
    write_k_cap = int(preset.get("search_top_k_write_max", 60))
    if write_k_cap <= 0:
        write_k_cap = 60
    write_k_cap = max(write_k_cap, preset_write_k)

    ui_write_k = 0
    try:
        ui_write_k = int((filters or {}).get("write_top_k") or 0)
    except (TypeError, ValueError):
        ui_write_k = 0

    if ui_write_k > 0:
        return min(max(preset_write_k, ui_write_k), write_k_cap)

    ui_step_k = 0
    try:
        ui_step_k = int((filters or {}).get("step_top_k") or 0)
    except (TypeError, ValueError):
        ui_step_k = 0

    effective_write_k = max(preset_write_k, int(ui_step_k * 1.5)) if ui_step_k > 0 else preset_write_k
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


def _accumulate_section_pool(
    state: DeepResearchState,
    section_title: str,
    chunks: List[Any],
    *,
    pool_source: str = "research_round",
) -> None:
    """Accumulate full-text evidence per section (dedup by chunk_id)."""
    if not section_title or not chunks:
        return
    section_pool = state.setdefault("section_evidence_pool", {})
    existing = section_pool.setdefault(section_title, [])
    seen_ids = set()
    for c in existing:
        if isinstance(c, dict):
            cid = str(c.get("chunk_id") or "")
        else:
            cid = str(getattr(c, "chunk_id", "") or "")
        if cid:
            seen_ids.add(cid)
    for c in chunks:
        if isinstance(c, dict):
            cid = str(c.get("chunk_id") or "")
        else:
            cid = str(getattr(c, "chunk_id", "") or "")
        if cid and cid in seen_ids:
            continue
        pool_chunk = copy.copy(c)
        if isinstance(pool_chunk, dict):
            pool_chunk["pool_source"] = pool_source
        else:
            try:
                setattr(pool_chunk, "pool_source", pool_source)
            except Exception:
                pass
        existing.append(pool_chunk)
        if cid:
            seen_ids.add(cid)


# pool_source values that represent gap-targeted evidence; these candidates get
# a relevance score boost and guaranteed quota in the final top-k selection.
_DR_GAP_POOL_SOURCES: frozenset = frozenset({"eval_supplement"})


def _rerank_section_pool_chunks(
    query: str,
    pool_chunks: List[Any],
    top_k: int,
    *,
    reranker_mode: Optional[str] = None,
) -> List[Any]:
    """Rerank section pool chunks with gap-pool protection.

    Chunks whose pool_source is in _DR_GAP_POOL_SOURCES (currently
    "eval_supplement") are treated as gap candidates: they receive a relevance
    score boost and a guaranteed minimum quota in the final top-k output via
    fuse_pools_with_gap_protection.  All other chunks go into the main pool.

    Returns the top-k original chunk objects (EvidenceChunk or dict) ordered
    by fused relevance score.
    """
    if not pool_chunks:
        return []

    # Lazy imports — avoids circular dependencies at module load time
    try:
        from src.retrieval.service import fuse_pools_with_gap_protection
        _has_fusion = True
    except Exception:
        _has_fusion = False
    try:
        from src.retrieval.hybrid_retriever import _rerank_candidates
        _has_reranker = True
    except Exception:
        _has_reranker = False

    if not _has_fusion and not _has_reranker:
        return list(pool_chunks[:top_k])

    main_candidates: List[Dict[str, Any]] = []
    gap_candidates: List[Dict[str, Any]] = []
    by_chunk_id: Dict[str, Any] = {}

    for idx, chunk in enumerate(pool_chunks):
        text = str(chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", "") or "")
        if not text.strip():
            continue
        if isinstance(chunk, dict):
            chunk_id = str(chunk.get("chunk_id") or f"pool::{idx}")
            doc_id = str(chunk.get("doc_id") or "")
            doc_title = chunk.get("doc_title")
            year = chunk.get("year")
            url = chunk.get("url")
            doi = chunk.get("doi")
            provider = chunk.get("provider")
            score = float(chunk.get("score", 0.0) or 0.0)
            source_type = str(chunk.get("source_type", "") or "")
            pool_source = str(chunk.get("pool_source", "") or "")
        else:
            chunk_id = str(getattr(chunk, "chunk_id", "") or f"pool::{idx}")
            doc_id = str(getattr(chunk, "doc_id", "") or "")
            doc_title = getattr(chunk, "doc_title", None)
            year = getattr(chunk, "year", None)
            url = getattr(chunk, "url", None)
            doi = getattr(chunk, "doi", None)
            provider = getattr(chunk, "provider", None)
            score = float(getattr(chunk, "score", 0.0) or 0.0)
            source_type = str(getattr(chunk, "source_type", "") or "")
            pool_source = str(getattr(chunk, "pool_source", "") or "")

        meta = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "paper_id": doc_id,
            "title": doc_title,
            "year": year,
            "url": url,
            "doi": doi,
            "provider": provider,
            "score": score,
        }
        cand = {
            "chunk_id": chunk_id,
            "content": text,
            "score": score,
            "source": "web" if source_type == "web" else "dense",
            "metadata": meta,
        }
        if pool_source in _DR_GAP_POOL_SOURCES:
            gap_candidates.append(cand)
        else:
            main_candidates.append(cand)
        by_chunk_id[chunk_id] = chunk

    all_candidates = main_candidates + gap_candidates
    if not all_candidates:
        return list(pool_chunks[:top_k])

    # Global fusion with gap protection
    fused: List[Dict[str, Any]] = []
    if _has_fusion:
        try:
            fused = fuse_pools_with_gap_protection(
                query=query,
                main_candidates=main_candidates,
                gap_candidates=gap_candidates,
                top_k=min(max(top_k, 1), len(all_candidates)),
                gap_ratio=float(getattr(settings.search, "research_gap_ratio", 0.25)),
                rank_pool_multiplier=float(getattr(settings.search, "research_rank_pool_multiplier", 3.0)),
                reranker_mode=reranker_mode,
            )
        except Exception as e:
            logger.warning(
                "_rerank_section_pool_chunks fusion failed (%s); falling back to plain rerank", e
            )

    # Fallback: plain rerank without gap protection
    if not fused:
        if _has_reranker:
            try:
                fused = _rerank_candidates(
                    query=query,
                    candidates=all_candidates,
                    top_k=min(max(top_k, 1), len(all_candidates)),
                    reranker_mode=reranker_mode,
                )
            except Exception:
                fused = sorted(
                    all_candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True
                )
        else:
            fused = all_candidates

    # Map fused raw dicts back to original chunk objects
    out: List[Any] = []
    for item in fused:
        meta = item.get("metadata") or {}
        cid = str(item.get("chunk_id") or meta.get("chunk_id") or "")
        chunk = by_chunk_id.get(cid)
        if chunk is not None:
            out.append(chunk)
    return out[:top_k] if out else list(pool_chunks[:top_k])


def _build_pack_from_chunks(query: str, chunks: List[Any]) -> Any:
    """Build an EvidencePack from selected chunks."""
    from src.retrieval.evidence import EvidencePack

    source_types = {str(getattr(c, "source_type", "") or "") for c in chunks}
    sources_used: List[str] = []
    if any(x in {"dense", "sparse"} for x in source_types):
        sources_used.append("dense")
    if "graph" in source_types:
        sources_used.append("graph")
    if "web" in source_types:
        sources_used.append("web")
    if not sources_used and chunks:
        sources_used.append("hybrid")
    return EvidencePack(
        query=query,
        chunks=chunks,
        total_candidates=len(chunks),
        retrieval_time_ms=0.0,
        sources_used=sources_used,
    )


def _resolve_text_citations(
    state: DeepResearchState,
    text: str,
    chunks: List[Any],
    include_unreferenced_documents: bool = False,
) -> tuple[str, List[Any]]:
    """
    使用共享的 doc_key->cite_key 映射做 [ref:xxxx] 占位符替换，确保跨阶段引用键稳定。
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


# ---------------------------------------------------------------------------
# Distinct-document counter (module-level so it can be used in helper fns)
# ---------------------------------------------------------------------------

def _count_distinct_docs(chunks: List[Any]) -> int:
    """Count unique source documents in a list of evidence chunks."""
    keys: set = set()
    for c in chunks:
        if getattr(c, "doi", None):
            keys.add(f"doi:{c.doi}")
        else:
            keys.add(c.doc_group_key)
    return len(keys)


def _build_gap_evidence_summary(
    state: DeepResearchState,
    query: str,
    chunks: List[Any],
    *,
    max_chunks: int = 20,
    max_total_chars: int = 80000,
    per_chunk_max_chars: int = 3000,
    compress_step: str = "evaluate",
) -> str:
    """
    Build evidence summary for gap/refined-query prompts with per-chunk length control.

    - Short chunks: keep original (light truncation at per_chunk_max_chars).
    - Long chunks: optionally compress with a cheap (lite) model before joining.
    - Final output: hard capped by max_total_chars.
    """
    if not chunks:
        return "(none)"

    cfg = getattr(settings, "content_fetcher", None)
    enable_compress = bool(getattr(cfg, "compress_long_fulltext", True))
    word_threshold = int(getattr(cfg, "compress_word_threshold", 300))
    max_output_words = int(getattr(cfg, "compress_max_output_words", 400))
    compress_client = None
    if enable_compress:
        try:
            from src.llm.llm_manager import get_manager
            manager = get_manager(str(_CONFIG_PATH))
            ultra_lite_provider = state.get("filters", {}).get("ultra_lite_provider")
            compress_client = manager.get_ultra_lite_client(ultra_lite_provider)
        except Exception:
            compress_client = None

    lines: List[str] = []
    compressed_chunks = 0
    truncated_chunks = 0
    input_chars = 0
    total_chars = 0
    for i, c in enumerate(chunks[:max_chunks]):
        title = getattr(c, "doc_title", "") or getattr(c, "title", "") or ""
        text = (getattr(c, "text", "") or "").strip()
        if not text:
            continue
        input_chars += len(text)

        # Long evidence is compressed to preserve substantive signal while avoiding 80k bloat.
        if (
            compress_client is not None
            and len(text) > per_chunk_max_chars
            and len(text.split()) > word_threshold
        ):
            text = compress_evidence_text_sync(
                text,
                query,
                compress_client,
                title=title,
                url=getattr(c, "url", "") or "",
                max_output_words=max_output_words,
                fallback_chars=per_chunk_max_chars,
            )
            compressed_chunks += 1
        elif len(text) > per_chunk_max_chars:
            text = text[:per_chunk_max_chars] + "..."
            truncated_chunks += 1

        ref = getattr(c, "ref_hash", f"ref:{i+1}")
        line = f"[{ref}] {title} — {text}" if title else f"[{ref}] {text}"

        if total_chars + len(line) + 1 > max_total_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1

    if not lines:
        return "(none)"
    out = "\n".join(lines)[:max_total_chars]
    logger.debug(
        "[gap_evidence_summary] step=%s chunks_in=%d chunks_used=%d compressed=%d truncated=%d "
        "input_chars=%d output_chars=%d max_total_chars=%d query=%r",
        compress_step,
        min(len(chunks), max_chunks),
        len(lines),
        compressed_chunks,
        truncated_chunks,
        input_chars,
        len(out),
        max_total_chars,
        (query or "")[:120],
    )
    return out


def _generate_refined_queries(
    state: DeepResearchState,
    section: SectionStatus,
    collected_chunks: List[Any],
    max_queries: int = 2,
) -> List[str]:
    """Generate 1-N refined keyword queries for Tier-3 (scholar/google) from existing evidence.

    Uses the LLM to identify the biggest remaining information gap and produce
    short keyword phrases suitable for Google Scholar / Google Search.
    Falls back to a simple topic+section query if the LLM call fails.
    """
    dashboard = state["dashboard"]
    topic = dashboard.brief.topic
    client, model_override = _resolve_step_lite_client(state, "research")

    # Build evidence summary from real retrieved evidence; fallback to collected_chunks.
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    eval_top_k = int(state.get("filters", {}).get("step_top_k") or preset.get("search_top_k_eval", 20))
    evidence_summary = ""
    try:
        svc = _get_retrieval_svc(state)
        ref_filters = dict(state.get("filters") or {})
        ref_filters["use_query_optimizer"] = False
        ref_filters["reranker_mode"] = "bge_only"
        pack = svc.search(
            query=f"{topic} {section.title}",
            mode=state.get("search_mode", "hybrid"),
            top_k=eval_top_k,
            filters=ref_filters,
        )
        evidence_summary = _build_gap_evidence_summary(
            state,
            query=f"{topic} {section.title} {'; '.join(section.gaps or [])}",
            chunks=list(pack.chunks),
            max_chunks=eval_top_k,
            max_total_chars=80000,
            per_chunk_max_chars=3000,
            compress_step="research",
        )
    except Exception:
        pass

    if not evidence_summary.strip():
        evidence_summary = _build_gap_evidence_summary(
            state,
            query=f"{topic} {section.title} {'; '.join(section.gaps or [])}",
            chunks=collected_chunks,
            max_chunks=20,
            max_total_chars=80000,
            per_chunk_max_chars=3000,
            compress_step="research",
        )

    gaps_block = "\n".join(f"- {g}" for g in section.gaps) if section.gaps else "(none)"
    # T3 targets scholar + google — always English academic keywords even for Chinese topics
    english_only_instruction = _BILINGUAL_HINT_ACADEMIC if _topic_is_chinese(topic) else ""

    try:
        prompt = _pm.render(
            "generate_refined_queries.txt",
            topic=topic,
            section_title=section.title,
            gaps_block=gaps_block,
            evidence_summary=evidence_summary,
            english_only_instruction=english_only_instruction,
            max_queries=max_queries,
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": "Output search queries ONLY, one per line."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,

        )
        raw = (resp.get("final_text") or "").strip()
        parsed: List[str] = []
        seen: set[str] = set()
        for line in raw.split("\n"):
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if len(cleaned) <= 2:
                continue
            # Apply stopword filter to enforce keyword-only output regardless
            # of whether the LLM followed the prompt instructions.
            is_zh = _topic_is_chinese(cleaned)
            if not is_zh:
                cleaned = _extract_en_keywords(cleaned, max_terms=10)
            if len(cleaned) <= 2 or cleaned in seen:
                continue
            seen.add(cleaned)
            parsed.append(cleaned)
            if len(parsed) >= max_queries:
                break

        valid, issues = _validate_queries_for_target(parsed, "scholar_google_keywords")
        repaired: List[str] = []
        if issues:
            repaired = _repair_queries_with_llm(
                state=state,
                section=section,
                original_queries=[x["query"] for x in issues],
                target="scholar_google_keywords",
                max_queries=max_queries,
            )
            valid2, _ = _validate_queries_for_target(repaired, "scholar_google_keywords")
            valid = (valid + valid2)[:max_queries]

        if not valid:
            # Last-resort fallback with minimal assumptions; keep it short and engine-safe.
            valid = [_fallback_short_query(topic, section.title)]

        _log_query_pack(
            "query_pack_refined",
            {
                "section": section.title,
                "target": "scholar_google_keywords",
                "raw_queries": parsed,
                "validation_issues": issues,
                "repaired_queries": repaired,
                "final_queries": valid,
            },
        )
        return valid
    except Exception:
        fallback = [_fallback_short_query(topic, section.title)]
        _log_query_pack(
            "query_pack_refined_fallback",
            {"section": section.title, "target": "scholar_google_keywords", "final_queries": fallback},
        )
        return fallback


def _generate_engine_gap_queries(
    state: DeepResearchState,
    section: SectionStatus,
    gap: str,
) -> Dict[str, str]:
    """Generate per-engine gap queries via LLM using the engine-aware prompt.

    Calls the LLM with ``generate_gap_queries.txt`` which outputs a 9-key JSON,
    each value being an engine-optimised query string (PubMed syntax for ncbi,
    comma-separated keywords for semantic, NL question for tavily, etc.).

    Returns a dict mapping engine keys to query strings.  On any failure the
    dict is empty and the caller falls back to keyword extraction.

    Active routing keys: ncbi_pubmed, semantic_keywords, tavily,
                         google_scholar, google
    Future keys (logged but not routed): semantic_umbrella,
                         semantic_ai4scholar_relevance, semantic_ai4scholar_bulk
    """
    dashboard = state["dashboard"]
    topic = dashboard.brief.topic
    client, model_override = _resolve_step_client_and_model(state, "research")

    is_zh = _topic_is_chinese(topic)
    language_instruction = _BILINGUAL_HINT_GAP if is_zh else ""
    background = f"{topic} — {section.title}"
    intent = str((state.get("filters") or {}).get("gap_query_intent") or "broad").strip().lower()
    if intent not in {"broad", "review_pref", "reviews_only"}:
        intent = "broad"

    try:
        prompt = _pm.render(
            "generate_gap_queries.txt",
            topic=topic,
            section_title=section.title,
            background=background,
            gap=gap,
            intent=intent,
            language_instruction=language_instruction,
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": "Output a single-line JSON object with exactly 9 keys. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
        )
        raw = (resp.get("final_text") or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        result: Dict[str, str] = {}
        for k in _GAP_ENGINE_ALL_KEYS:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                result[k] = _truncate_query_text(v.strip(), _MAX_QUERY_CHARS)
        _log_query_pack("gap_engine_queries", {
            "section": section.title,
            "gap": gap[:100],
            "engine_queries": result,
        })
        return result
    except Exception as e:
        logger.debug("_generate_engine_gap_queries failed (%s); falling back to keyword extraction", e)
        return {}


def _generate_section_queries(
    state: DeepResearchState,
    section: SectionStatus,
    max_queries: int = 8,
) -> List[Tuple[str, str]]:
    """Generate section queries using a recall + precision + discovery + gap strategy.

    Returns a list of (query, category) tuples. Category tags:

      Engine-specific gap (Round 2+, LLM-generated per engine):
        "gap_ncbi"      — PubMed syntax → ncbi only
        "gap_semantic"  — keyword phrases → semantic only
        "gap_tavily"    — NL question → tavily only
        "gap_scholar"   — boolean query → scholar only
        "gap_google"    — plain search → google only
      Fallback gap (when engine-aware generation fails):
        "gap"           — EN keyword phrases → ncbi + semantic
        "gap_discovery" — NL queries → tavily
      LLM-generated (all rounds):
        "recall"        — short keyword phrases → ncbi/semantic
        "precision"     — long constrained keywords → ncbi/semantic
        "discovery"     — NL questions → tavily

    When the topic is in Chinese, bilingual instruction is injected into the prompt so
    the LLM splits each category's budget evenly between Chinese and English queries.
    English topics generate English-only queries (no Chinese supplement needed).
    """
    dashboard = state["dashboard"]
    topic = dashboard.brief.topic
    gaps = section.gaps or []
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    round_idx = int(section.research_rounds or 0)
    gap_only_mode = round_idx > 1

    recall_budget = int(preset.get("recall_queries_per_section", 2))
    precision_budget = int(preset.get("precision_queries_per_section", 2))
    discovery_budget = 1

    # ── Priority 1: gap-targeted queries (LLM engine-aware, with keyword fallback) ──
    # For each gap, call the engine-aware prompt to get per-engine optimised queries.
    # On LLM failure, fall back to keyword extraction (safe, always works).
    # Cross-section reuse/dedup is handled during execution via _run_search cache.
    is_topic_zh = _topic_is_chinese(topic)
    gap_engine_queries: List[Tuple[str, str]] = []
    gap_kw_queries: List[str] = []
    gap_nl_queries: List[str] = []
    semantic_query_maps: Dict[str, Dict[str, str]] = {}
    history_dedup_hits = 0
    gap_seen: set = set()
    for gap in gaps[:max(max_queries // 3, 2)]:
        engine_result = _generate_engine_gap_queries(state, section, gap)
        if engine_result:
            for key, tag in _GAP_ENGINE_TAG_MAP.items():
                q = _truncate_query_text(engine_result.get(key, "").strip(), _MAX_QUERY_CHARS)
                sig = (tag, q)
                if q and len(q) > 2 and sig not in gap_seen:
                    gap_engine_queries.append((q, tag))
                    if tag == "gap_semantic":
                        rel_q = _truncate_query_text(
                            str(engine_result.get("semantic_ai4scholar_relevance", "")).strip(),
                            _MAX_QUERY_CHARS,
                        )
                        bulk_q = _truncate_query_text(
                            str(engine_result.get("semantic_ai4scholar_bulk", "")).strip(),
                            _MAX_QUERY_CHARS,
                        )
                        semantic_query_maps[q] = {
                            "relevance_query": rel_q or q,
                            "bulk_query": bulk_q or q,
                        }
                    gap_seen.add(sig)
        else:
            # Fallback: keyword extraction for academic engines
            en_q = _truncate_query_text(_extract_en_keywords(f"{topic} {gap}", max_terms=8), _MAX_QUERY_CHARS)
            if not en_q:
                en_q = _truncate_query_text(_extract_en_keywords(topic, max_terms=6), _MAX_QUERY_CHARS)
            if en_q and en_q not in gap_kw_queries:
                gap_kw_queries.append(en_q)
            # Fallback: NL for tavily
            if is_topic_zh:
                zh_q = _truncate_query_text(_extract_zh_keywords(f"{topic} {gap}", max_terms=8), _MAX_QUERY_CHARS)
                if zh_q and zh_q not in gap_nl_queries:
                    gap_nl_queries.append(zh_q)
            else:
                nl_q = _truncate_query_text(gap.strip(), _MAX_QUERY_CHARS)
                if nl_q and nl_q not in gap_nl_queries:
                    gap_nl_queries.append(nl_q)

    recall_queries: List[str] = []
    precision_queries: List[str] = []
    discovery_queries: List[str] = []
    if not gap_only_mode:
        # Round 1 keeps broad query generation; rounds 2+ are gap-only.
        client, model_override = _resolve_step_lite_client(state, "research")
        outline_block = "\n".join(f"- {s.title}" for s in dashboard.sections)
        other_sections = [s.title for s in dashboard.sections if s.title != section.title]
        gaps_block = "\n".join(f"- {g}" for g in gaps) if gaps else "(none)"
        avoid_overlap = ", ".join(other_sections[:4]) if other_sections else "(none)"
        temp_snippets = _retrieve_temp_snippets(state, f"{topic} {section.title}", top_k=3)
        temp_block = "\n\n".join(
            f"[{s['name']}] {s['text'][:350]}" for s in temp_snippets
        ) if temp_snippets else "(none)"

        is_zh = _topic_is_chinese(topic)
        bilingual_academic_instruction = _BILINGUAL_HINT_ACADEMIC if is_zh else ""
        bilingual_discovery_instruction = _BILINGUAL_HINT_DISCOVERY if is_zh else ""

        prompt = _pm.render(
            "generate_queries.txt",
            topic=topic,
            scope=dashboard.brief.scope,
            outline_block=outline_block,
            section_title=section.title,
            gaps_block=gaps_block,
            temp_block=temp_block,
            user_context_block=_build_user_context_block(state),
            recall_budget=recall_budget,
            precision_budget=precision_budget,
            discovery_budget=discovery_budget,
            bilingual_academic_instruction=bilingual_academic_instruction,
            bilingual_discovery_instruction=bilingual_discovery_instruction,
            avoid_overlap=avoid_overlap,
        )

        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": "Output search queries ONLY in the specified format."},
                    {"role": "user", "content": prompt},
                ],
                model=model_override,

            )
            raw = (resp.get("final_text") or "").strip()
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
                if lower in ("discovery", "category c", "discovery queries"):
                    current_bucket = "discovery"
                    continue
                if line.startswith("Category") or line.startswith("##"):
                    continue
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                if len(cleaned) <= 3:
                    continue
                cleaned = _truncate_query_text(cleaned, _MAX_QUERY_CHARS)
                seen = set(gap_kw_queries + gap_nl_queries + recall_queries + precision_queries + discovery_queries)
                if cleaned in seen:
                    continue
                if current_bucket == "precision" and len(precision_queries) < precision_budget:
                    precision_queries.append(cleaned)
                elif current_bucket == "discovery" and len(discovery_queries) < discovery_budget * 2:
                    discovery_queries.append(cleaned)
                elif len(recall_queries) < recall_budget:
                    recall_queries.append(cleaned)
                elif len(precision_queries) < precision_budget:
                    precision_queries.append(cleaned)
        except Exception:
            recall_queries = [f"{topic} {section.title}".strip()]

    # Assemble tagged tuples: (query, category) — priority order:
    # engine-specific gap → fallback gap → recall → precision → discovery
    result: List[Tuple[str, str]] = []
    for q, tag in gap_engine_queries:
        result.append((q, tag))
    for q in gap_kw_queries:
        result.append((q, "gap"))
    for q in gap_nl_queries:
        result.append((q, "gap_discovery"))
    llm_queries: List[Tuple[str, str]] = []
    for q in recall_queries:
        llm_queries.append((q, "recall"))
    for q in precision_queries:
        llm_queries.append((q, "precision"))
    for q in discovery_queries:
        llm_queries.append((q, "discovery"))
    result.extend(llm_queries)
    state["gap_semantic_query_maps"] = semantic_query_maps
    if not result:
        return [(f"{topic} {section.title}".strip(), "recall")]
    final_result = result
    _log_query_pack(
        "query_pack_section",
        {
            "section": section.title,
            "topic": topic,
            "round": round_idx,
            "gap_only_mode": gap_only_mode,
            "gap_engine_queries": [(q, t) for q, t in gap_engine_queries],
            "gap_semantic_query_maps": len(semantic_query_maps),
            "gap_kw_queries": gap_kw_queries,
            "gap_nl_queries": gap_nl_queries,
            "deduped_by_history": history_dedup_hits,
            "recall_queries": recall_queries,
            "precision_queries": precision_queries,
            "discovery_queries": discovery_queries,
            "total_queries": len(final_result),
        },
    )
    return final_result


# ────────────────────────────────────────────────
# Local Priority Revise: scan for fresh review signals
# ────────────────────────────────────────────────

def _scan_fresh_revise_signals(state: DeepResearchState) -> None:
    """Check review_waiter for newly submitted 'revise' actions and enqueue them.

    This is called at the start of research_node so that mid-run revisions
    are picked up without waiting for the global review_gate.
    """
    waiter = _resolve_runtime_callback(state, "review_waiter")
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
    t_scope_start = time.perf_counter()
    topic = state["topic"]
    logger.info(
        "[scoping_node] begin | topic=%r prelim_chars=%d clarify_answers=%d",
        topic[:80],
        len((state.get("preliminary_knowledge") or "").strip()),
        len(state.get("clarification_answers") or {}),
    )
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "scope")
    # 生成大纲使用 full client，不降级
    client, model_override = _resolve_step_client_and_model(state, "scope")
    clarification = state.get("clarification_answers") or {}
    clarification_lines = [f"- {k}: {v}" for k, v in clarification.items() if v]
    clarification_block = "\n".join(clarification_lines) if clarification_lines else "(none)"
    prelim_raw = (state.get("preliminary_knowledge") or "").strip()
    prelim_block = prelim_raw[:PRELIMINARY_KNOWLEDGE_MAX_CHARS] if prelim_raw else "(none)"

    prompt = _pm.render(
        "scope_research.txt",
        topic=topic,
        clarification_block=clarification_block,
        preliminary_knowledge_block=prelim_block,
    )

    try:
        t_scope_llm_start = time.perf_counter()
        logger.info(
            "[scoping_node] calling LLM | model=%s prompt_chars=%d",
            model_override or "(default)",
            len(prompt),
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are a research planning expert. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,

            response_model=_ScopingResponse,
        )
        logger.info(
            "[scoping_node] LLM response received | elapsed_ms=%.0f final_text_len=%d",
            (time.perf_counter() - t_scope_llm_start) * 1000.0,
            len(resp.get("final_text") or ""),
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
        topic_domain=_normalize_topic_domain(data.get("topic_domain", "general")),
    )

    dashboard = state.get("dashboard") or ResearchDashboard()
    dashboard.brief = brief
    state["dashboard"] = dashboard

    trajectory = state.get("trajectory") or ResearchTrajectory(topic=topic)
    state["trajectory"] = trajectory
    _emit_progress(state, "scope_done", {"topic": topic, "scope": brief.scope, "key_questions": brief.key_questions})
    logger.info(
        "[scoping_node] done | elapsed_ms=%.0f key_questions=%d scope_len=%d",
        (time.perf_counter() - t_scope_start) * 1000.0,
        len(brief.key_questions or []),
        len(brief.scope or ""),
    )

    return state


def plan_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 2: 规划 — 生成大纲并初始化 Dashboard"""
    t_plan_start = time.perf_counter()
    logger.info(
        "[plan_node] begin | topic=%r search_mode=%s max_sections=%s",
        (state.get("topic") or "")[:80],
        state.get("search_mode", "hybrid"),
        state.get("max_sections"),
    )
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "plan")
    client, model_override = _resolve_step_client_and_model(state, "plan")
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]
    max_sections = _normalize_max_sections(state.get("max_sections"), default=4)

    # Initial retrieval for background context
    # SmartQueryOptimizer is enabled here: it internally detects Chinese input and
    # generates bilingual (zh+en) queries with engine-aware styles when needed.
    # English input is unchanged (optimizer generates English-only queries).
    svc = _get_retrieval_svc(state)
    dr_filters = dict(state.get("filters") or {})
    dr_filters["use_query_optimizer"] = True
    # Research phase: force bge_only for speed; write nodes use the UI's reranker_mode.
    dr_filters["reranker_mode"] = "bge_only"
    # Use same step_top_k / local_top_k as rest of pipeline (no hidden override).
    step_top_k = dr_filters.get("step_top_k")
    local_top_k = dr_filters.get("local_top_k")
    plan_top_k = int(step_top_k or local_top_k or 15)
    logger.info(
        "[plan_node] background retrieval: mode=%s top_k=%d (step_top_k=%s local_top_k=%s) max_sections=%d providers=%s",
        state.get("search_mode", "hybrid"),
        plan_top_k,
        step_top_k,
        local_top_k,
        max_sections,
        dr_filters.get("web_providers"),
    )
    t_retrieval_start = time.perf_counter()
    pack = svc.search(
        query=dashboard.brief.topic,
        mode=state.get("search_mode", "hybrid"),
        top_k=plan_top_k,
        filters=dr_filters,
    )
    _accumulate_evidence_chunks(state, pack.chunks)
    context = pack.to_context_string(max_chunks=plan_top_k)
    logger.info(
        "[plan_node] background retrieval done: chunks=%d context_chars=%d elapsed_ms=%.0f",
        len(pack.chunks),
        len(context),
        (time.perf_counter() - t_retrieval_start) * 1000.0,
    )

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

    # Generate outline: preliminary knowledge has its own cap;
    # retrieval context keeps up to 30k chars, then summarizes to <=10k.
    prelim_raw = (state.get("preliminary_knowledge") or "").strip()
    prelim_block = prelim_raw[:PRELIMINARY_KNOWLEDGE_MAX_CHARS] if prelim_raw else "(none)"
    ultra_lite_provider = (state.get("filters") or {}).get("ultra_lite_provider")
    if len(context) > PLAN_CONTEXT_MAX_CHARS:
        context_for_prompt = summarize_if_needed(
            context,
            PLAN_CONTEXT_SUMMARIZE_TO,
            ultra_lite_provider=ultra_lite_provider,
            purpose="plan_context",
        )
    else:
        context_for_prompt = context
    prompt = _pm.render(
        "plan_outline.txt",
        topic=dashboard.brief.topic,
        scope=dashboard.brief.scope,
        key_questions=", ".join(dashboard.brief.key_questions),
        preliminary_knowledge_block=prelim_block,
        context=context_for_prompt,
        max_sections=max_sections,
    )

    logger.info(
        "[plan_node] calling LLM for outline: model=%s prompt_chars=%d max_sections=%d",
        model_override or "(default)",
        len(prompt),
        max_sections,
    )
    t_plan_llm_start = time.perf_counter()
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an expert at building academic review outlines."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,

        )
        # Use final_text; fall back to reasoning_text for thinking models that
        # may produce an empty final_text when the answer is in the reasoning field.
        raw = (resp.get("final_text") or resp.get("reasoning_text") or "").strip()
        logger.info(
            "[plan_node] LLM outline response: final_text_len=%d reasoning_text_len=%d raw_len=%d elapsed_ms=%.0f",
            len(resp.get("final_text") or ""),
            len(resp.get("reasoning_text") or ""),
            len(raw),
            (time.perf_counter() - t_plan_llm_start) * 1000.0,
        )
    except Exception as e:
        logger.warning("[plan_node] LLM call for outline generation failed: %s", e)
        raw = ""

    # 解析大纲 — 支持多种常见 LLM 输出格式
    sections = _parse_outline_sections(raw, dashboard.brief.topic)
    _emit_progress(
        state,
        "outline_generated",
        {
            "requested_max_sections": max_sections,
            "generated_sections": len(sections),
            "outline_preview": sections[:10],
        },
    )
    if len(sections) > max_sections:
        msg = (
            f"Outline exceeds requested max_sections: requested={max_sections}, "
            f"generated={len(sections)} (kept all, no truncation)."
        )
        logger.warning(msg)
        _emit_progress(
            state,
            "outline_max_sections_exceeded",
            {
                "message": msg,
                "requested_max_sections": max_sections,
                "generated_sections": len(sections),
                "outline_preview": sections[:10],
            },
        )

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
    _save_phase_checkpoint(state, "plan")
    _emit_progress(state, "checkpoint_saved", {
        "phase": "plan",
        "message": "Outline generated — checkpoint saved.",
    })
    logger.info(
        "[plan_node] done | elapsed_ms=%.0f sections=%d total_sources=%d",
        (time.perf_counter() - t_plan_start) * 1000.0,
        len(sections),
        dashboard.total_sources,
    )

    return state


def _filter_by_ui(tier_providers: List[str], ui_allowed: Optional[set]) -> List[str]:
    """Apply UI provider whitelist to a tier's provider list.

    If ui_allowed is None (user passed no web_providers), all providers are permitted.
    If ui_allowed is an empty set, all providers are blocked for this tier.
    """
    if ui_allowed is None:
        return tier_providers
    return [p for p in tier_providers if p in ui_allowed]


def _resolve_fetcher(ui_mode: str, tier_providers: List[str]) -> str:
    """Map the UI content-fetcher setting to a per-call value.

    T1/T2-Semantic return structured abstracts via API — fetching pages is never
    needed regardless of UI setting.  T3 (scholar/google) and Tavily respect the
    UI setting; in 'auto' mode the UnifiedWebSearcher decides lazily.
    """
    if all(p in _API_STRUCTURED_PROVIDERS for p in tier_providers):
        return "off"
    if ui_mode == "force":
        return "force"
    if ui_mode == "off":
        return "off"
    return "auto"


def _quick_coverage_check(
    state: DeepResearchState,
    section: "SectionStatus",
    new_chunks: List[Any],
    known_gaps: List[str],
) -> bool:
    """Quick LLM coverage check with Micro-CoT to prevent false positives.

    For each known gap, the LLM must extract a concrete answer verbatim from
    the evidence before it can mark the gap as addressed.  This prevents the
    common hallucination where the model sees a topic mention ('this paper
    studied X') and optimistically reports coverage.

    Returns True when >= 60% of gaps are substantively addressed.
    Returns False (conservatively escalate) on any parsing/LLM failure.
    Only called on Round 2+ (when section.gaps is non-empty from a prior evaluate).
    """
    if not known_gaps:
        return True

    dashboard = state["dashboard"]
    client, model_override = _resolve_step_lite_client(state, "evaluate")

    # Build evidence summary from real retrieved evidence (like evaluate_node).
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    eval_top_k = int(state.get("filters", {}).get("step_top_k") or preset.get("search_top_k_eval", 20))
    evidence_summary = ""
    try:
        svc = _get_retrieval_svc(state)
        qcc_filters = dict(state.get("filters") or {})
        qcc_filters["use_query_optimizer"] = False
        qcc_filters["reranker_mode"] = "bge_only"
        pack = svc.search(
            query=f"{dashboard.brief.topic} {section.title}",
            mode=state.get("search_mode", "hybrid"),
            top_k=eval_top_k,
            filters=qcc_filters,
        )
        evidence_summary = _build_gap_evidence_summary(
            state,
            query=f"{dashboard.brief.topic} {section.title} {'; '.join(known_gaps)}",
            chunks=list(pack.chunks),
            max_chunks=eval_top_k,
            max_total_chars=80000,
            per_chunk_max_chars=3000,
            compress_step="evaluate",
        )
    except Exception as e:
        logger.debug("quick_coverage_check evidence retrieval failed: %s", e)

    if not evidence_summary.strip():
        evidence_summary = _build_gap_evidence_summary(
            state,
            query=f"{dashboard.brief.topic} {section.title} {'; '.join(known_gaps)}",
            chunks=new_chunks,
            max_chunks=20,
            max_total_chars=80000,
            per_chunk_max_chars=3000,
            compress_step="evaluate",
        )

    gaps_block = "\n".join(f"- {g}" for g in known_gaps)

    try:
        prompt = _pm.render(
            "quick_coverage_check.txt",
            section_title=section.title,
            gaps_block=gaps_block,
            evidence_summary=evidence_summary,
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an evidence sufficiency evaluator. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,

        )
        raw = (resp.get("final_text") or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return False
        addressed = sum(
            1 for item in parsed
            if isinstance(item, dict)
            and item.get("addressed") is True
            and item.get("extracted_answer") not in (None, "", "null")
        )
        ratio = addressed / len(known_gaps)
        _emit_progress(state, "quick_coverage_check", {
            "section": section.title,
            "gaps_checked": len(known_gaps),
            "gaps_addressed": addressed,
            "ratio": round(ratio, 2),
        })
        return ratio >= 0.60
    except Exception as e:
        logger.debug("_quick_coverage_check failed (%s); conservatively escalating tier", e)
        return False


def _execute_tiered_search(
    state: DeepResearchState,
    section: SectionStatus,
    queries: List[Tuple[str, str]],
    svc: Any,
    base_dr_filters: Dict[str, Any],
    preset: Dict[str, Any],
    *,
    max_tier: int = 3,
    ui_allowed_providers: Optional[set] = None,
    ui_content_fetcher: str = "auto",
    ui_source_configs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], set]:
    """Execute progressive tiered search for a section.

    Tier 1 (NCBI)        — biomedical domains only; free REST API; fast
    Tier 2 (Semantic + Tavily) — cross-domain academic + broad discovery
    Tier 3 (Scholar + Google)  — widest coverage; Playwright-based; slowest

    Escalation logic:
      - After each tier, if we have known gaps from a prior evaluate_node call,
        run _quick_coverage_check (Micro-CoT LLM).  If gaps are substantially
        addressed, stop and return early.
      - Round 1 has no prior gaps, so tiers always run up to max_tier.
      - max_tier is set by research_node based on depth preset + round number.

    UI respect:
      - ui_allowed_providers: whitelist from frontend web_providers; None = all allowed
      - ui_source_configs: per-provider topK from frontend; falls back to default_per_provider_top_k
      - ui_content_fetcher: 'off'|'auto'|'force' from frontend; T1/T2-Semantic always 'off'
    """
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]
    branch_id = f"sec_{dashboard.sections.index(section) + 1}"
    search_mode = state.get("search_mode", "hybrid")
    # UI 穿透：step_top_k / local_top_k 优先，否则用 preset 的 default_per_provider_top_k
    ui_step_k = base_dr_filters.get("step_top_k")
    ui_local_k = base_dr_filters.get("local_top_k")
    default_top_k = int(ui_step_k or ui_local_k or preset.get("default_per_provider_top_k", 10))
    all_chunks: List[Any] = []
    all_sources: set = set()
    known_gaps = list(section.gaps) if section.research_rounds > 1 else []

    chunk_cache = _resolve_runtime_query_cache(state)

    semantic_query_maps = state.get("gap_semantic_query_maps") or {}

    def _run_search(
        q: str,
        providers: List[str],
        tier_label: str,
        *,
        semantic_query_map: Optional[Dict[str, str]] = None,
    ) -> None:
        _ensure_not_cancelled(state)
        if not providers:
            return

        q_norm = _truncate_query_text(q, _MAX_QUERY_CHARS)
        if not q_norm:
            return

        f = dict(base_dr_filters)
        # Research phase always uses BGE-only reranker for speed;
        # ColBERT (cascade) is reserved for the write stage.
        f["reranker_mode"] = "bge_only"
        f["web_providers"] = providers
        if semantic_query_map and "semantic" in providers:
            f["semantic_query_map"] = dict(semantic_query_map)
        if ui_source_configs:
            f["web_source_configs"] = ui_source_configs
        f["use_content_fetcher"] = _resolve_fetcher(ui_content_fetcher, providers)
        # 当 UI 传了 step_top_k 时，tier 检索统一用该值（UI 穿透）；否则用 per-provider topK 或 default
        if ui_step_k is not None:
            top_k = default_top_k
        else:
            top_k = default_top_k
            if ui_source_configs:
                provider_ks = [
                    ui_source_configs.get(p, {}).get("topK", default_top_k)
                    for p in providers
                ]
                top_k = max(provider_ks) if provider_ks else default_top_k

        provider_sig = "|".join(sorted(providers))
        executed_sig = f"{provider_sig}::{q_norm}"
        cache_key = json.dumps(
            {"query": q_norm, "providers": sorted(providers), "mode": search_mode, "top_k": top_k, "filters": f},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        cached_entry = chunk_cache.get(cache_key)
        if cached_entry is not None:
            if isinstance(cached_entry, dict):
                cached_chunks = list(cached_entry.get("chunks") or [])
                cached_sources = set(cached_entry.get("sources") or [])
            else:
                # Legacy value shape: raw chunk list
                cached_chunks = list(cached_entry)
                cached_sources = set()
            all_chunks.extend(cached_chunks)
            all_sources.update(cached_sources)
            section.source_count += len(cached_chunks)
            dashboard.total_sources += len(cached_chunks)
            trajectory.add_search_action(branch_id, SearchAction(
                query=q_norm,
                tool=f"{tier_label}:cache_hit",
                result_summary=f"Reused {len(cached_chunks)} cached chunks from prior section",
                source_count=len(cached_chunks),
            ))
            _log_query_pack("tier_query_cache_hit", {
                "section": section.title,
                "tier": tier_label,
                "providers": providers,
                "query": q_norm,
                "cached_chunks": len(cached_chunks),
            })
            eq = state.setdefault("executed_queries", {})
            eq.setdefault(section.title, [])
            if executed_sig not in eq[section.title]:
                eq[section.title].append(executed_sig)
            return
        _log_query_pack(
            "tier_query_execute",
            {
                "section": section.title,
                "tier": tier_label,
                "providers": providers,
                "query": q_norm,
                "top_k": top_k,
                "round": int(section.research_rounds),
            },
        )
        pack = svc.search(query=q_norm, mode=search_mode, top_k=top_k, filters=f)
        all_chunks.extend(pack.chunks)
        all_sources.update(pack.sources_used)
        section.source_count += len(pack.chunks)
        dashboard.total_sources += len(pack.chunks)
        trajectory.add_search_action(branch_id, SearchAction(
            query=q_norm,
            tool=tier_label,
            result_summary=f"Retrieved {len(pack.chunks)} chunks",
            source_count=len(pack.chunks),
        ))
        eq = state.setdefault("executed_queries", {})
        sec_eq = eq.setdefault(section.title, [])
        if executed_sig not in sec_eq:
            sec_eq.append(executed_sig)
        chunk_cache[cache_key] = {
            "chunks": list(pack.chunks),
            "sources": sorted(list(pack.sources_used)),
        }

    # ── Tier 1: NCBI (biomedical domains only) ──
    if max_tier >= 1 and _normalize_topic_domain(dashboard.brief.topic_domain) == "biomedical":
        t1_providers = _filter_by_ui(["ncbi"], ui_allowed_providers)
        if t1_providers:
            kw_queries = [(q, cat) for q, cat in queries if cat in ("gap", "gap_ncbi", "recall", "precision")]
            for q, cat in kw_queries:
                _run_search(q, t1_providers, "tier1_ncbi")
            if known_gaps and _quick_coverage_check(state, section, all_chunks, known_gaps):
                _emit_progress(state, "tier1_sufficient", {
                    "section": section.title,
                    "message": "Tier 1 (ncbi) coverage check passed; skipping Tier 2.",
                })
                return all_chunks, all_sources

    # ── Tier 2: Semantic Scholar (keyword) + Tavily (discovery) ──
    if max_tier >= 2:
        t2_kw = _filter_by_ui(["semantic"], ui_allowed_providers)
        t2_disc = _filter_by_ui(["tavily"], ui_allowed_providers)
        if t2_kw:
            kw_queries = [(q, cat) for q, cat in queries if cat in ("gap", "gap_semantic", "recall", "precision")]
            for q, cat in kw_queries:
                if cat == "gap_semantic":
                    q_norm = _truncate_query_text(q, _MAX_QUERY_CHARS)
                    semantic_map = semantic_query_maps.get(q_norm) or semantic_query_maps.get(q) or {}
                    _run_search(q, t2_kw, "tier2_semantic", semantic_query_map=semantic_map)
                else:
                    _run_search(q, t2_kw, "tier2_semantic")
        if t2_disc:
            disc_queries = [(q, cat) for q, cat in queries if cat in ("discovery", "gap_discovery", "gap_tavily")]
            for q, cat in disc_queries:
                _run_search(q, t2_disc, "tier2_tavily")
        if known_gaps and _quick_coverage_check(state, section, all_chunks, known_gaps):
            _emit_progress(state, "tier2_sufficient", {
                "section": section.title,
                "message": "Tier 2 (semantic+tavily) coverage check passed; skipping Tier 3.",
            })
            return all_chunks, all_sources

    # ── Tier 3: Google Scholar + Google (engine-specific gap + LLM-refined) ──
    if max_tier >= 3:
        t3_scholar = _filter_by_ui(["scholar"], ui_allowed_providers)
        t3_google = _filter_by_ui(["google"], ui_allowed_providers)
        t3_providers = _filter_by_ui(["scholar", "google"], ui_allowed_providers)
        if t3_providers:
            for q, cat in queries:
                if cat == "gap_scholar" and t3_scholar:
                    _run_search(q, t3_scholar, "tier3_gap_scholar")
                elif cat == "gap_google" and t3_google:
                    _run_search(q, t3_google, "tier3_gap_google")
            n_refined = int(preset.get("tier3_refined_queries", 2))
            refined = _generate_refined_queries(state, section, all_chunks, max_queries=n_refined)
            _emit_progress(state, "tier3_start", {
                "section": section.title,
                "refined_queries": refined,
                "message": f"Tier 3: {'+'.join(t3_providers)} with {len(refined)} refined queries",
            })
            for q in refined:
                _run_search(q, t3_providers, "tier3_scholar_google")

    return all_chunks, all_sources


def research_node(state: DeepResearchState) -> DeepResearchState:
    """Phase 3: 递归研究 — 对当前章节执行搜索 + 信息评估

    Tier ceiling per round (controlled by depth preset):
      Lite:          Round 1 → T1+T2;  Round 2..N-1 → T1+T2;  Round N → T1+T2+T3
      Comprehensive: Every round → T1+T2+T3

    Within each tier, escalation is coverage-driven (Micro-CoT check on Round 2+).
    """
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "research")
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]
    client = _resolve_runtime_llm_client(state)

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
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    svc = _get_retrieval_svc(state)
    base_dr_filters = dict(state.get("filters") or {})
    base_dr_filters["use_query_optimizer"] = False

    # ── Normal / Gap-fill round ──
    section.status = "researching"
    section.research_rounds += 1
    dashboard.total_iterations += 1
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    _emit_progress(state, "section_research_start", {"section": section.title, "round": section.research_rounds})

    # Build queries (recall + precision + gap + discovery)
    recall_q = int(preset.get("recall_queries_per_section", 2))
    precision_q = int(preset.get("precision_queries_per_section", 2))
    max_q = recall_q + precision_q + 4 + 2  # +4 gap buffer (kw + nl), +2 discovery
    queries = _generate_section_queries(state, section, max_queries=max_q)

    temp_hits = _retrieve_temp_snippets(state, f"{dashboard.brief.topic} {section.title}", top_k=4)
    branch_id = f"sec_{dashboard.sections.index(section) + 1}"
    for hit in temp_hits:
        trajectory.add_finding(branch_id, f"[temp:{hit['name']}] {hit['text'][:180]}")

    # ── Determine tier ceiling for this round ──
    max_rounds = int(preset.get("max_section_research_rounds", 3))
    is_first_round = section.research_rounds <= 1
    is_last_round = section.research_rounds >= max_rounds
    if is_last_round:
        max_tier = int(preset.get("last_round_max_tier", 3))
    elif is_first_round:
        max_tier = int(preset.get("round1_max_tier", 2))
    else:
        max_tier = int(preset.get("gapfill_max_tier", 2))

    # ── Extract UI settings from state filters ──
    ui_filters = state.get("filters") or {}
    ui_providers = ui_filters.get("web_providers")
    ui_allowed = set(ui_providers) if ui_providers else None
    ui_fetcher = ui_filters.get("use_content_fetcher", "auto")
    ui_source_configs = ui_filters.get("web_source_configs")

    _emit_progress(state, "research_tier_plan", {
        "section": section.title,
        "round": section.research_rounds,
        "max_tier": max_tier,
        "is_first_round": is_first_round,
        "is_last_round": is_last_round,
    })

    all_chunks, all_sources = _execute_tiered_search(
        state, section, queries, svc, base_dr_filters, preset,
        max_tier=max_tier,
        ui_allowed_providers=ui_allowed,
        ui_content_fetcher=ui_fetcher,
        ui_source_configs=ui_source_configs,
    )

    _finalise_research_round(state, section, all_chunks, all_sources, queries_repr=queries, client=client)
    _save_phase_checkpoint(state, "research", section.title)
    return state


def _finalise_research_round(
    state: DeepResearchState,
    section: SectionStatus,
    all_chunks: List[Any],
    all_sources: set,
    queries_repr: Any,
    client: Any,
) -> None:
    """Post-search bookkeeping shared by normal rounds and Completion Round."""
    dashboard = state["dashboard"]
    trajectory = state["trajectory"]

    distinct_docs = _count_distinct_docs(all_chunks)
    if len(all_chunks) < 3 or distinct_docs < 3:
        section.evidence_scarce = True
        _emit_progress(state, "evidence_insufficient", {
            "section": section.title,
            "message": "Evidence insufficient after search; section may be degraded in writing.",
            "chunks_found": len(all_chunks),
            "distinct_docs": distinct_docs,
            "gaps": section.gaps[:3],
        })
    else:
        section.evidence_scarce = False

    _accumulate_evidence_chunks(state, all_chunks)
    _accumulate_section_pool(state, section.title, all_chunks, pool_source="research_round")

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

    _sync_outline_status_to_canvas(state)

    if trajectory.needs_compression():
        compress_trajectory(trajectory, client, model=state.get("model_override"))

    _emit_progress(state, "section_research_done", {
        "section": section.title,
        "queries": queries_repr,
        "chunks_found": len(all_chunks),
        "sources_used": sorted(all_sources),
    })


def evaluate_node(state: DeepResearchState) -> DeepResearchState:
    """Evaluate information sufficiency and decide if more search is needed."""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "evaluate")
    dashboard = state["dashboard"]
    client, model_override = _resolve_step_lite_client(state, "evaluate")
    trajectory = state["trajectory"]

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    # Build findings context from per-section evidence pool first; fallback to retrieval.
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    eval_top_k = int(state.get("filters", {}).get("step_top_k") or preset.get("search_top_k_eval", 20))
    findings = ""
    query_text = f"{dashboard.brief.topic} {section.title}"
    eval_filters = dict(state.get("filters") or {})
    eval_filters["use_query_optimizer"] = False
    eval_filters["reranker_mode"] = "bge_only"  # 评估证据阶段固定 bge_only
    pool_chunks = list((state.get("section_evidence_pool") or {}).get(section.title) or [])
    if pool_chunks:
        reranked_pool = _rerank_section_pool_chunks(
            query=query_text,
            pool_chunks=pool_chunks,
            top_k=eval_top_k,
            reranker_mode=eval_filters.get("reranker_mode"),
        )
        pool_pack = _build_pack_from_chunks(query_text, reranked_pool)
        findings = cap_and_log(
            pool_pack.to_context_string(max_chunks=eval_top_k) or "", purpose="evaluate_findings"
        )
    if not findings.strip():
        try:
            svc = _get_retrieval_svc(state)
            pack = svc.search(
                query=query_text,
                mode=state.get("search_mode", "hybrid"),
                top_k=eval_top_k,
                filters=eval_filters,
            )
            findings = cap_and_log(
                pack.to_context_string(max_chunks=eval_top_k) or "", purpose="evaluate_findings"
            )
        except Exception as e:
            logger.debug("evaluate evidence retrieval failed: %s", e)

    if not findings.strip():
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

    # Coverage remains low with a thin pool: run constrained gap-targeted fresh search.
    cov_threshold = float(preset.get("coverage_threshold", 0.6))
    min_pool_size = int(preset.get("eval_pool_min_chunks", 5))
    should_supplement = (
        section.coverage_score < cov_threshold
        and len(pool_chunks) < min_pool_size
        and bool(section.gaps)
    )
    if should_supplement:
        supplement_chunks: List[Any] = []
        try:
            svc = _get_retrieval_svc(state)
            supplement_filters = dict(eval_filters)
            supplement_filters["reranker_mode"] = "bge_only"
            supplement_top_k = max(5, int(eval_top_k * 0.2))
            for gap in (section.gaps or [])[:3]:
                gap_q = f"{dashboard.brief.topic} {section.title} {str(gap).strip()}"
                gap_pack = svc.search(
                    query=gap_q,
                    mode=state.get("search_mode", "hybrid"),
                    top_k=supplement_top_k,
                    filters=supplement_filters,
                )
                supplement_chunks.extend(gap_pack.chunks)
            if supplement_chunks:
                _accumulate_section_pool(
                    state,
                    section.title,
                    supplement_chunks,
                    pool_source="eval_supplement",
                )
                _accumulate_evidence_chunks(state, supplement_chunks)
                _emit_progress(
                    state,
                    "evaluate_supplement_search_done",
                    {
                        "section": section.title,
                        "supplement_chunks": len(supplement_chunks),
                        "coverage": section.coverage_score,
                        "min_pool_size": min_pool_size,
                    },
                )
        except Exception as e:
            logger.debug("evaluate supplement search failed: %s", e)
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
    """Extract 3-5 core claims from section evidence (with [ref:xxxx] citations) before writing."""
    _ensure_not_cancelled(state)
    _tick_cost_monitor(state, "generate_claims")
    dashboard = state["dashboard"]
    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return state

    # 每个大纲内结合材料写作（claims 所用证据）：使用 UI 穿透的 reranker_mode
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    dr_filters = dict(state.get("filters") or {})
    write_top_k = _compute_effective_write_k(preset, dr_filters)
    query_text = f"{dashboard.brief.topic} {section.title}"
    pool_chunks = list((state.get("section_evidence_pool") or {}).get(section.title) or [])
    if pool_chunks:
        selected_chunks = _rerank_section_pool_chunks(
            query=query_text,
            pool_chunks=pool_chunks,
            top_k=write_top_k,
            reranker_mode=dr_filters.get("reranker_mode"),
        )
        pack = _build_pack_from_chunks(query_text, selected_chunks)
    else:
        svc = _get_retrieval_svc(state)
        dr_filters["use_query_optimizer"] = False
        dr_filters["web_queries_per_provider"] = _build_write_queries(
            dashboard.brief.topic, section.title,
        )
        pack = svc.search(
            query=query_text,
            mode=state.get("search_mode", "hybrid"),
            top_k=write_top_k,
            filters=dr_filters,
        )
    evidence_str = pack.to_context_string(max_chunks=write_top_k)

    client, model_override = _resolve_step_client_and_model(state, "write")
    user_content = _pm.render(
        "generate_claims.txt",
        section_title=section.title,
        evidence=cap_and_log(evidence_str or "", purpose="generate_claims"),
    )
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are an expert at extracting concise, citation-backed claims from evidence. Preserve every [ref:xxxx] citation marker in each claim."},
                {"role": "user", "content": user_content},
            ],
            model=model_override,

        )
        verified_claims = (resp.get("final_text") or "").strip()
    except Exception as e:
        logger.warning("generate_claims_node LLM call failed: %s", e)
        verified_claims = ""

    state["verified_claims"] = verified_claims
    
    _save_phase_checkpoint(state, "generate_claims", section.title)
    _emit_progress(state, "checkpoint_saved", {
        "phase": "generate_claims",
        "section": section.title,
        "message": f"Claims generated for \"{section.title}\" — checkpoint saved.",
    })
    
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

    # 收集上下文（用于 write 系统消息；超 70k 则 ultra_lite 总结）
    context_parts = [dashboard.to_system_prompt()]
    if trajectory.compressed_summaries:
        context_parts.append("\n".join(trajectory.compressed_summaries[-2:]))
    context = "\n\n".join(context_parts)
    ultra_lite_provider = (state.get("filters") or {}).get("ultra_lite_provider")

    # Retrieve section context — adaptive write-stage top_k
    # 每个大纲内结合材料写作：使用 UI 穿透的 reranker_mode（含 cascade），不在此处覆盖
    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    dr_filters = dict(state.get("filters") or {})
    write_top_k = _compute_effective_write_k(preset, dr_filters)
    verification_k = int(preset.get("verification_k", max(10, write_top_k)))
    base_query = f"{dashboard.brief.topic} {section.title}"
    verify_query = f"{dashboard.brief.topic} {section.title} data evidence citation verification"
    pool_chunks = list((state.get("section_evidence_pool") or {}).get(section.title) or [])
    if pool_chunks:
        write_chunks = _rerank_section_pool_chunks(
            query=base_query,
            pool_chunks=pool_chunks,
            top_k=write_top_k,
            reranker_mode=dr_filters.get("reranker_mode"),
        )
        verify_chunks = _rerank_section_pool_chunks(
            query=verify_query,
            pool_chunks=pool_chunks,
            top_k=verification_k,
            reranker_mode=dr_filters.get("reranker_mode"),
        )
        pack = _build_pack_from_chunks(base_query, write_chunks)
        verify_pack = _build_pack_from_chunks(verify_query, verify_chunks)
    else:
        svc = _get_retrieval_svc(state)
        dr_filters["use_query_optimizer"] = False
        write_qpp = _build_write_queries(dashboard.brief.topic, section.title)
        dr_filters["web_queries_per_provider"] = write_qpp
        pack = svc.search(
            query=base_query,
            mode=state.get("search_mode", "hybrid"),
            top_k=write_top_k,
            filters=dr_filters,
        )
        verify_qpp = _build_write_queries(
            dashboard.brief.topic, section.title, extra_kw_suffix="evidence data",
        )
        dr_filters["web_queries_per_provider"] = verify_qpp
        verify_pack = svc.search(
            query=verify_query,
            mode=state.get("search_mode", "hybrid"),
            top_k=verification_k,
            filters=dr_filters,
        )
    evidence_str = pack.to_context_string(max_chunks=write_top_k)
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
            "(e.g., 'A single study [ref:xxxx] suggests...' or 'Preliminary evidence from [ref:xxxx] indicates...').\n"
            "- Actively look for converging evidence from different authors/studies to strengthen conclusions.\n"
        )
        evidence_limited = summarize_if_needed(
            evidence_str or "", DR_SECTION_EVIDENCE_MAX_CHARS,
            ultra_lite_provider=ultra_lite_provider, purpose="dr_section_evidence",
        )
        verification_limited = summarize_if_needed(
            verification_evidence_str or "", DR_SECTION_EVIDENCE_MAX_CHARS,
            ultra_lite_provider=ultra_lite_provider, purpose="dr_section_verification",
        )
        context_limited = summarize_if_needed(
            context, DR_SECTION_EVIDENCE_MAX_CHARS,
            ultra_lite_provider=ultra_lite_provider, purpose="dr_section_context",
        )
        prompt = _pm.render(
            "write_section.txt",
            section_title=section.title,
            language_instruction=_language_instruction(state),
            user_context_block=_build_user_context_block(state),
            triangulation_block=triangulation_block,
            caution_block=caution_block,
            quantitative_block=quantitative_block,
            claims_block=claims_block,
            evidence=evidence_limited,
            verification_evidence=verification_limited,
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
                        f"{context_limited}"
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
                )
                section_text = (react_result.final_text or "").strip()
                if not section_text:
                    resp = client.chat(
                        messages=messages,
                        model=model_override,
                    )
                    section_text = (resp.get("final_text") or "").strip()
            else:
                resp = client.chat(
                    messages=messages,
                    model=model_override,
                )
                section_text = (resp.get("final_text") or "").strip()
        except Exception as e:
            section_text = f"(Section '{section.title}' generation failed: {e})"

    # 对章节文本做 hash->cite_key 后处理，并在 state 内保持引用键稳定
    section_chunks = list(pack.chunks) + list(verify_pack.chunks)
    _accumulate_section_pool(state, section.title, section_chunks, pool_source="write_stage")
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

    section.status = "reviewing"
    dashboard.update_confidence()
    _sync_outline_status_to_canvas(state)
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
    _save_phase_checkpoint(state, "write", section.title)

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
        _sync_outline_status_to_canvas(state)
        _emit_progress(state, "section_verify_done", {"section": section.title, "status": "done", "unsupported_claims": 0})
        _save_phase_checkpoint(state, "verify", section.title)
        completed = list(state.get("sections_completed") or [])
        total = len(dashboard.sections)
        _save_phase_checkpoint(state, "section_done", section.title)
        _emit_progress(state, "checkpoint_saved", {
            "phase": "section_done",
            "section": section.title,
            "completed": len(completed),
            "total": total,
            "message": f"Section \"{section.title}\" done — checkpoint saved ({len(completed)}/{total}).",
        })
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
                # ── SEVERE: full re-research (subject to rewrite cycle cap) ──
                max_rewrite = int(preset.get("max_verify_rewrite_cycles", 1))
                section.verify_rewrite_count += 1

                if section.verify_rewrite_count > max_rewrite:
                    # Cap exceeded — downgrade: record gaps, mark evidence_scarce, proceed
                    section.evidence_scarce = True
                    logger.info(
                        "Section '%s' verify-rewrite cap reached (%d/%d), downgrading SEVERE to proceed",
                        section.title, section.verify_rewrite_count, max_rewrite,
                    )
                    _emit_progress(
                        state,
                        "verify_severe_capped",
                        {
                            "section": section.title,
                            "unsup_ratio": round(unsup_ratio, 2),
                            "rewrite_count": section.verify_rewrite_count,
                            "max_rewrite": max_rewrite,
                            "message": f"Verify-rewrite cap reached ({section.verify_rewrite_count}/{max_rewrite}) — proceeding with available evidence.",
                        },
                    )
                    # Fall through to section.status = "done" below
                else:
                    section.status = "researching"
                    _emit_progress(
                        state,
                        "verify_severe",
                        {
                            "section": section.title,
                            "unsup_ratio": round(unsup_ratio, 2),
                            "message": f"Verification severe ({unsup_ratio:.0%} unsupported) — returning to research (rewrite {section.verify_rewrite_count}/{max_rewrite}).",
                            "unsupported_claims": result.unsupported_claims,
                            "total_claims": result.total_claims,
                        },
                    )
                    _save_phase_checkpoint(state, "verify", section.title)
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
    _sync_outline_status_to_canvas(state)
    _emit_progress(
        state,
        "section_verify_done",
        {
            "section": section.title,
            "status": "done",
            "coverage": section.coverage_score,
        },
    )
    _save_phase_checkpoint(state, "verify", section.title)

    # ── Checkpoint: section fully done (research → write → verify complete) ──
    completed = list(state.get("sections_completed") or [])
    total = len(dashboard.sections)
    _save_phase_checkpoint(state, "section_done", section.title)
    _emit_progress(state, "checkpoint_saved", {
        "phase": "section_done",
        "section": section.title,
        "completed": len(completed),
        "total": total,
        "message": f"Section \"{section.title}\" done — checkpoint saved ({len(completed)}/{total}).",
    })

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

    waiter = _resolve_runtime_callback(state, "review_waiter")
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
    _save_phase_checkpoint(state, "review_gate")
    _emit_progress(state, "checkpoint_saved", {
        "phase": "review_gate",
        "message": "Review gate reached — checkpoint saved.",
    })
    
    interrupt(
        {
            "reason": "waiting_for_review",
            "pending_sections": pending_sections,
        }
    )
    return state


# ---------------------------------------------------------------------------
# Coherence-refinement helpers (used by synthesize_node)
# ---------------------------------------------------------------------------

def _split_into_sections(body_md: str) -> List[Tuple[str, str]]:
    """
    Split a markdown document into sections by level-2 headings (## ...).

    Returns a list of (heading_line, body_text) pairs, where heading_line
    includes the leading '## ' and body_text is everything until the next ##
    heading (or end of document).  Content before the first ## heading is
    returned as ("", content) so it is never silently dropped.
    """
    if not body_md:
        return []

    sections: List[Tuple[str, str]] = []
    current_heading = ""
    current_lines: List[str] = []

    for line in body_md.splitlines(keepends=True):
        if line.startswith("## "):
            # Flush previous section
            if current_heading or current_lines:
                sections.append((current_heading, "".join(current_lines).rstrip()))
            current_heading = line.rstrip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_heading or current_lines:
        sections.append((current_heading, "".join(current_lines).rstrip()))

    return sections


def _extract_first_sentence(text: str, max_chars: int = 120) -> str:
    """Return the first sentence of *text*, truncated to *max_chars*."""
    t = (text or "").strip()
    if not t:
        return ""
    # Strip leading markdown emphasis / list markers
    t = re.sub(r"^[*_>#\-]+\s*", "", t)
    # Find end of first sentence
    for punct in ("。", ".", "！", "！", "？", "?"):
        idx = t.find(punct)
        if 0 < idx <= max_chars:
            return t[: idx + 1]
    return t[:max_chars]


def _build_document_blueprint(
    topic: str,
    abstract: str,
    sections: List[Tuple[str, str]],
    current_idx: int,
) -> str:
    """
    Build a lightweight Document Blueprint string to prepend to every
    sliding-window coherence prompt.

    The blueprint gives the LLM global narrative awareness of the full review
    without including the complete text of other sections.  It contains:

    - Topic
    - Abstract (first 300 chars as a condensed anchor)
    - Section outline: heading + first sentence of each section body,
      with a '>> [CURRENT] <<' marker on the section being refined.

    Estimated size: ~400-800 tokens for a 6-8 section document.
    Zero extra LLM calls — purely string extraction.
    """
    lines: List[str] = ["[DOCUMENT BLUEPRINT]", f"Topic: {topic}"]

    if abstract:
        abstract_snippet = abstract.strip()[:300].replace("\n", " ")
        lines.append(f"Abstract: {abstract_snippet}")

    lines.append("")
    lines.append("Section Outline:")

    for idx, (heading, body) in enumerate(sections):
        num = idx + 1
        display_heading = heading.lstrip("#").strip() if heading else f"Section {num}"
        snippet = _extract_first_sentence(body)
        snippet_part = f' -- "{snippet}"' if snippet else ""
        if idx == current_idx:
            lines.append(f'{num}. >> {display_heading} [CURRENT] <<{snippet_part}')
        else:
            lines.append(f"{num}. {display_heading}{snippet_part}")

    return "\n".join(lines)


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
        if t == "evidence limited":
            return True
        # Numeric citation style: [1], [23]
        if re.fullmatch(r"\d{1,4}", t):
            return True
        # Hash / legacy cite-key: [3dd4798b...], [smith2021], [doi:...]
        if re.fullmatch(r"[a-z0-9_.:/-]{6,120}", t):
            return True
        # Academic author-date: [Wang, 2018], [Wang and Li, 2015], [Wang et al., 2011a]
        if re.fullmatch(r"[A-Za-z\u4e00-\u9fff].{2,80},\s*(?:\d{4}|n\.d\.)[a-z]?", t, re.IGNORECASE):
            return True
        # Sequential web keys: [Web1], [Web2], ...
        if re.fullmatch(r"web\d+", t, re.IGNORECASE):
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
        effective_lang_rule = lang_hard_rule or "Respect the document's dominant language."

        # ── Token budget estimation ──
        try:
            from src.utils.token_counter import (
                count_tokens,
                get_context_window,
                compute_safe_budget,
                needs_sliding_window,
            )
            _token_counting_available = True
        except Exception:
            # Some environments may fail while importing `src.utils` package
            # due to unrelated side-effect imports in src/utils/__init__.py.
            # Fallback to direct module loading so token budgeting still works.
            try:
                import importlib.util

                token_counter_path = Path(__file__).resolve().parents[2] / "utils" / "token_counter.py"
                spec = importlib.util.spec_from_file_location("token_counter_direct", str(token_counter_path))
                if spec is None or spec.loader is None:
                    raise RuntimeError("failed to load token_counter module spec")
                tc_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tc_mod)

                count_tokens = tc_mod.count_tokens
                get_context_window = tc_mod.get_context_window
                compute_safe_budget = tc_mod.compute_safe_budget
                needs_sliding_window = tc_mod.needs_sliding_window
                _token_counting_available = True
            except Exception:
                _token_counting_available = False
                logger.debug("tiktoken unavailable; falling back to char-based estimation")

        def _estimate_tokens(text: str) -> int:
            if _token_counting_available:
                return count_tokens(text)
            return int(len(text) * 0.4)

        # Resolve model name for context-window lookup
        _active_model = model_override or ""
        if not _active_model:
            try:
                _active_model = client._default_model or ""  # type: ignore[attr-defined]
            except Exception:
                _active_model = ""

        if _token_counting_available:
            _context_window = get_context_window(_active_model)
        else:
            _context_window = 64_000  # conservative default

        # Build the single-pass prompt to measure its token footprint
        _single_pass_prompt = _pm.render(
            "coherence_refine.txt",
            language_instruction=_language_instruction(state),
            lang_hard_rule=effective_lang_rule,
            body_md=body_md,
        )
        _sys_tokens = _estimate_tokens("You are an expert in scholarly synthesis and coherence editing.")
        _prompt_tokens = _estimate_tokens(_single_pass_prompt) + _sys_tokens
        _body_tokens = _estimate_tokens(body_md)

        if _token_counting_available:
            _use_sliding = needs_sliding_window(
                _prompt_tokens, _context_window, safety_margin=0.10, min_output_tokens=1024
            )
        else:
            # Fallback heuristic: slide if body > 6000 tokens (~15k chars)
            _use_sliding = _body_tokens > 6000

        _emit_progress(
            state,
            "coherence_strategy_selected",
            {
                "strategy": "sliding_window" if _use_sliding else "single_pass",
                "body_tokens": _body_tokens,
                "prompt_tokens": _prompt_tokens,
                "context_window": _context_window,
            },
        )

        # ── Helper: extract tail tokens from a text ──
        def _tail_tokens(text: str, n_tokens: int) -> str:
            """Return approximately the last n_tokens worth of text."""
            approx_chars = n_tokens * 4
            return text[-approx_chars:] if len(text) > approx_chars else text

        # ── Helper: extract head tokens from a text ──
        def _head_tokens(text: str, n_tokens: int) -> str:
            """Return approximately the first n_tokens worth of text."""
            approx_chars = n_tokens * 4
            return text[:approx_chars] if len(text) > approx_chars else text

        # ===================================================================
        # PATH A — Single-pass (document fits within context window)
        # ===================================================================
        if not _use_sliding:
            if _token_counting_available:
                output_budget = compute_safe_budget(_prompt_tokens, _context_window)
                # A rewrite should not expand the document significantly
                output_budget = min(output_budget, _body_tokens + 500)
            else:
                output_budget = 3500

            try:
                resp_coherence = client.chat(
                    messages=[
                        {"role": "system", "content": "You are an expert in scholarly synthesis and coherence editing."},
                        {"role": "user", "content": _single_pass_prompt},
                    ],
                    model=model_override,
                    max_tokens=output_budget,
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
                                "strategy": "single_pass",
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
                logger.warning("Global coherence refine (single-pass) failed: %s", e)
                _emit_progress(
                    state,
                    "warning",
                    {"message": "全文连贯性整合失败，已回退到合成稿。"},
                )

        # ===================================================================
        # PATH B — Sliding window (document too long for a single call)
        # ===================================================================
        else:
            sections_list = _split_into_sections(body_md)
            n_sections = len(sections_list)
            topic_str = state.get("topic") or ""
            # Abstract is inserted at parts[1]; try to extract it from the assembled text
            _abstract_for_blueprint = abstract if abstract else ""

            refined_sections: List[str] = []
            window_errors = 0

            for idx, (heading, sec_body) in enumerate(sections_list):
                current_section_text = (heading + "\n" + sec_body).strip() if heading else sec_body.strip()

                # Build global context guide (blueprint with current marker)
                blueprint = _build_document_blueprint(
                    topic=topic_str,
                    abstract=_abstract_for_blueprint,
                    sections=sections_list,
                    current_idx=idx,
                )

                # Build local context: tail of previous section
                if idx > 0:
                    prev_heading, prev_body = sections_list[idx - 1]
                    prev_full = (prev_heading + "\n" + prev_body).strip()
                    prev_tail = _tail_tokens(prev_full, 300)
                else:
                    prev_tail = "(This is the first section — no preceding content.)"

                # Build local context: head of next section
                if idx < n_sections - 1:
                    next_heading, next_body = sections_list[idx + 1]
                    next_full = (next_heading + "\n" + next_body).strip()
                    next_preview = _head_tokens(next_full, 150)
                else:
                    next_preview = "(This is the last section — no following content.)"

                window_prompt = _pm.render(
                    "coherence_refine_window.txt",
                    language_instruction=_language_instruction(state),
                    lang_hard_rule=effective_lang_rule,
                    document_blueprint=blueprint,
                    prev_tail=prev_tail,
                    current_section=current_section_text,
                    next_preview=next_preview,
                )

                # Dynamic output budget per window: allow modest expansion
                sec_tokens = _estimate_tokens(current_section_text)
                if _token_counting_available:
                    win_prompt_tokens = _estimate_tokens(window_prompt) + _sys_tokens
                    win_budget = compute_safe_budget(win_prompt_tokens, _context_window)
                    win_budget = min(win_budget, sec_tokens + 600)
                else:
                    win_budget = min(3500, sec_tokens + 600)
                win_budget = max(1, min(win_budget, 16384))  # avoid 0 or over model limit

                try:
                    resp_win = client.chat(
                        messages=[
                            {"role": "system", "content": "You are an expert in scholarly synthesis and coherence editing."},
                            {"role": "user", "content": window_prompt},
                        ],
                        model=model_override,
                        max_tokens=win_budget,
                    )
                    refined_sec = (resp_win.get("final_text") or "").strip()
                    refined_sections.append(refined_sec if refined_sec else current_section_text)
                    _emit_progress(
                        state,
                        "coherence_window_done",
                        {"window": idx + 1, "total": n_sections, "section": heading.lstrip("#").strip()},
                    )
                except requests.exceptions.HTTPError as e:
                    body = (e.response.text or "")[:800] if getattr(e, "response", None) else ""
                    logger.warning(
                        "Coherence window %d/%d failed: %s | response: %s",
                        idx + 1, n_sections, e, body,
                    )
                    refined_sections.append(current_section_text)
                    window_errors += 1
                except Exception as e:
                    logger.warning("Coherence window %d/%d failed: %s", idx + 1, n_sections, e)
                    refined_sections.append(current_section_text)
                    window_errors += 1

            refined_body = "\n\n".join(refined_sections)
            candidate_markdown = refined_body + ("\n\n" + refs_md.strip() if refs_md.strip() else "")
            ok, guard_diag = _citation_guard_ok(assembled, candidate_markdown)
            lang_ok = _language_consistency_ok(refined_body)
            if ok and lang_ok:
                final_markdown = candidate_markdown
                _emit_progress(
                    state,
                    "global_refine_done",
                    {
                        "message": "已完成滑动窗口全文连贯性整合。",
                        "strategy": "sliding_window",
                        "windows": n_sections,
                        "window_errors": window_errors,
                        "open_gaps": len(aggregated_open_gaps),
                        "citation_guard": guard_diag,
                    },
                )
            else:
                logger.warning(
                    "Sliding window refine fallback (citation/lang guard): citation=%s, lang_ok=%s",
                    guard_diag,
                    lang_ok,
                )
                _emit_progress(
                    state,
                    "citation_guard_fallback",
                    {
                        "message": "滑动窗口整合后检测到语言/引用一致性风险，已回退到整合前版本。",
                        "guard": guard_diag,
                        "language_guard_ok": lang_ok,
                        "strategy": "sliding_window",
                    },
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

    # ── Persist final markdown + insights to Canvas ──
    if state.get("canvas_id"):
        try:
            from src.collaboration.canvas.canvas_manager import update_canvas
            update_kwargs: Dict[str, Any] = {
                "refined_markdown": final_markdown,
                "stage": "refine",
            }
            if has_insights:
                all_insight_texts = []
                for itype, items in insights_by_type.items():
                    for t in items[:10]:
                        all_insight_texts.append(f"[{itype}] {t}")
                update_kwargs["research_insights"] = all_insight_texts
            update_canvas(state["canvas_id"], **update_kwargs)
        except Exception:
            logger.debug("Failed to persist synthesis result to canvas", exc_info=True)

    _sync_outline_status_to_canvas(state)

    _emit_progress(
        state,
        "synthesize_done",
        {
            "sections_done": len(state.get("sections_completed", [])),
            "citations": len(state.get("citations", [])),
            "insights_consumed": sum(len(v) for v in insights_by_type.values()),
        },
    )
    _save_phase_checkpoint(state, "synthesize")
    _emit_progress(state, "checkpoint_saved", {
        "phase": "synthesize",
        "message": "Synthesis complete — final checkpoint saved.",
    })
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
    4. Diminishing-return plateau early stop

    When a guard fires (other than forced summarize), section is marked evidence_scarce
    if coverage is still below threshold, then routes directly to write.
    The last round is allowed to use T3 (last_round_max_tier=3 in preset), so there is
    no need for a separate Completion Round state machine.
    """
    dashboard = state.get("dashboard")
    if dashboard is None:
        return _write_or_claims(state)

    section = dashboard.get_section(state.get("current_section", ""))
    if section is None:
        return _write_or_claims(state)

    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    cov_threshold = float(preset.get("coverage_threshold", 0.6))

    def _mark_if_insufficient_and_write() -> str:
        if float(section.coverage_score or 0.0) < cov_threshold:
            section.evidence_scarce = True
            _emit_progress(state, "evidence_insufficient", {
                "section": section.title,
                "coverage": round(float(section.coverage_score or 0.0), 3),
                "message": "Evidence rounds exhausted below coverage threshold; proceeding to write with available evidence.",
            })
        return _write_or_claims(state)

    # Guard 0: forced summarize mode from cost monitor
    if bool(state.get("force_synthesize", False)):
        return "write"

    # Guard 1: global iteration cap
    max_iter = state.get("max_iterations", 30)
    if state.get("iteration_count", 0) >= max_iter:
        logger.info("Global iteration cap reached (%d), proceeding to write", max_iter)
        return _mark_if_insufficient_and_write()

    # Guard 2: per-section research round cap
    max_rounds = int(preset.get("max_section_research_rounds", 3))
    if section.research_rounds >= max_rounds:
        logger.info("Section '%s' hit per-section cap (%d rounds), proceeding to write", section.title, max_rounds)
        return _mark_if_insufficient_and_write()

    # Guard 3: coverage threshold reached — proceed to write
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
            _emit_progress(state, "coverage_plateau_early_stop", {
                "section": section.title,
                "coverage": round(float(history[-1]), 3),
                "gain_recent": round(gain_recent, 4),
                "gain_prev": round(gain_prev, 4),
                "message": "Coverage gain curve flattened; proceeding to write.",
            })
            return _write_or_claims(state)

    # Still have gaps and rounds remaining → continue gap-fill research
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
        logger.info(
            "Section '%s' sent back to research by verify (rewrite %d)",
            section.title, section.verify_rewrite_count,
        )
        return "research"

    next_section = dashboard.get_next_section()
    if next_section is None:
        # 第一轮（所有章节至少完成一次 write→verify）结束，清零全局迭代计数，
        # 使后续的「完善与提升」（如 review_gate、refine、或未来的补充研究）有独立预算
        if not state.get("iteration_count_reset_after_first_round"):
            state["iteration_count"] = 0
            state["iteration_count_reset_after_first_round"] = True
            preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
            max_r = int(preset.get("max_section_research_rounds", 3))
            max_rew = int(preset.get("max_verify_rewrite_cycles", 1))
            num_sec = len(dashboard.sections)
            state["max_iterations"] = (max_r + max_rew) * num_sec
            logger.info(
                "First round complete: iteration_count reset to 0, max_iterations=%d for refine/supplement phase",
                state["max_iterations"],
            )
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

def build_research_graph(include_scope_plan: bool = True, start_node: Optional[str] = None) -> StateGraph:
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
    else:
        allowed_entry = {"plan", "research", "write", "verify", "synthesize", "review_gate", "generate_claims"}
        resolved_entry = start_node if start_node in allowed_entry else "research"
        graph.set_entry_point(resolved_entry)
    graph.add_edge("plan", "research")
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
    preliminary_knowledge: str = "",
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
    max_sections: int = 4,
) -> DeepResearchState:
    resolved_depth = depth if depth in DEPTH_PRESETS else DEFAULT_DEPTH
    preset = get_depth_preset(resolved_depth)
    resolved_max_sections = _normalize_max_sections(max_sections, default=4)
    # max_iterations is set as a placeholder here; the real value is computed
    # dynamically in execute_deep_research() after num_sections is known:
    #   max_iterations = (max_section_research_rounds + max_verify_rewrite_cycles) × num_sections
    return {
        "topic": topic,
        "dashboard": ResearchDashboard(),
        "trajectory": ResearchTrajectory(topic=topic),
        "canvas_id": canvas_id or "",
        "session_id": session_id,
        "user_id": user_id,
        "search_mode": search_mode,
        "filters": filters or {},
        "write_top_k": (filters or {}).get("write_top_k"),
        "current_section": "",
        "sections_completed": [],
        "markdown_parts": [],
        "citations": [],
        "evidence_chunks": [],
        "section_evidence_pool": {},
        "evidence_chunk_empty_value": "",
        "citation_doc_key_map": {},
        "citation_existing_keys": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,  # placeholder; overridden in execute_deep_research()
        "max_sections": resolved_max_sections,
        "runtime_id": "",
        "model_override": model_override,
        "output_language": output_language,
        "clarification_answers": clarification_answers or {},
        "preliminary_knowledge": preliminary_knowledge or "",
        "user_context": user_context or "",
        "user_context_mode": user_context_mode or "supporting",
        "user_documents": user_documents or [],
        "step_models": step_models or {},
        "step_model_strict": bool(step_model_strict),
        "progress_callback": None,
        "cancel_check": None,
        "review_waiter": None,
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
    preliminary_knowledge: str = "",
    output_language: str = "auto",
    model_override: Optional[str] = None,
    step_models: Optional[Dict[str, Optional[str]]] = None,
    step_model_strict: bool = False,
    max_iterations: int = 30,
    max_sections: int = 4,
    progress_callback: Optional[Callable[[str, str, int], None]] = None,
) -> Dict[str, Any]:
    """Phase 1 only: scope + plan. Return outline and brief for confirmation."""
    def _emit(stage: str, message: str, percent: int) -> None:
        if progress_callback:
            try:
                progress_callback(stage, message, percent)
            except Exception:
                pass

    t_start_phase1 = time.perf_counter()
    _f = filters or {}
    logger.info(
        "[DR start] ▶ topic=%r | session=%s | search_mode=%s | model=%s"
        " | local_top_k=%s | step_top_k=%s | reranker_mode=%s | web_providers=%s | serpapi_ratio=%s"
        " | year=%s~%s | optimizer=%s | depth=n/a(start phase)",
        topic[:80], session_id[:12], search_mode, model_override or "default",
        _f.get("local_top_k"), _f.get("step_top_k"), _f.get("reranker_mode"),
        ",".join(_f["web_providers"]) if _f.get("web_providers") else "none",
        _f.get("serpapi_ratio"),
        _f.get("year_start"), _f.get("year_end"),
        _f.get("use_query_optimizer"),
    )
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
        preliminary_knowledge=preliminary_knowledge,
        step_models=None,  # Start phase: scope+plan always use the UI default model
        step_model_strict=False,
        max_sections=max_sections,
    )

    # Register llm_client so _resolve_runtime_llm_client can find it.
    # _build_initial_state leaves runtime_id empty; generate a temporary one.
    rid = f"dr-start-{session_id}-{int(time.time() * 1000)}"
    state["runtime_id"] = rid
    _register_runtime_llm_client(rid, llm_client)

    try:
        _emit("scoping", "正在分析研究范围和主题...", 10)
        t_scope = time.perf_counter()
        state = scoping_node(state)
        logger.info("[DR start] scope phase finished | elapsed_ms=%.0f", (time.perf_counter() - t_scope) * 1000.0)
        _emit("retrieval", "正在检索相关文献和资源...", 35)
        t_plan = time.perf_counter()
        state = plan_node(state)
        logger.info("[DR start] plan phase finished | elapsed_ms=%.0f", (time.perf_counter() - t_plan) * 1000.0)
        _emit("planning", "正在生成研究大纲框架...", 65)
        dashboard = state["dashboard"]
        _emit("finalizing", "正在完善研究规划...", 90)
        logger.info("[DR start] phase-1 finished | elapsed_ms=%.0f", (time.perf_counter() - t_start_phase1) * 1000.0)
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
    finally:
        _RUNTIME_LLM_CLIENTS.pop(rid, None)


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
    preliminary_knowledge: str = "",
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
    max_sections: int = 4,
    start_node: Optional[str] = None,
    initial_state_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare compiled graph runtime for deep-research execution/resume."""
    t0 = time.perf_counter()
    resolved_topic = topic.strip()
    preset = get_depth_preset(depth)
    graph = build_research_graph(include_scope_plan=False, start_node=start_node)
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
        preliminary_knowledge=preliminary_knowledge,
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
        max_sections=max_sections,
    )
    if initial_state_override:
        override = dict(initial_state_override)
        merged = {**initial_state, **override}
        dashboard_obj = merged.get("dashboard")
        if isinstance(dashboard_obj, dict):
            merged["dashboard"] = _build_dashboard_from_dict(dashboard_obj, fallback_topic=resolved_topic)
        elif not isinstance(dashboard_obj, ResearchDashboard):
            merged["dashboard"] = ResearchDashboard()
        trajectory_obj = merged.get("trajectory")
        if not isinstance(trajectory_obj, ResearchTrajectory):
            merged["trajectory"] = _build_trajectory_for_dashboard(
                str(merged.get("topic") or resolved_topic),
                merged["dashboard"],
            )
        # Runtime callbacks should always use the current process context.
        merged["progress_callback"] = progress_callback
        merged["cancel_check"] = cancel_check
        merged["review_waiter"] = review_waiter
        merged["job_id"] = job_id
        if model_override is not None:
            merged["model_override"] = model_override
        initial_state = merged

    _emit_progress(initial_state, "depth_resolved", {"depth": depth, "preset": preset})
    dashboard = initial_state["dashboard"]
    if not initial_state_override:
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
    else:
        outline = [s.title for s in dashboard.sections] or [resolved_topic]

    max_research_rounds = int(preset.get("max_section_research_rounds", 3))
    max_rewrite_cycles = int(preset.get("max_verify_rewrite_cycles", 1))
    iter_per_sec = max_research_rounds + max_rewrite_cycles
    num_sections = len(outline)
    scaled_max_iter = iter_per_sec * num_sections
    initial_state["max_iterations"] = scaled_max_iter
    logger.info(
        "Depth=%s | sections=%d | max_iterations=%d (%d rounds + %d rewrites × %d sections)",
        depth, num_sections, scaled_max_iter, max_research_rounds, max_rewrite_cycles, num_sections,
    )
    if not initial_state_override:
        initial_state["markdown_parts"] = [f"# {dashboard.brief.topic}\n"]

    if initial_state.get("canvas_id") and not initial_state_override:
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
    initial_state["runtime_id"] = thread_id
    _register_runtime_llm_client(thread_id, llm_client)
    _register_runtime_callbacks(thread_id, {
        "progress_callback": progress_callback,
        "cancel_check": cancel_check,
        "review_waiter": review_waiter,
    })
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
    max_sections: int = 4,
) -> Dict[str, Any]:
    """Phase 2: run research loop with confirmed brief/outline."""
    _f = filters or {}
    logger.info(
        "[DR execute] ▶ topic=%r | session=%s | job=%s | depth=%s | search_mode=%s | model=%s"
        " | local_top_k=%s | step_top_k=%s | reranker_mode=%s | web_providers=%s | serpapi_ratio=%s"
        " | year=%s~%s | optimizer=%s | sections=%d",
        topic[:80], session_id[:12], job_id[:16] if job_id else "", depth, search_mode, model_override or "default",
        _f.get("local_top_k"), _f.get("step_top_k"), _f.get("reranker_mode"),
        ",".join(_f["web_providers"]) if _f.get("web_providers") else "none",
        _f.get("serpapi_ratio"),
        _f.get("year_start"), _f.get("year_end"),
        _f.get("use_query_optimizer"),
        len(confirmed_outline) if confirmed_outline else 0,
    )
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
        max_sections=max_sections,
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

    # ── Checkpoint: confirmed outline (recovery anchor for the entire run) ──
    _save_phase_checkpoint(initial_state, "confirmed")
    _emit_progress(initial_state, "checkpoint_saved", {
        "phase": "confirmed",
        "message": "Outline confirmed — checkpoint saved.",
    })

    try:
        compiled.invoke(initial_state, config=config)
        state_snapshot = compiled.get_state(config)
    except Exception as e:
        logger.error(f"Deep Research execution failed: {e}")
        # ── Crash checkpoint: save whatever progress was made ──
        try:
            crash_state = getattr(compiled.get_state(config), "values", None) or initial_state
            if isinstance(crash_state, dict):
                crash_state["error"] = str(e)
                _save_phase_checkpoint(crash_state, "crash")
                _emit_progress(crash_state, "checkpoint_saved", {
                    "phase": "crash",
                    "message": f"Crash checkpoint saved — can resume from last completed section.",
                })
        except Exception:
            logger.debug("Failed to save crash checkpoint", exc_info=True)
        _cleanup_shared_browser_if_no_active_jobs(job_id)
        return {
            "markdown": f"# {topic}\n\nDeep Research execution failed: {e}",
            "canvas_id": canvas_id or "",
            "outline": runtime.get("outline") or [],
            "citations": [],
            "dashboard": {},
            "total_time_ms": (time.perf_counter() - float(runtime.get("started_at_perf") or time.perf_counter())) * 1000,
        }

    if getattr(state_snapshot, "next", ()):
        _cleanup_shared_browser_if_no_active_jobs(job_id)
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
    _cleanup_shared_browser_if_no_active_jobs(job_id)
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
    max_sections: int = 4,
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
        max_sections=max_sections,
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
        max_sections=max_sections,
    )


def optimize_section_evidence(
    job_id: str,
    section_title: str,
    llm_client: Any,
    search_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """User-triggered evidence optimization for a specific section.

    Loads the checkpoint for the given job, then executes a full T1+T2+T3 search
    using LLM-refined queries built from the section's accumulated evidence and gaps.
    All providers are used unconditionally (max_tier=3, no early-exit).

    This is the backend for the 'Evidence Optimization' button in the Research
    Progress Panel.  It can be called after writing is complete to enrich evidence
    for a section that had evidence_scarce=True or otherwise needs supplementation.

    Returns:
        {
            "section_title": str,
            "new_chunks": int,      # chunks retrieved in this optimization pass
            "sources_used": list[str],
        }
    """
    from src.collaboration.research.job_store import load_checkpoint

    checkpoint = load_checkpoint(job_id)
    if checkpoint is None:
        return {"error": f"No checkpoint found for job '{job_id}'"}

    # Reconstruct minimal state for retrieval
    state = checkpoint.get("state") or {}
    rid = str(state.get("runtime_id") or state.get("job_id") or "").strip()
    if rid:
        _register_runtime_llm_client(rid, llm_client)
        if progress_callback is not None:
            _register_runtime_callbacks(rid, {"progress_callback": progress_callback})
    else:
        state["llm_client"] = llm_client
        if progress_callback is not None:
            state["progress_callback"] = progress_callback

    dashboard = state.get("dashboard")
    if dashboard is None:
        return {"error": "Checkpoint has no dashboard"}

    section = dashboard.get_section(section_title)
    if section is None:
        return {"error": f"Section '{section_title}' not found in checkpoint"}

    preset = state.get("depth_preset") or get_depth_preset(state.get("depth", DEFAULT_DEPTH))
    svc = _get_retrieval_svc(state)
    effective_filters = dict(filters or state.get("filters") or {})
    effective_filters["use_query_optimizer"] = False

    ui_providers = effective_filters.get("web_providers")
    ui_allowed = set(ui_providers) if ui_providers else None
    ui_fetcher = effective_filters.get("use_content_fetcher", "auto")
    ui_source_configs = effective_filters.get("web_source_configs")

    _emit_progress(state, "evidence_optimization_start", {
        "section": section_title,
        "message": f"Evidence Optimization: full T1+T2+T3 for section '{section_title}'",
    })

    # Build fresh queries from accumulated evidence + known gaps
    n_refined = max(int(preset.get("tier3_refined_queries", 2)), 2)
    accumulated = list(state.get("evidence_chunks") or [])
    queries = _generate_section_queries(state, section, max_queries=8)

    all_chunks, all_sources = _execute_tiered_search(
        state, section, queries, svc, effective_filters, preset,
        max_tier=3,
        ui_allowed_providers=ui_allowed,
        ui_content_fetcher=ui_fetcher,
        ui_source_configs=ui_source_configs,
    )

    # Supplement with a dedicated T3 refined pass using full accumulated evidence
    t3_providers = _filter_by_ui(["scholar", "google"], ui_allowed)
    if t3_providers:
        refined = _generate_refined_queries(state, section, accumulated + all_chunks, max_queries=n_refined)
        _emit_progress(state, "evidence_optimization_t3", {
            "section": section_title,
            "refined_queries": refined,
            "message": f"Evidence Optimization T3 refined: {refined}",
        })
        for q in refined:
            f = dict(effective_filters)
            f["web_providers"] = t3_providers
            f["reranker_mode"] = "bge_only"  # 强制使用 bge_only 加快速度
            if ui_source_configs:
                f["web_source_configs"] = ui_source_configs
            f["use_content_fetcher"] = _resolve_fetcher(ui_fetcher, t3_providers)
            default_k = int(effective_filters.get("step_top_k") or preset.get("default_per_provider_top_k", 15))
            pack = svc.search(query=q, mode=search_mode, top_k=default_k, filters=f)
            all_chunks.extend(pack.chunks)
            all_sources.update(pack.sources_used)

    _accumulate_evidence_chunks(state, all_chunks)

    _emit_progress(state, "evidence_optimization_done", {
        "section": section_title,
        "new_chunks": len(all_chunks),
        "sources_used": sorted(all_sources),
        "message": f"Evidence Optimization complete: {len(all_chunks)} new chunks retrieved.",
    })

    return {
        "section_title": section_title,
        "new_chunks": len(all_chunks),
        "sources_used": sorted(all_sources),
    }
