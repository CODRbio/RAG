"""
显式指令定义与参数解析 + 统一上下文分析。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set

from .parser import ParsedIntent
from src.utils.prompt_manager import PromptManager
from src.log import get_logger

_pm = PromptManager()
_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ContextAnalysis: 单次 ultra-lite LLM 调用的结果
# ---------------------------------------------------------------------------

@dataclass
class ContextAnalysis:
    """Single ultra-lite LLM call result for multi-turn context management."""

    action: str = "rag"  # chat / rag / deep_research
    context_status: str = "self_contained"  # self_contained / resolved / needs_clarification
    rewritten_query: str = ""
    clarification: str = ""


def analyze_chat_context(
    message: str,
    rolling_summary: str,
    history: Optional[Iterable[Any]] = None,
    llm_client: Any = None,
    max_history_turns: int = 6,
) -> ContextAnalysis:
    """
    Single ultra-lite LLM call that combines:
      - Intent classification (chat / rag / deep_research)
      - Reference / pronoun detection (LLM-based, not regex)
      - Query rewriting when references are resolvable
      - Clarification question when references are not resolvable

    Replaces the old pipeline of: IntentParser(NL) + _classify_query + regex _has_unresolved_references + get_clarification_if_unresolved.
    """
    if not llm_client:
        return ContextAnalysis(action="rag")

    history_block = _format_history_block(history, max_turns=max_history_turns)

    prompt = _pm.render(
        "chat_context_analyze.txt",
        rolling_summary=rolling_summary or "（首轮对话）",
        history=history_block or "（首轮对话）",
        message=message,
    )

    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("chat_context_analyze_system.txt")},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.get("final_text") or "").strip()
    except Exception as exc:
        _logger.debug("analyze_chat_context LLM failed: %s", exc)
        return ContextAnalysis(action="rag")

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        _logger.debug("analyze_chat_context JSON parse failed: raw=%r", raw[:200])
        return ContextAnalysis(action="rag")

    action = (data.get("action") or "rag").strip().lower()
    if action not in ("chat", "rag", "deep_research"):
        action = "rag"

    context_status = (data.get("context_status") or "self_contained").strip().lower()
    if context_status not in ("self_contained", "resolved", "needs_clarification"):
        context_status = "self_contained"

    return ContextAnalysis(
        action=action,
        context_status=context_status,
        rewritten_query=(data.get("rewritten_query") or "").strip(),
        clarification=(data.get("clarification") or "").strip(),
    )


# ---------------------------------------------------------------------------
# 查询与本地库范围匹配检查（ultra-lite）
# ---------------------------------------------------------------------------

# 本地库名称 -> 简短范围描述，用于 LLM 判断 query 是否明显不符
COLLECTION_SCOPE: dict = {
    "DeepSea_symbiosis": "深海共生、海洋生物、共生关系、深海生态与相关研究",
    "deepsea_life": "深海生命、海洋生物、深海生态",
    "deepsea_ocean": "深海海洋、海洋环境",
    "deepsea_env": "深海环境、海洋环境",
}


def check_query_collection_scope(
    collection_name: str,
    query: str,
    llm_client: Any,
) -> str:
    """
    判断当前查询是否与本地库范围明显不符。
    返回 "match" | "mismatch" | "unclear"。
    优先使用建库/刷新时生成的 scope 摘要（collection_scope 模块），否则回退到内置 COLLECTION_SCOPE。
    """
    if not (collection_name and query and llm_client):
        return "match"
    try:
        from src.indexing.collection_scope import get_scope
        scope = get_scope(collection_name.strip())
    except Exception:
        scope = None
    if not scope:
        scope = COLLECTION_SCOPE.get(
            collection_name.strip(),
            collection_name or "本地知识库",
        )
    if isinstance(scope, str) and scope == (collection_name or "").strip():
        scope = f"主题：{collection_name}"
    prompt = _pm.render(
        "chat_local_scope_check.txt",
        collection_name=collection_name,
        scope_description=scope,
        query=query[:500],
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("chat_local_scope_check_system.txt")},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.get("final_text") or "").strip().lower()
    except Exception:
        return "match"
    if "mismatch" in raw:
        return "mismatch"
    if "unclear" in raw:
        return "unclear"
    return "match"


# ---------------------------------------------------------------------------
# 原有 query 构建逻辑（保留，供 rag 分支使用）
# ---------------------------------------------------------------------------

def get_search_query_from_intent(parsed: ParsedIntent, fallback: str) -> str:
    """
    从解析结果中取出用于检索的 query。
    显式指令时用 params["args"]，否则用 fallback（原始用户输入）。
    """
    if parsed.from_command and parsed.params:
        args = (parsed.params.get("args") or "").strip()
        if args:
            return args
    params_query = (parsed.params.get("query") if parsed.params else None) or ""
    if isinstance(params_query, str) and params_query.strip():
        return params_query.strip()
    return fallback


def build_search_query_from_context(
    parsed: ParsedIntent,
    fallback: str,
    history: Optional[Iterable[Any]] = None,
    max_user_turns: int = 3,
    max_len: int = 256,
    llm_client: Optional[Any] = None,
    enforce_english_if_input_english: bool = True,
    rolling_summary: str = "",
) -> str:
    """
    Build retrieval query with current-focus enforcement.

    If the caller already resolved references (e.g. via analyze_chat_context),
    pass the rewritten_query as ``fallback`` — the function will detect it is
    self-contained and skip the expensive LLM rewrite step.
    """
    base = get_search_query_from_intent(parsed, fallback)
    if parsed.from_command:
        return base

    needs_context = _has_unresolved_references(base)
    if not needs_context or not llm_client:
        return _truncate_query(base, max_len=max_len)

    recent_user_turns = _extract_recent_user_inputs(history, max_user_turns=max_user_turns)
    query = _generate_focused_query(
        llm_client=llm_client,
        current_msg=base,
        rolling_summary=rolling_summary,
        recent_user_turns=recent_user_turns[-2:],
        output_english=bool(enforce_english_if_input_english and base and not _is_chinese(base)),
        max_len=max_len,
    )

    if not query:
        return _truncate_query(base, max_len=max_len)
    if not _validates_current_focus(base, query):
        return _truncate_query(base, max_len=max_len)
    if enforce_english_if_input_english and base and not _is_chinese(base) and _is_chinese(query):
        return _truncate_query(base, max_len=max_len)
    return _truncate_query(query, max_len=max_len)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_recent_user_inputs(
    history: Optional[Iterable[Any]],
    max_user_turns: int = 3,
) -> List[str]:
    if not history:
        return []
    user_inputs: List[str] = []
    for t in history:
        role = getattr(t, "role", None)
        content = getattr(t, "content", None)
        if role == "user" and isinstance(content, str):
            user_inputs.append(content)
    if max_user_turns > 0:
        user_inputs = user_inputs[-max_user_turns:]
    return user_inputs


_REFERENCE_PATTERNS = re.compile(
    r"\b(it|this|that|these|those|them|its|the above|above|former|latter)\b"
    r"|前面|上面|之前|这个|那个|它|它们|那些|这些|该|此",
    re.IGNORECASE,
)


def _has_unresolved_references(text: str) -> bool:
    """Lightweight regex pre-filter for build_search_query_from_context."""
    return bool(_REFERENCE_PATTERNS.search(text or ""))


def _extract_content_words(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{2,}", text or "")
    if not tokens:
        return []
    stopwords: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for",
        "and", "or", "with", "about", "how", "what", "why", "when", "where", "which", "who",
        "do", "does", "did", "can", "could", "would", "should", "please", "compare",
    }
    zh_stopwords: Set[str] = {"这个", "那个", "一下", "什么", "怎么", "如何", "以及", "还有", "是否", "可以", "请问", "请"}
    out: List[str] = []
    for t in tokens:
        low = t.lower()
        if low in stopwords or t in zh_stopwords:
            continue
        if len(low) <= 1 and re.match(r"[a-zA-Z]", low):
            continue
        out.append(low)
    return out


def _validates_current_focus(current_msg: str, generated_query: str) -> bool:
    """Require overlap with current message to avoid topic drift."""
    current_tokens = set(_extract_content_words(current_msg.lower()))
    query_tokens = set(_extract_content_words(generated_query.lower()))
    if not current_tokens:
        return True
    overlap = current_tokens & query_tokens
    return (len(overlap) / max(1, len(current_tokens))) >= 0.3


def _truncate_query(text: str, max_len: int = 256) -> str:
    s = (text or "").strip()
    if max_len and len(s) > max_len:
        return s[:max_len].rstrip()
    return s


def _is_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _format_history_block(
    history: Optional[Iterable[Any]], max_turns: int = 6, max_len: int = 150,
) -> str:
    """Format recent turns for the context-analysis prompt."""
    if not history:
        return ""
    turns = list(history)
    if max_turns > 0:
        turns = turns[-max_turns:]
    lines: List[str] = []
    for t in turns:
        role = getattr(t, "role", "") or ""
        content = getattr(t, "content", "") or ""
        if not isinstance(content, str):
            continue
        text = content.strip().replace("\n", " ")
        if max_len and len(text) > max_len:
            text = text[:max_len] + "…"
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _generate_focused_query(
    llm_client: Any,
    current_msg: str,
    rolling_summary: str,
    recent_user_turns: List[str],
    output_english: bool,
    max_len: int = 256,
) -> str:
    recent_block = "\n".join(
        f"- {c.strip()}" for c in recent_user_turns if isinstance(c, str) and c.strip()
    )
    language_instruction = "English" if output_english else "the same language as current user question"
    system_content = (
        "Output only a single English search query."
        if output_english
        else "Output only a single search query."
    )
    prompt = _pm.render(
        "intent_search_query.txt",
        rolling_summary=rolling_summary or "(first message in session)",
        recent_block=recent_block or "(none)",
        current_msg=current_msg,
        language_instruction=language_instruction,
    )
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.get("final_text") or "").strip()
    except Exception:
        return ""
    text = _truncate_query(text.strip().strip('"').strip("'"), max_len=max_len)
    if not text:
        return ""
    return text
