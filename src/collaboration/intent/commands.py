"""
显式指令定义与参数解析。
"""

import re
from typing import Any, Iterable, List, Optional, Set

from .parser import ParsedIntent
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


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
    Build retrieval query with current-focus enforcement:
    - explicit command first
    - only use context when current message has unresolved references
    - validate generated query still overlaps current message semantics
    """
    base = get_search_query_from_intent(parsed, fallback)
    if parsed.from_command:
        return base

    # Self-contained input should not be polluted by history.
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
    r"\b(it|this|that|these|those|them|its|the above|above|former|latter)\b|前面|上面|之前|这个|那个|它|它们",
    re.IGNORECASE,
)


def _has_unresolved_references(text: str) -> bool:
    """Detect pronouns/references that require session context."""
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
            max_tokens=64,
        )
        text = (resp.get("final_text") or "").strip()
    except Exception:
        return ""
    text = _truncate_query(text.strip().strip('"').strip("'"), max_len=max_len)
    if not text:
        return ""
    return text
