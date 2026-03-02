"""
DEPRECATED: Moved to backup as of Chat/Research 1+1+1 unification.
The codebase no longer uses SmartQueryOptimizer or this module.
Use structured 1+1+1 query generation (e.g. chat_generate_queries / generate_queries) instead.

Provider-specific query optimizer for web search.
"""

import re
from typing import Tuple


_QWORDS_EN = {
    "what", "why", "how", "when", "where", "which", "who",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "may", "might",
}

_QWORDS_ZH = {
    "什么", "为何", "为什么", "如何", "怎么", "怎样", "是否", "可以", "能否", "原理", "机制",
}


def _normalize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_question(text: str) -> Tuple[str, bool]:
    """
    Remove common question leading words and punctuation.
    Returns (keywords, was_question).
    """
    t = _normalize(text)
    if not t:
        return "", False
    was_question = "?" in t or "？" in t

    # English question prefix
    lowered = t.lower()
    if any(lowered.startswith(w + " ") for w in _QWORDS_EN):
        was_question = True
        lowered = re.sub(r"^(what|why|how|when|where|which|who|is|are|was|were|do|does|did|can|could|should|would|may|might)\b\s*", "", lowered)
        t = lowered

    # Chinese question prefix
    for w in sorted(_QWORDS_ZH, key=len, reverse=True):
        if t.startswith(w):
            was_question = True
            t = t[len(w):].lstrip()
            break

    # Remove punctuation
    t = re.sub(r"[?？。．，,;；:：!！()（）\[\]{}\"']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, was_question


def optimize_query(provider: str, query: str) -> str:
    """
    Provider-specific query optimization.
    """
    q = _normalize(query)
    if not q:
        return q
    keywords, was_question = _strip_question(q)
    provider = (provider or "").lower()

    if provider in ("scholar", "semantic"):
        base = keywords or q
        # Do not force "review"/"survey" suffixes for academic engines.
        # They often bias results toward generic surveys and reduce snippet specificity.
        # 不再加双引号，避免 URL 编码导致 Scholar/Playwright 检索失败
        return base

    if provider == "google":
        base = keywords or q
        # Keep Google queries concise; avoid forcing "overview".
        return base

    if provider == "tavily":
        base = keywords or q
        if was_question and "overview" not in base.lower():
            base = f"{base} overview"
        return base

    return q
