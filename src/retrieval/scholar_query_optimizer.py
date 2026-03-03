"""
Scholar search query optimizer: rewrites a user query for a given data source
using ultra-lite LLM and per-source rules (see prompts/scholar_query_optimize.txt).
"""

from typing import Optional

from src.log import get_logger
from src.llm.llm_manager import get_manager
from src.utils.prompt_manager import PromptManager

logger = get_logger(__name__)
_pm = PromptManager()

VALID_SOURCES = frozenset({
    "google_scholar",
    "google",
    "semantic",
    "semantic_relevance",
    "semantic_bulk",
    "ncbi",
    "annas_archive",
})

# Map legacy "semantic" to relevance rules
_SOURCE_PROMPT_MAP = {
    "semantic": "semantic_relevance",
}


def optimize_scholar_query(
    query: str,
    source: str,
    ultra_lite_provider: Optional[str] = None,
) -> str:
    """
    Optimize a user query for the given scholar search source using ultra-lite LLM.
    Returns the optimized query string; on failure returns the original query.
    """
    query = (query or "").strip()
    if not query:
        return query
    source = (source or "google_scholar").strip().lower()
    if source not in VALID_SOURCES:
        logger.info("scholar_query_optimize: unknown source %r, skipping optimization", source)
        return query
    prompt_source = _SOURCE_PROMPT_MAP.get(source, source)

    try:
        manager = get_manager()
        client = manager.get_ultra_lite_client(ultra_lite_provider)
    except Exception as e:
        logger.warning("scholar_query_optimize: could not get ultra_lite client: %s", e)
        return query

    prompt = _pm.render("scholar_query_optimize.txt", source=prompt_source, query=query)
    try:
        resp = client.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.get("final_text") or "").strip()
    except Exception as e:
        logger.warning("scholar_query_optimize: LLM call failed: %s", e)
        return query

    optimized = raw.split("\n")[0].strip() if raw else ""
    if not optimized:
        return query
    logger.info(
        "scholar_query_optimize: source=%s original=%r optimized=%r",
        prompt_source, query[:80] + ("..." if len(query) > 80 else ""), optimized[:80] + ("..." if len(optimized) > 80 else "")
    )
    return optimized
