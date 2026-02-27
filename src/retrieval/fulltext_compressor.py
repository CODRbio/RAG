"""
Full-text evidence compressor.

After web full-text fetch, hits with content_type=="full_text" and word count > threshold
are compressed via a cheap LLM to <= max_output_words. Short texts pass through unchanged.
Rerank happens before compression so we only compress hits that will enter the context.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WORD_THRESHOLD_DEFAULT = 300
MAX_OUTPUT_WORDS_DEFAULT = 400
FALLBACK_TRUNCATE_CHARS = 2000
MAX_CONCURRENT_DEFAULT = 5
LLM_MAX_TOKENS = 800


def _word_count(text: str) -> int:
    """Approximate word count (whitespace-separated tokens)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def compress_evidence_text_sync(
    text: str,
    query: str,
    llm_client: Any,
    *,
    title: str = "(no title)",
    url: str = "",
    max_output_words: int = MAX_OUTPUT_WORDS_DEFAULT,
    fallback_chars: int = FALLBACK_TRUNCATE_CHARS,
) -> str:
    """
    Compress a single evidence text into a short, query-focused summary.

    Returns compressed text on success. On any failure, falls back to truncated
    original text so callers always get usable content.
    """
    content = (text or "").strip()
    if not content:
        return ""
    try:
        from src.utils.prompt_manager import PromptManager
        _pm = PromptManager()
        system = _pm.load("compress_fulltext_system.txt").strip()
        user = _pm.render(
            "compress_fulltext.txt",
            query=query or "",
            title=(title or "").strip() or "(no title)",
            url=(url or "").strip(),
            full_text=content[:12000],
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=LLM_MAX_TOKENS,
        )
        summary = (resp.get("final_text") or "").strip()
        if summary and _word_count(summary) <= max_output_words * 2:
            return summary
        raise ValueError("empty or too long summary")
    except Exception as e:
        logger.debug("evidence text compress failed, truncating: %s", e)
        return content[:fallback_chars] + ("..." if len(content) > fallback_chars else "")


def _compress_one_hit(
    hit: Dict[str, Any],
    query: str,
    llm_client: Any,
    *,
    max_output_words: int = MAX_OUTPUT_WORDS_DEFAULT,
    fallback_chars: int = FALLBACK_TRUNCATE_CHARS,
) -> None:
    """
    Mutate hit: replace hit["content"] with LLM-compressed summary (<= max_output_words).
    On failure, truncate to fallback_chars. Set metadata content_type and original_full_text_chars.
    """
    content = (hit.get("content") or "").strip()
    if not content:
        return
    metadata = hit.get("metadata") or {}
    title = (metadata.get("title") or "").strip() or "(no title)"
    url = (metadata.get("url") or "").strip() or ""

    compressed = compress_evidence_text_sync(
        content,
        query,
        llm_client,
        title=title,
        url=url,
        max_output_words=max_output_words,
        fallback_chars=fallback_chars,
    )
    hit["content"] = compressed
    metadata["content_type"] = "full_text_compressed"
    metadata["original_full_text_chars"] = len(content)
    hit["metadata"] = metadata
    logger.debug("compressed fulltext to %d words (was %d)", _word_count(compressed), _word_count(content))


def compress_fulltext_hits_sync(
    hits: List[Dict[str, Any]],
    query: str,
    llm_client: Any,
    *,
    word_threshold: int = WORD_THRESHOLD_DEFAULT,
    max_output_words: int = MAX_OUTPUT_WORDS_DEFAULT,
    fallback_chars: int = FALLBACK_TRUNCATE_CHARS,
    max_concurrent: int = MAX_CONCURRENT_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    In-place compress hits where metadata.content_type == "full_text" and word count > word_threshold.
    Uses llm_client (sync) with up to max_concurrent parallel calls. Returns the same list (mutated).
    """
    to_compress = []
    for h in hits:
        meta = h.get("metadata") or {}
        if meta.get("content_type") != "full_text":
            continue
        text = (h.get("content") or "").strip()
        if _word_count(text) <= word_threshold:
            continue
        to_compress.append(h)

    if not to_compress:
        return hits

    if max_concurrent <= 1:
        for h in to_compress:
            _compress_one_hit(h, query, llm_client, max_output_words=max_output_words, fallback_chars=fallback_chars)
        return hits

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(
                _compress_one_hit,
                h,
                query,
                llm_client,
                max_output_words=max_output_words,
                fallback_chars=fallback_chars,
            ): h
            for h in to_compress
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("fulltext compress task failed: %s", e)
    return hits
