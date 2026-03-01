"""
Context length limits and summarization when limits are exceeded.

Policy: use full content when under limit; when over limit, summarize via ultra_lite
before passing to the main LLM. Final integration (write_top_k / evidence assembly)
uses a 200k hard cap with logging when truncated.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Limit constants (characters) used across the codebase
AUTO_COMPLETE_CONTEXT_MAX_CHARS = 30_000
DR_USER_CONTEXT_MAX_CHARS = 40_000
DR_SECTION_EVIDENCE_MAX_CHARS = 70_000
CLAIM_VERIFICATION_MAX_CHARS = 70_000
COLLECTION_SCOPE_MAX_CHARS = 40_000
ENTITY_EXTRACTOR_MAX_CHARS = 15_000
FINAL_INTEGRATION_MAX_CHARS = 200_000
SESSION_MEMORY_TURN_MAX_CHARS = 20_000
CANVAS_REF_PREVIEW_MAX_CHARS = 500
QUERY_MAX_CHARS = 2_500
# Scope/plan prompt injection for preliminary knowledge block
PRELIMINARY_KNOWLEDGE_MAX_CHARS = 3_000
# plan_node retrieval context: keep rich context up to 30k chars,
# then summarize down to <=10k via ultra-lite when exceeded.
PLAN_CONTEXT_MAX_CHARS = 30_000
PLAN_CONTEXT_SUMMARIZE_TO = 10_000


def summarize_if_needed(
    text: str,
    max_chars: int,
    llm_client: Any = None,
    ultra_lite_provider: Optional[str] = None,
    purpose: str = "context",
) -> str:
    """
    If text is under max_chars, return as-is. Otherwise summarize via ultra_lite
    and return the summary. On summarization failure, truncate and log.

    Callers can pass llm_client (e.g. ultra_lite client) or leave None to resolve
    via get_manager().get_ultra_lite_client(ultra_lite_provider).
    """
    if not text or not isinstance(text, str):
        return text or ""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    client = llm_client
    if client is None:
        try:
            from src.llm.llm_manager import get_manager
            from pathlib import Path
            _config = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
            manager = get_manager(str(_config))
            client = manager.get_ultra_lite_client(ultra_lite_provider)
        except Exception as e:
            logger.warning("summarize_if_needed: could not get ultra_lite client, truncating: %s", e)
            out = text[:max_chars] + ("..." if len(text) > max_chars else "")
            logger.info("context_limits: truncated %s to %d chars (purpose=%s)", purpose, len(out), purpose)
            return out

    try:
        from src.utils.prompt_manager import PromptManager
        _pm = PromptManager()
        # Pass a prefix of the text to stay within typical model context
        input_slice = text[:150_000] if len(text) > 150_000 else text
        prompt = _pm.render(
            "context_summarize_ultra_lite.txt",
            text=input_slice,
            max_chars=max_chars,
        )
        resp = client.chat(
            messages=[
                {"role": "system", "content": _pm.load("context_summarize_ultra_lite_system.txt").strip()},
                {"role": "user", "content": prompt},
            ],
        )
        summary = (resp.get("final_text") or "").strip()
        if summary and len(summary) <= max_chars * 2:
            logger.info(
                "context_limits: summarized %s from %d to %d chars (purpose=%s)",
                purpose, len(text), len(summary), purpose,
            )
            return summary[:max_chars] if len(summary) > max_chars else summary
    except Exception as e:
        logger.warning("summarize_if_needed LLM failed (%s), truncating: %s", purpose, e)

    out = text[:max_chars] + ("..." if len(text) > max_chars else "")
    logger.info("context_limits: truncated %s to %d chars (purpose=%s)", purpose, len(out), purpose)
    return out


def cap_and_log(text: str, max_chars: int = FINAL_INTEGRATION_MAX_CHARS, purpose: str = "integration") -> str:
    """
    Hard cap text at max_chars for final integration (e.g. write_top_k evidence).
    Log when truncation occurs.
    """
    if not text or len(text) <= max_chars:
        return text or ""
    logger.info(
        "context_limits: final integration truncated from %d to %d chars (purpose=%s)",
        len(text), max_chars, purpose,
    )
    return text[:max_chars]
