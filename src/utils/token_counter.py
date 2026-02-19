"""
Token counting utilities for pre-call budget estimation.

Uses tiktoken (cl100k_base) as a unified approximation across all supported
providers (Claude, DeepSeek, Gemini, Kimi). The encoding error margin is well
under 15%, which is absorbed by the safety_margin parameter in compute_safe_budget.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Encoder — lazy-loaded to avoid import cost when module is first imported
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return approximate token count for *text* using cl100k_base encoding."""
    if not text:
        return 0
    try:
        enc = _get_encoder()
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        # Graceful fallback: character-based heuristic (same as trajectory.py)
        return int(len(text) * 0.4)


# ---------------------------------------------------------------------------
# Model context-window registry
# ---------------------------------------------------------------------------

# Values represent usable input context (not output limit).
# Sources: official provider documentation as of 2026-02.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude family
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-opus-4-6-20260205": 200_000,
    # DeepSeek family
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
    "deepseek-v3": 64_000,
    # Gemini family
    "gemini-pro-latest": 1_000_000,
    "gemini-flash-latest": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    # Kimi / Moonshot family
    "kimi-k2.5": 128_000,
    "moonshot-v1-128k": 128_000,
    # OpenAI family
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
}

# Safe default when model is unknown
DEFAULT_CONTEXT_WINDOW = 64_000

# Maximum output tokens we ever request (hard ceiling across all providers)
DEFAULT_MAX_OUTPUT_TOKENS = 8_192

# Minimum output budget we will ever pass to the LLM (prevents starving output)
MIN_OUTPUT_TOKENS = 512


def get_context_window(model: Optional[str]) -> int:
    """Return the context-window size for *model*, falling back to the default."""
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    # Exact match first
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Prefix / substring match for versioned model names
    m = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key in m or m.startswith(key.split("-")[0]):
            return window
    return DEFAULT_CONTEXT_WINDOW


def compute_safe_budget(
    prompt_tokens: int,
    context_window: int,
    safety_margin: float = 0.10,
    max_output: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> int:
    """
    Return the max output tokens we can safely request given the prompt size.

    Formula:
        available = floor(context_window * (1 - safety_margin)) - prompt_tokens
        clamped to [MIN_OUTPUT_TOKENS, max_output]

    The safety_margin reserves headroom for tokenizer approximation errors and
    any internal overhead tokens added by the provider SDK.
    """
    usable_context = int(context_window * (1.0 - safety_margin))
    available = usable_context - prompt_tokens
    return max(MIN_OUTPUT_TOKENS, min(max_output, available))


def needs_sliding_window(
    prompt_tokens: int,
    context_window: int,
    safety_margin: float = 0.10,
    min_output_tokens: int = 1024,
) -> bool:
    """
    Return True when the remaining output budget after placing the prompt would
    be smaller than *min_output_tokens*, indicating we cannot fit the full
    document rewrite in a single LLM call.
    """
    budget = compute_safe_budget(prompt_tokens, context_window, safety_margin)
    return budget < min_output_tokens
