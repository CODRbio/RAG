"""Prompt asset manager — singleton that loads and caches .txt prompt templates."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptManager:
    """Singleton prompt template manager.

    Loads ``.txt`` templates from ``src/prompts/``, renders them with
    ``str.format(**kwargs)``, and keeps loaded templates in an in-memory cache
    so each file is read from disk at most once per process lifetime.

    Usage::

        pm = PromptManager()
        text = pm.render("scope_research.txt", topic="deep-sea biology", clarification_block="...")
    """

    _instance: PromptManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "PromptManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._cache: Dict[str, str] = {}
                    cls._instance = inst
        return cls._instance

    def render(self, template_name: str, **kwargs: object) -> str:
        """Render a prompt template by name.

        Args:
            template_name: File name relative to ``src/prompts/`` (e.g. ``"scope_research.txt"``).
            **kwargs: Variables substituted into the template via ``str.format``.

        Returns:
            Rendered prompt string.
        """
        if template_name not in self._cache:
            path = _PROMPTS_DIR / template_name
            self._cache[template_name] = path.read_text(encoding="utf-8")
        return self._cache[template_name].format(**kwargs)

    def invalidate(self, template_name: str | None = None) -> None:
        """Clear one or all cached templates (useful in tests or hot-reload scenarios)."""
        if template_name is None:
            self._cache.clear()
        else:
            self._cache.pop(template_name, None)
