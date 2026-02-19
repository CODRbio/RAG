"""
Minimal tracing integration tests:
- disabled mode: traceable is a no-op fallback
- enabled mode: importing react_loop with run_type="agent" raises no warning
"""

import importlib
import sys
import warnings
from pathlib import Path

import pytest


# Support direct execution: `python tests/test_tracing_modes.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reload_module(module_name: str):
    """Drop module cache and import fresh."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_tracing_disabled_uses_noop_traceable(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    tracing = _reload_module("src.observability.tracing")
    assert tracing.langsmith_enabled is False

    @tracing.traceable(run_type="llm", name="dummy")
    def _fn(value):
        return value + 1

    assert _fn(1) == 2


def test_tracing_enabled_imports_react_loop_without_agent_run_type_warning(monkeypatch):
    pytest.importorskip("langsmith")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    # Load tracing first so react_loop consumes the wrapped traceable.
    _reload_module("src.observability.tracing")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        react_loop_module = _reload_module("src.llm.react_loop")
        assert callable(react_loop_module.react_loop)

    warning_texts = [str(w.message) for w in caught]
    assert not any("Unrecognized run_type: agent" in text for text in warning_texts)

