"""Tests for scholar downloader adapter: get_adapter(show_browser) returns distinct instances per mode."""

from src.retrieval.downloader.adapter import get_adapter, shutdown_adapter


def test_get_adapter_by_show_browser_returns_different_instances():
    """get_adapter(show_browser=True) and get_adapter(show_browser=False) must return different adapter instances."""
    try:
        headless = get_adapter(show_browser=False)
        headed = get_adapter(show_browser=True)
        assert headless is not headed
        assert headless._headed is False
        assert headed._headed is True
    finally:
        shutdown_adapter()


def test_get_adapter_none_and_false_same_instance():
    """get_adapter(None) and get_adapter(False) should return the same (headless) instance."""
    try:
        a = get_adapter(show_browser=None)
        b = get_adapter(show_browser=False)
        assert a is b
        assert a._headed is False
    finally:
        shutdown_adapter()
