"""
Tests for unified CDP context options (locale, timezone, viewport, etc.).

Ensures context_pool, web_content_fetcher, and browser_manager all use the same
canonical defaults from browser_service.get_cdp_context_options().
"""

import pytest

from src.retrieval.browser_service import (
    CDP_DEFAULT_LOCALE,
    CDP_DEFAULT_TIMEZONE_ID,
    get_cdp_context_options,
)


class TestCdpContextOptions:
    """Canonical CDP context options match paper_downloader style (Asia/zh-CN)."""

    def test_get_cdp_context_options_returns_canonical_locale_and_timezone(self):
        opts = get_cdp_context_options()
        assert opts["locale"] == CDP_DEFAULT_LOCALE
        assert opts["timezone_id"] == CDP_DEFAULT_TIMEZONE_ID
        assert opts["locale"] == "zh-CN"
        assert opts["timezone_id"] == "Asia/Shanghai"

    def test_get_cdp_context_options_includes_full_set(self):
        opts = get_cdp_context_options()
        assert "accept_downloads" in opts
        assert "user_agent" in opts
        assert "viewport" in opts
        assert "locale" in opts
        assert "timezone_id" in opts
        assert "geolocation" in opts
        assert opts.get("color_scheme") == "light"
        assert opts.get("reduced_motion") == "no-preference"
        assert opts.get("has_touch") is False

    def test_get_cdp_context_options_viewport_is_1280x720(self):
        opts = get_cdp_context_options()
        assert opts["viewport"] == {"width": 1280, "height": 720}

    def test_get_cdp_context_options_allows_user_agent_and_viewport_override(self):
        opts = get_cdp_context_options(user_agent="Custom/1.0", viewport={"width": 800, "height": 600})
        assert opts["user_agent"] == "Custom/1.0"
        assert opts["viewport"] == {"width": 800, "height": 600}
        assert opts["locale"] == "zh-CN"
        assert opts["timezone_id"] == "Asia/Shanghai"

    def test_browser_manager_cdp_path_uses_canonical_source(self):
        from src.retrieval.downloader.browser_manager import get_cdp_context_options as bm_get_opts
        from src.retrieval.browser_service import get_cdp_context_options
        assert bm_get_opts is get_cdp_context_options
