from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPerSlotHeadedStartup:
    def test_start_headed_starts_two_slots_with_distinct_ports_and_profiles(self):
        from src.retrieval.browser_service import SharedBrowserService

        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_context = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        launch_persistent = AsyncMock(return_value=mock_context)
        pw_instance = MagicMock()
        pw_instance.chromium.launch_persistent_context = launch_persistent

        mock_settings = SimpleNamespace(
            capsolver_extension_path="",
            shared_browser=SimpleNamespace(
                headed_context_pool_size=2,
                headed_window_width=1280,
                headed_window_height=900,
                headed_window_visible_strip_px=100,
                headed_window_y=80,
            ),
        )

        import sys
        settings_mod = sys.modules["config.settings"]
        orig_settings = settings_mod.settings
        settings_mod.settings = mock_settings
        orig_slots = SharedBrowserService._headed_slots
        orig_pw = SharedBrowserService._playwright_hd
        try:
            SharedBrowserService._headed_slots = {}
            SharedBrowserService._playwright_hd = None
            with patch("src.retrieval.browser_service.async_playwright") as p_pw:
                p_pw.return_value.start = AsyncMock(return_value=pw_instance)
                with patch.object(
                    SharedBrowserService,
                    "_ensure_headed_profile_dir",
                    side_effect=lambda sid=None: f"/tmp/{sid}",
                ):
                    _run(SharedBrowserService.start_headed(port=19000))
            assert launch_persistent.await_count == 2
            args0 = launch_persistent.await_args_list[0].kwargs.get("args", [])
            args1 = launch_persistent.await_args_list[1].kwargs.get("args", [])
            assert any("--remote-debugging-port=19000" in a for a in args0)
            assert any("--remote-debugging-port=19001" in a for a in args1)
            assert "headed-0" in SharedBrowserService._headed_slots
            assert "headed-1" in SharedBrowserService._headed_slots
            assert SharedBrowserService.get_cdp_url_headed("headed-0") == "http://127.0.0.1:19000"
            assert SharedBrowserService.get_cdp_url_headed("headed-1") == "http://127.0.0.1:19001"
        finally:
            settings_mod.settings = orig_settings
            SharedBrowserService._headed_slots = orig_slots
            SharedBrowserService._playwright_hd = orig_pw


class TestContextPoolHeadedSlotBinding:
    def test_create_headed_slot_uses_slot_specific_cdp_url(self):
        from src.retrieval.context_pool import SharedContextPool

        pool = SharedContextPool()
        pool._playwright = MagicMock()
        browser = MagicMock()
        browser.new_context = AsyncMock(return_value=MagicMock())
        pool._playwright.chromium.connect_over_cdp = AsyncMock(return_value=browser)
        pool._ensure_playwright = AsyncMock()

        with patch(
            "src.retrieval.context_pool.SharedBrowserService.get_cdp_url_headed",
            return_value="http://127.0.0.1:19901",
        ) as get_cdp:
            slot = _run(pool._create_headed_slot("headed-1", downloads_dir=None))
        assert slot is not None
        get_cdp.assert_called_with(slot_id="headed-1")
        pool._playwright.chromium.connect_over_cdp.assert_awaited_once_with("http://127.0.0.1:19901")
