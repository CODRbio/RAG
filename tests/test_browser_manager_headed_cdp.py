"""
Focused tests for BrowserManager headed CDP reuse vs local fallback.

- Headed CDP branch: when shared headed CDP is available, connect_over_cdp is used
  and extension_path is not used at attach time.
- Local fallback: when no shared headed CDP, local launch_persistent_context is used
  and --load-extension / --disable-extensions-except are passed when extension_path is valid.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.downloader.browser_manager import BrowserManager


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHeadedCdpReuse:
    """When get_cdp_url_headed() returns a URL, BrowserManager uses connect_over_cdp."""

    def test_headed_cdp_branch_uses_connect_over_cdp_and_ignores_extension_path(self):
        cdp_url = "http://127.0.0.1:9223"
        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=MagicMock())
        connect_mock = AsyncMock(return_value=mock_browser)
        launch_persistent_mock = AsyncMock()

        with patch(
            "src.retrieval.downloader.browser_manager.SharedBrowserService.get_cdp_url_headed",
            return_value=cdp_url,
        ):
            with patch(
                "src.retrieval.downloader.browser_manager.async_playwright"
            ) as p_playwright:
                mock_pw = MagicMock()
                mock_pw.chromium.connect_over_cdp = connect_mock
                mock_pw.chromium.launch_persistent_context = launch_persistent_mock
                p_playwright.return_value.start = AsyncMock(return_value=mock_pw)

                manager = BrowserManager()
                ctx = _run(
                    manager.launch_persistent_browser(
                        user_data_dir=tempfile.mkdtemp(),
                        browser_type="chrome",
                        headless=False,
                        reuse_shared_cdp=True,
                        extension_path="/some/capsolver/path",
                    )
                )
                assert ctx is not None
                connect_mock.assert_awaited_once()
                connect_mock.assert_awaited_with(cdp_url)
                launch_persistent_mock.assert_not_awaited()


class TestHeadedLocalFallbackExtensionArgs:
    """When no shared headed CDP, local launch receives extension args."""

    def test_local_fallback_injects_extension_args_when_extension_path_valid(self):
        with tempfile.TemporaryDirectory() as ext_dir:
            with open(os.path.join(ext_dir, "manifest.json"), "w") as f:
                f.write('{"manifest_version": 3, "name": "Cap", "version": "1.0"}')
            ext_path = os.path.abspath(ext_dir)

            launch_persistent = AsyncMock(return_value=MagicMock())
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch_persistent_context = launch_persistent

            with patch(
                "src.retrieval.downloader.browser_manager.SharedBrowserService.get_cdp_url_headed",
                return_value=None,
            ):
                with patch(
                    "src.retrieval.downloader.browser_manager.async_playwright"
                ) as p_playwright:
                    p_playwright.return_value.start = AsyncMock(
                        return_value=mock_pw_instance
                    )

                    manager = BrowserManager()
                    ctx = _run(
                        manager.launch_persistent_browser(
                            user_data_dir=tempfile.mkdtemp(),
                            browser_type="chromium",
                            headless=False,
                            reuse_shared_cdp=True,
                            extension_path=ext_path,
                        )
                    )
                    assert ctx is not None
                    launch_persistent.assert_awaited_once()
                    call_kw = launch_persistent.call_args[1]
                    args = call_kw.get("args") or []
                    assert any(
                        f"--load-extension={ext_path}" in (a or "") for a in args
                    ), f"Expected --load-extension in args: {args}"
                    assert any(
                        "--disable-extensions-except=" in (a or "") for a in args
                    ), f"Expected --disable-extensions-except in args: {args}"
