"""
Unit tests for the captcha capability upgrade.

Coverage:
  1. CaptchaSolver routing – correct primary provider order per captcha type.
  2. CaptchaSolver fallback – primary failure triggers secondary provider.
  3. CaptchaSolver – no providers configured returns a clean failure.
  4. CaptchaSolver – UNKNOWN type returns failure without calling any provider.
  5. SharedBrowserService._resolve_extension_args – absent/invalid/valid directory paths.
  6. BlockerType.CAPTCHA in _handle_blocker routes through solve_captcha_if_needed,
     NOT solve_cloudflare_if_needed.
  7. detect_captcha_type returns TURNSTILE and solve_captcha_if_needed delegates
     back to solve_cloudflare_if_needed.
  8. Regression: existing Cloudflare blocker still calls solve_cloudflare_if_needed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.downloader.captcha_solver import (
    CaptchaSolver,
    CaptchaSolveResult,
    CaptchaType,
    _CapSolverProvider,
    _TwoCaptchaProvider,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _ok(provider: str, ctype: CaptchaType) -> CaptchaSolveResult:
    return CaptchaSolveResult(success=True, token="tok123", provider=provider, captcha_type=ctype)


def _fail(ctype: CaptchaType) -> CaptchaSolveResult:
    return CaptchaSolveResult(success=False, error="boom", captcha_type=ctype)


# ---------------------------------------------------------------------------
# 1. Routing – correct primary provider per type
# ---------------------------------------------------------------------------

class TestCaptchaSolverRouting:
    def test_turnstile_primary_is_twocaptcha(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.TURNSTILE)
        assert isinstance(providers[0], _TwoCaptchaProvider)

    def test_recaptcha_v2_primary_is_capsolver(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.RECAPTCHA_V2)
        assert isinstance(providers[0], _CapSolverProvider)

    def test_recaptcha_v3_primary_is_capsolver(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.RECAPTCHA_V3)
        assert isinstance(providers[0], _CapSolverProvider)

    def test_hcaptcha_primary_is_capsolver(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.HCAPTCHA)
        assert isinstance(providers[0], _CapSolverProvider)

    def test_image_text_primary_is_twocaptcha(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.IMAGE_TEXT)
        assert isinstance(providers[0], _TwoCaptchaProvider)

    def test_funcaptcha_primary_is_capsolver(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.FUNCAPTCHA)
        assert isinstance(providers[0], _CapSolverProvider)

    def test_both_providers_present_in_chain(self):
        solver = CaptchaSolver(capsolver_api_key="cap-key", twocaptcha_api_key="tc-key")
        providers = solver._providers_for(CaptchaType.HCAPTCHA)
        types = [type(p) for p in providers]
        assert _CapSolverProvider in types
        assert _TwoCaptchaProvider in types


# ---------------------------------------------------------------------------
# 2. Fallback – primary failure triggers secondary
# ---------------------------------------------------------------------------

class TestCaptchaSolverFallback:
    def test_capsolver_failure_falls_back_to_twocaptcha(self):
        solver = CaptchaSolver(capsolver_api_key="cap", twocaptcha_api_key="tc")
        solver._cap.solve = AsyncMock(return_value=_fail(CaptchaType.RECAPTCHA_V2))
        solver._tc.solve = AsyncMock(return_value=_ok("twocaptcha", CaptchaType.RECAPTCHA_V2))

        result = _run(solver.solve(
            CaptchaType.RECAPTCHA_V2, {"sitekey": "sk", "pageurl": "https://x.com"}
        ))
        assert result.success is True
        assert result.provider == "twocaptcha"

    def test_twocaptcha_failure_falls_back_to_capsolver(self):
        solver = CaptchaSolver(capsolver_api_key="cap", twocaptcha_api_key="tc")
        solver._tc.solve = AsyncMock(return_value=_fail(CaptchaType.TURNSTILE))
        solver._cap.solve = AsyncMock(return_value=_ok("capsolver", CaptchaType.TURNSTILE))

        result = _run(solver.solve(
            CaptchaType.TURNSTILE, {"sitekey": "sk", "pageurl": "https://x.com"}
        ))
        assert result.success is True
        assert result.provider == "capsolver"

    def test_all_providers_fail_returns_failure(self):
        solver = CaptchaSolver(capsolver_api_key="cap", twocaptcha_api_key="tc")
        solver._cap.solve = AsyncMock(return_value=_fail(CaptchaType.HCAPTCHA))
        solver._tc.solve = AsyncMock(return_value=_fail(CaptchaType.HCAPTCHA))

        result = _run(solver.solve(
            CaptchaType.HCAPTCHA, {"sitekey": "sk", "pageurl": "https://x.com"}
        ))
        assert result.success is False


# ---------------------------------------------------------------------------
# 3. No providers configured
# ---------------------------------------------------------------------------

class TestCaptchaSolverNoProviders:
    def test_no_keys_returns_failure(self):
        solver = CaptchaSolver()
        assert solver.has_any_provider is False
        result = _run(solver.solve(CaptchaType.HCAPTCHA, {"sitekey": "sk", "pageurl": "u"}))
        assert result.success is False
        assert "no captcha API" in (result.error or "")

    def test_only_capsolver_key(self):
        solver = CaptchaSolver(capsolver_api_key="cap")
        assert solver._tc is None
        providers = solver._providers_for(CaptchaType.RECAPTCHA_V2)
        assert len(providers) == 1
        assert isinstance(providers[0], _CapSolverProvider)

    def test_only_twocaptcha_key(self):
        solver = CaptchaSolver(twocaptcha_api_key="tc")
        assert solver._cap is None
        providers = solver._providers_for(CaptchaType.RECAPTCHA_V2)
        assert len(providers) == 1
        assert isinstance(providers[0], _TwoCaptchaProvider)


# ---------------------------------------------------------------------------
# 4. UNKNOWN type short-circuits
# ---------------------------------------------------------------------------

class TestCaptchaSolverUnknown:
    def test_unknown_returns_failure_without_api_call(self):
        solver = CaptchaSolver(capsolver_api_key="cap", twocaptcha_api_key="tc")
        cap_mock = AsyncMock()
        solver._cap.solve = cap_mock

        result = _run(solver.solve(CaptchaType.UNKNOWN, {}))
        assert result.success is False
        cap_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 5. SharedBrowserService._resolve_extension_args
# ---------------------------------------------------------------------------

class TestResolveExtensionArgs:
    def test_none_returns_empty(self):
        from src.retrieval.browser_service import SharedBrowserService
        args, ignore = SharedBrowserService._resolve_extension_args(None)
        assert args == [] and ignore == []

    def test_missing_dir_returns_empty(self):
        from src.retrieval.browser_service import SharedBrowserService
        args, ignore = SharedBrowserService._resolve_extension_args("/nonexistent/path/xyz")
        assert args == [] and ignore == []

    def test_dir_without_manifest_returns_empty(self):
        from src.retrieval.browser_service import SharedBrowserService
        with tempfile.TemporaryDirectory() as d:
            args, ignore = SharedBrowserService._resolve_extension_args(d)
            assert args == [] and ignore == []

    def test_valid_extension_dir_returns_args(self):
        from src.retrieval.browser_service import SharedBrowserService
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "manifest.json"), "w") as f:
                f.write('{"manifest_version": 3, "name": "Test", "version": "1.0"}')
            args, ignore = SharedBrowserService._resolve_extension_args(d)
            resolved = os.path.abspath(d)
            assert f"--load-extension={resolved}" in args
            assert f"--disable-extensions-except={resolved}" in args
            assert "--disable-extensions" in ignore


# ---------------------------------------------------------------------------
# 6. _handle_blockers_smart CAPTCHA routes through solve_captcha_if_needed
# ---------------------------------------------------------------------------

class TestHandleBlockerCaptchaRouting:
    def test_captcha_blocker_calls_generic_solver_not_cloudflare(self):
        from src.retrieval.downloader.paper_downloader_refactored import (
            PaperDownloader,
            BlockerType,
            Blocker,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = PaperDownloader(download_dir=tmpdir)

            generic_mock = AsyncMock(return_value=(True, None))
            cf_mock = AsyncMock(return_value=(True, None))
            dl.solve_captcha_if_needed = generic_mock
            dl.solve_cloudflare_if_needed = cf_mock

            page_mock = AsyncMock()
            page_mock.title = AsyncMock(return_value="OK")
            blockers = [Blocker(BlockerType.CAPTCHA)]

            _run(dl._handle_blockers_smart(
                page=page_mock,
                blockers=blockers,
                task_download_dir=None,
            ))

            generic_mock.assert_awaited_once()
            cf_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Cloudflare blocker still uses solve_cloudflare_if_needed (regression)
# ---------------------------------------------------------------------------

class TestHandleBlockerCloudflareRegression:
    def test_cloudflare_blocker_still_calls_cf_solver(self):
        from src.retrieval.downloader.paper_downloader_refactored import (
            PaperDownloader,
            BlockerType,
            Blocker,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = PaperDownloader(download_dir=tmpdir)

            generic_mock = AsyncMock(return_value=(True, None))
            cf_mock = AsyncMock(return_value=(True, None))
            dl.solve_captcha_if_needed = generic_mock
            dl.solve_cloudflare_if_needed = cf_mock

            page_mock = AsyncMock()
            page_mock.title = AsyncMock(return_value="OK")
            blockers = [Blocker(BlockerType.CLOUDFLARE)]

            _run(dl._handle_blockers_smart(
                page=page_mock,
                blockers=blockers,
                task_download_dir=None,
            ))

            cf_mock.assert_awaited_once()
            generic_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. solve_captcha_if_needed delegates TURNSTILE to CF solver
# ---------------------------------------------------------------------------

class TestSolveCaptchaIfNeededTurnstileDelegation:
    def test_turnstile_delegates_to_cloudflare_solver(self):
        from src.retrieval.downloader.paper_downloader_refactored import PaperDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = PaperDownloader(download_dir=tmpdir)

            dl.detect_captcha_type = AsyncMock(return_value=CaptchaType.TURNSTILE)
            cf_mock = AsyncMock(return_value=(True, None))
            dl.solve_cloudflare_if_needed = cf_mock

            result = _run(dl.solve_captcha_if_needed(page=AsyncMock()))
            assert result == (True, None)
            cf_mock.assert_awaited_once()

    def test_unknown_captcha_returns_ok_without_api_call(self):
        from src.retrieval.downloader.paper_downloader_refactored import PaperDownloader

        with tempfile.TemporaryDirectory() as tmpdir:
            dl = PaperDownloader(download_dir=tmpdir)

            dl.detect_captcha_type = AsyncMock(return_value=CaptchaType.UNKNOWN)
            solver_mock = AsyncMock()
            dl._captcha_solver.solve = solver_mock

            result = _run(dl.solve_captcha_if_needed(page=AsyncMock()))
            assert result == (True, None)
            solver_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9. SharedBrowserService.start_headed extension path
# ---------------------------------------------------------------------------

class TestStartHeadedExtensionPath:
    """Headed CDP startup uses capsolver_extension_path; explicit arg overrides settings."""

    def test_start_headed_uses_settings_extension_path_when_no_explicit_arg(self):
        from src.retrieval.browser_service import SharedBrowserService

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "manifest.json"), "w") as f:
                f.write('{"manifest_version": 3, "name": "Cap", "version": "1.0"}')
            ext_path = os.path.abspath(d)

            _page = MagicMock()
            _page.goto = AsyncMock()
            _context = MagicMock(new_page=AsyncMock(return_value=_page))
            mock_launch_persistent = AsyncMock(return_value=_context)
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch_persistent_context = mock_launch_persistent

            mock_settings = MagicMock()
            mock_settings.capsolver_extension_path = ext_path
            mock_settings.shared_browser = None

            import sys
            _settings_mod = sys.modules["config.settings"]
            _orig_settings = _settings_mod.settings
            _settings_mod.settings = mock_settings
            try:
                with patch("src.retrieval.browser_service.async_playwright") as p_playwright:
                    p_playwright.return_value.start = AsyncMock(return_value=mock_pw_instance)
                    orig_ctx = SharedBrowserService._context_hd
                    orig_pw = SharedBrowserService._playwright_hd
                    orig_slots = SharedBrowserService._headed_slots
                    try:
                        SharedBrowserService._context_hd = None
                        SharedBrowserService._playwright_hd = None
                        SharedBrowserService._headed_slots = {}
                        _run(SharedBrowserService.start_headed(port=19923))
                        mock_launch_persistent.assert_awaited_once()
                        call_kw = mock_launch_persistent.call_args[1]
                        args = call_kw.get("args") or []
                        assert any(f"--load-extension={ext_path}" in (a or "") for a in args)
                        assert any("--disable-extensions-except=" in (a or "") for a in args)
                    finally:
                        SharedBrowserService._context_hd = orig_ctx
                        SharedBrowserService._playwright_hd = orig_pw
                        SharedBrowserService._headed_slots = orig_slots
            finally:
                _settings_mod.settings = _orig_settings

    def test_start_headed_explicit_extension_path_overrides_settings(self):
        from src.retrieval.browser_service import SharedBrowserService

        with tempfile.TemporaryDirectory() as d_settings:
            with tempfile.TemporaryDirectory() as d_explicit:
                with open(os.path.join(d_explicit, "manifest.json"), "w") as f:
                    f.write('{"manifest_version": 3, "name": "Cap", "version": "1.0"}')
                explicit_path = os.path.abspath(d_explicit)

                _page = MagicMock()
                _page.goto = AsyncMock()
                _context = MagicMock(new_page=AsyncMock(return_value=_page))
                mock_launch_persistent = AsyncMock(return_value=_context)
                mock_pw_instance = MagicMock()
                mock_pw_instance.chromium.launch_persistent_context = mock_launch_persistent

                mock_settings = MagicMock()
                mock_settings.capsolver_extension_path = os.path.abspath(d_settings)
                mock_settings.shared_browser = None

                import sys
                _settings_mod = sys.modules["config.settings"]
                _orig_settings = _settings_mod.settings
                _settings_mod.settings = mock_settings
                try:
                    with patch("src.retrieval.browser_service.async_playwright") as p_playwright:
                        p_playwright.return_value.start = AsyncMock(return_value=mock_pw_instance)
                        orig_ctx = SharedBrowserService._context_hd
                        orig_pw = SharedBrowserService._playwright_hd
                        orig_slots = SharedBrowserService._headed_slots
                        try:
                            SharedBrowserService._context_hd = None
                            SharedBrowserService._playwright_hd = None
                            SharedBrowserService._headed_slots = {}
                            _run(
                                SharedBrowserService.start_headed(
                                    port=19924,
                                    extension_path=explicit_path,
                                )
                            )
                            mock_launch_persistent.assert_awaited_once()
                            call_kw = mock_launch_persistent.call_args[1]
                            args = call_kw.get("args") or []
                            assert any(
                                f"--load-extension={explicit_path}" in (a or "") for a in args
                            )
                        finally:
                            SharedBrowserService._context_hd = orig_ctx
                            SharedBrowserService._playwright_hd = orig_pw
                            SharedBrowserService._headed_slots = orig_slots
                finally:
                    _settings_mod.settings = _orig_settings


# ---------------------------------------------------------------------------
# 10. Headed profile developer mode preference
# ---------------------------------------------------------------------------

class TestHeadedProfileDeveloperMode:
    """Shared headed browser uses one persistent profile with extensions.ui.developer_mode=true."""

    def test_ensure_headed_profile_dir_writes_developer_mode_preference(self):
        import json
        from pathlib import Path
        from src.utils.path_manager import PathManager
        from src.retrieval.browser_service import SharedBrowserService

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(PathManager, "get_shared_dir", return_value=root):
                path = SharedBrowserService._ensure_headed_profile_dir("headed-0")
            assert path
            assert (Path(path) / "Default" / "Preferences").exists()
            with open(Path(path) / "Default" / "Preferences", "r", encoding="utf-8") as f:
                prefs = json.load(f)
            assert prefs.get("extensions", {}).get("ui", {}).get("developer_mode") is True

    def test_ensure_headed_profile_dir_merges_with_existing_prefs(self):
        import json
        from pathlib import Path
        from src.utils.path_manager import PathManager
        from src.retrieval.browser_service import SharedBrowserService

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            profile_dir = root / "headed_browser_profile" / "headed-0"
            profile_dir.mkdir(parents=True)
            default_dir = profile_dir / "Default"
            default_dir.mkdir()
            existing = {"some_other_key": "value", "extensions": {"ui": {"other": 1}}}
            with open(default_dir / "Preferences", "w", encoding="utf-8") as f:
                json.dump(existing, f)
            with patch.object(PathManager, "get_shared_dir", return_value=root):
                SharedBrowserService._ensure_headed_profile_dir("headed-0")
            with open(default_dir / "Preferences", "r", encoding="utf-8") as f:
                prefs = json.load(f)
            assert prefs.get("some_other_key") == "value"
            assert prefs.get("extensions", {}).get("ui", {}).get("developer_mode") is True
            assert prefs.get("extensions", {}).get("ui", {}).get("other") == 1
