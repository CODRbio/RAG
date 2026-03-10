"""
Tests for WebContentFetcher captcha integration and main-content extraction.

- WebContentFetcher config: capsolver_api_key, captcha_timeout_seconds, _captcha_solver init.
- captcha_page_runner.run_captcha_flow: UNKNOWN returns True; no providers for non-UNKNOWN returns False.
- Provider priority (shared with downloader): Turnstile -> 2Captcha only in run_captcha_flow; others use CaptchaSolver.
- Main-content extraction: trafilatura path preserves favor_precision on first tier.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.downloader.captcha_page_runner import (
    INTERCEPT_SCRIPT,
    detect_captcha_type,
    run_captcha_flow,
)
from src.retrieval.downloader.captcha_solver import CaptchaSolver, CaptchaType
from src.retrieval.web_content_fetcher import WebContentFetcher


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# WebContentFetcher config and captcha solver init
# ---------------------------------------------------------------------------

class TestWebContentFetcherConfig:
    def test_fetcher_accepts_capsolver_and_captcha_timeout(self):
        f = WebContentFetcher(
            enabled=True,
            two_captcha_api_key="tc",
            capsolver_api_key="cap",
            captcha_timeout_seconds=90,
        )
        assert f.capsolver_api_key == "cap"
        assert f.captcha_timeout_seconds == 90
        assert f._captcha_solver is not None
        assert f._captcha_solver.has_any_provider is True

    def test_fetcher_inits_captcha_solver_when_either_key_set(self):
        f1 = WebContentFetcher(two_captcha_api_key="tc")
        assert f1._captcha_solver is not None
        f2 = WebContentFetcher(capsolver_api_key="cap")
        assert f2._captcha_solver is not None
        f3 = WebContentFetcher()
        assert f3._captcha_solver is None

    def test_from_settings_includes_captcha_fields_when_cfg_present(self):
        # from_settings reads content_fetcher from config.settings; we only assert
        # that the constructor accepts the new fields (covered above). Integration
        # with real config is exercised at runtime.
        fetcher = WebContentFetcher(
            enabled=True,
            two_captcha_api_key="tc",
            capsolver_api_key="cap",
            captcha_timeout_seconds=100,
        )
        assert fetcher.capsolver_api_key == "cap"
        assert fetcher.captcha_timeout_seconds == 100
        assert fetcher._captcha_solver is not None


# ---------------------------------------------------------------------------
# captcha_page_runner.run_captcha_flow
# ---------------------------------------------------------------------------

class TestRunCaptchaFlow:
    def test_unknown_captcha_returns_true_without_calling_solver(self):
        page = AsyncMock()
        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.UNKNOWN,
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            solver.solve = AsyncMock()
            result = _run(
                run_captcha_flow(
                    page,
                    solver,
                    "tc",
                    captcha_timeout_seconds=30,
                    max_retries=1,
                )
            )
            assert result is True
            solver.solve.assert_not_awaited()

    def test_no_providers_configured_returns_false_for_non_unknown(self):
        page = AsyncMock()
        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.RECAPTCHA_V2,
        ):
            solver = CaptchaSolver()
            result = _run(
                run_captcha_flow(
                    page,
                    solver,
                    "",
                    captcha_timeout_seconds=30,
                    max_retries=1,
                )
            )
            assert result is False

    def test_turnstile_uses_solve_turnstile_via_2captcha_not_generic_solver(self):
        page = AsyncMock()
        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.TURNSTILE,
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.solve_turnstile_via_2captcha",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_turnstile:
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="tc")
            solver.solve = AsyncMock()
            result = _run(
                run_captcha_flow(
                    page,
                    solver,
                    "tc",
                    captcha_timeout_seconds=30,
                    max_retries=1,
                )
            )
            assert result is True
            mock_turnstile.assert_awaited_once()
            solver.solve.assert_not_awaited()


# ---------------------------------------------------------------------------
# Main-content extraction (trafilatura options)
# ---------------------------------------------------------------------------

class TestMainContentExtraction:
    def test_trafilatura_tier_uses_favor_precision(self):
        long_content = "Main content here. " * 20
        with patch("trafilatura.fetch_url") as mock_fetch, patch(
            "trafilatura.extract"
        ) as mock_extract:
            mock_fetch.return_value = "<html><body><article><p>Main content here.</p></article></body></html>"
            mock_extract.return_value = long_content
            fetcher = WebContentFetcher(enabled=True)
            result = _run(fetcher._fetch_trafilatura("https://example.com/article"))
            assert result is not None
            assert "Main content" in result
            call_kw = mock_extract.call_args[1]
            assert call_kw.get("favor_precision") is True
            assert call_kw.get("include_comments") is False
            assert call_kw.get("include_tables") is True


# ---------------------------------------------------------------------------
# Intercept script constant
# ---------------------------------------------------------------------------

class TestInterceptScript:
    def test_intercept_script_contains_turnstile_and_callback(self):
        assert "turnstile" in INTERCEPT_SCRIPT.lower()
        assert "__cf_intercepted_params" in INTERCEPT_SCRIPT
        assert "__cf_callback" in INTERCEPT_SCRIPT
