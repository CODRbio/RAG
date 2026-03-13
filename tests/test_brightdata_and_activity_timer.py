"""
Tests for:
  1. BrightData REST API integration in WebContentFetcher._fetch_brightdata()
  2. ActivityTimer — extend / expired / remaining semantics
  3. run_captcha_flow on_progress callback milestones
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


class _ListHandler(logging.Handler):
    """Capture log records from a non-propagating logger."""
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def capture_logger(name: str):
    """Context manager that attaches a _ListHandler to a named logger and yields it."""
    handler = _ListHandler()
    lg = logging.getLogger(name)
    lg.addHandler(handler)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)

from src.retrieval.downloader.captcha_page_runner import run_captcha_flow
from src.retrieval.downloader.captcha_solver import CaptchaSolveResult, CaptchaSolver, CaptchaType
from src.retrieval.web_content_fetcher import ActivityTimer, WebContentFetcher


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ActivityTimer
# ---------------------------------------------------------------------------

class TestActivityTimer:
    def test_initially_not_expired(self):
        timer = ActivityTimer(idle_timeout=10.0)
        assert not timer.expired
        assert timer.remaining > 0

    def test_expires_after_deadline(self):
        timer = ActivityTimer(idle_timeout=0.0)
        # Zero timeout → already expired
        time.sleep(0.01)
        assert timer.expired

    def test_extend_pushes_deadline_forward(self):
        timer = ActivityTimer(idle_timeout=1.0)
        before = timer.remaining
        timer.extend(60.0, reason="test")
        assert timer.remaining > before
        assert timer.remaining > 50.0

    def test_extend_does_not_shorten_deadline(self):
        timer = ActivityTimer(idle_timeout=60.0)
        before = timer.remaining
        timer.extend(1.0, reason="short")  # 1s < current remaining
        # deadline should NOT have moved backward
        assert timer.remaining >= before - 0.1  # allow tiny clock drift

    def test_remaining_decreases_over_time(self):
        timer = ActivityTimer(idle_timeout=5.0)
        r1 = timer.remaining
        time.sleep(0.05)
        r2 = timer.remaining
        assert r2 < r1


# ---------------------------------------------------------------------------
# WebContentFetcher — brightdata_timeout_seconds field
# ---------------------------------------------------------------------------

class TestBrightDataConfig:
    def test_default_brightdata_timeout(self):
        f = WebContentFetcher()
        assert f.brightdata_timeout_seconds == 120

    def test_custom_brightdata_timeout(self):
        f = WebContentFetcher(brightdata_timeout_seconds=300)
        assert f.brightdata_timeout_seconds == 300

    def test_no_api_key_returns_none_immediately(self):
        f = WebContentFetcher(brightdata_api_key="")
        result = _run(f._fetch_brightdata("https://example.com"))
        assert result is None


# ---------------------------------------------------------------------------
# _fetch_brightdata — REST API integration
# ---------------------------------------------------------------------------

class TestFetchBrightdataRestApi:
    """Verify _fetch_brightdata() uses POST api.brightdata.com/request (REST API)."""

    def _make_resp(self, status: int, body: bytes):
        resp = AsyncMock()
        resp.status = status
        resp.read = AsyncMock(return_value=body)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    def _make_session(self, resp):
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(return_value=resp)
        return session

    def test_uses_post_not_get(self):
        """Must call session.post (REST API), not session.get (old proxy approach)."""
        html_body = b"<html><body><article>" + b"Content word. " * 30 + b"</article></body></html>"
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(
            brightdata_api_key="test-token",
            brightdata_zone="mx_webunlocker",
            brightdata_timeout_seconds=30,
        )

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value="Extracted content " * 10):
            _run(f._fetch_brightdata("https://example.com/article"))

        session.post.assert_called_once()
        call_args = session.post.call_args
        assert "api.brightdata.com/request" in call_args[0][0]
        # Must NOT use proxy parameter
        assert "proxy" not in (call_args.kwargs or {})

    def test_bearer_token_in_header(self):
        """Authorization: Bearer {api_key} must be sent."""
        html_body = b"<html><body><p>" + b"word " * 50 + b"</p></body></html>"
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(
            brightdata_api_key="my-secret-token",
            brightdata_timeout_seconds=30,
        )

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value="content " * 20):
            _run(f._fetch_brightdata("https://example.com"))

        call_kwargs = session.post.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-token"

    def test_payload_contains_zone_url_format(self):
        """Payload must include zone, url, format='raw'."""
        html_body = b"<html><body><p>" + b"text " * 50 + b"</p></body></html>"
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(
            brightdata_api_key="tok",
            brightdata_zone="my_zone",
            brightdata_timeout_seconds=30,
        )

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value="content " * 20):
            _run(f._fetch_brightdata("https://target.example.com/page"))

        call_kwargs = session.post.call_args.kwargs
        payload = call_kwargs.get("json", {})
        assert payload.get("zone") == "my_zone"
        assert payload.get("url") == "https://target.example.com/page"
        assert payload.get("format") == "raw"

    def test_non_200_returns_none(self):
        resp = self._make_resp(403, b"forbidden")
        session = self._make_session(resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=10)

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is None

    def test_non_200_logs_warning_not_debug(self):
        """HTTP error must log at WARNING level (not debug) so it's visible in default logging."""
        resp = self._make_resp(429, b"rate limited")
        session = self._make_session(resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=10)

        with capture_logger("src.retrieval.web_content_fetcher") as h, \
             patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"):
            _run(f._fetch_brightdata("https://example.com"))

        warn_records = [r for r in h.records if r.levelno >= logging.WARNING]
        assert any("429" in r.getMessage() or "返回" in r.getMessage() for r in warn_records)

    def test_empty_body_retries_up_to_3_times(self):
        """200 with empty body should retry; after 3 empty responses return None."""
        empty_resp = self._make_resp(200, b"")
        session = self._make_session(empty_resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=10)

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is None
        assert session.post.call_count == 3

    def test_empty_body_then_success_returns_text(self):
        """Retry should succeed if a later attempt returns content."""
        empty_resp = self._make_resp(200, b"")
        html_body = b"<html><body><article>" + b"Content. " * 30 + b"</article></body></html>"
        good_resp = self._make_resp(200, html_body)

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return empty_resp if call_count == 1 else good_resp

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(side_effect=_side_effect)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=30)

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("trafilatura.extract", return_value="Good content " * 15):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is not None
        assert "Good content" in result

    def test_success_logs_info_not_debug(self):
        """Successful fetch must log at INFO level."""
        html_body = b"<html><body><p>" + b"word " * 50 + b"</p></body></html>"
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=30)

        with capture_logger("src.retrieval.web_content_fetcher") as h, \
             patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value="Good article content " * 10):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is not None
        info_msgs = [r.getMessage() for r in h.records if r.levelno == logging.INFO]
        assert any("BrightData 成功" in m for m in info_msgs)

    def test_timeout_logs_warning(self):
        """asyncio.TimeoutError must log at WARNING level."""
        resp = AsyncMock()
        resp.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post = MagicMock(return_value=resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=10)

        with capture_logger("src.retrieval.web_content_fetcher") as h, \
             patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is None
        warn_msgs = [r.getMessage() for r in h.records if r.levelno >= logging.WARNING]
        assert any("超时" in m or "timeout" in m.lower() for m in warn_msgs)

    def test_trafilatura_extraction_failure_logs_warning(self):
        """Content received but no text extracted → WARNING, not silent."""
        html_body = b"<html><body><p>" + b"x " * 100 + b"</p></body></html>"  # >200 bytes to pass early check
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_timeout_seconds=30)

        with capture_logger("src.retrieval.web_content_fetcher") as h, \
             patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value=None):
            result = _run(f._fetch_brightdata("https://example.com"))

        assert result is None
        warn_msgs = [r.getMessage() for r in h.records if r.levelno >= logging.WARNING]
        assert any("提取失败" in m or "正文" in m for m in warn_msgs)

    def test_fallback_zone_default(self):
        """No zone configured → defaults to 'web_unlocker1'."""
        html_body = b"<html><body><p>" + b"word " * 50 + b"</p></body></html>"
        resp = self._make_resp(200, html_body)
        session = self._make_session(resp)

        f = WebContentFetcher(brightdata_api_key="tok", brightdata_zone="", brightdata_timeout_seconds=30)

        with patch("aiohttp.ClientSession", return_value=session), \
             patch("src.utils.aiohttp_tls_patch.apply_aiohttp_tls_in_tls_patch"), \
             patch("trafilatura.extract", return_value="content " * 20):
            _run(f._fetch_brightdata("https://example.com"))

        payload = session.post.call_args.kwargs.get("json", {})
        assert payload.get("zone") == "web_unlocker1"


# ---------------------------------------------------------------------------
# run_captcha_flow — on_progress callback
# ---------------------------------------------------------------------------

class TestRunCaptchaFlowOnProgress:
    def test_unknown_captcha_does_not_fire_progress(self):
        page = AsyncMock()
        progress_calls = []

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.UNKNOWN,
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            _run(run_captcha_flow(
                page, solver, "tc",
                on_progress=lambda m: progress_calls.append(m),
                max_retries=1,
            ))

        assert progress_calls == []

    def test_captcha_detected_milestone_fires(self):
        page = AsyncMock()
        progress_calls = []

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.RECAPTCHA_V2,
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.extract_captcha_params",
            new_callable=AsyncMock,
            return_value=None,  # params fail → loop ends
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            _run(run_captcha_flow(
                page, solver, "tc",
                on_progress=lambda m: progress_calls.append(m),
                max_retries=1,
            ))

        assert "captcha_detected" in progress_calls

    def test_solving_started_fires_before_solve(self):
        page = AsyncMock()
        progress_calls = []
        solve_called_at = {}

        async def _fake_solve(captcha_type, params):
            solve_called_at["idx"] = len(progress_calls)
            return CaptchaSolveResult(success=False, error="no token", captcha_type=captcha_type)

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.RECAPTCHA_V2,
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.extract_captcha_params",
            new_callable=AsyncMock,
            return_value={"sitekey": "k", "pageurl": "https://x.com", "action": "submit"},
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            solver.solve = _fake_solve
            _run(run_captcha_flow(
                page, solver, "tc",
                on_progress=lambda m: progress_calls.append(m),
                max_retries=1,
            ))

        assert "solving_started" in progress_calls
        # solving_started must appear before solve() is called
        assert progress_calls.index("solving_started") < solve_called_at["idx"]

    def test_token_received_and_captcha_solved_fire_on_success(self):
        page = AsyncMock()
        progress_calls = []

        async def _fake_solve(captcha_type, params):
            return CaptchaSolveResult(
                success=True, token="tok123", captcha_type=captcha_type
            )

        detect_returns = [CaptchaType.RECAPTCHA_V2, CaptchaType.UNKNOWN]
        detect_iter = iter(detect_returns)

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            side_effect=lambda _: next(detect_iter),
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.extract_captcha_params",
            new_callable=AsyncMock,
            return_value={"sitekey": "k", "pageurl": "https://x.com", "action": "submit"},
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.apply_captcha_token",
            new_callable=AsyncMock,
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            solver.solve = _fake_solve
            result = _run(run_captcha_flow(
                page, solver, "tc",
                on_progress=lambda m: progress_calls.append(m),
                max_retries=1,
            ))

        assert result is True
        assert "token_received" in progress_calls
        assert "captcha_solved" in progress_calls
        # order: detected → solving_started → token_received → captcha_solved
        assert progress_calls.index("token_received") < progress_calls.index("captcha_solved")

    def test_turnstile_fires_detected_and_solved(self):
        page = AsyncMock()
        progress_calls = []

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.TURNSTILE,
        ), patch(
            "src.retrieval.downloader.captcha_page_runner.solve_turnstile_via_2captcha",
            new_callable=AsyncMock,
            return_value=True,
        ):
            solver = CaptchaSolver(capsolver_api_key="c", twocaptcha_api_key="t")
            result = _run(run_captcha_flow(
                page, solver, "tc",
                on_progress=lambda m: progress_calls.append(m),
                max_retries=1,
            ))

        assert result is True
        assert "captcha_detected" in progress_calls
        assert "captcha_solved" in progress_calls

    def test_on_progress_none_does_not_raise(self):
        """on_progress=None (default) must not cause AttributeError."""
        page = AsyncMock()

        with patch(
            "src.retrieval.downloader.captcha_page_runner.detect_captcha_type",
            new_callable=AsyncMock,
            return_value=CaptchaType.UNKNOWN,
        ):
            solver = CaptchaSolver()
            # Should not raise
            result = _run(run_captcha_flow(page, solver, "", max_retries=1))
        assert result is True
