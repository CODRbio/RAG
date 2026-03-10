"""
Unified captcha solver abstraction.

Provides:
  - CaptchaType       – enum of recognised challenge types
  - CaptchaSolveResult – lightweight result DTO
  - CaptchaSolver      – async solver that routes by type:
      * reCAPTCHA v2/v3, hCaptcha, FunCaptcha  → CapSolver API first, 2Captcha fallback
      * Cloudflare Turnstile                   → 2Captcha first (existing proven path), CapSolver fallback
      * ImageToText (incl. Chinese)            → 2Captcha first, CapSolver fallback
      * Unknown                                → extension auto-pass only (no API call)

Both providers use the same async HTTP polling pattern so as not to block the event loop.
The TwoCaptcha SDK is used for the 2Captcha path when installed; otherwise a thin aiohttp
implementation is used so that the module degrades gracefully.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from src.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public enums / data classes
# ---------------------------------------------------------------------------

class CaptchaType(str, Enum):
    """Recognised captcha challenge types."""
    TURNSTILE = "turnstile"
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    IMAGE_TEXT = "image_text"
    FUNCAPTCHA = "funcaptcha"
    UNKNOWN = "unknown"


@dataclass
class CaptchaSolveResult:
    """Result of a single solve attempt."""
    success: bool
    token: Optional[str] = None          # token/text returned by the solver
    provider: Optional[str] = None       # "capsolver" | "twocaptcha"
    captcha_type: Optional[CaptchaType] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class _CapSolverProvider:
    """
    Thin async wrapper around the CapSolver REST API.
    https://docs.capsolver.com/
    """
    BASE = "https://api.capsolver.com"
    _TYPE_MAP = {
        CaptchaType.RECAPTCHA_V2:  "ReCaptchaV2TaskProxyless",
        CaptchaType.RECAPTCHA_V3:  "ReCaptchaV3TaskProxyless",
        CaptchaType.HCAPTCHA:      "HCaptchaTaskProxyless",
        CaptchaType.TURNSTILE:     "AntiTurnstileTaskProxyless",
        CaptchaType.FUNCAPTCHA:    "FunCaptchaTaskProxyless",
        CaptchaType.IMAGE_TEXT:    "ImageToTextTask",
    }

    def __init__(self, api_key: str, timeout: int = 120) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def solve(self, captcha_type: CaptchaType, params: Dict[str, Any]) -> CaptchaSolveResult:
        start = time.monotonic()
        task_type = self._TYPE_MAP.get(captcha_type)
        if not task_type:
            return CaptchaSolveResult(success=False, error=f"unsupported type: {captcha_type}", captcha_type=captcha_type)

        task: Dict[str, Any] = {"type": task_type}
        if captcha_type == CaptchaType.IMAGE_TEXT:
            img = params.get("image_base64") or params.get("body")
            if not img:
                return CaptchaSolveResult(success=False, error="missing image_base64", captcha_type=captcha_type)
            task["body"] = img
        elif captcha_type == CaptchaType.FUNCAPTCHA:
            task["websiteURL"] = params.get("pageurl", "")
            task["websitePublicKey"] = params.get("publickey", "")
        else:
            task["websiteURL"] = params.get("pageurl", "")
            task["websiteKey"] = params.get("sitekey", "")
            if captcha_type == CaptchaType.RECAPTCHA_V3:
                task["pageAction"] = params.get("action", "")
            if captcha_type == CaptchaType.TURNSTILE:
                if params.get("action"):
                    task["metadata"] = {"action": params["action"]}

        try:
            async with aiohttp.ClientSession() as session:
                # Create task
                create_resp = await session.post(
                    f"{self.BASE}/createTask",
                    json={"clientKey": self.api_key, "task": task},
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                create_data = await create_resp.json()
                if create_data.get("errorId", 1) != 0:
                    return CaptchaSolveResult(
                        success=False,
                        error=f"createTask error: {create_data.get('errorDescription')}",
                        captcha_type=captcha_type,
                    )
                task_id = create_data["taskId"]
                logger.debug("[capsolver] task created: %s type=%s", task_id, task_type)

                # Poll for result
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    await asyncio.sleep(3)
                    res_resp = await session.post(
                        f"{self.BASE}/getTaskResult",
                        json={"clientKey": self.api_key, "taskId": task_id},
                        timeout=aiohttp.ClientTimeout(total=30),
                    )
                    res_data = await res_resp.json()
                    if res_data.get("errorId", 1) != 0:
                        return CaptchaSolveResult(
                            success=False,
                            error=f"getTaskResult error: {res_data.get('errorDescription')}",
                            captcha_type=captcha_type,
                        )
                    if res_data.get("status") == "ready":
                        solution = res_data.get("solution") or {}
                        token = (
                            solution.get("gRecaptchaResponse")
                            or solution.get("token")
                            or solution.get("text")
                            or solution.get("userAgent")
                        )
                        latency = (time.monotonic() - start) * 1000
                        logger.info("[capsolver] solved type=%s in %.0f ms", captcha_type, latency)
                        return CaptchaSolveResult(
                            success=True,
                            token=token,
                            provider="capsolver",
                            captcha_type=captcha_type,
                            latency_ms=latency,
                            raw=res_data,
                        )
                return CaptchaSolveResult(success=False, error="timeout", captcha_type=captcha_type)
        except Exception as exc:
            return CaptchaSolveResult(success=False, error=str(exc), captcha_type=captcha_type)


class _TwoCaptchaProvider:
    """
    Async wrapper around the 2Captcha REST API.
    Falls back to direct HTTP calls if the 'twocaptcha' SDK is not installed.
    """
    BASE = "https://api.2captcha.com"

    def __init__(self, api_key: str, timeout: int = 120) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def solve(self, captcha_type: CaptchaType, params: Dict[str, Any]) -> CaptchaSolveResult:
        start = time.monotonic()
        try:
            # Try SDK first if available (reuses existing proven path for Turnstile)
            if captcha_type == CaptchaType.TURNSTILE:
                result = await self._solve_via_sdk(params)
            elif captcha_type == CaptchaType.IMAGE_TEXT:
                result = await self._solve_image_async(params)
            else:
                result = await self._solve_via_api(captcha_type, params)
            latency = (time.monotonic() - start) * 1000
            if result:
                return CaptchaSolveResult(
                    success=True,
                    token=result,
                    provider="twocaptcha",
                    captcha_type=captcha_type,
                    latency_ms=latency,
                )
            return CaptchaSolveResult(success=False, error="empty result", captcha_type=captcha_type)
        except Exception as exc:
            return CaptchaSolveResult(success=False, error=str(exc), captcha_type=captcha_type)

    async def _solve_via_sdk(self, params: Dict[str, Any]) -> Optional[str]:
        """Use the TwoCaptcha Python SDK (sync) wrapped in a thread."""
        try:
            from twocaptcha import TwoCaptcha  # type: ignore
        except ImportError:
            logger.warning("[twocaptcha] SDK not installed, falling back to HTTP API")
            return await self._solve_via_api(CaptchaType.TURNSTILE, params)

        def _sync() -> Optional[str]:
            solver = TwoCaptcha(self.api_key)
            result = solver.turnstile(
                sitekey=params.get("sitekey", ""),
                url=params.get("pageurl", ""),
                action=params.get("action"),
                data=params.get("data"),
                pagedata=params.get("pagedata"),
                useragent=params.get("userAgent"),
            )
            return result["code"] if result else None

        return await asyncio.to_thread(_sync)

    async def _solve_image_async(self, params: Dict[str, Any]) -> Optional[str]:
        """Solve an image/text captcha including Chinese characters."""
        img_b64 = params.get("image_base64") or params.get("body")
        if not img_b64:
            raise ValueError("missing image_base64 for ImageToText solve")
        # Detect Chinese hint from params or URL if provided
        lang_pool = "zh" if params.get("is_chinese") else "en"
        task: Dict[str, Any] = {
            "type": "ImageToTextTask",
            "body": img_b64,
            "phrase": False,
            "minLength": params.get("min_length", 2),
            "maxLength": params.get("max_length", 10),
            "comment": params.get("comment", ""),
        }
        async with aiohttp.ClientSession() as session:
            create_resp = await session.post(
                f"{self.BASE}/createTask",
                json={"clientKey": self.api_key, "task": task, "languagePool": lang_pool},
                timeout=aiohttp.ClientTimeout(total=30),
            )
            create_data = await create_resp.json()
            if create_data.get("errorId", 1) != 0:
                raise RuntimeError(create_data.get("errorDescription", "createTask failed"))
            task_id = create_data["taskId"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(3)
                res_resp = await session.post(
                    f"{self.BASE}/getTaskResult",
                    json={"clientKey": self.api_key, "taskId": task_id},
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                res_data = await res_resp.json()
                if res_data.get("errorId", 1) != 0:
                    raise RuntimeError(res_data.get("errorDescription", "getTaskResult failed"))
                if res_data.get("status") == "ready":
                    return (res_data.get("solution") or {}).get("text")
            raise TimeoutError("2Captcha image solve timed out")

    async def _solve_via_api(self, captcha_type: CaptchaType, params: Dict[str, Any]) -> Optional[str]:
        """Generic 2Captcha REST API call for reCAPTCHA, hCaptcha, FunCaptcha."""
        _task_type_map = {
            CaptchaType.RECAPTCHA_V2:  "RecaptchaV2TaskProxyless",
            CaptchaType.RECAPTCHA_V3:  "RecaptchaV3TaskProxyless",
            CaptchaType.HCAPTCHA:      "HCaptchaTaskProxyless",
            CaptchaType.FUNCAPTCHA:    "FunCaptchaTaskProxyless",
            CaptchaType.TURNSTILE:     "TurnstileTaskProxyless",
        }
        task_type = _task_type_map.get(captcha_type)
        if not task_type:
            raise ValueError(f"no 2captcha task type for {captcha_type}")

        task: Dict[str, Any] = {"type": task_type}
        if captcha_type == CaptchaType.FUNCAPTCHA:
            task["websiteURL"] = params.get("pageurl", "")
            task["websitePublicKey"] = params.get("publickey", "")
        else:
            task["websiteURL"] = params.get("pageurl", "")
            task["websiteKey"] = params.get("sitekey", "")
            if captcha_type == CaptchaType.RECAPTCHA_V3:
                task["pageAction"] = params.get("action", "")

        async with aiohttp.ClientSession() as session:
            create_resp = await session.post(
                f"{self.BASE}/createTask",
                json={"clientKey": self.api_key, "task": task},
                timeout=aiohttp.ClientTimeout(total=30),
            )
            create_data = await create_resp.json()
            if create_data.get("errorId", 1) != 0:
                raise RuntimeError(create_data.get("errorDescription", "createTask failed"))
            task_id = create_data["taskId"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(3)
                res_resp = await session.post(
                    f"{self.BASE}/getTaskResult",
                    json={"clientKey": self.api_key, "taskId": task_id},
                    timeout=aiohttp.ClientTimeout(total=30),
                )
                res_data = await res_resp.json()
                if res_data.get("errorId", 1) != 0:
                    raise RuntimeError(res_data.get("errorDescription", "getTaskResult failed"))
                if res_data.get("status") == "ready":
                    solution = res_data.get("solution") or {}
                    return (
                        solution.get("gRecaptchaResponse")
                        or solution.get("token")
                        or solution.get("text")
                    )
            raise TimeoutError(f"2Captcha solve timed out for {captcha_type}")


# ---------------------------------------------------------------------------
# Public solver router
# ---------------------------------------------------------------------------

class CaptchaSolver:
    """
    Unified async captcha solver.

    Routing strategy:
    - TURNSTILE    : 2Captcha first (existing proven path) → CapSolver fallback
    - reCAPTCHA*   : CapSolver first → 2Captcha fallback
    - hCaptcha     : CapSolver first → 2Captcha fallback
    - FunCaptcha   : CapSolver first → 2Captcha fallback
    - IMAGE_TEXT   : 2Captcha first (better language coverage) → CapSolver fallback
    - UNKNOWN      : returns failure immediately (extension auto-pass is the only hope)
    """

    # Primary provider by captcha type: "capsolver" | "twocaptcha"
    _PRIMARY: Dict[CaptchaType, str] = {
        CaptchaType.TURNSTILE:    "twocaptcha",
        CaptchaType.RECAPTCHA_V2: "capsolver",
        CaptchaType.RECAPTCHA_V3: "capsolver",
        CaptchaType.HCAPTCHA:     "capsolver",
        CaptchaType.FUNCAPTCHA:   "capsolver",
        CaptchaType.IMAGE_TEXT:   "twocaptcha",
    }

    def __init__(
        self,
        capsolver_api_key: str = "",
        twocaptcha_api_key: str = "",
        timeout_seconds: int = 120,
    ) -> None:
        self._cap = _CapSolverProvider(capsolver_api_key, timeout_seconds) if capsolver_api_key else None
        self._tc = _TwoCaptchaProvider(twocaptcha_api_key, timeout_seconds) if twocaptcha_api_key else None
        self.timeout = timeout_seconds

    @property
    def has_any_provider(self) -> bool:
        return self._cap is not None or self._tc is not None

    def _providers_for(self, captcha_type: CaptchaType) -> List[Any]:
        """Return providers in priority order for the given captcha type."""
        primary = self._PRIMARY.get(captcha_type, "capsolver")
        if primary == "capsolver":
            ordered = [self._cap, self._tc]
        else:
            ordered = [self._tc, self._cap]
        return [p for p in ordered if p is not None]

    async def solve(self, captcha_type: CaptchaType, params: Dict[str, Any]) -> CaptchaSolveResult:
        """
        Try providers in priority order and return the first successful result.
        If no provider is configured or the type is UNKNOWN, returns failure.
        """
        if captcha_type == CaptchaType.UNKNOWN:
            return CaptchaSolveResult(
                success=False,
                error="unknown captcha type – no API solve available",
                captcha_type=captcha_type,
            )

        providers = self._providers_for(captcha_type)
        if not providers:
            return CaptchaSolveResult(
                success=False,
                error="no captcha API keys configured",
                captcha_type=captcha_type,
            )

        last_result: Optional[CaptchaSolveResult] = None
        for provider in providers:
            pname = "capsolver" if isinstance(provider, _CapSolverProvider) else "twocaptcha"
            logger.info("[captcha-solver] trying %s for type=%s", pname, captcha_type)
            try:
                result = await provider.solve(captcha_type, params)
                if result.success:
                    return result
                logger.warning("[captcha-solver] %s failed: %s – trying fallback", pname, result.error)
                last_result = result
            except Exception as exc:
                logger.warning("[captcha-solver] %s raised: %s – trying fallback", pname, exc)
                last_result = CaptchaSolveResult(success=False, error=str(exc), captcha_type=captcha_type)

        return last_result or CaptchaSolveResult(
            success=False, error="all providers failed", captcha_type=captcha_type
        )
