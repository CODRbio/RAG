"""
Shared page-level captcha detection, param extraction, token application, and solve flow.

Used by WebContentFetcher and can be used by PaperDownloader to avoid duplication.
Provider policy: Cloudflare/Turnstile -> 2Captcha only; others -> CapSolver first, 2Captcha fallback (via CaptchaSolver).
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Dict, Optional

from src.log import get_logger

from .captcha_solver import CaptchaSolveResult, CaptchaSolver, CaptchaType

logger = get_logger(__name__)

# Intercept script for Cloudflare Turnstile param extraction (sitekey, callback, etc.)
INTERCEPT_SCRIPT = """
    const initTurnstileInterception = setInterval(()=>{
        if (window.turnstile) {
            clearInterval(initTurnstileInterception);
            const originalRender = window.turnstile.render;
            window.turnstile.render = function(container, options) {
                window.__cf_intercepted_params = {
                    sitekey: options.sitekey,
                    pageurl: window.location.href,
                    data: options.cData,
                    pagedata: options.chlPageData,
                    action: options.action,
                    userAgent: navigator.userAgent,
                };
                if (options.callback) {
                    window.__cf_callback = options.callback;
                }
                return originalRender.apply(this, arguments);
            };
        }
    }, 50);
"""


async def detect_captcha_type(page) -> CaptchaType:
    """
    Inspect the current page and return the most specific CaptchaType.
    Returns CaptchaType.UNKNOWN when nothing is recognised.
    """
    try:
        # --- Cloudflare Turnstile ---
        turnstile_selectors = [
            ".cf-turnstile",
            "div[data-sitekey]",
            'input[name="cf-turnstile-response"]',
            "#cf-hcaptcha-container",
            'iframe[src*="challenges.cloudflare.com"]',
            "#challenge-running",
            "#cf-challenge-running",
        ]
        for sel in turnstile_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    return CaptchaType.TURNSTILE
            except Exception:
                pass

        # --- reCAPTCHA ---
        try:
            rc_el = await page.query_selector('div.g-recaptcha[data-sitekey], iframe[src*="recaptcha"]')
            if rc_el:
                v3_el = await page.query_selector('div.g-recaptcha[data-size="invisible"]')
                if v3_el:
                    return CaptchaType.RECAPTCHA_V3
                return CaptchaType.RECAPTCHA_V2
        except Exception:
            pass

        try:
            has_rc = await page.evaluate(
                "() => !!(window.grecaptcha && (window.grecaptcha.render || window.grecaptcha.execute))"
            )
            if has_rc:
                invisible = await page.evaluate(
                    "() => !!document.querySelector('.g-recaptcha[data-size=\"invisible\"]')"
                )
                return CaptchaType.RECAPTCHA_V3 if invisible else CaptchaType.RECAPTCHA_V2
        except Exception:
            pass

        # --- hCaptcha ---
        try:
            hc_el = await page.query_selector('div.h-captcha[data-sitekey], iframe[src*="hcaptcha.com"]')
            if hc_el:
                return CaptchaType.HCAPTCHA
        except Exception:
            pass
        try:
            has_hc = await page.evaluate("() => !!(window.hcaptcha)")
            if has_hc:
                return CaptchaType.HCAPTCHA
        except Exception:
            pass

        # --- FunCaptcha ---
        try:
            fc_el = await page.query_selector(
                '#FunCaptcha, [data-pkey], iframe[src*="arkoselabs"], iframe[src*="funcaptcha"]'
            )
            if fc_el:
                return CaptchaType.FUNCAPTCHA
        except Exception:
            pass

        # --- Image/Text captcha ---
        try:
            img_captcha = await page.query_selector(
                'img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"], '
                'img[class*="captcha"], .captcha-image, #captcha-image, '
                'img[src*="验证码"], img[alt*="验证码"]'
            )
            if img_captcha:
                return CaptchaType.IMAGE_TEXT
        except Exception:
            pass

        # --- Text fallback ---
        try:
            body_text = await page.evaluate(
                '() => (document.body ? document.body.innerText : "").toLowerCase()'
            )
            if any(k in body_text for k in ("recaptcha", "g-recaptcha")):
                return CaptchaType.RECAPTCHA_V2
            if "hcaptcha" in body_text:
                return CaptchaType.HCAPTCHA
            if any(k in body_text for k in ("funcaptcha", "arkoselabs")):
                return CaptchaType.FUNCAPTCHA
            if any(k in body_text for k in ("captcha", "验证码", "robot", "human")):
                return CaptchaType.IMAGE_TEXT
        except Exception:
            pass

    except Exception as e:
        logger.debug("detect_captcha_type error: %s", e)
    return CaptchaType.UNKNOWN


async def _check_cloudflare_indicators(page) -> bool:
    """Check for Cloudflare challenge indicators on the page."""
    try:
        title = await page.title()
        cloudflare_titles = [
            "Just a moment",
            "Attention Required",
            "请稍候",
            "請稍候",
            "Un moment",
            "Einen Moment",
            "Un momento",
        ]
        for cf_title in cloudflare_titles:
            if cf_title in title:
                logger.debug("Cloudflare title detected: %s", title)
                return True

        key_selectors = [
            "#challenge-running",
            "#cf-challenge-running",
            'iframe[src*="challenges.cloudflare.com"]',
            '#turnstile-wrapper:not([style*="display: none"])',
            ".cf-browser-verification",
            "#cf-spinner-please-wait",
            'input[name="cf-turnstile-response"]',
            ".cf-turnstile",
            "div[data-sitekey]",
            "#cf-hcaptcha-container",
            ".hcaptcha-box",
        ]
        for selector in key_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    if "input[name=" in selector:
                        return True
                    if await element.is_visible():
                        return True
            except Exception:
                pass

        try:
            page_text = await page.evaluate('() => document.body ? document.body.innerText : ""')
            cf_indicators = [
                "are you a robot",
                "verify you are human",
                "checking your browser",
                "this process is automatic",
                "please complete the security check",
            ]
            for indicator in cf_indicators:
                if indicator in page_text.lower():
                    return True
        except Exception:
            pass

        has_cf = await page.evaluate(
            """() => !!(
                window._cf_chl_opt ||
                window.___cf_chl_ctx ||
                (window.cf && window.cf.chl_done) ||
                window.turnstile ||
                document.querySelector('[data-ray]') ||
                document.querySelector('meta[name="cf-2fa"]')
            )"""
        )
        if has_cf:
            return True
    except Exception as e:
        logger.debug("_check_cloudflare_indicators error: %s", e)
    return False


async def is_cloudflare_verifying(page, detection_duration: float = 1.5) -> bool:
    """Return True if the page appears to be showing a Cloudflare challenge."""
    try:
        if await _check_cloudflare_indicators(page):
            return True
        start = time.monotonic()
        while time.monotonic() - start < detection_duration:
            await asyncio.sleep(0.3)
            if await _check_cloudflare_indicators(page):
                return True
    except Exception as e:
        logger.debug("is_cloudflare_verifying error: %s", e)
    return False


async def get_turnstile_params(
    page,
    intercept_script: str = INTERCEPT_SCRIPT,
    timeout_seconds: float = 60,
) -> Optional[Dict[str, Any]]:
    """Inject intercept script, reload, and poll for window.__cf_intercepted_params."""
    try:
        await page.add_init_script(intercept_script)
        await page.reload()
        deadline = time.monotonic() + max(timeout_seconds, 1.0)
        while time.monotonic() < deadline:
            params = await page.evaluate("() => window.__cf_intercepted_params || null")
            if params:
                return params
            await asyncio.sleep(0.5)
        logger.warning("get_turnstile_params timed out")
    except Exception as e:
        logger.debug("get_turnstile_params error: %s", e)
    return None


async def send_token_to_page(page, token: str) -> None:
    """Inject Turnstile token: call __cf_callback or fill input and submit form."""
    await page.evaluate(
        """(token) => {
        if (window.__cf_callback) {
            try { window.__cf_callback(token); return; } catch (e) {}
        }
        let input = document.querySelector('input[name="cf-turnstile-response"]');
        if (input) {
            input.value = token;
            let form = input.closest('form');
            if (form) form.submit();
        }
    }""",
        token,
    )


async def _extract_recaptcha_params(page) -> Optional[Dict[str, Any]]:
    try:
        sitekey = await page.evaluate(
            """() => {
                const el = document.querySelector('.g-recaptcha[data-sitekey]');
                if (el) return el.getAttribute('data-sitekey');
                const scripts = Array.from(document.querySelectorAll('script'));
                for (const s of scripts) {
                    const m = s.textContent.match(/['"]sitekey['"]\\s*:\\s*['"]([^'"]+)['"]/);
                    if (m) return m[1];
                }
                return null;
            }"""
        )
        if not sitekey:
            iframe = await page.query_selector('iframe[src*="recaptcha"]')
            if iframe:
                import re
                src = await iframe.get_attribute("src") or ""
                m = re.search(r"[?&]k=([^&]+)", src)
                sitekey = m.group(1) if m else None
        if not sitekey:
            return None
        action = await page.evaluate(
            """() => {
                const el = document.querySelector('.g-recaptcha[data-action]');
                return el ? el.getAttribute('data-action') : 'submit';
            }"""
        ) or "submit"
        return {"sitekey": sitekey, "pageurl": page.url, "action": action}
    except Exception as e:
        logger.debug("_extract_recaptcha_params error: %s", e)
        return None


async def _extract_hcaptcha_params(page) -> Optional[Dict[str, Any]]:
    try:
        sitekey = await page.evaluate(
            """() => {
                const el = document.querySelector('.h-captcha[data-sitekey]');
                if (el) return el.getAttribute('data-sitekey');
                const iframe = document.querySelector('iframe[src*="hcaptcha.com"]');
                if (iframe) {
                    const m = iframe.src.match(/[?&]sitekey=([^&]+)/);
                    if (m) return m[1];
                }
                return null;
            }"""
        )
        return {"sitekey": sitekey, "pageurl": page.url} if sitekey else None
    except Exception as e:
        logger.debug("_extract_hcaptcha_params error: %s", e)
        return None


async def _extract_funcaptcha_params(page) -> Optional[Dict[str, Any]]:
    try:
        publickey = await page.evaluate(
            """() => {
                const el = document.querySelector('[data-pkey], #FunCaptcha');
                if (el) return el.getAttribute('data-pkey') || el.getAttribute('data-public-key');
                return null;
            }"""
        )
        return {"publickey": publickey, "pageurl": page.url} if publickey else None
    except Exception as e:
        logger.debug("_extract_funcaptcha_params error: %s", e)
        return None


async def _extract_image_captcha_params(page) -> Optional[Dict[str, Any]]:
    try:
        img_el = await page.query_selector(
            'img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"], '
            'img[class*="captcha"], .captcha-image, #captcha-image, '
            'img[src*="验证码"], img[alt*="验证码"]'
        )
        if img_el:
            img_bytes = await img_el.screenshot()
            img_b64 = base64.b64encode(img_bytes).decode()
            body_text = await page.evaluate('() => (document.body || {innerText: ""}).innerText')
            is_chinese = bool(any(ord(c) > 0x4E00 for c in (body_text or "")[:200]))
            return {
                "image_base64": img_b64,
                "pageurl": page.url,
                "is_chinese": is_chinese,
            }
    except Exception as e:
        logger.debug("_extract_image_captcha_params error: %s", e)
    return None


async def extract_captcha_params(
    captcha_type: CaptchaType,
    page,
    *,
    intercept_script: str = INTERCEPT_SCRIPT,
    turnstile_timeout_seconds: float = 60,
) -> Optional[Dict[str, Any]]:
    """Dispatch to the correct extractor for the given captcha type."""
    if captcha_type == CaptchaType.TURNSTILE:
        return await get_turnstile_params(page, intercept_script, turnstile_timeout_seconds)
    if captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
        return await _extract_recaptcha_params(page)
    if captcha_type == CaptchaType.HCAPTCHA:
        return await _extract_hcaptcha_params(page)
    if captcha_type == CaptchaType.FUNCAPTCHA:
        return await _extract_funcaptcha_params(page)
    if captcha_type == CaptchaType.IMAGE_TEXT:
        return await _extract_image_captcha_params(page)
    return None


async def apply_captcha_token(page, captcha_type: CaptchaType, token: str) -> None:
    """Inject a solved captcha token/text back into the page."""
    if captcha_type == CaptchaType.TURNSTILE:
        await send_token_to_page(page, token)
    elif captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
        await page.evaluate(
            """(token) => {
            const ta = document.getElementById('g-recaptcha-response');
            if (ta) { ta.value = token; ta.style.display = 'block'; }
            const widget = document.querySelector('.g-recaptcha');
            if (widget) {
                const cb = widget.getAttribute('data-callback');
                if (cb && typeof window[cb] === 'function') { try { window[cb](token); } catch(e) {} }
            }
            if (window.grecaptcha && window.grecaptcha.__oncomplete__) {
                try { window.grecaptcha.__oncomplete__(token); } catch(e) {}
            }
        }""",
            token,
        )
    elif captcha_type == CaptchaType.HCAPTCHA:
        await page.evaluate(
            """(token) => {
            const ta = document.querySelector('[name="h-captcha-response"]');
            if (ta) ta.value = token;
            if (window.hcaptcha) { try { window.hcaptcha.execute(); } catch(e) {} }
        }""",
            token,
        )
    elif captcha_type == CaptchaType.IMAGE_TEXT:
        await page.evaluate(
            """(text) => {
            const sel = 'input[name*="captcha"], input[id*="captcha"], input[placeholder*="captcha"],'
                      + 'input[name*="验证码"], input[id*="验证码"]';
            const inp = document.querySelector(sel);
            if (inp) {
                inp.value = text;
                const form = inp.closest('form');
                if (form) form.submit();
            }
        }""",
            token,
        )


def _solve_turnstile_sync(apikey: str, params: Dict[str, Any]) -> Optional[str]:
    """Sync 2Captcha Turnstile solve (run in thread)."""
    try:
        from twocaptcha import TwoCaptcha
    except ImportError:
        logger.warning("twocaptcha not installed; pip install 2captcha-python")
        return None
    solver = TwoCaptcha(apikey)
    try:
        result = solver.turnstile(
            sitekey=params.get("sitekey", ""),
            url=params.get("pageurl", ""),
            action=params.get("action"),
            data=params.get("data"),
            pagedata=params.get("pagedata"),
            useragent=params.get("userAgent"),
        )
        return result.get("code") if result else None
    except Exception as e:
        logger.debug("2Captcha Turnstile solve error: %s", e)
        return None


async def solve_turnstile_via_2captcha(
    page,
    twocaptcha_api_key: str,
    *,
    intercept_script: str = INTERCEPT_SCRIPT,
    timeout_seconds: float = 60,
    max_retries: int = 2,
) -> bool:
    """
    Solve Cloudflare Turnstile using 2Captcha only (no CapSolver).
    Returns True if the challenge was solved or not present, False on failure.
    """
    if not twocaptcha_api_key:
        logger.warning("No 2Captcha API key; cannot solve Turnstile")
        return False

    for attempt in range(max_retries):
        try:
            if not await is_cloudflare_verifying(page, detection_duration=0.5):
                return True

            params = await get_turnstile_params(page, intercept_script, timeout_seconds)
            if not params:
                logger.warning("Turnstile params extraction failed (attempt %s)", attempt + 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                continue

            token = await asyncio.to_thread(_solve_turnstile_sync, twocaptcha_api_key, params)
            if not token:
                if attempt < max_retries - 1:
                    await asyncio.sleep(4)
                continue

            await send_token_to_page(page, token)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                pass
            await asyncio.sleep(1)

            if not await is_cloudflare_verifying(page, detection_duration=1.0):
                logger.info("Turnstile solved successfully")
                return True
        except Exception as e:
            logger.debug("solve_turnstile_via_2captcha attempt %s: %s", attempt + 1, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(3)
    return False


async def run_captcha_flow(
    page,
    captcha_solver: CaptchaSolver,
    twocaptcha_api_key: str,
    *,
    intercept_script: str = INTERCEPT_SCRIPT,
    captcha_timeout_seconds: float = 120,
    max_retries: int = 2,
    load_state_timeout_ms: int = 10000,
) -> bool:
    """
    Single entry for content-fetcher / lightweight callers.

    - If no captcha or UNKNOWN: return True (page usable).
    - If Turnstile/Cloudflare: solve with 2Captcha only, then return success/fail.
    - Otherwise: detect type -> extract params -> CaptchaSolver (CapSolver first, 2Captcha fallback) -> apply token.

    Returns True if page is usable (no captcha, or solved); False if solving failed.
    """
    for attempt in range(max_retries):
        try:
            captcha_type = await detect_captcha_type(page)
            if captcha_type == CaptchaType.UNKNOWN:
                return True

            if captcha_type == CaptchaType.TURNSTILE:
                return await solve_turnstile_via_2captcha(
                    page,
                    twocaptcha_api_key,
                    intercept_script=intercept_script,
                    timeout_seconds=min(captcha_timeout_seconds, 60),
                    max_retries=max_retries,
                )

            if not captcha_solver.has_any_provider:
                logger.warning("No captcha API keys configured; cannot solve type=%s", captcha_type)
                return False

            params = await extract_captcha_params(
                captcha_type,
                page,
                intercept_script=intercept_script,
                turnstile_timeout_seconds=min(captcha_timeout_seconds, 60),
            )
            if not params:
                logger.warning("Could not extract params for type=%s", captcha_type)
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                continue

            result: CaptchaSolveResult = await captcha_solver.solve(captcha_type, params)
            if not result.success or not result.token:
                logger.warning("Solve failed type=%s error=%s", captcha_type, result.error)
                if attempt < max_retries - 1:
                    await asyncio.sleep(4)
                continue

            await apply_captcha_token(page, captcha_type, result.token)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=load_state_timeout_ms)
            except Exception:
                pass
            await asyncio.sleep(1)

            post_type = await detect_captcha_type(page)
            if post_type == CaptchaType.UNKNOWN:
                return True
        except Exception as e:
            logger.debug("run_captcha_flow attempt %s: %s", attempt + 1, e)
        if attempt < max_retries - 1:
            await asyncio.sleep(3)
    return False
