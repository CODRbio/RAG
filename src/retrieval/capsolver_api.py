"""
CapSolver API client for headless captcha solving.

Supports:
- ReCaptchaV2TaskProxyLess (Google reCAPTCHA v2)
- ImageToTextTask (image captcha OCR)
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Dict, Optional

import aiohttp

from src.log import get_logger

logger = get_logger(__name__)

# JS to extract sitekey from all known locations on Google sorry / reCAPTCHA pages
_EXTRACT_SITEKEY_JS = """
() => {
    // 1. data-sitekey attribute
    const byData = document.querySelector("[data-sitekey]");
    if (byData) {
        const k = (byData.getAttribute("data-sitekey") || "").trim();
        if (k) return k;
    }
    // 2. iframe src ?k= or ?sitekey=
    const frames = document.querySelectorAll('iframe[src*="recaptcha"]');
    for (const f of frames) {
        try {
            const u = new URL(f.src);
            const k = (u.searchParams.get("k") || u.searchParams.get("sitekey") || "").trim();
            if (k) return k;
        } catch (_e) {}
    }
    // 3. script src containing recaptcha/api.js?render=
    const scripts = document.querySelectorAll('script[src*="recaptcha"]');
    for (const s of scripts) {
        try {
            const u = new URL(s.src);
            const k = (u.searchParams.get("render") || "").trim();
            if (k && k !== "explicit") return k;
        } catch (_e) {}
    }
    // 4. inline script containing sitekey / render param
    const all = document.querySelectorAll("script:not([src])");
    for (const s of all) {
        const t = s.textContent || "";
        const m = t.match(/['"]sitekey['"]\\s*:\\s*['"]([A-Za-z0-9_-]{20,})['"]/);
        if (m) return m[1];
        const m2 = t.match(/grecaptcha\\.(?:render|execute)\\s*\\([^,]*,\\s*\\{[^}]*sitekey\\s*:\\s*['"]([A-Za-z0-9_-]{20,})['"]/);
        if (m2) return m2[1];
    }
    // 5. ___grecaptcha_cfg
    try {
        if (typeof ___grecaptcha_cfg !== "undefined" && ___grecaptcha_cfg.clients) {
            for (const c of Object.values(___grecaptcha_cfg.clients)) {
                for (const v of Object.values(c)) {
                    if (v && typeof v === "object" && v.sitekey) return v.sitekey;
                }
            }
        }
    } catch (_e) {}
    return "";
}
"""

# JS to inject reCAPTCHA token and trigger callback + form submit.
# Google sorry page requires BOTH setting the textarea AND submitting the form.
_INJECT_TOKEN_JS = """
(tk) => {
    // 1. Fill ALL g-recaptcha-response textareas (visible and hidden)
    const tas = document.querySelectorAll('textarea[id*="g-recaptcha-response"], textarea[name="g-recaptcha-response"]');
    for (const ta of tas) {
        ta.value = tk;
        ta.innerHTML = tk;
        ta.style.display = "block";
    }

    // 2. Try reCAPTCHA callback via ___grecaptcha_cfg (deep walk)
    let cbCalled = false;
    function walkObj(obj, depth) {
        if (!obj || typeof obj !== "object" || depth > 6) return;
        if (typeof obj.callback === "function") {
            try { obj.callback(tk); cbCalled = true; } catch (_) {}
        }
        for (const v of Object.values(obj)) {
            if (v && typeof v === "object") walkObj(v, depth + 1);
        }
    }
    try {
        if (typeof ___grecaptcha_cfg !== "undefined" && ___grecaptcha_cfg.clients) {
            walkObj(___grecaptcha_cfg.clients, 0);
        }
    } catch (_e) {}

    // 3. Try common window-level callback names
    if (!cbCalled) {
        for (const name of ["onCaptchaSuccess", "captchaCallback", "onRecaptchaSuccess", "submitCallback", "onSuccess"]) {
            try { if (typeof window[name] === "function") { window[name](tk); cbCalled = true; break; } } catch (_) {}
        }
    }

    // 4. ALWAYS also submit the form (Google sorry page needs form POST even after callback)
    const captchaForm = document.getElementById("captcha-form");
    if (captchaForm) {
        captchaForm.submit();
        return;
    }
    const btn = document.querySelector('input[type="submit"], button[type="submit"]');
    if (btn) { btn.click(); return; }
    const forms = document.querySelectorAll("form");
    if (forms.length > 0) forms[0].submit();
}
"""


async def _extract_sitekey(page) -> str:
    """Extract reCAPTCHA sitekey from page using multiple strategies."""
    try:
        return (await page.evaluate(_EXTRACT_SITEKEY_JS) or "").strip()
    except Exception as e:
        logger.debug("sitekey extraction failed: %s", e)
        return ""


async def _extract_data_s(page) -> str:
    """Extract Google sorry page data-s parameter (required by some CAPTCHA services)."""
    try:
        return (await page.evaluate("""
            () => {
                const el = document.querySelector('[data-s]');
                if (el) return el.getAttribute('data-s') || "";
                const m = document.documentElement.innerHTML.match(/data-s="([^"]+)"/);
                return m ? m[1] : "";
            }
        """) or "").strip()
    except Exception:
        return ""


async def _inject_recaptcha_token_and_submit(page, token: str) -> None:
    """Inject token into g-recaptcha-response and trigger callback / form submit."""
    try:
        await page.evaluate(_INJECT_TOKEN_JS, token)
    except Exception as e:
        logger.debug("token injection script failed: %s", e)


class CapSolverAPI:
    BASE_URL = "https://api.capsolver.com"

    def __init__(self, api_key: str, timeout_seconds: int = 120) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = max(10, int(timeout_seconds or 120))
        self.enabled = bool(self.api_key)

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.BASE_URL}{endpoint}", json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"CapSolver HTTP {resp.status}: {text[:300]}")
                data = await resp.json(content_type=None)
                if not isinstance(data, dict):
                    raise RuntimeError("CapSolver response is not JSON object")
                return data

    async def _create_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"clientKey": self.api_key, "task": task}
        return await self._post("/createTask", payload)

    async def _get_task_result(self, task_id: str) -> Dict[str, Any]:
        payload = {"clientKey": self.api_key, "taskId": task_id}
        return await self._post("/getTaskResult", payload)

    async def solve_recaptcha_v2(self, page, website_url: str, site_key: str, data_s: str = "") -> bool:
        if not self.enabled:
            return False
        if not site_key:
            logger.warning("CapSolver recaptcha skipped: empty site_key")
            return False

        try:
            task: dict = {
                "type": "ReCaptchaV2TaskProxyLess",
                "websiteURL": website_url,
                "websiteKey": site_key,
            }
            if data_s:
                task["recaptchaDataSValue"] = data_s
                logger.info("CapSolver: using data-s parameter for Google sorry page")
            created = await self._create_task(task)
            if created.get("errorId", 0) != 0:
                logger.warning("CapSolver createTask(recpatcha) error: %s", created)
                return False
            task_id = str(created.get("taskId") or "")
            if not task_id:
                logger.warning("CapSolver createTask(recpatcha) missing taskId")
                return False

            deadline = time.monotonic() + self.timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(2)
                result = await self._get_task_result(task_id)
                if result.get("errorId", 0) != 0:
                    logger.warning("CapSolver getTaskResult(recpatcha) error: %s", result)
                    return False
                if result.get("status") != "ready":
                    continue
                token = ((result.get("solution") or {}).get("gRecaptchaResponse") or "").strip()
                if not token:
                    logger.warning("CapSolver recaptcha ready but empty token")
                    return False
                await _inject_recaptcha_token_and_submit(page, token)
                logger.info("CapSolver recaptcha token injected")
                return True

            logger.warning("CapSolver recaptcha timeout after %ss", self.timeout_seconds)
            return False
        except Exception as e:
            logger.warning("CapSolver recaptcha solve failed: %s", e)
            return False

    async def solve_image_captcha(self, page, img_selector: str, input_selector: str) -> bool:
        if not self.enabled:
            return False
        try:
            img = await page.query_selector(img_selector)
            if img is None:
                return False
            raw = await img.screenshot(type="png")
            body_b64 = base64.b64encode(raw).decode("ascii")
            created = await self._create_task(
                {
                    "type": "ImageToTextTask",
                    "module": "common",
                    "body": body_b64,
                }
            )
            if created.get("errorId", 0) != 0:
                logger.warning("CapSolver createTask(image) error: %s", created)
                return False

            solution = created.get("solution") or {}
            text = (solution.get("text") or "").strip()
            if not text:
                # Fallback: some responses still require polling
                task_id = str(created.get("taskId") or "")
                if not task_id:
                    return False
                deadline = time.monotonic() + self.timeout_seconds
                while time.monotonic() < deadline:
                    await asyncio.sleep(2)
                    result = await self._get_task_result(task_id)
                    if result.get("errorId", 0) != 0:
                        logger.warning("CapSolver getTaskResult(image) error: %s", result)
                        return False
                    if result.get("status") != "ready":
                        continue
                    text = ((result.get("solution") or {}).get("text") or "").strip()
                    break

            if not text:
                logger.warning("CapSolver image solved with empty text")
                return False

            input_el = await page.query_selector(input_selector)
            if input_el is None:
                return False
            await input_el.fill("")
            await input_el.type(text)
            await input_el.press("Enter")
            logger.info("CapSolver image captcha text injected")
            return True
        except Exception as e:
            logger.warning("CapSolver image captcha solve failed: %s", e)
            return False

    async def solve_captcha_on_page(self, page) -> bool:
        """
        Auto-detect captcha type and solve.
        Returns True if a solving action was performed successfully.
        """
        if not self.enabled:
            return False
        try:
            has_recaptcha = await page.query_selector(
                'iframe[src*="recaptcha"], .g-recaptcha, #recaptcha, [data-sitekey]'
            )
            if has_recaptcha:
                site_key = await _extract_sitekey(page)
                data_s = await _extract_data_s(page)
                website_url = (page.url or "").strip()
                logger.info(
                    "captcha type=recaptcha sitekey=%s data_s=%s url=%s",
                    site_key[:12] + "..." if site_key else "(empty)",
                    "yes" if data_s else "no",
                    website_url[:60],
                )
                return await self.solve_recaptcha_v2(page, website_url, site_key, data_s=data_s)

            has_image = await page.query_selector('img[src*="sorry/image"], form img')
            has_input = await page.query_selector('input[name="captcha"], form input[type="text"]')
            if has_image and has_input:
                return await self.solve_image_captcha(
                    page,
                    'img[src*="sorry/image"], form img',
                    'input[name="captcha"], form input[type="text"]',
                )
            return False
        except Exception as e:
            logger.warning("CapSolver auto solve failed: %s", e)
            return False

