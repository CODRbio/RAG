"""
2Captcha API client for headless captcha solving.

Supports:
- reCAPTCHA v2 via in.php/res.php
- Image captcha OCR via base64 upload
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional

import aiohttp

from src.log import get_logger

logger = get_logger(__name__)


class TwoCaptchaAPI:
    IN_URL = "https://2captcha.com/in.php"
    RES_URL = "https://2captcha.com/res.php"

    def __init__(self, api_key: str, timeout_seconds: int = 120) -> None:
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = max(10, int(timeout_seconds or 120))
        self.enabled = bool(self.api_key)

    async def _submit_task(self, data: dict) -> Optional[str]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        payload = {"key": self.api_key, "json": 1}
        payload.update(data)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.IN_URL, data=payload) as resp:
                body = await resp.json(content_type=None)
        if not isinstance(body, dict) or body.get("status") != 1:
            logger.warning("2Captcha submit failed: %s", body)
            return None
        rid = str(body.get("request") or "")
        return rid or None

    async def _poll_result(self, request_id: str) -> Optional[str]:
        deadline = time.monotonic() + self.timeout_seconds
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time.monotonic() < deadline:
                await asyncio.sleep(3)
                params = {
                    "key": self.api_key,
                    "action": "get",
                    "id": request_id,
                    "json": 1,
                }
                async with session.get(self.RES_URL, params=params) as resp:
                    body = await resp.json(content_type=None)
                if not isinstance(body, dict):
                    continue
                if body.get("status") == 1:
                    text = str(body.get("request") or "").strip()
                    return text or None
                req = str(body.get("request") or "")
                if req == "CAPCHA_NOT_READY":
                    continue
                logger.warning("2Captcha poll failed: %s", body)
                return None
        logger.warning("2Captcha poll timeout after %ss", self.timeout_seconds)
        return None

    async def solve_recaptcha_v2(self, page, website_url: str, site_key: str, data_s: str = "") -> bool:
        if not self.enabled or not site_key:
            return False
        try:
            params: dict = {
                "method": "userrecaptcha",
                "googlekey": site_key,
                "pageurl": website_url,
            }
            if data_s:
                params["data-s"] = data_s
                logger.info("2Captcha: using data-s parameter for Google sorry page")
            req_id = await self._submit_task(params)
            if not req_id:
                return False
            token = await self._poll_result(req_id)
            if not token:
                return False
            from src.retrieval.capsolver_api import _inject_recaptcha_token_and_submit
            await _inject_recaptcha_token_and_submit(page, token)
            logger.info("2Captcha recaptcha token injected")
            return True
        except Exception as e:
            logger.warning("2Captcha recaptcha solve failed: %s", e)
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
            req_id = await self._submit_task(
                {
                    "method": "base64",
                    "body": body_b64,
                }
            )
            if not req_id:
                return False
            text = await self._poll_result(req_id)
            if not text:
                return False
            input_el = await page.query_selector(input_selector)
            if input_el is None:
                return False
            await input_el.fill("")
            await input_el.type(text)
            await input_el.press("Enter")
            logger.info("2Captcha image captcha text injected")
            return True
        except Exception as e:
            logger.warning("2Captcha image solve failed: %s", e)
            return False

    async def solve_captcha_on_page(self, page) -> bool:
        if not self.enabled:
            return False
        try:
            from src.retrieval.capsolver_api import _extract_sitekey

            has_recaptcha = await page.query_selector(
                'iframe[src*="recaptcha"], .g-recaptcha, #recaptcha, [data-sitekey]'
            )
            if has_recaptcha:
                from src.retrieval.capsolver_api import _extract_data_s
                site_key = await _extract_sitekey(page)
                data_s = await _extract_data_s(page)
                website_url = (page.url or "").strip()
                logger.info(
                    "2Captcha: type=recaptcha sitekey=%s data_s=%s",
                    site_key[:12] + "..." if site_key else "(empty)",
                    "yes" if data_s else "no",
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
            logger.warning("2Captcha auto solve failed: %s", e)
            return False

