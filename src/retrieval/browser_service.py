import asyncio
from typing import Optional

from playwright.async_api import async_playwright, Browser, Playwright

from src.log import get_logger

logger = get_logger(__name__)


class SharedBrowserService:
    """Process-wide shared Chromium service for CDP consumers."""

    _lock: asyncio.Lock = asyncio.Lock()
    _playwright: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _cdp_url: Optional[str] = None
    _headless: bool = True

    @classmethod
    async def start(cls, port: int = 9222, headless: bool = True) -> None:
        """Start a shared Chromium process with remote-debugging enabled."""
        async with cls._lock:
            if cls._browser is not None and cls._playwright is not None:
                return

            cls._headless = headless
            cls._playwright = await async_playwright().start()
            cls._browser = await cls._playwright.chromium.launch(
                headless=headless,
                args=[f"--remote-debugging-port={port}"],
            )
            cls._cdp_url = f"http://127.0.0.1:{port}"
            logger.info("[shared-browser] started at %s (headless=%s)", cls._cdp_url, headless)

    @classmethod
    def get_cdp_url(cls) -> Optional[str]:
        """Return local CDP endpoint when shared browser is ready."""
        return cls._cdp_url

    @classmethod
    async def stop(cls) -> None:
        """Stop shared Chromium and Playwright."""
        async with cls._lock:
            browser = cls._browser
            playwright = cls._playwright
            cls._browser = None
            cls._playwright = None
            cls._cdp_url = None

            if browser is not None:
                try:
                    await browser.close()
                except Exception as e:
                    logger.warning("[shared-browser] close browser failed: %s", e)

            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception as e:
                    logger.warning("[shared-browser] stop playwright failed: %s", e)

            logger.info("[shared-browser] stopped")
