import asyncio
from typing import Optional

from playwright.async_api import async_playwright, Browser, Playwright

from src.log import get_logger

logger = get_logger(__name__)


class SharedBrowserService:
    """Process-wide shared Chromium services for CDP consumers: headless + optional headed (minimized)."""

    _lock: asyncio.Lock = asyncio.Lock()

    # Headless instance (default)
    _playwright_hl: Optional[Playwright] = None
    _browser_hl: Optional[Browser] = None
    _cdp_url_hl: Optional[str] = None

    # Headed instance (optional, started minimized)
    _playwright_hd: Optional[Playwright] = None
    _browser_hd: Optional[Browser] = None
    _cdp_url_hd: Optional[str] = None

    # Backward compatibility: code that reads _headless expects the "default" shared browser to be headless
    _headless: bool = True

    @classmethod
    async def start(cls, port: int = 9222, headless: bool = True) -> None:
        """Start the headless shared Chromium process (default CDP)."""
        async with cls._lock:
            if cls._browser_hl is not None and cls._playwright_hl is not None:
                return

            cls._headless = True
            cls._playwright_hl = await async_playwright().start()
            cls._browser_hl = await cls._playwright_hl.chromium.launch(
                headless=True,
                args=[f"--remote-debugging-port={port}"],
            )
            cls._cdp_url_hl = f"http://127.0.0.1:{port}"
            logger.info("[shared-browser] headless started at %s", cls._cdp_url_hl)

    @classmethod
    async def start_headed(cls, port: int = 9223) -> None:
        """Start the headed shared Chromium process (minimized / off-screen)."""
        async with cls._lock:
            if cls._browser_hd is not None and cls._playwright_hd is not None:
                return

            cls._playwright_hd = await async_playwright().start()
            # Minimize: move window off-screen so it does not steal focus (macOS/Linux/Windows)
            args = [
                f"--remote-debugging-port={port}",
                "--window-position=-2000,-2000",
                "--window-size=1,1",
            ]
            cls._browser_hd = await cls._playwright_hd.chromium.launch(
                headless=False,
                args=args,
            )
            cls._cdp_url_hd = f"http://127.0.0.1:{port}"
            logger.info("[shared-browser] headed (minimized) started at %s", cls._cdp_url_hd)

    @classmethod
    def get_cdp_url(cls) -> Optional[str]:
        """Backward compatibility: return headless CDP endpoint."""
        return cls._cdp_url_hl

    @classmethod
    def get_cdp_url_headless(cls) -> Optional[str]:
        """Return headless CDP endpoint when ready."""
        return cls._cdp_url_hl

    @classmethod
    def get_cdp_url_headed(cls) -> Optional[str]:
        """Return headed CDP endpoint when ready."""
        return cls._cdp_url_hd

    @classmethod
    async def stop(cls) -> None:
        """Stop both headless and headed instances."""
        async with cls._lock:
            # Headless
            browser_hl = cls._browser_hl
            playwright_hl = cls._playwright_hl
            cls._browser_hl = None
            cls._playwright_hl = None
            cls._cdp_url_hl = None

            if browser_hl is not None:
                try:
                    await browser_hl.close()
                    logger.info("[shared-browser] headless stopped")
                except Exception as e:
                    logger.warning("[shared-browser] headless close failed: %s", e)

            if playwright_hl is not None:
                try:
                    await playwright_hl.stop()
                except Exception as e:
                    logger.warning("[shared-browser] headless playwright stop failed: %s", e)

            # Headed
            browser_hd = cls._browser_hd
            playwright_hd = cls._playwright_hd
            cls._browser_hd = None
            cls._playwright_hd = None
            cls._cdp_url_hd = None

            if browser_hd is not None:
                try:
                    await browser_hd.close()
                    logger.info("[shared-browser] headed stopped")
                except Exception as e:
                    logger.warning("[shared-browser] headed close failed: %s", e)

            if playwright_hd is not None:
                try:
                    await playwright_hd.stop()
                except Exception as e:
                    logger.warning("[shared-browser] headed playwright stop failed: %s", e)

            logger.info("[shared-browser] stopped")
