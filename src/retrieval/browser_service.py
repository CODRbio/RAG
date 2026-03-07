import asyncio
from typing import Any, Dict, Optional

import aiohttp
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.log import get_logger

logger = get_logger(__name__)


class SharedBrowserService:
    """Process-wide shared Chromium services for CDP consumers: headless + optional headed (parked off-screen)."""

    _lock: asyncio.Lock = asyncio.Lock()

    # Headless instance (default)
    _playwright_hl: Optional[Playwright] = None
    _browser_hl: Optional[Browser] = None
    _cdp_url_hl: Optional[str] = None

    # Headed instance (optional, started minimized)
    _playwright_hd: Optional[Playwright] = None
    _browser_hd: Optional[Browser] = None
    _cdp_url_hd: Optional[str] = None
    _headed_window_mode: str = "parked"
    _control_context_hd: Optional[BrowserContext] = None
    _control_page_hd: Optional[Page] = None

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
        """Start the headed shared Chromium process parked off-screen with a draggable visible strip."""
        async with cls._lock:
            if cls._browser_hd is not None and cls._playwright_hd is not None:
                return

            cls._playwright_hd = await async_playwright().start()
            from config.settings import settings

            sb = getattr(settings, "shared_browser", None)
            width = max(400, int(getattr(sb, "headed_window_width", 1280) if sb else 1280))
            height = max(300, int(getattr(sb, "headed_window_height", 900) if sb else 900))
            visible_strip = max(16, int(getattr(sb, "headed_window_visible_strip_px", 100) if sb else 100))
            pos_y = max(0, int(getattr(sb, "headed_window_y", 80) if sb else 80))
            # Park mostly off the left edge while leaving a small visible strip that can be dragged back.
            pos_x = -(width - visible_strip)
            args = [
                f"--remote-debugging-port={port}",
                f"--window-position={pos_x},{pos_y}",
                f"--window-size={width},{height}",
            ]
            cls._browser_hd = await cls._playwright_hd.chromium.launch(
                headless=False,
                args=args,
            )
            await cls._ensure_headed_control_page()
            cls._cdp_url_hd = f"http://127.0.0.1:{port}"
            cls._headed_window_mode = "parked"
            logger.info(
                "[shared-browser] headed parked at %s (x=%s, y=%s, w=%s, h=%s, visible_strip=%s)",
                cls._cdp_url_hd,
                pos_x,
                pos_y,
                width,
                height,
                visible_strip,
            )

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
    def get_headed_window_state(cls) -> Dict[str, Any]:
        """Return current headed browser window state for UI controls."""
        return {
            "available": cls._browser_hd is not None and cls._playwright_hd is not None and cls._cdp_url_hd is not None,
            "running": cls._browser_hd is not None and cls._playwright_hd is not None,
            "mode": cls._headed_window_mode if cls._browser_hd is not None else "unavailable",
            "cdp_url": cls._cdp_url_hd,
        }

    @classmethod
    def _get_headed_window_bounds(cls, mode: str) -> Dict[str, int]:
        from config.settings import settings

        sb = getattr(settings, "shared_browser", None)
        width = max(400, int(getattr(sb, "headed_window_width", 1280) if sb else 1280))
        height = max(300, int(getattr(sb, "headed_window_height", 900) if sb else 900))
        visible_strip = max(16, int(getattr(sb, "headed_window_visible_strip_px", 100) if sb else 100))
        pos_y = max(0, int(getattr(sb, "headed_window_y", 80) if sb else 80))
        if mode == "visible":
            return {"left": 80, "top": pos_y, "width": width, "height": height}
        return {"left": -(width - visible_strip), "top": pos_y, "width": width, "height": height}

    @classmethod
    async def _ensure_headed_control_page(cls) -> None:
        """Create a persistent blank page so CDP window control always has a target."""
        if cls._browser_hd is None:
            raise RuntimeError("headed shared browser is not running")
        try:
            if cls._control_page_hd is not None and not cls._control_page_hd.is_closed():
                return
        except Exception:
            cls._control_page_hd = None
        try:
            if cls._control_context_hd is None:
                cls._control_context_hd = await cls._browser_hd.new_context()
            page = await cls._control_context_hd.new_page()
            await page.goto("about:blank")
            cls._control_page_hd = page
            logger.debug("[shared-browser] headed control page initialized")
        except Exception as e:
            logger.warning("[shared-browser] init headed control page failed: %s", e)
            raise

    @classmethod
    async def _cdp_command(cls, ws: aiohttp.ClientWebSocketResponse, method: str, params: Optional[Dict[str, Any]] = None, msg_id: int = 1) -> Dict[str, Any]:
        await ws.send_json({"id": msg_id, "method": method, "params": params or {}})
        while True:
            msg = await ws.receive_json()
            if msg.get("id") != msg_id:
                continue
            if msg.get("error"):
                raise RuntimeError(f"{method} failed: {msg['error']}")
            return msg.get("result") or {}

    @classmethod
    async def _set_headed_window_mode(cls, mode: str) -> Dict[str, Any]:
        if cls._browser_hd is None or cls._playwright_hd is None or not cls._cdp_url_hd:
            raise RuntimeError("headed shared browser is not running")
        await cls._ensure_headed_control_page()

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{cls._cdp_url_hd}/json/version") as resp:
                data = await resp.json()
            ws_url = data.get("webSocketDebuggerUrl")
            if not ws_url:
                raise RuntimeError("headed shared browser websocket endpoint unavailable")

            async with session.ws_connect(ws_url) as ws:
                msg_id = 1
                target_id: Optional[str] = None
                try:
                    result = await cls._cdp_command(ws, "Browser.getWindowForTarget", msg_id=msg_id)
                except Exception:
                    msg_id += 1
                    targets = await cls._cdp_command(ws, "Target.getTargets", msg_id=msg_id)
                    page_target = next(
                        (
                            t for t in (targets.get("targetInfos") or [])
                            if t.get("type") == "page" and not str(t.get("url") or "").startswith("devtools://")
                        ),
                        None,
                    )
                    if not page_target:
                        raise RuntimeError("no active headed browser page available to control window")
                    target_id = str(page_target.get("targetId"))
                    msg_id += 1
                    result = await cls._cdp_command(
                        ws,
                        "Browser.getWindowForTarget",
                        {"targetId": target_id},
                        msg_id=msg_id,
                    )

                window_id = result.get("windowId")
                if window_id is None:
                    raise RuntimeError("headed browser window id unavailable")

                bounds = cls._get_headed_window_bounds(mode)
                msg_id += 1
                await cls._cdp_command(
                    ws,
                    "Browser.setWindowBounds",
                    {"windowId": window_id, "bounds": {"windowState": "normal"}},
                    msg_id=msg_id,
                )
                msg_id += 1
                await cls._cdp_command(
                    ws,
                    "Browser.setWindowBounds",
                    {"windowId": window_id, "bounds": bounds},
                    msg_id=msg_id,
                )
                if mode == "visible" and target_id:
                    msg_id += 1
                    try:
                        await cls._cdp_command(ws, "Target.activateTarget", {"targetId": target_id}, msg_id=msg_id)
                    except Exception as e:
                        logger.debug("[shared-browser] activate headed target failed (non-fatal): %s", e)

        cls._headed_window_mode = mode
        state = cls.get_headed_window_state()
        state["bounds"] = cls._get_headed_window_bounds(mode)
        return state

    @classmethod
    async def show_headed(cls) -> Dict[str, Any]:
        """Bring the headed shared browser window back on-screen."""
        return await cls._set_headed_window_mode("visible")

    @classmethod
    async def park_headed(cls) -> Dict[str, Any]:
        """Park the headed shared browser window mostly off-screen."""
        return await cls._set_headed_window_mode("parked")

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
            control_context_hd = cls._control_context_hd
            cls._browser_hd = None
            cls._playwright_hd = None
            cls._cdp_url_hd = None
            cls._headed_window_mode = "parked"
            cls._control_context_hd = None
            cls._control_page_hd = None

            if control_context_hd is not None:
                try:
                    await control_context_hd.close()
                except Exception as e:
                    logger.warning("[shared-browser] headed control context close failed: %s", e)

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
