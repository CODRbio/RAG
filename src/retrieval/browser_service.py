import asyncio
import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from src.log import get_logger

logger = get_logger(__name__)

# Subdir under shared data for headed browser profiles.
HEADED_PROFILE_SUBDIR = "headed_browser_profile"

# Chromium lock file that prevents multiple processes per profile; remove if owning process is dead.
SINGLETON_LOCK = "SingletonLock"


def _remove_stale_singleton_lock(profile_dir: Path) -> bool:
    """Remove Chromium SingletonLock in profile_dir if the owning process is dead.
    Returns True if lock was removed, False otherwise.
    """
    lock_path = profile_dir / SINGLETON_LOCK
    if not os.path.lexists(lock_path):
        return False
    try:
        target = os.readlink(lock_path)
        parts = target.rsplit("-", 1)
        if len(parts) == 2:
            pid = int(parts[1])
            os.kill(pid, 0)  # raises OSError if process is dead
            return False  # process still alive — do not remove
    except (OSError, ValueError, IndexError):
        pass
    try:
        lock_path.unlink()
        logger.info("[shared-browser] removed stale SingletonLock: %s", lock_path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Canonical default options for any context created over the shared CDP.
# Used by context_pool, web_content_fetcher, and browser_manager so all
# CDP consumers see the same locale, timezone, viewport, etc.
# ---------------------------------------------------------------------------
CDP_DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
CDP_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
CDP_DEFAULT_LOCALE = "zh-CN"
CDP_DEFAULT_TIMEZONE_ID = "Asia/Shanghai"
CDP_DEFAULT_GEOLOCATION = {"latitude": 39.9042, "longitude": 116.4074}  # Beijing


def get_cdp_context_options(
    *,
    user_agent: Optional[str] = None,
    viewport: Optional[Dict[str, int]] = None,
    accept_downloads: bool = True,
) -> Dict[str, Any]:
    """Return the canonical context options for new_context() over CDP.

    Callers can override user_agent and viewport; all other options are fixed
    so pool, fetcher, and downloader behave consistently.
    """
    opts: Dict[str, Any] = {
        "accept_downloads": accept_downloads,
        "user_agent": user_agent or CDP_DEFAULT_USER_AGENT,
        "viewport": viewport or CDP_DEFAULT_VIEWPORT,
        "locale": CDP_DEFAULT_LOCALE,
        "timezone_id": CDP_DEFAULT_TIMEZONE_ID,
        "geolocation": CDP_DEFAULT_GEOLOCATION,
        "color_scheme": "light",
        "reduced_motion": "no-preference",
        "has_touch": False,
    }
    return opts


class SharedBrowserService:
    """Process-wide shared Chromium services for CDP consumers: headless + optional headed (parked off-screen)."""

    _lock: asyncio.Lock = asyncio.Lock()

    # Headless instance (default)
    _playwright_hl: Optional[Playwright] = None
    _browser_hl: Optional[Browser] = None
    _cdp_url_hl: Optional[str] = None

    # Headed instances: one persistent profile per headed slot.
    _playwright_hd: Optional[Playwright] = None
    _headed_slots: Dict[str, Dict[str, Any]] = {}
    # Back-compat aliases for callers expecting single-headed attributes.
    _browser_hd: Optional[Browser] = None
    _context_hd: Optional[BrowserContext] = None
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
    def _headed_slot_ids(cls) -> List[str]:
        from config.settings import settings

        sb = getattr(settings, "shared_browser", None)
        n = max(1, int(getattr(sb, "headed_context_pool_size", 1) if sb else 1))
        return [f"headed-{i}" for i in range(n)]

    @classmethod
    def cleanup_headed_profile_locks_on_startup(cls) -> int:
        """Remove stale Chromium SingletonLock files in headed browser profile dirs.
        Call once at application startup before starting headed browser to avoid
        'profile already in use' when a previous process exited without cleanup.
        Returns the number of locks removed.
        """
        try:
            from src.utils.path_manager import PathManager
            root = PathManager.get_shared_dir()
        except Exception:
            root = Path("data").resolve() / "shared"
        profile_parent = root / HEADED_PROFILE_SUBDIR
        removed = 0
        if profile_parent.is_dir():
            for child in profile_parent.iterdir():
                if child.is_dir() and child.name.startswith("headed-") and _remove_stale_singleton_lock(child):
                    removed += 1
        if removed:
            logger.info("[shared-browser] startup cleanup: removed %d stale headed profile lock(s)", removed)
        return removed

    @classmethod
    def is_headed_profile_in_use(cls) -> bool:
        """Return True if any headed browser profile has a SingletonLock held by a live process.

        Used at startup to skip headed browser initialization when a prior worker
        process is still running the same profile (e.g. after a uvicorn worker restart
        while the Chromium child process of the old worker is still alive).
        """
        try:
            from src.utils.path_manager import PathManager
            root = PathManager.get_shared_dir()
        except Exception:
            root = Path("data").resolve() / "shared"
        profile_parent = root / HEADED_PROFILE_SUBDIR
        if not profile_parent.is_dir():
            return False
        for child in profile_parent.iterdir():
            if not (child.is_dir() and child.name.startswith("headed-")):
                continue
            lock_path = child / SINGLETON_LOCK
            if not os.path.lexists(lock_path):
                continue
            try:
                target = os.readlink(lock_path)
                parts = target.rsplit("-", 1)
                if len(parts) == 2:
                    pid = int(parts[1])
                    os.kill(pid, 0)  # raises OSError if process is dead
                    return True  # live process holds this profile
            except (OSError, ValueError, IndexError):
                pass
        return False

    @classmethod
    def _slot_index(cls, slot_id: str) -> int:
        try:
            return max(0, int(str(slot_id).split("-")[-1]))
        except Exception:
            return 0

    @classmethod
    def _default_headed_slot_id(cls) -> Optional[str]:
        if cls._headed_slots:
            return sorted(cls._headed_slots.keys(), key=cls._slot_index)[0]
        ids = cls._headed_slot_ids()
        return ids[0] if ids else None

    @classmethod
    def _get_headed_slot(cls, slot_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sid = slot_id or cls._default_headed_slot_id()
        if not sid:
            return None
        return cls._headed_slots.get(sid)

    @classmethod
    def _sync_headed_aliases(cls) -> None:
        """Keep legacy single-headed attributes aligned to slot headed-0 for compatibility."""
        slot = cls._get_headed_slot(None)
        if slot is None:
            cls._browser_hd = None
            cls._context_hd = None
            cls._cdp_url_hd = None
            cls._headed_window_mode = "parked"
            cls._control_context_hd = None
            cls._control_page_hd = None
            return
        cls._context_hd = slot.get("context")
        cls._cdp_url_hd = slot.get("cdp_url")
        cls._headed_window_mode = slot.get("mode", "parked")
        cls._control_page_hd = slot.get("control_page")
        cls._control_context_hd = cls._context_hd

    # Minimal Preferences skeleton so Chromium does not show "cannot read preference settings".
    _MINIMAL_PREFS: Dict[str, Any] = {
        "profile": {"exit_type": "Normal", "name": "Default"},
        "browser": {"check_default_browser": False},
    }

    @classmethod
    def _ensure_headed_profile_dir(cls, slot_id: Optional[str] = None) -> str:
        """Return deterministic user_data_dir for one headed slot and ensure
        Default/Preferences has a Chromium-compatible structure and
        extensions.ui.developer_mode=true.
        """
        try:
            from src.utils.path_manager import PathManager
            root = PathManager.get_shared_dir()
        except Exception:
            root = Path("data").resolve() / "shared"
        sid = slot_id or "headed-0"
        profile_dir = root / HEADED_PROFILE_SUBDIR / sid
        profile_dir.mkdir(parents=True, exist_ok=True)
        default_dir = profile_dir / "Default"
        default_dir.mkdir(parents=True, exist_ok=True)
        prefs_path = default_dir / "Preferences"
        prefs: Dict[str, Any] = {}
        if prefs_path.exists():
            try:
                with open(prefs_path, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
            except Exception as e:
                logger.debug("[shared-browser] headed profile prefs read failed (using defaults): %s", e)
        # Ensure minimal Chromium-expected keys to avoid "系统无法读取您的偏好设置"
        for key, value in cls._MINIMAL_PREFS.items():
            if key not in prefs:
                prefs[key] = dict(value) if isinstance(value, dict) else value
            elif isinstance(value, dict) and isinstance(prefs.get(key), dict):
                for k, v in value.items():
                    if k not in prefs[key]:
                        prefs[key][k] = v
        if "extensions" not in prefs:
            prefs["extensions"] = {}
        if "ui" not in prefs["extensions"]:
            prefs["extensions"]["ui"] = {}
        prefs["extensions"]["ui"]["developer_mode"] = True
        try:
            with open(prefs_path, "w", encoding="utf-8") as f:
                json.dump(prefs, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.warning("[shared-browser] headed profile prefs write failed: %s", e)
        return str(profile_dir.resolve())

    @classmethod
    def _resolve_extension_args(cls, extension_path: Optional[str]) -> tuple:
        """
        Validate a CapSolver-style extension directory and return the extra launch
        args and ignore_default_args needed to load it in Playwright Chromium.

        Returns (extra_args: List[str], ignore_default_args: List[str]).
        If the directory is absent or missing manifest.json an empty pair is returned.
        """
        if not extension_path:
            return [], []
        resolved = os.path.abspath(extension_path)
        if not os.path.isdir(resolved):
            logger.warning("[shared-browser] extension dir not found: %s – skipping", resolved)
            return [], []
        if not os.path.isfile(os.path.join(resolved, "manifest.json")):
            logger.warning("[shared-browser] extension missing manifest.json: %s – skipping", resolved)
            return [], []
        logger.info("[shared-browser] loading extension: %s", resolved)
        return (
            [
                f"--disable-extensions-except={resolved}",
                f"--load-extension={resolved}",
            ],
            ["--disable-extensions"],
        )

    @classmethod
    async def start_headed(
        cls,
        port: int = 9223,
        extension_path: Optional[str] = None,
        slot_id: Optional[str] = None,
    ) -> None:
        """Start headed persistent slot(s).

        If slot_id is provided, start only that slot. Otherwise start all configured headed slots.
        """
        async with cls._lock:
            if cls._playwright_hd is None:
                cls._playwright_hd = await async_playwright().start()
            from config.settings import settings

            sb = getattr(settings, "shared_browser", None)
            width = max(400, int(getattr(sb, "headed_window_width", 1280) if sb else 1280))
            height = max(300, int(getattr(sb, "headed_window_height", 900) if sb else 900))
            visible_strip = max(16, int(getattr(sb, "headed_window_visible_strip_px", 100) if sb else 100))
            pos_y = max(0, int(getattr(sb, "headed_window_y", 80) if sb else 80))
            ext_path = extension_path or getattr(settings, "capsolver_extension_path", None)
            ext_args, ignore_args = cls._resolve_extension_args(ext_path)
            target_slots = [slot_id] if slot_id else cls._headed_slot_ids()

            for sid in target_slots:
                if sid in cls._headed_slots and cls._headed_slots[sid].get("context") is not None:
                    continue
                idx = cls._slot_index(sid)
                slot_port = int(port) + idx
                user_data_dir = cls._ensure_headed_profile_dir(sid)
                pos_x = -(width - visible_strip)
                args: List[str] = [
                    f"--remote-debugging-port={slot_port}",
                    f"--window-position={pos_x},{pos_y}",
                    f"--window-size={width},{height}",
                ]
                args.extend(ext_args)
                launch_kwargs: Dict[str, Any] = {"headless": False, "args": args}
                if ignore_args:
                    launch_kwargs["ignore_default_args"] = ignore_args

                context = await cls._playwright_hd.chromium.launch_persistent_context(
                    user_data_dir,
                    **launch_kwargs,
                )
                cdp_url = f"http://127.0.0.1:{slot_port}"
                cls._headed_slots[sid] = {
                    "slot_id": sid,
                    "context": context,
                    "cdp_url": cdp_url,
                    "mode": "parked",
                    "control_page": None,
                    "window_bounds": {"x": pos_x, "y": pos_y, "w": width, "h": height},
                }
                await cls._ensure_headed_control_page(sid)
                logger.info(
                    "[shared-browser] headed slot=%s parked at %s (x=%s, y=%s, w=%s, h=%s, visible_strip=%s, extension=%s)",
                    sid,
                    cdp_url,
                    pos_x,
                    pos_y,
                    width,
                    height,
                    visible_strip,
                    ext_path or "none",
                )
            cls._sync_headed_aliases()

    @classmethod
    def get_cdp_url(cls) -> Optional[str]:
        """Backward compatibility: return headless CDP endpoint."""
        return cls._cdp_url_hl

    @classmethod
    def get_cdp_url_headless(cls) -> Optional[str]:
        """Return headless CDP endpoint when ready."""
        return cls._cdp_url_hl

    @classmethod
    def get_cdp_url_headed(cls, slot_id: Optional[str] = None) -> Optional[str]:
        """Return headed CDP endpoint for one slot, defaulting to headed-0."""
        slot = cls._get_headed_slot(slot_id)
        if slot is None:
            return None
        return str(slot.get("cdp_url") or "")

    @classmethod
    def get_headed_window_state(cls, slot_id: Optional[str] = None) -> Dict[str, Any]:
        """Return current headed browser window state for one slot."""
        slot = cls._get_headed_slot(slot_id)
        headed_ok = bool(slot is not None and slot.get("context") is not None and cls._playwright_hd is not None and slot.get("cdp_url"))
        return {
            "available": headed_ok,
            "running": headed_ok,
            "mode": (slot.get("mode", "parked") if slot else "unavailable"),
            "cdp_url": (slot.get("cdp_url") if slot else None),
            "slot_id": (slot.get("slot_id") if slot else None),
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
    async def _ensure_headed_control_page(cls, slot_id: Optional[str] = None) -> None:
        """Create a persistent blank page so CDP window control always has a target."""
        slot = cls._get_headed_slot(slot_id)
        if slot is None or slot.get("context") is None:
            raise RuntimeError("headed shared browser is not running")
        try:
            control_page = slot.get("control_page")
            if control_page is not None and not control_page.is_closed():
                return
        except Exception:
            slot["control_page"] = None
        try:
            page = await slot["context"].new_page()
            await page.goto("about:blank")
            slot["control_page"] = page
            logger.debug("[shared-browser] headed control page initialized: slot=%s", slot.get("slot_id"))
            cls._sync_headed_aliases()
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
    async def _set_headed_window_mode_macos_all_windows(cls, mode: str) -> None:
        """Best-effort macOS fallback: move all Google Chrome for Testing windows together."""
        if platform.system() != "Darwin":
            return

        bounds = cls._get_headed_window_bounds(mode)
        left = int(bounds["left"])
        top = int(bounds["top"])
        right = left + int(bounds["width"])
        bottom = top + int(bounds["height"])
        app_name = "Google Chrome for Testing"
        script_lines = [
            'tell application "System Events"',
            f'  if exists process "{app_name}" then',
            f'    tell process "{app_name}"',
            '      repeat with w in windows',
            f'        set position of w to {{{left}, {top}}}',
            f'        set size of w to {{{int(bounds["width"])}, {int(bounds["height"])}}}',
            '      end repeat',
            '    end tell',
            '  end if',
            'end tell',
        ]
        if mode == "visible":
            script_lines.extend(
                [
                    f'tell application "{app_name}" to activate',
                    'tell application "System Events"',
                    f'  if exists process "{app_name}" then',
                    f'    set frontmost of process "{app_name}" to true',
                    '  end if',
                    'end tell',
                ]
            )
        script = "\n".join(script_lines)
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            raise RuntimeError(err or f"osascript exit={proc.returncode}")
        logger.info(
            "[shared-browser] macOS window fallback applied mode=%s bounds=(%d,%d,%d,%d)",
            mode,
            left,
            top,
            right,
            bottom,
        )

    @classmethod
    async def _set_headed_window_mode(cls, mode: str, slot_id: Optional[str] = None) -> Dict[str, Any]:
        slot = cls._get_headed_slot(slot_id)
        cdp_url = str((slot or {}).get("cdp_url") or "")
        if slot is None or cls._playwright_hd is None or not cdp_url:
            raise RuntimeError("headed shared browser is not running")

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{cdp_url}/json/version") as version_resp:
                data = await version_resp.json()
            ws_url = data.get("webSocketDebuggerUrl")
            if not ws_url:
                raise RuntimeError("headed shared browser websocket endpoint unavailable")

            async with session.ws_connect(ws_url) as ws:
                msg_id = 1
                # Must enable discovery so getTargets returns targets from ALL browser contexts
                # (default + those created by context pool). GET /json often only lists default context.
                try:
                    await cls._cdp_command(ws, "Target.setDiscoverTargets", {"discover": True}, msg_id=msg_id)
                    msg_id += 1
                except Exception as e:
                    logger.debug("[shared-browser] setDiscoverTargets failed (non-fatal): %s", e)

                # Primary source: CDP Target.getTargets() — includes non-default contexts (all windows).
                targets_result = await cls._cdp_command(ws, "Target.getTargets", msg_id=msg_id)
                msg_id += 1
                target_infos = targets_result.get("targetInfos") or []
                page_target_ids = set()
                for t in target_infos:
                    if t.get("type") != "page":
                        continue
                    if str(t.get("url") or "").startswith("devtools://"):
                        continue
                    tid = t.get("targetId")
                    if tid:
                        page_target_ids.add(str(tid))

                # Fallback: also add any page IDs from GET /json (default-context pages).
                try:
                    async with session.get(f"{cdp_url}/json") as resp:
                        if resp.status == 200:
                            json_targets = await resp.json()
                            for t in json_targets or []:
                                if not isinstance(t, dict) or str(t.get("type", "")).lower() != "page":
                                    continue
                                if str(t.get("url") or "").startswith("devtools://"):
                                    continue
                                tid = t.get("id") or t.get("targetId")
                                if tid is not None:
                                    page_target_ids.add(str(tid))
                except Exception as e:
                    logger.debug("[shared-browser] /json fallback failed: %s", e)

                if not page_target_ids:
                    raise RuntimeError("no active headed browser page available to control window")

                # Resolve windowId for each page target (same window may host multiple tabs).
                seen_window_ids: set = set()
                window_ids: List[int] = []
                for target_id in page_target_ids:
                    msg_id += 1
                    try:
                        result = await cls._cdp_command(
                            ws,
                            "Browser.getWindowForTarget",
                            {"targetId": target_id},
                            msg_id=msg_id,
                        )
                    except Exception as e:
                        logger.debug("[shared-browser] getWindowForTarget %s failed: %s", target_id, e)
                        continue
                    wid = result.get("windowId")
                    if wid is None:
                        continue
                    wid_int = int(wid) if not isinstance(wid, int) else wid
                    if wid_int not in seen_window_ids:
                        seen_window_ids.add(wid_int)
                        window_ids.append(wid_int)

                if not window_ids:
                    raise RuntimeError("headed browser window id unavailable")

                logger.info(
                    "[shared-browser] headed pages=%d -> %d window(s), setting mode=%s",
                    len(page_target_ids),
                    len(window_ids),
                    mode,
                )

                bounds = cls._get_headed_window_bounds(mode)
                for window_id in window_ids:
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

                # Optionally activate the first target when bringing window on-screen.
                if mode == "visible" and page_target_ids:
                    first_id = next(iter(page_target_ids), None)
                    if first_id:
                        msg_id += 1
                        try:
                            await cls._cdp_command(
                                ws, "Target.activateTarget", {"targetId": first_id}, msg_id=msg_id
                            )
                        except Exception as e:
                            logger.debug("[shared-browser] activate headed target failed (non-fatal): %s", e)

        if len(cls._headed_slots) <= 1:
            try:
                await cls._set_headed_window_mode_macos_all_windows(mode)
            except Exception as e:
                logger.debug("[shared-browser] macOS all-window fallback failed (non-fatal): %s", e)

        slot["mode"] = mode
        cls._sync_headed_aliases()
        state = cls.get_headed_window_state(slot_id=slot.get("slot_id"))
        state["bounds"] = cls._get_headed_window_bounds(mode)
        return state

    @classmethod
    async def show_headed(cls, slot_id: Optional[str] = None) -> Dict[str, Any]:
        """Bring the headed shared browser window back on-screen."""
        return await cls._set_headed_window_mode("visible", slot_id=slot_id)

    @classmethod
    async def park_headed(cls, slot_id: Optional[str] = None) -> Dict[str, Any]:
        """Park the headed shared browser window mostly off-screen."""
        return await cls._set_headed_window_mode("parked", slot_id=slot_id)

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

            # Headed slots (persistent contexts: closing each stops the browser process)
            headed_slots = list(cls._headed_slots.items())
            playwright_hd = cls._playwright_hd
            cls._headed_slots = {}
            cls._browser_hd = None
            cls._context_hd = None
            cls._playwright_hd = None
            cls._cdp_url_hd = None
            cls._headed_window_mode = "parked"
            cls._control_context_hd = None
            cls._control_page_hd = None

            for sid, slot in headed_slots:
                context_hd = slot.get("context")
                if context_hd is None:
                    continue
                try:
                    await context_hd.close()
                    logger.info("[shared-browser] headed slot %s stopped", sid)
                except Exception as e:
                    logger.warning("[shared-browser] headed slot %s close failed: %s", sid, e)

            if playwright_hd is not None:
                try:
                    await playwright_hd.stop()
                except Exception as e:
                    logger.warning("[shared-browser] headed playwright stop failed: %s", e)

            logger.info("[shared-browser] stopped")
