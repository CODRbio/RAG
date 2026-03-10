"""
Resident context pool for shared CDP browsers.
One active task per context; acquire/release with randomized cooldown.
"""
import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from playwright.async_api import Browser, BrowserContext, Playwright

from src.log import get_logger
from src.retrieval.browser_service import SharedBrowserService, get_cdp_context_options

logger = get_logger(__name__)


@dataclass
class ContextLease:
    """Handle for an acquired context slot. Call release_context(lease) when done."""
    pool_type: str  # "headless" | "headed"
    slot_id: str
    context: BrowserContext
    browser: Browser
    job_id: str
    purpose: str
    _slot: "ContextSlot"
    _pool: "SharedContextPool"


@dataclass
class ContextSlot:
    """One resident context slot."""
    pool_type: str
    slot_id: str
    context: BrowserContext
    browser: Browser
    in_use: bool = False
    last_used_at: float = 0.0
    cooldown_until: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    lease_owner: str = ""
    job_id: str = ""
    downloads_dir: Optional[str] = None
    error_count: int = 0
    reserved_for: Optional[str] = None  # "search" | None


class SharedContextPool:
    """
    Resident context pools for headless and headed CDP browsers.
    Sizes and cooldown come from config (shared_browser).
    """

    _instance: Optional["SharedContextPool"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._headless_slots: List[ContextSlot] = []
        self._headed_slots: List[ContextSlot] = []
        self._headless_general: asyncio.Queue = asyncio.Queue()
        self._headless_search: asyncio.Queue = asyncio.Queue()
        self._headed_available: asyncio.Queue = asyncio.Queue()
        self._pool_lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> "SharedContextPool":
        if cls._instance is None:
            cls._instance = SharedContextPool()
        return cls._instance

    def _get_config(self) -> Any:
        from config.settings import settings
        return getattr(settings, "shared_browser", None)

    async def _ensure_playwright(self) -> None:
        if self._playwright is not None:
            return
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()

    async def _create_headless_slot(
        self, slot_id: str, reserved_for: Optional[str] = None
    ) -> Optional[ContextSlot]:
        cdp_url = SharedBrowserService.get_cdp_url_headless()
        if not cdp_url:
            return None
        await self._ensure_playwright()
        try:
            browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            context = await browser.new_context(**get_cdp_context_options())
            return ContextSlot(
                pool_type="headless",
                slot_id=slot_id,
                context=context,
                browser=browser,
                reserved_for=reserved_for,
            )
        except Exception as e:
            logger.warning("[context-pool] create headless slot %s failed: %s", slot_id, e)
            return None

    async def _create_headed_slot(self, slot_id: str, downloads_dir: Optional[str] = None) -> Optional[ContextSlot]:
        cdp_url = SharedBrowserService.get_cdp_url_headed(slot_id=slot_id)
        if not cdp_url:
            try:
                from config.settings import settings
                sb = getattr(settings, "shared_browser", None)
                base_port = int(getattr(sb, "headed_port", 9223) if sb else 9223)
                await SharedBrowserService.start_headed(port=base_port, slot_id=slot_id)
                cdp_url = SharedBrowserService.get_cdp_url_headed(slot_id=slot_id)
            except Exception as e:
                logger.warning("[context-pool] start headed slot %s failed: %s", slot_id, e)
                cdp_url = None
        if not cdp_url:
            return None
        await self._ensure_playwright()
        try:
            browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            if downloads_dir:
                os.makedirs(downloads_dir, exist_ok=True)
            context = await browser.new_context(**get_cdp_context_options())
            return ContextSlot(
                pool_type="headed",
                slot_id=slot_id,
                context=context,
                browser=browser,
                downloads_dir=downloads_dir,
            )
        except Exception as e:
            logger.warning("[context-pool] create headed slot %s failed: %s", slot_id, e)
            return None

    async def initialize(self) -> None:
        """Create resident contexts; call after SharedBrowserService.start() / start_headed()."""
        async with self._pool_lock:
            current_loop = asyncio.get_running_loop()
            if self._initialized:
                if self._owner_loop is not None and self._owner_loop is not current_loop:
                    logger.warning(
                        "[context-pool] initialize skipped: pool already bound to a different event loop"
                    )
                return
            cfg = self._get_config()
            if cfg is None:
                logger.warning("[context-pool] no shared_browser config, skip init")
                return
            n_headless = getattr(cfg, "headless_context_pool_size", 4)
            n_search_reserved = getattr(cfg, "headless_search_reserved_slots", 1)
            n_general = n_headless - n_search_reserved
            n_headed = getattr(cfg, "headed_context_pool_size", 2)
            for i in range(n_headless):
                slot_id = f"headless-{i}"
                reserved_for = "search" if i >= n_general else None
                slot = await self._create_headless_slot(slot_id, reserved_for=reserved_for)
                if slot:
                    self._headless_slots.append(slot)
                    if slot.reserved_for == "search":
                        await self._headless_search.put(slot)
                    else:
                        await self._headless_general.put(slot)
            base_downloads = "data/raw_papers"
            try:
                from config.settings import settings
                sd = getattr(settings, "scholar_downloader", None)
                if sd is not None:
                    base_downloads = getattr(sd, "download_dir", base_downloads) or base_downloads
            except Exception:
                pass
            for i in range(n_headed):
                slot_id = f"headed-{i}"
                downloads_dir = os.path.join(base_downloads, ".shared_downloads", slot_id)
                slot = await self._create_headed_slot(slot_id, downloads_dir=downloads_dir)
                if slot:
                    self._headed_slots.append(slot)
                    await self._headed_available.put(slot)
            self._initialized = True
            self._owner_loop = current_loop
            logger.info(
                "[context-pool] initialized headless=%d (general=%d, search_reserved=%d) headed=%d",
                len(self._headless_slots),
                n_general,
                n_search_reserved,
                len(self._headed_slots),
            )

    async def shutdown(self) -> None:
        """Close all pooled contexts and stop Playwright."""
        async with self._pool_lock:
            for slot in self._headless_slots + self._headed_slots:
                try:
                    await slot.context.close()
                except Exception as e:
                    logger.debug("[context-pool] close context %s: %s", slot.slot_id, e)
            self._headless_slots.clear()
            self._headed_slots.clear()
            while not self._headless_general.empty():
                try:
                    self._headless_general.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while not self._headless_search.empty():
                try:
                    self._headless_search.get_nowait()
                except asyncio.QueueEmpty:
                    break
            while not self._headed_available.empty():
                try:
                    self._headed_available.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    logger.warning("[context-pool] playwright stop: %s", e)
                self._playwright = None
            self._initialized = False
            self._owner_loop = None
            logger.info("[context-pool] shutdown done")

    def _cooldown_seconds(self) -> float:
        cfg = self._get_config()
        if cfg is None:
            return random.uniform(1.0, 2.0)
        lo = getattr(cfg, "context_cooldown_min_seconds", 1.0)
        hi = getattr(cfg, "context_cooldown_max_seconds", 2.0)
        return random.uniform(lo, hi)

    async def acquire_context(
        self,
        pool_type: str,
        *,
        timeout: Optional[float] = None,
        job_id: str = "",
        purpose: str = "",
        reserved_group: Optional[str] = None,
    ) -> Optional[ContextLease]:
        """
        Acquire a context lease. One task per context; caller must release when done.
        Returns None if pool not initialized or timeout.
        """
        cfg = self._get_config()
        acquire_timeout = timeout
        if acquire_timeout is None and cfg is not None:
            acquire_timeout = getattr(cfg, "context_acquire_timeout_seconds", 30.0)
        acquire_timeout = acquire_timeout or 30.0
        current_loop = asyncio.get_running_loop()
        if self._owner_loop is not None and self._owner_loop is not current_loop:
            logger.warning(
                "[context-pool] acquire %s skipped: pool event-loop mismatch (pool-created on another loop)",
                pool_type,
            )
            return None

        if pool_type == "headless":
            if reserved_group == "search":
                try:
                    slot = self._headless_search.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        slot = await asyncio.wait_for(
                            self._headless_general.get(), timeout=acquire_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[context-pool] acquire %s timeout (%.1fs)", pool_type, acquire_timeout
                        )
                        return None
            else:
                try:
                    slot = await asyncio.wait_for(
                        self._headless_general.get(), timeout=acquire_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[context-pool] acquire %s timeout (%.1fs)", pool_type, acquire_timeout
                    )
                    return None
        else:
            try:
                slot = await asyncio.wait_for(
                    self._headed_available.get(), timeout=acquire_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[context-pool] acquire %s timeout (%.1fs)", pool_type, acquire_timeout
                )
                return None

        now = time.monotonic()
        if slot.cooldown_until > now:
            wait = slot.cooldown_until - now
            await asyncio.sleep(wait)
        slot.in_use = True
        slot.last_used_at = now
        slot.lease_owner = purpose or "unknown"
        slot.job_id = job_id

        lease = ContextLease(
            pool_type=pool_type,
            slot_id=slot.slot_id,
            context=slot.context,
            browser=slot.browser,
            job_id=job_id,
            purpose=purpose,
            _slot=slot,
            _pool=self,
        )
        logger.debug("[context-pool] acquired %s slot=%s job_id=%s", pool_type, slot.slot_id, job_id)
        return lease

    async def release_context(self, lease: ContextLease, *, had_error: bool = False) -> None:
        """Return the context to the pool; apply cooldown then re-enqueue."""
        slot = lease._slot
        pool = lease._pool
        if lease.pool_type == "headless":
            queue = (
                pool._headless_search
                if getattr(slot, "reserved_for", None) == "search"
                else pool._headless_general
            )
        else:
            queue = pool._headed_available

        try:
            pages = slot.context.pages
            for p in pages:
                try:
                    await p.close()
                except Exception as e:
                    logger.debug("[context-pool] close page on release: %s", e)
        except Exception as e:
            logger.debug("[context-pool] list pages on release: %s", e)

        if had_error:
            slot.error_count += 1
            if slot.error_count >= 3:
                logger.warning("[context-pool] slot %s error_count=%d, recreating", slot.slot_id, slot.error_count)
                try:
                    await slot.context.close()
                except Exception:
                    pass
                new_slot: Optional[ContextSlot] = None
                if lease.pool_type == "headless":
                    new_slot = await pool._create_headless_slot(
                        slot.slot_id, reserved_for=getattr(slot, "reserved_for", None)
                    )
                    if new_slot:
                        pool._headless_slots = [s for s in pool._headless_slots if s.slot_id != slot.slot_id]
                        pool._headless_slots.append(new_slot)
                else:
                    new_slot = await pool._create_headed_slot(slot.slot_id, slot.downloads_dir)
                    if new_slot:
                        pool._headed_slots = [s for s in pool._headed_slots if s.slot_id != slot.slot_id]
                        pool._headed_slots.append(new_slot)
                if new_slot:
                    slot = new_slot
                slot.error_count = 0
        else:
            slot.error_count = 0

        slot.in_use = False
        slot.lease_owner = ""
        slot.job_id = ""
        cooldown = pool._cooldown_seconds()
        slot.cooldown_until = time.monotonic() + cooldown
        logger.debug("[context-pool] released %s slot=%s cooldown=%.2fs", lease.pool_type, slot.slot_id, cooldown)
        await asyncio.sleep(cooldown)
        await queue.put(slot)

    async def run_with_context(
        self,
        pool_type: str,
        worker,
        *,
        timeout: Optional[float] = None,
        job_id: str = "",
        purpose: str = "",
        reserved_group: Optional[str] = None,
    ):
        """
        Run worker(context) with a leased context. Acquire and release are always
        on the owner loop; if called from another loop, bridges via run_coroutine_threadsafe.
        """
        if not self._initialized or self._owner_loop is None:
            return None
        current_loop = asyncio.get_running_loop()

        async def _bridged():
            lease = await self.acquire_context(
                pool_type,
                timeout=timeout,
                job_id=job_id,
                purpose=purpose,
                reserved_group=reserved_group,
            )
            if lease is None:
                return None
            had_error = False
            try:
                return await worker(lease.context)
            except Exception:
                had_error = True
                raise
            finally:
                await self.release_context(lease, had_error=had_error)

        if current_loop is self._owner_loop:
            return await _bridged()
        logger.debug(
            "[context-pool] run_with_context bridging %s to owner loop", pool_type
        )
        future = asyncio.run_coroutine_threadsafe(_bridged(), self._owner_loop)
        return await asyncio.wrap_future(future)

    def is_initialized(self) -> bool:
        return self._initialized


async def run_with_headless_context(
    worker,
    *,
    timeout: Optional[float] = None,
    job_id: str = "",
    purpose: str = "",
    reserved_group: Optional[str] = None,
):
    """Run worker(context) with a headless context from the shared pool (bridge-safe)."""
    pool = SharedContextPool.get_instance()
    if not pool.is_initialized():
        return None
    return await pool.run_with_context(
        "headless",
        worker,
        timeout=timeout,
        job_id=job_id,
        purpose=purpose,
        reserved_group=reserved_group,
    )


async def run_with_headed_context(
    worker,
    *,
    timeout: Optional[float] = None,
    job_id: str = "",
    purpose: str = "",
    reserved_group: Optional[str] = None,
):
    """Run worker(context) with a headed context from the shared pool (bridge-safe)."""
    pool = SharedContextPool.get_instance()
    if not pool.is_initialized():
        return None
    return await pool.run_with_context(
        "headed",
        worker,
        timeout=timeout,
        job_id=job_id,
        purpose=purpose,
        reserved_group=reserved_group,
    )


async def acquire_headless_context(
    *,
    timeout: Optional[float] = None,
    job_id: str = "",
    purpose: str = "",
) -> Optional[ContextLease]:
    """Convenience: acquire a headless context lease from the shared pool."""
    pool = SharedContextPool.get_instance()
    if not pool.is_initialized():
        return None
    return await pool.acquire_context("headless", timeout=timeout, job_id=job_id, purpose=purpose)


async def acquire_headed_context(
    *,
    timeout: Optional[float] = None,
    job_id: str = "",
    purpose: str = "",
) -> Optional[ContextLease]:
    """Convenience: acquire a headed context lease from the shared pool."""
    pool = SharedContextPool.get_instance()
    if not pool.is_initialized():
        return None
    return await pool.acquire_context("headed", timeout=timeout, job_id=job_id, purpose=purpose)


async def release_context(lease: Optional[ContextLease], *, had_error: bool = False) -> None:
    """Release a lease; no-op if lease is None."""
    if lease is None:
        return
    await lease._pool.release_context(lease, had_error=had_error)
