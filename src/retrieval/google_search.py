"""
Google Scholar / Google 搜索模块

基于 Playwright 的浏览器自动化搜索，输出与 hybrid_retriever 兼容的 RAG 格式。
支持 capsolver 扩展自动处理验证码。

使用方法:
---------
from src.retrieval.google_search import google_searcher

# 异步搜索 Scholar
results = await google_searcher.search_scholar("deep learning", limit=5)

# 异步搜索 Google
results = await google_searcher.search_google("machine learning tutorial", limit=5)

输出格式:
--------
[
    {
        "content": "摘要/片段",
        "score": 0.8,
        "metadata": {
            "source": "scholar" | "google",
            "doc_id": "标题",
            "title": "标题",
            "url": "https://...",
            "domain": "scholar.google.com",
            "search_query": "检索词",
            ...
        }
    }
]
"""

import asyncio
import json
import os
import platform
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup

from config.settings import settings
from src.log import get_logger
from src.retrieval.browser_service import SharedBrowserService
from src.retrieval.capsolver_api import CapSolverAPI
from src.retrieval.two_captcha_api import TwoCaptchaAPI
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)

# 浏览器复用：全局单例与空闲超时
_shared_browser_manager: Optional["_BrowserManager"] = None
_shared_browser_last_used: float = 0.0
_shared_browser_lock = asyncio.Lock()
# 引用计数：有活跃使用者时不关闭
_browser_ref_count: int = 0
_browser_active_jobs: set = set()  # job_id / chat session 标识
_delayed_cleanup_task: Optional[asyncio.Task] = None
# Cross-job guard for Chromium profile operations.
_browser_profile_lock = threading.Lock()
# Scholar Playwright operations share one Chromium instance (page navigation, keyboard
# focus, pagination all conflict).  Serialize them completely so only one runs at a time.
# Waiters block here without consuming the operation's own timeout budget.
_playwright_scholar_lock = asyncio.Lock()


class _ProfileLockGuard:
    async def __aenter__(self):
        await asyncio.to_thread(_browser_profile_lock.acquire)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        _browser_profile_lock.release()
        return False


_profile_lock_guard = _ProfileLockGuard()


def _cleanup_stale_singleton_lock(user_data_dir: str) -> None:
    """Remove a stale Chromium ``SingletonLock`` if the owning process is dead.

    Chromium creates a symlink ``SingletonLock`` whose target is
    ``<hostname>-<pid>``.  If the process crashed without cleanup, the
    lock file prevents any new instance from using the same profile.
    """
    lock_path = os.path.join(user_data_dir, "SingletonLock")
    if not os.path.lexists(lock_path):
        return
    try:
        target = os.readlink(lock_path)
        parts = target.rsplit("-", 1)
        if len(parts) == 2:
            pid = int(parts[1])
            os.kill(pid, 0)  # raises OSError if process is dead
            return  # process still alive — do not remove
    except (OSError, ValueError, IndexError):
        pass
    try:
        os.unlink(lock_path)
        logger.info("Removed stale Chromium SingletonLock: %s", lock_path)
    except OSError:
        pass


def _singleton_lock_held_by_live_process(user_data_dir: str) -> bool:
    """True if SingletonLock exists and the owning process is still alive."""
    lock_path = os.path.join(user_data_dir, "SingletonLock")
    if not os.path.lexists(lock_path):
        return False
    try:
        target = os.readlink(lock_path)
        parts = target.rsplit("-", 1)
        if len(parts) == 2:
            pid = int(parts[1])
            os.kill(pid, 0)  # raises OSError if process is dead
            return True  # process alive
    except (OSError, ValueError, IndexError):
        pass
    return False


async def _do_close_shared_browser() -> None:
    """Actually close the shared browser and reset global state. Caller must hold _shared_browser_lock."""
    global _shared_browser_manager, _shared_browser_last_used, _browser_ref_count, _browser_active_jobs
    if _shared_browser_manager is None:
        return
    try:
        await _shared_browser_manager.close()
        logger.info("共享浏览器实例已关闭")
    except Exception as e:
        logger.warning("关闭共享浏览器时出错: %s", e)
    finally:
        _shared_browser_manager = None
        _shared_browser_last_used = 0
        _browser_ref_count = 0
        _browser_active_jobs = set()


async def _close_shared_browser_if_exists() -> None:
    """Close the global shared browser (if any) to free the profile directory."""
    async with _profile_lock_guard:
        async with _shared_browser_lock:
            global _shared_browser_manager, _shared_browser_last_used
            if _shared_browser_manager is not None:
                try:
                    await _shared_browser_manager.close()
                except Exception:
                    pass
                _shared_browser_manager = None
                _shared_browser_last_used = 0


async def _get_or_create_shared_browser_manager(
    timeout: int,
    user_data_dir: str,
    headless: Optional[bool],
    proxy: Optional[str],
    extension_path: Optional[str],
) -> Tuple[Optional["_BrowserManager"], Any]:
    """当 performance.google_search.browser_reuse 为 True 时返回复用的 (manager, context)。"""
    perf = getattr(settings, "perf_google_search", None)
    if not perf or not getattr(perf, "browser_reuse", True):
        return None, None
    
    max_idle = getattr(perf, "max_idle_seconds", 300) or 300
    async with _profile_lock_guard:
        async with _shared_browser_lock:
            global _shared_browser_manager, _shared_browser_last_used
            now = time.monotonic()
            if _shared_browser_manager is not None:
                if (now - _shared_browser_last_used) > max_idle:
                    try:
                        await _shared_browser_manager.close()
                    except Exception:
                        pass
                    _shared_browser_manager = None
                else:
                    _shared_browser_last_used = now
                    if _shared_browser_manager.browsers:
                        return _shared_browser_manager, _shared_browser_manager.browsers[0]
                    try:
                        await _shared_browser_manager.close()
                    except Exception:
                        pass
                    _shared_browser_manager = None
            manager = _BrowserManager(timeout=timeout)
            context = await manager.launch_persistent_browser(
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                stealth_mode=True,
                extension_path=extension_path,
            )
            _shared_browser_manager = manager
            _shared_browser_last_used = time.monotonic()
            return manager, context


async def acquire_shared_browser(
    job_id: str = "",
    timeout: int = 120000,
    user_data_dir: str = "",
    headless: Optional[bool] = None,
    proxy: Optional[str] = None,
    extension_path: Optional[str] = None,
) -> Tuple["_BrowserManager", Any, bool]:
    """
    获取共享浏览器，引用计数 +1（仅当 is_shared=True 时）。
    - browser_reuse 且已有共享实例：复用，ref_count++，返回 (manager, context, True)。
    - browser_reuse 且无共享实例：创建并设为共享，ref_count=1，返回 (manager, context, True)。
    - 非 browser_reuse：创建独立实例，返回 (manager, context, False)，调用方在 finally 中关闭 manager。
    """
    perf = getattr(settings, "perf_google_search", None)
    reuse = bool(perf and getattr(perf, "browser_reuse", True))
    jid = (job_id or "").strip() or f"_anon_{id(object())}"

    if not reuse:
        manager = _BrowserManager(timeout=timeout)
        async with _profile_lock_guard:
            context = await manager.launch_persistent_browser(
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                stealth_mode=True,
                extension_path=extension_path,
            )
        return manager, context, False

    async with _profile_lock_guard:
        async with _shared_browser_lock:
            global _shared_browser_manager, _shared_browser_last_used
            global _browser_ref_count, _browser_active_jobs, _delayed_cleanup_task
            now = time.monotonic()
            if _shared_browser_manager is not None and _shared_browser_manager.browsers:
                _browser_ref_count += 1
                _browser_active_jobs.add(jid)
                _shared_browser_last_used = now
                if _delayed_cleanup_task and not _delayed_cleanup_task.done():
                    _delayed_cleanup_task.cancel()
                    _delayed_cleanup_task = None
                return _shared_browser_manager, _shared_browser_manager.browsers[0], True
            manager = _BrowserManager(timeout=timeout)
            try:
                context = await manager.launch_persistent_browser(
                    user_data_dir=user_data_dir,
                    headless=headless,
                    proxy=proxy,
                    stealth_mode=True,
                    extension_path=extension_path,
                )
            except Exception as launch_err:
                cdp_port = getattr(perf, "cdp_port", None) if perf else None
                if (
                    cdp_port
                    and _singleton_lock_held_by_live_process(user_data_dir)
                ):
                    try:
                        from playwright.async_api import async_playwright
                        playwright = await async_playwright().start()
                        browser = await playwright.chromium.connect_over_cdp(
                            f"http://127.0.0.1:{cdp_port}"
                        )
                        ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
                        manager.playwright = playwright
                        manager.attach_cdp_browser(browser, ctx)
                        context = ctx
                        logger.info("launch 失败，已通过 CDP 连接已有浏览器 (port=%s)", cdp_port)
                    except Exception as cdp_err:
                        logger.warning("CDP 回退失败: %s", cdp_err)
                        raise launch_err from cdp_err
                else:
                    raise
            _shared_browser_manager = manager
            _shared_browser_last_used = now
            _browser_ref_count = 1
            _browser_active_jobs = {jid}
            if _delayed_cleanup_task and not _delayed_cleanup_task.done():
                _delayed_cleanup_task.cancel()
                _delayed_cleanup_task = None
            return manager, context, True


async def release_shared_browser(
    job_id: str = "",
    is_shared: bool = True,
    manager: Optional["_BrowserManager"] = None,
) -> None:
    """
    释放浏览器引用。is_shared=False 且 manager 非空时直接关闭 manager；
    否则 ref_count--，若归零则调度延迟关闭（max_idle_seconds 后再真正关闭）。
    """
    if not is_shared and manager is not None:
        try:
            async with _profile_lock_guard:
                await manager.close()
        except Exception as e:
            logger.warning("关闭独立浏览器时出错: %s", e)
        return

    jid = (job_id or "").strip()
    perf = getattr(settings, "perf_google_search", None)
    max_idle = (getattr(perf, "max_idle_seconds", None) or 300) if perf else 300

    async with _profile_lock_guard:
        async with _shared_browser_lock:
            global _browser_ref_count, _browser_active_jobs, _shared_browser_last_used
            global _delayed_cleanup_task
            _browser_ref_count = max(0, _browser_ref_count - 1)
            if jid:
                _browser_active_jobs.discard(jid)
            _shared_browser_last_used = time.monotonic()

            if _browser_ref_count <= 0:

                async def _delayed_close() -> None:
                    await asyncio.sleep(max_idle)
                    async with _profile_lock_guard:
                        async with _shared_browser_lock:
                            global _shared_browser_manager, _delayed_cleanup_task
                            if _browser_ref_count <= 0 and _shared_browser_manager is not None:
                                await _do_close_shared_browser()
                            _delayed_cleanup_task = None

                _delayed_cleanup_task = asyncio.create_task(_delayed_close())
                logger.debug("已调度共享浏览器延迟关闭 (idle %ds)", max_idle)


# ============================================================
# 配置读取
# ============================================================
def _get_google_search_config() -> Dict[str, Any]:
    """从 config 或 settings 读取 google_search 配置"""
    try:
        from config.settings import settings
        gs = getattr(settings, "google_search", None)
        if gs is not None:
            return {
                "enabled": getattr(gs, "enabled", True),
                "scholar_enabled": getattr(gs, "scholar_enabled", True),
                "google_enabled": getattr(gs, "google_enabled", False),
                "extension_path": getattr(settings, "capsolver_extension_path", "extra_tools/CapSolverExtension"),
                "headless": getattr(gs, "headless", None),
                "proxy": getattr(gs, "proxy", None),
                "timeout": getattr(gs, "timeout", 60000),
                "max_results": getattr(gs, "max_results", 5),
                "user_data_dir": getattr(gs, "user_data_dir", None),
                "headed_browser_port": getattr(gs, "headed_browser_port", 9223),
                "start_headed_browser": getattr(gs, "start_headed_browser", False),
            }
    except Exception:
        pass
    
    config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    gs = raw.get("google_search") or {}
    return {
        "enabled": gs.get("enabled", True),
        "scholar_enabled": gs.get("scholar_enabled", True),
        "google_enabled": gs.get("google_enabled", False),
        "extension_path": raw.get("capsolver_extension_path", gs.get("extension_path", "extra_tools/CapSolverExtension")),
        "headless": gs.get("headless"),
        "proxy": gs.get("proxy"),
        "timeout": gs.get("timeout", 60000),
        "max_results": min(int(gs.get("max_results", 5)), 20),
        "user_data_dir": gs.get("user_data_dir"),
        "headed_browser_port": int(gs.get("headed_browser_port", 9223)),
        "start_headed_browser": bool(gs.get("start_headed_browser", False)),
    }


# ============================================================
# 显示管理（虚拟显示器支持 - Linux 服务器）
# ============================================================
_virtual_display = None
_display_mode = None


def _is_display_available() -> bool:
    """检测是否有可用的显示器"""
    if platform.system() in ("Darwin", "Windows"):
        return True
    
    display = os.environ.get("DISPLAY")
    if display:
        try:
            import subprocess
            result = subprocess.run(["xdpyinfo"], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return True
    return False


def _start_virtual_display(width: int = 1920, height: int = 1080) -> Optional[object]:
    """启动虚拟显示器（仅 Linux）"""
    global _virtual_display
    
    if _virtual_display is not None:
        return _virtual_display
    
    if platform.system() != "Linux":
        return None
    
    try:
        from pyvirtualdisplay import Display
        logger.info(f"启动虚拟显示器 ({width}x{height})...")
        _virtual_display = Display(visible=False, size=(width, height), backend="xvfb")
        _virtual_display.start()
        logger.info(f"虚拟显示器已启动: DISPLAY={os.environ.get('DISPLAY')}")
        return _virtual_display
    except ImportError:
        logger.warning("pyvirtualdisplay 未安装，无法启动虚拟显示器")
        return None
    except Exception as e:
        logger.error(f"启动虚拟显示器失败: {e}")
        return None


def _stop_virtual_display():
    """停止虚拟显示器"""
    global _virtual_display
    if _virtual_display is not None:
        try:
            _virtual_display.stop()
        except Exception:
            pass
        _virtual_display = None


def _ensure_display(force_mode: Optional[str] = None) -> Tuple[bool, str]:
    """确保有可用的显示器"""
    global _display_mode
    
    mode = force_mode or os.environ.get("DISPLAY_MODE", "auto").lower()
    
    if mode == "headless":
        _display_mode = "headless"
        return False, "headless"
    
    if platform.system() in ("Darwin", "Windows"):
        _display_mode = "real"
        return True, "real"
    
    # Linux
    if _is_display_available():
        _display_mode = "real"
        return True, "real"
    else:
        display = _start_virtual_display()
        if display:
            _display_mode = "virtual"
            return True, "virtual"
        else:
            _display_mode = "headless"
            return False, "headless"


def _get_display_mode() -> str:
    """获取当前显示模式"""
    return _display_mode or "unknown"


# ============================================================
# Stealth 模式支持
# ============================================================
_STEALTH_AVAILABLE = "none"
_Stealth = None

try:
    from playwright_stealth import Stealth as _StealthClass
    _Stealth = _StealthClass
    _STEALTH_AVAILABLE = "v2"
except Exception:
    try:
        from playwright_stealth import stealth_async
        _STEALTH_AVAILABLE = "async"
    except Exception:
        _STEALTH_AVAILABLE = "none"


# ============================================================
# 随机化配置
# ============================================================
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]


def _get_random_user_agent() -> str:
    return random.choice(_USER_AGENTS)


def _get_random_viewport() -> dict:
    return {
        "width": 1280 + random.randint(-50, 50),
        "height": 720 + random.randint(-50, 50)
    }


# ============================================================
# 人类行为模拟
# ============================================================
async def _simulate_human_behavior(page):
    """模拟人类用户行为"""
    try:
        await page.evaluate("""
            () => {
                const scrollAmount = Math.floor(Math.random() * window.innerHeight * 0.8);
                window.scrollBy(0, scrollAmount);
                setTimeout(() => {
                    window.scrollBy(0, -scrollAmount * 0.7);
                }, 500 + Math.random() * 1000);
            }
        """)
        
        viewport = await page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
        if viewport:
            for _ in range(3):
                x = random.randint(0, viewport['width'] - 100)
                y = random.randint(0, viewport['height'] - 100)
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.1, 0.3))
        
        await asyncio.sleep(random.uniform(0.5, 2.0))
    except Exception as e:
        logger.debug(f"模拟人类行为时出错: {e}")


async def _simulate_human_scroll_to_bottom(page):
    """模拟滚动到页面底部"""
    try:
        await page.evaluate("""
            () => {
                const scrollHeight = document.body.scrollHeight;
                const duration = 1000 + Math.random() * 1500;
                const startTime = performance.now();
                const startPos = window.pageYOffset;
                
                function scrollStep(timestamp) {
                    const elapsed = timestamp - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const ease = progress < 0.5 
                        ? 2 * progress * progress 
                        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                    window.scrollTo(0, startPos + (scrollHeight - startPos) * ease);
                    if (progress < 1) window.requestAnimationFrame(scrollStep);
                }
                window.requestAnimationFrame(scrollStep);
            }
        """)
        await asyncio.sleep(random.uniform(0.5, 1.5))
    except Exception as e:
        logger.debug(f"滚动到底部时出错: {e}")


# ============================================================
# 浏览器管理器
# ============================================================
class _BrowserManager:
    """浏览器管理器，负责创建和管理浏览器实例（含 CDP 连接）"""
    
    def __init__(self, timeout: int = 120000):
        self.playwright = None
        self.browsers = []
        self.browser_pids = []  # 记录 chromium 进程 PID
        self._cdp_browser = None  # 通过 connect_over_cdp 连接时持有，close 时仅断开不杀进程
        self.timeout = timeout
        self._playwright_lock = asyncio.Lock()

    def attach_cdp_browser(self, browser: Any, context: Any) -> None:
        """附加通过 connect_over_cdp 得到的 browser 与默认 context。"""
        self._cdp_browser = browser
        self.browsers = [context]

    def is_cdp_connected(self) -> bool:
        """是否通过 CDP 连接共享浏览器（无本地窗口，不应按有头模式等待验证码）。"""
        return self._cdp_browser is not None
    
    async def close(self):
        """关闭所有浏览器实例：先关所有页面，再关 context，最后 stop playwright（CDP 时仅断开连接）"""
        import signal
        
        # CDP 连接：只断开，不杀进程
        if self._cdp_browser is not None:
            try:
                await asyncio.wait_for(self._cdp_browser.close(), timeout=10.0)
                logger.debug("CDP 浏览器连接已断开")
            except Exception as e:
                logger.warning("断开 CDP 浏览器时出错: %s", e)
            self._cdp_browser = None
            self.browsers = []
            if self.playwright:
                try:
                    await asyncio.wait_for(self.playwright.stop(), timeout=10.0)
                except Exception:
                    pass
                self.playwright = None
            return

        # 保存 PID 列表副本（关闭时使用）
        pids_to_check = list(self.browser_pids)
        
        # 1. 关闭所有页面
        for ctx in self.browsers:
            try:
                pages = getattr(ctx, "pages", []) or []
                for p in pages:
                    try:
                        if not p.is_closed():
                            await asyncio.wait_for(p.close(), timeout=5.0)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # 2. 关闭 context
        for ctx in self.browsers:
            try:
                await asyncio.wait_for(ctx.close(), timeout=10.0)
                logger.debug("BrowserContext closed successfully")
            except Exception as e:
                logger.warning(f"关闭 BrowserContext 时出错: {e}")
        self.browsers = []
        await asyncio.sleep(0.5)
        
        # 3. 停止 playwright（这会清理底层进程）
        if self.playwright:
            try:
                await asyncio.wait_for(self.playwright.stop(), timeout=10.0)
                logger.debug("Playwright stopped successfully")
            except Exception as e:
                logger.warning(f"停止 Playwright 时出错: {e}")
            self.playwright = None
        
        # 4. 最后手段：如果还有残留进程，强制 kill
        await asyncio.sleep(0.5)
        for pid in pids_to_check:
            try:
                os.kill(pid, 0)  # 检查进程是否还存在
                logger.warning(f"检测到残留 chromium 进程 {pid}，尝试强制终止")
                os.kill(pid, signal.SIGTERM)
                await asyncio.sleep(0.5)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"已强制终止 chromium 进程 {pid}")
                except OSError:
                    logger.info(f"chromium 进程 {pid} 已通过 SIGTERM 退出")
            except OSError:
                pass  # 进程已不存在，正常
        
        self.browser_pids = []
    
    async def _ensure_playwright(self):
        """确保 Playwright 已初始化"""
        async with self._playwright_lock:
            if not self.playwright:
                from playwright.async_api import async_playwright
                self.playwright = await async_playwright().start()
    
    async def launch_persistent_browser(
        self,
        user_data_dir: str,
        headless: Optional[bool] = None,
        proxy: Optional[str] = None,
        stealth_mode: bool = True,
        extension_path: Optional[str] = None,
        **kwargs
    ):
        """启动持久化浏览器实例"""
        await self._ensure_playwright()

        _cleanup_stale_singleton_lock(user_data_dir)

        # 未配置时默认无头；显式 headless=False 时尝试启用有头（Linux 可走 Xvfb）
        if headless is None:
            headless = True
            logger.debug("google_search headless 未配置，默认无头")
        elif headless is False:
            use_headed, display_mode = _ensure_display()
            if use_headed:
                logger.info("显式请求有头模式，显示模式=%s", display_mode)
                headless = False
            else:
                logger.warning("显式请求有头模式但无可用显示器，降级为无头")
                headless = True
        
        os.makedirs(user_data_dir, exist_ok=True)

        context_options = {
            "accept_downloads": True,
            "user_agent": _get_random_user_agent(),
            "viewport": _get_random_viewport(),
            "locale": "en-US",
            "timezone_id": "America/New_York",
        }
        if proxy:
            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
            context_options["proxy"] = {"server": proxy}

        if headless is not False:
            cdp_url = SharedBrowserService.get_cdp_url_headless()
            if cdp_url:
                logger.info("使用无头 CDP 连接: %s", cdp_url)
                browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
                context = await browser.new_context(**context_options)
                self.attach_cdp_browser(browser, context)
                return context
        else:
            cdp_url = SharedBrowserService.get_cdp_url_headed()
            if not cdp_url:
                try:
                    from config.settings import settings
                    gs = getattr(settings, "google_search", None)
                    port = getattr(gs, "headed_browser_port", 9223)
                    await SharedBrowserService.start_headed(port=port)
                    cdp_url = SharedBrowserService.get_cdp_url_headed()
                except Exception as e:
                    logger.warning("懒启动有头 CDP 失败: %s", e)
            if cdp_url:
                logger.info("使用有头 CDP 连接: %s", cdp_url)
                browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
                context = await browser.new_context(**context_options)
                self.attach_cdp_browser(browser, context)
                return context
            logger.info("有头 CDP 不可用，将本地 launch 有头实例")

        context_options = {
            "headless": headless,
            "accept_downloads": True,
            "user_agent": _get_random_user_agent(),
            "viewport": _get_random_viewport(),
            "locale": "en-US",
            "timezone_id": "America/New_York",
            **kwargs
        }
        
        # 代理配置
        if proxy:
            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
            
            proxy_config = {"server": proxy}
            if '@' in proxy and '://' in proxy:
                try:
                    protocol, rest = proxy.split('://', 1)
                    if '@' in rest:
                        auth, server = rest.split('@', 1)
                        if ':' in auth:
                            username, password = auth.split(':', 1)
                            proxy_config = {
                                "server": f"{protocol}://{server}",
                                "username": username,
                                "password": password
                            }
                except Exception:
                    pass
            context_options["proxy"] = proxy_config
            logger.info(f"使用代理: {proxy}")
        
        # 隐身模式参数
        args = []
        if stealth_mode:
            args.extend([
                "--disable-blink-features=AutomationControlled",
                "--disable-features=AutomationControlled",
                "--no-sandbox"
            ])
        
        # 扩展加载
        if extension_path and os.path.exists(extension_path):
            manifest_path = os.path.join(extension_path, 'manifest.json')
            if os.path.exists(manifest_path):
                logger.info(f"加载 capsolver 扩展: {extension_path}")
                args.extend([
                    f'--disable-extensions-except={extension_path}',
                    f'--load-extension={extension_path}',
                ])
                context_options["ignore_default_args"] = ["--disable-extensions"]
        
        if args:
            context_options["args"] = args
        
        browser = await self.playwright.chromium.launch_persistent_context(
            user_data_dir,
            **context_options
        )
        
        # 尝试获取并记录 chromium 进程 PID
        pid = None
        try:
            # Playwright 内部属性路径（可能因版本而异）
            impl = getattr(browser, "_impl_obj", None)
            if impl:
                br = getattr(impl, "_browser", None)
                if br:
                    conn = getattr(br, "_connection", None)
                    if conn:
                        transport = getattr(conn, "_transport", None)
                        if transport:
                            proc = getattr(transport, "_proc", None)
                            if proc and hasattr(proc, "pid"):
                                pid = proc.pid
        except Exception:
            pass
        
        if pid:
            self.browser_pids.append(pid)
            logger.debug(f"记录 chromium 进程 PID: {pid}")
        
        # 应用隐身模式
        if stealth_mode and browser.pages:
            for page in browser.pages:
                await self._apply_stealth_mode(page)
        
        self.browsers.append(browser)
        return browser
    
    async def _apply_stealth_mode(self, page):
        """应用隐身模式脚本"""
        if _STEALTH_AVAILABLE == "v2" and _Stealth is not None:
            try:
                s = _Stealth()
                await s.apply_stealth_async(page)
                return
            except Exception:
                pass
        
        # 备用方案
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)


# ============================================================
# Scholar HTML 解析器
# ============================================================
class _ScholarParser:
    """Google Scholar 页面解析器"""
    
    @staticmethod
    def extract_results(html_content: str) -> List[Dict[str, Any]]:
        """从 HTML 内容中提取搜索结果"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        for result in soup.find_all('div', class_='gs_r'):
            try:
                ri = result.find('div', class_='gs_ri')
                if not ri:
                    continue
                
                title, link, doi = _ScholarParser._extract_title_link_doi(ri)
                if not title:
                    continue
                
                pdf_link = _ScholarParser._extract_pdf_link(result)
                if not doi and pdf_link:
                    doi = _ScholarParser._extract_doi_from_url(pdf_link)
                
                cited_count = _ScholarParser._extract_cited_count(result)
                authors, pub_info = _ScholarParser._extract_author_publisher(ri)
                snippet = _ScholarParser._extract_snippet(ri)
                year = _ScholarParser._extract_year(pub_info)
                
                results.append({
                    'title': title,
                    'link': link,
                    'pdf_link': pdf_link,
                    'authors': authors,
                    'year': year,
                    'snippet': snippet,
                    'cited_count': cited_count,
                    'publication_info': pub_info,
                    'doi': doi
                })
            except Exception as e:
                logger.debug(f"解析结果时出错: {e}")
                continue
        
        return results
    
    @staticmethod
    def _extract_title_link_doi(ri) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """提取标题、链接和 DOI"""
        title_tag = ri.find('h3', class_='gs_rt')
        if not title_tag:
            return None, None, None
        
        a_tag = title_tag.find('a')
        if a_tag:
            title = ' '.join(a_tag.get_text(separator=' ').split())
            link = a_tag.get('href')
            doi = _ScholarParser._extract_doi_from_url(link)
        else:
            title = ' '.join(title_tag.get_text(separator=' ').split())
            link = None
            doi = None
        
        return title, link, doi
    
    @staticmethod
    def _extract_pdf_link(result) -> Optional[str]:
        """提取 PDF 链接"""
        pdf_div = result.find('div', class_='gs_or_ggsm')
        if pdf_div:
            a_pdf = pdf_div.find('a', href=True)
            if a_pdf:
                if re.search(r'\[PDF\]', a_pdf.get_text(), re.I):
                    return a_pdf.get('href')
        return None
    
    @staticmethod
    def _extract_cited_count(result) -> int:
        """提取引用次数"""
        cited_link = result.find('a', string=re.compile(r'(被引用次数：|Cited by)', re.I))
        if cited_link:
            m = re.search(r'(?:被引用次数：|Cited by)\s*(\d+)', cited_link.get_text(), re.I)
            if m:
                return int(m.group(1))
        return 0
    
    @staticmethod
    def _extract_author_publisher(ri) -> Tuple[List[str], Optional[str]]:
        """提取作者和出版信息"""
        gs_a = ri.find('div', class_='gs_a')
        if not gs_a:
            return [], None
        
        text = ' '.join(gs_a.get_text(separator=' ').split())
        parts = [p.strip() for p in text.split(" - ")]
        
        if not parts:
            return [], None
        
        authors = [a.strip() for a in parts[0].split(',') if a.strip()]
        pub_info = " - ".join(parts[1:]) if len(parts) > 1 else None
        
        return authors, pub_info
    
    @staticmethod
    def _extract_snippet(ri) -> Optional[str]:
        """提取摘要"""
        snippet = ri.find('div', class_='gs_rs')
        if snippet:
            return ' '.join(snippet.get_text(separator=' ').split())
        return None
    
    @staticmethod
    def _extract_year(pub_info: Optional[str]) -> Optional[int]:
        """从出版信息中提取年份"""
        if not pub_info:
            return None
        match = re.search(r'\b(19|20)\d{2}\b', pub_info)
        if match:
            return int(match.group(0))
        return None
    
    @staticmethod
    def _extract_doi_from_url(url: Optional[str]) -> Optional[str]:
        """从 URL 中提取 DOI（doi.org 规范链接 → 任意出版商路径）"""
        if not url:
            return None

        # 1. 标准 doi.org / dx.doi.org 链接（最可靠）
        match = re.search(r'(?:doi\.org|dx\.doi\.org)/+(10\.\d{4,}/[^\s&?#]+)', url, re.I)
        if match:
            return re.sub(r'[/.),:;\s]+$', '', match.group(1))

        # 2. 出版商 URL 路径中直接嵌入 DOI（如 Springer、Wiley、Frontiers、T&F）
        match = re.search(r'/(10\.\d{4,}/[a-zA-Z0-9._\-/();:]+)', url)
        if match:
            doi = re.sub(r'[/.),:;\s]+$', '', match.group(1))
            if '/' in doi and 10 < len(doi) < 120:
                return doi

        return None

    @staticmethod
    def extract_total_results(html_content: str) -> Optional[int]:
        """提取总结果数"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results_div = soup.find('div', {'class': 'gs_ab_mdw'})
            if results_div:
                text = results_div.get_text()
                match = re.search(r'(?:About|约)\s*([\d,]+)', text)
                if match:
                    return int(match.group(1).replace(',', ''))
        except Exception:
            pass
        return None


# ============================================================
# Google 搜索 HTML 解析器
# ============================================================
class _GoogleParser:
    """Google 搜索页面解析器"""
    
    @staticmethod
    def extract_results(html_content: str) -> List[Dict[str, Any]]:
        """从 Google 搜索结果 HTML 中提取结果"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Google 搜索结果通常在 div.g 中
        for result in soup.find_all('div', class_='g'):
            try:
                # 提取标题和链接
                title_tag = result.find('h3')
                link_tag = result.find('a', href=True)
                
                if not title_tag or not link_tag:
                    continue
                
                title = title_tag.get_text(strip=True)
                link = link_tag.get('href', '')
                
                # 过滤无效链接
                if not link or link.startswith('/search') or 'google.com' in link:
                    continue
                
                # 提取摘要
                snippet = ""
                snippet_div = result.find('div', {'data-sncf': True}) or result.find('span', class_='aCOpRe')
                if snippet_div:
                    snippet = snippet_div.get_text(strip=True)
                else:
                    # 备用方案
                    for div in result.find_all('div'):
                        text = div.get_text(strip=True)
                        if len(text) > 50 and title not in text:
                            snippet = text
                            break
                
                results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                })
            except Exception as e:
                logger.debug(f"解析 Google 结果时出错: {e}")
                continue
        
        return results


# ============================================================
# Google 搜索器主类
# ============================================================
@dataclass
class GoogleSearcher:
    """
    Google/Scholar 搜索器，输出与 RAG 检索结果统一格式。
    """
    
    _config: Dict[str, Any] = field(default_factory=dict)
    _browser_manager: Optional[_BrowserManager] = field(default=None, repr=False)
    _cache: Optional[TTLCache] = field(default=None, repr=False)
    _capsolver_api: Optional[CapSolverAPI] = field(default=None, repr=False)
    _two_captcha_api: Optional[TwoCaptchaAPI] = field(default=None, repr=False)
    
    # 默认分数
    SCHOLAR_DEFAULT_SCORE = 0.8
    GOOGLE_DEFAULT_SCORE = 0.6
    
    def __post_init__(self):
        if not self._config:
            self._config = _get_google_search_config()
        cap_cfg = getattr(settings, "capsolver", None)
        if cap_cfg and getattr(cap_cfg, "enabled", False) and getattr(cap_cfg, "api_key", ""):
            self._capsolver_api = CapSolverAPI(
                api_key=getattr(cap_cfg, "api_key", ""),
                timeout_seconds=getattr(cap_cfg, "timeout_seconds", 120),
            )
        two_key = (
            getattr(getattr(settings, "content_fetcher", None), "two_captcha_api_key", "") or
            getattr(getattr(settings, "scholar_downloader", None), "twocaptcha_api_key", "")
        )
        if two_key:
            self._two_captcha_api = TwoCaptchaAPI(api_key=two_key, timeout_seconds=120)
        perf = getattr(settings, "perf_google_search", None)
        self._cache = (
            get_cache(
                getattr(perf, "cache_enabled", False),
                getattr(perf, "cache_ttl_seconds", 1800),
                prefix="google_search",
            )
            if perf else None
        )
    
    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", True))
    
    @property
    def scholar_enabled(self) -> bool:
        return bool(self._config.get("scholar_enabled", True))
    
    @property
    def google_enabled(self) -> bool:
        return bool(self._config.get("google_enabled", False))
    
    def _get_extension_path(self) -> Optional[str]:
        """获取扩展路径（使用全局 capsolver_extension_path）"""
        ext_path = getattr(settings, "capsolver_extension_path", None) or self._config.get("extension_path", "extra_tools/CapSolverExtension")
        if ext_path:
            # 支持相对路径
            if not os.path.isabs(ext_path):
                base_path = Path(__file__).resolve().parents[2]
                ext_path = str(base_path / ext_path)
            if os.path.exists(ext_path):
                return ext_path
        return None
    
    def _get_user_data_dir(self) -> str:
        """获取浏览器用户数据目录"""
        user_data_dir = self._config.get("user_data_dir")
        if user_data_dir:
            return user_data_dir
        return os.path.join(os.path.expanduser("~"), ".google_search_browser")
    
    async def search_scholar(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        job_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        搜索 Google Scholar

        Args:
            query: 搜索查询
            limit: 最大结果数
            year_start: 开始年份
            year_end: 结束年份
            job_id: 调用方 job/session 标识，用于浏览器引用计数

        Returns:
            RAG 格式的结果列表
        """
        if not self.enabled or not self.scholar_enabled:
            logger.warning("Google Scholar search skipped: disabled in config")
            return []

        limit = limit or self._config.get("max_results", 5)
        cache_key = _make_key("google_scholar", query, limit, year_start, year_end)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()

        results = []
        browser_manager = None
        context = None
        page = None
        is_shared = True
        do_headed_retry = None

        # 等待全局串行锁——等待期间不占用 Playwright 操作的 timeout 预算
        logger.debug(f"[scholar-lock] waiting  query={query!r}")
        async with _playwright_scholar_lock:
            logger.debug(f"[scholar-lock] acquired query={query!r}")
            try:
                browser_manager, context, is_shared = await acquire_shared_browser(
                    job_id=job_id,
                    timeout=timeout,
                    user_data_dir=user_data_dir,
                    headless=headless,
                    proxy=proxy,
                    extension_path=self._get_extension_path(),
                )

                page = await context.new_page()
                page.set_default_timeout(timeout)
                await browser_manager._apply_stealth_mode(page)

                # 通过搜索框输入方式搜索（避免 URL 携带搜索词触发封控）
                await self._navigate_scholar_via_searchbox(
                    page, query, year_start, year_end, timeout
                )

                # 检查验证码（等待 JS 异步重定向稳定）
                if await self._wait_and_check_captcha(page):
                    logger.warning("检测到验证码")
                    captcha_result = await self._handle_captcha_with_tiers(
                        page=page,
                        browser_manager=browser_manager,
                        headless=headless,
                        result_selector=".gs_r",
                        scope="scholar",
                    )
                    if captcha_result == "needs_headed":
                        do_headed_retry = ("scholar", query, limit, year_start, year_end, timeout, "scholar")
                    elif captcha_result == "failed":
                        return results
                    # else "solved" -> fall through

                if do_headed_retry is not None:
                    # Skip extraction; will retry with headed browser after releasing lock
                    pass
                else:
                    # 获取总结果数
                    html_content = await page.content()
                    total = _ScholarParser.extract_total_results(html_content)
                    if total:
                        logger.info(f"总共约 {total} 条结果")

                    # 多页抓取
                    current_page = 1
                    needed_pages = (limit + 9) // 10

                    while current_page <= needed_pages and len(results) < limit:
                        logger.info(f"处理第 {current_page} 页 (已收集 {len(results)}/{limit})")

                        try:
                            await page.wait_for_selector('div.gs_r.gs_or.gs_scl', timeout=timeout/4)
                        except Exception:
                            pass

                        html = await page.content()
                        page_results = _ScholarParser.extract_results(html)

                        if not page_results:
                            logger.warning(f"第 {current_page} 页无结果")
                            break

                        for r in page_results[:limit - len(results)]:
                            rag_item = self._to_scholar_rag_format(r, query)
                            results.append(rag_item)

                        logger.info(f"第 {current_page} 页提取 {len(page_results)} 条")

                        if len(results) >= limit or current_page >= needed_pages:
                            break

                        # 翻页
                        await self._random_delay(1.0, 3.0)
                        await _simulate_human_scroll_to_bottom(page)

                        if not await self._click_next_page(page, timeout):
                            break

                        current_page += 1
                        await self._random_delay()
                        await _simulate_human_behavior(page)

                    logger.info(f"Scholar 搜索完成，共 {len(results)} 条结果")
                    if self._cache:
                        self._cache.set(cache_key, results)
                    return results

            except Exception as e:
                logger.error(f"Scholar 搜索出错: {e}")
                return results
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                if browser_manager is not None:
                    await release_shared_browser(
                        job_id=job_id,
                        is_shared=is_shared,
                        manager=None if is_shared else browser_manager,
                    )

        if do_headed_retry is not None:
            _fn, _q, _lim, _ys, _ye, _to, _sc = do_headed_retry
            return await self._retry_with_headed_browser(
                search_fn=_fn,
                query=_q,
                limit=_lim,
                year_start=_ys,
                year_end=_ye,
                result_selector=".gs_r",
                timeout=_to,
                scope=_sc,
            )
        return results

    async def search_scholar_batch(
        self,
        queries: List[str],
        limit_per_query: int = 5,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        job_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        批量搜索 Google Scholar（串行执行）

        在同一个浏览器实例中依次执行多个查询，每个查询返回 limit_per_query 条结果。
        搜索完成后通过 release_shared_browser 释放引用。

        Args:
            queries: 查询列表
            limit_per_query: 每个查询的最大结果数
            year_start: 开始年份
            year_end: 结束年份
            job_id: 调用方 job/session 标识，用于浏览器引用计数

        Returns:
            合并后的 RAG 格式结果列表（len = queries数 × limit_per_query）
        """
        if not self.enabled or not self.scholar_enabled:
            logger.warning("Google Scholar batch search skipped: disabled in config")
            return []

        if not queries:
            return []

        # 检查缓存：如果所有查询都有缓存，直接返回
        all_results = []
        queries_to_search = []
        for q in queries:
            cache_key = _make_key("google_scholar", q, limit_per_query, year_start, year_end)
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    all_results.extend(cached)
                    continue
            queries_to_search.append(q)

        if not queries_to_search:
            logger.info("[retrieval] playwright scholar_batch skip all_cached queries=%d", len(queries))
            logger.info(f"Scholar 批量搜索：所有 {len(queries)} 个查询都命中缓存")
            return all_results

        logger.info("[retrieval] playwright scholar_batch start queries=%d cached=%d", len(queries_to_search), len(queries) - len(queries_to_search))
        logger.info(f"Scholar 批量搜索：{len(queries_to_search)} 个查询待执行，{len(queries) - len(queries_to_search)} 个命中缓存")

        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()

        browser_manager = None
        context = None
        page = None
        is_shared = True
        headed_retries: List[Tuple[str, int, Optional[int], Optional[int], int, str]] = []

        # 等待全局串行锁——等待期间不占用 Playwright 操作的 timeout 预算
        logger.debug(f"[scholar-lock] waiting  batch queries={len(queries_to_search)}")
        async with _playwright_scholar_lock:
            logger.debug(f"[scholar-lock] acquired batch queries={len(queries_to_search)}")
            try:
                browser_manager, context, is_shared = await acquire_shared_browser(
                    job_id=job_id,
                    timeout=timeout,
                    user_data_dir=user_data_dir,
                    headless=headless,
                    proxy=proxy,
                    extension_path=self._get_extension_path(),
                )

                page = await context.new_page()
                page.set_default_timeout(timeout)
                await browser_manager._apply_stealth_mode(page)

                # 串行执行每个查询
                for idx, query in enumerate(queries_to_search):
                    logger.info(f"Scholar 批量搜索 [{idx+1}/{len(queries_to_search)}]: {query!r}")

                    query_results = []
                    try:
                        # 通过搜索框输入方式搜索（避免 URL 携带搜索词触发封控）
                        await self._navigate_scholar_via_searchbox(
                            page, query, year_start, year_end, timeout
                        )

                        # 检查验证码（等待 JS 异步重定向稳定）
                        if await self._wait_and_check_captcha(page):
                            logger.warning(f"检测到验证码 (query={query!r})")
                            captcha_result = await self._handle_captcha_with_tiers(
                                page=page,
                                browser_manager=browser_manager,
                                headless=headless,
                                result_selector=".gs_r",
                                scope=f"scholar_batch:{query}",
                            )
                            if captcha_result == "needs_headed":
                                headed_retries.append((query, limit_per_query, year_start, year_end, timeout, f"scholar_batch:{query}"))
                                continue
                            if captcha_result == "failed":
                                continue

                        # 提取结果（支持多页抓取，每页 10 条）
                        needed_pages = max(1, (limit_per_query + 9) // 10)
                        current_page = 1

                        while current_page <= needed_pages and len(query_results) < limit_per_query:
                            try:
                                await page.wait_for_selector('div.gs_r.gs_or.gs_scl', timeout=timeout/4)
                            except Exception:
                                pass

                            html = await page.content()
                            page_results = _ScholarParser.extract_results(html)

                            if not page_results:
                                logger.warning(f"  Scholar batch: 第 {current_page} 页无结果 (query={query!r})")
                                break

                            for r in page_results[:limit_per_query - len(query_results)]:
                                rag_item = self._to_scholar_rag_format(r, query)
                                query_results.append(rag_item)

                            logger.info(f"  Scholar batch: 第 {current_page} 页提取 {len(page_results)} 条 (累计 {len(query_results)}/{limit_per_query})")

                            if len(query_results) >= limit_per_query or current_page >= needed_pages:
                                break

                            # 翻页
                            await self._random_delay(1.0, 3.0)
                            await _simulate_human_scroll_to_bottom(page)
                            if not await self._click_next_page(page, timeout):
                                break
                            current_page += 1
                            await self._random_delay()

                        logger.info(f"  -> 获取 {len(query_results)} 条结果 (query={query!r})")

                        # 缓存单个查询的结果
                        if self._cache and query_results:
                            cache_key = _make_key("google_scholar", query, limit_per_query, year_start, year_end)
                            self._cache.set(cache_key, query_results)

                        all_results.extend(query_results)

                        # 查询间延迟（避免被封）
                        if idx < len(queries_to_search) - 1:
                            await self._random_delay(2.0, 4.0)

                    except Exception as e:
                        logger.error(f"Scholar 批量搜索单个查询出错 (query={query!r}): {e}")
                        continue

                logger.info("[retrieval] playwright scholar_batch done total_hits=%d", len(all_results))
                logger.info(f"Scholar 批量搜索完成，共 {len(all_results)} 条结果")

            except Exception as e:
                logger.error(f"Scholar 批量搜索出错: {e}")
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                if browser_manager is not None:
                    await release_shared_browser(
                        job_id=job_id,
                        is_shared=is_shared,
                        manager=None if is_shared else browser_manager,
                    )

        for _q, _lim, _ys, _ye, _to, _sc in headed_retries:
            retry_results = await self._retry_with_headed_browser(
                search_fn="scholar",
                query=_q,
                limit=_lim,
                year_start=_ys,
                year_end=_ye,
                result_selector=".gs_r",
                timeout=_to,
                scope=_sc,
            )
            if retry_results and self._cache:
                cache_key = _make_key("google_scholar", _q, _lim, _ys, _ye)
                self._cache.set(cache_key, retry_results)
            all_results.extend(retry_results)

        logger.info("[retrieval] playwright scholar_batch done total_hits=%d", len(all_results))
        return all_results

    async def search_google(
        self,
        query: str,
        limit: Optional[int] = None,
        job_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        搜索 Google

        Args:
            query: 搜索查询
            limit: 最大结果数
            job_id: 调用方 job/session 标识，用于浏览器引用计数

        Returns:
            RAG 格式的结果列表
        """
        if not self.enabled or not self.google_enabled:
            logger.warning("Google search skipped: disabled in config")
            return []

        limit = limit or self._config.get("max_results", 5)
        cache_key = _make_key("google_web", query, limit)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()

        results = []
        browser_manager = None
        context = None
        page = None
        is_shared = True
        do_headed_retry = False

        try:
            browser_manager, context, is_shared = await acquire_shared_browser(
                job_id=job_id,
                timeout=timeout,
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                extension_path=self._get_extension_path(),
            )

            page = await context.new_page()
            page.set_default_timeout(timeout)
            await browser_manager._apply_stealth_mode(page)

            # 构建搜索 URL
            encoded = quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded}&hl=en"
            logger.info(f"Google 搜索 URL: {search_url}")
            
            await page.goto(search_url, wait_until="domcontentloaded")
            await self._random_delay()
            await _simulate_human_behavior(page)
            
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout/2)
            except Exception:
                logger.warning("等待页面加载超时，继续处理...")
            
            # 检查验证码
            if await self._wait_and_check_captcha(page):
                logger.warning("检测到验证码")
                captcha_result = await self._handle_captcha_with_tiers(
                    page=page,
                    browser_manager=browser_manager,
                    headless=headless,
                    result_selector="div.g",
                    scope="google",
                )
                if captcha_result == "needs_headed":
                    do_headed_retry = True
                elif captcha_result == "failed":
                    return results
                # else "solved" -> fall through
            
            if not do_headed_retry:
                # 提取结果
                html = await page.content()
                page_results = _GoogleParser.extract_results(html)
                
                for r in page_results[:limit]:
                    rag_item = self._to_google_rag_format(r, query)
                    results.append(rag_item)
                
                logger.info(f"Google 搜索完成，共 {len(results)} 条结果")
                if self._cache:
                    self._cache.set(cache_key, results)
                return results

        except Exception as e:
            logger.error(f"Google 搜索出错: {e}")
            return results
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if browser_manager is not None:
                await release_shared_browser(
                    job_id=job_id,
                    is_shared=is_shared,
                    manager=None if is_shared else browser_manager,
                )

        if do_headed_retry:
            return await self._retry_with_headed_browser(
                search_fn="google",
                query=query,
                limit=limit,
                result_selector="div.g",
                timeout=timeout,
                scope="google",
            )
        return results

    async def search_google_batch(
        self,
        queries: List[str],
        limit_per_query: int = 5,
        job_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        批量搜索 Google（串行执行）

        在同一个浏览器实例中依次执行多个查询，每个查询返回 limit_per_query 条结果。
        搜索完成后通过 release_shared_browser 释放引用。

        Args:
            queries: 查询列表
            limit_per_query: 每个查询的最大结果数
            job_id: 调用方 job/session 标识，用于浏览器引用计数

        Returns:
            合并后的 RAG 格式结果列表（len = queries数 × limit_per_query）
        """
        if not self.enabled or not self.google_enabled:
            logger.warning("Google batch search skipped: disabled in config")
            return []

        if not queries:
            return []

        # 检查缓存
        all_results = []
        queries_to_search = []
        for q in queries:
            cache_key = _make_key("google_web", q, limit_per_query)
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    all_results.extend(cached)
                    continue
            queries_to_search.append(q)

        if not queries_to_search:
            logger.info("[retrieval] playwright google_batch skip all_cached queries=%d", len(queries))
            logger.info(f"Google 批量搜索：所有 {len(queries)} 个查询都命中缓存")
            return all_results

        logger.info("[retrieval] playwright google_batch start queries=%d cached=%d", len(queries_to_search), len(queries) - len(queries_to_search))
        logger.info(f"Google 批量搜索：{len(queries_to_search)} 个查询待执行，{len(queries) - len(queries_to_search)} 个命中缓存")

        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()

        browser_manager = None
        context = None
        page = None
        is_shared = True
        headed_retries_google: List[Tuple[str, int, int, str]] = []

        try:
            browser_manager, context, is_shared = await acquire_shared_browser(
                job_id=job_id,
                timeout=timeout,
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                extension_path=self._get_extension_path(),
            )

            page = await context.new_page()
            page.set_default_timeout(timeout)
            await browser_manager._apply_stealth_mode(page)

            # 串行执行每个查询
            for idx, query in enumerate(queries_to_search):
                logger.info(f"Google 批量搜索 [{idx+1}/{len(queries_to_search)}]: {query!r}")
                
                query_results = []
                try:
                    # 构建搜索 URL
                    encoded = quote_plus(query)
                    search_url = f"https://www.google.com/search?q={encoded}&hl=en"
                    
                    await page.goto(search_url, wait_until="domcontentloaded")
                    await self._random_delay()
                    await _simulate_human_behavior(page)
                    
                    try:
                        await page.wait_for_load_state("networkidle", timeout=timeout/2)
                    except Exception:
                        pass
                    
                    # 检查验证码
                    if await self._wait_and_check_captcha(page):
                        logger.warning(f"检测到验证码 (query={query!r})")
                        captcha_result = await self._handle_captcha_with_tiers(
                            page=page,
                            browser_manager=browser_manager,
                            headless=headless,
                            result_selector="div.g",
                            scope=f"google_batch:{query}",
                        )
                        if captcha_result == "needs_headed":
                            headed_retries_google.append((query, limit_per_query, timeout, f"google_batch:{query}"))
                            continue
                        if captcha_result == "failed":
                            continue
                    
                    # 提取结果
                    html = await page.content()
                    page_results = _GoogleParser.extract_results(html)
                    
                    for r in page_results[:limit_per_query]:
                        rag_item = self._to_google_rag_format(r, query)
                        query_results.append(rag_item)
                    
                    logger.info(f"  -> 获取 {len(query_results)} 条结果")
                    
                    # 缓存单个查询的结果
                    if self._cache and query_results:
                        cache_key = _make_key("google_web", query, limit_per_query)
                        self._cache.set(cache_key, query_results)
                    
                    all_results.extend(query_results)
                    
                    # 查询间延迟（避免被封）
                    if idx < len(queries_to_search) - 1:
                        await self._random_delay(2.0, 4.0)
                
                except Exception as e:
                    logger.error(f"Google 批量搜索单个查询出错 (query={query!r}): {e}")
                    continue
            
            logger.info("[retrieval] playwright google_batch done total_hits=%d", len(all_results))
            logger.info(f"Google 批量搜索完成，共 {len(all_results)} 条结果")
        
        except Exception as e:
            logger.error(f"Google 批量搜索出错: {e}")
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if browser_manager is not None:
                await release_shared_browser(
                    job_id=job_id,
                    is_shared=is_shared,
                    manager=None if is_shared else browser_manager,
                )

        for _q, _lim, _to, _sc in headed_retries_google:
            retry_results = await self._retry_with_headed_browser(
                search_fn="google",
                query=_q,
                limit=_lim,
                result_selector="div.g",
                timeout=_to,
                scope=_sc,
            )
            if retry_results and self._cache:
                self._cache.set(_make_key("google_web", _q, _lim), retry_results)
            all_results.extend(retry_results)

        return all_results

    def _build_scholar_url(
        self, 
        query: str, 
        year_start: Optional[int] = None, 
        year_end: Optional[int] = None
    ) -> str:
        """构建 Scholar 搜索 URL（仅作为搜索框方式的 fallback）"""
        encoded = quote_plus(query)
        url = f"https://scholar.google.com/scholar?q={encoded}&hl=en&start=0"
        
        if year_start:
            url += f"&as_ylo={year_start}"
        if year_end:
            url += f"&as_yhi={year_end}"
        
        return url
    
    async def _human_type(self, page, text: str):
        """模拟人类逐字输入，带随机延迟"""
        for char in text:
            await page.keyboard.type(char)
            delay = random.uniform(0.05, 0.18)
            # 空格和标点后略长暂停
            if char in ' ,.;:':
                delay += random.uniform(0.05, 0.15)
            # 偶尔更长暂停（模拟思考）
            if random.random() < 0.05:
                delay += random.uniform(0.3, 0.6)
            await asyncio.sleep(delay)
    
    async def _navigate_scholar_via_searchbox(
        self,
        page,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        timeout: int = 60000,
    ) -> None:
        """
        通过搜索框输入方式执行 Scholar 搜索，避免 URL 携带搜索词触发反爬封控。
        
        流程: 加载首页(或复用已有结果页) → 搜索框输入 → 提交 → (可选)年份过滤
        若搜索框不可用（如验证码页面），自动回退到 URL 方式。

        调用方（search_scholar / search_scholar_batch）已持有 _playwright_scholar_lock，
        此处无需额外加锁，直接顺序执行即可。
        """
        current_url = page.url or ""
        on_scholar = "scholar.google.com" in current_url
        
        if not on_scholar:
            # 加载 Scholar 首页（不含搜索词，不易触发封控）
            logger.info("加载 Scholar 首页...")
            await page.goto("https://scholar.google.com/", wait_until="domcontentloaded")
            await self._random_delay(1.5, 3.0)
            await _simulate_human_behavior(page)
        
        # 查找搜索框（首页和结果页通用选择器）
        search_input = None
        for sel in ['input#gs_hdr_tsi', 'input[name="q"]', 'textarea[name="q"]']:
            search_input = await page.query_selector(sel)
            if search_input:
                break
        
        if not search_input:
            # 搜索框未找到（可能是验证码页面），回退到 URL 方式
            logger.warning("Scholar 搜索框未找到，回退到 URL 直接跳转")
            url = self._build_scholar_url(query, year_start, year_end)
            await page.goto(url, wait_until="domcontentloaded")
            await self._random_delay()
            await _simulate_human_behavior(page)
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout / 2)
            except Exception:
                pass
            return
        
        # 点击搜索框
        await search_input.click()
        await asyncio.sleep(random.uniform(0.3, 0.6))
        
        # 全选并清空已有内容
        modifier = "Meta" if platform.system() == "Darwin" else "Control"
        await page.keyboard.press(f"{modifier}+KeyA")
        await asyncio.sleep(random.uniform(0.1, 0.2))
        await page.keyboard.press("Backspace")
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # 逐字输入查询词（模拟人类打字）
        await self._human_type(page, query)
        await asyncio.sleep(random.uniform(0.5, 1.0))
        
        # 提交搜索
        await page.keyboard.press("Enter")
        logger.info(f"Scholar 搜索框提交: {query!r}")
        
        # 等待结果加载
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout / 2)
        except Exception:
            pass
        await self._random_delay(1.0, 2.0)
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout / 2)
        except Exception:
            logger.debug("等待 networkidle 超时，继续处理")

        # sorry 重定向可能在 networkidle 后才完成，多等一下让页面稳定
        await asyncio.sleep(1.5)
        
        # 年份过滤（在已建立 session 的基础上追加参数，不触发封控）
        if year_start or year_end:
            await self._apply_scholar_year_filter(page, year_start, year_end, timeout)
    
    async def _apply_scholar_year_filter(
        self, page, year_start: Optional[int], year_end: Optional[int], timeout: int
    ) -> None:
        """在已有搜索结果页上追加年份过滤（session 已建立，URL 参数修改安全）"""
        current_url = page.url
        params = ""
        if year_start:
            params += f"&as_ylo={year_start}"
        if year_end:
            params += f"&as_yhi={year_end}"
        if not params:
            return
        
        new_url = current_url + params
        logger.info(f"应用年份过滤: as_ylo={year_start}, as_yhi={year_end}")
        await page.goto(new_url, wait_until="domcontentloaded")
        await self._random_delay(1.0, 2.0)
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout / 2)
        except Exception:
            pass
    
    def _to_scholar_rag_format(self, result: Dict, query: str) -> Dict[str, Any]:
        """转换为 RAG 格式（Scholar）"""
        url = result.get('link') or ''
        domain = urlparse(url).netloc if url else 'scholar.google.com'
        
        metadata = {
            "source": "scholar",
            "provider": "scholar",
            "doc_id": result.get('title', ''),
            "title": result.get('title', ''),
            "url": url,
            "domain": domain,
            "search_query": query
        }
        
        if result.get('authors'):
            metadata["authors"] = result['authors']
        if result.get('year'):
            metadata["year"] = result['year']
        if result.get('publication_info'):
            metadata["venue"] = result['publication_info']
        if result.get('cited_count'):
            metadata["cited_by"] = result['cited_count']
        if result.get('pdf_link'):
            metadata["pdf_url"] = result['pdf_link']
        if result.get('doi'):
            metadata["doi"] = result['doi']
        
        return {
            "content": result.get('snippet') or result.get('title', ''),
            "score": self.SCHOLAR_DEFAULT_SCORE,
            "metadata": metadata
        }
    
    def _to_google_rag_format(self, result: Dict, query: str) -> Dict[str, Any]:
        """转换为 RAG 格式（Google）"""
        url = result.get('link') or ''
        domain = urlparse(url).netloc if url else 'google.com'
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return {
            "content": result.get('snippet') or result.get('title', ''),
            "score": self.GOOGLE_DEFAULT_SCORE,
            "metadata": {
                "source": "google",
                "provider": "google",
                "doc_id": result.get('title', ''),
                "title": result.get('title', ''),
                "url": url,
                "domain": domain,
                "search_query": query
            }
        }
    
    async def _check_captcha(self, page) -> bool:
        """检查是否有验证码（DOM 元素 + sorry URL + HTML 文本回退）"""
        try:
            url = (page.url or "").lower()
            if "/sorry/" in url or "google.com/sorry" in url:
                return True
            selector = "form#captcha-form, #gs_captcha_ccl, div.g-recaptcha, #recaptcha, [data-sitekey]"
            if await page.query_selector(selector) is not None:
                return True
            # JS 异步跳转可能还没完成，回退检查 HTML 文本
            html = await page.content()
            if "captcha-form" in html or "g-recaptcha" in html or "sorry/image" in html:
                return True
            return False
        except Exception:
            return False

    async def _wait_and_check_captcha(self, page, max_wait: float = 5.0) -> bool:
        """等待页面稳定后检查验证码（处理 JS 异步重定向）"""
        if await self._check_captcha(page):
            return True
        deadline = asyncio.get_event_loop().time() + max_wait
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1)
            if await self._check_captcha(page):
                return True
            # 有结果元素出现说明不是验证码页
            if await page.query_selector(".gs_r, div.g"):
                return False
        return False
    
    async def _check_results(self, page, selector: str) -> bool:
        """检查是否有搜索结果"""
        try:
            return await page.query_selector(selector) is not None
        except Exception:
            return False

    def _get_capsolver_api(self) -> Optional[CapSolverAPI]:
        api = self._capsolver_api
        if api and api.enabled:
            return api
        return None

    def _get_two_captcha_api(self) -> Optional[TwoCaptchaAPI]:
        api = self._two_captcha_api
        if api and api.enabled:
            return api
        return None

    async def _handle_captcha_with_tiers(
        self,
        page,
        browser_manager: _BrowserManager,
        headless: Optional[bool],
        result_selector: str,
        scope: str,
    ) -> str:
        """
        Tiered captcha handling:
        1) CapSolver API (headless friendly)
        2) 2Captcha API (headless friendly)
        3) If already in headed non-CDP browser: extension fallback
        4) Else: return "needs_headed" so caller can retry with headed browser

        Returns:
            "solved" | "needs_headed" | "failed"
        """
        capsolver = self._get_capsolver_api()
        if capsolver:
            solved = await capsolver.solve_captcha_on_page(page)
            if solved:
                # Give challenge page some time to navigate/refresh after token injection.
                for _ in range(20):
                    await asyncio.sleep(1)
                    if not await self._check_captcha(page):
                        break
                if not await self._check_captcha(page):
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                    except Exception:
                        pass
                    try:
                        await page.wait_for_selector(result_selector, timeout=12000)
                    except Exception:
                        pass
                    if await self._check_results(page, result_selector):
                        logger.info("captcha solved by CapSolver API (%s)", scope)
                        return "solved"
                logger.warning("CapSolver API attempted but captcha/results not cleared (%s)", scope)
            else:
                logger.warning("CapSolver API solve failed (%s)", scope)

        two_captcha = self._get_two_captcha_api()
        if two_captcha:
            solved = await two_captcha.solve_captcha_on_page(page)
            if solved:
                for _ in range(20):
                    await asyncio.sleep(1)
                    if not await self._check_captcha(page):
                        break
                if not await self._check_captcha(page):
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                    except Exception:
                        pass
                    try:
                        await page.wait_for_selector(result_selector, timeout=12000)
                    except Exception:
                        pass
                    if await self._check_results(page, result_selector):
                        logger.info("captcha solved by 2Captcha API (%s)", scope)
                        return "solved"
                logger.warning("2Captcha API attempted but captcha/results not cleared (%s)", scope)
            else:
                logger.warning("2Captcha API solve failed (%s)", scope)

        # Only do inline headed extension-wait when already in a headed non-CDP browser
        is_headed = (
            (headless is False) and not browser_manager.is_cdp_connected()
        )
        if is_headed:
            logger.info("有头模式兜底验证码处理（CapSolver->2Captcha 失败后）(%s)...", scope)
            await self._inject_capsolver_attributes(page)
            for i in range(90):
                await asyncio.sleep(1)
                if not await self._check_captcha(page):
                    logger.info("验证码已解决 (%s)", scope)
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                        await page.wait_for_selector(result_selector, timeout=10000)
                    except Exception:
                        pass
                    break
                if i > 0 and i % 15 == 0:
                    await self._inject_capsolver_attributes(page)
            if await self._check_results(page, result_selector):
                return "solved"
            logger.error("验证码处理后仍无结果 (%s)", scope)
            return "failed"

        logger.info(
            "无头/CDP 模式下 API 解验证码失败 (%s)，将尝试有头重试",
            scope,
        )
        return "needs_headed"

    async def _retry_with_headed_browser(
        self,
        search_fn: str,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        result_selector: str = ".gs_r",
        timeout: int = 60000,
        scope: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retry a single search in a fresh local headed browser (no CDP).
        Used when CapSolver + 2Captcha fail in headless/CDP mode.
        """
        use_headed, display_mode = _ensure_display()
        if not use_headed:
            logger.warning("有头重试跳过: 无可用显示器 (display=%s)", display_mode)
            return []

        user_data_dir = self._get_user_data_dir()
        proxy = self._config.get("proxy")
        extension_path = self._get_extension_path()
        manager = _BrowserManager(timeout=timeout)
        context = None
        page = None
        results: List[Dict[str, Any]] = []

        try:
            async with _profile_lock_guard:
                context = await manager.launch_persistent_browser(
                    user_data_dir=user_data_dir,
                    headless=False,
                    proxy=proxy,
                    stealth_mode=True,
                    extension_path=extension_path,
                )
            page = await context.new_page()
            page.set_default_timeout(timeout)
            await manager._apply_stealth_mode(page)

            if search_fn == "scholar":
                await self._navigate_scholar_via_searchbox(
                    page, query, year_start, year_end, timeout
                )
            else:
                encoded = quote_plus(query)
                search_url = f"https://www.google.com/search?q={encoded}&hl=en"
                await page.goto(search_url, wait_until="domcontentloaded")
                await self._random_delay()
                await _simulate_human_behavior(page)
                try:
                    await page.wait_for_load_state("networkidle", timeout=timeout // 2)
                except Exception:
                    pass

            if await self._wait_and_check_captcha(page):
                logger.info("有头重试: 检测到验证码，等待扩展处理 (%s)", scope)
                await self._inject_capsolver_attributes(page)
                for i in range(90):
                    await asyncio.sleep(1)
                    if not await self._check_captcha(page):
                        logger.info("有头重试: 验证码已解决 (%s)", scope)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=15000)
                            await page.wait_for_selector(result_selector, timeout=10000)
                        except Exception:
                            pass
                        break
                    if i > 0 and i % 15 == 0:
                        await self._inject_capsolver_attributes(page)
                if await self._check_captcha(page) or not await self._check_results(page, result_selector):
                    logger.warning("有头重试: 验证码未解决或无结果 (%s)", scope)
                    return results

            if search_fn == "scholar":
                needed_pages = (limit + 9) // 10
                current_page = 1
                while current_page <= needed_pages and len(results) < limit:
                    try:
                        await page.wait_for_selector("div.gs_r.gs_or.gs_scl", timeout=timeout // 4)
                    except Exception:
                        pass
                    html = await page.content()
                    page_results = _ScholarParser.extract_results(html)
                    if not page_results:
                        break
                    for r in page_results[: limit - len(results)]:
                        results.append(self._to_scholar_rag_format(r, query))
                    if len(results) >= limit or current_page >= needed_pages:
                        break
                    await self._random_delay(1.0, 3.0)
                    await _simulate_human_scroll_to_bottom(page)
                    if not await self._click_next_page(page, timeout):
                        break
                    current_page += 1
                    await self._random_delay()
                    await _simulate_human_behavior(page)
            else:
                html = await page.content()
                page_results = _GoogleParser.extract_results(html)
                for r in page_results[:limit]:
                    results.append(self._to_google_rag_format(r, query))

            if results:
                logger.info("有头重试成功 (%s): %d 条结果", scope, len(results))
        except Exception as e:
            logger.warning("有头重试异常 (%s): %s", scope, e)
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            await release_shared_browser(
                job_id="",
                is_shared=False,
                manager=manager,
            )
        return results

    async def _inject_capsolver_attributes(self, page):
        """注入 capsolver 属性"""
        try:
            await page.evaluate("""
                () => {
                    const recaptcha = document.querySelector('iframe[src*="recaptcha"]') ||
                                     document.querySelector('.g-recaptcha');
                    if (recaptcha) return { success: true, isRecaptcha: true };
                    
                    const img = document.querySelector('img[src*="sorry/image"]') ||
                               document.querySelector('form img');
                    const input = document.querySelector('input[name="captcha"]') ||
                                 document.querySelector('form input[type="text"]');
                    
                    if (img && input) {
                        const id = 'gs-captcha-' + Date.now();
                        img.setAttribute('capsolver-image-to-text-source', id);
                        input.setAttribute('capsolver-image-to-text-result', id);
                        return { success: true };
                    }
                    return { success: false };
                }
            """)
        except Exception as e:
            logger.debug(f"注入 capsolver 属性失败: {e}")
    
    async def _click_next_page(self, page, timeout: int) -> bool:
        """点击下一页按钮"""
        try:
            selectors = [
                '#gs_n td:last-child a',
                'a.gs_btnPR',
                '.gs_btnPR'
            ]
            
            for selector in selectors:
                btn = await page.query_selector(selector)
                if btn:
                    url_before = page.url
                    await btn.scroll_into_view_if_needed()
                    await asyncio.sleep(0.5)
                    await btn.click()
                    
                    try:
                        await page.wait_for_load_state("networkidle", timeout=timeout/2)
                    except Exception:
                        pass
                    
                    if page.url != url_before:
                        return True
            
            return False
        except Exception as e:
            logger.debug(f"点击下一页失败: {e}")
            return False
    
    async def _random_delay(self, min_s: float = 1.0, max_s: float = 2.0):
        """随机延迟"""
        await asyncio.sleep(random.uniform(min_s, max_s))


async def cleanup_shared_browser():
    """清理全局共享的浏览器实例。在程序退出或测试完成后调用。ref_count > 0 时不强关。"""
    global _shared_browser_manager, _shared_browser_last_used, _browser_ref_count, _browser_active_jobs
    async with _profile_lock_guard:
        async with _shared_browser_lock:
            if _browser_ref_count > 0:
                logger.warning(
                    "Skip cleanup_shared_browser: ref_count=%d active_jobs=%s",
                    _browser_ref_count,
                    _browser_active_jobs,
                )
                return
            if _shared_browser_manager is not None:
                await _do_close_shared_browser()


def release_shared_browser_sync(job_id: str = "") -> None:
    """同步释放本 job 对共享浏览器的引用（供 DR 结束等同步上下文调用）。"""
    from src.retrieval.unified_web_search import _get_bg_loop
    try:
        bg = _get_bg_loop()
        fut = asyncio.run_coroutine_threadsafe(
            release_shared_browser(job_id=job_id, is_shared=True, manager=None),
            bg,
        )
        fut.result(timeout=10)
    except Exception:
        pass


def cleanup_shared_browser_sync():
    """同步版本的清理函数（在非异步上下文中使用）"""
    from src.retrieval.unified_web_search import _get_bg_loop
    try:
        bg = _get_bg_loop()
        fut = asyncio.run_coroutine_threadsafe(cleanup_shared_browser(), bg)
        fut.result(timeout=10)
    except Exception:
        pass


# 注册退出时自动清理
import atexit
atexit.register(cleanup_shared_browser_sync)


# 全局单例
google_searcher = GoogleSearcher()
