"""
WebContentFetcher — 网络搜索结果全文抓取模块

三级提取策略自动降级：
1. trafilatura — 纯 HTTP 抓取 + 正文提取，轻量快速，无需浏览器
2. BrightData Web Unlocker — 对付反爬站点，需配置 API Key
3. Playwright 浏览器 — JS 渲染页面，含 Cloudflare 检测、stealth 反检测

URL 过滤机制：
- is_qualifying_url() 自动跳过搜索引擎、社交媒体、二进制文件、登录页
- 支持 only_academic=True 模式仅提取学术域名

与 RAG 项目完全兼容：
- 输入/输出格式与 UnifiedWebSearcher 一致（List[Dict]，含 content + metadata）
- 提取成功后回写 hit["content"] 为全文，原始片段保留在 metadata.original_snippet
- 通过 metadata.content_type = "full_text" 标记已提取的结果
- 支持 TTLCache、get_logger、from_settings() 配置加载
- 提供 enrich_results_sync() 同步版本兼容 RetrievalService

使用方法:
---------
from src.retrieval.web_content_fetcher import WebContentFetcher

# 从配置加载
fetcher = WebContentFetcher.from_settings()

# 异步增强搜索结果
enriched = await fetcher.enrich_results(results, query="deep sea microbiome")

# 同步版本（兼容 RetrievalService）
enriched = fetcher.enrich_results_sync(results, query="deep sea microbiome")
"""

import asyncio
import concurrent.futures
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.log import get_logger
from src.retrieval.browser_service import SharedBrowserService, get_cdp_context_options
from src.retrieval.downloader.browser_manager import apply_stealth_to_page
from src.retrieval.downloader.captcha_page_runner import run_captcha_flow
from src.retrieval.downloader.captcha_solver import CaptchaSolver
from src.utils.cache import DiskCache, TTLCache, _make_key, get_cache, get_disk_cache
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

logger = get_logger(__name__)


class ActivityTimer:
    """
    活跃度感知的截止时间控制器。

    初始 deadline = now + idle_timeout。每次调用 extend() 会把 deadline
    推到 max(当前值, now + extra_seconds)，防止正在进行中的验证码解决被提前中断。
    """

    def __init__(self, idle_timeout: float) -> None:
        self._deadline: float = time.monotonic() + idle_timeout
        self._idle_timeout = idle_timeout

    def extend(self, extra_seconds: float, reason: str = "") -> None:
        """将 deadline 推到 max(当前值, now + extra_seconds)。"""
        new_deadline = time.monotonic() + extra_seconds
        if new_deadline > self._deadline:
            logger.debug(
                "[activity_timer] extended +%.1fs reason=%r, remaining=%.1fs",
                extra_seconds,
                reason,
                new_deadline - time.monotonic(),
            )
            self._deadline = new_deadline

    @property
    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self._deadline


async def _run_with_activity_timer(
    coro: Any,
    timer: ActivityTimer,
    poll_interval: float = 1.0,
) -> Any:
    """
    运行 coro（包装为 Task），当 timer.expired 时取消任务。

    watchdog 每 poll_interval 秒检查一次 timer，过期则取消。
    返回协程结果；若被 watchdog 取消则返回 None。
    """
    task = asyncio.create_task(coro)

    async def _watchdog() -> None:
        while not task.done():
            if timer.expired:
                logger.debug("[activity_timer] watchdog: timer expired, cancelling task")
                task.cancel()
                return
            await asyncio.sleep(poll_interval)

    wd = asyncio.create_task(_watchdog())
    try:
        return await task
    except asyncio.CancelledError:
        logger.debug("[activity_timer] task cancelled by watchdog (timer expired)")
        return None
    finally:
        wd.cancel()
        try:
            await wd
        except asyncio.CancelledError:
            pass


class _FetchDecisionResponse(BaseModel):
    urls_to_fetch: List[str] = Field(default_factory=list)

# ============================================================
# URL 过滤常量
# ============================================================

# 跳过的域名（搜索引擎、社交媒体、视频、论坛等）
SKIP_DOMAINS = {
    # 搜索引擎
    "google.com", "google.co", "bing.com", "yahoo.com", "baidu.com",
    "duckduckgo.com", "yandex.com", "sogou.com",
    # 社交媒体
    "twitter.com", "x.com", "facebook.com", "linkedin.com",
    "weibo.com", "zhihu.com",
    # 视频 / 图片
    "youtube.com", "tiktok.com", "instagram.com", "vimeo.com",
    # 论坛 / 问答
    "reddit.com", "quora.com", "stackexchange.com", "stackoverflow.com",
    # 代码托管
    "github.com", "gitlab.com", "bitbucket.org",
}

# 域名中包含这些子串时跳过（登录页等）
SKIP_DOMAIN_SUBSTRINGS = {"login.", "signin.", "auth.", "account.", "sso."}

# 跳过的文件扩展名（二进制 / 多媒体）
SKIP_EXTENSIONS = {
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".exe", ".dmg", ".msi", ".deb", ".rpm",
    ".mp4", ".mp3", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}

# 学术域名白名单（only_academic 模式）
ACADEMIC_DOMAINS = {
    "scholar.google.com", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov", "doi.org", "dx.doi.org",
    "sciencedirect.com", "springer.com", "link.springer.com",
    "wiley.com", "onlinelibrary.wiley.com",
    "nature.com", "science.org", "pnas.org",
    "cell.com", "thelancet.com", "bmj.com",
    "frontiersin.org", "mdpi.com", "plos.org", "peerj.com",
    "academic.oup.com", "tandfonline.com", "sagepub.com",
    "biomedcentral.com", "biorxiv.org", "medrxiv.org",
    "researchgate.net", "semanticscholar.org",
    "jstor.org", "ieee.org", "ieeexplore.ieee.org",
    "acm.org", "dl.acm.org",
}

# 每页抓取总时长上限：默认 1 分钟；启用 BrightData + 2captcha 时为 2 分钟
PER_PAGE_TIMEOUT_CAP_SECONDS = 60
PER_PAGE_TIMEOUT_CAP_EXTENDED_SECONDS = 120

# 学术域名子串匹配
ACADEMIC_DOMAIN_SUBSTRINGS = {
    "scholar", "arxiv", "pubmed", "doi.org", "sciencedirect",
    "springer", "wiley", "nature.com", "science.org",
    "ncbi.nlm.nih.gov", "frontiersin", "mdpi", "plos",
    "biomedcentral", "biorxiv", "medrxiv", "researchgate",
    "semanticscholar", "ieee", "acm.org",
}


def _extract_domain(url: str) -> str:
    """从 URL 提取域名（不含端口）"""
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _get_path_ext(url: str) -> str:
    """从 URL 路径提取扩展名"""
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        dot_idx = path.rfind(".")
        if dot_idx >= 0:
            return path[dot_idx:].lower().split("?")[0].split("#")[0]
    except Exception:
        pass
    return ""


def is_qualifying_url(url: str, only_academic: bool = False) -> bool:
    """
    判断 URL 是否值得抓取全文。

    过滤逻辑：
    - 跳过搜索引擎、社交媒体、视频、二进制文件、登录页
    - only_academic=True 时仅保留学术域名

    Args:
        url: 待检查的 URL
        only_academic: 是否仅允许学术域名

    Returns:
        True 表示可以抓取
    """
    if not url or not url.startswith(("http://", "https://")):
        return False

    domain = _extract_domain(url)
    if not domain:
        return False

    # 检查域名子串黑名单（登录页等）
    for sub in SKIP_DOMAIN_SUBSTRINGS:
        if sub in domain:
            return False

    # 检查域名黑名单
    for skip in SKIP_DOMAINS:
        if domain == skip or domain.endswith("." + skip):
            return False

    # 检查文件扩展名
    ext = _get_path_ext(url)
    if ext in SKIP_EXTENSIONS:
        return False

    # 学术域名过滤
    if only_academic:
        # 精确匹配
        for acad in ACADEMIC_DOMAINS:
            if domain == acad or domain.endswith("." + acad):
                return True
        # 子串匹配
        for sub in ACADEMIC_DOMAIN_SUBSTRINGS:
            if sub in domain:
                return True
        return False

    return True


# ============================================================
# WebContentFetcher
# ============================================================

@dataclass
class WebContentFetcher:
    """
    网络搜索结果全文抓取器。

    三级提取策略自动降级：trafilatura → Playwright → BrightData
    """

    enabled: bool = True
    only_academic: bool = False
    max_content_length: int = 8000
    timeout_seconds: int = 60
    brightdata_api_key: str = ""
    brightdata_zone: str = ""
    two_captcha_api_key: str = ""  # 与 BrightData 同时启用时，单页抓取上限提高到 2 分钟；Cloudflare/Turnstile 仅用 2Captcha
    capsolver_api_key: str = ""  # 其他验证码优先 CapSolver，失败再 2Captcha
    captcha_timeout_seconds: int = 120
    # 动态软超时参数
    idle_timeout_seconds: int = 30          # Playwright 无任何活动时的基础超时
    captcha_detect_extra_seconds: int = 90  # 检测到验证码后额外给予的时间
    captcha_solving_extra_seconds: int = 120  # 调用解码 API 后额外给予的时间
    captcha_token_extra_seconds: int = 45   # token 应用后额外给予的时间
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    disk_cache_enabled: bool = True
    disk_cache_ttl_seconds: int = 2592000   # 30 days; entries promoted to permanent after promote_threshold hits
    disk_cache_promote_threshold: int = 3   # hits within TTL window before entry becomes permanent
    disk_cache_dir: str = "data/cache"
    max_concurrent: int = 5
    _cache: Optional[TTLCache] = field(default=None, repr=False, init=False)
    _disk_cache: Optional[DiskCache] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.cache_enabled:
            self._cache = get_cache(
                enabled=True,
                ttl_seconds=self.cache_ttl_seconds,
                maxsize=512,
                prefix="content_fetcher",
            )
        if self.disk_cache_enabled:
            import os
            db_path = os.path.join(self.disk_cache_dir, "web_content.db")
            self._disk_cache = get_disk_cache(
                enabled=True,
                db_path=db_path,
                ttl_seconds=self.disk_cache_ttl_seconds,
                promote_threshold=self.disk_cache_promote_threshold,
            )
        self._inflight: Dict[str, asyncio.Future] = {}
        self._captcha_solver: Optional[CaptchaSolver] = None
        if self.two_captcha_api_key or self.capsolver_api_key:
            self._captcha_solver = CaptchaSolver(
                capsolver_api_key=self.capsolver_api_key,
                twocaptcha_api_key=self.two_captcha_api_key,
                timeout_seconds=self.captcha_timeout_seconds,
            )

    def _per_page_timeout_cap_seconds(self) -> int:
        """单页抓取总时长上限：启用 BrightData 且配置 2captcha 时为 2 分钟，否则 1 分钟。"""
        if self.brightdata_api_key and self.two_captcha_api_key:
            return PER_PAGE_TIMEOUT_CAP_EXTENDED_SECONDS
        return PER_PAGE_TIMEOUT_CAP_SECONDS

    @property
    def effective_timeout_seconds(self) -> int:
        """实际使用的单页超时：不超过配置的 timeout_seconds，且不超过每页总上限。"""
        return min(self.timeout_seconds, self._per_page_timeout_cap_seconds())

    @classmethod
    def from_settings(cls) -> "WebContentFetcher":
        """从 config/settings 加载配置。取值优先级：请求/UI 入参（若有）> config > 代码默认；此处仅使用 config + 默认。"""
        try:
            from config.settings import settings
            cfg = getattr(settings, "content_fetcher", None)
            if cfg is not None:
                return cls(
                    enabled=getattr(cfg, "enabled", False),
                    only_academic=getattr(cfg, "only_academic", False),
                    max_content_length=getattr(cfg, "max_content_length", 8000),
                    timeout_seconds=getattr(cfg, "timeout_seconds", 30),
                    brightdata_api_key=getattr(cfg, "brightdata_api_key", ""),
                    brightdata_zone=getattr(cfg, "brightdata_zone", ""),
                    two_captcha_api_key=getattr(cfg, "two_captcha_api_key", ""),
                    capsolver_api_key=getattr(cfg, "capsolver_api_key", ""),
                    captcha_timeout_seconds=getattr(cfg, "captcha_timeout_seconds", 120),
                    idle_timeout_seconds=int(getattr(cfg, "idle_timeout_seconds", 30)),
                    captcha_detect_extra_seconds=int(getattr(cfg, "captcha_detect_extra_seconds", 90)),
                    captcha_solving_extra_seconds=int(getattr(cfg, "captcha_solving_extra_seconds", 120)),
                    captcha_token_extra_seconds=int(getattr(cfg, "captcha_token_extra_seconds", 45)),
                    cache_enabled=getattr(cfg, "cache_enabled", True),
                    cache_ttl_seconds=getattr(cfg, "cache_ttl_seconds", 3600),
                    disk_cache_enabled=getattr(cfg, "disk_cache_enabled", True),
                    disk_cache_ttl_seconds=getattr(cfg, "disk_cache_ttl_seconds", 2592000),
                    disk_cache_promote_threshold=getattr(cfg, "disk_cache_promote_threshold", 3),
                    disk_cache_dir=getattr(cfg, "disk_cache_dir", "data/cache"),
                    max_concurrent=getattr(cfg, "max_concurrent", 5),
                )
        except Exception as e:
            logger.warning(f"从 settings 加载 content_fetcher 配置失败: {e}")
        return cls(enabled=False)

    # --------------------------------------------------------
    # 第一级：trafilatura（纯 HTTP + 正文提取）
    # --------------------------------------------------------

    async def _fetch_trafilatura(self, url: str) -> Optional[str]:
        """
        使用 trafilatura 抓取并提取正文。

        轻量快速，不启动浏览器，适合绝大多数正常网页。
        """
        try:
            import trafilatura

            loop = asyncio.get_event_loop()
            # trafilatura.fetch_url 是同步的，放到 executor 避免阻塞
            downloaded = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: trafilatura.fetch_url(url),
                ),
                timeout=self.effective_timeout_seconds,
            )
            if not downloaded:
                return None

            text = await loop.run_in_executor(
                None,
                lambda: trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                    favor_precision=True,
                ),
            )
            if text and len(text.strip()) > 100:
                logger.debug(f"trafilatura 成功: {url} ({len(text)} chars)")
                return text.strip()
            return None
        except asyncio.TimeoutError:
            logger.debug(f"trafilatura 超时: {url}")
            return None
        except ImportError:
            logger.warning("trafilatura 未安装，跳过第一级提取")
            return None
        except Exception as e:
            logger.debug(f"trafilatura 失败: {url} - {e}")
            return None

    # --------------------------------------------------------
    # 第三级：BrightData Web Unlocker
    # --------------------------------------------------------

    async def _fetch_brightdata(self, url: str) -> Optional[str]:
        """
        使用 BrightData Web Unlocker API 抓取。

        适合反爬站点，需要配置 brightdata_api_key。
        """
        if not self.brightdata_api_key:
            return None

        try:
            from src.utils.aiohttp_tls_patch import apply_aiohttp_tls_in_tls_patch
            apply_aiohttp_tls_in_tls_patch()
            import aiohttp
            import trafilatura

            zone = self.brightdata_zone or "web_unlocker1"
            proxy_url = f"https://brd-customer-{self.brightdata_api_key}-zone-{zone}:@brd.superproxy.io:33335"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=self.effective_timeout_seconds),
                    headers={"User-Agent": "Mozilla/5.0 (compatible; DeepSeaRAG/1.0)"},
                    ssl=False,
                ) as resp:
                    if resp.status != 200:
                        logger.debug(f"BrightData 返回 {resp.status}: {url}")
                        return None
                    html = await resp.text()

            if not html or len(html) < 200:
                return None

            # 用 trafilatura 从 HTML 提取正文
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                ),
            )
            if text and len(text.strip()) > 100:
                logger.debug(f"BrightData 成功: {url} ({len(text)} chars)")
                return text.strip()
            return None
        except asyncio.TimeoutError:
            logger.debug(f"BrightData 超时: {url}")
            return None
        except ImportError as e:
            logger.debug(f"BrightData 依赖缺失: {e}")
            return None
        except Exception as e:
            logger.debug(f"BrightData 失败: {url} - {e}")
            return None

    # --------------------------------------------------------
    # 第二级：Playwright 浏览器
    # --------------------------------------------------------

    async def _fetch_playwright(self, url: str) -> Optional[str]:
        """
        使用 Playwright headless 浏览器抓取。

        支持 JS 渲染页面，含 stealth 反检测。
        优先从 resident context 池借出；不可用时回退到临时浏览器。
        """
        html: Optional[str] = None
        try:
            from playwright.async_api import async_playwright

            # 创建活跃度感知 timer；验证码各里程碑通过回调推延 deadline
            timer = ActivityTimer(float(self.idle_timeout_seconds))

            def _on_captcha_progress(milestone: str) -> None:
                extras: Dict[str, int] = {
                    "captcha_detected": self.captcha_detect_extra_seconds,
                    "solving_started":  self.captcha_solving_extra_seconds,
                    "token_received":   self.captcha_token_extra_seconds,
                }
                if milestone in extras:
                    timer.extend(float(extras[milestone]), reason=milestone)

            # Prefer headless context pool (bridge-safe: worker runs on owner loop)
            try:
                from config.settings import settings
                from src.retrieval.context_pool import run_with_headless_context

                cfg = getattr(settings, "shared_browser", None)
                acquire_timeout = getattr(cfg, "context_acquire_timeout_seconds", 30.0) if cfg else 30.0
                timeout_ms = self.effective_timeout_seconds * 1000

                async def _pool_worker(context):
                    page = await context.new_page()
                    try:
                        await apply_stealth_to_page(page)
                        await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=timeout_ms,
                        )
                        await page.wait_for_timeout(2000)
                        if self._captcha_solver and self._captcha_solver.has_any_provider:
                            await run_captcha_flow(
                                page,
                                self._captcha_solver,
                                self.two_captcha_api_key,
                                captcha_timeout_seconds=self.captcha_timeout_seconds,
                                max_retries=2,
                                on_progress=_on_captcha_progress,
                            )
                        return await page.content()
                    finally:
                        await page.close()

                # 注：若 context pool 跨事件循环（bridge 模式），timer 取消外层 task 后
                # pool 侧协程仍会继续执行直至自然结束（浏览器槽位短暂占用，不影响正确性）
                pool_html = await _run_with_activity_timer(
                    run_with_headless_context(
                        _pool_worker,
                        timeout=acquire_timeout,
                        purpose="content_fetcher",
                    ),
                    timer,
                )
                if pool_html and len(pool_html) >= 200:
                    html = pool_html
            except Exception as e:
                logger.debug("[web_content_fetcher] pooled fetch failed, fallback: %s", e)

            # Fallback: ephemeral browser/context (only if pool did not produce html)
            if not (html and len(html) >= 200):
                async def _ephemeral_fetch() -> Optional[str]:
                    async with async_playwright() as p:
                        cdp_url = SharedBrowserService.get_cdp_url_headless()
                        own_browser = False
                        if cdp_url:
                            browser = await p.chromium.connect_over_cdp(cdp_url)
                        else:
                            browser = await p.chromium.launch(headless=True)
                            own_browser = True
                        try:
                            context = await browser.new_context(**get_cdp_context_options())
                            page = await context.new_page()
                            await apply_stealth_to_page(page)
                            await page.goto(
                                url,
                                wait_until="domcontentloaded",
                                timeout=self.effective_timeout_seconds * 1000,
                            )
                            await page.wait_for_timeout(2000)
                            if self._captcha_solver and self._captcha_solver.has_any_provider:
                                await run_captcha_flow(
                                    page,
                                    self._captcha_solver,
                                    self.two_captcha_api_key,
                                    captcha_timeout_seconds=self.captcha_timeout_seconds,
                                    max_retries=2,
                                    on_progress=_on_captcha_progress,
                                )
                            result_html = await page.content()
                            await context.close()
                            return result_html
                        finally:
                            if own_browser:
                                await browser.close()

                html = await _run_with_activity_timer(_ephemeral_fetch(), timer)

            if not html or len(html) < 200:
                return None

            # 用 trafilatura 从渲染后的 HTML 提取正文
            try:
                import trafilatura

                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    lambda: trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                    ),
                )
                if text and len(text.strip()) > 100:
                    logger.debug(f"Playwright 成功: {url} ({len(text)} chars)")
                    return text.strip()
            except ImportError:
                # trafilatura 不可用时用 BeautifulSoup 降级
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html, "html.parser")
                    # 移除 script/style
                    for tag in soup(["script", "style", "nav", "header", "footer"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    if text and len(text) > 100:
                        logger.debug(f"Playwright+BS4 成功: {url} ({len(text)} chars)")
                        return text
                except Exception:
                    pass

            return None
        except asyncio.TimeoutError:
            logger.debug(f"Playwright 超时: {url}")
            return None
        except ImportError:
            logger.debug("Playwright 未安装，跳过第三级提取")
            return None
        except Exception as e:
            logger.debug(f"Playwright 失败: {url} - {e}")
            return None

    # --------------------------------------------------------
    # 三级自动降级
    # --------------------------------------------------------

    async def fetch_content(self, url: str) -> Optional[str]:
        """
        抓取 URL 全文内容，三级策略自动降级。

        trafilatura → Playwright → BrightData

        同 URL 并发请求自动合并（in-flight dedup），周期内 TTLCache 防止重复抓取。

        Args:
            url: 目标 URL

        Returns:
            提取的正文文本，失败返回 None
        """
        cache_key = _make_key("content", url)

        # ── L1: 内存 TTLCache（进程内，毫秒级）──
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("[content_cache] L1 hit: %s", url)
                return cached

        # ── L2: 磁盘 SQLite 缓存（跨重启，秒级）──
        if self._disk_cache:
            disk_cached = self._disk_cache.get(cache_key)
            if disk_cached is not None:
                logger.info("[content_cache] L2 disk hit: %s", url)
                if self._cache:
                    self._cache.set(cache_key, disk_cached)   # 回填 L1
                return disk_cached

        # ── In-flight dedup: 同 URL 已有在途请求则等待 ──
        inflight = self._inflight.get(url)
        if inflight is not None:
            logger.debug("等待同 URL 在途请求: %s", url)
            try:
                return await asyncio.shield(inflight)
            except Exception:
                pass

        # ── 发起新请求，注册 Future 供其他协程复用 ──
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Optional[str]] = loop.create_future()
        self._inflight[url] = fut

        async def _fetch_with_cap() -> Optional[str]:
            text = await self._fetch_trafilatura(url)
            if text is None:
                text = await self._fetch_playwright(url)
            if text is None:
                text = await self._fetch_brightdata(url)
            return text

        try:
            # trafilatura / brightdata 各自内部已有独立超时，此处用松弛兜底防止意外挂死。
            # playwright 路径由内部 ActivityTimer + watchdog 管理，不依赖此硬上限。
            _outer_cap = max(float(self.idle_timeout_seconds) * 5, 300.0)
            try:
                text = await asyncio.wait_for(
                    _fetch_with_cap(),
                    timeout=_outer_cap,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "抓取总超时放弃: %s (外层兜底上限 %.0fs)",
                    url,
                    _outer_cap,
                )
                text = None

            if text and len(text) > self.max_content_length:
                text = text[: self.max_content_length]

            if text:
                if self._cache:
                    self._cache.set(cache_key, text)        # 写 L1
                if self._disk_cache:
                    self._disk_cache.set(cache_key, text)   # 写 L2
                    logger.debug("[content_cache] stored to disk: %s", url)

            if not fut.done():
                fut.set_result(text)
            return text
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(url, None)

    # --------------------------------------------------------
    # LLM 预判：哪些 URL 需要抓取全文
    # --------------------------------------------------------

    async def evaluate_snippets_need_fetch(
        self,
        query: str,
        results: List[Dict[str, Any]],
        llm_client: Any,
    ) -> List[str]:
        """
        使用 LLM 预判搜索摘要是否足够，返回需要抓取全文的 URL 列表。

        只评估来源为 scholar/google 的条目（这两种来源摘要最容易被截断）。
        LLM 返回 JSON: {"urls_to_fetch": ["url1", "url2"]}

        若 LLM 调用失败，降级为返回所有候选 URL（等同全量抓取）。

        Args:
            query:      原始用户查询
            results:    UnifiedWebSearcher 返回的结果列表
            llm_client: LLMManager 客户端实例

        Returns:
            需要抓取全文的 URL 集合（list）
        """
        # 只评估 scholar / google 来源且有合法 URL 或 DOI 的条目；有 DOI 时用 doi.org 链接作为抓取目标
        candidates = []
        for hit in results:
            metadata = hit.get("metadata") or {}
            source = metadata.get("source", "")
            url = (metadata.get("url") or "").strip()
            doi = (metadata.get("doi") or "").strip()
            snippet = (hit.get("content") or "").strip()
            if source not in ("scholar", "google") or not snippet:
                continue
            # DOI 获取后优先用 doi.org 链接，便于自动拉取
            if doi and "/" in doi and doi.startswith("10.") and len(doi) < 120:
                candidate_url = "https://doi.org/" + doi
            else:
                candidate_url = url
            if candidate_url:
                candidates.append({"url": candidate_url, "snippet": snippet})

        if not candidates:
            return []

        items_text = "\n".join(
            f"{i + 1}. URL: {c['url']}\n   摘要: {c['snippet'][:400]}"
            for i, c in enumerate(candidates)
        )
        prompt = _pm.render("web_content_fetch_decide.txt", query=query, items_text=items_text)

        try:
            logger.info(
                "[retrieval] model_call content_fetcher llm_fetch_decide candidates=%d",
                len(candidates),
            )
            resp = llm_client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("web_content_fetch_decide_system.txt")},
                    {"role": "user", "content": prompt},
                ],

                response_model=_FetchDecisionResponse,
            )
            parsed: Optional[_FetchDecisionResponse] = resp.get("parsed_object")
            if parsed is None:
                raw_text = (resp.get("final_text") or "").strip()
                if raw_text:
                    parsed = _FetchDecisionResponse.model_validate_json(raw_text)

            urls = parsed.urls_to_fetch if parsed is not None else []
            allowed = {c["url"] for c in candidates}
            urls = [u for u in urls if isinstance(u, str) and u in allowed]
            logger.info(
                f"LLM 预判：{len(urls)}/{len(candidates)} 条需抓取全文"
                + (f" (query={query!r})" if query else "")
            )
            return urls
        except Exception as e:
            logger.warning(f"LLM 预判失败，降级为全量抓取: {e}")
            return [c["url"] for c in candidates]

    # --------------------------------------------------------
    # 批量增强搜索结果
    # --------------------------------------------------------

    async def enrich_results(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
        llm_client: Optional[Any] = None,
        use_content_fetcher: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        异步批量抓取搜索结果的全文内容。

        抓取策略由 use_content_fetcher 和 llm_client 共同决定：

        - use_content_fetcher='force'：硬强制，跳过来源限制和 LLM 预判，对所有合格 URL 全量抓取。
        - use_content_fetcher='auto' / None（智能模式）且提供 llm_client：
            仅对 scholar/google 来源调用 LLM 预判，选择性抓取。
        - use_content_fetcher='auto' / None 且无 llm_client：对 scholar/google 来源全量抓取。

        成功抓取后：
        - 原始 content 保存到 metadata.original_snippet
        - 全文替换 hit["content"]
        - 标记 metadata.content_type = "full_text"

        Args:
            results:              UnifiedWebSearcher 返回的搜索结果列表
            query:                原始查询（用于日志与 LLM prompt）
            llm_client:           LLMManager 客户端，不为 None 时启用智能预判
            use_content_fetcher:  'force'=强制全量, 'auto'/None=智能, 'off' 由调用层拦截

        Returns:
            增强后的结果列表（原地修改）
        """
        if not results:
            return results

        # 智能模式 + LLM 可用：先预判，再选择性抓取
        urls_to_fetch: Optional[set] = None  # None 表示"全量抓取"
        if use_content_fetcher in (None, "auto") and llm_client is not None:
            fetching_urls = await self.evaluate_snippets_need_fetch(
                query or "", results, llm_client
            )
            urls_to_fetch = set(fetching_urls)

            # 对不需要抓取的合格 URL 打上 snippet_sufficient 标记
            for hit in results:
                metadata = hit.get("metadata") or {}
                url = (metadata.get("url") or "").strip()
                if (
                    url
                    and url not in urls_to_fetch
                    and is_qualifying_url(url, only_academic=self.only_academic)
                    and metadata.get("content_type") not in ("full_text",)
                ):
                    metadata["content_type"] = "snippet_sufficient"
                    hit["metadata"] = metadata

        is_force = use_content_fetcher == "force"

        sem = asyncio.Semaphore(self.max_concurrent)

        def _fetch_url_for_hit(metadata: Dict[str, Any]) -> Optional[str]:
            """有 DOI 时优先用 doi.org 链接拉取，否则用原始 url。"""
            url = (metadata.get("url") or "").strip()
            doi = (metadata.get("doi") or "").strip()
            if doi and "/" in doi and doi.startswith("10.") and len(doi) < 120:
                return "https://doi.org/" + doi
            return url if url else None

        async def _process_one(hit: Dict[str, Any]) -> None:
            metadata = hit.get("metadata") or {}
            source = metadata.get("source", "")
            url = (metadata.get("url") or "").strip()
            fetch_url = _fetch_url_for_hit(metadata)

            if not is_force and source not in ("scholar", "google", "serpapi_scholar", "serpapi_google"):
                return

            if not fetch_url or not is_qualifying_url(fetch_url, only_academic=self.only_academic):
                return

            # 已经是全文的跳过
            if metadata.get("content_type") == "full_text":
                return

            # 智能模式：跳过未被 LLM 选中的 URL（LLM 返回的可能是 doi_url 或原始 url）
            if urls_to_fetch is not None and url not in urls_to_fetch and fetch_url not in urls_to_fetch:
                return

            async with sem:
                try:
                    full_text = await self.fetch_content(fetch_url)
                except Exception as e:
                    logger.debug(f"抓取失败 {fetch_url}: {e}")
                    return

                if full_text:
                    original = (hit.get("content") or "").strip()
                    if original:
                        metadata["original_snippet"] = original
                    hit["content"] = full_text
                    metadata["content_type"] = "full_text"
                    hit["metadata"] = metadata

        tasks = [_process_one(hit) for hit in results]
        await asyncio.gather(*tasks, return_exceptions=True)

        enriched_count = sum(
            1 for h in results
            if (h.get("metadata") or {}).get("content_type") == "full_text"
        )
        sufficient_count = sum(
            1 for h in results
            if (h.get("metadata") or {}).get("content_type") == "snippet_sufficient"
        )
        logger.info(
            f"全文抓取: {enriched_count} 条全文 / {sufficient_count} 条摘要已足够 / "
            f"{len(results)} 条总计"
            + (f" (query={query!r})" if query else "")
        )

        return results

    def enrich_results_sync(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
        llm_client: Optional[Any] = None,
        use_content_fetcher: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        同步版本的 enrich_results，兼容 RetrievalService。
        """
        if not results or not self.enabled:
            return results

        coro = self.enrich_results(
            results,
            query=query,
            llm_client=llm_client,
            use_content_fetcher=use_content_fetcher,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
