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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)

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

    三级提取策略自动降级：trafilatura → BrightData → Playwright
    """

    enabled: bool = True
    only_academic: bool = False
    max_content_length: int = 8000
    timeout_seconds: int = 15
    brightdata_api_key: str = ""
    brightdata_zone: str = ""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent: int = 5
    _cache: Optional[TTLCache] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self.cache_enabled:
            self._cache = get_cache(
                enabled=True,
                ttl_seconds=self.cache_ttl_seconds,
                maxsize=512,
                prefix="content_fetcher",
            )

    @classmethod
    def from_settings(cls) -> "WebContentFetcher":
        """从 config/settings.py 全局 settings 加载配置"""
        try:
            from config.settings import settings
            cfg = getattr(settings, "content_fetcher", None)
            if cfg is not None:
                return cls(
                    enabled=getattr(cfg, "enabled", False),
                    only_academic=getattr(cfg, "only_academic", False),
                    max_content_length=getattr(cfg, "max_content_length", 8000),
                    timeout_seconds=getattr(cfg, "timeout_seconds", 15),
                    brightdata_api_key=getattr(cfg, "brightdata_api_key", ""),
                    brightdata_zone=getattr(cfg, "brightdata_zone", ""),
                    cache_enabled=getattr(cfg, "cache_enabled", True),
                    cache_ttl_seconds=getattr(cfg, "cache_ttl_seconds", 3600),
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
                timeout=self.timeout_seconds,
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
    # 第二级：BrightData Web Unlocker
    # --------------------------------------------------------

    async def _fetch_brightdata(self, url: str) -> Optional[str]:
        """
        使用 BrightData Web Unlocker API 抓取。

        适合反爬站点，需要配置 brightdata_api_key。
        """
        if not self.brightdata_api_key:
            return None

        try:
            import aiohttp
            import trafilatura

            zone = self.brightdata_zone or "web_unlocker1"
            proxy_url = f"https://brd-customer-{self.brightdata_api_key}-zone-{zone}:@brd.superproxy.io:33335"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
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
    # 第三级：Playwright 浏览器
    # --------------------------------------------------------

    async def _fetch_playwright(self, url: str) -> Optional[str]:
        """
        使用 Playwright headless 浏览器抓取。

        支持 JS 渲染页面，含 stealth 反检测。
        浏览器实例临时创建，不复用 google_search.py 的全局浏览器。
        """
        try:
            from playwright.async_api import async_playwright

            # 可选 stealth 插件
            stealth_fn = None
            try:
                from playwright_stealth import stealth_async
                stealth_fn = stealth_async
            except ImportError:
                pass

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        viewport={"width": 1280, "height": 800},
                    )
                    page = await context.new_page()

                    if stealth_fn:
                        await stealth_fn(page)

                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_seconds * 1000,
                    )

                    # 等待正文加载
                    await page.wait_for_timeout(2000)

                    html = await page.content()
                    await context.close()
                finally:
                    await browser.close()

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

        trafilatura → BrightData → Playwright

        Args:
            url: 目标 URL

        Returns:
            提取的正文文本，失败返回 None
        """
        # 检查缓存
        if self._cache:
            cache_key = _make_key("content", url)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"缓存命中: {url}")
                return cached

        # 第一级：trafilatura
        text = await self._fetch_trafilatura(url)

        # 第二级：BrightData
        if text is None:
            text = await self._fetch_brightdata(url)

        # 第三级：Playwright
        if text is None:
            text = await self._fetch_playwright(url)

        # 截断到最大长度
        if text and len(text) > self.max_content_length:
            text = text[: self.max_content_length]

        # 写入缓存
        if text and self._cache:
            cache_key = _make_key("content", url)
            self._cache.set(cache_key, text)

        return text

    # --------------------------------------------------------
    # 批量增强搜索结果
    # --------------------------------------------------------

    async def enrich_results(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        异步批量抓取搜索结果的全文内容。

        对每个合格的 URL 调用 fetch_content，成功则：
        - 原始 content 保存到 metadata.original_snippet
        - 全文替换 hit["content"]
        - 标记 metadata.content_type = "full_text"

        Args:
            results: UnifiedWebSearcher 返回的搜索结果列表
            query: 原始查询（用于日志）

        Returns:
            增强后的结果列表（原地修改）
        """
        if not results:
            return results

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _process_one(hit: Dict[str, Any]) -> None:
            metadata = hit.get("metadata") or {}
            url = (metadata.get("url") or "").strip()

            if not url or not is_qualifying_url(url, only_academic=self.only_academic):
                return

            # 已经是全文的跳过
            if metadata.get("content_type") == "full_text":
                return

            async with sem:
                try:
                    full_text = await self.fetch_content(url)
                except Exception as e:
                    logger.debug(f"抓取失败 {url}: {e}")
                    return

                if full_text:
                    # 保留原始片段
                    original = (hit.get("content") or "").strip()
                    if original:
                        metadata["original_snippet"] = original
                    # 替换为全文
                    hit["content"] = full_text
                    metadata["content_type"] = "full_text"
                    hit["metadata"] = metadata

        tasks = [_process_one(hit) for hit in results]
        await asyncio.gather(*tasks, return_exceptions=True)

        enriched_count = sum(
            1 for h in results
            if (h.get("metadata") or {}).get("content_type") == "full_text"
        )
        logger.info(
            f"全文抓取: {enriched_count}/{len(results)} 条成功"
            + (f" (query={query!r})" if query else "")
        )

        return results

    def enrich_results_sync(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        同步版本的 enrich_results，兼容 RetrievalService。
        """
        if not results or not self.enabled:
            return results

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.enrich_results(results, query=query),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.enrich_results(results, query=query)
                )
        except RuntimeError:
            return asyncio.run(self.enrich_results(results, query=query))
