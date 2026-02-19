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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup

from config.settings import settings
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)

# 浏览器复用：全局单例与空闲超时
_shared_browser_manager: Optional["_BrowserManager"] = None
_shared_browser_last_used: float = 0.0
_shared_browser_lock = asyncio.Lock()


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
                "extension_path": getattr(gs, "extension_path", "extra_tools/CapSolverExtension"),
                "headless": getattr(gs, "headless", None),
                "proxy": getattr(gs, "proxy", None),
                "timeout": getattr(gs, "timeout", 60000),
                "max_results": getattr(gs, "max_results", 5),
                "user_data_dir": getattr(gs, "user_data_dir", None),
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
        "extension_path": gs.get("extension_path", "extra_tools/CapSolverExtension"),
        "headless": gs.get("headless"),
        "proxy": gs.get("proxy"),
        "timeout": gs.get("timeout", 60000),
        "max_results": min(int(gs.get("max_results", 5)), 20),
        "user_data_dir": gs.get("user_data_dir"),
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
    """浏览器管理器，负责创建和管理浏览器实例"""
    
    def __init__(self, timeout: int = 120000):
        self.playwright = None
        self.browsers = []
        self.browser_pids = []  # 记录 chromium 进程 PID
        self.timeout = timeout
        self._playwright_lock = asyncio.Lock()
    
    async def close(self):
        """关闭所有浏览器实例：先关所有页面，再关 context，最后 stop playwright"""
        import signal
        
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
        
        # 自动检测显示模式
        if headless is None:
            use_headed, display_mode = _ensure_display()
            headless = not use_headed
            logger.info(f"显示模式: {display_mode}, headless={headless}")
        
        os.makedirs(user_data_dir, exist_ok=True)
        
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
        """从 URL 中提取 DOI"""
        if not url:
            return None
        
        match = re.search(r'(?:doi\.org|dx\.doi\.org)/+(10\.\d{4,}/[^\s&?#]+)', url, re.I)
        if match:
            return match.group(1).rstrip('/.')
        
        match = re.search(r'/(10\.\d{4,}/[a-zA-Z0-9._\-/()]+)', url)
        if match:
            doi = match.group(1).rstrip('/.')
            if '/' in doi and 10 < len(doi) < 100:
                return re.sub(r'[/.\-]+$', '', doi)
        
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
    
    # 默认分数
    SCHOLAR_DEFAULT_SCORE = 0.8
    GOOGLE_DEFAULT_SCORE = 0.6
    
    def __post_init__(self):
        if not self._config:
            self._config = _get_google_search_config()
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
        """获取扩展路径"""
        ext_path = self._config.get("extension_path", "extra_tools/CapSolverExtension")
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
    ) -> List[Dict[str, Any]]:
        """
        搜索 Google Scholar
        
        Args:
            query: 搜索查询
            limit: 最大结果数
            year_start: 开始年份
            year_end: 结束年份
        
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
        reuse_browser = getattr(getattr(settings, "perf_google_search", None), "browser_reuse", True)

        try:
            shared_mgr, shared_ctx = await _get_or_create_shared_browser_manager(
                timeout=timeout,
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                extension_path=self._get_extension_path(),
            )
            if shared_mgr is not None and shared_ctx is not None:
                browser_manager = shared_mgr
                context = shared_ctx
            else:
                browser_manager = _BrowserManager(timeout=timeout)
                context = await browser_manager.launch_persistent_browser(
                    user_data_dir=self._get_user_data_dir(),
                    headless=headless,
                    proxy=proxy,
                    stealth_mode=True,
                    extension_path=self._get_extension_path()
                )

            page = await context.new_page()
            page.set_default_timeout(timeout)
            await browser_manager._apply_stealth_mode(page)

            # 通过搜索框输入方式搜索（避免 URL 携带搜索词触发封控）
            await self._navigate_scholar_via_searchbox(
                page, query, year_start, year_end, timeout
            )
            
            # 检查验证码
            if await self._check_captcha(page):
                logger.warning("检测到验证码")
                display_mode = _get_display_mode()
                is_headed = display_mode in ("real", "virtual") or headless == False
                
                if is_headed:
                    logger.info("有头模式，等待验证码解决...")
                    await self._inject_capsolver_attributes(page)
                    
                    for i in range(90):
                        await asyncio.sleep(1)
                        if not await self._check_captcha(page):
                            logger.info("验证码已解决")
                            try:
                                await page.wait_for_load_state("domcontentloaded", timeout=15000)
                                await page.wait_for_selector('.gs_r', timeout=10000)
                            except Exception:
                                pass
                            break
                        if i > 0 and i % 15 == 0:
                            await self._inject_capsolver_attributes(page)
                    
                    if not await self._check_results(page, '.gs_r'):
                        logger.error("验证码处理后仍无结果")
                        return results
                else:
                    logger.error("无头模式下无法处理验证码")
                    return results
            
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
            if browser_manager and not reuse_browser:
                await browser_manager.close()
            elif browser_manager and reuse_browser:
                global _shared_browser_last_used
                _shared_browser_last_used = time.monotonic()

    async def search_scholar_batch(
        self,
        queries: List[str],
        limit_per_query: int = 5,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量搜索 Google Scholar（串行执行）
        
        在同一个浏览器实例中依次执行多个查询，每个查询返回 limit_per_query 条结果。
        搜索完成后关闭浏览器。
        
        Args:
            queries: 查询列表
            limit_per_query: 每个查询的最大结果数
            year_start: 开始年份
            year_end: 结束年份
        
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
            logger.info(f"Scholar 批量搜索：所有 {len(queries)} 个查询都命中缓存")
            return all_results
        
        logger.info(f"Scholar 批量搜索：{len(queries_to_search)} 个查询待执行，{len(queries) - len(queries_to_search)} 个命中缓存")
        
        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()
        
        browser_manager = None
        context = None
        page = None
        
        try:
            # 启动浏览器（只启动一次）
            browser_manager = _BrowserManager(timeout=timeout)
            context = await browser_manager.launch_persistent_browser(
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                stealth_mode=True,
                extension_path=self._get_extension_path()
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
                    
                    # 检查验证码
                    if await self._check_captcha(page):
                        logger.warning(f"检测到验证码 (query={query!r})")
                        display_mode = _get_display_mode()
                        is_headed = display_mode in ("real", "virtual") or headless == False
                        
                        if is_headed:
                            await self._inject_capsolver_attributes(page)
                            for i in range(90):
                                await asyncio.sleep(1)
                                if not await self._check_captcha(page):
                                    logger.info("验证码已解决")
                                    try:
                                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                                        await page.wait_for_selector('.gs_r', timeout=10000)
                                    except Exception:
                                        pass
                                    break
                                if i > 0 and i % 15 == 0:
                                    await self._inject_capsolver_attributes(page)
                            
                            if not await self._check_results(page, '.gs_r'):
                                logger.error(f"验证码处理后仍无结果 (query={query!r})")
                                continue
                        else:
                            logger.error(f"无头模式下无法处理验证码 (query={query!r})")
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
            
            logger.info(f"Scholar 批量搜索完成，共 {len(all_results)} 条结果")
            return all_results
        
        except Exception as e:
            logger.error(f"Scholar 批量搜索出错: {e}")
            return all_results
        finally:
            # 关闭页面
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            # 关闭浏览器（批量搜索完成后始终关闭）
            if browser_manager:
                try:
                    await browser_manager.close()
                    logger.info("Scholar 批量搜索：浏览器已关闭")
                except Exception as e:
                    logger.warning(f"关闭浏览器时出错: {e}")

    async def search_google(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索 Google
        
        Args:
            query: 搜索查询
            limit: 最大结果数
        
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
        reuse_browser = getattr(getattr(settings, "perf_google_search", None), "browser_reuse", True)

        try:
            shared_mgr, shared_ctx = await _get_or_create_shared_browser_manager(
                timeout=timeout,
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                extension_path=self._get_extension_path(),
            )
            if shared_mgr is not None and shared_ctx is not None:
                browser_manager = shared_mgr
                context = shared_ctx
            else:
                browser_manager = _BrowserManager(timeout=timeout)
                context = await browser_manager.launch_persistent_browser(
                    user_data_dir=self._get_user_data_dir(),
                    headless=headless,
                    proxy=proxy,
                    stealth_mode=True,
                    extension_path=self._get_extension_path()
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
            if await self._check_captcha(page):
                logger.warning("检测到验证码")
                display_mode = _get_display_mode()
                is_headed = display_mode in ("real", "virtual") or headless == False
                
                if is_headed:
                    logger.info("有头模式，等待验证码解决...")
                    for i in range(90):
                        await asyncio.sleep(1)
                        if not await self._check_captcha(page):
                            logger.info("验证码已解决")
                            break
                    
                    if not await self._check_results(page, 'div.g'):
                        logger.error("验证码处理后仍无结果")
                        return results
                else:
                    logger.error("无头模式下无法处理验证码")
                    return results
            
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
            if browser_manager and not reuse_browser:
                await browser_manager.close()
            elif browser_manager and reuse_browser:
                global _shared_browser_last_used
                _shared_browser_last_used = time.monotonic()

    async def search_google_batch(
        self,
        queries: List[str],
        limit_per_query: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        批量搜索 Google（串行执行）
        
        在同一个浏览器实例中依次执行多个查询，每个查询返回 limit_per_query 条结果。
        搜索完成后关闭浏览器。
        
        Args:
            queries: 查询列表
            limit_per_query: 每个查询的最大结果数
        
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
            logger.info(f"Google 批量搜索：所有 {len(queries)} 个查询都命中缓存")
            return all_results
        
        logger.info(f"Google 批量搜索：{len(queries_to_search)} 个查询待执行，{len(queries) - len(queries_to_search)} 个命中缓存")
        
        timeout = self._config.get("timeout", 60000)
        headless = self._config.get("headless")
        proxy = self._config.get("proxy")
        user_data_dir = self._get_user_data_dir()
        
        browser_manager = None
        context = None
        page = None
        
        try:
            # 启动浏览器（只启动一次）
            browser_manager = _BrowserManager(timeout=timeout)
            context = await browser_manager.launch_persistent_browser(
                user_data_dir=user_data_dir,
                headless=headless,
                proxy=proxy,
                stealth_mode=True,
                extension_path=self._get_extension_path()
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
                    if await self._check_captcha(page):
                        logger.warning(f"检测到验证码 (query={query!r})")
                        display_mode = _get_display_mode()
                        is_headed = display_mode in ("real", "virtual") or headless == False
                        
                        if is_headed:
                            for i in range(90):
                                await asyncio.sleep(1)
                                if not await self._check_captcha(page):
                                    logger.info("验证码已解决")
                                    break
                            
                            if not await self._check_results(page, 'div.g'):
                                logger.error(f"验证码处理后仍无结果 (query={query!r})")
                                continue
                        else:
                            logger.error(f"无头模式下无法处理验证码 (query={query!r})")
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
            
            logger.info(f"Google 批量搜索完成，共 {len(all_results)} 条结果")
            return all_results
        
        except Exception as e:
            logger.error(f"Google 批量搜索出错: {e}")
            return all_results
        finally:
            # 关闭页面
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            # 关闭浏览器（批量搜索完成后始终关闭）
            if browser_manager:
                try:
                    await browser_manager.close()
                    logger.info("Google 批量搜索：浏览器已关闭")
                except Exception as e:
                    logger.warning(f"关闭浏览器时出错: {e}")

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
                "doc_id": result.get('title', ''),
                "title": result.get('title', ''),
                "url": url,
                "domain": domain,
                "search_query": query
            }
        }
    
    async def _check_captcha(self, page) -> bool:
        """检查是否有验证码"""
        try:
            selector = "form#captcha-form, #gs_captcha_ccl, div.g-recaptcha, #recaptcha"
            return await page.query_selector(selector) is not None
        except Exception:
            return False
    
    async def _check_results(self, page, selector: str) -> bool:
        """检查是否有搜索结果"""
        try:
            return await page.query_selector(selector) is not None
        except Exception:
            return False
    
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
    """清理全局共享的浏览器实例。在程序退出或测试完成后调用。"""
    global _shared_browser_manager, _shared_browser_last_used
    async with _shared_browser_lock:
        if _shared_browser_manager is not None:
            try:
                await _shared_browser_manager.close()
                logger.info("共享浏览器实例已关闭")
            except Exception as e:
                logger.warning(f"关闭共享浏览器时出错: {e}")
            finally:
                _shared_browser_manager = None
                _shared_browser_last_used = 0


def cleanup_shared_browser_sync():
    """同步版本的清理函数（在非异步上下文中使用）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(cleanup_shared_browser())
        else:
            loop.run_until_complete(cleanup_shared_browser())
    except RuntimeError:
        # 没有事件循环，创建新的
        asyncio.run(cleanup_shared_browser())


# 注册退出时自动清理
import atexit
atexit.register(cleanup_shared_browser_sync)


# 全局单例
google_searcher = GoogleSearcher()
