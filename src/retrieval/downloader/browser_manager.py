import asyncio
from typing import Optional, Tuple, Dict, Any, List, Union
from playwright.async_api import async_playwright, Browser, Playwright, BrowserContext, Page
from .log_utils import logger
import platform
import random
import os
import json
import tempfile

# 导入显示器管理模块
from .display_manager import ensure_display, get_display_mode, should_use_headed_mode
from src.retrieval.browser_service import SharedBrowserService, get_cdp_context_options

# 尝试导入 playwright-stealth，如果失败则使用备用方案
#
# 兼容说明：
# - playwright-stealth v2.x: 导出 Stealth 类（apply_stealth_async/apply_stealth_sync）
# - 少数旧版本：可能存在 stealth_async / stealth_sync
STEALTH_AVAILABLE: str = "none"  # "v2" | "async" | "sync" | "none"
Stealth = None
stealth_async = None
stealth_sync = None

try:
    # v2.x API（已验证：2.0.0）
    from playwright_stealth import Stealth as _Stealth
    Stealth = _Stealth
    STEALTH_AVAILABLE = "v2"
except Exception:
    try:
        # 历史版本 API（部分版本可能提供）
        from playwright_stealth import stealth_async as _stealth_async
        stealth_async = _stealth_async
        STEALTH_AVAILABLE = "async"
    except Exception:
        try:
            from playwright_stealth import stealth_sync as _stealth_sync
            stealth_sync = _stealth_sync
            STEALTH_AVAILABLE = "sync"
        except Exception:
            STEALTH_AVAILABLE = "none"
            logger.warning("playwright-stealth 未安装或 API 不兼容，隐身模式将使用基础脚本")

# 随机用户代理列表（包含不同浏览器和操作系统）
USER_AGENTS = [
    # Chrome Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
]

# 国内主要城市地理坐标
CITY_GEOLOCATIONS = [
    {"latitude": 39.9042, "longitude": 116.4074},  # 北京
    {"latitude": 31.2304, "longitude": 121.4737},  # 上海
    {"latitude": 23.1291, "longitude": 113.2644},  # 广州
    {"latitude": 30.5728, "longitude": 104.0668}   # 成都
]

def get_random_user_agent() -> str:
    """获取随机用户代理"""
    return random.choice(USER_AGENTS)

def get_random_viewport(base_width=1280, base_height=720) -> dict:
    """生成随机视口尺寸，在基础尺寸上±50像素"""
    return {
        "width": base_width + random.randint(-50, 50),
        "height": base_height + random.randint(-50, 50)
    }

def get_random_geolocation() -> dict:
    """获取随机地理位置"""
    return random.choice(CITY_GEOLOCATIONS)

# 修改原有常量定义
DEFAULT_USER_AGENT = get_random_user_agent()
DEFAULT_VIEWPORT = get_random_viewport()
DEFAULT_GEOLOCATION = get_random_geolocation()
DEFAULT_LOCALE = random.choice(["zh-CN", "zh-TW", "en-US"])  # 增加语言随机
DEFAULT_TIMEZONE = random.choice(["Asia/Shanghai", "Asia/Chongqing"])  # 国内时区

class BrowserManager:
    """浏览器管理器，负责创建和管理浏览器实例"""
    
    def __init__(self,timeout:int=120000):
        self.playwright = None
        self.browsers = []
        self.timeout = 120000
        self._temp_files = []  # 用于跟踪需要清理的临时文件
        self._playwright_lock = asyncio.Lock()  # 保护 Playwright 初始化的锁
        self._ui_action_lock = asyncio.Lock()  # 有头模式下串行化拟人化交互
        self._cdp_browser_handle = None  # CDP 连接到共享浏览器（不拥有进程）
        
    async def close(self, force_cleanup: bool = False):
        """关闭所有浏览器实例和Playwright
        
        Args:
            force_cleanup: 是否强制清理残留进程（默认 False，避免触发崩溃报告）
        """
        logger.info(f"开始关闭 {len(self.browsers)} 个浏览器实例...")
        
        for i, browser in enumerate(self.browsers):
            try:
                logger.info(f"关闭浏览器实例 {i + 1}/{len(self.browsers)}...")
                await asyncio.wait_for(browser.close(), timeout=10.0)
                logger.info(f"浏览器实例 {i + 1} 已关闭")
                # 等待 Chrome 完成内部清理，避免崩溃
                await asyncio.sleep(1.0)
            except asyncio.TimeoutError:
                logger.warning(f"关闭浏览器实例 {i + 1} 超时")
            except Exception as e:
                # 关闭时的异常通常可以忽略（资源正在释放）
                logger.debug(f"关闭浏览器时出错（可忽略）: {e}")
        
        # 在停止 Playwright 前等待，让浏览器进程优雅退出
        if self.browsers:
            await asyncio.sleep(0.5)
                
        if self.playwright:
            try:
                logger.info("停止 Playwright...")
                await asyncio.wait_for(self.playwright.stop(), timeout=10.0)
                self.playwright = None
                logger.info("Playwright 已停止")
            except asyncio.TimeoutError:
                logger.warning("停止 Playwright 超时")
                self.playwright = None
            except Exception as e:
                # 停止时的异常通常可以忽略
                logger.debug(f"停止Playwright时出错（可忽略）: {e}")
                self.playwright = None
        
        # 清理临时文件
        self._cleanup_temp_files()
                
        self.browsers = []
        self._cdp_browser_handle = None
        
        # 仅在明确需要时才强制清理（默认不调用，避免 pkill 触发崩溃报告）
        if force_cleanup:
            await self.force_cleanup()
    
    async def force_cleanup(self):
        """强制清理所有残留的 Chromium/Chrome 进程"""
        if self._cdp_browser_handle is not None:
            logger.info("检测到共享 CDP 浏览器连接，跳过 force_cleanup")
            return

        import subprocess
        import platform
        
        system = platform.system()
        logger.info(f"强制清理残留浏览器进程 (系统: {system})...")
        
        try:
            if system == "Darwin" or system == "Linux":
                # macOS 和 Linux 使用 pkill
                # 只杀死由 playwright 启动的进程（包含特定路径）
                kill_commands = [
                    ['pkill', '-f', 'playwright.*chromium'],
                    ['pkill', '-f', 'playwright.*chrome'],
                    ['pkill', '-f', '.cache/ms-playwright'],
                ]
                for cmd in kill_commands:
                    try:
                        result = subprocess.run(cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            logger.info(f"已终止进程: {' '.join(cmd)}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"执行 {' '.join(cmd)} 超时")
                    except Exception as e:
                        logger.debug(f"执行 {' '.join(cmd)} 失败: {e}")
                        
            elif system == "Windows":
                # Windows 使用 taskkill
                kill_commands = [
                    ['taskkill', '/F', '/IM', 'chromium.exe'],
                    ['taskkill', '/F', '/IM', 'chrome.exe'],
                ]
                for cmd in kill_commands:
                    try:
                        result = subprocess.run(cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            logger.info(f"已终止进程: {' '.join(cmd)}")
                    except Exception as e:
                        logger.debug(f"执行 {' '.join(cmd)} 失败: {e}")
                        
            logger.info("强制清理完成")
        except Exception as e:
            logger.warning(f"强制清理进程时出错: {e}")
    
    def _cleanup_temp_files(self):
        """清理所有临时文件"""
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.error(f"删除临时文件时出错: {file_path}, {e}")
        
        # 清空列表
        self._temp_files = []
        
    def create_pdf_preferences_file(self, always_open_externally=True) -> str:
        """
        创建包含PDF首选项的临时文件
        
        Args:
            always_open_externally: 是否总是外部打开PDF
            
        Returns:
            str: 临时文件路径
        """
        # 创建临时首选项文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            preferences = {
                "plugins": {
                    "always_open_pdf_externally": always_open_externally
                }
            }
            json.dump(preferences, f)
            prefs_file_path = f.name
            
        # 添加到需要清理的文件列表
        self._temp_files.append(prefs_file_path)
        logger.info(f"已创建PDF首选项文件: {prefs_file_path}")
        
        return prefs_file_path
        
    async def _ensure_playwright(self):
        """确保Playwright已初始化（线程安全）"""
        async with self._playwright_lock:
            if not self.playwright:
                self.playwright = await async_playwright().start()

    async def execute_with_ui_lock(self, coro):
        """在有头模式下串行执行拟人化交互，避免焦点抢占。"""
        async with self._ui_action_lock:
            return await coro
            
    async def launch_browser(self, 
                           browser_type: str = "chromium", 
                           headless: Optional[bool] = None, 
                           proxy: Optional[str] = None,
                           stealth_mode: bool = False,
                           always_download_pdf: bool = False,
                           reuse_shared_cdp: bool = True,
                           **kwargs) -> Browser:
        """
        启动浏览器实例
        
        Args:
            browser_type: 浏览器类型，可选 "chromium"、"chrome"、"firefox"、"webkit"
            headless: 是否使用无头模式
                - None: 自动检测（推荐，支持 capsolver 扩展）
                - True: 强制无头模式（capsolver 可能无法工作）
                - False: 强制有头模式
            proxy: 代理服务器地址
            stealth_mode: 是否启用隐身模式
            always_download_pdf: 是否总是下载PDF而不是在浏览器中打开
            **kwargs: 其他浏览器配置选项
            
        Returns:
            Browser: 浏览器实例
        """
        await self._ensure_playwright()

        logger.info(
            "[headed-diag] launch_browser: browser_type=%r, headless=%s, proxy=%s, "
            "stealth_mode=%s, reuse_shared_cdp=%s, always_download_pdf=%s, kwargs_keys=%s",
            browser_type,
            headless,
            proxy,
            stealth_mode,
            reuse_shared_cdp,
            always_download_pdf,
            list(kwargs.keys()) if kwargs else [],
        )

        # 先做自动检测，再判断 CDP，避免 headless=None 被误判为“非 False”走错分支
        if headless is None:
            use_headed, display_mode = ensure_display()
            headless = not use_headed
            logger.info(f"自动检测显示模式: {display_mode}, headless={headless}")

        # 按需复用共享 CDP 浏览器。显式禁用时始终本地启动，适合需要真实可见窗口
        # 或依赖 user_data_dir / persistent context 语义的场景。
        if reuse_shared_cdp:
            headed_slot_id = kwargs.get("headed_slot_id")
            if headless:
                cdp_url = SharedBrowserService.get_cdp_url_headless()
            else:
                cdp_url = SharedBrowserService.get_cdp_url_headed(slot_id=headed_slot_id)
            if cdp_url and browser_type in ("chromium", "chrome"):
                logger.info(f"连接共享浏览器 CDP: {cdp_url}")
                browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
                self._cdp_browser_handle = browser
                return browser

        context_options = {}
        
        # 基本启动选项
        launch_options = {
            "headless": headless,
            **kwargs
        }
        
        # 如果需要总是下载PDF
        if always_download_pdf and "chrom" in browser_type:
            prefs_file_path = self.create_pdf_preferences_file(True)
            args = launch_options.get("args", [])
            args.append(f'--initial-preferences-file="{prefs_file_path}"')
            launch_options["args"] = args
            logger.info("已设置总是外部打开PDF")
        
        # 添加代理配置
        if proxy:
            logger.info(f"使用代理: {proxy}")
            
            # 处理 socks5h 协议
            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
                logger.info(f"已将代理格式从socks5h://转换为socks5://: {proxy}")
            
            # 解析代理配置
            proxy_config = {
                "server": proxy
            }
            
            # 如果有用户名密码，添加到配置中
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
                            logger.info(f"已提取代理用户名和密码")
                except Exception as e:
                    logger.error(f"解析代理认证信息失败: {e}")
            
            context_options["proxy"] = proxy_config
            
            # Firefox特殊处理
            if browser_type == "firefox":
                # Firefox在使用代理时的特殊设置
                firefox_args = context_options.get("args", [])
                firefox_args.extend([
                    "-purgecaches",  # 清除缓存
                ])
                context_options["args"] = firefox_args
                
                # 添加Firefox代理特殊设置
                if "firefox_user_prefs" not in context_options:
                    context_options["firefox_user_prefs"] = {}
                
                context_options["firefox_user_prefs"].update({
                    "network.proxy.socks_remote_dns": True,  # 使用SOCKS代理进行DNS解析
                    "network.proxy.no_proxies_on": "",       # 不跳过任何域名
                    "network.dns.disablePrefetch": True,     # 禁用DNS预取
                    "network.proxy.type": 1                  # 使用手动代理配置
                })
                logger.info("已应用Firefox的代理配置")
        
        logger.info(
            "[headed-diag] launch_browser: launch_options passed to Playwright: headless=%s, args=%s, keys=%s",
            launch_options.get("headless"),
            launch_options.get("args", []),
            list(launch_options.keys()),
        )
        
        # 根据浏览器类型选择启动方法
        # 注意：Chrome 137+ 已移除扩展加载支持，建议使用 Playwright 内置 Chromium
        if browser_type == "chrome":
            # 使用 Playwright 内置 Chromium（不使用系统 Chrome）
            # 因为 Chrome 137+ 已移除 --load-extension 标志
            logger.info("使用 Playwright 内置 Chromium（推荐，支持扩展加载）")
            browser = await self.playwright.chromium.launch(**launch_options)
        elif browser_type == "firefox":
            browser = await self.playwright.firefox.launch(**launch_options)
        elif browser_type == "webkit":
            browser = await self.playwright.webkit.launch(**launch_options)
        else:
            # 默认使用 Playwright 内置 Chromium
            browser = await self.playwright.chromium.launch(**launch_options)
        
        logger.info(
            "[headed-diag] Playwright launch returned: type=%s, launch_options.headless=%s, launch_options.args=%s",
            browser_type,
            launch_options.get("headless"),
            launch_options.get("args", []),
        )
            
        self.browsers.append(browser)
        return browser
        
    async def create_context(self, 
                           browser: Browser,
                           user_agent: Optional[str] = None,
                           viewport: Optional[Dict[str, int]] = None,
                           user_data_dir: Optional[str] = None,
                           downloads_path: Optional[str] = None,
                           **kwargs) -> BrowserContext:
        """
        创建浏览器上下文
        
        Args:
            browser: 浏览器实例
            user_agent: 用户代理
            viewport: 视口大小
            user_data_dir: 用户数据目录
            downloads_path: 下载路径
            **kwargs: 其他上下文配置选项
            
        Returns:
            BrowserContext: 浏览器上下文
        """
        # 设置用户代理
        final_user_agent = user_agent or get_random_user_agent()
        
        # 设置视口大小
        final_viewport = viewport or get_random_viewport()
        
        # 设置地理位置
        final_geolocation = get_random_geolocation()
        
        context_options = {
            **kwargs,
            "user_agent": final_user_agent,
            "viewport": final_viewport,
            "locale": DEFAULT_LOCALE,
            "timezone_id": DEFAULT_TIMEZONE,
            "geolocation": final_geolocation,
            "color_scheme": "light",
            "reduced_motion": "no-preference",
            "has_touch": False
        }
        
        # 设置下载路径
        if downloads_path:
            context_options["accept_downloads"] = True
            os.makedirs(downloads_path, exist_ok=True)
            logger.info(f"设置下载路径: {downloads_path}")
        
        # 创建上下文
        context = await browser.new_context(**context_options)
        return context
        
    async def launch_persistent_browser(self,
                                      user_data_dir: str,
                                      browser_type: str = "chrome",
                                      headless: Optional[bool] = None,
                                      proxy: Optional[str] = None,
                                      extension_path: Optional[str] = None,
                                      stealth_mode: bool = True,
                                      user_agent: Optional[str] = None,
                                      viewport: Optional[Dict[str, int]] = None,
                                      timeout: int = 120000,
                                      always_download_pdf: bool = True,
                                      downloads_path: Optional[str] = None,
                                      reuse_shared_cdp: bool = True,
                                      **kwargs) -> BrowserContext:
        """
        启动持久化浏览器实例
        
        Args:
            user_data_dir: 用户数据目录
            browser_type: 浏览器类型
            headless: 是否使用无头模式
                - None: 自动检测（推荐，支持 capsolver 扩展）
                - True: 强制无头模式（capsolver 可能无法工作）
                - False: 强制有头模式
            proxy: 代理服务器地址
            extension_path: Capsolver 等扩展目录路径，不传则使用默认 extra_tools/CapSolverExtension
            stealth_mode: 是否启用隐身模式
            user_agent: 用户代理
            viewport: 视口大小
            timeout: 超时时间
            always_download_pdf: 是否总是下载PDF而不是在浏览器中打开
            downloads_path: 下载文件保存路径（会设置为浏览器的默认下载目录）
            **kwargs: 其他配置选项
            
        Returns:
            BrowserContext: 浏览器上下文
        """
        await self._ensure_playwright()

        logger.info(
            "[headed-diag] launch_persistent_browser: user_data_dir=%r, browser_type=%r, "
            "headless=%s, reuse_shared_cdp=%s, stealth_mode=%s, extension_path=%s, "
            "downloads_path=%s, timeout=%s, kwargs_keys=%s",
            user_data_dir,
            browser_type,
            headless,
            reuse_shared_cdp,
            stealth_mode,
            extension_path,
            downloads_path,
            timeout,
            list(kwargs.keys()) if kwargs else [],
        )

        # 先做自动检测，再判断 CDP，避免 headless=None 被误判为“非 False”走错分支
        if headless is None:
            use_headed, display_mode = ensure_display()
            headless = not use_headed
            logger.info(f"自动检测显示模式: {display_mode}, headless={headless}")

        # 对 persistent context，显式允许时优先复用共享 CDP（有头/无头双端口）；无 CDP 时回退到本地 launch。
        if reuse_shared_cdp:
            headed_slot_id = kwargs.get("headed_slot_id")
            if headless:
                cdp_url = SharedBrowserService.get_cdp_url_headless()
            else:
                cdp_url = SharedBrowserService.get_cdp_url_headed(slot_id=headed_slot_id)
            if cdp_url and browser_type in ("chromium", "chrome"):
                logger.info(f"通过 CDP 复用共享浏览器: {cdp_url}")
                if user_data_dir:
                    logger.info("CDP 模式不使用 user_data_dir 持久化配置，改为独立 context 隔离")

                browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
                self._cdp_browser_handle = browser
                context_options_cdp = get_cdp_context_options(
                    user_agent=user_agent,
                    viewport=viewport,
                    accept_downloads=True,
                )
                if downloads_path:
                    os.makedirs(downloads_path, exist_ok=True)
                    logger.info(f"CDP context 将通过 download.save_as 使用下载目录: {downloads_path}")
                context = await browser.new_context(**context_options_cdp)
                return context
        
        # 确保用户数据目录存在
        os.makedirs(user_data_dir, exist_ok=True)
        
        # 准备上下文选项
        context_options = {
            "headless": headless,
            "accept_downloads": True,  # 关键：启用下载功能
            **kwargs
        }
        
        # 如果指定了下载路径，设置到 context_options
        if downloads_path:
            os.makedirs(downloads_path, exist_ok=True)
            logger.info(f"已准备下载目录: {downloads_path}")
        
        # 如果需要总是下载PDF或设置下载路径
        if (always_download_pdf or downloads_path) and "chrom" in browser_type:
            # 对于持久化上下文，在用户数据目录中创建/更新首选项文件。
            # 必须包含 Chromium 期望的最小结构，否则有头模式会弹出「系统无法读取您的偏好设置」。
            _MINIMAL_PREFS = {
                "profile": {"exit_type": "Normal", "name": "Default"},
                "browser": {"check_default_browser": False},
            }
            default_dir = os.path.join(user_data_dir, "Default")
            os.makedirs(default_dir, exist_ok=True)
            preferences_path = os.path.join(default_dir, "Preferences")
            if os.path.exists(preferences_path):
                try:
                    with open(preferences_path, "r", encoding="utf-8") as f:
                        preferences = json.load(f)
                except Exception as e:
                    logger.warning("读取现有首选项文件失败: %s，将使用最小有效结构", e)
                    preferences = {}
            else:
                preferences = {}
            for key, value in _MINIMAL_PREFS.items():
                if key not in preferences:
                    preferences[key] = dict(value) if isinstance(value, dict) else value
                elif isinstance(value, dict) and isinstance(preferences.get(key), dict):
                    for k, v in value.items():
                        if k not in preferences[key]:
                            preferences[key][k] = v
            if "plugins" not in preferences:
                preferences["plugins"] = {}
            if "download" not in preferences:
                preferences["download"] = {}
            if always_download_pdf:
                preferences["plugins"]["always_open_pdf_externally"] = True
            if downloads_path:
                os.makedirs(downloads_path, exist_ok=True)
                abs_downloads_path = os.path.abspath(downloads_path)
                preferences["download"]["default_directory"] = abs_downloads_path
                preferences["download"]["prompt_for_download"] = False
                preferences["savefile"] = {"default_directory": abs_downloads_path}
                logger.info(f"已设置浏览器默认下载目录: {abs_downloads_path}")
            with open(preferences_path, "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            logger.info("已更新浏览器首选项: %s", preferences_path)
        
        # 添加代理配置
        if proxy:
            logger.info(f"使用代理: {proxy}")
            
            # 处理 socks5h 协议
            if proxy.startswith("socks5h://"):
                proxy = proxy.replace("socks5h://", "socks5://")
                logger.info(f"已将代理格式从socks5h://转换为socks5://: {proxy}")
            
            # 解析代理配置
            proxy_config = {
                "server": proxy
            }
            
            # 如果有用户名密码，添加到配置中
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
                            logger.info(f"已提取代理用户名和密码")
                except Exception as e:
                    logger.error(f"解析代理认证信息失败: {e}")
            
            context_options["proxy"] = proxy_config
            
            # Firefox特殊处理
            if browser_type == "firefox":
                # Firefox在使用代理时的特殊设置
                firefox_args = context_options.get("args", [])
                firefox_args.extend([
                    "-purgecaches",  # 清除缓存
                ])
                context_options["args"] = firefox_args
                
                # 添加Firefox代理特殊设置
                if "firefox_user_prefs" not in context_options:
                    context_options["firefox_user_prefs"] = {}
                
                context_options["firefox_user_prefs"].update({
                    "network.proxy.socks_remote_dns": True,  # 使用SOCKS代理进行DNS解析
                    "network.proxy.no_proxies_on": "",       # 不跳过任何域名
                    "network.dns.disablePrefetch": True,     # 禁用DNS预取
                    "network.proxy.type": 1                  # 使用手动代理配置
                })
                logger.info("已应用Firefox的代理配置")
        
        # 设置用户代理
        if user_agent:
            context_options["user_agent"] = user_agent
        else:
            context_options["user_agent"] = DEFAULT_USER_AGENT
            
        # 设置视口大小
        if viewport:
            context_options["viewport"] = viewport
        else:
            context_options["viewport"] = DEFAULT_VIEWPORT
            
        # 添加其他默认选项
        context_options.update({
            "locale": DEFAULT_LOCALE,
            "timezone_id": DEFAULT_TIMEZONE,
            "geolocation": DEFAULT_GEOLOCATION,
            "color_scheme": "light",
            "reduced_motion": "no-preference",
            "has_touch": False,
            "timeout": self.timeout
        })
        
        args = []
        # 隐藏自动化信息
        if stealth_mode and "chrom" in browser_type:
            logger.info("启用隐身模式，隐藏自动化信息")
            # 添加额外的 Chromium 参数来隐藏自动化信息
            args = context_options.get("args", [])
            args.extend([
                "--disable-blink-features=AutomationControlled",
                "--disable-features=AutomationControlled",
                "--no-sandbox"
            ])
            context_options["args"] = args
        
        logger.info(
            "[headed-diag] launch_persistent_browser: context_options passed to Playwright: headless=%s, args=%s, keys=%s",
            context_options.get("headless"),
            context_options.get("args", []),
            list(context_options.keys()),
        )
        
        # 根据浏览器类型选择启动方法
        try:
            # 重要：Chrome 137+ 已移除 --load-extension 支持
            # 必须使用 Playwright 内置的 Chromium 来加载扩展
            if browser_type == "chrome" or browser_type == "chromium":
                extension = (extension_path or 'extra_tools/CapSolverExtension').strip() or 'extra_tools/CapSolverExtension'
                resolved_extension_path = os.path.abspath(extension)
                
                # 检查扩展目录是否存在
                load_extension = False
                if not os.path.exists(resolved_extension_path):
                    logger.warning(f"扩展目录不存在: {resolved_extension_path}，将不加载扩展")
                elif not os.path.exists(os.path.join(resolved_extension_path, 'manifest.json')):
                    logger.warning(f"扩展目录缺少 manifest.json: {resolved_extension_path}，将不加载扩展")
                else:
                    logger.info(f"加载 capsolver 扩展: {resolved_extension_path}")
                    load_extension = True
                    
                    # 确保 args 存在并添加扩展参数
                    if "args" not in context_options:
                        context_options["args"] = []
                    context_options["args"].extend([
                        '--disable-extensions-except=' + resolved_extension_path,
                        '--load-extension=' + resolved_extension_path,
                    ])
                    
                    # 关键：排除 Playwright 默认的 --disable-extensions 参数
                    # 否则会与 --load-extension 冲突导致浏览器崩溃
                    context_options["ignore_default_args"] = ["--disable-extensions"]
                    logger.info("已排除默认参数 --disable-extensions 以支持扩展加载")
                
                logger.info(f"浏览器启动参数: {context_options.get('args', [])}")
                
                # 关键修复：使用 Playwright 内置 Chromium，而非系统 Chrome
                # Chrome 137+ 已移除 --load-extension 标志，只有 Playwright Chromium 支持
                # 注意：不传入 channel 参数，使用 Playwright 自带的 Chromium
                browser = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    # 不使用 channel="chrome"，这样会使用 Playwright 内置 Chromium
                    **context_options
                )
            elif browser_type == "firefox":
                browser = await self.playwright.firefox.launch_persistent_context(
                    user_data_dir,
                    **context_options
                )
            elif browser_type == "webkit":
                browser = await self.playwright.webkit.launch_persistent_context(
                    user_data_dir,
                    **context_options
                )
            else:
                # 默认使用Chromium
                browser = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir,
                    **context_options
                )
            
            logger.info(
                "[headed-diag] Playwright launch_persistent_context returned: type=%s, pages=%d, "
                "context_options.headless=%s, context_options.args=%s",
                browser_type,
                len(browser.pages),
                context_options.get("headless"),
                context_options.get("args", []),
            )
                
            # 如果启用了隐身模式，进一步修改浏览器属性以隐藏自动化信息
            if stealth_mode and browser.pages:
                for page in browser.pages:
                    await self._apply_stealth_mode(page)
                    
            self.browsers.append(browser)
            return browser
        except Exception as e:
            logger.error(f"启动持久化浏览器时出错: {e}")
            # 如果是代理错误，提供更详细的信息
            if "proxy" in str(e).lower() or "ERR_NO_SUPPORTED_PROXIES" in str(e):
                logger.error("代理设置错误。如果您不需要使用代理，请移除 --proxy 参数")
            raise

    async def _apply_stealth_mode(self, page: Page):
        """Apply stealth to page (delegates to apply_stealth_to_page)."""
        await apply_stealth_to_page(page)


async def apply_stealth_to_page(page: Page) -> None:
    """
    Apply stealth / anti-detection to a Playwright page.
    Shared by BrowserManager and WebContentFetcher. Tries playwright-stealth v2 -> async -> sync, then fallback script.
    """
    logger.debug("Applying stealth to page...")
    if STEALTH_AVAILABLE == "v2" and Stealth is not None:
        try:
            s = Stealth()
            await s.apply_stealth_async(page)
            logger.info("隐身模式脚本应用完成（playwright-stealth v2）")
            return
        except Exception as e:
            logger.warning(f"playwright-stealth v2 应用失败，将回退到基础脚本: {e}")
    elif STEALTH_AVAILABLE == "async" and stealth_async is not None:
        try:
            await stealth_async(page)
            logger.info("隐身模式脚本应用完成（playwright-stealth async API）")
            return
        except Exception as e:
            logger.warning(f"playwright-stealth async API 应用失败，将回退到基础脚本: {e}")
    elif STEALTH_AVAILABLE == "sync" and stealth_sync is not None:
        try:
            await page.evaluate("() => { }")  # 确保页面已加载
            stealth_sync(page)
            logger.info("隐身模式脚本应用完成（playwright-stealth sync API）")
            return
        except Exception as e:
            logger.warning(f"playwright-stealth sync API 应用失败，将回退到基础脚本: {e}")
    else:
        await page.add_init_script("""
                // 隐藏 webdriver 属性
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // 修改 permissions API
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                
                // 伪装 plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // 伪装 languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['zh-CN', 'zh', 'en']
                });
            """)
        logger.info("隐身模式脚本应用完成（备用方案）")

# 修改原有函数以支持PDF下载选项
async def setup_browser(
    browser_type: str = "chrome", 
    headless: Optional[bool] = None, 
    proxy: Optional[str] = None,
    browser_channel: Optional[str] = None,
    user_agent: Optional[str] = None,
    viewport: Optional[Dict[str, int]] = None,
    stealth_mode: bool = True,
    user_data_dir: Optional[str] = None,
    downloads_path: Optional[str] = None,
    always_download_pdf: bool = True,
    **kwargs
) -> Union[Browser, BrowserContext]:
    """
    创建并配置浏览器实例
    
    Args:
        browser_type: 浏览器类型，可选 "chromium"、"chrome"、"firefox"、"webkit"
        headless: 是否使用无头模式
            - None: 自动检测（推荐，支持 capsolver 扩展）
            - True: 强制无头模式（capsolver 可能无法工作）
            - False: 强制有头模式
        proxy: 代理服务器地址
        browser_channel: 浏览器渠道（已弃用，保留参数兼容性）
        user_agent: 自定义用户代理字符串
        viewport: 自定义视口大小 {"width": width, "height": height}
        stealth_mode: 是否启用增强的隐身模式
        user_data_dir: Chrome用户配置文件路径，用于使用已登录的Google账户
        downloads_path: 下载文件保存路径
        always_download_pdf: 是否总是下载PDF而不是在浏览器中打开，默认为True
        **kwargs: 其他浏览器配置选项
    
    Returns:
        Union[Browser, BrowserContext]: 
        - 如果使用了user_data_dir，返回BrowserContext对象
        - 否则返回Browser对象
    """
    # 创建浏览器管理器
    manager = BrowserManager()
    
    # 启动浏览器
    browser = await manager.launch_browser(
        browser_type=browser_type,
        headless=headless,
        proxy=proxy,
        stealth_mode=stealth_mode,
        always_download_pdf=always_download_pdf,
        **kwargs
    )
    
    # 创建上下文
    context = await manager.create_context(
        browser=browser,
        user_agent=user_agent,
        viewport=viewport,
        user_data_dir=user_data_dir,
        downloads_path=downloads_path,
        **kwargs
    )
    
    return context

async def simulate_human_behavior(page):
    """
    模拟人类用户行为，减少被检测为机器人的可能性
    
    Args:
        page: Playwright页面对象
    """
    try:
        logger.info("模拟人类行为...")
        
        # 随机滚动
        await page.evaluate("""
            () => {
                const scrollAmount = Math.floor(Math.random() * window.innerHeight * 0.8);
                window.scrollBy(0, scrollAmount);
                
                // 随机延迟后再滚动回来
                setTimeout(() => {
                    window.scrollBy(0, -scrollAmount * 0.7);
                }, 500 + Math.random() * 1000);
            }
        """)
        
        # 随机移动鼠标
        viewport = await page.evaluate("""
            () => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            }
        """)
        
        if viewport:
            # 生成几个随机点进行鼠标移动
            points = []
            for _ in range(3):
                points.append({
                    'x': random.randint(0, viewport['width'] - 100),
                    'y': random.randint(0, viewport['height'] - 100)
                })
            
            # 执行鼠标移动
            for point in points:
                await page.mouse.move(point['x'], point['y'])
                await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # 随机等待
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        logger.info("人类行为模拟完成")
    except Exception as e:
        logger.warning(f"模拟人类行为时出错: {e}")

async def simulate_human_behavior_with_scroll_to_bottom(page):
    """
    模拟人类用户行为，包括滚动到页面底部，减少被检测为机器人的可能性
    
    Args:
        page: Playwright页面对象
    """
    try:
        logger.info("模拟人类行为，包括滚动到底部...")
        
        # 先进行一些随机滚动
        await page.evaluate("""
            () => {
                const scrollAmount = Math.floor(Math.random() * window.innerHeight * 0.6);
                window.scrollBy(0, scrollAmount);
                
                // 随机延迟后再滚动更多
                setTimeout(() => {
                    window.scrollBy(0, scrollAmount * 0.8);
                }, 300 + Math.random() * 800);
            }
        """)
        
        await asyncio.sleep(random.uniform(0.3, 0.8))
        
        # 滚动到页面底部（平滑滚动）
        await page.evaluate("""
            () => {
                // 平滑滚动到底部
                const scrollHeight = document.body.scrollHeight;
                const duration = (1000 + Math.random() * 1500) / 1.7; // 加快到原来的1.7倍速度，约0.59-1.47秒
                const startTime = performance.now();
                const startPos = window.pageYOffset;
                
                function scrollStep(timestamp) {
                    const elapsed = timestamp - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const easeInOutQuad = progress < 0.5 
                        ? 2 * progress * progress 
                        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                    
                    window.scrollTo(0, startPos + (scrollHeight - startPos) * easeInOutQuad);
                    
                    if (progress < 1) {
                        window.requestAnimationFrame(scrollStep);
                    }
                }
                
                window.requestAnimationFrame(scrollStep);
            }
        """)
        
        # 等待1-3秒的随机时间，模拟用户阅读页面底部内容
        bottom_wait_time = random.uniform(0.5, 1.5)
        logger.info(f"滚动到页面底部，等待 {bottom_wait_time:.2f} 秒...")
        await asyncio.sleep(bottom_wait_time)
        
        # 随机移动鼠标
        viewport = await page.evaluate("""
            () => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            }
        """)
        
        if viewport:
            # 生成几个随机点进行鼠标移动（底部区域的点较多）
            points = []
            for _ in range(2):
                # 更倾向于底部区域
                points.append({
                    'x': random.randint(0, viewport['width'] - 100),
                    'y': random.randint(viewport['height'] * 0.6, viewport['height'] - 50)
                })
            
            # 执行鼠标移动
            for point in points:
                await page.mouse.move(point['x'], point['y'])
                await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # 最后的短暂停顿
        await asyncio.sleep(random.uniform(0.3, 0.7))
        
        logger.info("人类行为模拟完成，已滚动到页面底部")
    except Exception as e:
        logger.warning(f"模拟人类行为（滚动到底部）时出错: {e}")
        # 尝试基本的行为模拟作为备选方案
        try:
            await simulate_human_behavior(page)
        except Exception as e2:
            logger.warning(f"备选人类行为模拟也失败: {e2}")

async def progressive_input(page, selector, text, min_delay=50, max_delay=150):
    """
    渐进式输入文本，模拟人类打字行为
    
    Args:
        page: Playwright页面对象
        selector: 输入框选择器
        text: 要输入的文本
        min_delay: 最小延迟（毫秒）
        max_delay: 最大延迟（毫秒）
    """
    try:
        logger.info(f"渐进式输入文本到 {selector}")
        
        # 先点击输入框
        await page.click(selector)
        await asyncio.sleep(random.uniform(0.3, 0.7))
        
        # 清除现有内容
        await page.fill(selector, "")
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        # 逐个字符输入
        for char in text:
            # 随机延迟
            delay = random.randint(min_delay, max_delay)
            await page.type(selector, char, delay=delay)
            
            # 偶尔添加更长的停顿，模拟思考
            if random.random() < 0.1:
                await asyncio.sleep(random.uniform(0.3, 0.8))
        
        # 输入完成后的短暂停顿
        await asyncio.sleep(random.uniform(0.5, 1.0))
        
        logger.info("渐进式输入完成")
    except Exception as e:
        logger.warning(f"渐进式输入时出错: {e}")
        # 回退到标准输入方法
        await page.fill(selector, text) 