import os
import html
import json
import time
import hashlib
import uuid
from time import sleep
import argparse
import shutil
import subprocess
import asyncio
import aiohttp
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any
import sys
import re
import signal
from urllib.parse import urljoin, urlparse
from .browser_manager import BrowserManager, simulate_human_behavior, setup_browser
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from .log_utils import setup_logger, cleanup_old_logs
import logging
from bs4 import BeautifulSoup
from twocaptcha import TwoCaptcha

# playwright-stealth 兼容层：
# - v2.x：Stealth().apply_stealth_async(page)
# - 少数旧版本：stealth_async(page) / stealth_sync(page)
_STEALTH_MODE: str = "none"  # "v2" | "async" | "sync" | "none"
_Stealth = None
_stealth_async = None
_stealth_sync = None
try:
    from playwright_stealth import Stealth as _Stealth  # type: ignore
    _STEALTH_MODE = "v2"
except Exception:
    try:
        from playwright_stealth import stealth_async as _stealth_async  # type: ignore
        _STEALTH_MODE = "async"
    except Exception:
        try:
            from playwright_stealth import stealth_sync as _stealth_sync  # type: ignore
            _STEALTH_MODE = "sync"
        except Exception:
            _STEALTH_MODE = "none"
from datetime import datetime
import random

# 配置日志
logger = setup_logger('PaperDownloader', level = "INFO")

# Sci-Hub 下载结果原因码（用于省钱：遇到明确 not_in_db 时跳过 Bright Data + Sci-Hub）
SCIHUB_OK = "ok"
SCIHUB_NOT_IN_DB = "not_in_db"
SCIHUB_BLOCKED = "blocked_or_challenge"
SCIHUB_TIMEOUT = "timeout"
SCIHUB_ERROR = "error"

# Sci-Hub 现在通过 BrightData + sci-hub.st 实现，无需开关
# Playwright 版本已废弃（DDoS-Guard 不稳定）


def load_config(config_path: str = "config.json") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"无法加载配置文件 {config_path}: {e}")
        return {}


# ===== 阶段1：页面分析能力（仅用于诊断日志，不改变流程）=====
class BlockerType(Enum):
    """阻断类型"""
    CLOUDFLARE = "cloudflare"
    CAPTCHA = "captcha"
    PAYWALL = "paywall"
    LOGIN_REQUIRED = "login_required"
    GEO_BLOCKED = "geo_blocked"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"


class ContentType(Enum):
    """页面内容类型"""
    PDF_INLINE = "pdf_inline"
    PDF_VIEWER = "pdf_viewer"
    ARTICLE_PAGE = "article_page"
    LANDING_PAGE = "landing_page"
    DOWNLOAD_PAGE = "download_page"
    ERROR_PAGE = "error_page"
    UNKNOWN = "unknown"


@dataclass
class Blocker:
    """阻断因素"""
    type: BlockerType
    solvable: bool = True
    details: str = ""


@dataclass
class ActionableElement:
    """可操作元素"""
    selector: str
    tag: str
    text: str
    href: Optional[str] = None
    is_visible: bool = True
    position_y: int = 0
    score: int = 0
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class PageAnalysis:
    """页面分析结果"""
    url: str = ""
    domain: str = ""
    title: str = ""
    content_type: ContentType = ContentType.UNKNOWN
    blockers: List[Blocker] = field(default_factory=list)
    actionable_elements: List[ActionableElement] = field(default_factory=list)
    download_detected: bool = False
    redirect_chain: List[str] = field(default_factory=list)
    raw_info: Dict[str, Any] = field(default_factory=dict)


class PageAnalyzer:
    """页面分析器：检测页面状态和可操作元素"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.pdf_keywords = [
            "pdf", "download", "full text", "full-text", "view", "open", "read",
            "下载", "打开", "全文", "查看", "保存", "télécharger", "descargar", "herunterladen"
        ]
        self.pdf_url_patterns = [
            ".pdf", "/pdf/", "/pdf?", "pdf=", "pdfft", "pdfdirect",
            "/epdf/", "/reader/", "/doi/pdf/", "/article/pdf/",
            "download=true", "viewcontent"
        ]

    async def analyze(self, page) -> PageAnalysis:
        """全面分析当前页面"""
        analysis = PageAnalysis()
        try:
            analysis.url = page.url
            analysis.domain = self._extract_domain(page.url)
            analysis.title = await page.title() or ""
            analysis.blockers = await self._detect_blockers(page)
            analysis.content_type = await self._detect_content_type(page)
            analysis.actionable_elements = await self._find_actionable_elements(page)
            analysis.raw_info = await self._extract_raw_info(page)
        except Exception as e:
            self.logger.warning(f"页面分析出错: {e}")
        return analysis

    def _extract_domain(self, url: str) -> str:
        """提取域名"""
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    async def _detect_blockers(self, page) -> List[Blocker]:
        """检测各种阻断因素"""
        blockers: List[Blocker] = []
        try:
            title = (await page.title() or "").lower()
            url = page.url.lower()
            try:
                body_text = await page.evaluate(
                    '() => document.body ? document.body.innerText.slice(0, 5000) : ""'
                )
                body_text = (body_text or "").lower()
            except Exception:
                body_text = ""

            cloudflare_indicators = [
                ("title", ["just a moment", "attention required", "请稍候", "checking your browser"]),
                ("text", ["verify you are human", "checking your browser", "ddos protection", "cloudflare"]),
            ]
            cf_detected = False
            for check_type, patterns in cloudflare_indicators:
                source = title if check_type == "title" else body_text
                if any(p in source for p in patterns):
                    cf_detected = True
                    break
            if not cf_detected:
                cf_selectors = [
                    "#challenge-running", "#cf-challenge-running",
                    'iframe[src*="challenges.cloudflare.com"]',
                    ".cf-browser-verification", "#turnstile-wrapper"
                ]
                for sel in cf_selectors:
                    try:
                        elem = await page.query_selector(sel)
                        if elem:
                            cf_detected = True
                            break
                    except Exception:
                        pass
            if cf_detected:
                blockers.append(Blocker(BlockerType.CLOUDFLARE, solvable=True, details="Cloudflare验证"))

            captcha_indicators = ["captcha", "recaptcha", "hcaptcha", "验证码", "robot"]
            if any(ind in body_text for ind in captcha_indicators) and not cf_detected:
                blockers.append(Blocker(BlockerType.CAPTCHA, solvable=True, details="检测到验证码"))

            paywall_indicators = [
                "subscribe to read", "subscription required", "purchase this article",
                "buy this article", "rent this article", "pay per view",
                "institutional access", "access through your institution",
                "订阅", "购买全文", "get full access"
            ]
            if any(ind in body_text for ind in paywall_indicators):
                blockers.append(Blocker(BlockerType.PAYWALL, solvable=False, details="需要付费/订阅"))

            login_indicators = [
                "sign in to", "log in to", "login required", "please log in",
                "authentication required", "请登录", "需要登录"
            ]
            if any(ind in body_text for ind in login_indicators):
                blockers.append(Blocker(BlockerType.LOGIN_REQUIRED, solvable=False, details="需要登录"))

            error_indicators = ["page not found", "404", "not found", "does not exist", "页面不存在"]
            if any(ind in body_text or ind in title for ind in error_indicators):
                blockers.append(Blocker(BlockerType.NOT_FOUND, solvable=False, details="页面不存在"))

            server_error_indicators = ["500", "502", "503", "server error", "service unavailable", "服务器错误"]
            if any(ind in body_text or ind in title for ind in server_error_indicators):
                blockers.append(Blocker(BlockerType.SERVER_ERROR, solvable=False, details="服务器错误"))

            rate_limit_indicators = ["rate limit", "too many requests", "429", "请求过于频繁"]
            if any(ind in body_text or ind in title for ind in rate_limit_indicators):
                blockers.append(Blocker(BlockerType.RATE_LIMITED, solvable=True, details="请求频率限制"))
        except Exception as e:
            self.logger.debug(f"阻断检测出错: {e}")
        return blockers

    async def _detect_content_type(self, page) -> ContentType:
        """检测页面内容类型"""
        try:
            url = page.url.lower()
            
            # 1. 检测 PDF 内联显示（URL 直接指向 PDF）
            pdf_inline_patterns = [
                ".pdf", "/pdf/", "/pdf?", "pdf=", "/content/pdf/",
                "/article-pdf/", "/pdfdirect", "pdfft"
            ]
            if any(p in url for p in pdf_inline_patterns):
                return ContentType.PDF_INLINE

            # 2. 检测 PDF 查看器页面
            pdf_viewer_patterns = [
                "/epdf/", "/pdf-viewer", "/reader/", "/pdfviewer",
                "/doi/reader/", "/pdfreader"
            ]
            if any(p in url for p in pdf_viewer_patterns):
                return ContentType.PDF_VIEWER

            viewer_selectors = [
                "#pdf-viewer", ".pdf-viewer", '[class*="pdfViewer"]',
                'embed[type="application/pdf"]', 'iframe[src*=".pdf"]'
            ]
            for sel in viewer_selectors:
                try:
                    elem = await page.query_selector(sel)
                    if elem:
                        return ContentType.PDF_VIEWER
                except Exception:
                    pass

            article_patterns = [
                "/doi/full/", "/doi/abs/", "/doi/10.", "/article/",
                "/papers/", "/publication/", "/abstract/"
            ]
            if any(p in url for p in article_patterns):
                return ContentType.ARTICLE_PAGE

            download_patterns = ["/download", "download=", "/getpdf", "/viewcontent"]
            if any(p in url for p in download_patterns):
                return ContentType.DOWNLOAD_PAGE

            landing_patterns = ["doi.org/", "/resolve/", "/redirect"]
            if any(p in url for p in landing_patterns):
                return ContentType.LANDING_PAGE

            title = (await page.title() or "").lower()
            if "error" in title or "not found" in title or "404" in title:
                return ContentType.ERROR_PAGE
            return ContentType.UNKNOWN
        except Exception as e:
            self.logger.debug(f"内容类型检测出错: {e}")
            return ContentType.UNKNOWN

    async def _find_actionable_elements(self, page) -> List[ActionableElement]:
        """发现所有可能与PDF下载相关的元素"""
        elements: List[ActionableElement] = []
        try:
            raw_elements = await page.evaluate('''() => {
                const results = [];
                const seen = new Set();
                function getKey(el) {
                    return el.tagName + '|' + (el.href || '') + '|' + (el.innerText || '').slice(0, 50);
                }
                function isPdfRelated(el) {
                    const text = (el.innerText || '').toLowerCase();
                    const href = (el.href || '').toLowerCase();
                    const className = (el.className || '').toLowerCase();
                    const title = (el.title || '').toLowerCase();
                    const ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
                    const keywords = ['pdf', 'download', '下载', '打开', '查看', 'full text', 'view', 'open',
                                     'read', 'save', 'export', 'get', '全文', '保存'];
                    return keywords.some(kw =>
                        text.includes(kw) || href.includes(kw) ||
                        className.includes(kw) || title.includes(kw) ||
                        ariaLabel.includes(kw)
                    );
                }
                function buildSelector(el) {
                    if (el.id) return '#' + el.id;
                    let selector = el.tagName.toLowerCase();
                    if (el.className && typeof el.className === 'string') {
                        const classes = el.className.trim().split(/\\s+/).slice(0, 3);
                        if (classes.length > 0 && classes[0]) {
                            selector += '.' + classes.join('.');
                        }
                    }
                    if (el.href && el.href.includes('pdf')) {
                        selector += '[href*="pdf"]';
                    }
                    if (el.title) {
                        selector += '[title="' + el.title.replace(/"/g, '\\"') + '"]';
                    }
                    return selector;
                }
                document.querySelectorAll('a[href]').forEach(el => {
                    if (!isPdfRelated(el)) return;
                    const key = getKey(el);
                    if (seen.has(key)) return;
                    seen.add(key);
                    const rect = el.getBoundingClientRect();
                    results.push({
                        selector: buildSelector(el),
                        tag: 'a',
                        text: (el.innerText || '').slice(0, 100).trim(),
                        href: el.href || '',
                        is_visible: rect.width > 0 && rect.height > 0 && el.offsetParent !== null,
                        position_y: rect.top + window.scrollY,
                        attributes: {
                            class: el.className || '',
                            title: el.title || '',
                            download: el.hasAttribute('download') ? 'true' : '',
                            target: el.target || ''
                        }
                    });
                });
                document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]').forEach(el => {
                    if (!isPdfRelated(el)) return;
                    const key = getKey(el);
                    if (seen.has(key)) return;
                    seen.add(key);
                    const rect = el.getBoundingClientRect();
                    results.push({
                        selector: buildSelector(el),
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || el.value || '').slice(0, 100).trim(),
                        href: null,
                        is_visible: rect.width > 0 && rect.height > 0 && el.offsetParent !== null,
                        position_y: rect.top + window.scrollY,
                        attributes: {
                            class: el.className || '',
                            title: el.title || '',
                            type: el.type || '',
                            'aria-label': el.getAttribute('aria-label') || ''
                        }
                    });
                });
                return results.slice(0, 50);
            }''')
            for raw in raw_elements:
                elem = ActionableElement(
                    selector=raw["selector"],
                    tag=raw["tag"],
                    text=raw["text"],
                    href=raw.get("href"),
                    is_visible=raw["is_visible"],
                    position_y=int(raw["position_y"]),
                    attributes=raw.get("attributes", {})
                )
                elem.score = self._calculate_score(elem)
                elements.append(elem)
            elements.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            self.logger.debug(f"元素发现出错: {e}")
        return elements

    def _calculate_score(self, elem: ActionableElement) -> int:
        """计算元素的PDF相关性评分"""
        score = 0
        text = elem.text.lower()
        href = (elem.href or "").lower()
        attrs = elem.attributes
        attr_class = attrs.get("class", "").lower()
        attr_title = attrs.get("title", "").lower()
        attr_aria = attrs.get("aria-label", "").lower()
        
        # ====== 加分逻辑 ======
        if "pdf" in text:
            score += 25
        if "download" in text or "下载" in text:
            score += 15
        if "full text" in text or "full-text" in text or "全文" in text:
            score += 10
        if "view" in text or "open" in text or "打开" in text or "查看" in text:
            score += 10
        if ".pdf" in href:
            score += 30
        if "/pdf/" in href or "/pdf?" in href:
            score += 25
        if "download" in href:
            score += 15
        if "epdf" in href or "pdfdirect" in href:
            score += 20
        if attrs.get("download"):
            score += 15
        if "pdf" in attr_class:
            score += 10
        if "pdf" in attr_title:
            score += 10
        if "pdf" in attr_aria:
            score += 10
        if elem.is_visible:
            score += 10
        if elem.position_y < 300:
            score += 5
        elif elem.position_y < 600:
            score += 3
        if elem.tag == "a":
            score += 2
        
        # ====== 降权逻辑：补充材料 ======
        supplementary_keywords = [
            'supplementary', 'supplement', 'supporting', 'appendix',
            'esm', 'moesm', 'mediaobject', '-sup-', '-si-', '/si/',
            '附件', '附录', '补充材料', '支持材料'
        ]
        combined_text = f"{text} {href} {attr_class} {attr_title}"
        for kw in supplementary_keywords:
            if kw in combined_text:
                score -= 50
                break
        
        # ====== 降权逻辑：参考文献链接 ======
        reference_keywords = [
            'reference', 'bibliography', 'cited-by', 'citation',
            'ref-list', 'reflist', '/ref/', '#ref',
            '参考文献', '引用'
        ]
        for kw in reference_keywords:
            if kw in combined_text:
                score -= 40
                break
        
        # ====== 降权逻辑：其他干扰 ======
        noise_keywords = ['flyer', 'cover', 'toc', 'abstract-only', 'preview-only']
        for kw in noise_keywords:
            if kw in combined_text:
                score -= 30
                break
        
        return max(0, min(score, 100))

    async def _extract_raw_info(self, page) -> Dict[str, Any]:
        """提取原始页面信息（用于调试）"""
        try:
            return await page.evaluate('''() => {
                return {
                    url: location.href,
                    title: document.title,
                    has_pdf_embed: !!document.querySelector('embed[type="application/pdf"]'),
                    has_pdf_iframe: !!document.querySelector('iframe[src*=".pdf"]'),
                    button_count: document.querySelectorAll('button').length,
                    link_count: document.querySelectorAll('a[href]').length,
                    form_count: document.querySelectorAll('form').length,
                    iframe_count: document.querySelectorAll('iframe').length
                };
            }''')
        except Exception:
            return {}

    def format_analysis(self, analysis: PageAnalysis) -> str:
        """格式化分析结果为可读字符串"""
        lines = [
            "=== 页面分析结果 ===",
            f"URL: {analysis.url[:80]}{'...' if len(analysis.url) > 80 else ''}",
            f"域名: {analysis.domain}",
            f"标题: {analysis.title[:50]}{'...' if len(analysis.title) > 50 else ''}",
            f"内容类型: {analysis.content_type.value}",
            f"阻断因素: {[f'{b.type.value}({b.details})' for b in analysis.blockers] if analysis.blockers else '无'}",
            f"可操作元素: {len(analysis.actionable_elements)} 个",
        ]
        if analysis.actionable_elements:
            lines.append("Top 5 元素:")
            for i, elem in enumerate(analysis.actionable_elements[:5]):
                lines.append(
                    f"  {i+1}. [分数:{elem.score}] {elem.tag} - {elem.text[:30]} - {elem.selector[:50]}"
                )
        return "\n".join(lines)


class ExperienceStore:
    """经验库：记录和复用成功的下载路径"""
    
    def __init__(self, store_path: str, logger: logging.Logger):
        self.store_path = store_path
        self.logger = logger
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """加载经验库"""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logger.info(f"已加载经验库，包含 {len(data.get('domains', {}))} 个域名记录")
                    return data
            except Exception as e:
                self.logger.warning(f"加载经验库失败: {e}")
        return {
            'domains': {},      # 域名 -> [成功路径]
            'selectors': {},    # 选择器 -> {success: int, fail: int}
            'version': 1
        }
    
    def _save(self):
        """保存经验库"""
        try:
            # 使用临时文件确保原子写入
            temp_path = self.store_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            # 原子重命名
            if sys.platform == 'win32' and os.path.exists(self.store_path):
                os.remove(self.store_path)
            os.rename(temp_path, self.store_path)
        except Exception as e:
            self.logger.warning(f"保存经验库失败: {e}")
    
    def record_success(self, domain: str, url: str, action_sequence: List[Dict[str, Any]]):
        """
        记录成功的下载经验
        
        Args:
            domain: 域名
            url: 完整URL
            action_sequence: 操作序列，每个元素是 {type, selector, text, ...}
        """
        if not domain or not action_sequence:
            return
        
        # 泛化URL模式
        url_pattern = self._generalize_url(url)
        
        if domain not in self.data['domains']:
            self.data['domains'][domain] = []
        
        # 检查是否已有相同模式的记录
        existing = None
        for exp in self.data['domains'][domain]:
            if exp.get('pattern') == url_pattern:
                existing = exp
                break
        
        if existing:
            # 更新现有记录
            existing['success_count'] = existing.get('success_count', 0) + 1
            existing['last_success'] = datetime.now().isoformat()
            existing['actions'] = action_sequence  # 用最新的操作序列
        else:
            # 新增记录
            self.data['domains'][domain].append({
                'pattern': url_pattern,
                'actions': action_sequence,
                'success_count': 1,
                'last_success': datetime.now().isoformat(),
                'created': datetime.now().isoformat()
            })
        
        # 更新选择器统计
        for action in action_sequence:
            selector = action.get('selector')
            if selector:
                if selector not in self.data['selectors']:
                    self.data['selectors'][selector] = {'success': 0, 'fail': 0}
                self.data['selectors'][selector]['success'] += 1
        
        self._save()
        self.logger.debug(f"[经验库] 记录成功: {domain} - {url_pattern}")
    
    def record_failure(self, domain: str, failed_selectors: List[str]):
        """记录失败的选择器"""
        for selector in failed_selectors:
            if selector:
                if selector not in self.data['selectors']:
                    self.data['selectors'][selector] = {'success': 0, 'fail': 0}
                self.data['selectors'][selector]['fail'] += 1
        self._save()
    
    def query_experience(self, domain: str, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        查询是否有可复用的经验
        
        Args:
            domain: 域名
            url: 完整URL
        
        Returns:
            成功的操作序列，或 None
        """
        if domain not in self.data['domains']:
            return None
        
        url_pattern = self._generalize_url(url)
        
        # 优先精确匹配
        for exp in self.data['domains'][domain]:
            if exp.get('pattern') == url_pattern:
                self.logger.info(f"[经验库] 找到精确匹配: {url_pattern} (成功{exp.get('success_count', 0)}次)")
                return exp.get('actions')
        
        # 其次模糊匹配（同域名下成功次数最多的）
        best_exp = None
        best_count = 0
        for exp in self.data['domains'][domain]:
            count = exp.get('success_count', 0)
            if count > best_count:
                best_count = count
                best_exp = exp
        
        if best_exp and best_count >= 2:
            self.logger.info(f"[经验库] 使用同域名最佳经验 (成功{best_count}次)")
            return best_exp.get('actions')
        
        return None
    
    def get_selector_priority(self) -> List[Tuple[str, float]]:
        """
        获取选择器优先级排序
        
        Returns:
            [(selector, success_rate), ...] 按成功率降序
        """
        result = []
        for selector, stats in self.data['selectors'].items():
            total = stats['success'] + stats['fail']
            if total >= 3:  # 至少3次才有统计意义
                rate = stats['success'] / total
                result.append((selector, rate))
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def boost_element_scores(self, elements: List[ActionableElement]) -> List[ActionableElement]:
        """
        根据历史成功率调整元素评分
        
        Args:
            elements: 原始元素列表
        
        Returns:
            调整评分后的元素列表
        """
        selector_stats = self.data.get('selectors', {})
        
        for elem in elements:
            stats = selector_stats.get(elem.selector)
            if stats:
                total = stats['success'] + stats['fail']
                if total >= 2:
                    rate = stats['success'] / total
                    # 根据成功率调整分数 (-20 到 +20)
                    bonus = int((rate - 0.5) * 40)
                    elem.score = max(0, min(100, elem.score + bonus))
        
        return sorted(elements, key=lambda x: x.score, reverse=True)
    
    def _generalize_url(self, url: str) -> str:
        """泛化URL模式，去除具体ID"""
        pattern = url
        try:
            # 替换DOI: 10.1002/ece3.70522 → 10.{doi}
            pattern = re.sub(r'10\.\d{4,9}/[^\s/?#]+', '10.{doi}', pattern)
            # 替换长数字ID: /123456/ → /{id}/
            pattern = re.sub(r'/\d{5,}(/|$)', '/{id}\\1', pattern)
            # 替换UUID风格ID
            pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '{uuid}', pattern, flags=re.IGNORECASE)
            # 去除查询参数中的session/token
            pattern = re.sub(r'[?&](session|token|sid|auth)[^&]*', '', pattern, flags=re.IGNORECASE)
        except Exception:
            pass
        return pattern
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'domains': len(self.data.get('domains', {})),
            'total_experiences': sum(len(v) for v in self.data.get('domains', {}).values()),
            'selectors_tracked': len(self.data.get('selectors', {}))
        }


class LLMAssistant:
    """LLM辅助：在常规方法失败时分析页面并建议操作"""
    
    def __init__(self, api_key: str, logger: logging.Logger, model: str = None, provider: str = "auto",
                 timeout: int = 20):
        self.api_key = api_key
        self.logger = logger
        self.enabled = bool(api_key)
        self.timeout = timeout
        
        # 自动检测 provider
        if provider == "auto" and api_key:
            if api_key.startswith("sk-") and len(api_key) > 20:
                # DeepSeek API key 格式通常是 sk- 开头
                self.provider = "deepseek"
                self.model = model or "deepseek-chat"
            else:
                # Anthropic API key 格式
                self.provider = "anthropic"
                self.model = model or "claude-sonnet-4-20250514"
        else:
            self.provider = provider if provider != "auto" else "anthropic"
            self.model = model or ("deepseek-chat" if self.provider == "deepseek" else "claude-sonnet-4-20250514")
        
        if not self.enabled:
            self.logger.info("[LLM] 未配置API密钥，LLM辅助功能禁用")
        else:
            self.logger.info(f"[LLM] 使用 {self.provider} API，模型: {self.model}")
    
    async def analyze_and_suggest(self, page, analysis: PageAnalysis, 
                                   action_history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        调用LLM分析页面并建议下一步操作
        
        Args:
            page: Playwright页面对象
            analysis: PageAnalysis对象
            action_history: 已尝试的操作历史
        
        Returns:
            建议的操作 {type, selector, reason} 或 None
        """
        if not self.enabled:
            return None
        
        try:
            # 1. 提取页面关键信息（控制token）
            page_summary = await self._extract_page_summary(page)
            
            # 2. 构建prompt
            prompt = self._build_prompt(page_summary, analysis, action_history)
            
            # 3. 调用LLM
            response = await self._call_llm(prompt)
            
            # 4. 解析响应
            suggestion = self._parse_response(response)
            
            if suggestion:
                self.logger.info(f"[LLM] 建议操作: {suggestion.get('type')} - {suggestion.get('reason', '')[:50]}")
            
            return suggestion
            
        except Exception as e:
            self.logger.warning(f"[LLM] 分析失败: {e}")
            return None
    
    async def _extract_page_summary(self, page) -> Dict[str, Any]:
        """提取页面关键信息（精简版）"""
        try:
            return await page.evaluate('''() => {
                const result = {
                    url: location.href,
                    title: document.title,
                    buttons: [],
                    links: [],
                    text_hints: []
                };
                
                // 提取按钮（限制数量）
                document.querySelectorAll('button, [role="button"]').forEach((btn, i) => {
                    if (i < 15) {
                        const text = (btn.innerText || '').trim().slice(0, 50);
                        if (text) {
                            result.buttons.push({
                                text: text,
                                class: (btn.className || '').slice(0, 80),
                                visible: btn.offsetParent !== null
                            });
                        }
                    }
                });
                
                // 提取可能相关的链接
                document.querySelectorAll('a[href]').forEach((a, i) => {
                    const href = (a.href || '').toLowerCase();
                    const text = (a.innerText || '').toLowerCase();
                    
                    if (i < 20 && (
                        href.includes('pdf') || href.includes('download') ||
                        text.includes('pdf') || text.includes('download') ||
                        text.includes('view') || text.includes('full')
                    )) {
                        result.links.push({
                            text: (a.innerText || '').trim().slice(0, 50),
                            href: a.href.slice(0, 150),
                            visible: a.offsetParent !== null
                        });
                    }
                });
                
                // 检测关键文本
                const bodyText = (document.body?.innerText || '').toLowerCase();
                ['download', 'pdf', 'full text', 'access', 'subscribe', 'login', 'purchase'].forEach(kw => {
                    if (bodyText.includes(kw)) {
                        result.text_hints.push(kw);
                    }
                });
                
                return result;
            }''')
        except Exception:
            return {'url': page.url, 'title': '', 'buttons': [], 'links': [], 'text_hints': []}
    
    def _build_prompt(self, page_summary: Dict, analysis: PageAnalysis, 
                      action_history: List[Dict]) -> str:
        """构建LLM prompt"""
        
        history_str = ""
        if action_history:
            history_str = "\n".join([
                f"  - {a.get('type', '?')}: {a.get('selector', a.get('text', '?'))[:40]} -> {a.get('result', '?')}"
                for a in action_history[-5:]  # 只显示最近5个
            ])
        else:
            history_str = "  (无)"
        
        blockers_str = ', '.join([b.type.value for b in analysis.blockers]) if analysis.blockers else "无"
        
        prompt = f"""你是一个PDF下载助手。分析以下网页信息，告诉我下一步该怎么做才能下载PDF。

## 当前页面
URL: {page_summary.get('url', '')[:100]}
标题: {page_summary.get('title', '')[:80]}

## 页面元素
按钮: {json.dumps(page_summary.get('buttons', [])[:10], ensure_ascii=False)}
相关链接: {json.dumps(page_summary.get('links', [])[:10], ensure_ascii=False)}
关键词检测: {page_summary.get('text_hints', [])}

## 检测到的问题
{blockers_str}

## 已尝试的操作
{history_str}

请返回JSON格式的建议（只返回JSON，不要其他文字）:
{{
    "action": {{
        "type": "click|navigate|give_up",
        "selector": "CSS选择器（如果type是click）",
        "url": "目标URL（如果type是navigate）",
        "reason": "简短解释"
    }},
    "confidence": 0-100
}}

注意：
1. 如果检测到付费墙/需要登录，type应为"give_up"
2. 优先选择明确包含"pdf"或"download"的元素
3. 如果是PDF查看器页面，找下载/保存按钮
4. selector要尽量精确，优先用class或id"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        if self.provider == "deepseek":
            # DeepSeek API
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"DeepSeek API返回 {resp.status}: {error_text[:200]}")
                    result = await resp.json()
                    # DeepSeek 返回格式: {choices: [{message: {content: "..."}}]}
                    return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            # Anthropic API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        raise Exception(f"Anthropic API返回 {resp.status}")
                    result = await resp.json()
                    return result.get('content', [{}])[0].get('text', '')
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM响应（支持 DeepSeek thinking 模式）"""
        try:
            response = response.strip()
            
            # 处理 DeepSeek thinking 模式：提取 <think>...</think> 之后的内容
            if '<think>' in response.lower() or '<thinking>' in response.lower():
                # 移除 thinking 标签及其内容
                response = re.sub(r'<think[^>]*>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
                response = re.sub(r'<thinking[^>]*>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
                response = response.strip()
                self.logger.debug(f"[LLM] 已移除 thinking 标签，剩余内容长度: {len(response)}")
            
            # 尝试提取JSON（可能在最后）
            # 方法1: 查找最后一个完整的 { ... } 块（支持嵌套）
            # 从后往前找第一个 {，然后匹配到对应的 }
            brace_count = 0
            start_idx = -1
            for i in range(len(response) - 1, -1, -1):
                if response[i] == '}':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif response[i] == '{':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        # 找到了完整的 JSON 对象
                        response = response[i:start_idx + 1]
                        break
            
            # 去除markdown代码块
            if response.startswith('```'):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
                response = response.strip()
            
            # 尝试解析JSON
            data = json.loads(response)
            action = data.get('action', {})
            
            if action.get('type') in ['click', 'navigate', 'give_up']:
                return action
            else:
                self.logger.debug(f"[LLM] 解析出的 action.type 无效: {action.get('type')}")
            
        except json.JSONDecodeError as e:
            # JSON解析失败，尝试更宽松的提取
            self.logger.debug(f"[LLM] JSON解析失败，尝试提取: {str(e)[:100]}")
            
            # 尝试查找 JSON 对象（更宽松）
            json_patterns = [
                r'\{[^{}]*"action"[^{}]*\{[^{}]*"type"[^{}]*"[^"]*"[^{}]*\}[^{}]*\}',  # 嵌套JSON
                r'\{[^}]*"type"\s*:\s*"(click|navigate|give_up)"[^}]*\}',  # 简单JSON
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        action = data.get('action', {})
                        if action.get('type') in ['click', 'navigate', 'give_up']:
                            return action
                    except:
                        continue
            
            self.logger.debug(f"[LLM] 无法从响应中提取有效JSON，原始响应前200字符: {response[:200]}")
        
        except Exception as e:
            self.logger.debug(f"[LLM] 解析响应失败: {e}")
        
        return None


# 通过该下载器下载的论文会保存在download_dir目录下，download_dir目录下会创建一个tmp目录，用于下载临时文件
# download_dir目录下会创建一个papers目录，用于保存下载的论文
# download_dir目录下会创建一个logs目录，用于保存下载的日志
# download_dir目录下会创建一个browser_data目录，用于保存浏览器数据
# download_dir目录下会创建一个history.json文件，用于保存下载的历史
# download_dir目录下会创建一个download_history.json文件，用于保存下载的历史

class PaperDownloader:
    # ==================== PDF 选择器配置（类级别常量）====================
    # 统一管理所有PDF按钮/链接选择器，避免在多个方法中重复定义
    
    PDF_SELECTORS = {
        # 网站特定选择器（高优先级，精确匹配）
        'site_specific': {
            'wiley': [
                'a.coolBar__ctrl.pdf-download',
                'a[title="ePDF"][class*="coolBar"]',
                'a.coolBar__ctrl[href*="/doi/epdf/"]',
                '#pdfviewer-toolbar button[aria-label*="Download"]',
                '.pdf-viewer-toolbar button[aria-label*="download"]',
                'a[href*="/doi/pdf/"][title="Download PDF"]',
            ],
            'cedar_digital_commons': [
                'a#pdf',
                'a.btn[href*="viewcontent"]',
                'a[href*="viewcontent.cgi"]',
                'a.btn:has-text("Download")',
            ],
            'springer_nature': [
                'a.c-pdf-download__link[href*="/pdf/"][data-article-pdf="true"]',
                'a.u-button[data-article-pdf="true"][data-test="pdf-link"]',
                'div.c-pdf-download a.c-pdf-download__link[href*="/pdf/"]',
                'a.c-pdf-download__link',
                'a[data-track-action="download pdf"]',
                'div.pdf-link a[href*="/pdf/"]',
            ],
            'elsevier': [
                'a.link-button[href*="pdfft"]',
                'a.link-button[aria-label*="View PDF"]',
                'a.link-button-primary[href*="pdfft"]',
            ],
            'frontiersin': [
                'a.download-files-pdf',
                'a[data-action="download-pdf"]',
            ],
            'ojs_pkp': [  # OJS/PKP系统（europeanjournaloftaxonomy, zootaxa等）
                'a.download',
                '.galley-link.download',
                'a.obj_galley_link',
                'a.obj_galley_link.pdf',
                'a.download span.label',
            ],
            'mdpi': [
                'a.download-pdf-link',
                'a[title="Download PDF"]',
                'a.UD_ArticlePDF[href*="/pdf"]',
            ],
            'bmc': [
                'a.c-pdf-download__link[href*="content/pdf"]',
            ],
            'cambridge': [
                'a[href*="/core/services/aop-cambridge-core/content/view"]',
            ],
            'oxford': [
                'a.article-pdfLink',
                'a[href*="/article-pdf/"]',
            ],
            'jstor': [
                'a.download-pdf',
            ],
            'taylor_francis': [
                'a.show-pdf[href*="/epdf/"]',
            ],
            'ieee': [
                'a.doc-actions-link.stats-document-lh-action-downloadPdf_3',
            ],
            'sage': [
                'a#downloadPdfUrl[href*="/pdf/"]',
            ],
            'proquest': [
                'a.contents-menu-tab-title[href*="fulltextPDF"]',
                'a[id*="addFlashPageParameterformat_fulltextPDF"]',
                'a[title*="Full text - PDF from ProQuest"]',
            ],
            'pnas': [
                'a.btn.btn--pdf[href*="/doi/reader/"][aria-label="Open full-text in eReader"]',
            ],
            'the_innovation': [
                'a[href^="/data/article/"][href*="/preview/pdf/"]',
                'span.hidden-lg a[href*="/pdf/"]',
            ],
            'ssrn': [
                'a[href*="Delivery.cfm"][href*=".pdf"]',
            ],
            'science': [
                'a.btn[aria-label="PDF"]',
            ],
        },
        
        # 通用选择器（按类型分组）
        'generic': {
            'direct_links': [
                'a[href$=".pdf"]',
                'a[href*="/pdf/"][download]',
                'a[download]',
                'a[download][href*=".pdf"]',
                'a[href*="/pdf/"][href*="download"]',
                'a[href*=".pdf"][target="_blank"]',
                'a[href*="/doi/pdf/"]',
            ],
            'buttons': [
                'button[title*="Download"]',
                'button[title*="download"]',
                'button:has-text("Download")',
                'button:has-text("Download PDF")',
                'button:has-text("Open PDF")',
                'button:has-text("Open")',
                'button:has-text("打开")',
                'button:has-text("下载")',
                'button:has-text("打开 PDF")',
                'button:has-text("下载 PDF")',
                'button[aria-label*="PDF"]',
                'button#open-button',
            ],
            'links': [
                'a[title*="PDF"]',
                'a[title*="pdf"]',
                'a[title*="Download PDF"]',
                'a[title*="download PDF"]',
                'a:has-text("Download")',
                'a:has-text("Download PDF")',
                'a:has-text("Open PDF")',
                'a:has-text("打开")',
                'a:has-text("Download This Paper")',
                'a:has-text("Open PDF in Browser")',
                'a.pdf-download',
                '.download-pdf-btn',
                'a.btn[href*="pdf"]',
                'a:has(button#open-button)',
                '[data-testid="download-pdf"]',
                'div.article-content a[href*="pdf"]',
                '.download-link[href*="pdf"]',
                'div.download-pdf a',
                'a.c-pdf-download__link[href*=".pdf"]',
            ],
            'text_matchers': [
                ':text("View PDF")',
                ':text("Download PDF")',
                ':text("Full Text PDF")',
            ],
        },
        
        # 下拉菜单触发器
        'dropdown_triggers': [
            'button:has-text("Download article")',
            'button:has-text("Download")',
            '[aria-haspopup="true"]:has-text("Download")',
            'button:has-text("下载文章")',
            'button.ActionsDropDown__button',
        ],
        
        # 下拉菜单中的PDF链接
        'dropdown_pdf_links': [
            'a:has-text("Download PDF")',
            'a:has-text("PDF")',
            'a[href*=".pdf"]',
            'a:has-text("下载 PDF")',
            'a[data-event*="pdf"]',
            '.ActionsDropDown__menu a[href*="/pdf"]',
        ],
        
        # 模态框/弹窗内按钮
        'modal_buttons': [
            '.modal button:has-text("打开")',
            '.modal button:has-text("下载")',
            '.modal button:has-text("Open")',
            '.modal button:has-text("Download")',
            '.modal a:has-text("Open PDF")',
            '.dialog button:has-text("打开")',
            '.dialog button:has-text("Open")',
            '[role="dialog"] button:has-text("打开")',
            '[role="dialog"] button:has-text("Open")',
            '[role="dialog"] button:has-text("Download")',
            '.overlay button:has-text("打开")',
            '.overlay button:has-text("Open")',
            '.popup button:has-text("打开")',
            '.popup button:has-text("Open")',
        ],
        
        # 页面中央/PDF容器内按钮
        'center_buttons': [
            '.pdf-container button:has-text("打开")',
            '.pdf-viewer button:has-text("打开")',
            '.pdf-container button:has-text("Open")',
            '.pdf-viewer button:has-text("Open")',
            '#pdf-viewer button',
            '.reader-body button:has-text("打开")',
            '.reader-body button:has-text("Open")',
            'button:has-text("打开")',
            'button:has-text("下载")',
            'button.open-pdf',
            'button.view-pdf',
            'a.open-pdf-link',
        ],
    }
    
    # 多步点击配置：针对特定站点的多步操作
    MULTI_STEP_SITES = {
        'wiley': {
            'detect': lambda url: 'onlinelibrary.wiley.com' in url.lower(),
            'steps': [
                {'selector': 'div.coolBar__section.coolBar--download.PdfLink a.coolBar__ctrl.pdf-download, a.coolBar__ctrl.pdf-download, a[title="ePDF"].coolBar__ctrl, a:has-text("Download PDF")', 'wait': 2},
                {'selector': 'a[href*="pdfdirect"][href*="download=true"], a[href*="pdfdirect"], a[aria-label*="Download"]', 'wait': 2}
            ]
        },
        'elife': {
            'detect': lambda url: 'elife' in url.lower(),
            'steps': [
                {'selector': 'a.button.button--default.button--action.icon.icon-download#button-action-download', 'wait': 1},
                {'selector': 'a.article-download-links-list__link:has-text("Article PDF")', 'wait': 1}
            ]
        },
        'zootaxa': {
            'detect': lambda url: 'mapress.com' in url.lower(),
            'steps': [
                {'selector': 'a.obj_galley_link.pdf', 'wait': 1},
                {'selector': 'a.download[download]', 'wait': 1}
            ]
        },
    }
    
    # 网站URL匹配规则
    SITE_URL_PATTERNS = {
        'wiley': ['onlinelibrary.wiley.com'],
        'cedar_digital_commons': ['cedar', 'digitalcommons'],
        'springer_nature': ['springer.com', 'nature.com', 'link.springer.com'],
        'elsevier': ['sciencedirect.com', 'elsevier.com'],
        'frontiersin': ['frontiersin.org'],
        'ojs_pkp': ['europeanjournaloftaxonomy.eu', 'mapress.com', 'biotaxa.org'],
        'mdpi': ['mdpi.com'],
        'bmc': ['biomedcentral.com', 'bmj.com'],
        'cambridge': ['cambridge.org'],
        'oxford': ['academic.oup.com', 'oxfordjournals.org'],
        'jstor': ['jstor.org'],
        'taylor_francis': ['tandfonline.com'],
        'ieee': ['ieeexplore.ieee.org'],
        'sage': ['sagepub.com'],
        'proquest': ['proquest.com'],
        'pnas': ['pnas.org'],
        'the_innovation': ['the-innovation.org'],
        'ssrn': ['ssrn.com'],
        'science': ['science.org', 'sciencemag.org'],
    }
    
    def __init__(self, download_dir: str = "papers", max_concurrent: int = 5, show_browser: bool = False, 
                 persist_browser: bool = False, download_timeout: int = 200, timeout: int = 10, max_retries: int = 3,
                 browser_type: str = "chrome", stealth_mode: bool = True, config_path: str = "config.json",
                 logger: logging.Logger = logger):
        """初始化论文下载器
        
        Args:
            download_dir: 下载文件保存的目录
            max_concurrent: 最大并发下载数
            show_browser: 是否显示浏览器窗口
            persist_browser: 是否保持浏览器会话
            download_timeout: 下载超时时间(秒)
            max_retries: 最大重试次数
            browser_type: 浏览器类型 ("chrome", "chromium", "firefox")
            stealth_mode: 是否启用隐身模式以避免网站检测自动化
            config_path: 配置文件路径
        """
        # 设置日志记录器
        self.logger = logger
        
        # 加载配置文件
        self.config = load_config(config_path)
        # 统一超时配置（可在 config.json -> downloader.timeouts 覆盖）
        default_timeouts = {
            "session_acquire_timeout": 15,
            "page_create_timeout": 10,
            "goto_timeout": 20,
            "load_state_timeout": 10,
            "redirect_chain_timeout": 15,
            "page_stable_wait": 2,
            "cookie_consent_timeout": 8,
            "cloudflare_timeout": 60,
            "cloudflare_retry_wait": 10,
            "captcha_timeout": 60,
            "download_event_timeout": 15,
            "inline_pdf_fetch_timeout": 15,
            "inline_pdf_body_timeout": 10,
            "button_appear_timeout": 8,
            "button_click_timeout": 5,
            "post_click_nav_timeout": 10,
            "pdf_viewer_timeout": 12,
            "download_complete_timeout": 20,
            "download_poll_interval": 1,
            "rate_limit_wait": 30,
            "smart_loop_total_timeout": 60,
            "smart_loop_iteration_wait": 2,
            "llm_timeout": 20,
            "paper_total_timeout": 120
        }
        self.timeouts = default_timeouts
        self.timeouts.update(self.config.get("downloader", {}).get("timeouts", {}))
        api_keys = self.config.get("api_keys", {})
        self.apikey = api_keys.get("twocaptcha", "")
        self.brightdata_api_key = api_keys.get("brightdata", "")
        self.download_dir = os.path.abspath(download_dir)
        # 注：download_tmp_dir 已废弃，改用任务专属临时目录 (.task_xxx)
        self.max_concurrent = max_concurrent
        self.show_browser = show_browser
        self.persist_browser = persist_browser
        self.download_timeout = download_timeout
        self.timeout = timeout
        self.max_retries = max_retries
        self.browser_type = browser_type.lower()
        self.headless = not show_browser  # 根据 show_browser 设置 headless 属性
        self.stealth_mode = stealth_mode  # 添加 stealth_mode 参数
        
        # 确保下载目录存在
        os.makedirs(self.download_dir, exist_ok=True)
        self.logger.debug(f"初始化下载目录: {self.download_dir}")
        
        # Debug 文件日志（仅在 DEBUG 级别时创建）
        self.debug_log_file = None
        if self.logger.level == logging.DEBUG:
            debug_log_path = os.path.join(os.path.dirname(self.download_dir), "logs", "debug.log")
            os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
            file_handler = logging.FileHandler(debug_log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.debug_log_file = debug_log_path
            self.logger.debug(f"=== Debug 模式已启用，日志文件: {debug_log_path} ===")
        
        # 持久化浏览器相关属性
        self._persistent_user_data_dir = os.path.join(os.path.dirname(self.download_dir), ".browser_data")
        self._browser_context = None  # 统一的浏览器上下文（兼容旧逻辑）
        self._browser_manager = None  # 浏览器管理器
        self._context_lock = asyncio.Lock()  # 添加锁来保护context的创建和使用
        self._active_pages = set()  # 跟踪所有活动的页面
        
        # 会话池相关属性（解决 cookie/重定向问题 + PDF 设置生效）
        self._session_pool_size = max_concurrent  # 会话池大小与并发数保持一致
        self._session_pool: asyncio.Queue = asyncio.Queue()  # 会话队列
        self._session_pool_data: List[Dict] = []  # 会话元数据 [{context, user_data_dir, index, last_used}]
        self._session_pool_initialized = False  # 会话池是否已初始化
        
        # 动态资源管理（优化：空闲超时自动释放）
        self._session_idle_timeout = 60  # 会话空闲超时（秒），空闲超过此时间后关闭
        self._cleanup_task: Optional[asyncio.Task] = None  # 清理任务
        self._last_activity_time = time.time()  # 最后活动时间
        self._use_lightweight_mode = True  # 轻量模式：单次下载使用临时上下文，快速释放
        
        # 下载历史
        self.history_file = os.path.join(self.download_dir, ".download_history.json")
        self.download_history = self.load_download_history()
        
        # MD5 索引：用于检测重复文件（相同内容的 PDF）
        self.file_md5_index = {}  # MD5 -> filepath 映射
        self._load_md5_index()
        
        # Initialize the browser manager to use its functionalities (initialization, stealth mode, human behavior simulation)
        self.browser_manager = BrowserManager()
        
        # 页面分析器（阶段1：仅用于诊断）
        self.page_analyzer = PageAnalyzer(self.logger)

        # ============================================================
        # 经验库（阶段3：智能学习）
        # 存储成功的下载策略，可跨项目共享
        # 配置项：config.json -> downloader.experience_store_path
        # - 设为 null：使用默认路径 {download_dir}/.experience_store.json
        # - 设为绝对路径：如 "/Users/shared/.experience_store.json" 可跨项目共享
        # - 设为相对路径：相对于项目根目录
        # ============================================================
        configured_path = self.config.get("downloader", {}).get("experience_store_path")
        if configured_path:
            # 用户自定义路径（支持绝对路径和相对路径）
            if os.path.isabs(configured_path):
                self.experience_store_path = configured_path
            else:
                # 相对路径：相对于配置文件所在目录
                config_dir = os.path.dirname(os.path.abspath(config_path))
                self.experience_store_path = os.path.join(config_dir, configured_path)
            self.logger.info(f"[经验库] 使用自定义路径: {self.experience_store_path}")
        else:
            # 默认路径：下载目录下的隐藏文件
            self.experience_store_path = os.path.join(self.download_dir, ".experience_store.json")
        self.experience_store = ExperienceStore(self.experience_store_path, self.logger)

        # LLM辅助（阶段3，可选）
        # 优先读取 deepseek，否则读取 anthropic（向后兼容）
        api_keys = self.config.get("api_keys", {})
        llm_api_key = api_keys.get("deepseek", "") or api_keys.get("anthropic", "")
        llm_provider = self.config.get("llm", {}).get("provider", "auto")  # 可选配置：auto/deepseek/anthropic
        llm_model = self.config.get("llm", {}).get("model", None)  # 可选配置：自定义模型
        self.llm_assistant = LLMAssistant(
            llm_api_key,
            self.logger,
            model=llm_model,
            provider=llm_provider,
            timeout=self.timeouts.get("llm_timeout", 20)
        )

        # 其他初始化代码...

# 拦截script，用于获取 Cloudflare JS Challenge 参数，包括非常关键的sitekey，后续将用于解决验证码
    intercept_script = """
        console.clear = () => console.log('Console was cleared');
        const i = setInterval(()=>{
            if (window.turnstile) {
                console.log('success!!');
                clearInterval(i);
                window.turnstile.render = (a, b) => {
                    let params = {
                        sitekey: b.sitekey,
                        pageurl: window.location.href,
                        data: b.cData,
                        pagedata: b.chlPageData,
                        action: b.action,
                        userAgent: navigator.userAgent,
                    };
                    console.log('intercepted-params:' + JSON.stringify(params));
                    window.cfCallback = b.callback;
                }
            }
        },50);
    """

    # ==================== 选择器辅助方法 ====================
    
    def _match_site(self, url: str, site_key: str) -> bool:
        """检查URL是否匹配指定网站
        
        Args:
            url: 待检查的URL
            site_key: 网站标识键（如'wiley', 'springer_nature'等）
            
        Returns:
            bool: 是否匹配
        """
        if site_key not in self.SITE_URL_PATTERNS:
            return False
        
        url_lower = url.lower()
        patterns = self.SITE_URL_PATTERNS[site_key]
        return any(pattern in url_lower for pattern in patterns)
    
    def _get_site_specific_selectors(self, url: str) -> list:
        """根据URL返回网站特定的选择器列表
        
        Args:
            url: 当前页面URL
            
        Returns:
            list: 网站特定的选择器列表
        """
        for site_key, selectors in self.PDF_SELECTORS['site_specific'].items():
            if self._match_site(url, site_key):
                self.logger.debug(f"匹配到网站特定选择器: {site_key}")
                return selectors
        return []
    
    def _build_selector_list(self, url: str) -> list:
        """构建完整的选择器列表（网站特定 + 通用）
        
        Args:
            url: 当前页面URL
            
        Returns:
            list: 完整的选择器列表，按优先级排序
        """
        selectors = []
        
        # 1. 网站特定选择器（最高优先级）
        site_specific = self._get_site_specific_selectors(url)
        if site_specific:
            selectors.extend(site_specific)
        
        # 2. 通用选择器（按优先级添加）
        selectors.extend(self.PDF_SELECTORS['generic']['direct_links'])
        selectors.extend(self.PDF_SELECTORS['generic']['buttons'])
        selectors.extend(self.PDF_SELECTORS['generic']['links'])
        selectors.extend(self.PDF_SELECTORS['generic']['text_matchers'])
        
        return selectors

# 尝试在动态加载的 iframe 中查找并点击 id="open-button" 的按钮，这在很多学术网站中非常重要，
# 特别是Elsevier，Springer等网站,许多时候并没有直接提供pdf链接，而是需要点击pdf下载或者类似的才可以。
    async def click_open_button_if_found(self, page, task_download_dir: Optional[str] = None):
        """
        尝试在页面或 iframe 中查找并点击下载/打开 PDF 的按钮。
        支持多种网站的按钮样式，包括 Wiley、Elsevier 等。
        """
        try:
            # 等待页面完全加载
            await page.wait_for_load_state(
                'domcontentloaded',
                timeout=self.timeouts.get("load_state_timeout", 10) * 1000
            )
            self.logger.debug("页面网络活动已稳定")
            
            # 先等待一段时间，让动态内容有机会加载
            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
            
            # ===== 方法1：检测主页面中的通用下载/打开按钮 =====
            # 使用统一配置的选择器
            element_timeout = self.timeouts.get("button_appear_timeout", 8) * 1000
            main_page_selectors = self._build_selector_list(page.url)
            
            for selector in main_page_selectors:
                try:
                    btn = page.locator(selector).first
                    await btn.wait_for(state='attached', timeout=element_timeout)
                    try:
                        await btn.scroll_into_view_if_needed(timeout=element_timeout)
                    except:
                        pass
                    
                    if await btn.is_visible(timeout=element_timeout):
                        self.logger.info(f"在主页面找到潜在的按钮: {selector}")
                        tag_name = await btn.evaluate("el => el.tagName.toLowerCase()")
                        target_attr = await btn.evaluate("el => el.getAttribute('target')")
                        href_attr = await btn.evaluate("el => el.getAttribute('href')")

                        if tag_name == 'button':
                            try:
                                parent_link = await btn.evaluate(
                                    "el => { const a = el.closest('a'); if (!a) return null; return {href: a.getAttribute('href'), target: a.getAttribute('target')}; }"
                                )
                                if parent_link and parent_link.get("href"):
                                    tag_name = 'a'
                                    href_attr = parent_link.get("href")
                                    target_attr = parent_link.get("target")
                                    self.logger.info("按钮位于链接内，使用链接导航")
                            except Exception:
                                pass

                        if tag_name == 'a' and href_attr:
                            full_url = href_attr if href_attr.startswith('http') else urljoin(page.url, href_attr)
                            if self._is_likely_pdf_url(full_url) or target_attr == '_blank':
                                self.logger.info(f"链接直达尝试导航: {full_url}")
                                try:
                                    await page.goto(
                                        full_url,
                                        wait_until='domcontentloaded',
                                        timeout=self.timeouts.get("goto_timeout", 20) * 1000
                                    )
                                    await asyncio.sleep(2)
                                    return True
                                except Exception as nav_e:
                                    if "Download is starting" in str(nav_e):
                                        self.logger.info("链接导航触发下载")
                                        return True
                                    raise

                        else:
                            await btn.click()
                            self.logger.info(f"已点击: {selector}")
                            await asyncio.sleep(2)
                            # Wiley专属：处理Cloudflare验证并等待PDF页面加载
                            if 'onlinelibrary.wiley.com' in page.url and ('/doi/pdf/' in page.url or '/doi/epdf/' in page.url):
                                try:
                                    self.logger.info("检测到Wiley PDF页面，使用通用方法处理Cloudflare验证...")
                                    cf_success, _ = await self.solve_cloudflare_if_needed(
                                        page, task_download_dir=task_download_dir
                                    )
                                    if cf_success:
                                        self.logger.info("Wiley PDF页面Cloudflare验证已处理，等待下载按钮...")
                                        await page.wait_for_selector(
                                            'a[href*="pdfdirect"], a[aria-label*="Download"]',
                                            timeout=self.timeouts.get("button_appear_timeout", 8) * 1000
                                        )
                                        self.logger.info("Wiley下载按钮已加载")
                                except Exception as e:
                                    self.logger.warning(f"Wiley PDF页面处理失败: {e}")
                            return True
                    else:
                        self.logger.debug(f"元素 {selector} 不可见，尝试强制点击")
                        await btn.click(force=True)
                        self.logger.info(f"强制点击: {selector}")
                        await asyncio.sleep(2)
                        # Wiley专属：处理Cloudflare验证并等待PDF页面加载
                        if 'onlinelibrary.wiley.com' in page.url and ('/doi/pdf/' in page.url or '/doi/epdf/' in page.url):
                            try:
                                self.logger.info("检测到Wiley PDF页面，使用通用方法处理Cloudflare验证...")
                                cf_success, _ = await self.solve_cloudflare_if_needed(
                                    page, task_download_dir=task_download_dir
                                )
                                if cf_success:
                                    self.logger.info("Wiley PDF页面Cloudflare验证已处理，等待下载按钮...")
                                    await page.wait_for_selector(
                                        'a[href*="pdfdirect"], a[aria-label*="Download"]',
                                        timeout=self.timeouts.get("button_appear_timeout", 8) * 1000
                                    )
                                    self.logger.info("Wiley下载按钮已加载")
                            except Exception as e:
                                self.logger.warning(f"Wiley PDF页面处理失败: {e}")
                        return True

                except Exception as e:
                    continue
            
            # ===== 方法2：检测弹窗/模态框中的按钮 =====
            modal_selectors = self.PDF_SELECTORS['modal_buttons']
            
            for selector in modal_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        self.logger.info(f"在模态框/弹窗中找到并点击了按钮: {selector}")
                        await asyncio.sleep(1)
                        return True
                except:
                    continue
            
            # ===== 方法3：滚动页面并检测中间的打开按钮 =====
            # 有些网站会在 PDF 区域显示一个"点击打开"的覆盖层
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(0.5)
            
            center_button_selectors = self.PDF_SELECTORS['center_buttons']
            
            for selector in center_button_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        self.logger.info(f"在页面中央找到并点击了按钮: {selector}")
                        await asyncio.sleep(1)
                        return True
                except:
                    continue
            
            # ===== 方法4：原有逻辑 - 检查 iframe 中的按钮 =====
            frames = page.frames
            frame_count = len(frames)
            self.logger.debug(f"当前页面中有 {frame_count} 个 frame")
                
            for i, frame in enumerate(frames):
                if frame != page.main_frame:
                    try:
                        # 在 iframe 中查找 open-button
                        button = await frame.wait_for_selector('button#open-button', timeout=1000, state='visible')
                        if button:
                            try:
                                parent_link = await button.evaluate(
                                    "el => { const a = el.closest('a'); if (!a) return null; return {href: a.getAttribute('href'), target: a.getAttribute('target')}; }"
                                )
                                if parent_link and parent_link.get("href"):
                                    full_url = urljoin(frame.url, parent_link.get("href"))
                                    self.logger.info(f"在 iframe 中找到链接，尝试导航: {full_url}")
                                    try:
                                        await page.goto(
                                            full_url,
                                            wait_until='domcontentloaded',
                                            timeout=self.timeouts.get("goto_timeout", 20) * 1000
                                        )
                                        await asyncio.sleep(2)
                                        return True
                                    except Exception as nav_e:
                                        if "Download is starting" in str(nav_e):
                                            self.logger.info("iframe 链接导航触发下载")
                                            return True
                                        raise
                            except Exception:
                                pass
                            frame_element = await page.query_selector(f"iframe[src*='{frame.url.split('/')[-1]}']")
                            if frame_element:
                                await frame_element.scroll_into_view_if_needed()
                            
                            await button.click()
                            self.logger.info(f"在第 {i} 个 iframe 中找到并点击了 open-button 按钮")
                            return True
                    except Exception as e:
                        self.logger.debug(f"在第 {i} 个 iframe 中未找到按钮: {str(e)[:50]}")
            
            # ===== 方法5：处理下拉菜单式下载按钮 =====
            # 某些网站（如Frontiersin）需要先点击下拉菜单触发器，再点击PDF链接
            dropdown_triggers = self.PDF_SELECTORS['dropdown_triggers']
            
            for trigger in dropdown_triggers:
                try:
                    btn = page.locator(trigger).first
                    if await btn.is_visible(timeout=1000):
                        self.logger.info(f"找到下拉菜单触发器: {trigger}")
                        await btn.click()
                        await asyncio.sleep(0.5)  # 等待下拉菜单展开
                        
                        # 在下拉菜单中查找PDF链接
                        pdf_link_selectors = self.PDF_SELECTORS['dropdown_pdf_links']
                        
                        for pdf_selector in pdf_link_selectors:
                            try:
                                pdf_link = page.locator(pdf_selector).first
                                if await pdf_link.is_visible(timeout=1000):
                                    href = await pdf_link.get_attribute("href")
                                    if href:
                                        try:
                                            from urllib.parse import urljoin
                                            if href.startswith("//"):
                                                href = "https:" + href
                                            elif href.startswith("/"):
                                                href = urljoin(page.url, href)
                                        except Exception:
                                            pass
                                        try:
                                            self.logger.info(f"在下拉菜单中找到PDF直链，尝试导航: {href[:120]}")
                                            await page.goto(href, wait_until="domcontentloaded")
                                            return True
                                        except Exception as nav_e:
                                            if "Download is starting" in str(nav_e):
                                                self.logger.info("下拉菜单PDF直链触发下载")
                                                return True
                                    await pdf_link.click()
                                    self.logger.info(f"在下拉菜单中找到并点击了PDF链接: {pdf_selector}")
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            self.logger.debug("未找到任何可点击的下载/打开按钮")
            return False
         
        except Exception as e:
            self.logger.error(f"搜索下载按钮时发生错误: {str(e)}")
            return False

    async def _handle_cookie_consent(self, page, context_label: str = "") -> bool:
        """检测并尝试点击 cookie 同意按钮（自动同意）"""
        cookie_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("Accept Cookies")',
            'button:has-text("I Accept")',
            'button:has-text("OK")',
            'button:has-text("同意")',
            'button:has-text("接受")',
            '[id*="accept"]',
            '[class*="accept"]',
            '[data-testid*="accept"]',
            '.cc-btn.cc-dismiss',
            '#onetrust-accept-btn-handler',
            '.cookie-consent-accept',
        ]
        try:
            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
            for selector in cookie_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=self.timeouts.get("cookie_consent_timeout", 8) * 1000):
                        await btn.click()
                        label = f"({context_label})" if context_label else ""
                        self.logger.info(f"点击了 cookie 同意按钮{label}: {selector}")
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        return True
                except Exception:
                    continue
        except Exception as e:
            self.logger.debug(f"cookie 同意处理失败: {e}")
        return False
        
# 尝试点击页面中的下载按钮，用于sci-hub等网站
    async def click_onclick_button(self, page): #sci-hub使用
        """尝试点击页面中的下载按钮
        
        支持通过以下方式查找按钮：
        1. 按钮带有onclick属性的按钮
        2. 按钮文本包含下载相关关键词
        3. 按钮在特定div内（如id="buttons"）
        
        Args:
            page: Playwright页面对象
            
        Returns:
            bool: 是否成功点击按钮
        """
        try:
            # 等待页面完全加载
            await page.wait_for_load_state(
                'networkidle',
                timeout=self.timeouts.get("load_state_timeout", 10) * 1000
            )
            self.logger.debug("页面网络活动已稳定")
            
            # 等待动态内容加载
            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
            
            # 滚动到页面中间
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")

            # 1. 尝试查找在id="buttons"区域内的按钮
            button = await page.query_selector('#buttons button')
            if button:
                self.logger.info("在buttons区域找到下载按钮，尝试点击")
                await button.click()
                return True

            # 2. 尝试查找带有onclick属性且文本包含下载关键词的按钮
            buttons_with_onclick = await page.query_selector_all('button[onclick]')
            for button in buttons_with_onclick:
                button_text = await button.inner_text()
                if any(keyword in button_text.lower() for keyword in ['save', 'download', '下载']):
                    self.logger.info("找到带onclick属性且包含下载关键词的按钮，尝试点击")
                    await button.click()
                    return True

            # 3. 尝试查找任何包含下载关键词的按钮
            all_buttons = await page.query_selector_all('button')
            for button in all_buttons:
                button_text = await button.inner_text()
                if any(keyword in button_text.lower() for keyword in ['save', 'download', '下载']):
                    self.logger.info("找到包含下载关键词的按钮，尝试点击")
                    await button.click()
                    return True

            self.logger.info("在页面中未找到任何下载按钮")
            return False
            
        except Exception as e:
            self.logger.error(f"点击下载按钮时发生错误: {str(e)}")
            return False

# 检查页面中的Cloudflare验证高置信度指标，由于Cloudflare验证的复杂性，需要多次检查，
# 以及等待一段时间后再次检查，确保检测到Cloudflare验证。
    async def is_cloudflare_verifying(self, page, detection_duration=1.5):
        """
        在指定时间内动态检测页面是否正在进行Cloudflare验证
        Args:
            page: Playwright页面对象
            detection_duration: 检测持续时间(秒)
            wait_timeout: 超时等待时间(秒)，超过此时间将停止检测
            retry_interval: 重试间隔时间(秒)，每次检查的间隔
        
            
        Returns:
            bool: 是否正在进行Cloudflare验证
        """
        try:
            # 初始状态检查（立即检查一次）
            initial_state = await self._check_cloudflare_indicators(page)
            if initial_state:
                return True
                
            # 动态监控变化
            start_time = time.time()
            while time.time() - start_time < detection_duration:
                # 等待一小段时间再次检查
                await asyncio.sleep(0.3)
                
                # 再次检查指标
                current_state = await self._check_cloudflare_indicators(page)
                if current_state:
                    return True
                    
            # 如果在指定时间内没有检测到验证
            return False
            
        except Exception as e:
            logger.error(f"检测Cloudflare验证时出错: {e}")
            return False

  # 检查页面中的Cloudflare验证高置信度指标          
    async def _check_cloudflare_indicators(self, page):
        """检查页面中的Cloudflare验证高置信度指标"""
        
        # 1. 检查页面标题（支持多语言）
        title = await page.title()
        cloudflare_titles = [
            "Just a moment",      # 英文
            "Attention Required", # 英文
            "请稍候",             # 中文简体
            "請稍候",             # 中文繁体
            "Un moment",          # 法语
            "Einen Moment",       # 德语
            "Un momento",         # 西班牙语/意大利语
        ]
        for cf_title in cloudflare_titles:
            if cf_title in title:
                self.logger.info(f"检测到Cloudflare页面标题: {title}")
                return True
            
        # 2. 检查关键DOM元素（扩展选择器列表）
        key_selectors = [
            '#challenge-running', 
            '#cf-challenge-running',
            'iframe[src*="challenges.cloudflare.com"]',
            '#turnstile-wrapper:not([style*="display: none"])',
            '.cf-browser-verification',
            '#cf-spinner-please-wait',
            # 新增：Turnstile 验证框相关选择器
            'input[name="cf-turnstile-response"]',
            '.cf-turnstile',
            'div[data-sitekey]',  # Turnstile widget
            '#cf-hcaptcha-container',
            '.hcaptcha-box',
        ]
        
        for selector in key_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    # 某些元素不需要可见性检查（如 hidden input）
                    if 'input[name=' in selector:
                        self.logger.info(f"检测到Cloudflare验证元素: {selector}")
                        return True
                    if await element.is_visible():
                        self.logger.info(f"检测到Cloudflare验证元素: {selector}")
                        return True
            except Exception:
                pass
        
        # 3. 检查页面文本内容（Are you a robot? / Verify you are human）
        try:
            page_text = await page.evaluate('() => document.body ? document.body.innerText : ""')
            cf_text_indicators = [
                "Are you a robot",
                "Verify you are human",
                "Please verify you are a human",
                "Checking your browser",
                "This process is automatic",
                "Please complete the security check",
                "confirm you are a human",
                "Enable JavaScript and cookies",
            ]
            for indicator in cf_text_indicators:
                if indicator.lower() in page_text.lower():
                    self.logger.info(f"检测到Cloudflare验证文本: {indicator}")
                    return True
        except Exception:
            pass
        
        # 4. 检查Cloudflare JavaScript对象
        has_cf_objects = await page.evaluate('''() => {
            return !!(
                window._cf_chl_opt || 
                window.___cf_chl_ctx || 
                (window.cf && window.cf.chl_done) ||
                window.turnstile ||
                document.querySelector('[data-ray]') ||
                document.querySelector('meta[name="cf-2fa"]')
            );
        }''')
        
        if has_cf_objects:
            self.logger.info("检测到Cloudflare验证JS对象")
            return True
            
        return False
    
    # 获取 Cloudflare JS Challenge 参数，包括非常关键的sitekey，后续将用于解决验证码
    async def get_captcha_params(self, page, script):  
        """
        刷新页面，注入拦截 JavaScript，并通过监听控制台日志获取 Turnstile 参数。
        """
        self.logger.info("开始获取 Cloudflare JS Challenge 参数")
        await page.reload()  # 刷新页面
        await page.evaluate(script)  # 注入拦截脚本
        try:
            # 等待控制台输出包含"intercepted-params:"的日志信息
            msg = await page.wait_for_event("console", lambda m: "intercepted-params:" in m.text, timeout=20*1000)
            log_entry = msg.text
            match = re.search(r'intercepted-params:({.*?})', log_entry)
            if match:
                json_string = match.group(1)
                params = json.loads(json_string)
                self.logger.info("Parameters received")
                if params:
                    return params
                else:
                    self.logger.error("No parameters found in console output")
                    raise Exception("No parameters found in console output")
            else:
                self.logger.error("No parameters found in console output")
                raise Exception("No parameters found in console output")
        except Exception as e:
            self.logger.error(f"Failed to intercept parameters: {str(e)}")
            raise
            
    # 使用 2Captcha 服务解决 Turnstile 验证码    
    def solver_captcha(self,apikey, params):  # 使用 2Captcha 服务解决 Turnstile 验证码    
        """
        使用 2Captcha 服务解决 Turnstile 验证码。
        """
        solver = TwoCaptcha(apikey)
        try:
            result = solver.turnstile(sitekey=params["sitekey"],
                                      url=params["pageurl"],
                                      action=params["action"],
                                      data=params["data"],
                                      pagedata=params["pagedata"],
                                      useragent=params["userAgent"])
            self.logger.info("Captcha solved")
            return result['code']
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return None
        
# 将解码后的 token 传入页面回调函数，从而正式解决验证码问题
    async def send_token_callback(self, page, token):
        """
        将解码后的 token 传入页面回调函数。
        """
        script = f"cfCallback('{token}')"
        await page.evaluate(script)
        self.logger.info("The token is sent to the callback function")

# ==================== 通用 Cloudflare 验证码自动处理 ====================
    async def solve_cloudflare_if_needed(self, page, filepath: str = None,
                                          max_retries: int = 2,
                                          wait_after_solve: bool = True,
                                          task_download_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        通用的 Cloudflare 验证码检测和解决方法。
        可以在任何页面导航后调用，自动检测并解决 Cloudflare Turnstile 验证码。
        
        Args:
            page: Playwright 页面对象
            filepath: 可选的文件保存路径（如果验证后触发下载）
            max_retries: 最大重试次数
            wait_after_solve: 解决后是否等待页面加载
            task_download_dir: 任务专属临时目录（并发安全）
            
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None): 无验证码或验证码已解决，页面正常
                - (True, filepath): 验证码解决后触发了下载，文件已保存
                - (False, None): 验证码解决失败
        """
        for attempt in range(max_retries):
            try:
                cf_detected = await self.is_cloudflare_verifying(page)
                if not cf_detected:
                    self.logger.debug("未检测到 Cloudflare 验证码")
                    return (True, None)
                
                self.logger.info(f"检测到 Cloudflare 验证码，尝试解决 (尝试 {attempt + 1}/{max_retries})")
                
                if not self.apikey:
                    self.logger.error("未配置 2Captcha API Key，无法解决验证码")
                    return (False, None)
                
                params = None
                try:
                    params = await self.get_captcha_params(page, self.intercept_script)
                except Exception as param_e:
                    self.logger.error(f"提取验证码参数失败 (尝试 {attempt + 1}): {param_e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        continue
                    return (False, None)
                
                if not params:
                    self.logger.error("未能获取验证码参数")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        continue
                    return (False, None)
                
                self.logger.info(f"成功提取验证码参数: sitekey={params.get('sitekey', 'N/A')[:20]}...")
                
                token = None
                try:
                    token = self.solver_captcha(self.apikey, params)
                except Exception as solve_e:
                    self.logger.error(f"验证码解决失败 (尝试 {attempt + 1}): {solve_e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        continue
                    return (False, None)
                
                if not token:
                    self.logger.error("未能获取验证码 token")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        continue
                    return (False, None)
                
                self.logger.info("验证码解决成功，发送 token...")
                
                if filepath:
                    local_task_dir = task_download_dir
                    cleanup_task_dir = False
                    if not local_task_dir:
                        local_task_dir, _, _ = self._create_task_download_dir()
                        cleanup_task_dir = True
                    try:
                        async with page.expect_download(timeout=self.timeouts.get("download_event_timeout", 15) * 1000) as download_info:
                            await self.send_token_callback(page, token)
                        download = await download_info.value
                        if await self._save_and_validate_download(
                            download, filepath, local_task_dir, "cloudflare_callback"
                        ):
                            self.logger.info(f"验证码回调后触发下载，文件已保存: {filepath}")
                            return (True, filepath)
                        else:
                            self.logger.warning("下载的文件无效，继续流程")
                    except asyncio.TimeoutError:
                        await self.send_token_callback(page, token)
                    except Exception as dl_e:
                        self.logger.debug(f"验证码回调下载探测: {dl_e}")
                        try:
                            await self.send_token_callback(page, token)
                        except Exception:
                            pass
                    finally:
                        if cleanup_task_dir:
                            self._cleanup_task_download_dir(local_task_dir)
                else:
                    await self.send_token_callback(page, token)
                
                if wait_after_solve:
                    try:
                        await page.wait_for_load_state(
                            'domcontentloaded',
                            timeout=self.timeouts.get("load_state_timeout", 10) * 1000
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                
                still_has_cf = await self.is_cloudflare_verifying(page, detection_duration=1.0)
                if not still_has_cf:
                    self.logger.info("Cloudflare 验证码已成功解决")
                    return (True, None)
                else:
                    self.logger.warning("验证码发送后仍检测到 Cloudflare，可能需要重试")
                    
            except Exception as e:
                self.logger.error(f"解决 Cloudflare 验证码时出错 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        self.logger.error(f"Cloudflare 验证码解决失败，已尝试 {max_retries} 次")
        return (False, None)

    async def navigate_with_cf_bypass(self, page, url: str, filepath: str = None,
                                       timeout: int = None,
                                       wait_until: str = 'domcontentloaded',
                                       task_download_dir: Optional[str] = None) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        导航到URL并自动处理 Cloudflare 验证码。
        
        Args:
            page: Playwright 页面对象
            url: 目标 URL
            filepath: 可选的文件保存路径（如果验证后触发下载）
            timeout: 导航超时时间（毫秒），默认使用 self.timeout
            wait_until: 等待状态 ('load', 'domcontentloaded', 'networkidle')
            task_download_dir: 任务专属临时目录（并发安全）
            
        Returns:
            Tuple[bool, Optional[Response], Optional[str]]:
                - (True, response, None): 导航成功，无下载
                - (True, response, filepath): 导航成功，触发了下载
                - (False, response, None): 导航失败或验证码解决失败
        """
        if timeout is None:
            timeout = self.timeout * 1000
            
        response = None
        try:
            response = await page.goto(url, timeout=timeout, wait_until=wait_until)
            await asyncio.sleep(0.5)
            
            cf_result, downloaded_file = await self.solve_cloudflare_if_needed(
                page, filepath, task_download_dir=task_download_dir
            )
            
            if not cf_result:
                self.logger.warning(f"页面 {url} 的 Cloudflare 验证码解决失败")
                return (False, response, None)
            
            if downloaded_file:
                return (True, response, downloaded_file)
            
            return (True, response, None)
            
        except PlaywrightTimeoutError as e:
            self.logger.warning(f"导航到 {url} 超时: {e}")
            try:
                cf_result, downloaded_file = await self.solve_cloudflare_if_needed(
                    page, filepath, task_download_dir=task_download_dir
                )
                if cf_result:
                    return (True, response, downloaded_file)
            except Exception:
                pass
            return (False, response, None)
        except Exception as e:
            self.logger.error(f"导航到 {url} 时出错: {e}")
            return (False, response, None)

    async def ensure_page_ready(self, page, filepath: str = None,
                                task_download_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        确保页面已准备好（无 Cloudflare 验证码阻挡）。
        
        Args:
            page: Playwright 页面对象
            filepath: 可选的文件保存路径
            task_download_dir: 任务专属临时目录（并发安全）
            
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None): 页面已准备好
                - (True, filepath): 页面准备过程中触发了下载
                - (False, None): 页面未准备好（验证码解决失败）
        """
        return await self.solve_cloudflare_if_needed(
            page, filepath, task_download_dir=task_download_dir
        )

# 直接下载文件（对于直接提供PDF下载链接的情况），对于开放资源的最简洁方式，但是很多学术网站都收到保护
    async def download_direct(self, url: str, filepath: str, headers: Optional[Dict] = None,
                              title: Optional[str] = None, authors: Optional[List[str]] = None, 
                              year: Optional[int] = None) -> bool:   # 直接下载文件（对于直接提供PDF下载链接的情况）
        """直接下载文件（对于直接提供PDF下载链接的情况）
        
        Args:
            url: 下载链接
            filepath: 保存路径
            headers: 请求头
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            
        Returns:
            bool: 是否下载成功
        """
        self.logger.info(f"尝试直接下载: {url}")

        # 确保文件所在目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 设置默认请求头
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        
        try:
            # 使用aiohttp替代requests
            # 注意：aiohttp timeout 单位是秒，不是毫秒
            timeout = aiohttp.ClientTimeout(total=min(self.download_timeout, 60))  # 最多60秒
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    if response.status == 200:
                        # 使用异步方式写入文件
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                        
                        # 验证下载的文件
                        verified_path = filepath
                        is_valid = self.is_valid_pdf_file(filepath)
                        if not is_valid:
                            self.logger.warning(f"直接下载的文件验证失败: {verified_path}")
                            os.remove(filepath)
                            return False
                        else:
                            self.logger.info(f"直接下载的文件验证成功: {filepath}")
                        
                        # 更新文件路径（如果验证过程中修改了）
                        if verified_path != filepath:
                            filepath = verified_path
                        
                        # 计算并记录 MD5（用于去重/过滤）
                        md5 = self._compute_file_md5(filepath)
                        if md5:
                            existing_path = self.file_md5_index.get(md5)
                            if existing_path and existing_path != filepath and os.path.exists(existing_path):
                                self.logger.info(f"检测到重复文件 (MD5: {md5[:8]}...): {filepath}")
                            else:
                                self.file_md5_index[md5] = filepath
                                self._save_md5_index()
                        
                        # 更新下载历史
                        if title and authors:
                            file_hash = self._generate_paper_hash(title, authors)
                        else:
                            file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                        
                        self.download_history[file_hash] = {
                            'path': filepath,
                            'url': url,
                            'title': title,
                            'timestamp': time.time(),
                            'status': 'success',  # 添加状态字段
                            'md5': md5
                        }
                        self.save_download_history()
                        
                        self.logger.info(f"直接下载成功: {filepath}")
                        return True
                    else:
                        self.logger.warning(f"下载失败，HTTP状态码: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"直接下载过程中出错: {str(e)}")
            return False

    def _is_likely_pdf_url(self, url: str) -> bool:
        """
        判断 URL 是否可能指向 PDF 文件
        
        支持的模式：
        - 直接 .pdf 结尾
        - 包含 "pdf" 关键字
        - 已知出版商的特殊 URL 模式
        
        Args:
            url: 要检测的 URL
            
        Returns:
            True 如果 URL 可能指向 PDF，否则 False
        """
        if not url:
            return False
        
        url_lower = url.lower()
        
        # 1. 常规模式
        if url_lower.endswith('.pdf') or url_lower.endswith('.epub'):
            return True
        
        if "pdf" in url_lower:
            return True
        
        # 2. 已知出版商的特殊模式
        known_patterns = [
            "type=printable",      # PLOS: article/file?id=...&type=printable
            "type=pdf",            # 其他出版商
            "/file?",              # 通用文件下载接口
            "/download?",          # 下载接口
            "/download/",          # 下载路径（OJS/PKP 等期刊系统）
            "/article/download/",  # OJS 期刊系统下载链接
            "format=pdf",          # 指定格式参数
            "outputformat=pdf",    # BMJ 等
            "/getpdf/",            # Wiley 等
            "/pdfdirect/",         # 某些期刊
            "article_file",        # PLOS 旧版
            "/bitstreams/",        # DSpace/Scholarlis 机构仓库（如 memorial.scholaris.ca）
        ]
        
        for pattern in known_patterns:
            if pattern in url_lower:
                return True
        
        # 3. 域名特殊处理（某些域名的所有链接都是 PDF）
        pdf_domains = [
            "europepmc.org/articles/",  # Europe PMC 文章页直接是 PDF
            "biorxiv.org/content/",     # bioRxiv 内容页
            "medrxiv.org/content/",     # medRxiv 内容页
        ]
        
        for domain_pattern in pdf_domains:
            if domain_pattern in url_lower:
                return True
        
        return False

    def _is_ssrn_url(self, url: str) -> bool:
        """检测 URL 是否是 SSRN 链接

        Args:
            url: 要检测的 URL

        Returns:
            True 如果是 SSRN 链接，否则 False
        """
        if not url:
            return False
        url_lower = url.lower()
        return 'ssrn.com' in url_lower or 'papers.ssrn.com' in url_lower

# 使用brightdata的验证码解决器下载文件，用于解决Cloudflare验证码问题
    async def _brightdata_web_unlocker_raw(
        self,
        url: str,
        ZONE_NAME: str = "mx_webunlocker",
        api_key: Optional[str] = None,
        timeout_total: int = 300
    ) -> Tuple[Optional[bytes], int]:
        """
        使用 Bright Data Web Unlocker 拉取指定 URL 的原始响应内容（raw bytes）。

        注意：该接口为付费请求，请谨慎调用频率与重试次数。
        """
        if api_key is None:
            api_key = self.brightdata_api_key

        if not api_key:
            self.logger.warning("未配置 brightdata API key，无法使用 Web Unlocker")
            return None, 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "zone": ZONE_NAME,
            "url": url,
            "format": "raw"
        }

        try:
            # Bright Data 偶发返回 200 但 body 为空；此时做小幅重试可显著提升成功率
            for attempt in range(1, 4):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.brightdata.com/request",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout_total)
                    ) as response:
                        body = await response.read()
                        if response.status == 200:
                            if body:
                                return body, response.status
                            # 200 但空 body：短暂退避后重试
                            if attempt < 3:
                                self.logger.warning(
                                    f"Bright Data 返回 200 但内容为空，将重试 ({attempt}/3): {url}"
                                )
                                await asyncio.sleep(1.0 * attempt)
                                continue
                            return None, response.status
                        return None, response.status
        except Exception as e:
            self.logger.error(f"Bright Data Web Unlocker 请求失败: {e}")
            return None, 0

    def _normalize_scihub_url(self, maybe_url: str, mirror: str) -> str:
        """将 Sci-Hub 页面中提取到的相对/协议相对链接规范化为绝对 URL。"""
        if not maybe_url:
            return maybe_url
        if maybe_url.startswith("//"):
            return "https:" + maybe_url
        if maybe_url.startswith("/"):
            return mirror.rstrip("/") + maybe_url
        return maybe_url

    def _extract_scihub_pdf_url(self, html: str, mirror: str) -> Optional[str]:
        """
        从 Sci-Hub HTML 中提取 PDF 下载链接。

        Sci-Hub 页面结构可能变化，这里采用多策略兜底：
        - meta(citation_pdf_url)
        - iframe src
        - a[href*=".pdf"]
        - /download/...pdf 的正则匹配
        """
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")

            # 1) meta citation_pdf_url
            meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta and meta.get("content"):
                content = meta.get("content", "").strip()
                if content and "{pdf}" not in content and ".pdf" in content.lower():
                    return self._normalize_scihub_url(content, mirror)

            # 2) iframe src（Sci-Hub 常见）
            iframe = soup.find("iframe", src=True)
            if iframe and iframe.get("src"):
                src = iframe.get("src", "").strip()
                if src:
                    return self._normalize_scihub_url(src, mirror)

            # 3) a[href] 直接包含 pdf
            a_pdf = soup.find("a", href=lambda x: x and ".pdf" in x.lower())
            if a_pdf and a_pdf.get("href"):
                href = a_pdf.get("href", "").strip()
                if href:
                    return self._normalize_scihub_url(href, mirror)

            # 4) regex: /download/...pdf
            m = re.search(r"(/download/[^\"'\s>]+\.pdf[^\"'\s>]*)", html, flags=re.IGNORECASE)
            if m:
                return self._normalize_scihub_url(m.group(1), mirror)

        except Exception as e:
            self.logger.debug(f"解析 Sci-Hub HTML 提取 PDF 链接时出错: {e}")

        return None

    async def download_with_solver(self, url: str, filepath: str, api_key: str = None,
                                   ZONE_NAME: str = "mx_webunlocker", title: Optional[str] = None,
                                   authors: Optional[List[str]] = None, year: Optional[int] = None) -> bool:
        """使用验证码解决器下载文件

        Args:
            url: 下载链接  
            filepath: 保存路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份

        Returns:
            bool: 是否下载成功
        """
        # 如果未提供 api_key，使用配置文件中的值
        if api_key is None:
            api_key = self.brightdata_api_key
            
        self.logger.info(f"尝试使用验证码解决器下载: {url}")

        async def web_unlocker_request(url):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "zone": ZONE_NAME,
                "url": url,
                "format": "raw"
            }
            # Bright Data 偶发返回 200 但 body 为空；这里做小幅重试
            for attempt in range(1, 4):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.brightdata.com/request",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        body = await response.read()
                        if response.status == 200:
                            if body:
                                return body, response.status
                            if attempt < 3:
                                self.logger.warning(
                                    f"Bright Data 返回 200 但内容为空，将重试 ({attempt}/3): {url}"
                                )
                                await asyncio.sleep(1.0 * attempt)
                                continue
                            return None, response.status
                        return None, response.status

        try:
            # 获取解锁后的页面内容
            content, status = await web_unlocker_request(url)
            if status == 200 and content:
                # 直接将内容写入文件
                with open(filepath, 'wb') as f:
                    f.write(content)
                verified_path = filepath
                is_valid = self.is_valid_pdf_file(filepath)
                if not is_valid:
                    self.logger.warning(f"验证码解决器下载的文件验证失败: {verified_path}")
                    os.remove(filepath)
                    return False
                else:
                    self.logger.info(f"验证码解决器下载的文件验证成功: {filepath}")

                # 计算并记录 MD5（用于去重/过滤）
                md5 = self._compute_file_md5(filepath)
                if md5:
                    existing_path = self.file_md5_index.get(md5)
                    if existing_path and existing_path != filepath and os.path.exists(existing_path):
                        self.logger.info(f"检测到重复文件 (MD5: {md5[:8]}...): {filepath}")
                    else:
                        self.file_md5_index[md5] = filepath
                        self._save_md5_index()

                # 更新下载历史
                if title and authors:
                    file_hash = self._generate_paper_hash(title, authors)
                else:
                    file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()

                self.download_history[file_hash] = {
                    'path': filepath,
                    'url': url,
                    'title': title,
                    'timestamp': time.time(),
                    'status': 'success',  # 添加状态字段
                    'md5': md5
                }
                self.save_download_history()

                self.logger.info(f"使用解决器直接下载成功: {filepath}")
                return True
            else:
                self.logger.warning(f"验证码解决器下载失败，状态码: {status}")
                return False
        except Exception as e:
            self.logger.error(f"验证码解决器下载过程中出错: {str(e)}")
            return False
        
 # 通过 Sci-Hub 下载文件：传统直连经常被 DDoS-Guard/验证码拦截，这里直接使用 Bright Data Web Unlocker
    async def download_with_sci_hub(self, paper_url: str, filepath: str, 
                                   title: Optional[str] = None, 
                                   authors: Optional[List[str]] = None,
                                   year: Optional[int] = None) -> bool:
        """使用Sci-Hub下载论文
        
        Args:
            paper_url: 论文URL或DOI
            filepath: 保存路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            
        Returns:
            bool: 是否下载成功
        """
        self.logger.info(f"尝试通过Sci-Hub下载: {paper_url}")
        
        # 确保文件所在目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Sci-Hub镜像网址
        sci_hub_mirrors = [
            "https://sci-hub.st/"
        ]
        
        try:
            # 构建查询URL
            if paper_url.startswith('http'):
                query_url = paper_url
            elif paper_url.startswith('10.'):
                # DOI：优先使用原始 DOI（更贴近 Sci-Hub 的常见用法）
                query_url = paper_url
            else:
                query_url = paper_url
            
            # 尝试每个镜像
            for mirror in sci_hub_mirrors:
                try:
                    # 访问Sci-Hub
                    sci_hub_url = f"{mirror}{query_url}"
                    self.logger.info(f"访问Sci-Hub镜像（Bright Data Web Unlocker）: {sci_hub_url}")

                    html_bytes, status = await self._brightdata_web_unlocker_raw(
                        sci_hub_url,
                        ZONE_NAME="mx_webunlocker"
                    )

                    if status != 200 or not html_bytes:
                        self.logger.warning(f"通过 Bright Data 访问 Sci-Hub 失败，状态码: {status}")
                        continue

                    html = html_bytes.decode("utf-8", errors="ignore")

                    # 常见不可用提示
                    if "The requested resource is not available" in html:
                        self.logger.warning("Sci-Hub 页面提示资源不可用")
                        break

                    # 防御/挑战页特征（理论上 Web Unlocker 应该能处理，但这里兜底）
                    if re.search(r"ddos-guard|js-challenge|check\.ddos-guard|Checking your browser", html, flags=re.IGNORECASE):
                        self.logger.warning("Sci-Hub 仍返回 DDoS-Guard/挑战页，跳过该镜像")
                        continue

                    pdf_url = self._extract_scihub_pdf_url(html, mirror)
                    if not pdf_url:
                        self.logger.warning("在Sci-Hub页面中未找到PDF链接（Bright Data）")
                        continue

                    self.logger.info(f"已从 Sci-Hub 页面提取 PDF 链接: {pdf_url}")

                    # 直接复用现有 Bright Data 下载逻辑（会验证 PDF 并写入历史）
                    ok = await self.download_with_solver(
                        pdf_url,
                        filepath,
                        api_key=self.brightdata_api_key,
                        ZONE_NAME="mx_webunlocker",
                        title=title,
                        authors=authors,
                        year=year
                    )
                    if ok:
                        self.logger.info(f"从Sci-Hub下载成功（Bright Data）: {filepath}")
                        return True
                    
                except Exception as e:
                    self.logger.warning(f"尝试镜像 {mirror} 时出错: {str(e)}")
            
            self.logger.warning("所有Sci-Hub镜像尝试均失败")
            return False
            
        except Exception as e:
            self.logger.error(f"Sci-Hub下载过程中出错: {str(e)}")
            return False

    async def download_with_sci_hub_browser(self, paper_url: str, filepath: str, 
                                            title: Optional[str] = None, 
                                            authors: Optional[List[str]] = None,
                                            year: Optional[int] = None) -> Tuple[bool, str]:
        """[已废弃] Playwright 版 Sci-Hub 下载
        
        由于 DDoS-Guard 对自动化极不稳定，此方法已废弃。
        Sci-Hub 现统一通过 download_with_sci_hub (BrightData + sci-hub.st) 实现。
        保留此方法签名是为了兼容现有调用，直接返回失败。
        """
        self.logger.debug(f"[已废弃] Playwright Sci-Hub 方法被调用，直接跳过: {paper_url[:50]}...")
        return False, SCIHUB_ERROR

    # ==================== 会话池相关方法 ====================
    
    def _write_pdf_preferences(self, user_data_dir: str):
        """写入 PDF 下载首选项到用户数据目录
        
        确保 always_open_pdf_externally=True 生效，
        这样 PDF 链接会触发下载事件而不是在浏览器中内联显示。
        
        Args:
            user_data_dir: 浏览器用户数据目录
        """
        default_dir = os.path.join(user_data_dir, "Default")
        os.makedirs(default_dir, exist_ok=True)
        
        preferences_path = os.path.join(default_dir, "Preferences")
        preferences = {
            "plugins": {
                "always_open_pdf_externally": True
            },
            "download": {
                "prompt_for_download": False,
                "default_directory": self.download_dir
            },
            "profile": {
                "default_content_setting_values": {
                    "automatic_downloads": 1
                }
            }
        }
        
        # 如果已有 Preferences 文件，尝试合并
        if os.path.exists(preferences_path):
            try:
                with open(preferences_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                # 深度合并
                for key, value in preferences.items():
                    if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
                        existing[key].update(value)
                    else:
                        existing[key] = value
                preferences = existing
            except Exception as e:
                self.logger.warning(f"读取现有 Preferences 失败，将覆盖: {e}")
        
        with open(preferences_path, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2)
        self.logger.debug(f"已写入 PDF 首选项到: {preferences_path}")

    async def _init_session_pool(self):
        """初始化浏览器会话池（动态模式：按需创建，空闲释放）
        
        创建多个独立的持久化浏览器上下文，每个有独立的用户数据目录。
        这样可以：
        1. 保持 cookie/session（解决重定向问题）
        2. 正确应用 PDF 下载首选项
        3. 避免锁文件冲突（独立目录）
        """
        self.logger.debug(f"[DEBUG] _init_session_pool called, already_initialized={self._session_pool_initialized}")
        if self._session_pool_initialized:
            return
        
        self.logger.info(f"正在初始化会话池，大小: {self._session_pool_size}")
        
        for i in range(self._session_pool_size):
            # 每个会话有独立的用户数据目录
            user_data_dir = os.path.join(
                os.path.dirname(self.download_dir), 
                ".browser_data", 
                f"session_pool_{i}"
            )
            os.makedirs(user_data_dir, exist_ok=True)
            
            # 清理锁文件
            self._clean_lock_files_in_dir(user_data_dir)
            
            # 写入 PDF 首选项（必须在启动浏览器之前）
            self._write_pdf_preferences(user_data_dir)
            
            try:
                # 启动持久化上下文
                context = await self.browser_manager.launch_persistent_browser(
                    user_data_dir=user_data_dir,
                    browser_type=self.browser_type,
                    headless=self.headless,
                    stealth_mode=self.stealth_mode,
                    viewport={'width': 1280, 'height': 720},
                    downloads_path=self.download_dir,
                    timeout=self.download_timeout * 1000
                )
                
                session_data = {
                    'context': context,
                    'user_data_dir': user_data_dir,
                    'index': i,
                    'last_used': time.time(),  # 记录最后使用时间
                    'in_use': False  # 是否正在使用
                }
                self._session_pool_data.append(session_data)
                await self._session_pool.put(session_data)
                self.logger.info(f"会话池 [{i}] 初始化成功: {user_data_dir}")
                
            except Exception as e:
                self.logger.error(f"会话池 [{i}] 初始化失败: {e}")
                # 继续初始化其他会话
        
        self._session_pool_initialized = True
        self.logger.info(f"会话池初始化完成，可用会话数: {self._session_pool.qsize()}")
        
        # 启动后台清理任务
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动后台清理任务，定期检查并关闭空闲会话"""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return  # 已有清理任务在运行
        
        async def cleanup_loop():
            """定期检查空闲会话并释放资源"""
            while self._session_pool_initialized:
                await asyncio.sleep(30)  # 每 30 秒检查一次
                
                idle_time = time.time() - self._last_activity_time
                
                # 如果空闲超过阈值，关闭所有会话释放资源
                if idle_time > self._session_idle_timeout:
                    # 检查是否所有会话都空闲（不在使用中）
                    all_idle = all(not s.get('in_use', False) for s in self._session_pool_data)
                    
                    if all_idle and self._session_pool_initialized:
                        self.logger.info(f"会话池空闲超过 {self._session_idle_timeout} 秒，自动释放资源")
                        await self._close_session_pool()
                        break
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # 如果没有事件循环，忽略
            pass

    async def _acquire_session(self) -> Optional[Dict]:
        """从会话池获取一个可用会话
        
        如果池中没有可用会话，会阻塞等待直到有会话被释放。
        支持动态重新初始化（如果会话池已被空闲释放）。
        
        Returns:
            会话数据字典 {'context': BrowserContext, 'user_data_dir': str, 'index': int, ...}
            或 None（如果会话池未初始化）
        """
        # 更新最后活动时间
        self._last_activity_time = time.time()
        
        # 如果会话池已关闭，重新初始化
        if not self._session_pool_initialized:
            await self._init_session_pool()
        
        if self._session_pool.empty() and not self._session_pool_data:
            self.logger.error("会话池为空且无法初始化")
            return None
        
        try:
            # 阻塞等待可用会话
            session = await asyncio.wait_for(
                self._session_pool.get(),
                timeout=self.timeouts.get("session_acquire_timeout", 15)
            )
            session['in_use'] = True
            session['last_used'] = time.time()
            self.logger.debug(f"获取会话 [{session['index']}]")
            return session
        except asyncio.TimeoutError:
            self.logger.error("获取会话超时")
            return None

    async def _release_session(self, session: Dict):
        """释放会话回会话池
        
        Args:
            session: 之前通过 _acquire_session 获取的会话数据
        """
        if session:
            session['in_use'] = False
            session['last_used'] = time.time()
            self._last_activity_time = time.time()  # 更新最后活动时间
            await self._session_pool.put(session)
            self.logger.debug(f"释放会话 [{session['index']}]")

    async def _close_session_pool(self):
        """关闭会话池中的所有上下文，立即释放资源"""
        if not self._session_pool_initialized:
            return
        
        self.logger.info("正在关闭会话池...")
        
        # 取消清理任务
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 并行关闭所有会话以加速
        close_tasks = []
        for session_data in self._session_pool_data:
            close_tasks.append(self._close_single_session(session_data))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._session_pool_data.clear()
        # 清空队列
        while not self._session_pool.empty():
            try:
                self._session_pool.get_nowait()
            except:
                break
        
        self._session_pool_initialized = False
        self.logger.info("会话池已关闭，资源已释放")

    async def _close_single_session(self, session_data: Dict):
        """关闭单个会话（辅助方法）"""
        try:
            context = session_data.get('context')
            if context:
                # 快速关闭：直接关闭上下文（会自动关闭所有页面）
                await asyncio.wait_for(context.close(), timeout=5)
                self.logger.debug(f"已关闭会话 [{session_data['index']}]")
        except asyncio.TimeoutError:
            self.logger.warning(f"关闭会话 [{session_data.get('index', '?')}] 超时，强制跳过")
        except Exception as e:
            self.logger.warning(f"关闭会话 [{session_data.get('index', '?')}] 时出错: {e}")

    async def _create_lightweight_context(self):
        """创建轻量级临时上下文（非持久化，下载完立即释放）
        
        适用于：
        - 单次下载任务
        - 不需要保持 cookie 的场景
        - 需要快速释放资源的场景
        
        Returns:
            (context, cleanup_func) - 上下文和清理函数
        """
        self.logger.debug("创建轻量级临时上下文")
        
        try:
            # 使用临时目录
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="browser_temp_")
            
            # 写入 PDF 首选项
            self._write_pdf_preferences(temp_dir)
            
            # 创建临时持久化上下文（为了正确处理下载）
            context = await self.browser_manager.launch_persistent_browser(
                user_data_dir=temp_dir,
                browser_type=self.browser_type,
                headless=self.headless,
                stealth_mode=self.stealth_mode,
                viewport={'width': 1280, 'height': 720},
                downloads_path=self.download_dir,
                timeout=30000  # 较短的超时时间
            )
            
            async def cleanup():
                """清理临时上下文和目录"""
                try:
                    await asyncio.wait_for(context.close(), timeout=5)
                except:
                    pass
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
                self.logger.debug("轻量级上下文已清理")
            
            return context, cleanup
            
        except Exception as e:
            self.logger.error(f"创建轻量级上下文失败: {e}")
            return None, None

    def set_idle_timeout(self, timeout_seconds: int):
        """动态设置空闲超时时间
        
        Args:
            timeout_seconds: 空闲超时秒数（建议 30-300）
        """
        self._session_idle_timeout = max(10, timeout_seconds)
        self.logger.info(f"空闲超时已设置为 {self._session_idle_timeout} 秒")

    async def force_release_all(self):
        """强制释放所有浏览器资源（立即执行）
        
        用于：
        - 批量下载完成后主动释放资源
        - 出错后清理资源
        """
        self.logger.info("强制释放所有浏览器资源...")
        
        # 关闭会话池
        if self._session_pool_initialized:
            await self._close_session_pool()
        
        # 关闭旧的浏览器上下文（兼容）
        if self._browser_context:
            try:
                await asyncio.wait_for(self._browser_context.close(), timeout=5)
            except:
                pass
            self._browser_context = None
        
        self.logger.info("所有浏览器资源已释放")

    # ==================== 会话池相关方法结束 ====================

    async def get_browser_context(self):
        """获取或创建浏览器上下文"""
        async with self._context_lock:
            if self._browser_context is not None:
                try:
                    # 验证现有的会话是否有效
                    page = await self._browser_context.new_page()
                    await page.goto('about:blank')
                    await page.close()
                    self.logger.info("使用现有的浏览器上下文")
                    return self._browser_context
                except Exception as e:
                    # 会话已失效，需要重新创建
                    self.logger.warning(f"现有浏览器上下文已失效，需要重新创建: {str(e)}")
                    try:
                        await self._browser_context.close()
                    except:
                        pass
                    self._browser_context = None
                    self._browser_manager = None

            try:
                if self.persist_browser:
                    # 确保用户数据目录存在
                    os.makedirs(self._persistent_user_data_dir, exist_ok=True)
                    
                    # 清理锁定文件
                    self._clean_lock_files_in_dir(self._persistent_user_data_dir)
                    
                    self._browser_context = await self.browser_manager.launch_persistent_browser(
                        user_data_dir=self._persistent_user_data_dir,
                        browser_type=self.browser_type,
                        headless=self.headless,  # 确保这个参数被正确传递和使用
                        stealth_mode=self.stealth_mode,
                        viewport={'width': 1280, 'height': 720},
                        downloads_path=self.download_dir,  # 使用主下载目录
                        timeout=self.download_timeout*1000
                    )
                else:
                    self._browser_context = await setup_browser(
                        browser_type=self.browser_type,
                        headless=self.headless,  # 确保这个参数被正确传递和使用
                        browser_channel=self.browser_type,
                        stealth_mode=self.stealth_mode,
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                        viewport={'width': 1280, 'height': 720},
                        downloads_path=self.download_dir
                    )
                self.logger.info("成功创建浏览器上下文")
                return self._browser_context
            except Exception as e:
                self.logger.error(f"创建浏览器上下文失败: {str(e)}")
                self._browser_context = None
                self._browser_manager = None
                return None

    async def cleanup_resources(self, force_close=False, ask_user=False):
        """清理所有资源，包括浏览器上下文和临时文件"""
        self.logger.info("开始清理资源...")
        
        # 检查是否有正在进行的下载
        has_active_downloads = False
        
        # 检查下载目录中是否有部分下载的文件
        download_dir = self.download_dir
        if os.path.exists(download_dir):
            part_files = [f for f in os.listdir(download_dir) 
                        if f.lower().endswith('.part') or 
                        f.lower().endswith('.crdownload') or
                        f.lower().endswith('.download')]
            
            if part_files:
                self.logger.info(f"检测到正在下载的文件: {part_files}")
                has_active_downloads = True

        # 如果有活动下载且不是强制关闭，并且用户需要询问
        if has_active_downloads and not force_close and ask_user and self.persist_browser:
            self.logger.info("检测到正在进行的下载")
            
            if self.show_browser:
                try:
                    response = input("检测到正在进行的下载。是否关闭浏览器？(y/N): ").strip().lower()
                    if response != 'y':
                        self.logger.info("用户选择保留浏览器会话")
                        try:
                            self.save_download_history()
                            self.logger.info("下载历史已保存")
                        except Exception as e:
                            self.logger.error(f"保存下载历史时出错: {str(e)}")
                        
                        self.logger.info("资源部分清理完成（保留浏览器会话以完成下载）")
                        return
                except Exception as e:
                    self.logger.warning(f"询问用户时出错: {str(e)}")

        # 如果有活动下载且不是强制关闭，则不关闭浏览器上下文
        if has_active_downloads and not force_close and self.persist_browser:
            self.logger.info("检测到正在进行的下载，保留浏览器会话")
            try:
                self.save_download_history()
                self.logger.info("下载历史已保存")
            except Exception as e:
                self.logger.error(f"保存下载历史时出错: {str(e)}")
            
            self.logger.info("资源部分清理完成（保留浏览器会话以完成下载）")
            return

        # 关闭会话池（新增）
        if self._session_pool_initialized:
            try:
                await self._close_session_pool()
            except Exception as e:
                self.logger.warning(f"关闭会话池时出错: {e}")

        # 关闭所有活动页面（兼容旧逻辑）
        if self._browser_context:
            try:
                # 获取所有页面（包括blank页面）
                all_pages = self._browser_context.pages
                for page in all_pages:
                    try:
                        # 先导航到空白页，确保页面完全卸载
                        try:
                            await page.goto('about:blank', timeout=3000)
                        except:
                            pass
                        await page.close()
                        self.logger.debug(f"已关闭页面")
                    except Exception as e:
                        self.logger.warning(f"关闭页面时出错: {str(e)}")
                
                # 增加延迟到 2 秒，让浏览器完成所有后台操作
                # 这对于 persistent context 的优雅关闭很重要
                await asyncio.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"关闭页面时出错: {str(e)}")

        # 关闭浏览器（通过 BrowserManager 统一管理，避免双重关闭）
        # 对于 persistent context，_browser_context 和 browser_manager.browsers[0] 是同一个对象
        # 只通过 browser_manager.close() 关闭一次，避免崩溃
        if self.browser_manager is not None:
            try:
                self.logger.info("正在关闭 BrowserManager...")
                await self.browser_manager.close()
                self._browser_context = None  # 清除引用
                self.logger.info("BrowserManager 已关闭")
            except Exception as e:
                self.logger.warning(f"关闭 BrowserManager 时出错: {str(e)}")
                self._browser_context = None

        # 清理残留的任务临时目录
        try:
            for name in os.listdir(self.download_dir):
                if name.startswith('.task_'):
                    task_dir = os.path.join(self.download_dir, name)
                    if os.path.isdir(task_dir):
                        self.logger.info(f"清理残留任务目录: {task_dir}")
                        shutil.rmtree(task_dir, ignore_errors=True)
        except Exception as e:
            self.logger.debug(f"清理任务目录时出错: {str(e)}")

        # 保存下载历史
        try:
            self.save_download_history()
            self.logger.info("下载历史已保存")
        except Exception as e:
            self.logger.error(f"保存下载历史时出错: {str(e)}")

        self.logger.info("资源清理完成")

    async def pdf_download_with_browser(self, url: str, filepath: str) -> bool:
        """使用浏览器下载 PDF 文件
        
        使用会话池获取浏览器上下文，保证 cookie/session 持续性和 PDF 设置生效。
        """
        self.logger.debug(f"[DEBUG] pdf_download_with_browser: url={url}, filepath={filepath}")
        downloads_dir = os.path.dirname(filepath)
        os.makedirs(downloads_dir, exist_ok=True)
        finished = False
        start_time = time.time()
        max_total_time = self.timeouts.get("paper_total_timeout", 120)
        page = None
        session = None
        
        # 创建任务专属临时目录（并发安全）
        task_download_dir, task_start_time, initial_main_files = self._create_task_download_dir()
        
        try:
            # 从会话池获取上下文（替代单一 _browser_context）
            session = await self._acquire_session()
            if session is None:
                self.logger.error("无法从会话池获取浏览器上下文")
                return False
            
            context = session['context']

            # 创建新页面
            page = await context.new_page()
            self._active_pages.add(page)

            # 根据 stealth_mode 应用相应模式（兼容不同 playwright-stealth 版本）
            if self.stealth_mode:
                self.logger.debug("应用 stealth 模式到页面")
                try:
                    if _STEALTH_MODE == "v2" and _Stealth is not None:
                        await _Stealth().apply_stealth_async(page)
                    elif _STEALTH_MODE == "async" and _stealth_async is not None:
                        await _stealth_async(page)
                    elif _STEALTH_MODE == "sync" and _stealth_sync is not None:
                        _stealth_sync(page)
                except Exception as e:
                    self.logger.warning(f"playwright-stealth 应用失败，将继续执行（不致命）: {e}")
            else:
                self.logger.info("不应用 stealth 模式到页面")

            cloudflare_protection = False
            # 先检测是否存在 Cloudflare 保护（JS Challenge 或 Turnstile）
            
            # 预热步骤：先访问网站主页处理 cookie 同意，避免 ERR_TOO_MANY_REDIRECTS
            try:
                if time.time() - start_time > max_total_time:
                    self.logger.warning("pdf_download_with_browser 超时，提前退出")
                    return False
                from urllib.parse import urlparse
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                self.logger.info(f"预热：先访问主站 {base_url} 处理 cookie")
                self.logger.debug(f"[DEBUG] 预热访问: base_url={base_url}")
                warmup_response = await page.goto(
                    base_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeouts.get("goto_timeout", 20) * 1000
                )
                self.logger.debug(f"[DEBUG] 预热完成: status={warmup_response.status if warmup_response else None}, url={page.url}")
                
                # 尝试处理 cookie 同意弹窗
                await self._handle_cookie_consent(page, context_label="预热")
                
                # 检测并处理预热页面的 Cloudflare 验证
                cf_success, _ = await self.solve_cloudflare_if_needed(
                    page, task_download_dir=task_download_dir
                )
                if not cf_success:
                    self.logger.warning("预热页面 Cloudflare 验证解决失败，继续尝试")
                
                self.logger.info("预热完成，开始访问 PDF URL")
            except Exception as warmup_e:
                self.logger.warning(f"预热访问失败，继续尝试直接下载: {warmup_e}")
            
            # 设置下载事件监听器（确保所有下载都被捕获到任务目录）
            latest_download = {"obj": None}
            def _on_download_capture(download):
                latest_download["obj"] = download
                self.logger.debug(f"[下载监听] 捕获下载事件: {download.suggested_filename}")
            page.on("download", _on_download_capture)
            
            # 尝试初始下载
            self.logger.info("开始尝试初始下载")
            response = None
            try:
                # 初始下载等待时间上限，避免过久阻塞
                download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                async with page.expect_download(timeout=download_timeout_ms) as download_info:
                    try: 
                        response = await page.goto(url, wait_until="domcontentloaded") 
                        await self._handle_cookie_consent(page, context_label="跳转后")
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                    except Exception as goto_e:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                download = await download_info.value
                # 使用统一的保存方法（并发安全）
                if await self._save_and_validate_download(download, filepath, task_download_dir, "initial_download"):
                    self.logger.info(f"初始下载尝试成功保存到: {filepath}")
                    finished = True
                    return True
                else:
                    self.logger.warning(f"初始下载的文件无效")
                    finished = False

            except Exception as e:
                self.logger.error(f"初始下载失败：{e}, 检查是否404错误")
                # expect_download 超时后，下载可能已在后台落盘；先等待其稳定，避免过早关闭页面导致 Target closed
                if latest_download.get("obj") is not None:
                    dl = latest_download["obj"]
                    try:
                        await asyncio.wait_for(dl.path(), timeout=30.0)
                    except Exception:
                        pass
                try:
                    analysis = await self.page_analyzer.analyze(page)
                    self.logger.info(f"[观察点-初始下载失败] {self.page_analyzer.format_analysis(analysis)}")
                except Exception as e:
                    self.logger.debug(f"页面分析失败: {e}")
                
                # 检查是否是404错误（注意：403 可能是 Cloudflare 验证，不要直接返回）
                try:
                    if response and response.status == 404:
                        self.logger.error(f"URL返回404错误，终止下载尝试: {url}")
                        return False
                    elif response and response.status == 403:
                        # 403 可能是 Cloudflare 验证，继续后续处理
                        self.logger.warning(f"URL返回403，可能是Cloudflare验证，继续尝试: {url}")
                except Exception as nav_e:
                    self.logger.warning(f"检查页面状态时出错: {nav_e}")
                
                # 检查是否是 PDF 内联显示（Content-Disposition: inline）
                # 某些网站（如 Cell.com）直接在浏览器中渲染 PDF 而不触发下载事件
                # 增强版：使用 JS fetch 替代 response.body()，更可靠
                try:
                    # 先检查当前页面的 content-type
                    content_type = ''
                    if response:
                        content_type = response.headers.get('content-type', '')
                    
                    # 如果响应是 PDF 或页面 URL 以 .pdf 结尾
                    is_pdf_url = url.lower().endswith('.pdf')
                    is_pdf_content = 'application/pdf' in content_type.lower()
                    
                    if is_pdf_content or is_pdf_url:
                        self.logger.info(f"检测到 PDF 内联显示 (content-type: {content_type}, url ends with .pdf: {is_pdf_url})")
                        self.logger.info("尝试使用 JS fetch 获取 PDF 内容...")
                        
                        try:
                            # 使用 JS fetch 在页面上下文中获取 PDF（保持 cookie/session）
                            pdf_bytes = await asyncio.wait_for(
                                page.evaluate("""
                                async (url) => {
                                    try {
                                        const resp = await fetch(url, {
                                            credentials: 'include',
                                            headers: {
                                                'Accept': 'application/pdf'
                                            }
                                        });
                                        if (!resp.ok) {
                                            console.error('Fetch failed:', resp.status);
                                            return null;
                                        }
                                        const buf = await resp.arrayBuffer();
                                        return Array.from(new Uint8Array(buf));
                                    } catch (e) {
                                        console.error('Fetch error:', e);
                                        return null;
                                    }
                                }
                            """, url),
                                timeout=self.timeouts.get("inline_pdf_fetch_timeout", 15)
                            )
                            
                            if pdf_bytes and len(pdf_bytes) > 1000:
                                with open(filepath, 'wb') as f:
                                    f.write(bytes(pdf_bytes))
                                if self.is_valid_pdf_file(filepath):
                                    self.logger.info(f"JS fetch 回退下载成功: {filepath}")
                                    finished = True
                                    return True
                                else:
                                    self.logger.warning("JS fetch 下载的文件验证失败，继续尝试其他方法")
                                    try:
                                        os.remove(filepath)
                                    except:
                                        pass
                            else:
                                self.logger.warning(f"JS fetch 返回内容过小或为空: {len(pdf_bytes) if pdf_bytes else 0} bytes")
                        except Exception as fetch_e:
                            self.logger.warning(f"JS fetch 获取 PDF 失败: {fetch_e}")
                            
                            # 回退：尝试 response.body()
                            if response:
                                try:
                                    pdf_content = await asyncio.wait_for(
                                        response.body(),
                                        timeout=self.timeouts.get("inline_pdf_body_timeout", 10)
                                    )
                                    if pdf_content and len(pdf_content) > 1000:
                                        with open(filepath, 'wb') as f:
                                            f.write(pdf_content)
                                        if self.is_valid_pdf_file(filepath):
                                            self.logger.info(f"response.body() 回退下载成功: {filepath}")
                                            finished = True
                                            return True
                                except Exception as body_e:
                                    self.logger.warning(f"response.body() 回退也失败: {body_e}")
                except Exception as inline_e:
                    self.logger.warning(f"内联 PDF 检测/获取过程出错: {inline_e}")
                
                self.logger.info("非404错误，继续尝试其他下载方法")
                finished = False

            # 使用通用方法检测和解决 Cloudflare 保护
            try:
                self.logger.info("使用通用方法检测 Cloudflare 保护状态")
                cf_success, cf_downloaded_file = await self.solve_cloudflare_if_needed(
                    page, filepath, task_download_dir=task_download_dir
                )
                
                if cf_downloaded_file and self.is_valid_pdf_file(cf_downloaded_file):
                    self.logger.info(f"Cloudflare 验证后触发下载，文件已保存: {cf_downloaded_file}")
                    finished = True
                    return True
                elif not cf_success:
                    self.logger.warning("Cloudflare 验证码解决失败，继续尝试其他方法")
                    finished = False
            except Exception as e:
                self.logger.error(f"解决Cloudflare下载策略失败: {e}")
                finished = False

            # 如果初始下载失败，使用智能循环处理复杂场景
            if not finished:
                self.logger.info("初始下载未成功，启用智能循环...")
                try:
                    finished = await self._smart_download_loop(
                        page=page, filepath=filepath, url=url,
                        max_iterations=3, max_total_time=60,
                        task_download_dir=task_download_dir,
                        initial_main_files=initial_main_files,
                        task_start_time=task_start_time
                    )
                    if finished:
                        self.logger.info(f"智能循环下载成功: {filepath}")
                        return True
                except Exception as smart_e:
                    self.logger.warning(f"智能循环下载失败: {smart_e}")
                    finished = False

        except Exception as e:
            self.logger.exception(f"下载 PDF 文件时发生未预料的错误: {e}")
            finished = False
        finally:
            # 等待可能的下载完成（避免过早清理/关闭）
            try:
                if not finished:
                    if latest_download.get("obj") is not None:
                        dl = latest_download["obj"]
                        latest_download["obj"] = None
                        if await self._save_and_validate_download(
                            dl, filepath, task_download_dir, "finally-cleanup"
                        ):
                            finished = True
                    if not finished and await self._check_download_in_progress(
                        filepath, task_download_dir=task_download_dir
                    ):
                        await self._wait_for_download_complete(
                            filepath,
                            timeout=min(self.timeouts.get("download_complete_timeout", 20), 10),
                            task_download_dir=task_download_dir,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time
                        )
            except Exception as wait_e:
                self.logger.debug(f"等待下载完成失败（清理前）: {wait_e}")

            # 清理任务专属临时目录
            self._cleanup_task_download_dir(task_download_dir)
            
            # 移除下载监听器
            if page:
                try:
                    page.off("download", _on_download_capture)
                except Exception:
                    pass
            
            # 关闭页面
            if page:
                try:
                    await page.close()
                    self._active_pages.discard(page)
                except Exception as close_e:
                    self.logger.warning(f"关闭页面时出错: {close_e}")
            
            # 释放会话回会话池（重要！否则会话会被耗尽）
            if session:
                await self._release_session(session)

        return finished


    def _generate_filename(self, title: Optional[str], authors: Optional[List[str]], year: Optional[int]) -> str:
        """根据论文信息生成标准化文件名
        
        Args:
            title: 论文标题
            authors: 论文作者列表
            year: 发表年份
            
        Returns:
            str: 标准化的文件名（不含扩展名）
        """
        filename_parts = []
        
        # 添加年份（如果有）
        if year:
            filename_parts.append(f"{year}")
        
        # 添加作者（如果有）
        if authors and len(authors) > 0:
            # 使用第一作者姓氏
            first_author = authors[0].strip()
            if " " in first_author:
                # 假设格式是"名 姓"，取姓
                last_name = first_author.split(" ")[-1]
            else:
                last_name = first_author
            
            # 如果有多个作者，添加"等"
            if len(authors) > 1:
                author_part = f"{last_name}_et_al"
            else:
                author_part = last_name
            
            filename_parts.append(author_part)
        
        # 添加标题（如果有）
        if title:
            # 清理标题，移除非法字符
            clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
            # 缩短标题（最多50个字符）
            if len(clean_title) > 35:
                clean_title = "_".join(clean_title.split(" ")[0:6])
            
            filename_parts.append(clean_title)
        
        # 如果没有任何有效部分，使用时间戳
        if not filename_parts:
            filename_parts.append(f"paper_{int(time.time())}")
        
        # 组合文件名部分
        filename = "_".join(filename_parts)
        
        # 确保文件名不超过255个字符（考虑到后面会加.pdf）
        if len(filename) > 250:
            filename = filename[:250]
        
        return filename

            
    def _clean_lock_files_in_dir(self, directory: str):
        """清理目录中的锁定文件
        
        Args:
            directory: 浏览器用户数据目录
        """
        try:
            if os.path.exists(directory):
                self.logger.info(f"清理目录中的锁定文件: {directory}")
                # 删除可能的锁定文件
                lock_files = [
                    "SingletonLock",
                    "SingletonCookie",
                    "SingletonSocket",
                    ".com.google.Chrome.sUifpX",
                    "lockfile"
                ]
                
                for lock_file in lock_files:
                    lock_path = os.path.join(directory, lock_file)
                    if os.path.exists(lock_path):
                        try:
                            os.remove(lock_path)
                            self.logger.info(f"已删除锁定文件: {lock_path}")
                        except Exception as e:
                            self.logger.warning(f"删除锁定文件失败: {lock_path}, 错误: {str(e)}")
        except Exception as e:
            self.logger.warning(f"清理锁定文件时出错: {str(e)}")
            
    def is_valid_pdf_file(self, file_path: str) -> bool:
        """检查文件是否是有效的PDF文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否是有效的PDF文件
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                return False
                
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # PDF头部至少需要几十个字节
                self.logger.warning(f"文件过小 ({file_size} 字节)，不可能是有效的PDF")
                return False
                
            # 读取文件头部
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # 读取前1KB
                
            # PDF文件头通常以%PDF-开头，后面跟着版本号（如%PDF-1.4）
            if not header.startswith(b'%PDF-'):
                self.logger.warning(f"文件不是PDF格式: {file_path}")
                return False
                
            # 检查PDF是否包含基本结构
            if not (b'obj' in header or b'stream' in header or b'/Type' in header or b'/Pages' in header):
                self.logger.warning(f"文件缺少基本PDF结构: {file_path}")
                return False
                
            # 文件大小合理性检查
            if file_size < 1000:  # 小于1KB的PDF文件很可能是无效的
                self.logger.warning(f"文件大小异常 ({file_size} 字节): {file_path}")
                return False
                
            self.logger.debug(f"文件验证通过: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"检查PDF文件时出错: {str(e)}")
            return False

    def verify_downloaded_file(self, file_path: str) -> Tuple[bool, str]:
        """验证下载的文件是否是有效的学术论文
        
        Args:
            file_path: 文件路径
            
        Returns:
            Tuple[bool, str]: (是否有效, 文件路径或错误消息)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"文件不存在: {file_path}"
                
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # 小于1KB
                return False, f"文件过小 ({file_size} 字节)，可能不是有效论文"
                
            # 检查文件扩展名
            file_ext = os.path.splitext(file_path.lower())[1]
            
            # 根据不同扩展名进行不同的验证
            if file_ext == '.pdf':
                is_valid = self.is_valid_pdf_file(file_path)
                if not is_valid:
                    return False, f"文件不是有效的PDF格式: {file_path}"
            elif file_ext == '.epub':
                # EPUB格式通常不需要特别验证，扩展名和文件大小验证就足够了
                # 文件至少应该有基本的EPUB结构，简单检查一下
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if not (header.startswith(b'PK\x03\x04') and (b'mimetype' in header or b'.epub' in header.lower())):
                        return False, f"文件不是有效的EPUB格式: {file_path}"
            elif file_ext == '.djvu':
                # DJVU格式检查
                with open(file_path, 'rb') as f:
                    header = f.read(16)
                    if not header.startswith(b'AT&TFORM'):
                        return False, f"文件不是有效的DjVu格式: {file_path}"
            elif file_ext in ('.zip', '.rar', '.7z'):
                # 压缩文件可能包含论文，但我们无法验证内容
                self.logger.info(f"下载的是压缩文件，无法验证内容: {file_path}")
            else:
                # 其他格式，尝试检查是否为PDF但扩展名不正确
                if self.is_valid_pdf_file(file_path):
                    # 是PDF但扩展名不正确，添加.pdf扩展名
                    new_path = f"{file_path}.pdf"
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    shutil.move(file_path, new_path)
                    self.logger.info(f"文件是PDF格式但扩展名不正确，已重命名: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
                    return True, new_path
                # 尝试检查是否为EPUB
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if header.startswith(b'PK\x03\x04') and (b'mimetype' in header or b'.epub' in header.lower()):
                        new_path = f"{file_path}.epub"
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        shutil.move(file_path, new_path)
                        self.logger.info(f"文件是EPUB格式但扩展名不正确，已重命名: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
                        return True, new_path
                # 不是已知的论文格式
                return False, f"文件格式不符合论文类型（非PDF/EPUB/DjVu）: {file_path}"
                    
            # 通过所有检查
            return True, file_path
            
        except Exception as e:
            return False, f"验证文件时出错: {str(e)}"

    # =========================================================================
    # MD5 去重方法
    # =========================================================================
    
    def _compute_file_md5(self, filepath: str) -> Optional[str]:
        """计算文件 MD5"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.warning(f"计算文件 MD5 失败: {e}")
            return None
    
    def _load_md5_index(self):
        """从下载目录加载已有文件的 MD5 索引"""
        self.file_md5_index = {}
        md5_index_file = os.path.join(self.download_dir, ".md5_index.json")
        
        # 尝试从缓存文件加载
        if os.path.exists(md5_index_file):
            try:
                with open(md5_index_file, 'r') as f:
                    self.file_md5_index = json.load(f)
                self.logger.info(f"已加载 {len(self.file_md5_index)} 个文件的 MD5 索引")
                return
            except Exception as e:
                self.logger.warning(f"加载 MD5 索引缓存失败: {e}")
        
        # 扫描目录构建索引
        pdf_count = 0
        for filename in os.listdir(self.download_dir):
            if filename.lower().endswith(('.pdf', '.epub', '.djvu')):
                filepath = os.path.join(self.download_dir, filename)
                md5 = self._compute_file_md5(filepath)
                if md5:
                    self.file_md5_index[md5] = filepath
                    pdf_count += 1
        
        self.logger.info(f"已扫描并索引 {pdf_count} 个文件")
        self._save_md5_index()
    
    def _save_md5_index(self):
        """保存 MD5 索引到文件"""
        md5_index_file = os.path.join(self.download_dir, ".md5_index.json")
        try:
            with open(md5_index_file, 'w') as f:
                json.dump(self.file_md5_index, f, indent=2)
        except Exception as e:
            self.logger.warning(f"保存 MD5 索引失败: {e}")
    
    def check_file_duplicate(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        检查文件是否与已有文件重复（基于 MD5）
        
        Args:
            filepath: 要检查的文件路径
        
        Returns:
            (is_duplicate, existing_filepath) - 如果重复，返回 (True, 已有文件路径)
        """
        md5 = self._compute_file_md5(filepath)
        if not md5:
            return False, None
        
        if md5 in self.file_md5_index:
            existing_path = self.file_md5_index[md5]
            if existing_path != filepath and os.path.exists(existing_path):
                self.logger.info(f"检测到重复文件 (MD5: {md5[:8]}...): {filepath}")
                return True, existing_path
        
        # 不重复，添加到索引
        self.file_md5_index[md5] = filepath
        self._save_md5_index()
        return False, None

    
    # =========================================================================
    # Anna's Archive 下载方法
    # =========================================================================
    
    async def _annas_search_md5(self, query: str) -> Optional[str]:
        """在 Anna's Archive 搜索并提取 MD5（仅限 DOI 查询，用于 Sci-Hub 联合）"""
        import urllib.parse
        
        doi_pattern = re.compile(r'10\.\d{4,9}/\S+', re.IGNORECASE)
        doi_match = doi_pattern.search(query or "")
        if not doi_match:
            self.logger.info("Anna's Archive MD5 搜索跳过：query 不是 DOI")
            return None

        target_doi = doi_match.group(0).lower().rstrip(').,;')
        encoded_query = urllib.parse.quote(target_doi, safe='')

        counts_url = f"https://annas-archive.pm/dyn/search_counts?q={encoded_query}"
        search_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        json_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json,text/plain,*/*",
        }

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except Exception:
                return 0

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(counts_url, headers=json_headers) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Anna's Archive counts 返回状态码: {resp.status}")
                        return None
                    data = await resp.json(content_type=None)

                journals_value = _safe_int(data.get("aarecords_journals", {}).get("value", 0))
                records_value = _safe_int(data.get("aarecords", {}).get("value", 0))

                if journals_value > 0:
                    target_index = "journals"
                elif records_value > 0:
                    target_index = ""
                else:
                    self.logger.info("Anna's Archive counts 未命中该 DOI")
                    return None

                if target_index:
                    search_url = f"https://annas-archive.pm/search?index={target_index}&q={encoded_query}"
                else:
                    search_url = f"https://annas-archive.pm/search?index=&q={encoded_query}"

                async with session.get(search_url, headers=search_headers) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Anna's Archive 搜索返回状态码: {resp.status}")
                        return None

                    html = await resp.text()
                    if "ddos-guard" in html.lower():
                        self.logger.warning("Anna's Archive 被 DDoS-Guard 阻挡")
                        return None

                    patterns = [
                        re.compile(r'href="(?:https?://[^"]*)?/md5/([a-f0-9]{32})"', re.IGNORECASE),
                        re.compile(r'const\s+md5\s*=\s*"([a-f0-9]{32})"', re.IGNORECASE),
                        re.compile(r'aarecord_id\s*=\s*"md5:([a-f0-9]{32})"', re.IGNORECASE),
                    ]
                    for pattern in patterns:
                        matches = pattern.findall(html)
                        if matches:
                            md5 = matches[0].lower()
                            if len(matches) > 1:
                                self.logger.debug(
                                    f"Anna's Archive 匹配到多个 MD5（返回第一个）: {len(matches)}"
                                )
                            self.logger.info(f"Anna's Archive 找到 MD5: {md5}")
                            return md5

                    self.logger.info("Anna's Archive 未找到 MD5")
                    return None
        except Exception as e:
            self.logger.warning(f"Anna's Archive 搜索出错: {e}")
            return None

    async def annas_keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """使用关键词在 Anna's Archive 搜索，支持多页结果汇总"""
        import urllib.parse

        if not query:
            return []

        encoded_query = urllib.parse.quote(query, safe='')
        base_url = "https://annas-archive.li"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }

        try:
            results: List[Dict] = []
            seen_md5 = set()
            max_pages = int(self.config.get("annas_keyword_max_pages", 5))
            page = 1

            # 匹配 MD5 链接（同时支持相对和绝对路径）
            md5_pattern = re.compile(
                r'href="(?:https?://[^"]*)?/md5/([a-f0-9]{32})"',
                re.IGNORECASE
            )

            next_pattern = re.compile(r'aria-label="Next"|>\s*Next\s*<|rel="next"', re.IGNORECASE)

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while page <= max_pages and len(results) < limit:
                    search_url = f"{base_url}/search?index=&page={page}&sort=&display=&q={encoded_query}"
                    async with session.get(search_url, headers=headers) as resp:
                        if resp.status != 200:
                            self.logger.warning(f"Anna's Archive 搜索失败: HTTP {resp.status}")
                            break
                        html_text = await resp.text()

                    page_added = 0
                    for m in md5_pattern.findall(html_text):
                        m = m.lower()
                        if m in seen_md5:
                            continue
                        seen_md5.add(m)
                        info = self._extract_anna_entry_info(html_text, m)
                        if info:
                            results.append(info)
                            page_added += 1
                        if len(results) >= limit:
                            break

                    if page_added == 0:
                        break
                    if not next_pattern.search(html_text):
                        break
                    page += 1

            self.logger.info(f"Anna's Archive 关键词搜索完成: 找到 {len(results)} 条结果")
            return results
        except Exception as e:
            self.logger.warning(f"Anna's Archive 关键词搜索失败: {e}")
            return []

    def _extract_anna_entry_info(self, html_text: str, md5: str) -> Optional[Dict]:
        """从 Anna's Archive HTML 中提取单个条目的详细信息"""
        # 标题链接模式（优先 js-vim-focus 类）
        title_pattern = re.compile(
            rf'href="(?:https?://[^"]*)?/md5/{md5}"[^>]*class="[^"]*js-vim-focus[^"]*"[^>]*>([^<]+)</a>',
            re.IGNORECASE | re.DOTALL
        )
        title_match = title_pattern.search(html_text)
        
        if not title_match:
            # 备用：尝试任何锚点文本
            fallback_pattern = re.compile(
                rf'href="(?:https?://[^"]*)?/md5/{md5}"[^>]*>([^<]+)</a>',
                re.IGNORECASE | re.DOTALL
            )
            title_match = fallback_pattern.search(html_text)
        
        title = html.unescape(title_match.group(1).strip()) if title_match else "Unknown"
        
        # 在 MD5 位置后的 HTML 块中查找其他字段
        md5_pos = html_text.lower().find(f'/md5/{md5}')
        if md5_pos == -1:
            md5_pos = 0
        search_block = html_text[md5_pos:md5_pos + 3000]
        
        # 作者（user-edit 图标后）
        authors_pattern = re.compile(
            r'icon-\[mdi--user-edit\][^>]*></span>\s*([^<]+)</a>',
            re.IGNORECASE
        )
        authors_match = authors_pattern.search(search_block)
        authors_str = html.unescape(authors_match.group(1).strip()) if authors_match else ""
        
        authors = []
        if authors_str:
            for sep in [' & ', ', ', '; ']:
                if sep in authors_str:
                    authors = [a.strip() for a in authors_str.split(sep) if a.strip()]
                    break
            if not authors:
                authors = [authors_str]
        
        # 出版信息（company 图标后）
        publisher_pattern = re.compile(
            r'icon-\[mdi--company\][^>]*></span>\s*([^<]+)</a>',
            re.IGNORECASE
        )
        publisher_match = publisher_pattern.search(search_block)
        publication_info = html.unescape(publisher_match.group(1).strip()) if publisher_match else ""
        
        # 年份：从多个来源提取
        year = None
        year_pattern1 = re.compile(r'·\s*(\d{4})\s*·')
        year_match = year_pattern1.search(search_block)
        if year_match:
            year = int(year_match.group(1))
        if not year and publication_info:
            year_pattern2 = re.compile(r',?\s*(\d{4})\s*$')
            year_match2 = year_pattern2.search(publication_info)
            if year_match2:
                year = int(year_match2.group(1))
        
        # 摘要/描述
        snippet = ""
        snippet_pattern = re.compile(
            r'text-gray-600[^>]*>(.*?)</div>',
            re.IGNORECASE | re.DOTALL
        )
        snippet_match = snippet_pattern.search(search_block)
        if snippet_match:
            snippet = snippet_match.group(1).strip()
            snippet = html.unescape(snippet)
            snippet = re.sub(r'<[^>]+>', ' ', snippet)
            snippet = re.sub(r'&lt;[^&]*?&gt;', '', snippet)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            snippet = snippet[:500] if snippet else ""
        
        # 文件类型和大小
        filetype = ""
        filesize = ""
        filetype_pattern = re.compile(
            r'·\s*(pdf|epub|djvu|mobi|zip|azw3?)\s*·\s*([\d.]+\s*[KMG]?B)',
            re.IGNORECASE
        )
        filetype_match = filetype_pattern.search(search_block)
        if filetype_match:
            filetype = filetype_match.group(1).upper()
            filesize = filetype_match.group(2)
        else:
            ft_only = re.search(r'·\s*(pdf|epub|djvu|mobi)\s*·', search_block, re.IGNORECASE)
            if ft_only:
                filetype = ft_only.group(1).upper()

        return {
            "md5": md5,
            "title": title,
            "authors": authors,
            "year": year,
            "publication_info": publication_info,
            "snippet": snippet,
            "filetype": filetype,
            "filesize": filesize,
            "link": f"https://annas-archive.pm/md5/{md5}"
        }

    async def _annas_get_download_url(self, md5: str) -> Optional[str]:
        """调用 Anna's Archive Fast Download API 获取下载 URL"""
        api_key = self.config.get("api_keys", {}).get("annas_archive", "")
        if not api_key:
            self.logger.error("Anna's Archive API Key 未配置")
            return None
        
        api_url = f"https://annas-archive.li/dyn/api/fast_download.json?md5={md5}&key={api_key}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, headers=headers) as resp:
                    if resp.status == 200:
                        try:
                            data = await resp.json(content_type=None)
                        except Exception:
                            text = await resp.text()
                            try:
                                data = json.loads(text)
                            except Exception:
                                self.logger.warning(
                                    "Anna's Archive API 返回非 JSON 响应，解析失败"
                                )
                                return None
                        return data.get("download_url")
                    self.logger.warning(f"Anna's Archive API 返回状态码: {resp.status}")
                    return None
        except Exception as e:
            self.logger.warning(f"Anna's Archive API 出错: {e}")
            return None
    
    async def _annas_download_file(
        self,
        download_url: str,
        filepath: str,
        task_download_dir: Optional[str] = None,
        annas_md5: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """从 Anna's Archive 下载文件（优先 curl），写入任务临时目录后校验并原子移动到目标路径。
        
        Args:
            download_url: 下载 URL
            filepath: 最终目标路径
            task_download_dir: 任务专属临时目录（与浏览器下载一致，并发隔离）
            annas_md5: Anna 的 MD5，用于临时文件名唯一性
        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 最终文件路径)
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        # 任务级隔离：临时文件放入 task_download_dir，文件名唯一避免多任务冲突
        if task_download_dir:
            os.makedirs(task_download_dir, exist_ok=True)
            unique_suffix = (annas_md5 or uuid.uuid4().hex[:12]).lower()
            tmp_path = os.path.join(task_download_dir, f"annas_{unique_suffix}.part")
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            tmp_path = f"{filepath}.part"
        
        try:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            curl_path = shutil.which("curl")
            if curl_path:
                cmd = [
                    curl_path,
                    "-L",
                    "--fail",
                    "--retry", "2",
                    "--connect-timeout", "15",
                    "--max-time", "300",
                    "-o", tmp_path,
                    download_url,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.warning(
                        f"Anna's Archive CLI 下载失败: {result.stderr.strip()}"
                    )
                else:
                    return await self._annas_finalize_download(tmp_path, filepath)
            # curl 不可用或失败时，回退到 aiohttp
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(download_url, headers=headers) as resp:
                    if resp.status != 200:
                        return False, None
                    with open(tmp_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                return await self._annas_finalize_download(tmp_path, filepath)
        except Exception as e:
            self.logger.warning(f"Anna's Archive 下载出错: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False, None

    async def _annas_finalize_download(self, tmp_path: str, filepath: str) -> Tuple[bool, Optional[str]]:
        """校验 Anna 下载的临时文件并原子移动到目标路径，更新 MD5 索引（与 _save_and_validate_download 一致）。"""
        try:
            if not os.path.exists(tmp_path):
                return False, None
            is_valid, path_or_err = self.verify_downloaded_file(tmp_path)
            if not is_valid:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                return False, None
            # path_or_err 可能是扩展名修正后的路径
            final_temp = path_or_err if os.path.exists(path_or_err) else tmp_path
            final_ext = os.path.splitext(final_temp)[1].lower()
            target_path = filepath
            if final_ext and final_ext != ".pdf":
                target_path = os.path.splitext(filepath)[0] + final_ext
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass
            shutil.move(final_temp, target_path)
            md5 = self._compute_file_md5(target_path)
            if md5:
                existing_path = self.file_md5_index.get(md5)
                if existing_path and existing_path != target_path and os.path.exists(existing_path):
                    self.logger.info(f"[Anna 下载验证] MD5重复: {md5[:8]}... 已存在 {existing_path}")
                else:
                    self.file_md5_index[md5] = target_path
                    self._save_md5_index()
            self.logger.info(f"[Anna 下载验证] 成功保存: {os.path.basename(target_path)}")
            return True, target_path
        except Exception as e:
            self.logger.debug(f"[Anna 下载验证] 失败: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False, None
    
    async def download_with_annas_archive(
        self, md5: str, filepath: str,
        title: Optional[str] = None, authors: Optional[List[str]] = None, year: Optional[int] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """使用 Anna's Archive API 下载论文（需要已知 MD5）
        
        此方法仅在用户手动选择"失败且有MD5"的论文并点击"Anna's Retry"时调用。
        使用与浏览器下载一致的任务级临时目录，完成后标准化命名并清理临时目录。
        
        Args:
            md5: Anna's Archive 文件的 MD5 哈希
            filepath: 保存路径
            title: 论文标题（用于记录历史）
            authors: 作者列表
            year: 发表年份
            
        Returns:
            Tuple[bool, str, Optional[str]]: (是否成功, 原因码, 最终文件路径)
        """
        self.logger.info(f"尝试通过 Anna's Archive API 下载: MD5={md5}")
        
        api_key = self.config.get("api_keys", {}).get("annas_archive", "")
        if not api_key:
            return False, "no_api_key", None
        
        if not md5:
            return False, "no_md5", None
        
        # 如果目标文件已存在且有效，直接返回成功
        if os.path.exists(filepath):
            if self.is_valid_pdf_file(filepath):
                return True, "already_exists", filepath
            try:
                os.remove(filepath)
            except Exception:
                pass

        task_download_dir, _, _ = self._create_task_download_dir()
        try:
            download_url = await self._annas_get_download_url(md5)
            if not download_url:
                return False, "api_error", None
            
            success, final_path = await self._annas_download_file(
                download_url, filepath,
                task_download_dir=task_download_dir,
                annas_md5=md5,
            )
            resolved_path = final_path or filepath if success else None
            if success:
                if title or authors:
                    ext = os.path.splitext(resolved_path)[1]
                    resolved_path = await self._rename_file_if_needed(
                        resolved_path, filepath,
                        title=title, authors=authors, year=year,
                        ext=ext
                    )
                if title:
                    self.download_history[title] = {
                        "title": title, "authors": authors or [], "year": year,
                        "method": "annas_archive", "md5": md5, "filepath": resolved_path,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.save_download_history()
            return (True, "ok", resolved_path) if success else (False, "download_failed", None)
        finally:
            self._cleanup_task_download_dir(task_download_dir)


    async def process_papers(self, papers: List[Dict]) -> Dict[str, int]:
        """处理论文数据列表，下载所有论文
        
        Args:
            papers: 论文数据列表
            
        Returns:
            Dict[str, int]: 下载统计信息
        """
        # 确保浏览器上下文已初始化
        await self.initialize()
        
        stats = {
            'total': len(papers),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'verified': 0,
            'corrupted': 0,
            'retry': 0,
            'already_downloaded': 0
        }
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def create_download_task(paper):
            # 添加0.5-1秒的随机延迟
            delay = random.uniform(0.5, 1.0)
            await asyncio.sleep(delay)
            return self._download_paper(paper, semaphore, stats)
        
        # 异步创建所有任务
        tasks = await asyncio.gather(
            *(create_download_task(paper) for paper in papers)
        )
        
        # 等待所有下载任务完成
        await asyncio.gather(*tasks)
        
        return stats

    async def _download_paper(self, paper: Dict, semaphore: asyncio.Semaphore, stats: Dict[str, int]) -> None:
        """下载单篇论文"""
        title = paper.get('title', '').strip()
        authors = paper.get('authors', [])
        year = paper.get('year')
        url = paper.get('url', '')
        paper_id = paper.get('id', '')
        
        # 生成文件名
        filename = self._generate_filename(title, authors, year)
        filepath = os.path.join(self.download_dir, f"{filename}.pdf")
        
        # 检查是否已经下载过此论文
        paper_hash = self._generate_paper_hash(title, authors)
        if paper_hash in self.download_history:
            history_entry = self.download_history[paper_hash]
            history_path = history_entry.get('path', filepath)
            self.logger.debug(f"检查历史记录: hash={paper_hash}")
            self.logger.debug(f"历史路径: {history_path}")
            self.logger.debug(f"当前路径: {filepath}")
            
            if os.path.exists(history_path):
                if self.is_valid_pdf_file(history_path):
                    self.logger.info(f"论文已成功下载过: {title}")
                    stats['already_downloaded'] += 1
                    return  # 确保这里直接返回，不继续下载
                else:
                    self.logger.warning(f"文件存在但验证失败，将重新下载: {history_path}")
                    try:
                        os.remove(history_path)  # 删除无效文件
                    except Exception as e:
                        self.logger.warning(f"删除无效文件失败: {str(e)}")
            else:
                self.logger.warning(f"历史记录中的文件不存在，将重新下载: {history_path}")
        
        # 检查目标路径是否已存在有效文件
        if os.path.exists(filepath):
            if self.is_valid_pdf_file(filepath):
                self.logger.info(f"目标路径已存在有效文件: {filepath}")
                stats['already_downloaded'] += 1
                
                # 更新下载历史
                if paper_hash not in self.download_history:
                    self.download_history[paper_hash] = {
                        'path': filepath,
                        'url': url,
                        'title': title,
                        'timestamp': time.time(),
                        'status': 'success'
                    }
                    self.save_download_history()
                return  # 确保这里直接返回，不继续下载
            else:
                self.logger.warning(f"目标路径存在无效文件，将重新下载: {filepath}")
                try:
                    os.remove(filepath)  # 删除无效文件
                except Exception as e:
                    self.logger.warning(f"删除无效文件失败: {str(e)}")

        try:
            async with semaphore:  # 后续的下载逻辑保持不变
                # 提取论文信息
                title = paper.get('title')
                authors = paper.get('authors', [])
                year = paper.get('year')
                
                # 直接映射字段名
                url = paper.get('url') or paper.get('link')
                pdf_url = paper.get('pdf_url') or paper.get('pdf_link')
                doi = paper.get('doi')
                source = paper.get('source', '')  # 提取来源标识
                
                # 确保至少有URL或DOI
                if not url and not pdf_url and not doi:
                    self.logger.warning(f"跳过没有URL、PDF链接和DOI的论文: {title}")
                    stats['skipped'] += 1
                    return
                
                success = False
                annas_md5 = paper.get('annas_md5')
                is_semantic_source = source == "Semantic Scholar"
                is_semantic_link = bool(url and "semanticscholar.org" in url.lower())
                if is_semantic_source and is_semantic_link:
                    self.logger.info("Semantic Scholar 来源链接不作为下载入口，将仅使用 pdf_url/doi")
                    url = None
                scihub_input = doi or url
                
                # 使用共享的浏览器上下文
                if self._browser_context is None:
                    self.logger.error("浏览器上下文未初始化")
                    stats['failed'] += 1
                    return
                
                # ===== 简化后的下载顺序 =====
                # 优先级：pdf_link 直链 > 浏览器 > DOI/Sci-Hub > Anna's Archive
                
                if pdf_url:
                    # 有 pdf_url：先尝试免费方法
                    self.logger.debug(f"检测到 pdf_url: {pdf_url}")
                    is_likely_pdf = self._is_likely_pdf_url(pdf_url)
                    is_ssrn = self._is_ssrn_url(pdf_url)
                    force_pdf_attempt = is_semantic_source and not is_likely_pdf
                    self.logger.debug(f"URL 识别结果: is_pdf={is_likely_pdf}, is_ssrn={is_ssrn}")
                    
                    # 1) 直接下载（扩展支持：不仅 .pdf，也包括其他可能的 PDF URL）
                    if not success and (is_likely_pdf or force_pdf_attempt):
                        try:
                            self.logger.info(f"尝试直接下载PDF链接: {pdf_url}")
                            success = await self.download_direct(pdf_url, filepath, title=title, authors=authors, year=year)
                            if success:
                                stats['success'] += 1
                                await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.warning(f"PDF直链下载失败: {str(e)}")

                    # 2) 浏览器下载 pdf_url（带预热和 cookie 处理）
                    if not success and (is_likely_pdf or force_pdf_attempt):
                        try:
                            self.logger.info(f"尝试通过浏览器下载PDF链接: {pdf_url}")
                            success = await self.pdf_download_with_browser(pdf_url, filepath)
                            if success:
                                stats['success'] += 1
                                await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.warning(f"浏览器下载pdf_url失败: {str(e)}")

                    # 3) BrightData 下载 pdf_url（付费）
                    if not success and (is_likely_pdf or force_pdf_attempt):
                        # ResearchGate 需要多次重试（偶发返回空响应）
                        max_retries = 3 if 'researchgate.net' in pdf_url.lower() else 1
                        
                        for attempt in range(max_retries):
                            try:
                                if attempt > 0:
                                    self.logger.info(f"ResearchGate 重试 ({attempt + 1}/{max_retries}): {pdf_url[:60]}...")
                                    await asyncio.sleep(2)  # 重试前等待
                                else:
                                    self.logger.info(f"尝试BrightData下载pdf_url: {pdf_url}")
                                
                                success = await self.download_with_solver(pdf_url, filepath, title=title, authors=authors, year=year)
                                if success:
                                    stats['success'] += 1
                                    break
                            except Exception as e:
                                self.logger.warning(f"BrightData下载pdf_url失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                        
                        if success:
                            await asyncio.sleep(1)

                    # 4) SSRN 特殊处理：pdf_url 指向页面但包含 PDF 下载链接
                    if not success and is_ssrn and not is_likely_pdf:
                        try:
                            self.logger.info(f"检测到 SSRN pdf_url，使用页面查找 PDF: {pdf_url}")
                            success = await self.find_and_download_pdf_with_browser(
                                pdf_url, filepath, title=title, authors=authors, year=year, source=source
                            )
                            if success:
                                stats['success'] += 1
                                await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.warning(f"SSRN pdf_url 页面查找 PDF 失败: {str(e)}")
                            await asyncio.sleep(1)

                # Semantic Scholar 专属：pdf_url 失败/缺失后，尝试 doi.org 解析跳转下载
                # 说明：仅对 Semantic Scholar 生效；其他来源保持原有逻辑不变
                if not success and is_semantic_source and doi:
                    try:
                        doi_str = str(doi).strip()
                        # 兼容：如果上游已给出 doi.org URL，则直接使用；否则拼接成可访问链接
                        if doi_str.lower().startswith("http"):
                            doi_url = doi_str
                        else:
                            doi_url = f"https://doi.org/{doi_str}"
                        self.logger.info(f"Semantic Scholar：尝试通过 DOI 解析下载: {doi_url}")
                        success = await self.find_and_download_pdf_with_browser(
                            doi_url, filepath, title=title, authors=authors, year=year, source=source
                        )
                        if success:
                            stats['success'] += 1
                            await asyncio.sleep(1)
                    except Exception as e:
                        self.logger.warning(f"Semantic Scholar：DOI 解析下载失败（继续后续策略）: {e}")

                if not success and not pdf_url and doi:
                    # 没有 pdf_url：提前查询 Anna MD5（后续 Sci-Hub 仅在有 MD5 时尝试）
                    try:
                        if not annas_md5:
                            md5 = await self._annas_search_md5(doi)
                            if md5:
                                annas_md5 = md5
                                paper['annas_md5'] = md5
                                self.logger.info(f"Anna's Archive 找到 MD5: {md5}（已保存，供后续 Sci-Hub 使用）")
                            else:
                                self.logger.info("Anna's Archive 未匹配 MD5，后续将跳过 Sci-Hub")
                    except Exception as e:
                        self.logger.warning(f"DOI 下载前 MD5 查询失败: {str(e)}")
                        self.logger.info("Anna's MD5 查询失败，后续将跳过 Sci-Hub")
                        await asyncio.sleep(1)

                if not success and url:
                    # 没有 pdf_url 或 pdf_url 下载失败：尝试从论文页面查找 PDF
                    if is_semantic_source and is_semantic_link and not pdf_url:
                        self.logger.info("检测到Semantic Scholar链接，跳过页面抓取，继续其他策略")
                    else:
                        try:
                            self.logger.info(f"尝试从论文页面查找PDF: {url}")
                            success = await self.find_and_download_pdf_with_browser(
                                url, filepath, title=title, authors=authors, year=year, source=source
                            )
                            if success:
                                stats['success'] += 1
                                await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.warning(f"浏览器查找PDF失败: {str(e)}")
                            await asyncio.sleep(1)

                # 4) BrightData + Sci-Hub（付费，通过 sci-hub.st）：仅在已有 Anna MD5 且前序均失败后尝试
                if not success and scihub_input:
                    # 搜索 Anna's Archive 获取 MD5（只搜索不下载，用于决定是否允许 Sci-Hub）
                    try:
                        if not annas_md5:
                            annas_query = doi or title
                            if annas_query:
                                md5 = await self._annas_search_md5(annas_query)
                                if md5:
                                    annas_md5 = md5
                                    paper['annas_md5'] = md5
                                    self.logger.info(f"Anna's Archive 找到 MD5: {md5}（已保存，供后续手动重试）")
                                else:
                                    self.logger.info("Anna's Archive 未匹配 MD5，跳过 Sci-Hub 以节省资源")
                            else:
                                self.logger.info("缺少 DOI/标题，无法查询 MD5，跳过 Sci-Hub 以节省资源")
                    except Exception as e:
                        self.logger.warning(f"Anna's MD5 搜索失败: {e}")
                        annas_md5 = None
                    
                    if annas_md5:
                        try:
                            self.logger.info(f"尝试BrightData + Sci-Hub下载: {scihub_input}")
                            success = await self.download_with_sci_hub(
                                scihub_input, filepath, title=title, authors=authors, year=year
                            )
                            if success:
                                stats['success'] += 1
                                await asyncio.sleep(1)
                        except Exception as e:
                            self.logger.warning(f"BrightData + Sci-Hub下载失败: {str(e)}")
                            await asyncio.sleep(1)
                    else:
                        self.logger.info("已跳过 Sci-Hub（需要 MD5 才尝试）")
                
                # 注意：Anna's Archive API 下载已移除自动调用
                # 只有用户手动选择"失败且有MD5"的论文并点击"Anna's Retry"时才会触发
              
                # 所有方法都失败
                if not success:
                    self.logger.error(f"所有下载方法均失败: {title}")
                    stats['failed'] += 1
        except Exception as e:
            self.logger.error(f"下载论文时发生未处理的异常: {str(e)}")
            stats['failed'] += 1

    def load_download_history(self) -> Dict[str, Dict]:
        """加载下载历史记录
        
        Returns:
            Dict: 下载历史字典
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.logger.info(f"已加载下载历史，共 {len(history)} 条记录")
                return history
            except json.JSONDecodeError:
                self.logger.warning(f"历史记录文件损坏，创建新的历史记录")
                # 备份损坏的文件
                backup_file = f"{self.history_file}.bak.{int(time.time())}"
                try:
                    shutil.copy2(self.history_file, backup_file)
                    self.logger.info(f"已备份损坏的历史记录到 {backup_file}")
                except:
                    pass
                return {}
            except Exception as e:
                self.logger.warning(f"加载下载历史失败: {str(e)}，创建新的历史记录")
                return {}
        else:
            self.logger.info("下载历史文件不存在，创建新的历史记录")
            return {}

    def save_download_history(self):
        """安全地保存下载历史到文件
        
        使用临时文件和原子重命名操作确保写入安全
        """
        try:
            # 使用临时文件
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_history, f, ensure_ascii=False, indent=2)
            
            # 在Windows上，可能需要先删除目标文件
            if sys.platform == 'win32' and os.path.exists(self.history_file):
                os.remove(self.history_file)
                
            # 原子重命名操作
            os.rename(temp_file, self.history_file)
            self.logger.debug(f"下载历史已保存到 {self.history_file}")
        except Exception as e:
            self.logger.error(f"保存下载历史失败: {str(e)}")

    async def start_auto_save(self, interval=60):
        """启动定期自动保存
        
        Args:
            interval: 自动保存间隔(秒)
        """
        self._auto_save_task = asyncio.create_task(self._auto_save_loop(interval))
    
    async def _auto_save_loop(self, interval):
        """自动保存循环
        
        Args:
            interval: 保存间隔(秒)
        """
        while True:
            await asyncio.sleep(interval)
            self.logger.debug(f"执行自动保存...")
            self.save_download_history()
            
    def stop_auto_save(self):
        """停止自动保存任务"""
        if hasattr(self, '_auto_save_task') and self._auto_save_task:
            self._auto_save_task.cancel()

    # 添加缺失的函数定义
    def _generate_paper_hash(self, title: str, authors: List[str]) -> str:
        """根据论文标题和作者生成唯一哈希值
        
        Args:
            title: 论文标题
            authors: 作者列表
            
        Returns:
            str: 论文的唯一哈希值
        """
        # 标准化标题和作者
        normalized_title = title.lower().strip() if title else ""
        normalized_authors = [a.lower().strip() for a in authors if a]
        
        # 组合标题和第一作者（如果有）
        hash_input = normalized_title
        if normalized_authors:
            hash_input += "_" + normalized_authors[0]
            
        # 生成哈希
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    async def _rename_file_if_needed(
        self,
        current_path: str,
        desired_path: str,
        title=None,
        authors=None,
        year=None,
        ext: Optional[str] = None
    ) -> str:
        """重命名文件并返回新路径
        
        Args:
            current_path: 当前文件路径
            desired_path: 期望的文件路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            
        Returns:
            str: 最终文件路径
        """
        # 生成标准化文件名
        if title or authors:
            base_filename = self._generate_filename(title, authors, year)
            normalized_ext = (ext or ".pdf").strip()
            if not normalized_ext.startswith("."):
                normalized_ext = f".{normalized_ext}"
            new_filepath = f"{os.path.dirname(desired_path)}/{base_filename}{normalized_ext}"
            
            # 如果文件路径不同，进行重命名
            if os.path.normpath(current_path) != os.path.normpath(new_filepath):
                try:
                    # 如果目标文件已存在，先删除
                    if os.path.exists(new_filepath):
                        os.remove(new_filepath)
                    
                    # 移动文件
                    shutil.move(current_path, new_filepath)
                    self.logger.info(f"文件已重命名: {os.path.basename(current_path)} -> {os.path.basename(new_filepath)}")
                    return new_filepath
                except Exception as e:
                    self.logger.warning(f"重命名文件失败: {str(e)}")
                    return current_path
        
        return current_path

    # =========================================================================
    # 统一的任务下载目录管理（并发安全）
    # =========================================================================
    
    def _create_task_download_dir(self) -> Tuple[str, float, set]:
        """创建任务专属临时目录
        
        Returns:
            Tuple[str, float, set]: (任务目录路径, 任务开始时间, 主目录初始文件集合)
        """
        task_id = uuid.uuid4().hex[:8]
        task_download_dir = os.path.join(self.download_dir, f".task_{task_id}")
        os.makedirs(task_download_dir, exist_ok=True)
        task_start_time = time.time()
        
        # 记录主目录当前文件列表（用于后续过滤）
        initial_main_files = set()
        try:
            initial_main_files = set(os.listdir(self.download_dir))
        except Exception:
            pass
        
        self.logger.debug(f"[任务目录] 创建: {task_download_dir}")
        return task_download_dir, task_start_time, initial_main_files
    
    def _cleanup_task_download_dir(self, task_download_dir: str) -> None:
        """清理任务专属临时目录
        
        Args:
            task_download_dir: 任务目录路径
        """
        try:
            if task_download_dir and os.path.exists(task_download_dir):
                shutil.rmtree(task_download_dir, ignore_errors=True)
                self.logger.debug(f"[任务目录] 已清理: {task_download_dir}")
        except Exception:
            pass
    
    async def _save_and_validate_download(
        self, 
        download_obj, 
        filepath: str, 
        task_download_dir: str,
        reason: str = "download"
    ) -> bool:
        """统一的下载保存和验证方法（并发安全）
        
        将下载先保存到任务临时目录，验证后再移动到最终位置。
        
        Args:
            download_obj: Playwright download 对象
            filepath: 最终目标路径
            task_download_dir: 任务专属临时目录
            reason: 下载原因（用于日志）
            
        Returns:
            bool: 是否成功
        """
        try:
            # 先保存到任务专属临时目录
            suggested_name = download_obj.suggested_filename or "download.pdf"
            temp_filepath = os.path.join(task_download_dir, suggested_name)
            
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass
            
            await download_obj.save_as(temp_filepath)
            
            # 验证是否是有效PDF
            if not self.is_valid_pdf_file(temp_filepath):
                self.logger.warning(f"[下载验证] 文件无效，删除: {reason}")
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass
                return False
            
            # 移动到最终位置
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            shutil.move(temp_filepath, filepath)
            
            # 计算并记录 MD5（用于去重/过滤）
            md5 = self._compute_file_md5(filepath)
            if md5:
                existing_path = self.file_md5_index.get(md5)
                if existing_path and existing_path != filepath and os.path.exists(existing_path):
                    self.logger.info(f"[下载验证] MD5重复: {md5[:8]}... 已存在 {existing_path}")
                else:
                    self.file_md5_index[md5] = filepath
                    self._save_md5_index()
            
            self.logger.info(f"[下载验证] 成功保存: {os.path.basename(filepath)} ({reason})")
            return True
            
        except Exception as e:
            self.logger.debug(f"[下载验证] 失败({reason}): {e}")
            return False
    
    async def _check_download_in_task_dir(self, task_download_dir: str) -> bool:
        """检查任务目录中是否有下载正在进行
        
        Args:
            task_download_dir: 任务专属临时目录
            
        Returns:
            bool: 是否有下载进行中
        """
        if not os.path.exists(task_download_dir):
            return False
        
        temp_extensions = ['.crdownload', '.part', '.download', '.tmp']
        try:
            for f in os.listdir(task_download_dir):
                if any(f.lower().endswith(ext) for ext in temp_extensions):
                    return True
        except Exception:
            pass
        
        return False
    
    async def _wait_for_download_in_task_dir(
        self, 
        task_download_dir: str, 
        filepath: str, 
        timeout: Optional[int] = None
    ) -> bool:
        """等待任务目录中的下载完成并移动到最终位置
        
        Args:
            task_download_dir: 任务专属临时目录
            filepath: 最终目标路径
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功
        """
        start_time = time.time()
        
        if timeout is None:
            timeout = self.timeouts.get("download_complete_timeout", 20)
        poll_interval = self.timeouts.get("download_poll_interval", 1)
        
        while time.time() - start_time < timeout:
            # 检查目标文件是否已存在且有效
            if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                self.logger.info(f"[任务目录等待] 目标文件已就绪: {filepath}")
                return True
            
            # 检查任务目录中的文件
            if os.path.exists(task_download_dir):
                try:
                    for f in os.listdir(task_download_dir):
                        # 跳过临时文件
                        if any(f.lower().endswith(ext) for ext in ['.crdownload', '.part', '.download', '.tmp']):
                            continue
                        
                        fpath = os.path.join(task_download_dir, f)
                        if os.path.isdir(fpath):
                            continue
                        
                        # 检查是否是有效PDF
                        if self.is_valid_pdf_file(fpath):
                            self.logger.info(f"[任务目录等待] 发现新PDF文件: {f}")
                            try:
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                                shutil.move(fpath, filepath)
                                return True
                            except Exception as e:
                                self.logger.warning(f"[任务目录等待] 移动文件失败: {e}")
                except Exception:
                    pass
            
            await asyncio.sleep(poll_interval)
        
        return False

    async def initialize(self):
        """初始化浏览器上下文"""
        if self._browser_context is None:
            await self.get_browser_context()

    def clean_browser_lock_files(self):
        """清理浏览器的锁定文件"""
        if not self.persist_browser:
            self.logger.info("非持久化模式，无需清理浏览器锁定文件")
            return
        if hasattr(self, '_persistent_user_data_dir'):
            self._clean_lock_files_in_dir(self._persistent_user_data_dir)
        else:
            self.logger.warning("未找到浏览器用户数据目录")

    # ==================== 阶段2：智能下载循环 ====================
    
    async def _smart_download_loop(self, page, filepath: str, url: str, 
                                   title: Optional[str] = None, 
                                   authors: Optional[List[str]] = None,
                                   max_iterations: int = 6,
                                   max_total_time: Optional[int] = None,
                                   task_download_dir: Optional[str] = None,
                                   initial_main_files: Optional[set] = None,
                                   task_start_time: Optional[float] = None) -> bool:
        """
        智能下载循环：根据页面分析结果决定下一步操作
        
        Args:
            page: Playwright页面对象
            filepath: 目标保存路径
            url: 原始URL（用于日志）
            title: 论文标题
            authors: 作者列表
            max_iterations: 最大迭代次数
            max_total_time: 最大总时间（秒）
            task_download_dir: 任务专属临时目录（并发安全）
            initial_main_files: 主目录初始文件集合（用于回退过滤）
            task_start_time: 任务开始时间（用于回退过滤）
        
        Returns:
            bool: 是否成功下载
        """
        visited_urls = set()
        last_action = None
        if max_total_time is None:
            max_total_time = self.timeouts.get("smart_loop_total_timeout", 60)
        start_time = time.time()
        action_history = []  # 新增：记录操作历史
        failed_selectors = []  # 新增：记录失败的选择器
        
        # === 阶段3：查询经验库 ===
        domain = urlparse(page.url).netloc.lower()
        learned_actions = self.experience_store.query_experience(domain, page.url)
        if learned_actions:
            self.logger.info(f"[智能循环] 找到历史经验，尝试复用...")
            for learned_action in learned_actions:
                selector = learned_action.get('selector')
                if selector:
                    try:
                        locator = page.locator(selector).first
                        if await locator.count() > 0:
                            self.logger.info(f"[智能循环] 执行经验操作: {selector[:50]}")
                            clicked = await self._click_element_safe(
                                page, 
                                ActionableElement(selector=selector, tag='a', text=learned_action.get('text', '')),
                                filepath,
                                task_download_dir=task_download_dir
                            )
                            if clicked:
                                await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                                if await self._wait_for_download_complete(
                                    filepath,
                                    timeout=self.timeouts.get("download_complete_timeout", 20),
                                    task_download_dir=task_download_dir,
                                    initial_main_files=initial_main_files,
                                    task_start_time=task_start_time
                                ):
                                    # 成功！记录这次成功
                                    self.experience_store.record_success(domain, url, learned_actions)
                                    return True
                    except Exception as e:
                        self.logger.debug(f"[智能循环] 经验操作执行失败: {e}")
            self.logger.info(f"[智能循环] 经验操作未成功，继续常规流程")
        # === 经验查询结束 ===
        
        for iteration in range(max_iterations):
            # 检查总时间限制
            elapsed = time.time() - start_time
            if elapsed > max_total_time:
                self.logger.warning(f"[智能循环] 超过总时间限制 {max_total_time}秒，退出")
                return False
            
            current_url = page.url
            self.logger.info(f"[智能循环] 迭代 {iteration + 1}/{max_iterations}, "
                            f"已用时 {elapsed:.1f}s, URL: {current_url[:60]}...")
            
            # 防止URL死循环
            url_key = current_url.split('#')[0].split('?')[0]  # 去掉锚点和查询参数
            if url_key in visited_urls and iteration > 0:
                self.logger.debug(f"[智能循环] URL重复访问，尝试不同策略")
            visited_urls.add(url_key)
            
            # 1. 分析当前页面
            analysis = await self.page_analyzer.analyze(page)
            
            # === 阶段3：应用经验库选择器加成 ===
            if analysis.actionable_elements:
                analysis.actionable_elements = self.experience_store.boost_element_scores(
                    analysis.actionable_elements
                )
            # === 加成结束 ===
            
            # the-innovation.org: Standard/Extended PDF 优先规则
            if "the-innovation.org" in domain:
                custom_buttons = await self._collect_innovation_pdf_buttons(page)
                if custom_buttons:
                    existing_selectors = {e.selector for e in analysis.actionable_elements}
                    merged = custom_buttons + [
                        e for e in analysis.actionable_elements
                        if e.selector not in existing_selectors
                    ]
                    analysis.actionable_elements = merged
                    self.logger.info("[the-innovation] 优先 Standard PDF，Extended 作为兜底")
            
            self.logger.debug(f"[智能循环] 内容类型: {analysis.content_type.value}, "
                             f"阻断: {[b.type.value for b in analysis.blockers]}, "
                             f"可操作元素: {len(analysis.actionable_elements)} 个")
            
            # 2. 处理阻断因素
            if analysis.blockers:
                resolved = await self._handle_blockers_smart(
                    page, analysis.blockers, task_download_dir=task_download_dir
                )
                if resolved == 'unsolvable':
                    self.logger.error(f"[智能循环] 遇到不可解决的阻断: {[b.type.value for b in analysis.blockers]}")
                    return False
                elif resolved == 'solved':
                    self.logger.info(f"[智能循环] 阻断已解决，重新分析页面")
                    await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                    continue  # 重新分析
                # resolved == 'continue' 表示可以继续尝试
            
            # 3. 检查是否已经触发下载（检查下载目录）
            if await self._check_download_in_progress(filepath, task_download_dir=task_download_dir):
                self.logger.info(f"[智能循环] 检测到下载进行中")
                if await self._wait_for_download_complete(
                    filepath,
                    timeout=self.timeouts.get("download_complete_timeout", 20),
                    task_download_dir=task_download_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time
                ):
                    return True
            
            # 4. 根据内容类型决定操作
            if analysis.content_type == ContentType.PDF_INLINE:
                self.logger.info(f"[智能循环] 检测到PDF内联显示，尝试直接获取")
                if await self._download_inline_pdf_smart(page, filepath):
                    return True
            
            if analysis.content_type == ContentType.PDF_VIEWER:
                self.logger.info(f"[智能循环] 检测到PDF查看器，查找下载按钮")
                # PDF查看器中优先找下载按钮
                if analysis.actionable_elements:
                    # 重新排序，优先下载按钮
                    sorted_elements = self._sort_elements_prefer_download(analysis.actionable_elements)
                    if sorted_elements:
                        best = sorted_elements[0]
                        clicked = await self._click_element_safe(page, best, filepath, task_download_dir=task_download_dir)
                        if clicked:
                            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                            if await self._wait_for_download_complete(
                                filepath,
                                timeout=self.timeouts.get("download_complete_timeout", 20),
                                task_download_dir=task_download_dir,
                                initial_main_files=initial_main_files,
                                task_start_time=task_start_time
                            ):
                                return True
                            continue
            
            # 5. 如果有高分可操作元素，点击它
            if analysis.actionable_elements:
                best = analysis.actionable_elements[0]
                
                # 跳过已经点击过的相同元素
                action_key = f"{best.selector}|{best.text[:20] if best.text else ''}"
                if action_key == last_action:
                    self.logger.debug(f"[智能循环] 跳过重复操作，尝试下一个元素")
                    if len(analysis.actionable_elements) > 1:
                        best = analysis.actionable_elements[1]
                        action_key = f"{best.selector}|{best.text[:20] if best.text else ''}"
                    else:
                        self.logger.info(f"[智能循环] 没有更多元素可尝试")
                        break
                
                if best.score >= 25:  # 置信度阈值
                    self.logger.info(f"[智能循环] 点击元素: 分数={best.score}, "
                                    f"文本='{best.text[:30] if best.text else ''}', "
                                    f"选择器={best.selector[:50]}")
                    
                    clicked = await self._click_element_safe(page, best, filepath, task_download_dir=task_download_dir)
                    
                    # === 阶段3：记录操作 ===
                    action_history.append({
                        'type': 'click',
                        'selector': best.selector,
                        'text': best.text[:30] if best.text else '',
                        'result': 'success' if clicked else 'fail'
                    })
                    if not clicked:
                        failed_selectors.append(best.selector)
                    # === 记录结束 ===
                    
                    if clicked:
                        last_action = action_key
                        
                        # 优先检查：如果最终文件已存在且有效，直接返回成功
                        if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                            self.logger.info(f"[智能循环] 下载已完成（点击后文件直接可用）")
                            self.experience_store.record_success(domain, url, action_history)
                            return True
                        
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        
                        # 再次检查最终文件
                        if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                            self.logger.info(f"[智能循环] 下载已完成（等待后文件可用）")
                            self.experience_store.record_success(domain, url, action_history)
                            return True
                        
                        # 检查是否触发了下载
                        if await self._wait_for_download_complete(
                            filepath,
                            timeout=self.timeouts.get("download_complete_timeout", 20),
                            task_download_dir=task_download_dir,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time
                        ):
                            # === 阶段3：记录成功经验 ===
                            self.experience_store.record_success(domain, url, action_history)
                            # === 记录结束 ===
                            return True
                        
                        # 检查页面是否变化（可能跳转到新页面）
                        if page.url != current_url:
                            self.logger.info(f"[智能循环] 页面已跳转到: {page.url[:70]}...")
                            continue  # 重新分析新页面
                    else:
                        self.logger.info(f"[智能循环] 点击失败，尝试下一个元素")
                        last_action = action_key
                        continue
                else:
                    self.logger.debug(f"[智能循环] 最高分元素分数过低 ({best.score})，尝试通用方法")
            
            # 6. 通用兜底：使用现有的 click_open_button_if_found
            self.logger.info(f"[智能循环] 尝试通用按钮查找")
            try:
                clicked = await self.click_open_button_if_found(
                    page, task_download_dir=task_download_dir
                )
                if clicked:
                    await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                    if await self._wait_for_download_complete(
                        filepath,
                        timeout=self.timeouts.get("download_complete_timeout", 20),
                        task_download_dir=task_download_dir,
                        initial_main_files=initial_main_files,
                        task_start_time=task_start_time
                    ):
                        return True
                    if page.url != current_url:
                        continue
            except Exception as e:
                self.logger.debug(f"[智能循环] 通用按钮查找失败: {e}")
            
            # 如果这一轮什么都没做成，等待一下再试
            self.logger.debug(f"[智能循环] 本轮未找到有效操作，等待后重试")
            await asyncio.sleep(self.timeouts.get("smart_loop_iteration_wait", 2))
        
        # 循环结束，常规方法都失败了
        
        # === 阶段3：LLM兜底 ===
        if self.llm_assistant.enabled and len(action_history) > 0:
            self.logger.info(f"[智能循环] 常规方法失败，尝试LLM分析...")
            try:
                analysis = await self.page_analyzer.analyze(page)
                suggestion = await self.llm_assistant.analyze_and_suggest(page, analysis, action_history)
                
                if suggestion and suggestion.get('type') == 'click':
                    selector = suggestion.get('selector')
                    if selector:
                        self.logger.info(f"[智能循环] LLM建议点击: {selector[:50]}")
                        try:
                            locator = page.locator(selector).first
                            if await locator.count() > 0:
                                await locator.click(
                                    timeout=self.timeouts.get("button_click_timeout", 5) * 1000,
                                    force=True
                                )
                                await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                                if await self._wait_for_download_complete(
                                    filepath,
                                    timeout=self.timeouts.get("download_complete_timeout", 20),
                                    task_download_dir=task_download_dir,
                                    initial_main_files=initial_main_files,
                                    task_start_time=task_start_time
                                ):
                                    # LLM建议成功！记录经验
                                    action_history.append({
                                        'type': 'click',
                                        'selector': selector,
                                        'text': '',
                                        'result': 'success',
                                        'source': 'llm'
                                    })
                                    self.experience_store.record_success(domain, url, action_history)
                                    return True
                        except Exception as e:
                            self.logger.debug(f"[智能循环] LLM建议执行失败: {e}")
                
                elif suggestion and suggestion.get('type') == 'give_up':
                    self.logger.warning(f"[智能循环] LLM建议放弃: {suggestion.get('reason', '')}")
            except Exception as e:
                self.logger.warning(f"[智能循环] LLM分析出错: {e}")
        
        # 记录失败的选择器
        if failed_selectors:
            self.experience_store.record_failure(domain, failed_selectors)
        # === LLM兜底结束 ===
        
        self.logger.warning(f"[智能循环] 达到最大迭代次数，下载失败")
        return False

    async def _handle_blockers_smart(self, page, blockers: List[Blocker],
                                     task_download_dir: Optional[str] = None) -> str:
        """
        智能处理阻断因素
        
        Returns:
            'solved': 阻断已解决
            'unsolvable': 阻断不可解决
            'continue': 可以继续尝试（阻断可能是误报）
        """
        for blocker in blockers:
            if blocker.type == BlockerType.CLOUDFLARE:
                self.logger.info(f"[阻断处理] 检测到Cloudflare，尝试解决...")
                success, _ = await self.solve_cloudflare_if_needed(
                    page, task_download_dir=task_download_dir
                )
                if success:
                    return 'solved'
                else:
                    # Cloudflare可能需要更长时间，等待后重试
                    self.logger.info(f"[阻断处理] Cloudflare未立即解决，等待10秒...")
                    await asyncio.sleep(self.timeouts.get("cloudflare_retry_wait", 10))
                    # 检查是否自动通过了
                    new_title = await page.title()
                    if '请稍候' not in new_title and 'moment' not in new_title.lower():
                        return 'solved'
                    return 'continue'  # 让循环继续尝试
            
            elif blocker.type == BlockerType.CAPTCHA:
                self.logger.info(f"[阻断处理] 检测到验证码，尝试解决...")
                success, _ = await self.solve_cloudflare_if_needed(
                    page, task_download_dir=task_download_dir
                )
                return 'solved' if success else 'continue'
            
            elif blocker.type in [BlockerType.PAYWALL, BlockerType.LOGIN_REQUIRED, BlockerType.GEO_BLOCKED]:
                self.logger.warning(f"[阻断处理] 不可解决的阻断: {blocker.type.value}")
                return 'unsolvable'
            
            elif blocker.type == BlockerType.NOT_FOUND:
                self.logger.error(f"[阻断处理] 页面不存在")
                return 'unsolvable'
            
            elif blocker.type == BlockerType.SERVER_ERROR:
                # 服务器错误可能是误报，继续尝试
                self.logger.warning(f"[阻断处理] 检测到服务器错误（可能误报），继续尝试")
                return 'continue'
            
            elif blocker.type == BlockerType.RATE_LIMITED:
                self.logger.warning(f"[阻断处理] 频率限制，等待30秒...")
                await asyncio.sleep(self.timeouts.get("rate_limit_wait", 30))
                return 'solved'
        
        return 'continue'

    async def _click_element_safe(self, page, element: ActionableElement, filepath: str, 
                                   task_download_dir: Optional[str] = None) -> bool:
        """
        安全点击元素，同时监听下载事件
        
        Args:
            page: 页面对象
            element: 要点击的元素
            filepath: 目标文件路径
            task_download_dir: 任务专属临时目录（并发安全）
        
        Returns:
            bool: 是否成功点击
        """
        try:
            locator = None
            
            # 方式1: 使用选择器
            try:
                locator = page.locator(element.selector).first
                if await locator.count() == 0:
                    locator = None
            except Exception:
                locator = None
            
            # 方式2: 使用文本匹配
            if locator is None and element.text:
                try:
                    text_escaped = element.text[:30].replace('"', '\\"')
                    if element.tag == 'a':
                        locator = page.locator(f'a:has-text("{text_escaped}")').first
                    elif element.tag == 'button':
                        locator = page.locator(f'button:has-text("{text_escaped}")').first
                    else:
                        locator = page.locator(f':has-text("{text_escaped}")').first
                    if locator and await locator.count() == 0:
                        locator = None
                except Exception:
                    locator = None
            
            # 方式3: 使用href
            if locator is None and element.href:
                try:
                    href_key = element.href.split('?')[0][-50:]
                    locator = page.locator(f'a[href*="{href_key}"]').first
                    if await locator.count() == 0:
                        locator = None
                except Exception:
                    locator = None
            
            if locator is None:
                self.logger.debug(f"[点击] 无法定位元素: {element.selector}")
                return False
            
            # 1. 等待页面加载稳定
            try:
                await page.wait_for_load_state(
                    'domcontentloaded',
                    timeout=self.timeouts.get("load_state_timeout", 10) * 1000
                )
            except Exception:
                pass
            
            # 2. 滚动到元素位置（可能在页面下方）
            try:
                await locator.scroll_into_view_if_needed(
                    timeout=self.timeouts.get("button_appear_timeout", 8) * 1000
                )
                await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
            except Exception as e:
                self.logger.debug(f"[点击] 滚动到元素失败: {e}")
            
            # 3. 等待元素可见
            use_force = False
            try:
                await locator.wait_for(
                    state='visible',
                    timeout=self.timeouts.get("button_appear_timeout", 8) * 1000
                )
            except Exception as e:
                self.logger.debug(f"[点击] 等待元素可见超时: {e}，将尝试强制点击")
                use_force = True
            
            # 4. 用短窗口探测是否触发下载
            try:
                async with page.expect_download(
                    timeout=self.timeouts.get("download_event_timeout", 15) * 1000
                ) as download_info:
                    await locator.click(
                        timeout=self.timeouts.get("button_click_timeout", 5) * 1000,
                        force=use_force
                    )
                download = await download_info.value
                # 使用统一的保存方法（并发安全）
                local_task_dir = task_download_dir
                cleanup_task_dir = False
                if not local_task_dir:
                    local_task_dir, _, _ = self._create_task_download_dir()
                    cleanup_task_dir = True
                try:
                    success = await self._save_and_validate_download(
                        download, filepath, local_task_dir, "click_element"
                    )
                    if success:
                        self.logger.info(f"[点击] 点击触发了下载，已保存")
                        return True
                    else:
                        self.logger.warning(f"[点击] 下载的文件无效")
                finally:
                    if cleanup_task_dir:
                        self._cleanup_task_download_dir(local_task_dir)
            except Exception as e:
                error_msg = str(e).lower()
                
                # 区分：点击本身失败 vs 点击成功但没下载
                if 'not visible' in error_msg or 'not attached' in error_msg or 'element is outside' in error_msg:
                    # 元素不可见/不可点击，尝试直接导航到 href
                    if element.href and ('pdf' in element.href.lower() or '/doi/' in element.href.lower()):
                        self.logger.info(f"[点击] 元素不可见，直接导航到: {element.href[:70]}...")
                        direct_download = {"obj": None}
                        def _on_direct_download(download):
                            direct_download["obj"] = download
                        page.on("download", _on_direct_download)
                        try:
                            await page.goto(
                                element.href,
                                wait_until='domcontentloaded',
                                timeout=self.timeouts.get("goto_timeout", 20) * 1000
                            )
                            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                            # 检查是否到达 PDF 页面
                            new_url = page.url.lower()
                            if '.pdf' in new_url or '/pdf/' in new_url or '/pdf?' in new_url:
                                self.logger.info(f"[点击] 导航成功，到达PDF页面")
                                if await self._download_inline_pdf_smart(page, filepath):
                                    return True
                            return True  # 导航成功，让循环继续分析新页面
                        except Exception as nav_e:
                            error_msg = str(nav_e)
                            if "Download is starting" in error_msg:
                                self.logger.info("[点击] 导航触发下载，等待监听器捕获")
                                for _ in range(10):  # 最多等待 2 秒
                                    if direct_download.get("obj") is not None:
                                        break
                                    await asyncio.sleep(0.2)
                                download = direct_download.get("obj")
                                if download is None:
                                    self.logger.warning("[点击] 下载捕获失败或超时")
                                    return False
                                direct_download["obj"] = None
                                local_task_dir = task_download_dir
                                cleanup_task_dir = False
                                if not local_task_dir:
                                    local_task_dir, _, _ = self._create_task_download_dir()
                                    cleanup_task_dir = True
                                try:
                                    if await self._save_and_validate_download(
                                        download, filepath, local_task_dir, "direct_goto_not_visible"
                                    ):
                                        return True
                                finally:
                                    if cleanup_task_dir:
                                        self._cleanup_task_download_dir(local_task_dir)
                                self.logger.warning("[点击] 下载捕获失败或文件无效")
                                return False
                            self.logger.warning(f"[点击] 导航失败: {nav_e}")
                            return False
                        finally:
                            try:
                                page.off("download", _on_direct_download)
                            except Exception:
                                pass
                    else:
                        self.logger.debug(f"[点击] 元素不可见且无有效href，跳过")
                        return False
                
                # 点击成功但没有立即触发下载（可能是页面跳转）
                self.logger.debug(f"[点击] 点击完成，未立即触发下载: {e}")
                
                # 检查是否跳转到了 PDF 页面
                await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                new_url = page.url.lower()
                if '.pdf' in new_url or '/pdf/' in new_url or '/pdf?' in new_url:
                    self.logger.info(f"[点击] 检测到跳转到PDF页面，尝试直接下载")
                    if await self._download_inline_pdf_smart(page, filepath):
                        return True
            
            return True  # 点击本身成功了
            
        except Exception as e:
            self.logger.warning(f"[点击] 点击元素失败: {e}")
            return False

    async def _check_download_in_progress(self, filepath: str, task_download_dir: Optional[str] = None) -> bool:
        """检查是否有下载正在进行
        
        Args:
            filepath: 目标文件路径
            task_download_dir: 任务专属临时目录（优先检查）
        """
        # 优先检查任务目录
        if task_download_dir:
            return await self._check_download_in_task_dir(task_download_dir)
        
        # 降级：检查主下载目录
        download_dir = os.path.dirname(filepath)
        if not os.path.exists(download_dir):
            return False
        
        temp_extensions = ['.crdownload', '.part', '.download', '.tmp']
        try:
            for f in os.listdir(download_dir):
                if any(f.lower().endswith(ext) for ext in temp_extensions):
                    return True
        except Exception:
            pass
        
        return False

    async def _wait_for_download_complete(self, filepath: str, timeout: Optional[int] = None,
                                          task_download_dir: Optional[str] = None,
                                          initial_main_files: Optional[set] = None,
                                          task_start_time: Optional[float] = None) -> bool:
        """
        等待下载完成
        
        Args:
            filepath: 目标文件路径
            timeout: 超时时间（秒）
            task_download_dir: 任务专属临时目录（优先检查，并发安全）
            initial_main_files: 主目录初始文件集合（用于回退过滤）
            task_start_time: 任务开始时间（用于回退过滤）
        """
        # 优先使用任务目录
        if task_download_dir:
            if await self._wait_for_download_in_task_dir(task_download_dir, filepath, timeout):
                return True
            
            # 回退：任务目录未捕获下载，尝试在主目录中严格过滤
            if initial_main_files is not None or task_start_time is not None:
                try:
                    for name in os.listdir(self.download_dir):
                        # 跳过临时文件和任务目录
                        if name.lower().endswith(('.crdownload', '.download', '.part', '.tmp')):
                            continue
                        if name.startswith('.task_'):
                            continue
                        
                        if initial_main_files is not None and name in initial_main_files:
                            continue
                        
                        p = os.path.join(self.download_dir, name)
                        if os.path.isdir(p):
                            continue
                        
                        try:
                            mtime = os.path.getmtime(p)
                            size = os.path.getsize(p)
                        except Exception:
                            continue
                        
                        if task_start_time is not None and mtime < task_start_time - 1:
                            continue
                        
                        # 检查是否是PDF
                        try:
                            with open(p, 'rb') as f:
                                if f.read(5) != b'%PDF-':
                                    continue
                        except Exception:
                            continue
                        
                        if size < 1000 or not self.is_valid_pdf_file(p):
                            continue
                        
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            shutil.move(p, filepath)
                            return True
                        except Exception as e:
                            self.logger.warning(f"[下载等待] 回退移动文件失败: {e}")
                except Exception:
                    pass
            
            return False
        
        # 降级：使用原有逻辑（扫描主下载目录）
        download_dir = os.path.dirname(filepath)
        start_time = time.time()
        
        # 记录初始文件列表
        initial_files = set()
        if os.path.exists(download_dir):
            try:
                initial_files = set(os.listdir(download_dir))
            except Exception:
                pass
        
        if timeout is None:
            timeout = self.timeouts.get("download_complete_timeout", 20)
        poll_interval = self.timeouts.get("download_poll_interval", 1)
        while time.time() - start_time < timeout:
            # 检查目标文件是否存在且有效
            if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                self.logger.info(f"[下载等待] 目标文件已就绪: {filepath}")
                return True
            
            # 检查下载目录中的新文件
            if os.path.exists(download_dir):
                try:
                    current_files = set(os.listdir(download_dir))
                    new_files = current_files - initial_files
                    
                    for f in new_files:
                        # 跳过临时文件
                        if any(f.lower().endswith(ext) for ext in ['.crdownload', '.part', '.download', '.tmp']):
                            continue
                        
                        fpath = os.path.join(download_dir, f)
                        
                        # 检查是否是有效PDF
                        if self.is_valid_pdf_file(fpath):
                            self.logger.info(f"[下载等待] 发现新PDF文件: {f}")
                            try:
                                if fpath != filepath:
                                    if os.path.exists(filepath):
                                        os.remove(filepath)
                                    shutil.move(fpath, filepath)
                                return True
                            except Exception as e:
                                self.logger.warning(f"[下载等待] 移动文件失败: {e}")
                except Exception:
                    pass
            
            await asyncio.sleep(poll_interval)
        
        return False

    async def _download_inline_pdf_smart(self, page, filepath: str) -> bool:
        """智能处理内联PDF - 优先使用 download_direct"""
        current_url = page.url
        self.logger.info(f"[内联PDF] 尝试下载: {current_url[:70]}...")
        
        # 方法1: 优先使用 download_direct（最可靠）
        try:
            success = await self.download_direct(current_url, filepath)
            if success and self.is_valid_pdf_file(filepath):
                self.logger.info(f"[内联PDF] download_direct 成功")
                return True
        except Exception as e:
            self.logger.debug(f"[内联PDF] download_direct 失败: {e}")
        
        # 方法2: 查找页面中的 iframe/embed PDF URL
        try:
            pdf_url = await page.evaluate('''() => {
                // 检查 embed 元素
                const embed = document.querySelector('embed[type="application/pdf"], embed[src*=".pdf"]');
                if (embed && embed.src) return embed.src;
                
                // 检查 iframe 元素
                const iframe = document.querySelector('iframe[src*=".pdf"], iframe[src*="/pdf/"]');
                if (iframe && iframe.src) return iframe.src;
                
                // 检查 object 元素
                const obj = document.querySelector('object[type="application/pdf"], object[data*=".pdf"]');
                if (obj && obj.data) return obj.data;
                
                return null;
            }''')
            
            if pdf_url:
                self.logger.info(f"[内联PDF] 发现嵌入的PDF URL: {pdf_url[:70]}...")
                success = await self.download_direct(pdf_url, filepath)
                if success and self.is_valid_pdf_file(filepath):
                    self.logger.info(f"[内联PDF] 嵌入PDF下载成功")
                    return True
        except Exception as e:
            self.logger.debug(f"[内联PDF] 查找嵌入PDF失败: {e}")
        
        # 方法3: JS fetch 作为最后兜底
        try:
            import base64
            pdf_data = await page.evaluate('''async () => {
                try {
                    const response = await fetch(location.href, {
                        credentials: 'include',
                        headers: { 'Accept': 'application/pdf' }
                    });
                    const blob = await response.blob();
                    if (blob.type === 'application/pdf' || blob.type.includes('pdf')) {
                        const buffer = await blob.arrayBuffer();
                        const bytes = new Uint8Array(buffer);
                        let binary = '';
                        for (let i = 0; i < bytes.length; i++) {
                            binary += String.fromCharCode(bytes[i]);
                        }
                        return btoa(binary);
                    }
                } catch (e) {
                    console.error('PDF fetch error:', e);
                }
                return null;
            }''')
            
            if pdf_data:
                pdf_bytes = base64.b64decode(pdf_data)
                with open(filepath, 'wb') as f:
                    f.write(pdf_bytes)
                
                if self.is_valid_pdf_file(filepath):
                    self.logger.info(f"[内联PDF] JS fetch 成功")
                    return True
                else:
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.debug(f"[内联PDF] JS fetch 失败: {e}")
        
        self.logger.warning(f"[内联PDF] 所有方法均失败")
        return False

    def _sort_elements_prefer_download(self, elements: List[ActionableElement]) -> List[ActionableElement]:
        """重新排序元素，优先下载按钮，降权补充材料和参考文献"""
        def download_score(elem):
            score = elem.score
            text = (elem.text or '').lower()
            href = (elem.href or '').lower()
            combined = f"{text} {href}"
            
            # 加分
            if 'download' in text or '下载' in text:
                score += 50
            if 'save' in text or '保存' in text:
                score += 30
            if 'open' in text or '打开' in text or '查看' in text:
                score += 25
            
            # 降权：补充材料
            supp_kw = ['supplementary', 'supplement', 'supporting', 'appendix', 'esm', 'moesm', '-sup-', '附件', '附录']
            for kw in supp_kw:
                if kw in combined:
                    score -= 60
                    break
            
            # 降权：参考文献
            ref_kw = ['reference', 'bibliography', 'cited', 'citation', 'ref-list', '参考文献', '引用']
            for kw in ref_kw:
                if kw in combined:
                    score -= 50
                    break
            
            return score
        
        return sorted(elements, key=download_score, reverse=True)

    # ==================== 阶段2结束 ====================

    async def _collect_innovation_pdf_buttons(self, page) -> List[ActionableElement]:
        """the-innovation.org 站点：提取 Standard/Extended PDF 按钮"""
        elements: List[ActionableElement] = []
        try:
            raw_elements = await page.evaluate('''() => {
                const results = [];
                document.querySelectorAll('div.download-pdf').forEach(el => {
                    const a = el.querySelector('a');
                    const text = (a?.innerText || el.innerText || '').trim();
                    const dataType = el.getAttribute('data-type') || '';
                    const selector = dataType
                        ? `div.download-pdf[data-type="${dataType}"] a`
                        : 'div.download-pdf:not([data-type]) a';
                    results.push({ selector, text, dataType });
                });
                return results;
            }''')
            for raw in raw_elements:
                elem = ActionableElement(
                    selector=raw["selector"],
                    tag='a',
                    text=raw.get("text", ""),
                    href=None,
                    is_visible=True,
                    position_y=0,
                    attributes={"data-type": raw.get("dataType", "")}
                )
                text = (elem.text or "").lower()
                if "standard" in text:
                    elem.score = 120
                elif "extended" in text:
                    elem.score = 90
                else:
                    elem.score = 60
                elements.append(elem)
        except Exception as e:
            self.logger.debug(f"[the-innovation] 按钮提取失败: {e}")
        elements.sort(key=lambda x: x.score, reverse=True)
        return elements

    async def find_and_download_pdf_with_browser(self, url: str, filepath: str, 
                                           title: Optional[str] = None, 
                                           authors: Optional[List[str]] = None,
                                           year: Optional[int] = None,
                                           source: Optional[str] = None) -> bool:
        """使用浏览器查找并下载PDF文件
        
        查找页面中的PDF下载链接按钮，点击后下载PDF文件
        
        Args:
            url: 论文页面URL
            filepath: 保存路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            
        Returns:
            bool: 是否下载成功
        """
        self.logger.info(f"尝试查找并下载PDF: {url}")
        downloads_dir = os.path.dirname(filepath)
        os.makedirs(downloads_dir, exist_ok=True)
        
        # ====== Bug修复：创建任务专属临时目录 ======
        task_id = uuid.uuid4().hex[:8]
        task_download_dir = os.path.join(self.download_dir, f".task_{task_id}")
        os.makedirs(task_download_dir, exist_ok=True)
        task_start_time = time.time()
        
        # 记录主目录当前文件列表（用于salvage时过滤）
        initial_main_files = set()
        try:
            initial_main_files = set(os.listdir(self.download_dir))
        except Exception:
            pass
        # ====== Bug修复结束 ======
        
        finished = False
        page = None
        new_page = None  # 用于跟踪可能打开的新页面
        new_page_download_handler = None
        
        try:
            # 使用共享的浏览器上下文（必要时初始化）
            await self.initialize()
            if self._browser_context is None:
                self.logger.error("浏览器上下文未初始化")
                return False
                
            # 创建新页面
            page = await self._browser_context.new_page()
            self._active_pages.add(page)

            # 捕获“非常快的下载”：不依赖 expect_download，避免错过瞬时 download 事件
            latest_download = {"obj": None}
            def _on_download(download):
                latest_download["obj"] = download
            page.on("download", _on_download)

            async def _finalize_download_obj(download_obj, reason: str) -> bool:
                """将 Playwright download 保存为目标文件名并写入下载历史
                
                【Bug修复】先保存到任务临时目录，验证后再移动到最终位置
                """
                nonlocal finished
                try:
                    # 先保存到任务专属临时目录
                    suggested_name = download_obj.suggested_filename or "download.pdf"
                    temp_filepath = os.path.join(task_download_dir, suggested_name)
                    
                    if os.path.exists(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                    await download_obj.save_as(temp_filepath)
                    
                    # 验证是否是有效PDF
                    if not self.is_valid_pdf_file(temp_filepath):
                        try:
                            os.remove(temp_filepath)
                        except Exception:
                            pass
                        return False

                    # 移动到最终位置
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
                    shutil.move(temp_filepath, filepath)

                    # 计算并记录 MD5（用于去重/过滤）
                    md5 = self._compute_file_md5(filepath)
                    if md5:
                        existing_path = self.file_md5_index.get(md5)
                        if existing_path and existing_path != filepath and os.path.exists(existing_path):
                            self.logger.info(f"[finalize] MD5重复: {md5[:8]}... 已存在 {existing_path}")
                        else:
                            self.file_md5_index[md5] = filepath
                            self._save_md5_index()

                    # 写入下载历史：这一步是后续“导入/标记成功”的关键
                    try:
                        if title and authors:
                            file_hash = self._generate_paper_hash(title, authors)
                        else:
                            file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                        self.download_history[file_hash] = {
                            'path': filepath,
                            'url': url,
                            'title': title,
                            'timestamp': time.time(),
                            'status': 'success',
                            'method': f'browser_download_event:{reason}',
                            'md5': md5
                        }
                        self.save_download_history()
                    except Exception as e:
                        self.logger.warning(f"写入下载历史失败（不影响文件已保存）: {e}")

                    finished = True
                    return True
                except Exception as e:
                    self.logger.debug(f"finalize_download 失败({reason}): {e}")
                    return False

            async def _consume_download_event(reason: str) -> bool:
                """如果 download 事件已经发生，则立刻接管并保存到 filepath"""
                dl = latest_download.get("obj")
                if dl is None:
                    return False
                latest_download["obj"] = None
                return await _finalize_download_obj(dl, reason)

            async def _log_analysis(tag: str, level: str = "info") -> None:
                try:
                    analysis = await self.page_analyzer.analyze(page)
                    msg = self.page_analyzer.format_analysis(analysis)
                    if level == "warning":
                        self.logger.warning(f"[观察点-{tag}] {msg}")
                    else:
                        self.logger.info(f"[观察点-{tag}] {msg}")
                except Exception as e:
                    self.logger.debug(f"页面分析失败: {e}")

            def _doi_hint(d: Optional[str]) -> Optional[str]:
                if not d:
                    return None
                d = d.strip().lower()
                # 10.1371/journal.pbio.3003291 -> 3003291
                m = re.search(r'10\.\d{4,9}/\S+', d)
                if not m:
                    return None
                tail = m.group(0).split("/")[-1]
                return tail

            async def _salvage_downloaded_pdf(reason: str) -> bool:
                """
                兜底：有些站点下载发生在 Playwright download 事件之外。
                
                【Bug修复】：
                1. 优先扫描任务专属临时目录（完全隔离）
                2. 如果临时目录为空，再扫描主目录（使用严格过滤）
                """
                nonlocal finished
                
                # 首先检查目标文件是否已存在
                if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                    self.logger.info(f"[salvage] 目标文件已存在: {filepath}")
                    return True
                
                # ====== 策略1：扫描任务专属临时目录（完全安全）======
                task_candidates = []
                try:
                    for name in os.listdir(task_download_dir):
                        if name.lower().endswith(('.crdownload', '.download', '.part', '.tmp')):
                            continue
                        p = os.path.join(task_download_dir, name)
                        if os.path.isdir(p):
                            continue
                        try:
                            with open(p, 'rb') as f:
                                if f.read(5) != b'%PDF-':
                                    continue
                        except Exception:
                            continue
                        if self.is_valid_pdf_file(p):
                            mtime = os.path.getmtime(p)
                            md5 = self._compute_file_md5(p)
                            if md5:
                                existing_path = self.file_md5_index.get(md5)
                                if existing_path and existing_path != p and os.path.exists(existing_path):
                                    self.logger.info(f"[salvage] 跳过重复PDF(MD5): {name}")
                                    continue
                            task_candidates.append((mtime, p, name, md5))
                except Exception as e:
                    self.logger.debug(f"[salvage] 扫描任务目录出错: {e}")
                
                if task_candidates:
                    # 任务目录有文件，选择最新的（因为都是本任务的，安全）
                    task_candidates.sort(key=lambda x: x[0], reverse=True)
                    _, best_path, best_name, _ = task_candidates[0]
                    self.logger.info(f"[salvage] 从任务目录选择: {best_name}")
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        shutil.move(best_path, filepath)
                        if self.is_valid_pdf_file(filepath):
                            md5 = self._compute_file_md5(filepath)
                            if md5:
                                existing_path = self.file_md5_index.get(md5)
                                if existing_path and existing_path != filepath and os.path.exists(existing_path):
                                    self.logger.info(f"[salvage] 发现重复PDF(MD5): {best_name}")
                                else:
                                    self.file_md5_index[md5] = filepath
                                    self._save_md5_index()
                            # 写入历史
                            try:
                                if title and authors:
                                    file_hash = self._generate_paper_hash(title, authors)
                                else:
                                    file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                                self.download_history[file_hash] = {
                                    'path': filepath,
                                    'url': url,
                                    'title': title,
                                    'timestamp': time.time(),
                                    'status': 'success',
                                    'method': f'salvage_task_dir:{reason}',
                                    'md5': md5
                                }
                                self.save_download_history()
                            except Exception:
                                pass
                            finished = True
                            return True
                    except Exception as e:
                        self.logger.error(f"[salvage] 移动文件失败: {e}")
                    return False
                
                # ====== 策略2：扫描主目录（严格过滤）======
                self.logger.debug(f"[salvage] 任务目录为空，扫描主目录（严格过滤）")
                
                # 提取URL中的DOI/期刊hint
                hint = None
                try:
                    if url:
                        m1 = re.search(r'journal\.[a-z0-9]+\.(\d+)', url.lower())
                        if m1:
                            hint = m1.group(1)
                    if not hint:
                        hint = _doi_hint(url)
                except Exception:
                    hint = None
                
                main_candidates = []
                try:
                    for name in os.listdir(self.download_dir):
                        # 跳过临时文件和其他任务的目录
                        if name.lower().endswith(('.crdownload', '.download', '.part', '.tmp')):
                            continue
                        if name.startswith('.task_'):  # 跳过其他任务的临时目录
                            continue
                        
                        p = os.path.join(self.download_dir, name)
                        if os.path.isdir(p):
                            continue
                        
                        # 【关键过滤1】必须是任务开始后的新文件
                        if name in initial_main_files:
                            continue
                        
                        try:
                            mtime = os.path.getmtime(p)
                            size = os.path.getsize(p)
                        except Exception:
                            continue
                        
                        # 【关键过滤2】只选任务开始后创建的文件
                        if mtime < task_start_time - 1:  # 允许1秒误差
                            continue
                        
                        # 检查是否是PDF
                        try:
                            with open(p, 'rb') as f:
                                if f.read(5) != b'%PDF-':
                                    continue
                        except Exception:
                            continue
                        
                        if size < 1000 or not self.is_valid_pdf_file(p):
                            continue
                        
                        md5 = self._compute_file_md5(p)
                        if md5:
                            existing_path = self.file_md5_index.get(md5)
                            if existing_path and existing_path != p and os.path.exists(existing_path):
                                self.logger.info(f"[salvage] 跳过重复PDF(MD5): {name}")
                                continue
                        
                        # 计算匹配分数（DOI hint匹配）
                        score = 0
                        if hint and hint in name.lower():
                            score += 10
                        
                        main_candidates.append({
                            'path': p,
                            'name': name,
                            'mtime': mtime,
                            'score': score,
                            'time_since_start': mtime - task_start_time,
                            'md5': md5
                        })
                except Exception as e:
                    self.logger.debug(f"[salvage] 扫描主目录出错: {e}")
                    return False
                
                if not main_candidates:
                    self.logger.debug(f"[salvage] 主目录未找到符合条件的文件")
                    return False
                
                # 按分数排序，分数相同则选最早创建的（更可能是自己的）
                main_candidates.sort(key=lambda x: (-x['score'], x['mtime']))
                
                best = main_candidates[0]
                
                # 警告：如果有多个候选且分数相同，可能选错
                if len(main_candidates) > 1 and main_candidates[0]['score'] == main_candidates[1]['score']:
                    self.logger.warning(
                        f"[salvage] 主目录有多个候选文件，可能存在并发风险！"
                        f"选择: {best['name']}, 共{len(main_candidates)}个候选"
                    )
                else:
                    self.logger.info(
                        f"[salvage] 从主目录选择: {best['name']}, "
                        f"创建于任务开始后{best['time_since_start']:.1f}秒"
                    )
                
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    shutil.move(best['path'], filepath)
                    if self.is_valid_pdf_file(filepath):
                        md5 = self._compute_file_md5(filepath)
                        if md5:
                            existing_path = self.file_md5_index.get(md5)
                            if existing_path and existing_path != filepath and os.path.exists(existing_path):
                                self.logger.info(f"[salvage] 发现重复PDF(MD5): {best['name']}")
                            else:
                                self.file_md5_index[md5] = filepath
                                self._save_md5_index()
                        # 写入历史
                        try:
                            if title and authors:
                                file_hash = self._generate_paper_hash(title, authors)
                            else:
                                file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                            self.download_history[file_hash] = {
                                'path': filepath,
                                'url': url,
                                'title': title,
                                'timestamp': time.time(),
                                'status': 'success',
                                'method': f'salvage_main_dir:{reason}',
                                'md5': md5
                            }
                            self.save_download_history()
                        except Exception:
                            pass
                        finished = True
                        return True
                except Exception as e:
                    self.logger.error(f"[salvage] 移动文件失败: {e}")
                
                return False
            
            # 应用 stealth 模式（兼容不同 playwright-stealth 版本）
            if self.stealth_mode:
                self.logger.debug("应用 stealth 模式到页面")
                try:
                    if _STEALTH_MODE == "v2" and _Stealth is not None:
                        await _Stealth().apply_stealth_async(page)
                    elif _STEALTH_MODE == "async" and _stealth_async is not None:
                        await _stealth_async(page)
                    elif _STEALTH_MODE == "sync" and _stealth_sync is not None:
                        _stealth_sync(page)
                except Exception as e:
                    self.logger.warning(f"playwright-stealth 应用失败，将继续执行（不致命）: {e}")
            
            # 预热步骤：先访问网站主页处理 cookie 同意，避免阻塞
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                self.logger.info(f"预热：先访问主站 {base_url} 处理 cookie")
                warmup_response = await page.goto(
                    base_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeouts.get("goto_timeout", 20) * 1000
                )
                self.logger.debug(f"预热完成: status={warmup_response.status if warmup_response else None}")
                
                # 尝试处理 cookie 同意弹窗
                await self._handle_cookie_consent(page, context_label="预热")
                await _consume_download_event("warmup")
                
                # 检测并处理预热页面的 Cloudflare 验证
                cf_success, _ = await self.solve_cloudflare_if_needed(
                    page, task_download_dir=task_download_dir
                )
                if not cf_success:
                    self.logger.warning("预热页面 Cloudflare 验证解决失败，继续尝试")
                
                self.logger.info("预热完成，开始访问论文页面")
            except Exception as warmup_e:
                self.logger.warning(f"预热访问失败，继续尝试直接访问: {warmup_e}")
            
            # 检测是否为DOI链接
            is_doi_link = 'doi.org' in url.lower() or url.startswith('10.')
            is_semantic_scholar = source == "Semantic Scholar"
            is_wiley = 'onlinelibrary.wiley.com' in url.lower()
            
            # 检测是否为可能的直接下载链接（OJS viewFile、/download/、/pdf/ 等模式）
            is_direct_download_url = any(pattern in url.lower() for pattern in [
                '/viewfile/', '/article/viewfile/', '/download/', '/attachment/',
                '/getpdf', '/fulltextpdf', '/pdf/download'
            ])
            
            # 直接下载链接的优先处理（如 OJS viewFile）
            if is_direct_download_url and not is_doi_link and not is_wiley:
                self.logger.info(f"检测到可能的直接下载链接: {url}")
                try:
                    download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                    async with page.expect_download(timeout=download_timeout_ms) as download_info:
                        await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=self.timeouts.get("goto_timeout", 20) * 1000
                        )
                        await self._handle_cookie_consent(page, context_label="直接下载链接")
                        # 检测并处理 Cloudflare 验证
                        cf_success, cf_file = await self.solve_cloudflare_if_needed(
                            page, filepath, task_download_dir=task_download_dir
                        )
                        if cf_file and self.is_valid_pdf_file(cf_file):
                            self.logger.info(f"直接下载链接 Cloudflare 验证触发下载: {cf_file}")
                            return True
                    download = await download_info.value
                    # 使用统一的保存方法（并发安全）
                    if await self._save_and_validate_download(download, filepath, task_download_dir, "direct_download_url"):
                        self.logger.info(f"直接下载链接成功: {filepath}")
                        # 写入下载历史
                        try:
                            if title and authors:
                                file_hash = self._generate_paper_hash(title, authors)
                            else:
                                file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                            self.download_history[file_hash] = {
                                'path': filepath,
                                'url': url,
                                'title': title,
                                'timestamp': time.time(),
                                'status': 'success',
                                'method': 'direct_download_url',
                            }
                            self.save_download_history()
                        except Exception:
                            pass
                        return True
                    else:
                        self.logger.warning("直接下载链接文件无效，继续常规流程")
                except asyncio.TimeoutError:
                    self.logger.debug("直接下载链接未触发下载事件，继续常规流程")
                except Exception as direct_dl_e:
                    self.logger.debug(f"直接下载链接尝试失败: {direct_dl_e}")
                    # 检查全局监听器是否捕获了下载
                    if await _consume_download_event("direct-dl-exception-fallback"):
                        return True

            # Wiley专属：优先尝试构造 pdfdirect 直达下载（绕开按钮点击/Cloudflare渲染差异）
            # 运行证据：浏览器打开 /doi/pdfdirect/{doi}?download=true 会触发 "Download is starting"
            if is_wiley and '/doi/pdfdirect/' not in url.lower():
                try:
                    from urllib.parse import urlparse
                    parsed_wiley = urlparse(url)
                    # 兼容 /doi/full/{doi}、/doi/abs/{doi}、/doi/pdf/{doi}、/doi/epdf/{doi}
                    m = re.search(r'/doi/(?:full|abs|pdf|epdf)/([^?]+)', parsed_wiley.path, flags=re.IGNORECASE)
                    wiley_doi = m.group(1) if m else None
                    if wiley_doi:
                        pdfdirect_url = f"{parsed_wiley.scheme}://{parsed_wiley.netloc}/doi/pdfdirect/{wiley_doi}?download=true"

                        download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                        async with page.expect_download(timeout=download_timeout_ms) as download_info:
                            await page.goto(pdfdirect_url, wait_until="domcontentloaded")
                            await self._handle_cookie_consent(page, context_label="Wiley pdfdirect")
                            # 检测并处理 Cloudflare 验证
                            cf_success, cf_file = await self.solve_cloudflare_if_needed(
                                page, filepath, task_download_dir=task_download_dir
                            )
                            if cf_file and self.is_valid_pdf_file(cf_file):
                                self.logger.info(f"Wiley pdfdirect Cloudflare 验证触发下载: {cf_file}")
                                return True
                        download = await download_info.value
                        # 使用统一的保存方法（并发安全）
                        if await self._save_and_validate_download(download, filepath, task_download_dir, "wiley_pdfdirect"):
                            self.logger.info(f"Wiley pdfdirect 直达下载成功: {filepath}")
                            return True
                        else:
                            self.logger.warning("Wiley pdfdirect 下载文件无效，继续走原有流程")
                except Exception as wiley_direct_e:
                    self.logger.warning(f"Wiley pdfdirect 尝试失败（继续走原有流程）: {wiley_direct_e}")
            
            # 尝试访问URL，如果直接触发下载则捕获
            response = None
            direct_download_triggered = False
            try:
                # 对于DOI链接，可能直接触发下载，使用 expect_download
                if is_doi_link:
                    try:
                        async with page.expect_download(timeout=5000) as download_info:
                            response = await page.goto(url, wait_until="domcontentloaded")
                            await self._handle_cookie_consent(page, context_label="DOI跳转后")
                        if await _consume_download_event("goto-doi"):
                            return True
                        # 如果到这里说明确实触发了下载
                        self.logger.info("DOI链接直接触发PDF下载")
                        download = await download_info.value
                        # 使用统一的保存方法（并发安全）
                        if await self._save_and_validate_download(download, filepath, task_download_dir, "doi_direct"):
                            self.logger.info(f"DOI直接下载成功: {filepath}")
                            # 更新下载历史
                            if title and authors:
                                file_hash = self._generate_paper_hash(title, authors)
                            else:
                                file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                            self.download_history[file_hash] = {
                                'path': filepath,
                                'url': url,
                                'title': title,
                                'timestamp': time.time(),
                                'status': 'success'
                            }
                            self.save_download_history()
                            return True
                        else:
                            self.logger.warning("DOI直接下载的文件无效")
                            if os.path.exists(filepath):
                                os.remove(filepath)
                    except (asyncio.TimeoutError, PlaywrightTimeoutError):
                        # 没有触发下载，继续正常流程
                        self.logger.debug("DOI链接未直接触发下载，继续页面解析流程")
                        response = await page.goto(url, wait_until="domcontentloaded")
                        await self._handle_cookie_consent(page, context_label="DOI页面解析")
                        await _log_analysis("导航后")
                        # DOI 导航后立即检测并处理 Cloudflare 验证
                        cf_success, cf_file = await self.solve_cloudflare_if_needed(
                            page, filepath, task_download_dir=task_download_dir
                        )
                        if cf_file and self.is_valid_pdf_file(cf_file):
                            self.logger.info(f"DOI 导航后 Cloudflare 验证触发下载: {cf_file}")
                            return True
                        if await _consume_download_event("goto-doi-parse"):
                            return True
                else:
                    response = await page.goto(url, wait_until="domcontentloaded")
                    await self._handle_cookie_consent(page, context_label="页面解析")
                    await _log_analysis("导航后")
                    # 导航后立即检测并处理 Cloudflare 验证
                    cf_success, cf_file = await self.solve_cloudflare_if_needed(
                        page, filepath, task_download_dir=task_download_dir
                    )
                    if cf_file and self.is_valid_pdf_file(cf_file):
                        self.logger.info(f"导航后 Cloudflare 验证触发下载: {cf_file}")
                        return True
                    if await _consume_download_event("goto-article"):
                        return True
            except Exception as e:
                error_msg = str(e)
                if "Download is starting" in error_msg:
                    # 这意味着URL是直接的PDF下载链接
                    self.logger.info("检测到URL直接触发下载，尝试捕获")
                    direct_download_triggered = True
                    
                    # 【修复】等待 download 事件触发（异步延迟问题）
                    # page.goto() 抛出异常比 download 事件触发更快，需要等待
                    for wait_attempt in range(10):  # 最多等待 2 秒
                        if latest_download.get("obj") is not None:
                            break
                        await asyncio.sleep(0.2)
                    
                    # 现在检查全局监听器是否已捕获下载
                    if await _consume_download_event("download-starting-fallback"):
                        return True
                    
                    # 如果仍然没有捕获到下载，尝试从下载目录恢复
                    self.logger.warning("等待 download 事件超时，尝试从下载目录恢复")
                    if await _salvage_downloaded_pdf("download-starting-salvage"):
                        return True
                    await _log_analysis("下载失败", level="warning")
                    return False
                else:
                    raise
            
            if is_doi_link:
                if is_semantic_scholar:
                    self.logger.info(f"Semantic Scholar DOI链接，使用增强等待策略")
                    # Semantic Scholar 的DOI链接需要更长的等待时间
                    await asyncio.sleep(4)
                else:
                    self.logger.info(f"检测到DOI链接，等待重定向到出版商页面")
                    await asyncio.sleep(3)
                
                # 等待网络空闲，确保重定向完成
                try:
                    await page.wait_for_load_state('networkidle', timeout=15000 if is_semantic_scholar else 10000)
                except Exception as e:
                    self.logger.debug(f"等待网络空闲超时（非致命）: {e}")
                await self._handle_cookie_consent(page, context_label="重定向后")
                self.logger.info(f"重定向完成，当前URL: {page.url}")
                await _log_analysis("重定向后")
                if await _consume_download_event("after-redirect"):
                    return True
                
                # ========== 阶段2：智能循环优先 ==========
                smart_success = await self._smart_download_loop(
                    page=page, filepath=filepath, url=url,
                    title=title, authors=authors, max_iterations=3,
                    task_download_dir=task_download_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time
                )
                if smart_success:
                    # 记录下载历史
                    try:
                        if title and authors:
                            file_hash = self._generate_paper_hash(title, authors)
                        else:
                            file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                        self.download_history[file_hash] = {
                            'path': filepath,
                            'url': url,
                            'title': title,
                            'timestamp': time.time(),
                            'status': 'success',
                            'method': 'smart_loop'
                        }
                        self.save_download_history()
                    except Exception as e:
                        self.logger.warning(f"记录下载历史失败: {e}")
                    return True
                self.logger.info(f"[兜底] 智能循环未成功，继续传统流程...")
                # ========== 阶段2结束 ==========
            else:
                await asyncio.sleep(2)
                await self._handle_cookie_consent(page, context_label="页面稳定后")
                if await _consume_download_event("after-stabilize"):
                    return True
                
                # ========== 阶段2：智能循环优先（非DOI链接）==========
                smart_success = await self._smart_download_loop(
                    page=page, filepath=filepath, url=url,
                    title=title, authors=authors, max_iterations=3,
                    task_download_dir=task_download_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time
                )
                if smart_success:
                    try:
                        if title and authors:
                            file_hash = self._generate_paper_hash(title, authors)
                        else:
                            file_hash = hashlib.md5(os.path.basename(filepath).encode()).hexdigest()
                        self.download_history[file_hash] = {
                            'path': filepath,
                            'url': url,
                            'title': title,
                            'timestamp': time.time(),
                            'status': 'success',
                            'method': 'smart_loop'
                        }
                        self.save_download_history()
                    except Exception as e:
                        self.logger.warning(f"记录下载历史失败: {e}")
                    return True
                self.logger.info(f"[兜底] 智能循环未成功，继续传统流程...")
                # ========== 阶段2结束 ==========
            
            # Springer专属：等待PDF下载区域加载（JavaScript动态插入）
            if 'link.springer.com' in page.url.lower():
                try:
                    self.logger.info("检测到Springer页面，等待PDF下载区域加载...")
                    await page.wait_for_selector('.c-pdf-download, a.c-pdf-download__link, a[data-track-action="download pdf"]', timeout=10000)
                    self.logger.info("Springer PDF下载区域已加载")
                except Exception as e:
                    self.logger.warning(f"等待Springer PDF区域超时（可能需要订阅）: {e}")
                    # 继续尝试，也许有其他方式
            
            if response and (response.status == 404 or response.status == 403):
                self.logger.error(f"URL返回404或403错误，终止下载尝试: {url}")
                await _log_analysis("下载失败", level="warning")
                return False
            # 使用通用方法检测和解决 Cloudflare 保护（JS Challenge 或 Turnstile）
            cf_success, cf_downloaded_file = await self.solve_cloudflare_if_needed(
                page, filepath, task_download_dir=task_download_dir
            )
            if cf_downloaded_file and self.is_valid_pdf_file(cf_downloaded_file):
                self.logger.info(f"Cloudflare 验证后触发下载，文件已保存: {cf_downloaded_file}")
                return True
            elif cf_success:
                if not self.headless:
                    await self.browser_manager.execute_with_ui_lock(
                        simulate_human_behavior(page)
                    )
                else:
                    await simulate_human_behavior(page)
            else:
                self.logger.warning("Cloudflare 验证码解决失败，继续尝试其他方法")
            # 检查是否是需要多步操作的站点（使用类配置）
            for site_id, site_config in self.MULTI_STEP_SITES.items():
                if site_config['detect'](url):
                    self.logger.info(f"检测到需要多步操作的站点: {site_id}")
                    wiley_pdfdirect_fallback_logged = False
                    # 多步操作：每一步都“智能判断”
                    # - 等待元素用 attached（避免元素存在但不可见导致超时）
                    # - 点击用 force=True（Wiley coolBar 常见不可见/覆盖层）
                    # - 点击前用短窗 expect_download 探测是否触发下载，触发则立即保存返回
                    # - 未触发下载则继续，允许跳转/弹窗/下一步作为兜底
                    download_probe_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                    step_timeout = self.timeouts.get("button_appear_timeout", 8) * 1000
                    for step_index, step in enumerate(site_config['steps']):
                        try:
                            analysis = await self.page_analyzer.analyze(page)
                            self.logger.debug(f"[观察点-步骤{step_index+1}前] 当前URL: {page.url[:60]}")
                            self.logger.debug(f"[观察点-步骤{step_index+1}前] 内容类型: {analysis.content_type.value}")
                            self.logger.debug(
                                f"[观察点-步骤{step_index+1}前] 阻断: {[b.type.value for b in analysis.blockers]}"
                            )
                            if analysis.actionable_elements:
                                top = analysis.actionable_elements[0]
                                self.logger.debug(
                                    f"[观察点-步骤{step_index+1}前] 最佳元素: 分数={top.score}, {top.text[:30]}"
                                )
                        except Exception as e:
                            self.logger.debug(f"页面分析失败: {e}")
                        step_selector = step['selector']
                        self.logger.info(f"执行{site_id}站点的第{step_index+1}步操作: {step_selector}")

                        try:
                            all_loc = page.locator(step_selector)
                            await all_loc.first.wait_for(state='attached', timeout=step_timeout)
                            element_count = await all_loc.count()

                            # Wiley 常见：第一个命中是隐藏的 cloned 元素；优先选可见的那个
                            loc = all_loc.first
                            is_visible = None
                            for i in range(min(element_count, 5)):
                                cand = all_loc.nth(i)
                                try:
                                    if await cand.is_visible(timeout=self.timeouts.get("button_appear_timeout", 8) * 1000):
                                        loc = cand
                                        is_visible = True
                                        break
                                except Exception:
                                    continue
                            if is_visible is None:
                                try:
                                    is_visible = await loc.is_visible(timeout=self.timeouts.get("button_appear_timeout", 8) * 1000)
                                except Exception:
                                    is_visible = None
                            try:
                                await loc.scroll_into_view_if_needed(timeout=step_timeout)
                            except Exception:
                                pass

                            # 点击前短窗探测下载是否被触发
                            download_event_caught = False
                            try:
                                async with page.expect_download(timeout=download_probe_ms) as step_download_info:
                                    try:
                                        await loc.click(timeout=step_timeout, force=True)
                                    except Exception as click_e:
                                        # 元素不可见/被遮挡时，force click 仍可能失败，降级为 JS click
                                        try:
                                            await loc.evaluate("el => el.click()")
                                        except Exception:
                                            raise click_e
                                step_download = await step_download_info.value
                                download_event_caught = True
                                # 使用统一的保存方法（并发安全）
                                if await self._save_and_validate_download(step_download, filepath, task_download_dir, f"multi_step_{step_index+1}"):
                                    self.logger.info(f"多步操作第{step_index+1}步已触发下载并保存: {filepath}")
                                    return True
                            except (asyncio.TimeoutError, PlaywrightTimeoutError):
                                # 没捕获到 download 事件很常见：很多站点先跳转到 (e)pdf 页面，再由下载按钮触发真正下载
                                # 不要重复点击，继续走后续“跳转/兜底”逻辑即可
                                pass
                            # 点击后等待页面可操作（避免 sleep）
                            try:
                                await page.wait_for_load_state('domcontentloaded', timeout=10000)
                            except Exception:
                                pass

                            # 如果下载事件太快，可能绕过 expect_download；这里兜底接管并保存
                            if latest_download.get("obj") is not None:
                                dl = latest_download["obj"]; latest_download["obj"] = None
                                await dl.save_as(filepath)
                                if self.is_valid_pdf_file(filepath):
                                    return True

                            # Wiley 专属兜底：进入 /doi/(e)pdf 后，构造 pdfdirect 直达下载并捕获下载事件
                            try:
                                current_url = page.url
                                if (site_id == 'wiley'
                                    and 'onlinelibrary.wiley.com' in current_url.lower()
                                    and ('/doi/pdf/' in current_url.lower() or '/doi/epdf/' in current_url.lower())):
                                    from urllib.parse import urlparse
                                    parsed = urlparse(current_url)
                                    m = re.search(r'/doi/(?:e?pdf)/([^?]+)', parsed.path, flags=re.IGNORECASE)
                                    wiley_doi = m.group(1) if m else None
                                    if wiley_doi:
                                        pdfdirect_url = f"{parsed.scheme}://{parsed.netloc}/doi/pdfdirect/{wiley_doi}?download=true"

                                        download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                                        async with page.expect_download(timeout=download_timeout_ms) as wiley_download_info:
                                            await page.goto(pdfdirect_url, wait_until="domcontentloaded", timeout=30000)
                                            await self._handle_cookie_consent(page, context_label="Wiley pdfdirect from (e)pdf")
                                        wiley_download = await wiley_download_info.value
                                        # 使用统一的保存方法（并发安全）
                                        if await self._save_and_validate_download(wiley_download, filepath, task_download_dir, "wiley_pdfdirect_fallback"):
                                            self.logger.info(f"Wiley pdfdirect 兜底下载成功: {filepath}")
                                            return True
                            except Exception as wiley_fallback_e:
                                if not wiley_pdfdirect_fallback_logged:
                                    self.logger.debug(f"Wiley pdfdirect 兜底失败（继续流程）: {wiley_fallback_e}")
                                    wiley_pdfdirect_fallback_logged = True

                        except Exception as e:
                            self.logger.error(f"执行多步操作第{step_index+1}步异常: {e}")
                            continue
                    
            # ===== 预先检查页面上是否有PDF下载按钮/链接 =====
            has_pdf_button = False
            clicked_selector = None
            valid_pdf_link_element = None # Store the actual element to click

            reference_ancestor_xpath = 'ancestor::*[contains(@id, "reference") or contains(@class, "reference") or contains(@id, "bibliography") or contains(@class, "bibliography")]'

            def _is_springer_supplementary(href: Optional[str]) -> bool:
                if not href:
                    return False
                lower = href.lower()
                return ('mediaobjects' in lower) or ('/esm/' in lower) or ('moesm' in lower)

            def _is_springer_main_pdf(href: Optional[str], attrs: Optional[dict] = None) -> bool:
                if not href:
                    return False
                lower = href.lower()
                if '/content/pdf/' in lower:
                    return True
                if attrs:
                    if attrs.get('data-article-pdf') == 'true' or attrs.get('data-test') == 'pdf-link':
                        return True
                return False

            def _is_innovation_main_pdf(href: Optional[str]) -> bool:
                if not href:
                    return False
                lower = href.lower()
                return '/data/article/' in lower and '/preview/pdf/' in lower

            def _is_wiley_supplementary(href: Optional[str]) -> bool:
                """检查是否是Wiley的补充材料"""
                if not href:
                    return False
                lower = href.lower()
                return '/action/downloadsupplement' in lower or 'supplement' in lower or '-sup-' in lower

            def _is_supplementary_generic(href: Optional[str], text: Optional[str] = None) -> bool:
                """通用补充材料检测（适用于所有网站）"""
                if not href and not text:
                    return False
                combined = f"{(href or '').lower()} {(text or '').lower()}"
                supp_keywords = [
                    'supplementary', 'supplement', 'supporting', 'appendix',
                    'esm', 'moesm', 'mediaobject', '-sup-', '-si-', '/si/',
                    '附件', '附录', '补充材料'
                ]
                for kw in supp_keywords:
                    if kw in combined:
                        return True
                return False

            def _is_reference_link(href: Optional[str], text: Optional[str] = None) -> bool:
                """检测是否是参考文献链接"""
                if not href and not text:
                    return False
                combined = f"{(href or '').lower()} {(text or '').lower()}"
                ref_keywords = [
                    '/reference', '#reference', 'ref-list', 'reflist',
                    'bibliography', 'cited-by', '/citation',
                    '参考文献', '引用文献'
                ]
                for kw in ref_keywords:
                    if kw in combined:
                        return True
                return False

            async def _try_external_pdf_fallback(reason: str, href_value: Optional[str] = None) -> bool:
                """外链 PDF 直链兜底：保持原有逻辑不变，仅在失败时尝试"""
                nonlocal finished
                try:
                    candidate = href_value
                    if not candidate and valid_pdf_link_element:
                        candidate = await valid_pdf_link_element.get_attribute('href')
                    if not candidate:
                        return False
                    resolved = urljoin(page.url, candidate)
                    if 'link.springer.com' in page.url.lower() and _is_springer_supplementary(resolved):
                        return False
                    if 'the-innovation.org' in page.url.lower() and not _is_innovation_main_pdf(resolved):
                        return False
                    if not resolved.lower().endswith(".pdf"):
                        return False
                    if 'mdpi' in page.url.lower() and 'flyer.pdf' in resolved.lower():
                        return False
                    self.logger.info(f"外链PDF兜底下载({reason}): {resolved}")
                    ok = await self.download_direct(
                        resolved, filepath, title=title, authors=authors, year=year
                    )
                    if ok:
                        finished = True
                        return True
                    return False
                except Exception as e:
                    self.logger.debug(f"外链PDF兜底下载失败({reason}): {e}")
                    return False

            # 使用统一配置构建选择器列表
            selectors = self._build_selector_list(url)
            
            for selector in selectors:
                try:
                    # 使用 locator 而不是 query_selector_all，以便后续操作
                    pdf_locator = page.locator(selector)
                    link_count = await pdf_locator.count()
                    if link_count > 0:
                        self.logger.debug(f"Selector '{selector}' found {link_count} potential links.")
                        # Check each link to see if it's inside a reference section
                        for i in range(link_count):
                            link_locator = pdf_locator.nth(i)
                            # Check if the link has a reference-like ancestor
                            reference_ancestors = await link_locator.locator(f'xpath={reference_ancestor_xpath}').count()
                            if reference_ancestors == 0:
                                # 针对 MDPI 的误选 pdf（如 flyer.pdf）进行过滤
                                try:
                                    href_value = await link_locator.get_attribute('href')
                                    if href_value and 'mdpi' in page.url.lower() and 'flyer.pdf' in href_value.lower():
                                        continue
                                except Exception:
                                    pass
                                if 'link.springer.com' in page.url.lower():
                                    try:
                                        href_value = await link_locator.get_attribute('href')
                                        attrs = {
                                            'data-article-pdf': await link_locator.get_attribute('data-article-pdf'),
                                            'data-test': await link_locator.get_attribute('data-test'),
                                        }
                                        if _is_springer_supplementary(href_value) and not _is_springer_main_pdf(href_value, attrs):
                                            continue
                                    except Exception:
                                        pass
                                if 'onlinelibrary.wiley.com' in page.url.lower():
                                    try:
                                        href_value = await link_locator.get_attribute('href')
                                        if _is_wiley_supplementary(href_value):
                                            self.logger.debug(f"跳过Wiley补充材料: {href_value}")
                                            continue
                                    except Exception:
                                        pass
                                if 'the-innovation.org' in page.url.lower():
                                    try:
                                        href_value = await link_locator.get_attribute('href')
                                        if not _is_innovation_main_pdf(href_value):
                                            continue
                                    except Exception:
                                        pass
                                # 通用补充材料和参考文献过滤（适用于所有网站）
                                try:
                                    href_value = await link_locator.get_attribute('href')
                                    link_text = await link_locator.inner_text()
                                    if _is_supplementary_generic(href_value, link_text):
                                        self.logger.debug(f"跳过通用补充材料: {href_value or link_text[:30]}")
                                        continue
                                    if _is_reference_link(href_value, link_text):
                                        self.logger.debug(f"跳过参考文献链接: {href_value or link_text[:30]}")
                                        continue
                                except Exception:
                                    pass
                                # Found a valid link not in a reference section
                                self.logger.info(f"Found valid PDF link/button: {selector}")
                                has_pdf_button = True
                                clicked_selector = selector
                                valid_pdf_link_element = link_locator # Store the valid locator
                                break # Stop checking links for this selector
                            else:
                                self.logger.debug(f"Skipping link for selector '{selector}' as it appears to be in a reference section.")

                    if has_pdf_button:
                        break # Stop checking other selectors once a valid link is found

                except Exception as e:
                    self.logger.debug(f"尝试选择器 {selector} 时出错: {str(e)}")

            # 多轮查找逻辑：如果第一轮未找到PDF按钮，尝试其他策略
            if not has_pdf_button:
                self.logger.info("第一轮未找到PDF按钮，尝试多轮查找策略")
                
                # 第二轮：滚动页面后重试（某些按钮延迟加载）
                try:
                    self.logger.info("第二轮：滚动页面到底部，等待延迟加载内容")
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)
                    
                    # 重新查找（使用相同的选择器列表）
                    for selector in selectors:
                        try:
                            pdf_locator = page.locator(selector)
                            link_count = await pdf_locator.count()
                            if link_count > 0:
                                for i in range(link_count):
                                    link_locator = pdf_locator.nth(i)
                                    reference_ancestors = await link_locator.locator(f'xpath={reference_ancestor_xpath}').count()
                                    if reference_ancestors == 0:
                                        # 针对 MDPI 的误选 pdf（如 flyer.pdf）进行过滤
                                        try:
                                            href_value = await link_locator.get_attribute('href')
                                            if href_value and 'mdpi' in page.url.lower() and 'flyer.pdf' in href_value.lower():
                                                continue
                                        except Exception:
                                            pass
                                        if 'link.springer.com' in page.url.lower():
                                            try:
                                                href_value = await link_locator.get_attribute('href')
                                                attrs = {
                                                    'data-article-pdf': await link_locator.get_attribute('data-article-pdf'),
                                                    'data-test': await link_locator.get_attribute('data-test'),
                                                }
                                                if _is_springer_supplementary(href_value) and not _is_springer_main_pdf(href_value, attrs):
                                                    continue
                                            except Exception:
                                                pass
                                        if 'the-innovation.org' in page.url.lower():
                                            try:
                                                href_value = await link_locator.get_attribute('href')
                                                if not _is_innovation_main_pdf(href_value):
                                                    continue
                                            except Exception:
                                                pass
                                        # 通用补充材料和参考文献过滤
                                        try:
                                            href_value = await link_locator.get_attribute('href')
                                            link_text = await link_locator.inner_text()
                                            if _is_supplementary_generic(href_value, link_text):
                                                continue
                                            if _is_reference_link(href_value, link_text):
                                                continue
                                        except Exception:
                                            pass
                                        self.logger.info(f"第二轮找到PDF链接: {selector}")
                                        has_pdf_button = True
                                        clicked_selector = selector
                                        valid_pdf_link_element = link_locator
                                        break
                            if has_pdf_button:
                                break
                        except Exception as e:
                            self.logger.debug(f"第二轮尝试选择器 {selector} 时出错: {str(e)}")
                except Exception as e:
                    self.logger.debug(f"第二轮查找出错: {e}")
                
                # 第三轮：尝试更宽松的文本匹配（仅针对Semantic Scholar或DOI链接）
                if not has_pdf_button and (is_semantic_scholar or is_doi_link):
                    try:
                        self.logger.info("第三轮：使用宽松的文本匹配查找PDF链接")
                        loose_selectors = [
                            'a:has-text("PDF")',
                            'a:has-text("pdf")',
                            'button:has-text("PDF")',
                            'a:has-text("Download")',
                            'a[href*="pdf"]',
                        ]
                        
                        for selector in loose_selectors:
                            try:
                                pdf_locator = page.locator(selector)
                                link_count = await pdf_locator.count()
                                if link_count > 0:
                                    for i in range(link_count):
                                        link_locator = pdf_locator.nth(i)
                                        reference_ancestors = await link_locator.locator(f'xpath={reference_ancestor_xpath}').count()
                                        if reference_ancestors == 0:
                                            # 通用补充材料和参考文献过滤
                                            try:
                                                href_value = await link_locator.get_attribute('href')
                                                link_text = await link_locator.inner_text()
                                                if _is_supplementary_generic(href_value, link_text):
                                                    continue
                                                if _is_reference_link(href_value, link_text):
                                                    continue
                                            except Exception:
                                                pass
                                            self.logger.info(f"第三轮找到PDF链接: {selector}")
                                            has_pdf_button = True
                                            clicked_selector = selector
                                            valid_pdf_link_element = link_locator
                                            break
                                if has_pdf_button:
                                    break
                            except Exception as e:
                                self.logger.debug(f"第三轮尝试选择器 {selector} 时出错: {str(e)}")
                    except Exception as e:
                        self.logger.debug(f"第三轮查找出错: {e}")
            
            if not has_pdf_button:
                self.logger.warning(f"所有查找策略均未找到PDF下载按钮: {url}")
            else:
                self.logger.info(f"已找到PDF下载按钮: {clicked_selector}")

            # 检测是否可能打开新页面
            will_open_new_page = False
            if has_pdf_button and valid_pdf_link_element: # Check using the stored valid element
                # 检查链接是否有target="_blank"或其他会导致新页面的属性
                target_attr = None
                href_attr = None
                try:
                    # 获取target属性
                    target_attr = await valid_pdf_link_element.get_attribute('target')
                    # 获取href属性以检查是否是外部链接
                    href_attr = await valid_pdf_link_element.get_attribute('href')
                    is_pdf_link = bool(href_attr and href_attr.lower().endswith(".pdf"))

                    # 判断是否会打开新页面
                    if target_attr == '_blank':
                        will_open_new_page = True
                    elif href_attr and (href_attr.startswith('http') or href_attr.startswith('//')):
                        # 外部链接不一定会开新页；PDF 直链优先走直接下载分支
                        will_open_new_page = not is_pdf_link
                    else:
                        will_open_new_page = False
                    if will_open_new_page:
                        self.logger.info(f"检测到链接可能会打开新页面: target='{target_attr}', href='{href_attr}'")
                except Exception as e:
                    self.logger.debug(f"检查链接属性时出错: {str(e)}")

            # 处理可能打开新页面的情况
            if has_pdf_button and will_open_new_page and valid_pdf_link_element:
                self.logger.info("准备处理打开新页面的情况")

                # 准备监听新页面事件（先点击，再等待 popup）
                async with page.expect_popup() as popup_info:
                    try:
                        await valid_pdf_link_element.click()
                        self.logger.info(f"已点击链接(可能打开新页面): {clicked_selector}")
                    except Exception as e:
                        self.logger.error(f"尝试点击选择器 {clicked_selector} 时出错: {str(e)}")
                    # 等待新页面加载，新页面可能直接出发下载事件，进行检测
                    try:
                        new_page = await popup_info.value
                        self._active_pages.add(new_page)
                        self.logger.info(f"成功捕获新页面: {new_page.url}")
                        await self._handle_cookie_consent(new_page, context_label="新页面打开后")
                        download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                        new_page_download = {"obj": None}
                        def _on_new_page_download(download):
                            new_page_download["obj"] = download
                        new_page_download_handler = _on_new_page_download
                        new_page.on("download", _on_new_page_download)
                        try:
                            download_wait_seconds = max(1, download_timeout_ms / 1000)
                            wait_loops = int(download_wait_seconds / 0.2)
                            for _ in range(wait_loops):
                                if new_page_download.get("obj") is not None:
                                    break
                                await asyncio.sleep(0.2)
                            download = new_page_download.get("obj")
                            if download is not None:
                                new_page_download["obj"] = None
                                # 使用统一的保存方法（并发安全）
                                if await self._save_and_validate_download(download, filepath, task_download_dir, "new_page_download"):
                                    finished = True
                                    return True
                                else:
                                    self.logger.error(f"新页面下载的文件不是有效的PDF: {filepath}")
                                    finished = False
                            else:
                                self.logger.info("新页面未触发下载事件，继续后续处理")
                        except Exception as e:
                            self.logger.error(f"处理新页面下载时出错: {str(e)}")
                        
                        # 使用通用方法处理潜在的验证码问题
                        try:
                            self.logger.info("使用通用方法检测新页面的Cloudflare验证...")
                            cf_success, cf_downloaded_file = await self.solve_cloudflare_if_needed(
                                new_page, filepath, task_download_dir=task_download_dir
                            )
                            
                            if cf_downloaded_file and self.is_valid_pdf_file(cf_downloaded_file):
                                self.logger.info(f"新页面Cloudflare验证后触发下载，文件已保存: {cf_downloaded_file}")
                                finished = True
                                return True
                            elif not cf_success:
                                self.logger.warning("新页面Cloudflare验证码解决失败")
                                finished = False
                        except Exception as e:
                            self.logger.error(f"处理新页面验证码时出错: {str(e)}")
                            

                        #尝试点击潜在的按钮
                        try:
                            async with new_page.expect_download(timeout=download_timeout_ms) as download_button_info:
                                try:
                                    await self._handle_cookie_consent(new_page, context_label="新页面点击按钮前")
                                    await self.click_open_button_if_found(
                                        new_page, task_download_dir=task_download_dir
                                    )
                                except Exception as e:
                                    pass
                            download = await download_button_info.value
                            # 使用统一的保存方法（并发安全）
                            if await self._save_and_validate_download(download, filepath, task_download_dir, "button_download"):
                                finished = True
                                return True
                            else:
                                self.logger.error(f"新页面下载的文件不是有效的PDF: {filepath}")
                                finished = False

                        except Exception as e:
                            self.logger.error(f"处理新页面时出错: {str(e)}")
                            finished = False
                    except Exception as e:
                        finished = False
                        self.logger.error(f"处理新页面时出错: {str(e)}")

                if not finished:
                    await _try_external_pdf_fallback("popup失败", href_attr)
            #现在处理直接打开页面的情况

            if has_pdf_button and valid_pdf_link_element and not will_open_new_page:
                try:
                    async with page.expect_download(timeout = self.timeout * 2000) as download_info:
                        await self._handle_cookie_consent(page, context_label="页面点击前")
                        try: 
                            await valid_pdf_link_element.click()
                        except Exception as e:
                            pass
                    download = await download_info.value
                    # 使用统一的保存方法（并发安全）
                    if await self._save_and_validate_download(download, filepath, task_download_dir, "direct_open_page"):
                        finished = True
                        return True
                    else:
                        self.logger.error(f"直接打开页面下载的文件不是有效的PDF: {filepath}")
                        finished = False
                except Exception as e:
                    self.logger.error(f"处理直接打开页面时出错: {str(e)}")
                    finished = False
                #尝试解决潜在的点击问题
                try:
                    async with page.expect_download(timeout=self.timeout * 2000) as download_info:
                        try:
                            await self._handle_cookie_consent(page, context_label="页面按钮前")
                            await self.click_open_button_if_found(
                                page, task_download_dir=task_download_dir
                            )
                        except:
                            pass
                    download = await download_info.value
                    # 使用统一的保存方法（并发安全）
                    if await self._save_and_validate_download(download, filepath, task_download_dir, "direct_open_page_2"):
                        finished = True
                        return True
                    else:
                        self.logger.error(f"直接打开页面下载的文件不是有效的PDF: {filepath}")
                        finished = False
                except Exception as e:
                    self.logger.error(f"处理直接打开页面时出错: {str(e)}")
                    finished = False

                if not finished:
                    await _try_external_pdf_fallback("直接下载失败")

            # === 兜底：如果下载“发生了但没被接管”，在这里统一接管并落到 filepath ===
            if not finished:
                if await _consume_download_event("end-of-flow"):
                    return True
                if await _salvage_downloaded_pdf("end-of-flow"):
                    return True

        except Exception as e:
            self.logger.error(f"查找和下载PDF时出错: {str(e)}")
            await _log_analysis("下载失败", level="warning")
            return False
        finally:
            # 等待可能的下载完成（避免过早关闭）
            try:
                if not finished:
                    if latest_download.get("obj") is not None:
                        try:
                            saved = await asyncio.wait_for(
                                _consume_download_event("finally-cleanup"),
                                timeout=10.0
                            )
                            if saved:
                                finished = True
                        except asyncio.TimeoutError:
                            pass
                    if not finished and await self._check_download_in_progress(
                        filepath, task_download_dir=task_download_dir
                    ):
                        await self._wait_for_download_complete(
                            filepath,
                            timeout=min(self.timeouts.get("download_complete_timeout", 20), 10),
                            task_download_dir=task_download_dir,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time
                        )
            except Exception as wait_e:
                self.logger.debug(f"等待下载完成失败（关闭前）: {wait_e}")

            # ====== Bug修复：清理任务专属临时目录 ======
            try:
                if 'task_download_dir' in dir() and os.path.exists(task_download_dir):
                    shutil.rmtree(task_download_dir, ignore_errors=True)
            except Exception:
                pass
            # ====== 清理结束 ======
            
            # 关闭所有打开的页面
            if new_page:
                try:
                    if new_page_download_handler:
                        try:
                            new_page.off("download", new_page_download_handler)
                        except Exception:
                            pass
                    if not new_page.is_closed():
                        await asyncio.wait_for(new_page.close(), timeout=3.0)
                    self._active_pages.discard(new_page)
                except Exception as e:
                    self.logger.warning(f"关闭新页面时出错: {str(e)}")
            
            if page:
                try:
                    try:
                        page.off("download", _on_download)
                    except Exception:
                        pass
                    await page.close()
                    self._active_pages.discard(page)
                except Exception as e:
                    self.logger.warning(f"关闭页面时出错: {str(e)}")
            
        return finished

def setup_signal_handlers(downloader):
    """设置信号处理器以捕获中断信号
    
    Args:
        downloader: PaperDownloader实例
    """
    def signal_handler(sig, frame):
        logger.info("接收到中断信号，保存历史记录...")
        downloader.save_download_history()
        logger.info("历史记录已保存，程序退出")
        sys.exit(0)
    
    # 注册SIGINT (Ctrl+C) 和 SIGTERM (终止信号)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """主函数，处理命令行参数并执行下载任务"""
    parser = argparse.ArgumentParser(description="从JSON搜索结果下载论文PDF文件")
    parser.add_argument("input", help="输入JSON文件路径或包含JSON文件的目录")
    parser.add_argument("--output", "-o", default="papers", help="下载文件保存目录")
    parser.add_argument("--concurrent", "-c", type=int, default=5, help="最大并发下载数")
    parser.add_argument("--show-browser", action="store_true", help="显示浏览器窗口，允许手动操作")
    parser.add_argument("--no-persist-browser", action="store_false", dest="persist_browser", help="禁用浏览器会话持久化")
    parser.add_argument("--cleanup", action="store_true", help="启动前清理所有浏览器锁定文件")
    parser.add_argument("--force-cleanup", action="store_true", help="程序结束后强制清理资源，包括正在进行的下载")
    parser.add_argument("--retry-failed", action="store_true", help="重试之前失败的下载")
    parser.add_argument("--timeout", type=int, default=10, help="网络加载时间(秒)")
    parser.add_argument("--download_timeout", type=int, default=180, help="单个文件下载超时时间(秒)")
    parser.add_argument("--max-retries", type=int, default=3, help="下载失败时的最大重试次数")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help="日志级别")
    parser.add_argument("--ask-on-exit", action="store_true", help="程序结束时询问是否关闭浏览器")
    parser.add_argument("--browser", choices=['chromium', 'chrome', 'firefox'], default='chrome',
                        help="")
    parser.add_argument("--no-stealth", action="store_false", dest="stealth_mode", 
                       help="禁用浏览器隐身模式（可能会被网站检测为机器人）", default=True)
    

    args = parser.parse_args()

    cleanup_old_logs(log_dir = "logs", days = 1)
    
    # 设置日志级别
    # 显示启动参数
    logger.info("\n下载参数:")
    logger.info(f"输入: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"并发数: {args.concurrent}")
    logger.info(f"显示浏览器: {args.show_browser}")
    logger.info(f"持久化浏览器: {args.persist_browser}")
    logger.info(f"清理锁文件: {args.cleanup}")
    logger.info(f"强制清理: {args.force_cleanup}")
    logger.info(f"退出时询问: {args.ask_on_exit}")
    logger.info(f"网络加载超时时间: {args.timeout}秒")
    logger.info(f"单个文件下载超时时间: {args.download_timeout}秒")
    logger.info(f"最大重试: {args.max_retries}次")
    logger.info(f"日志级别: {args.log_level}\n")
    logger.info(f"隐身模式: {args.stealth_mode}")

    log_level = args.log_level.upper()
    logger.setLevel(log_level)
    
    try:
        # 检查输入路径
        if not os.path.exists(args.input):
            logger.error(f"输入路径不存在: {args.input}")
            return 1
        
        # 检查并规范化输出目录路径
        args.output = os.path.abspath(args.output)
        if not os.path.exists(args.output):
            try:
                os.makedirs(args.output)
                logger.info(f"创建输出目录: {args.output}")
            except Exception as e:
                logger.error(f"创建输出目录失败: {str(e)}")
                return 1
        elif not os.path.isdir(args.output):
            logger.error(f"输出路径不是目录: {args.output}")
            return 1
        elif not os.access(args.output, os.W_OK):
            logger.error(f"输出目录没有写入权限: {args.output}")
            return 1
        
        # 验证并调整并发数
        if args.concurrent < 1:
            logger.warning("并发数不能小于1，已调整为1")
            args.concurrent = 1
        elif args.concurrent > 10:
            logger.warning("并发数过大可能导致不稳定，建议不超过10")
        
        # 验证超时时间
        if args.timeout < 10:
            logger.warning("超时时间过短，已调整为60秒")
            args.timeout = 10
        elif args.timeout > 1800:  # 30分钟
            logger.warning("超时时间过长，已调整为1800秒(30分钟)")
            args.timeout = 1800
        
        # 验证重试次数
        if args.max_retries < 1:
            logger.warning("重试次数不能小于1，已调整为1")
            args.max_retries = 1
        elif args.max_retries > 5:
            logger.warning("重试次数过多，已调整为5")
            args.max_retries = 5
        
        # 如果指定了cleanup，先清理所有锁文件
        if args.cleanup:
            logger.info("清理所有浏览器锁定文件...")
            try:
                downloader = PaperDownloader(
                    args.output,
                    args.concurrent,
                    args.show_browser,
                    args.persist_browser,
                    download_timeout=args.download_timeout,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    browser_type=args.browser
                )
                downloader.clean_browser_lock_files()
            except Exception as e:
                logger.error(f"清理锁定文件失败: {str(e)}")
                return 1
        
        # 初始化downloader变量，确保finally块可以访问
        downloader = None
        
        # 处理输入文件
        if os.path.isdir(args.input):
            # 如果是目录，处理所有JSON文件
            json_files = [f for f in os.listdir(args.input) if f.endswith('.json')]
            if not json_files:
                logger.error(f"目录中没有JSON文件: {args.input}")
                return 1
            
            total_stats = {
                'total': 0,
                'success': 0,
                'failed': 0,
                'skipped': 0,
                'verified': 0,
                'corrupted': 0,
                'retry': 0,
                'already_downloaded': 0
            }
            
            for json_file in json_files:
                file_path = os.path.join(args.input, json_file)
                logger.info(f"\n处理JSON文件: {json_file}")
                
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        papers = json.load(f)
                    
                    # 创建下载器实例
                    downloader = PaperDownloader(
                        args.output,
                        args.concurrent,
                        args.show_browser,
                        args.persist_browser,
                        download_timeout=args.download_timeout,
                        max_retries=args.max_retries,
                        browser_type=args.browser,
                        stealth_mode=args.stealth_mode # 传递命令行参数
                    )
                    
                    # 设置信号处理器
                    setup_signal_handlers(downloader)
                    
                    # 启动自动保存任务
                    await downloader.start_auto_save(interval=30)  # 每30秒自动保存
                    
                    # 执行下载
                    stats = await downloader.process_papers(papers)
                    
                    # 更新总统计
                    for key in total_stats:
                        if key in stats:
                            total_stats[key] += stats[key]
                    
                    # 清理资源（保留浏览器会话，除非是最后一个文件）
                    is_last_file = json_file == json_files[-1]
                    if is_last_file:
                        # 最后一个文件处理完毕后，根据参数决定如何清理资源
                        force_close = args.force_cleanup
                        ask_user = args.ask_on_exit and args.show_browser
                        await downloader.cleanup_resources(force_close=force_close, ask_user=ask_user)
                    else:
                        # 不是最后一个文件，只保存历史，不关闭浏览器会话
                        downloader.save_download_history()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON文件格式错误 ({json_file}): {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"处理文件时出错 ({json_file}): {str(e)}")
                    continue
            
            # 显示总统计信息
            success_rate = (total_stats['success'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0
            logger.info("\n总下载统计:")
            logger.info(f"总数: {total_stats['total']}")
            logger.info(f"成功: {total_stats['success']} ({success_rate:.1f}%)")
            logger.info(f"失败: {total_stats['failed']}")
            logger.info(f"跳过: {total_stats['skipped']}")
            logger.info(f"验证: {total_stats['verified']}")
            logger.info(f"损坏: {total_stats['corrupted']}")
            logger.info(f"重试: {total_stats['retry']}")
            logger.info(f"已下载: {total_stats['already_downloaded']}")
        else:
            # 单个JSON文件
            try:
                # 读取JSON文件
                with open(args.input, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                
                # 创建下载器实例
                downloader = PaperDownloader(
                    args.output,
                    args.concurrent,
                    args.show_browser,
                    args.persist_browser,
                    download_timeout=args.download_timeout,
                    max_retries=args.max_retries,
                    browser_type=args.browser,
                    stealth_mode=args.stealth_mode  # 传递命令行参数
                )
                
                # 设置信号处理器
                setup_signal_handlers(downloader)
                
                # 启动自动保存任务
                await downloader.start_auto_save(interval=30)  # 每30秒自动保存
                
                # 执行下载
                stats = await downloader.process_papers(papers)
                
                # 显示统计信息
                success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
                logger.info("\n下载统计:")
                logger.info(f"总数: {stats['total']}")
                logger.info(f"成功: {stats['success']} ({success_rate:.1f}%)")
                logger.info(f"失败: {stats['failed']}")
                logger.info(f"跳过: {stats['skipped']}")
                logger.info(f"验证: {stats['verified']}")
                logger.info(f"损坏: {stats['corrupted']}")
                logger.info(f"重试: {stats['retry']}")
                logger.info(f"已下载: {stats['already_downloaded']}")
                
                # 清理资源，根据参数决定如何清理
                force_close = args.force_cleanup
                ask_user = args.ask_on_exit and args.show_browser
                await downloader.cleanup_resources(force_close=force_close, ask_user=ask_user)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON文件格式错误: {str(e)}")
                return 1
            except Exception as e:
                logger.error(f"处理文件时出错: {str(e)}")
                return 1
        
        # 保存历史记录
        downloader.save_download_history()
        downloader.stop_auto_save()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n用户中断，正在清理资源...")
        if 'downloader' in locals() and downloader is not None:
            try:
                # 检查是否有正在进行的下载
                logger.info("检查是否有正在进行的下载...")
                # 给用户5秒时间决定是否立即终止
                logger.info("按Ctrl+C再次中断将立即终止所有下载")
                logger.info("否则将等待当前下载完成后再清理资源")
                try:
                    await asyncio.sleep(2)
                except KeyboardInterrupt:
                    logger.info("用户确认立即终止")
                    # 强制终止
                    await downloader.cleanup_resources(force_close=True)
                    return 130
                
                # 用户没有再次中断，尝试等待当前下载完成
                logger.info("等待当前下载完成后再清理资源...")
                # 询问用户是否关闭浏览器
                ask_user = args.ask_on_exit and args.show_browser
                await downloader.cleanup_resources(force_close=False, ask_user=ask_user)
            except Exception as e:
                logger.error(f"清理资源时出错: {str(e)}")
        return 130
    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # 确保所有资源都被正确释放
        if 'downloader' in locals() and downloader is not None:
            try:
                if hasattr(downloader, '_playwright') and downloader._playwright:
                    await downloader._playwright.stop()
            except Exception as e:
                logger.error(f"关闭playwright时出错: {str(e)}")

# 运行主函数
if __name__ == "__main__":
    import sys
    import asyncio
    
    if sys.platform == 'win32':
        # Windows平台需要特殊处理asyncio事件循环
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)