import os
import html
import json
import time
import hashlib
import uuid
import argparse
import shutil
import subprocess
import asyncio
import aiohttp
import base64
import tempfile
import contextvars
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
import sys
import re
import signal
import threading
from urllib.parse import quote, unquote, urljoin, urlparse
from .browser_manager import BrowserManager, simulate_human_behavior, setup_browser
from playwright.async_api import TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from src.log import get_logger
from src.llm.llm_manager import get_manager
import logging
from bs4 import BeautifulSoup
try:
    from twocaptcha import TwoCaptcha
except ImportError:
    TwoCaptcha = None  # optional: captcha solving disabled when not installed

from .captcha_solver import CaptchaType, CaptchaSolver, CaptchaSolveResult  # noqa: E402

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

# 配置日志（与项目 src.log 统一）
logger = get_logger("scholar_downloader.PaperDownloader")

# Sci-Hub 下载结果原因码（用于省钱：遇到明确 not_in_db 时跳过 Bright Data + Sci-Hub）
SCIHUB_OK = "ok"
SCIHUB_NOT_IN_DB = "not_in_db"
SCIHUB_BLOCKED = "blocked_or_challenge"
SCIHUB_TIMEOUT = "timeout"
SCIHUB_ERROR = "error"

# Sci-Hub 现在通过 BrightData + sci-hub.st 实现，无需开关
# Playwright 版本已废弃（DDoS-Guard 不稳定）


def _project_config_root() -> str:
    """项目 config 目录（与 config/settings、src/db/engine 等一致：repo/config）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config")


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并 override 到 base，与 config/settings 行为一致。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_rag_config() -> Tuple[dict, str]:
    """
    加载项目全局配置：config/rag_config.json + config/rag_config.local.json（本地覆盖）。
    返回 (合并后的原始配置, 使用的 config 目录路径)。
    """
    root = _project_config_root()
    base_path = os.path.join(root, "rag_config.json")
    local_path = os.path.join(root, "rag_config.local.json")
    raw = {}
    if os.path.exists(base_path):
        try:
            with open(base_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning(f"加载 {base_path} 失败: {e}")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local = json.load(f)
            raw = _deep_merge(raw, local)
        except Exception as e:
            logger.warning(f"加载 {local_path} 失败: {e}")
    return raw, root


def _normalize_rag_config_to_downloader(raw: dict) -> dict:
    """
    将 rag_config 结构转为 PaperDownloader 使用的结构：
    api_keys (twocaptcha, capsolver, brightdata), downloader (proxy, timeouts, experience_store_path, capsolver_extension_path)。
    与 adapter 中 initial_config 的构建逻辑对齐。
    """
    sd = raw.get("scholar_downloader") or {}
    cf = raw.get("content_fetcher") or {}
    twocaptcha = (sd.get("twocaptcha_api_key") or "").strip()
    if not twocaptcha:
        twocaptcha = (cf.get("two_captcha_api_key") or "").strip()
    # CapSolver API key from top-level capsolver section
    capsolver = (raw.get("capsolver") or {}).get("api_key", "")
    capsolver = (capsolver or sd.get("capsolver_api_key") or "").strip()
    brightdata = (cf.get("brightdata_api_key") or "").strip()
    api_keys = {
        "twocaptcha": twocaptcha,
        "capsolver": capsolver,
        "brightdata": brightdata,
    }
    dl = {
        "proxy": sd.get("proxy"),
        "experience_store_path": sd.get("experience_store_path"),
        "timeouts": sd.get("timeouts") or {},
    }
    ext = (raw.get("capsolver_extension_path") or sd.get("capsolver_extension_path") or "").strip()
    if ext:
        dl["capsolver_extension_path"] = ext
    return {
        "api_keys": api_keys,
        "downloader": dl,
        "annas_keyword_max_pages": sd.get("annas_keyword_max_pages", 5),
    }


def load_config(config_path: Optional[str] = "config.json") -> dict:
    """
    加载配置。与项目全局配置一致：
    - 若 config_path 为 None、默认 "config.json" 或该路径不存在，则加载
      config/rag_config.json + config/rag_config.local.json 并归一化为下载器结构。
    - 否则从给定路径加载；若内容含 scholar_downloader 则同样归一化。
    """
    use_project = config_path is None or config_path == "config.json"
    if use_project or not (config_path and os.path.exists(config_path)):
        raw, _ = _load_rag_config()
        if raw:
            return _normalize_rag_config_to_downloader(raw)
        if use_project:
            logger.debug("未找到项目 config，使用空配置")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception as e:
        logger.warning(f"无法加载配置文件 {config_path}: {e}")
        return {}
    if "scholar_downloader" in loaded or "content_fetcher" in loaded:
        return _normalize_rag_config_to_downloader(loaded)
    return loaded


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
    """可操作元素（含上下文与相对位置，供启发式打分与 LLM rerank 使用）"""
    selector: str
    tag: str
    text: str
    element_id: str = ""
    href: Optional[str] = None
    is_visible: bool = True
    is_fixed: bool = False
    rect_top: float = 0.0
    position_y: int = 0
    score: float = 0.0
    attributes: Dict[str, str] = field(default_factory=dict)
    # 上下文与相对位置（enrichment）
    position_ratio: Optional[float] = None  # 元素顶部在文档中的相对位置 (0~1)
    in_viewport: bool = False
    ancestor_path: str = ""  # 祖先节点 id/class 路径，逗号分隔
    nearby_text_before: str = ""
    nearby_text_after: str = ""
    ancestor_features: str = ""
    container_role: str = ""  # header, main, sidebar, footer, reference, unknown
    site_source: str = ""  # 当前页 host，由调用方从 page.url 填入


@dataclass
class ScoreConfig:
    """PDF 主文下载候选的启发式打分配置（零侵入式可调）。"""
    fixed_bonus: float = 40.0
    fixed_top_bonus: float = 20.0
    early_position_max_bonus: float = 30.0
    late_position_penalty: float = 40.0
    good_container_bonus: float = 20.0
    sidebar_top_bonus: float = 15.0
    danger_zone_penalty: float = 80.0
    positive_text_bonus: float = 50.0
    file_size_regex_bonus: float = 20.0
    negative_text_penalty: float = 100.0
    positive_href_bonus: float = 40.0
    negative_href_penalty: float = 100.0
    hard_reject_score: float = -9999.0
    # 低置信度时返回空，不强行瞎选
    min_confidence_to_accept: float = 0.50

    positive_text_keywords: Tuple[str, ...] = (
        "open pdf", "download pdf", "full text", "article pdf", "epdf",
        "view pdf", "下载 pdf", "打开 pdf",
    )
    negative_text_keywords: Tuple[str, ...] = (
        "supplement", "support", "reference", "citation", "export",
        "figure", "table", "appendix", "data",
    )
    positive_href_keywords: Tuple[str, ...] = (
        "/doi/pdf/", ".pdf", "/content/pdf/", "pdf?", "/epdf/",
    )
    negative_href_keywords: Tuple[str, ...] = (
        "/suppl/", "supplementary", "appendix", "citation", "ris", "bib",
    )
    danger_zone_keywords: Tuple[str, ...] = (
        "footer", "references", "related", "recommendation",
        "metrics", "history", "advertisement",
    )
    sidebar_keywords: Tuple[str, ...] = (
        "sidebar", "aside", "article-tools", "panel",
    )
    primary_zone_keywords: Tuple[str, ...] = (
        "header", "navbar", "toolbar", "main",
    )
    file_size_regex: str = r"\d+(\.\d+)?\s*(mb|kb)"


@dataclass
class Candidate:
    """PDF 候选元素。

    注：前端提取 text 时应合并 innerText、aria-label、title 及内部 img/svg alt，
    并在代码中统一转小写比对。
    """
    element_id: str
    text: str
    href: str
    is_fixed: bool
    rect_top: float
    position_ratio: float
    ancestor_features: str
    is_visible: bool
    score: float = 0.0


class PDFExtractor:
    """两阶段选择器：启发式粗排 + 单轮轻量 LLM 精排。"""

    _RERANK_PROMPT_CACHE: Optional[str] = None

    def __init__(self, logger: logging.Logger, score_config: Optional[ScoreConfig] = None):
        self.logger = logger
        self.score_config = score_config or ScoreConfig()
        self._file_size_re = re.compile(self.score_config.file_size_regex, re.IGNORECASE)

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    @staticmethod
    def _coerce_ratio(value: Optional[float]) -> float:
        if value is None:
            return 0.5
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.5

    def to_candidate(self, elem: ActionableElement) -> Candidate:
        return Candidate(
            element_id=(getattr(elem, "element_id", "") or getattr(elem, "selector", "") or "").strip(),
            text=self._normalize_text(getattr(elem, "text", "")),
            href=self._normalize_text(getattr(elem, "href", "")),
            is_fixed=bool(getattr(elem, "is_fixed", False)),
            rect_top=float(getattr(elem, "rect_top", 0.0) or 0.0),
            position_ratio=self._coerce_ratio(getattr(elem, "position_ratio", None)),
            ancestor_features=self._normalize_text(
                getattr(elem, "ancestor_features", "") or getattr(elem, "ancestor_path", "")
            ),
            is_visible=bool(getattr(elem, "is_visible", True)),
            score=float(getattr(elem, "score", 0.0) or 0.0),
        )

    def _calculate_score(self, candidate: Candidate) -> float:
        cfg = self.score_config
        text = self._normalize_text(candidate.text)
        href = self._normalize_text(candidate.href)
        ancestor_features = self._normalize_text(candidate.ancestor_features)

        if not candidate.is_visible or (not text and not href):
            return cfg.hard_reject_score

        score = 0.0

        # 1) 位置与固定特征
        if candidate.is_fixed:
            score += cfg.fixed_bonus
            if candidate.rect_top < 200:
                score += cfg.fixed_top_bonus
            # fixed/sticky 不能因整页位置靠后被扣分
        else:
            score += max(0.0, (1.0 - candidate.position_ratio) * cfg.early_position_max_bonus)
            if candidate.position_ratio > 0.8:
                score -= cfg.late_position_penalty

        # 2) DOM 上下文区域
        hit_danger_zone = any(kw in ancestor_features for kw in cfg.danger_zone_keywords)
        if hit_danger_zone:
            score -= cfg.danger_zone_penalty

        hit_sidebar_zone = any(kw in ancestor_features for kw in cfg.sidebar_keywords)
        if hit_sidebar_zone and not hit_danger_zone:
            if candidate.is_fixed or candidate.position_ratio < 0.3:
                score += cfg.sidebar_top_bonus

        if any(kw in ancestor_features for kw in cfg.primary_zone_keywords):
            score += cfg.good_container_bonus

        # 3) 文案特征
        if any(kw in text for kw in cfg.positive_text_keywords):
            score += cfg.positive_text_bonus
        if self._file_size_re.search(text):
            score += cfg.file_size_regex_bonus
        if any(kw in text for kw in cfg.negative_text_keywords):
            score -= cfg.negative_text_penalty

        # 4) 链接特征
        if any(kw in href for kw in cfg.positive_href_keywords):
            score += cfg.positive_href_bonus
        if any(kw in href for kw in cfg.negative_href_keywords):
            score -= cfg.negative_href_penalty

        return score

    def score_and_filter(self, candidates: List[Candidate]) -> List[Candidate]:
        ranked: List[Candidate] = []
        for candidate in candidates:
            candidate.score = self._calculate_score(candidate)
            if candidate.score >= 0:
                ranked.append(candidate)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _get_llm_clients(
        self,
        mode: str,
        provider: Optional[str] = None,
    ) -> List[Any]:
        manager = get_manager()
        normalized_mode = (mode or "ultra-lite").strip().lower()
        if normalized_mode == "lite":
            return [manager.get_lite_client(provider)]
        if normalized_mode == "auto-upgrade":
            # 先 ultra-lite，低置信度再升级到 lite
            return [manager.get_ultra_lite_client(provider), manager.get_lite_client(provider)]
        return [manager.get_ultra_lite_client(provider)]

    @staticmethod
    def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        # Find outermost {...} to tolerate leading/trailing prose from the LLM
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return None

    async def rerank_candidates(
        self,
        candidates: List[Candidate],
        *,
        top_n: int = 3,
        mode: str = "ultra-lite",
        provider: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ranked = self.score_and_filter(candidates)
        if not ranked:
            return None
        selected = ranked[:max(1, top_n)]
        payload = [
            {
                "id": item.element_id,
                "text": item.text,
                "href": item.href,
                "is_fixed": item.is_fixed,
                "position_ratio": item.position_ratio,
                "score": item.score,
            }
            for item in selected
        ]
        if PDFExtractor._RERANK_PROMPT_CACHE is None:
            prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts")
            prompt_path = os.path.join(prompt_dir, "downloader_candidate_rerank.txt")
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    PDFExtractor._RERANK_PROMPT_CACHE = f.read()
            except Exception:
                PDFExtractor._RERANK_PROMPT_CACHE = (
                    "你是一个顶尖的学术网站解析专家。"
                    "从候选 JSON 中仅选择一个正文 PDF 入口，证据不足返回 null。\n"
                    "候选列表: {json_data}\n"
                    "只返回 JSON: {{\"best_candidate_id\": null, \"confidence\": 0.0, \"reason\": \"...\"}}"
                )
        prompt = PDFExtractor._RERANK_PROMPT_CACHE.format(
            json_data=json.dumps(payload, ensure_ascii=False),
        )
        for client in self._get_llm_clients(mode, provider=provider):
            if not client:
                continue
            try:
                resp = await asyncio.to_thread(
                    client.chat,
                    [{"role": "user", "content": prompt}],
                    model=model_override or None,
                )
                data = self._extract_first_json((resp or {}).get("final_text", ""))
                if not data:
                    continue
                best_candidate_id = data.get("best_candidate_id")
                confidence = float(data.get("confidence", 0.0) or 0.0)
                reason = str(data.get("reason", "") or "")
                if best_candidate_id is None:
                    if confidence >= self.score_config.min_confidence_to_accept:
                        return {
                            "best_candidate_id": None,
                            "confidence": confidence,
                            "reason": reason,
                        }
                    continue
                valid_ids = {item.element_id for item in selected}
                if best_candidate_id in valid_ids and confidence >= self.score_config.min_confidence_to_accept:
                    return {
                        "best_candidate_id": best_candidate_id,
                        "confidence": confidence,
                        "reason": reason,
                    }
            except Exception as e:
                self.logger.debug(f"[PDFExtractor] LLM rerank failed: {e}")
                continue
        return None


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
        self.pdf_extractor = PDFExtractor(logger)
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

            error_indicators_text = ["page not found", "does not exist", "页面不存在", "the page you requested"]
            error_indicators_title = ["404 not found", "page not found", "not found", "页面不存在"]
            if (any(ind in body_text for ind in error_indicators_text) or
                any(ind in title for ind in error_indicators_title)):
                blockers.append(Blocker(BlockerType.NOT_FOUND, solvable=False, details="页面不存在"))

            server_error_indicators_text = ["internal server error", "server error", "service unavailable",
                                            "bad gateway", "服务器错误", "服务不可用"]
            server_error_indicators_title = ["500 internal", "502 bad gateway", "503 service",
                                             "server error", "服务器错误"]
            if (any(ind in body_text for ind in server_error_indicators_text) or
                any(ind in title for ind in server_error_indicators_title)):
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
                function getAllElements(root) {
                    const elements = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
                    let node;
                    while (node = walker.nextNode()) {
                        elements.push(node);
                        if (node.shadowRoot) {
                            elements.push(...getAllElements(node.shadowRoot));
                        }
                    }
                    return elements;
                }
                function getKey(el) {
                    return el.tagName + '|' + (el.href || '') + '|' + (buildElementText(el) || '').slice(0, 50);
                }
                function buildElementText(el) {
                    const chunks = [];
                    const inner = (el.innerText || el.textContent || '').trim();
                    const aria = (el.getAttribute('aria-label') || '').trim();
                    const title = (el.getAttribute('title') || '').trim();
                    if (inner) chunks.push(inner);
                    if (aria) chunks.push(aria);
                    if (title) chunks.push(title);
                    try {
                        const media = el.querySelectorAll('img[alt], svg[alt], [role="img"][aria-label]');
                        media.forEach(m => {
                            const alt = (m.getAttribute('alt') || m.getAttribute('aria-label') || '').trim();
                            if (alt) chunks.push(alt);
                        });
                    } catch (_) {}
                    return chunks.join(' ').replace(/\\s+/g, ' ').trim();
                }
                function isPdfRelated(el) {
                    const text = (buildElementText(el) || '').toLowerCase();
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
                const allNodes = getAllElements(document.body);
                let uidCounter = 0;
                allNodes.forEach(el => {
                    const tag = (el.tagName || '').toLowerCase();
                    const isLink = tag === 'a' && el.hasAttribute('href');
                    const isBtn = tag === 'button' || el.getAttribute('role') === 'button' ||
                                  (tag === 'input' && (el.type === 'button' || el.type === 'submit'));
                    const className = typeof el.className === 'string' ? el.className.toLowerCase() : '';
                    const mergedText = buildElementText(el);
                    const exactText = (mergedText || '').trim().toLowerCase();
                    const isPseudoBtn = (tag === 'div' || tag === 'span' || tag === 'p') && (
                        exactText.includes('download free pdf') ||
                        exactText.includes('download pdf') ||
                        className.includes('ds-work-cover') ||
                        className.includes('ds2-5-button') ||
                        className.includes('toolbarbutton') ||
                        className.includes('js-swp-download-button') ||
                        (exactText.length > 0 && exactText.length < 50 && (className.includes('btn') || className.includes('button') || className.includes('download')))
                    );
                    if (!isLink && !isBtn && !isPseudoBtn) return;
                    if (!isPdfRelated(el)) return;
                    const key = getKey(el);
                    if (seen.has(key)) return;
                    seen.add(key);
                    const uid = 'dl-btn-' + (++uidCounter);
                    el.setAttribute('data-dl-uid', uid);
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    const is_fixed = style && (style.position === 'fixed' || style.position === 'sticky');
                    const docHeight = Math.max(1, document.body.scrollHeight || document.documentElement.scrollHeight);
                    const topDoc = rect.top + window.scrollY;
                    const position_ratio = Math.max(0, Math.min(1, topDoc / docHeight));
                    const in_viewport = rect.top >= 0 && rect.top < window.innerHeight && rect.left >= 0 && rect.left < window.innerWidth;
                    let ancestor_path = [];
                    let node = el.parentElement;
                    for (let d = 0; d < 5 && node; d++) {
                        const id = node.id ? '#' + node.id : '';
                        const cls = (typeof node.className === 'string' && node.className) ? node.className.trim().split(/\\s+/)[0] : '';
                        if (id || cls) ancestor_path.push(id || (cls ? '.' + cls : ''));
                        node = node.parentElement;
                    }
                    const ancestor_path_str = ancestor_path.join(',');
                    let container_role = 'unknown';
                    const ancLower = ancestor_path_str.toLowerCase();
                    if (/header|article-toolbar|article__btn|toolbar|pdf-download|main-pdf/.test(ancLower)) container_role = 'header';
                    else if (/main|content|article-body|article-content/.test(ancLower)) container_role = 'main';
                    else if (/sidebar|aside|secondary|right-col|left-col/.test(ancLower)) container_role = 'sidebar';
                    else if (/footer|bottom|ref-list|reference|bibliography|citation|cite/.test(ancLower)) container_role = 'footer';
                    else if (/reference|bibliography|ref-list|reflist|cited-by/.test(ancLower)) container_role = 'reference';
                    const prevSib = el.previousElementSibling;
                    const nextSib = el.nextElementSibling;
                    const nearby_text_before = (prevSib ? (prevSib.innerText || '').trim() : '').slice(0, 80);
                    const nearby_text_after = (nextSib ? (nextSib.innerText || '').trim() : '').slice(0, 80);
                    results.push({
                        element_id: uid,
                        selector: '[data-dl-uid="' + uid + '"]',
                        tag: tag,
                        text: (mergedText || el.value || '').replace(/\\s+/g, ' ').slice(0, 160).trim(),
                        href: isLink ? (el.href || '') : null,
                        is_visible: rect.width > 0 && rect.height > 0 && el.offsetParent !== null,
                        is_fixed: !!is_fixed,
                        rect_top: Number(rect.top || 0),
                        position_y: Math.round(topDoc),
                        position_ratio: position_ratio,
                        in_viewport: in_viewport,
                        ancestor_path: ancestor_path_str,
                        ancestor_features: ancestor_path_str.toLowerCase().replace(/[,.#]/g, ' '),
                        nearby_text_before: nearby_text_before,
                        nearby_text_after: nearby_text_after,
                        container_role: container_role,
                        attributes: {
                            class: el.className || '',
                            title: el.title || '',
                            download: el.hasAttribute('download') ? 'true' : '',
                            target: isLink ? (el.target || '') : '',
                            'aria-label': el.getAttribute('aria-label') || '',
                            'data-track-action': el.getAttribute('data-track-action') || '',
                            'data-item-name': el.getAttribute('data-item-name') || '',
                            'data-aa-name': el.getAttribute('data-aa-name') || '',
                            'data-testid': el.getAttribute('data-testid') || '',
                            'data-id': el.getAttribute('data-id') || ''
                        }
                    });
                });
                return results.slice(0, 50);
            }''')
            page_url = page.url if hasattr(page, "url") else ""
            try:
                site_source = (urlparse(page_url).netloc or "").strip().lower() or ""
            except Exception:
                site_source = ""
            for raw in raw_elements:
                elem = ActionableElement(
                    selector=raw["selector"],
                    tag=raw["tag"],
                    text=raw["text"],
                    element_id=(raw.get("element_id") or "").strip(),
                    href=raw.get("href"),
                    is_visible=raw["is_visible"],
                    is_fixed=bool(raw.get("is_fixed", False)),
                    rect_top=float(raw.get("rect_top", 0.0) or 0.0),
                    position_y=int(raw.get("position_y", 0)),
                    attributes=raw.get("attributes", {}),
                    position_ratio=raw.get("position_ratio"),
                    in_viewport=bool(raw.get("in_viewport", False)),
                    ancestor_path=(raw.get("ancestor_path") or "").strip(),
                    ancestor_features=(raw.get("ancestor_features") or "").strip().lower(),
                    nearby_text_before=(raw.get("nearby_text_before") or "").strip(),
                    nearby_text_after=(raw.get("nearby_text_after") or "").strip(),
                    container_role=(raw.get("container_role") or "unknown").strip().lower(),
                    site_source=site_source,
                )
                elem.score = self._calculate_score(elem)
                elements.append(elem)
            elements.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            self.logger.debug(f"元素发现出错: {e}")
        return elements

    def _calculate_score(self, elem: ActionableElement) -> float:
        """按 ScoreConfig 的统一规则计算分数。"""
        candidate = self.pdf_extractor.to_candidate(elem)
        score = self.pdf_extractor._calculate_score(candidate)
        nearby = f"{(elem.nearby_text_before or '').lower()} {(elem.nearby_text_after or '').lower()}".strip()
        if nearby:
            if any(kw in nearby for kw in self.pdf_extractor.score_config.negative_text_keywords):
                score -= 20
            if any(kw in nearby for kw in self.pdf_extractor.score_config.positive_text_keywords):
                score += 10
        return score

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
        self._io_lock = threading.RLock()

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
    
    async def _save(self):
        """保存经验库 (异步安全)"""
        temp_path = None
        try:
            temp_path = self.store_path + f".{uuid.uuid4().hex[:8]}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data.copy(), f, ensure_ascii=False, indent=2)

            if sys.platform == 'win32' and os.path.exists(self.store_path):
                os.remove(self.store_path)
            os.rename(temp_path, self.store_path)
            temp_path = None
        except Exception as e:
            self.logger.warning(f"保存经验库失败: {e}")
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    
    async def record_success(self, domain: str, url: str, action_sequence: List[Dict[str, Any]]):
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
        
        await self._save()
        self.logger.debug(f"[经验库] 记录成功: {domain} - {url_pattern}")
    
    async def record_failure(self, domain: str, failed_selectors: List[str]):
        """记录失败的选择器"""
        for selector in failed_selectors:
            if selector:
                if selector not in self.data['selectors']:
                    self.data['selectors'][selector] = {'success': 0, 'fail': 0}
                self.data['selectors'][selector]['fail'] += 1
        await self._save()
    
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
    """LLM辅助：在常规方法失败时分析页面并建议操作。使用项目 src.llm 框架，支持多 provider/模型与统一配置。"""

    _PROMPT_TEMPLATE_CACHE: Optional[str] = None

    def __init__(self, logger: logging.Logger, provider: Optional[str] = None, model: Optional[str] = None,
                 timeout: int = 20):
        self.logger = logger
        self._client = None
        self._model = model
        self._timeout = timeout
        self.enabled = False
        try:
            manager = get_manager()
            # 兼容旧配置：anthropic -> claude
            resolved_provider = (provider or "qwen-thinking").strip()
            if resolved_provider == "anthropic":
                resolved_provider = "claude"
            self._client = manager.get_client(resolved_provider)
            self._model = model
            self.enabled = True
            self.logger.info(f"[LLM] 使用项目 LLM 框架，provider: {resolved_provider}")
        except Exception as e:
            self.logger.info(f"[LLM] 未配置或不可用，LLM辅助功能禁用: {e}")
    
    async def analyze_and_suggest(self, page, analysis: PageAnalysis,
                                   action_history: List[Dict[str, Any]],
                                   llm_provider_override: Optional[str] = None,
                                   model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        调用LLM分析页面并建议下一步操作
        
        Args:
            page: Playwright页面对象
            analysis: PageAnalysis对象
            action_history: 已尝试的操作历史
            llm_provider_override: 本次调用使用的 provider（覆盖实例默认）
            model_override: 本次调用使用的 model（覆盖实例默认）
        
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
            
            # 3. 调用LLM（支持单次 override）
            response = await self._call_llm(
                prompt,
                provider_override=llm_provider_override,
                model_override=model_override,
            )
            
            # 4. 解析响应
            suggestion = self._parse_response(response)
            
            if suggestion:
                self.logger.info(f"[LLM] 建议操作: {suggestion.get('type')} - {suggestion.get('reason', '')[:50]}")
            
            return suggestion
            
        except Exception as e:
            self.logger.warning(f"[LLM] 分析失败: {e}")
            return None
    
    async def _extract_page_summary(self, page) -> Dict[str, Any]:
        """提取页面关键信息（精简版）；穿透 Shadow DOM 收集按钮与链接"""
        try:
            return await page.evaluate('''() => {
                const result = {
                    url: location.href,
                    title: document.title,
                    buttons: [],
                    links: [],
                    text_hints: []
                };
                function getAllElements(root) {
                    const elements = [];
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
                    let node;
                    while (node = walker.nextNode()) {
                        elements.push(node);
                        if (node.shadowRoot) {
                            elements.push(...getAllElements(node.shadowRoot));
                        }
                    }
                    return elements;
                }
                const allNodes = getAllElements(document.body);
                let btnCount = 0, linkCount = 0;
                allNodes.forEach(el => {
                    const tag = (el.tagName || '').toLowerCase();
                    const className = typeof el.className === 'string' ? el.className.toLowerCase() : '';
                    const exactText = (el.innerText || '').trim().toLowerCase();
                    const isBtn = tag === 'button' || el.getAttribute('role') === 'button';
                    const isPseudoBtn = (tag === 'div' || tag === 'span' || tag === 'button') && (
                        exactText.includes('download free pdf') || exactText.includes('download pdf') ||
                        className.includes('ds-work-cover') || className.includes('ds2-5-button') || className.includes('toolbarbutton') ||
                        (exactText.length > 0 && exactText.length < 50 && (className.includes('btn') || className.includes('button') || className.includes('download')))
                    );
                    if (btnCount < 15 && (isBtn || isPseudoBtn)) {
                        const text = (el.innerText || '').trim().slice(0, 50);
                        if (text) {
                            result.buttons.push({
                                text: text,
                                class: (el.className || '').slice(0, 80),
                                visible: el.offsetParent !== null
                            });
                            btnCount++;
                        }
                    }
                    if (linkCount < 20 && tag === 'a' && el.href) {
                        const href = (el.href || '').toLowerCase();
                        const text = (el.innerText || '').toLowerCase();
                        if (href.includes('pdf') || href.includes('download') || text.includes('pdf') || text.includes('download') || text.includes('view') || text.includes('full')) {
                            result.links.push({
                                text: (el.innerText || '').trim().slice(0, 50),
                                href: el.href.slice(0, 150),
                                visible: el.offsetParent !== null
                            });
                            linkCount++;
                        }
                    }
                });
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
        """构建LLM prompt（模板来自 src/prompts/downloader_llm_assist.txt）"""
        if LLMAssistant._PROMPT_TEMPLATE_CACHE is None:
            prompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts", "downloader_llm_assist.txt"
            )
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    LLMAssistant._PROMPT_TEMPLATE_CACHE = f.read()
            except Exception as e:
                self.logger.warning(f"[LLM] 加载 prompt 模板失败，使用内联模板: {e}")
                LLMAssistant._PROMPT_TEMPLATE_CACHE = (
                    "你是一个PDF下载助手。分析以下网页信息，告诉我下一步该怎么做才能下载PDF。\n\n"
                    "## 当前页面\nURL: {url}\n标题: {title}\n\n## 页面元素\n按钮: {buttons_json}\n"
                    "相关链接: {links_json}\n关键词检测: {text_hints}\n\n## 检测到的问题\n{blockers_str}\n\n"
                    "## 已尝试的操作\n{history_str}\n\n请返回JSON格式的建议（只返回JSON，不要其他文字）:\n"
                    '{{"action": {{"type": "click|navigate|give_up", "selector": "...", "url": "...", "reason": "..."}}, "confidence": 0-100}}'
                )
        history_str = ""
        if action_history:
            history_str = "\n".join([
                f"  - {a.get('type', '?')}: {a.get('selector', a.get('text', '?'))[:40]} -> {a.get('result', '?')}"
                for a in action_history[-5:]
            ])
        else:
            history_str = "  (无)"
        blockers_str = ", ".join([b.type.value for b in analysis.blockers]) if analysis.blockers else "无"
        return LLMAssistant._PROMPT_TEMPLATE_CACHE.format(
            url=(page_summary.get("url") or "")[:100],
            title=(page_summary.get("title") or "")[:80],
            buttons_json=json.dumps(page_summary.get("buttons", [])[:10], ensure_ascii=False),
            links_json=json.dumps(page_summary.get("links", [])[:10], ensure_ascii=False),
            text_hints=page_summary.get("text_hints", []),
            blockers_str=blockers_str,
            history_str=history_str,
        )
    
    async def _call_llm(self, prompt: str,
                        provider_override: Optional[str] = None,
                        model_override: Optional[str] = None) -> str:
        """调用项目 LLM 框架（同步 client.chat 放入 asyncio.to_thread 避免阻塞事件循环）。
        单次调用可传入 provider_override / model_override 覆盖实例默认。"""
        client = self._client
        model = self._model
        if provider_override:
            try:
                from src.llm.llm_manager import get_manager
                client = get_manager().get_client(provider_override)
                model = model_override or self._model
            except Exception as e:
                self.logger.debug(f"[LLM] override provider 不可用，使用默认: {e}")
        elif model_override:
            model = model_override
        if not client:
            return ""
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = await asyncio.to_thread(client.chat, messages, model=model)
            return (resp.get("final_text") or "").strip()
        except Exception as e:
            self.logger.debug(f"[LLM] 调用失败: {e}")
            return ""
    
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
            
            # 去除markdown代码块
            if response.startswith('```'):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
                response = response.strip()

            # 优先：从左到右扫描 JSON 起点，逐个尝试解析后缀，兼容前置解释文本
            for idx, ch in enumerate(response):
                if ch != "{":
                    continue
                candidate = response[idx:].strip()
                try:
                    data = json.loads(candidate)
                    action = data.get('action', {})
                    if action.get('type') in ['click', 'navigate', 'give_up']:
                        return action
                except Exception:
                    continue

            # 回退：保留原逆向大括号提取逻辑
            brace_count = 0
            start_idx = -1
            extracted = response
            for i in range(len(response) - 1, -1, -1):
                if response[i] == '}':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif response[i] == '{':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        extracted = response[i:start_idx + 1]
                        break

            data = json.loads(extracted)
            action = data.get('action', {})
            if action.get('type') in ['click', 'navigate', 'give_up']:
                return action
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
                    except Exception:
                        continue
            
            self.logger.debug(f"[LLM] 无法从响应中提取有效JSON，原始响应前200字符: {response[:200]}")
        
        except Exception as e:
            self.logger.debug(f"[LLM] 解析响应失败: {e}")
        
        return None

    _RERANK_PROMPT_CACHE: Optional[str] = None

    async def rerank_candidates(
        self,
        elements: List[ActionableElement],
        top_n: int = 5,
        ultra_lite_provider: Optional[str] = None,
    ) -> Optional[List[ActionableElement]]:
        """用 ultra-lite 对 top-N 候选做 rerank，返回重排后的列表；失败则返回 None 保持启发式顺序。"""
        if not elements or top_n <= 0:
            return None
        try:
            manager = get_manager()
            client = manager.get_ultra_lite_client(ultra_lite_provider)
            if not client:
                return None
        except Exception as e:
            self.logger.debug(f"[LLM] ultra_lite 不可用，跳过 rerank: {e}")
            return None
        subset = elements[:top_n]
        site_source = (getattr(subset[0], "site_source", None) or "") if subset else ""
        candidates = []
        for i, el in enumerate(subset):
            candidates.append({
                "index": i,
                "text": (el.text or "")[:80],
                "href": (el.href or "")[:100],
                "position_ratio": getattr(el, "position_ratio", None),
                "container_role": getattr(el, "container_role", "") or "unknown",
                "nearby_before": (getattr(el, "nearby_text_before", None) or "")[:60],
                "nearby_after": (getattr(el, "nearby_text_after", None) or "")[:60],
            })
        if LLMAssistant._RERANK_PROMPT_CACHE is None:
            prompt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prompts")
            try:
                with open(os.path.join(prompt_dir, "downloader_candidate_rerank.txt"), "r", encoding="utf-8") as f:
                    LLMAssistant._RERANK_PROMPT_CACHE = f.read()
            except Exception as e:
                self.logger.debug(f"[LLM] 加载 rerank 模板失败: {e}")
                LLMAssistant._RERANK_PROMPT_CACHE = (
                    "You are a PDF download helper. Pick the best candidate for main article PDF.\n\n"
                    "Site: {site_source}\nCandidates: {candidates_json}\n\n"
                    "Reply JSON only: {{\"best_index\": 0, \"confidence\": 0-100}}"
                )
        prompt = LLMAssistant._RERANK_PROMPT_CACHE.format(
            site_source=site_source or "unknown",
            candidates_json=json.dumps(candidates, ensure_ascii=False),
        )
        try:
            resp = await asyncio.to_thread(client.chat, [{"role": "user", "content": prompt}], model=None)
            text = (resp.get("final_text") or "").strip()
            if not text:
                return None
            for start in range(len(text)):
                if start < len(text) and text[start] == "{":
                    try:
                        data = json.loads(text[start:])
                        best_index = data.get("best_index", -1)
                        confidence = data.get("confidence", 0)
                        if confidence >= 50 and 0 <= best_index < len(subset):
                            reordered = [subset[best_index]] + [e for i, e in enumerate(subset) if i != best_index]
                            self.logger.info(f"[LLM] rerank 选择 index={best_index}, confidence={confidence}")
                            return reordered
                        break
                    except json.JSONDecodeError:
                        continue
            return None
        except Exception as e:
            self.logger.debug(f"[LLM] rerank 调用失败，使用启发式顺序: {e}")
            return None


@dataclass
class _PaperDownloadContext:
    paper: Dict
    filepath: str
    title: str
    authors: List[str]
    year: Optional[int]
    url: Optional[str]
    pdf_url: Optional[str]
    doi: Optional[str]
    source: str
    is_likely_pdf: bool
    is_ssrn: bool
    is_academia_pdf: bool
    is_semantic_source: bool
    force_pdf_attempt: bool
    annas_md5: Optional[str] = None


@dataclass
class DownloadAttemptTrace:
    events: List[Dict[str, Any]] = field(default_factory=list)
    blocker_history: List[str] = field(default_factory=list)
    navigation_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    download_observations: List[Dict[str, Any]] = field(default_factory=list)
    final_outcome: Optional[str] = None

    def add_event(self, payload: Dict[str, Any]) -> None:
        self.events.append(dict(payload))


class DownloadBudgetController:
    def __init__(
        self,
        *,
        base_deadline_seconds: float,
        challenge_deadline_seconds: float,
        hard_deadline_seconds: float,
    ) -> None:
        self.started_at = time.monotonic()
        self.absolute_deadline_seconds = float(base_deadline_seconds)
        self.challenge_deadline_seconds = float(max(challenge_deadline_seconds, base_deadline_seconds))
        self.hard_deadline_seconds = float(max(hard_deadline_seconds, self.challenge_deadline_seconds))
        self.current_phase = "request_init"
        self.phase_started_at = self.started_at
        self.phase_budget_seconds = float(base_deadline_seconds)
        self.progress_grants: Dict[str, int] = {}

    def start_phase(self, name: str, base_seconds: float) -> None:
        self.current_phase = name
        self.phase_started_at = time.monotonic()
        self.phase_budget_seconds = max(float(base_seconds), 0.1)

    def promote_challenge_window(self) -> None:
        self.absolute_deadline_seconds = min(
            self.hard_deadline_seconds,
            max(self.absolute_deadline_seconds, self.challenge_deadline_seconds),
        )

    def grant_progress(self, reason: str, extra_seconds: float, *, allow_count: Optional[int] = None) -> bool:
        if allow_count is not None and self.progress_grants.get(reason, 0) >= allow_count:
            return False
        self.progress_grants[reason] = self.progress_grants.get(reason, 0) + 1
        extra = max(float(extra_seconds), 0.0)
        self.phase_budget_seconds += extra
        self.absolute_deadline_seconds = min(
            self.hard_deadline_seconds,
            self.absolute_deadline_seconds + extra,
        )
        return True

    def remaining_seconds(self) -> float:
        now = time.monotonic()
        total_elapsed = now - self.started_at
        phase_elapsed = now - self.phase_started_at
        absolute_remaining = self.absolute_deadline_seconds - total_elapsed
        phase_remaining = self.phase_budget_seconds - phase_elapsed
        return max(0.0, min(absolute_remaining, phase_remaining))

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def should_abort(self) -> bool:
        return self.remaining_seconds() <= 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "remaining_seconds": round(self.remaining_seconds(), 2),
            "elapsed_seconds": round(self.elapsed_seconds(), 2),
            "absolute_deadline_seconds": round(self.absolute_deadline_seconds, 2),
            "phase_budget_seconds": round(self.phase_budget_seconds, 2),
            "progress_grants": dict(self.progress_grants),
        }


@dataclass
class DownloadRequestContext:
    paper_id: str
    strategy: str
    title: str = ""
    url: Optional[str] = None
    filepath: Optional[str] = None
    show_browser_override: Optional[bool] = None
    llm_provider_override: Optional[str] = None
    llm_model_override: Optional[str] = None
    assist_llm_mode: Optional[str] = None
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    budget_controller: Optional[DownloadBudgetController] = None
    trace: DownloadAttemptTrace = field(default_factory=DownloadAttemptTrace)
    current_phase: str = "request_init"
    session_slot_index: Optional[int] = None
    task_download_dir: Optional[str] = None
    failed_selectors: List[str] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


_REQUEST_CONTEXT: contextvars.ContextVar[Optional[DownloadRequestContext]] = contextvars.ContextVar(
    "paper_download_request_context",
    default=None,
)


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
                'a[data-test-id="article-pdf-link"]',
                'a[href*="/doi/pdfdirect/"]',
                'a[href*="/doi/epdf/"]',
                'a[href*="/doi/pdf/"]',
                'a.coolBar__ctrl.pdf-download',
                'a[title="ePDF"][class*="coolBar"]',
                'a.coolBar__ctrl[href*="/doi/epdf/"]',
            ],
            'cedar_digital_commons': [
                'a#pdf',
                'a.btn[href*="viewcontent"]',
                'a[href*="viewcontent.cgi"]',
                'a.btn:has-text("Download")',
            ],
            'springer_nature': [
                'a[data-track-action="download pdf"]',
                'a[data-track-action="Pdf download"]',
                'a[data-track="pdf download"]',
                'a[data-test="pdf-link"]',
                'a[href*="/content/pdf/"][href$=".pdf"]',
                'a.c-pdf-download__link[href*="/pdf/"][data-article-pdf="true"]',
                'a.u-button[data-article-pdf="true"][data-test="pdf-link"]',
                'div.c-pdf-download a.c-pdf-download__link[href*="/pdf/"]',
                'a.c-pdf-download__link',
            ],
            'elsevier': [
                '[data-aa-name="srm-pdf-download"]',
                '[data-test-id="pdf-link"]',
                'button#pdfLink',
                'a#pdfLink',
                'a[href*="pdfft"][data-internalname]',
                'a[data-aa-name="btn-download-pdf"]',
                'a[title="Download full text"]',
                'a.link-button[href*="pdfft"]',
                'a.link-button[aria-label*="View PDF"]',
                'a.link-button-primary[href*="pdfft"]',
                'button#pdf-download-button',
            ],
            'frontiersin': [
                'button[data-test-id="download-pdf-btn"]',
                'a[data-action="download-pdf"]',
                'a.download-files-pdf',
            ],
            'ojs_pkp': [  # OJS/PKP系统（europeanjournaloftaxonomy, zootaxa等）
                'a.download',
                '.galley-link.download',
                'a.obj_galley_link',
                'a.obj_galley_link.pdf',
                'a.download span.label',
            ],
            'mdpi': [
                'a.UD_ArticlePDF',
                '.article-pdf-download a',
                'a[href*="/pdf"]:not([href*="flyer"])',
                'a.download-pdf-link',
                'a[title="Download PDF"]',
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
                'a[data-panel-name="article-pdf"]',
                'a[data-title="Download PDF"]',
                'a[href*="/doi/pdf/"]',
                'a.btn--pdf[href*="/doi/pdf/"]',
            ],
            'the_innovation': [
                'a[href^="/data/article/"][href*="/preview/pdf/"]',
                'span.hidden-lg a[href*="/pdf/"]',
            ],
            'ssrn': [
                'a[href*="Delivery.cfm"][href*=".pdf"]',
                'a.abstract-buttons__delivery-button',
                'button:has-text("Download This Paper")',
            ],
            'science': [
                'a[data-item-name="download-pdf"]',
                'a[data-toggle="tooltip"][title*="PDF"]',
                'a[href*="/doi/pdf/"]',
                'a.article-tools__pdf',
                'a.btn[aria-label="PDF"]',
            ],
            'asm': [
                'a.btn--pdf',
                'a[title="Open full-text in eReader"]',
                'a[data-panel-name="article-pdf"]',
                'a[data-item-name="download-pdf"]',
                'a[data-toggle="tooltip"][title*="PDF"]',
                'a[href*="/doi/pdf/"]',
                'a[href*="/doi/epdf/"]',
                'a[href*="/doi/reader/"]',
            ],
            'acs': [
                'a[data-id="article_header_OpenPDF"]',
                'a.article__btn__secondary--pdf[href*="/doi/pdf/"]',
                'a[href*="/doi/pdf/"]',
            ],
            'researchgate': [
                'a[data-testid="pdf-download-button"]',
                'button:has-text("Download full-text PDF")',
                'a.nova-legacy-c-button[href*="fulltext"]',
                'a:has-text("Download full-text PDF")',
                'a:has-text("Download file PDF")',
                '.js-target-download-btn',
            ],
            'arxiv': [
                'a.download-pdf',
                'a[href^="/pdf/"]',
            ],
            'biorxiv_medrxiv': [
                'a.article-dl-pdf-link',
                'a.article-dl-pdf-link-main',
                'a[href$=".full.pdf"]',
            ],
            'chemrxiv': [
                'button:has-text("Download PDF")',
                'a[href*="/api/articles/"][href*="/pdf"]',
            ],
            'research_square': [
                'a[data-test="download-pdf-button"]',
                'a.download-pdf',
            ],
            'osf_preprints': [
                'a:has-text("Download preprint")',
                'a.btn-primary:has-text("Download")',
            ],
            'pmc': [
                'a.int-view[href*="/pdf/"]',
                'a[data-action="pdf_link"]',
                'a.article-pdf',
                'a[data-panel-name="article-pdf"]',
                'a.article-dl-pdf-link-ext',
            ],
            'academia': [
                'button.ds2-5-button:has-text("Download PDF")',
                'button[class*="ToolbarButton"]:has-text("Download PDF")',
                'button.ds2-5-button',
                'a.js-swp-download-button',
                'div.ds-work-cover--hover-container',
                'div.js-swp-download-button',
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
        'atypon_publishers': {
            'detect': lambda url: any(d in url.lower() for d in ['science.org', 'pnas.org', 'tandfonline.com', 'journals.asm.org']),
            'steps': [
                {'selector': 'a.btn--pdf, a[title="Open full-text in eReader"], a[data-item-name="download-pdf"], a[data-panel-name="article-pdf"], a[data-toggle="tooltip"][title*="PDF"]', 'wait': 2},
                {'selector': 'a.btn-primary:has-text("Proceed"), button:has-text("Accept and Download"), form#pdf-terms input[type="submit"]', 'wait': 2},
            ]
        },
        'researchgate': {
            'detect': lambda url: 'researchgate.net' in url.lower(),
            'steps': [
                {'selector': 'a[data-testid="pdf-download-button"], a:has-text("Download full-text PDF"), button:has-text("Download full-text PDF"), a.nova-legacy-c-button[href*="download"]', 'wait': 2},
                {'selector': 'button:has-text("Download without"), span:has-text("Download without"), button:has-text("Continue to download")', 'wait': 2},
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
        'asm': ['journals.asm.org'],
        'acs': ['pubs.acs.org'],
        'researchgate': ['researchgate.net'],
        'arxiv': ['arxiv.org'],
        'biorxiv_medrxiv': ['biorxiv.org', 'medrxiv.org'],
        'chemrxiv': ['chemrxiv.org'],
        'research_square': ['researchsquare.com'],
        'osf_preprints': ['osf.io'],
        'pmc': ['ncbi.nlm.nih.gov/pmc', 'europepmc.org'],
        'academia': ['academia.edu'],
    }
    
    def __init__(self, download_dir: str = "papers", max_concurrent: int = 5, show_browser: bool = False,
                 persist_browser: bool = False, download_timeout: int = 200, timeout: int = 10, max_retries: int = 3,
                 browser_type: str = "chrome", stealth_mode: bool = True, config_path: str = "config.json",
                 initial_config: Optional[Dict[str, Any]] = None, logger: logging.Logger = logger):
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
            config_path: 配置文件路径（当 initial_config 为 None 时使用）
            initial_config: 可选配置字典，与当前项目统一时由 adapter 传入，优先于 config_path
            logger: 日志记录器
        """
        # 设置日志记录器
        self.logger = logger

        # 配置：优先使用 initial_config（与 RAG 项目统一），否则从 config_path 加载
        if initial_config is not None:
            self.config = dict(initial_config)
            self._config_dir = None  # adapter 传入时相对路径用 download_dir
        else:
            self.config = load_config(config_path)
            # 使用项目 config 时，相对路径以 config 目录为基准
            self._config_dir = _project_config_root() if (config_path is None or config_path == "config.json") else None
        selector_cfg = self._load_selector_config()
        self.pdf_selectors = selector_cfg.get("pdf_selectors", self.PDF_SELECTORS)
        self.site_url_patterns = selector_cfg.get("site_url_patterns", self.SITE_URL_PATTERNS)
        self.multi_step_sites = self._build_multi_step_sites(
            selector_cfg.get("multi_step_sites", {})
        )
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
            "paper_total_timeout": 150,
            "paper_challenge_timeout": 210,
            "paper_hard_timeout": 240,
            "download_growth_grant_limit": 2,
        }
        self.timeouts = default_timeouts
        self.timeouts.update(self.config.get("downloader", {}).get("timeouts", {}))
        api_keys = self.config.get("api_keys", {})
        self.apikey = api_keys.get("twocaptcha", "")
        self._capsolver_api_key = api_keys.get("capsolver", "")
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
        dl_cfg = self.config.get("downloader") or {}
        self._proxy = dl_cfg.get("proxy") or None
        if isinstance(self._proxy, str):
            self._proxy = self._proxy.strip() or None
        _ext = dl_cfg.get("capsolver_extension_path") or None
        self._capsolver_extension_path = (_ext or "").strip() or None

        # Build the unified captcha solver shared across all sessions
        _cap_timeout = self.timeouts.get("captcha_timeout", 120)
        self._captcha_solver = CaptchaSolver(
            capsolver_api_key=self._capsolver_api_key,
            twocaptcha_api_key=self.apikey,
            timeout_seconds=_cap_timeout,
        )

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
            if not any(
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", "") == debug_log_path
                for h in self.logger.handlers
            ):
                self.logger.addHandler(file_handler)
            self.debug_log_file = debug_log_path
            self.logger.debug(f"=== Debug 模式已启用，日志文件: {debug_log_path} ===")
        
        # 持久化浏览器相关属性
        self._persistent_user_data_dir = os.path.join(os.path.dirname(self.download_dir), ".browser_data")
        self._browser_manager = None  # 浏览器管理器
        self._context_lock = asyncio.Lock()  # 串行化会话池 init/close，避免重复创建共享 context
        self._active_pages = set()  # 跟踪所有活动的页面
        
        # 会话池相关属性（单上下文多标签：共享 cookie/登录态）
        self._shared_browser_context = None  # 唯一的全局浏览器上下文，所有 slot 共享
        self._session_pool_size = max_concurrent  # 会话池大小与并发数保持一致
        self._session_pool: asyncio.Queue = asyncio.Queue()  # 会话队列
        self._session_pool_data: List[Dict] = []  # 会话元数据 [{context, user_data_dir, index, last_used}]
        self._session_pool_initialized = False  # 会话池是否已初始化
        self._pool_was_headless: Optional[bool] = None  # 会话池当前使用的 headless（用于检测 override 变更后重建）
        self._show_browser_override: Optional[bool] = None  # 运行时覆盖：有头/无头（由前端或 API 设置）
        self._warmed_domains: Set[str] = set()  # 兼容旧字段名：当前会话中“已可复用的平台 key”集合
        self._http_session: Optional[aiohttp.ClientSession] = None
        # 防并发惊群：按平台 key 排队，仅第一个任务执行预热，其余等待后直接放行
        self._domain_locks: Dict[str, asyncio.Lock] = {}  # 兼容旧字段名：实际按平台 key 使用
        self._domain_locks_lock = asyncio.Lock()

        # 动态资源管理（优化：空闲超时自动释放）
        self._session_idle_timeout = 60  # 会话空闲超时（秒），空闲超过此时间后关闭
        self._cleanup_task: Optional[asyncio.Task] = None  # 清理任务
        self._last_activity_time = time.time()  # 最后活动时间
        # Resident context pool：当 shared_browser 池可用且 headed 时，从池借还 context
        self._use_context_pool: bool = False
        self._pool_sem: Optional[asyncio.Semaphore] = None
        
        # 下载历史
        self.history_file = os.path.join(self.download_dir, ".download_history.json")
        self.download_history = self.load_download_history()

        # 并发写安全：历史/MD5/经验库 JSON 写入锁
        self._io_lock = threading.RLock()
        
        # MD5 索引：用于检测重复文件（相同内容的 PDF）
        self.file_md5_index = {}  # MD5 -> filepath 映射
        self._load_md5_index()
        
        # Initialize the browser manager to use its functionalities (initialization, stealth mode, human behavior simulation)
        self.browser_manager = BrowserManager()
        
        # 页面分析器（阶段1：仅用于诊断）
        self.page_analyzer = PageAnalyzer(self.logger)
        self.pdf_extractor = self.page_analyzer.pdf_extractor

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
                # 相对路径：项目 config 目录、或 config_path 所在目录、或 download_dir
                if getattr(self, "_config_dir", None) and os.path.isdir(self._config_dir):
                    config_dir = self._config_dir
                elif config_path and os.path.exists(config_path):
                    config_dir = os.path.dirname(os.path.abspath(config_path))
                else:
                    config_dir = self.download_dir
                self.experience_store_path = os.path.join(config_dir, configured_path)
            self.logger.info(f"[经验库] 使用自定义路径: {self.experience_store_path}")
        else:
            # 默认路径：下载目录下的隐藏文件
            self.experience_store_path = os.path.join(self.download_dir, ".experience_store.json")
        self.experience_store = ExperienceStore(self.experience_store_path, self.logger)

        # LLM辅助（阶段3，可选）— 使用项目 src.llm 框架，provider/model 从 config 传入，API key 由框架统一加载
        llm_cfg = self.config.get("llm", {})
        self.llm_assistant = LLMAssistant(
            self.logger,
            provider=llm_cfg.get("provider"),
            model=llm_cfg.get("model"),
            timeout=self.timeouts.get("llm_timeout", 20),
        )

        # 其他初始化代码...

    def set_show_browser_override(self, value: Optional[bool]) -> None:
        """运行时覆盖：有头(True)/无头(False)。由 API 请求传入，影响下次会话池创建或重建。"""
        old = self._show_browser_override
        self._show_browser_override = value
        self.logger.info(
            "[headed-diag] set_show_browser_override: old=%r -> new=%r",
            old,
            value,
        )

    def get_pool_headless(self) -> bool:
        """会话池/共享 context 应使用的 headless。

        这里只看实例级 override，不读取 request context，避免单次请求把共享
        context 误判成“模式变更”而触发不必要的重建。
        """
        if self._show_browser_override is not None:
            result = not self._show_browser_override
            self.logger.info(
                "[headed-diag] get_pool_headless: from override _show_browser_override=%r -> headless=%s",
                self._show_browser_override,
                result,
            )
            return result
        self.logger.info(
            "[headed-diag] get_pool_headless: from config self.headless=%s",
            self.headless,
        )
        return self.headless

    def get_effective_headless(self) -> bool:
        """当前请求应使用的 headless。

        请求级 request context 可以覆盖实例级默认值，但该值不应用于共享
        context / session pool 的重建判定。
        """
        request_context = self._get_request_context()
        if request_context and request_context.show_browser_override is not None:
            return not request_context.show_browser_override
        if self._show_browser_override is not None:
            return not self._show_browser_override
        return self.headless

    def _get_request_budget_settings(self) -> Dict[str, float]:
        base = float(self.timeouts.get("paper_total_timeout", 150))
        challenge = float(self.timeouts.get("paper_challenge_timeout", max(base + 60, 210)))
        hard = float(self.timeouts.get("paper_hard_timeout", max(challenge + 30, 240)))
        return {
            "base": base,
            "challenge": max(challenge, base),
            "hard": max(hard, challenge),
        }

    def _get_request_context(self) -> Optional[DownloadRequestContext]:
        return _REQUEST_CONTEXT.get()

    def _build_request_context(
        self,
        *,
        paper_id: Optional[str],
        strategy: str,
        title: str = "",
        url: Optional[str] = None,
        filepath: Optional[str] = None,
        show_browser: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
        assist_llm_mode: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> DownloadRequestContext:
        limits = self._get_request_budget_settings()
        budget = DownloadBudgetController(
            base_deadline_seconds=limits["base"],
            challenge_deadline_seconds=limits["challenge"],
            hard_deadline_seconds=limits["hard"],
        )
        return DownloadRequestContext(
            paper_id=paper_id or os.path.splitext(os.path.basename(filepath or ""))[0] or uuid.uuid4().hex[:12],
            strategy=strategy,
            title=title,
            url=url,
            filepath=filepath,
            show_browser_override=show_browser,
            llm_provider_override=llm_provider,
            llm_model_override=model_override,
            assist_llm_mode=assist_llm_mode,
            progress_callback=progress_callback,
            budget_controller=budget,
            metadata={"assist_llm_mode": (assist_llm_mode or "ultra-lite")},
        )

    def _emit_progress_event(
        self,
        stage: str,
        *,
        request_context: Optional[DownloadRequestContext] = None,
        phase: Optional[str] = None,
        **extra: Any,
    ) -> None:
        ctx = request_context or self._get_request_context()
        payload: Dict[str, Any] = {"stage": stage}
        if ctx:
            payload["paper_id"] = ctx.paper_id
            payload["strategy"] = ctx.strategy
            active_phase = phase or ctx.current_phase
            if active_phase:
                payload["phase"] = active_phase
            if ctx.budget_controller is not None:
                payload["remaining_budget"] = round(ctx.budget_controller.remaining_seconds(), 2)
                payload["elapsed"] = round(ctx.budget_controller.elapsed_seconds(), 2)
        elif phase:
            payload["phase"] = phase
        payload.update(extra)
        if ctx:
            ctx.trace.add_event(payload)
        callback = ctx.progress_callback if ctx else None
        if callback:
            try:
                callback(dict(payload))
            except Exception as cb_error:
                self.logger.debug(f"progress_callback 调用失败: {cb_error}")

    def _start_phase(
        self,
        phase: str,
        *,
        base_seconds: float,
        request_context: Optional[DownloadRequestContext] = None,
        **extra: Any,
    ) -> float:
        ctx = request_context or self._get_request_context()
        if ctx and ctx.budget_controller is not None:
            ctx.current_phase = phase
            ctx.budget_controller.start_phase(phase, base_seconds)
            self._emit_progress_event(
                "phase_started",
                request_context=ctx,
                phase=phase,
                phase_budget=round(base_seconds, 2),
                **extra,
            )
            return ctx.budget_controller.remaining_seconds()
        return max(float(base_seconds), 0.1)

    def _grant_progress(
        self,
        reason: str,
        *,
        extra_seconds: float,
        request_context: Optional[DownloadRequestContext] = None,
        allow_count: Optional[int] = None,
        **extra: Any,
    ) -> bool:
        ctx = request_context or self._get_request_context()
        if not ctx or ctx.budget_controller is None:
            return False
        granted = ctx.budget_controller.grant_progress(reason, extra_seconds, allow_count=allow_count)
        if granted:
            self._emit_progress_event(
                "progress_granted",
                request_context=ctx,
                reason=reason,
                extra_seconds=round(extra_seconds, 2),
                **extra,
            )
        return granted

    def _classify_failure(
        self,
        *,
        blocker: Optional[Blocker] = None,
        response_status: Optional[int] = None,
        exception: Optional[BaseException] = None,
        invalid_artifact: bool = False,
        download_stalled: bool = False,
    ) -> str:
        if invalid_artifact:
            return "permanent_failure"
        if blocker is not None:
            if blocker.type in (BlockerType.CLOUDFLARE, BlockerType.CAPTCHA):
                return "anti_bot_challenge"
            if blocker.type in (BlockerType.RATE_LIMITED,):
                return "soft_block"
            if blocker.type in (BlockerType.PAYWALL, BlockerType.LOGIN_REQUIRED, BlockerType.GEO_BLOCKED, BlockerType.NOT_FOUND):
                return "permanent_failure"
        if response_status in (403, 429):
            return "soft_block"
        if response_status == 404:
            return "permanent_failure"
        if response_status is not None and response_status >= 500:
            return "transient_network"
        if download_stalled:
            return "download_stalled"
        if isinstance(exception, (asyncio.TimeoutError, PlaywrightTimeoutError, aiohttp.ClientError)):
            return "transient_network"
        return "selector_or_action_miss"

    def _record_failure(
        self,
        failure_class: str,
        *,
        request_context: Optional[DownloadRequestContext] = None,
        **extra: Any,
    ) -> int:
        ctx = request_context or self._get_request_context()
        if not ctx:
            return 0
        new_count = ctx.failure_counts.get(failure_class, 0) + 1
        ctx.failure_counts[failure_class] = new_count
        self._emit_progress_event(
            "failure_classified",
            request_context=ctx,
            failure_class=failure_class,
            failure_count=new_count,
            **extra,
        )
        return new_count

    def _retry_policy_for_failure(self, failure_class: str) -> Dict[str, Any]:
        policies = {
            "transient_network": {"max_attempts": 2, "backoff_seconds": 2.0},
            "anti_bot_challenge": {"max_attempts": 2, "backoff_seconds": self.timeouts.get("page_stable_wait", 2)},
            "soft_block": {"max_attempts": 2, "backoff_seconds": self.timeouts.get("rate_limit_wait", 30)},
            "selector_or_action_miss": {"max_attempts": 1, "backoff_seconds": 0.0},
            "download_stalled": {"max_attempts": 1, "backoff_seconds": 0.0},
            "permanent_failure": {"max_attempts": 1, "backoff_seconds": 0.0},
        }
        return policies.get(failure_class, {"max_attempts": 1, "backoff_seconds": 0.0})

    def _should_retry_failure(self, failure_class: str, attempt: int) -> Tuple[bool, float]:
        policy = self._retry_policy_for_failure(failure_class)
        return attempt < policy["max_attempts"], float(policy["backoff_seconds"])

    def _get_current_llm_override(self) -> Tuple[Optional[str], Optional[str]]:
        request_context = self._get_request_context()
        if request_context:
            return request_context.llm_provider_override, request_context.llm_model_override
        return (None, None)

# 拦截script，用于获取 Cloudflare JS Challenge 参数，包括非常关键的sitekey，后续将用于解决验证码
    # 保留原生 render 调用，防止页面卡死；参数挂载到全局变量供 Python 轮询（比 console.log 稳定）
    intercept_script = """
        const initTurnstileInterception = setInterval(()=>{
            if (window.turnstile) {
                clearInterval(initTurnstileInterception);
                const originalRender = window.turnstile.render;
                window.turnstile.render = function(container, options) {
                    window.__cf_intercepted_params = {
                        sitekey: options.sitekey,
                        pageurl: window.location.href,
                        data: options.cData,
                        pagedata: options.chlPageData,
                        action: options.action,
                        userAgent: navigator.userAgent,
                    };
                    if (options.callback) {
                        window.__cf_callback = options.callback;
                    }
                    return originalRender.apply(this, arguments);
                };
            }
        }, 50);
    """

    # ==================== 选择器辅助方法 ====================
    def _default_multi_step_site_config(self) -> Dict[str, Dict[str, Any]]:
        return {
            "wiley": {
                "detect_contains": ["onlinelibrary.wiley.com"],
                "steps": [
                    {"selector": 'div.coolBar__section.coolBar--download.PdfLink a.coolBar__ctrl.pdf-download, a.coolBar__ctrl.pdf-download, a[title="ePDF"].coolBar__ctrl, a:has-text("Download PDF")', "wait": 2},
                    {"selector": 'a[href*="pdfdirect"][href*="download=true"], a[href*="pdfdirect"], a[aria-label*="Download"]', "wait": 2},
                ],
            },
            "elife": {
                "detect_contains": ["elife"],
                "steps": [
                    {"selector": 'a.button.button--default.button--action.icon.icon-download#button-action-download', "wait": 1},
                    {"selector": 'a.article-download-links-list__link:has-text("Article PDF")', "wait": 1},
                ],
            },
            "zootaxa": {
                "detect_contains": ["mapress.com"],
                "steps": [
                    {"selector": "a.obj_galley_link.pdf", "wait": 1},
                    {"selector": "a.download[download]", "wait": 1},
                ],
            },
            "atypon_publishers": {
                "detect_contains": ["science.org", "pnas.org", "tandfonline.com", "journals.asm.org"],
                "steps": [
                    {"selector": 'a.btn--pdf, a[title="Open full-text in eReader"], a[data-item-name="download-pdf"], a[data-panel-name="article-pdf"], a[data-toggle="tooltip"][title*="PDF"]', "wait": 2},
                    {"selector": 'a.btn-primary:has-text("Proceed"), button:has-text("Accept and Download"), form#pdf-terms input[type="submit"]', "wait": 2},
                ],
            },
            "researchgate": {
                "detect_contains": ["researchgate.net"],
                "steps": [
                    {"selector": 'a[data-testid="pdf-download-button"], a:has-text("Download full-text PDF"), button:has-text("Download full-text PDF"), a.nova-legacy-c-button[href*="download"]', "wait": 2},
                    {"selector": 'button:has-text("Download without"), span:has-text("Download without"), button:has-text("Continue to download")', "wait": 2},
                ],
            },
        }

    def _build_multi_step_sites(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        built: Dict[str, Dict[str, Any]] = {}
        for site_id, site_cfg in (config or {}).items():
            detect_contains = [
                str(item).lower()
                for item in (site_cfg.get("detect_contains") or [])
                if str(item).strip()
            ]
            steps = site_cfg.get("steps") or []
            if not steps:
                continue

            def _make_detect(patterns: List[str]):
                return lambda u: any(p in (u or "").lower() for p in patterns)

            detect = _make_detect(detect_contains)
            built[site_id] = {"detect": detect, "steps": steps}

        return built or self.MULTI_STEP_SITES

    def _load_selector_config(self) -> Dict[str, Any]:
        defaults = {
            "pdf_selectors": self.PDF_SELECTORS,
            "site_url_patterns": self.SITE_URL_PATTERNS,
            "multi_step_sites": self._default_multi_step_site_config(),
        }
        config_path = os.path.join(_project_config_root(), "pdf_selectors.json")
        if not os.path.exists(config_path):
            return defaults
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                self.logger.warning("pdf_selectors.json 格式错误（非对象），将使用内置默认配置")
                return defaults
            return _deep_merge(defaults, loaded)
        except Exception as e:
            self.logger.warning("加载 pdf_selectors.json 失败，使用内置默认配置: %s", e)
            return defaults
    
    def _match_site(self, url: str, site_key: str) -> bool:
        """检查URL是否匹配指定网站
        
        Args:
            url: 待检查的URL
            site_key: 网站标识键（如'wiley', 'springer_nature'等）
            
        Returns:
            bool: 是否匹配
        """
        if site_key not in self.site_url_patterns:
            return False
        
        url_lower = url.lower()
        patterns = self.site_url_patterns[site_key]
        return any(pattern in url_lower for pattern in patterns)
    
    def _get_site_specific_selectors(self, url: str) -> list:
        """根据URL返回网站特定的选择器列表
        
        Args:
            url: 当前页面URL
            
        Returns:
            list: 网站特定的选择器列表
        """
        for site_key, selectors in self.pdf_selectors['site_specific'].items():
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
        selectors.extend(self.pdf_selectors['generic']['direct_links'])
        selectors.extend(self.pdf_selectors['generic']['buttons'])
        selectors.extend(self.pdf_selectors['generic']['links'])
        selectors.extend(self.pdf_selectors['generic']['text_matchers'])
        
        return selectors

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """从 DOI 或出版商文章 URL 中提取 DOI。"""
        if not url:
            return None

        parsed = urlparse(url)
        path = unquote(parsed.path or "")
        match = re.search(r'(10\.\d{4,9}/[^?#]+)', path, flags=re.IGNORECASE)
        if not match:
            return None

        doi = match.group(1).rstrip("/")
        return doi or None

    def _is_doi_like_url(self, url: str) -> bool:
        """判断 URL/字符串是否可视为 DOI 入口。"""
        if not url:
            return False

        candidate = str(url).strip()
        if not candidate:
            return False

        lowered = candidate.lower()
        if lowered.startswith("10.") or "doi.org/" in lowered:
            return True

        return self._extract_doi_from_url(candidate) is not None

    def _extract_host_from_url(self, url: Optional[str]) -> str:
        """提取并标准化 URL host（不含端口）。"""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or parsed.netloc or "").strip().lower()
            if ":" in host:
                host = host.split(":", 1)[0]
            return host
        except Exception:
            return ""

    def _normalize_platform_key(self, host: str) -> str:
        """将 host 归一为平台 key（别名合并 + 去 www）。"""
        normalized = (host or "").strip().lower()
        if not normalized:
            return ""
        if normalized.startswith("www."):
            normalized = normalized[4:]

        aliases = {
            "dx.doi.org": "doi.org",
            "doi.org.cn": "doi.org",
        }
        return aliases.get(normalized, normalized)

    def _is_low_value_warmup_key(self, platform_key: str) -> bool:
        """低价值预热 key：仅作为跳转入口，不值得单独预热。"""
        return platform_key in {"doi.org"}

    def _resolve_platform_key(self, url: Optional[str], page_url: Optional[str] = None) -> str:
        """按“最终站点优先”解析平台 key，避免将 doi.org 误当目标平台。"""
        url_key = self._normalize_platform_key(self._extract_host_from_url(url))
        page_key = self._normalize_platform_key(self._extract_host_from_url(page_url))

        if page_key and not self._is_low_value_warmup_key(page_key):
            return page_key
        if url_key and not self._is_low_value_warmup_key(url_key):
            return url_key
        return page_key or url_key

    def _is_platform_ready(self, platform_key: str) -> bool:
        return bool(platform_key) and platform_key in self._warmed_domains

    def _mark_platform_ready(self, platform_key: str, reason: str = "") -> None:
        if not platform_key:
            return
        if platform_key in self._warmed_domains:
            return
        self._warmed_domains.add(platform_key)
        if reason:
            self.logger.info("平台 %s 已标记为可复用（%s）", platform_key, reason)
        else:
            self.logger.info("平台 %s 已标记为可复用", platform_key)

    async def _get_platform_lock(self, platform_key: str) -> asyncio.Lock:
        async with self._domain_locks_lock:
            if platform_key not in self._domain_locks:
                self._domain_locks[platform_key] = asyncio.Lock()
            return self._domain_locks[platform_key]

    async def _warmup_platform_once(
        self,
        page,
        *,
        request_url: str,
        task_download_dir: Optional[str] = None,
        request_context: Optional["DownloadRequestContext"] = None,
        consume_download_event: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """按平台 key 预热一次；已可复用平台直接跳过。"""
        platform_key = self._resolve_platform_key(request_url)
        if not platform_key:
            return
        if self._is_low_value_warmup_key(platform_key):
            self.logger.info("入口 %s 为跳转域，跳过预热，等待进入目标平台后再处理", platform_key)
            return
        if self._is_platform_ready(platform_key):
            self.logger.info("平台 %s 已可复用，跳过预热", platform_key)
            return

        self._start_phase(
            "page_warmup",
            base_seconds=self.timeouts.get("goto_timeout", 20),
            request_context=request_context,
            domain=platform_key,
        )

        platform_lock = await self._get_platform_lock(platform_key)
        async with platform_lock:
            if self._is_platform_ready(platform_key):
                self.logger.info("平台 %s 已由并发任务完成预热，直接放行", platform_key)
                return

            parsed = urlparse(request_url)
            scheme = parsed.scheme or "https"
            base_url = f"{scheme}://{parsed.netloc}"
            self.logger.info("首入平台，执行全局探路预热：%s (key=%s)", base_url, platform_key)
            try:
                warmup_response = await page.goto(
                    base_url,
                    wait_until="domcontentloaded",
                    timeout=self.timeouts.get("goto_timeout", 20) * 1000,
                )
                self.logger.debug(
                    "[DEBUG] 预热完成: status=%s, url=%s",
                    warmup_response.status if warmup_response else None,
                    page.url,
                )
                await self._handle_cookie_consent(page, context_label="预热")
                if consume_download_event is not None:
                    await consume_download_event("warmup")
                cf_success, _ = await self.solve_cloudflare_if_needed(
                    page,
                    task_download_dir=task_download_dir,
                    request_context=request_context,
                )
                still_blocked = False
                if cf_success:
                    try:
                        still_blocked = await self.is_cloudflare_verifying(page, detection_duration=0.5)
                    except Exception:
                        still_blocked = False
                if cf_success and not still_blocked:
                    self._mark_platform_ready(platform_key, reason="warmup")
                    self._grant_progress(
                        "warmup_completed",
                        extra_seconds=8,
                        request_context=request_context,
                        allow_count=1,
                        domain=platform_key,
                    )
                    self._emit_progress_event(
                        "warmup_completed",
                        request_context=request_context,
                        domain=platform_key,
                    )
                    self.logger.info("预热完成，开始访问目标 URL")
                else:
                    self.logger.warning("平台 %s 预热未完全通过验证，本次不标记可复用", platform_key)
            except Exception as e:
                self.logger.warning("平台 %s 预热访问失败（不影响后续流程）: %s", platform_key, e)

    def _infer_source_flags(
        self,
        source: Optional[str],
        url: Optional[str] = None,
        pdf_url: Optional[str] = None,
    ) -> Dict[str, bool]:
        """基于 source + URL 联合推断来源特征，避免完全依赖上游 source 标签。"""
        source_lower = (source or "").strip().lower()
        url_candidates = [part.lower() for part in [url or "", pdf_url or ""] if part]
        url_blob = " ".join(url_candidates)

        return {
            "is_semantic_source": (
                "semantic scholar" in source_lower or "semanticscholar.org" in url_blob
            ),
            "is_ssrn_source": (
                "ssrn" in source_lower or any(self._is_ssrn_url(candidate) for candidate in url_candidates)
            ),
            "is_academia_source": (
                "academia" in source_lower or "academia.edu" in url_blob
            ),
            "is_wiley_source": (
                "wiley" in source_lower or "onlinelibrary.wiley.com" in url_blob
            ),
            "is_sciencedirect_source": (
                "sciencedirect" in source_lower
                or "elsevier" in source_lower
                or "sciencedirect.com" in url_blob
            ),
            "is_asm_source": (
                "asm" in source_lower or "journals.asm.org" in url_blob
            ),
        }

    def _build_preferred_pdf_entrypoints(self, url: str) -> List[str]:
        """为已知 DOI 站点构造优先尝试的正文 PDF 入口。"""
        if not url:
            return []

        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        scheme = parsed.scheme or "https"
        doi = self._extract_doi_from_url(url)
        if not doi:
            return []

        encoded_doi = quote(doi, safe="/")

        if host.endswith("journals.asm.org"):
            return [f"{scheme}://{parsed.netloc}/doi/pdf/{encoded_doi}"]

        if host.endswith("onlinelibrary.wiley.com"):
            return [
                f"{scheme}://{parsed.netloc}/doi/pdfdirect/{encoded_doi}?download=true",
                f"{scheme}://{parsed.netloc}/doi/pdf/{encoded_doi}",
                f"{scheme}://{parsed.netloc}/doi/epdf/{encoded_doi}",
            ]

        if host.endswith("pubs.acs.org"):
            return [f"{scheme}://{parsed.netloc}/doi/pdf/{encoded_doi}"]

        return []

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
                    except Exception:
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
                            await self._handle_wiley_post_click(page, task_download_dir=task_download_dir)
                            return True
                    else:
                        self.logger.debug(f"元素 {selector} 不可见，尝试强制点击")
                        await btn.click(force=True)
                        self.logger.info(f"强制点击: {selector}")
                        await asyncio.sleep(2)
                        await self._handle_wiley_post_click(page, task_download_dir=task_download_dir)
                        return True

                except Exception as e:
                    continue
            
            # ===== 方法2：检测弹窗/模态框中的按钮 =====
            modal_selectors = self.pdf_selectors['modal_buttons']
            
            for selector in modal_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        self.logger.info(f"在模态框/弹窗中找到并点击了按钮: {selector}")
                        await asyncio.sleep(1)
                        return True
                except Exception:
                    continue
            
            # ===== 方法3：滚动页面并检测中间的打开按钮 =====
            # 有些网站会在 PDF 区域显示一个"点击打开"的覆盖层
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(0.5)
            
            center_button_selectors = self.pdf_selectors['center_buttons']
            
            for selector in center_button_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        self.logger.info(f"在页面中央找到并点击了按钮: {selector}")
                        await asyncio.sleep(1)
                        return True
                except Exception:
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
            dropdown_triggers = self.pdf_selectors['dropdown_triggers']
            
            for trigger in dropdown_triggers:
                try:
                    btn = page.locator(trigger).first
                    if await btn.is_visible(timeout=1000):
                        self.logger.info(f"找到下拉菜单触发器: {trigger}")
                        await btn.click()
                        await asyncio.sleep(0.5)  # 等待下拉菜单展开
                        
                        # 在下拉菜单中查找PDF链接
                        pdf_link_selectors = self.pdf_selectors['dropdown_pdf_links']
                        
                        for pdf_selector in pdf_link_selectors:
                            try:
                                pdf_link = page.locator(pdf_selector).first
                                if await pdf_link.is_visible(timeout=1000):
                                    href = await pdf_link.get_attribute("href")
                                    if href:
                                        try:
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
                            except Exception:
                                continue
                except Exception:
                    continue
            
            self.logger.debug("未找到任何可点击的下载/打开按钮")
            return False
         
        except Exception as e:
            self.logger.error(f"搜索下载按钮时发生错误: {str(e)}")
            return False

    async def _handle_wiley_post_click(self, page, task_download_dir: Optional[str] = None):
        """Wiley 专属：点击 PDF 按钮后处理 Cloudflare 验证并等待下载按钮"""
        if 'onlinelibrary.wiley.com' not in page.url:
            return
        if '/doi/pdf/' not in page.url and '/doi/epdf/' not in page.url:
            return
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

# ==================== Generic captcha type detection ====================

    async def detect_captcha_type(self, page) -> CaptchaType:
        """
        Inspect the current page and return the most specific CaptchaType.

        Checks are ordered roughly by specificity so the first match wins.
        Returns CaptchaType.UNKNOWN when nothing is recognised.
        """
        try:
            # --- Cloudflare Turnstile ---
            turnstile_selectors = [
                '.cf-turnstile',
                'div[data-sitekey]',
                'input[name="cf-turnstile-response"]',
                '#cf-hcaptcha-container',  # CF sometimes wraps hCaptcha
                'iframe[src*="challenges.cloudflare.com"]',
                '#challenge-running',
                '#cf-challenge-running',
            ]
            for sel in turnstile_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        return CaptchaType.TURNSTILE
                except Exception:
                    pass

            # --- reCAPTCHA ---
            try:
                rc_el = await page.query_selector('div.g-recaptcha[data-sitekey], iframe[src*="recaptcha"]')
                if rc_el:
                    # v3 usually has no visible widget; detect by absence of checkbox
                    v3_el = await page.query_selector('div.g-recaptcha[data-size="invisible"]')
                    if v3_el:
                        return CaptchaType.RECAPTCHA_V3
                    return CaptchaType.RECAPTCHA_V2
            except Exception:
                pass

            # Try JS globals for reCAPTCHA
            try:
                has_rc = await page.evaluate(
                    "() => !!(window.grecaptcha && (window.grecaptcha.render || window.grecaptcha.execute))"
                )
                if has_rc:
                    invisible = await page.evaluate(
                        "() => !!document.querySelector('.g-recaptcha[data-size=\"invisible\"]')"
                    )
                    return CaptchaType.RECAPTCHA_V3 if invisible else CaptchaType.RECAPTCHA_V2
            except Exception:
                pass

            # --- hCaptcha ---
            try:
                hc_el = await page.query_selector('div.h-captcha[data-sitekey], iframe[src*="hcaptcha.com"]')
                if hc_el:
                    return CaptchaType.HCAPTCHA
            except Exception:
                pass

            try:
                has_hc = await page.evaluate("() => !!(window.hcaptcha)")
                if has_hc:
                    return CaptchaType.HCAPTCHA
            except Exception:
                pass

            # --- FunCaptcha / Arkose Labs ---
            try:
                fc_el = await page.query_selector(
                    '#FunCaptcha, [data-pkey], iframe[src*="arkoselabs"], iframe[src*="funcaptcha"]'
                )
                if fc_el:
                    return CaptchaType.FUNCAPTCHA
            except Exception:
                pass

            # --- Image/Text captcha (generic) ---
            try:
                img_captcha = await page.query_selector(
                    'img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"], '
                    'img[class*="captcha"], .captcha-image, #captcha-image, '
                    'img[src*="验证码"], img[alt*="验证码"]'
                )
                if img_captcha:
                    return CaptchaType.IMAGE_TEXT
            except Exception:
                pass

            # --- Text-based fallback ---
            try:
                body_text = await page.evaluate(
                    '() => (document.body ? document.body.innerText : "").toLowerCase()'
                )
                if any(k in body_text for k in ("recaptcha", "g-recaptcha")):
                    return CaptchaType.RECAPTCHA_V2
                if "hcaptcha" in body_text:
                    return CaptchaType.HCAPTCHA
                if any(k in body_text for k in ("funcaptcha", "arkoselabs")):
                    return CaptchaType.FUNCAPTCHA
                if any(k in body_text for k in ("captcha", "验证码", "robot", "human")):
                    return CaptchaType.IMAGE_TEXT  # generic fallback to image solve
            except Exception:
                pass

        except Exception as e:
            self.logger.debug("detect_captcha_type error: %s", e)

        return CaptchaType.UNKNOWN

    # ==================== Generic captcha parameter extraction ====================

    async def _extract_recaptcha_params(self, page) -> Optional[Dict[str, Any]]:
        """Extract reCAPTCHA sitekey and action from the current page."""
        try:
            # Try DOM attribute first
            sitekey = await page.evaluate(
                """() => {
                    const el = document.querySelector('.g-recaptcha[data-sitekey]');
                    if (el) return el.getAttribute('data-sitekey');
                    const scripts = Array.from(document.querySelectorAll('script'));
                    for (const s of scripts) {
                        const m = s.textContent.match(/['"]sitekey['"]\\s*:\\s*['"]([^'"]+)['"]/);
                        if (m) return m[1];
                    }
                    return null;
                }"""
            )
            if not sitekey:
                # Try iframe src parameter
                iframe = await page.query_selector('iframe[src*="recaptcha"]')
                if iframe:
                    src = await iframe.get_attribute("src") or ""
                    import re as _re
                    m = _re.search(r"[?&]k=([^&]+)", src)
                    sitekey = m.group(1) if m else None

            if not sitekey:
                return None

            action = await page.evaluate(
                """() => {
                    const el = document.querySelector('.g-recaptcha[data-action]');
                    return el ? el.getAttribute('data-action') : 'submit';
                }"""
            ) or "submit"

            return {"sitekey": sitekey, "pageurl": page.url, "action": action}
        except Exception as e:
            self.logger.debug("_extract_recaptcha_params error: %s", e)
            return None

    async def _extract_hcaptcha_params(self, page) -> Optional[Dict[str, Any]]:
        """Extract hCaptcha sitekey from the current page."""
        try:
            sitekey = await page.evaluate(
                """() => {
                    const el = document.querySelector('.h-captcha[data-sitekey]');
                    if (el) return el.getAttribute('data-sitekey');
                    const iframe = document.querySelector('iframe[src*="hcaptcha.com"]');
                    if (iframe) {
                        const m = iframe.src.match(/[?&]sitekey=([^&]+)/);
                        if (m) return m[1];
                    }
                    return null;
                }"""
            )
            return {"sitekey": sitekey, "pageurl": page.url} if sitekey else None
        except Exception as e:
            self.logger.debug("_extract_hcaptcha_params error: %s", e)
            return None

    async def _extract_funcaptcha_params(self, page) -> Optional[Dict[str, Any]]:
        """Extract FunCaptcha public key from the current page."""
        try:
            publickey = await page.evaluate(
                """() => {
                    const el = document.querySelector('[data-pkey], #FunCaptcha');
                    if (el) return el.getAttribute('data-pkey') || el.getAttribute('data-public-key');
                    return null;
                }"""
            )
            return {"publickey": publickey, "pageurl": page.url} if publickey else None
        except Exception as e:
            self.logger.debug("_extract_funcaptcha_params error: %s", e)
            return None

    async def _extract_image_captcha_params(self, page) -> Optional[Dict[str, Any]]:
        """Screenshot the captcha image element and return it as base64."""
        try:
            img_el = await page.query_selector(
                'img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"], '
                'img[class*="captcha"], .captcha-image, #captcha-image, '
                'img[src*="验证码"], img[alt*="验证码"]'
            )
            if img_el:
                img_bytes = await img_el.screenshot()
                img_b64 = base64.b64encode(img_bytes).decode()
                # Detect Chinese page context for language hint
                body_text = await page.evaluate('() => (document.body || {innerText: ""}).innerText')
                is_chinese = bool(
                    any(ord(c) > 0x4E00 for c in body_text[:200])
                )
                return {
                    "image_base64": img_b64,
                    "pageurl": page.url,
                    "is_chinese": is_chinese,
                }
            return None
        except Exception as e:
            self.logger.debug("_extract_image_captcha_params error: %s", e)
            return None

    async def _extract_captcha_params(
        self, captcha_type: CaptchaType, page
    ) -> Optional[Dict[str, Any]]:
        """Dispatch to the correct extractor for the given captcha type."""
        if captcha_type == CaptchaType.TURNSTILE:
            return await self.get_captcha_params(page, self.intercept_script)
        if captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
            return await self._extract_recaptcha_params(page)
        if captcha_type == CaptchaType.HCAPTCHA:
            return await self._extract_hcaptcha_params(page)
        if captcha_type == CaptchaType.FUNCAPTCHA:
            return await self._extract_funcaptcha_params(page)
        if captcha_type == CaptchaType.IMAGE_TEXT:
            return await self._extract_image_captcha_params(page)
        return None

    # ==================== Generic captcha token application ====================

    async def _apply_captcha_token(
        self, page, captcha_type: CaptchaType, token: str
    ) -> None:
        """Inject a solved captcha token/text back into the page."""
        if captcha_type == CaptchaType.TURNSTILE:
            await self.send_token_callback(page, token)
        elif captcha_type in (CaptchaType.RECAPTCHA_V2, CaptchaType.RECAPTCHA_V3):
            await page.evaluate(
                """(token) => {
                    const ta = document.getElementById('g-recaptcha-response');
                    if (ta) {
                        ta.value = token;
                        ta.style.display = 'block';
                    }
                    // Fire the grecaptcha callback if registered
                    const widget = document.querySelector('.g-recaptcha');
                    if (widget) {
                        const cb = widget.getAttribute('data-callback');
                        if (cb && typeof window[cb] === 'function') {
                            try { window[cb](token); } catch(e) {}
                        }
                    }
                    if (window.grecaptcha && window.grecaptcha.__oncomplete__) {
                        try { window.grecaptcha.__oncomplete__(token); } catch(e) {}
                    }
                }""",
                token,
            )
        elif captcha_type == CaptchaType.HCAPTCHA:
            await page.evaluate(
                """(token) => {
                    const ta = document.querySelector('[name="h-captcha-response"]');
                    if (ta) ta.value = token;
                    if (window.hcaptcha) {
                        try { window.hcaptcha.execute(); } catch(e) {}
                    }
                }""",
                token,
            )
        elif captcha_type == CaptchaType.IMAGE_TEXT:
            # Fill the text input and submit the closest form
            await page.evaluate(
                """(text) => {
                    const sel = 'input[name*="captcha"], input[id*="captcha"], input[placeholder*="captcha"],'
                              + 'input[name*="验证码"], input[id*="验证码"]';
                    const inp = document.querySelector(sel);
                    if (inp) {
                        inp.value = text;
                        const form = inp.closest('form');
                        if (form) form.submit();
                    }
                }""",
                token,
            )
        # FUNCAPTCHA and UNKNOWN: no generic injection available

    # ==================== Unified generic captcha solver ====================

    async def solve_captcha_if_needed(
        self,
        page,
        filepath: str = None,
        max_retries: int = 2,
        wait_after_solve: bool = True,
        task_download_dir: Optional[str] = None,
        request_context=None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Unified captcha handler for all non-Cloudflare challenge types.

        Flow:
          1. Detect the specific captcha type on the page.
          2. If Turnstile/Cloudflare detected, delegate to solve_cloudflare_if_needed().
          3. Wait a few seconds for extension auto-pass (CapSolver extension in headed mode).
          4. If still present, extract params and call CaptchaSolver API (primary + fallback).
          5. Apply token/text to the page.

        Returns:
            (True, None)      – no captcha, or captcha solved, page is usable
            (True, filepath)  – captcha solved and triggered a download
            (False, None)     – failed to solve
        """
        ctx = request_context or self._get_request_context()

        for attempt in range(max_retries):
            try:
                captcha_type = await self.detect_captcha_type(page)

                if captcha_type == CaptchaType.UNKNOWN:
                    self.logger.debug("[captcha] no recognisable captcha – returning ok")
                    return (True, None)

                # Route Turnstile back to the dedicated Cloudflare solver
                if captcha_type == CaptchaType.TURNSTILE:
                    return await self.solve_cloudflare_if_needed(
                        page,
                        filepath=filepath,
                        max_retries=max_retries,
                        wait_after_solve=wait_after_solve,
                        task_download_dir=task_download_dir,
                        request_context=request_context,
                    )

                self._emit_progress_event(
                    "blocker_detected",
                    request_context=ctx,
                    blocker=captcha_type.value,
                    attempt=attempt + 1,
                )
                self.logger.info(
                    "[captcha] detected type=%s attempt=%d/%d",
                    captcha_type.value,
                    attempt + 1,
                    max_retries,
                )

                # Give the CapSolver extension 5-8 s to auto-solve in headed mode
                if not self.headless:
                    self.logger.info("[captcha] waiting for extension auto-pass…")
                    await asyncio.sleep(6)
                    post_type = await self.detect_captcha_type(page)
                    if post_type == CaptchaType.UNKNOWN:
                        self.logger.info("[captcha] extension auto-passed type=%s", captcha_type.value)
                        self._emit_progress_event(
                            "challenge_solved", request_context=ctx, method="extension_auto", captcha_type=captcha_type.value
                        )
                        return (True, None)

                # API-based solving
                if not self._captcha_solver.has_any_provider:
                    self.logger.warning(
                        "[captcha] no API keys configured; cannot solve type=%s", captcha_type.value
                    )
                    return (False, None)

                params = await self._extract_captcha_params(captcha_type, page)
                if not params:
                    self.logger.error("[captcha] could not extract params for type=%s", captcha_type.value)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(3)
                        continue
                    return (False, None)

                solve_result: CaptchaSolveResult = await self._captcha_solver.solve(captcha_type, params)

                if not solve_result.success or not solve_result.token:
                    self.logger.error(
                        "[captcha] solve failed type=%s error=%s",
                        captcha_type.value,
                        solve_result.error,
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(4)
                        continue
                    return (False, None)

                self.logger.info(
                    "[captcha] solved type=%s provider=%s latency=%.0f ms",
                    captcha_type.value,
                    solve_result.provider,
                    solve_result.latency_ms,
                )

                # Apply token/text to the page
                await self._apply_captcha_token(page, captcha_type, solve_result.token)

                if wait_after_solve:
                    try:
                        await page.wait_for_load_state(
                            "domcontentloaded",
                            timeout=self.timeouts.get("load_state_timeout", 10) * 1000,
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(1)

                # Verify the captcha is gone
                post_type = await self.detect_captcha_type(page)
                if post_type == CaptchaType.UNKNOWN:
                    self._emit_progress_event(
                        "challenge_solved",
                        request_context=ctx,
                        method=f"api_{solve_result.provider}",
                        captcha_type=captcha_type.value,
                        latency_ms=solve_result.latency_ms,
                    )
                    return (True, None)
                else:
                    self.logger.warning(
                        "[captcha] still present after solve, type=%s", post_type.value
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)
                        continue

            except Exception as exc:
                self.logger.error("[captcha] unexpected error attempt=%d: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                    continue

        return (False, None)

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
            self.logger.error(f"检测Cloudflare验证时出错: {e}")
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
    async def get_captcha_params(
        self,
        page,
        script,
        request_context: Optional[DownloadRequestContext] = None,
    ):
        """
        在加载前注入拦截脚本，刷新页面后轮询 window.__cf_intercepted_params 获取 Turnstile 参数。
        """
        self.logger.info("开始获取 Cloudflare JS Challenge 参数")
        await page.add_init_script(script)
        await page.reload()
        try:
            ctx = request_context or self._get_request_context()
            wait_seconds = self._start_phase(
                "captcha_param_extract",
                base_seconds=self.timeouts.get("captcha_timeout", 60),
                request_context=ctx,
            )
            deadline = time.monotonic() + max(wait_seconds, 1.0)
            while time.monotonic() < deadline:
                params = await page.evaluate("() => window.__cf_intercepted_params || null")
                if params:
                    self.logger.info("Parameters received")
                    self._grant_progress(
                        "captcha_params_extracted",
                        extra_seconds=15,
                        request_context=ctx,
                        allow_count=1,
                    )
                    self._emit_progress_event(
                        "captcha_params_extracted",
                        request_context=ctx,
                        sitekey_present=bool(params.get("sitekey")),
                    )
                    return params
                await asyncio.sleep(0.5)
            raise Exception("获取 Turnstile 参数超时")
        except Exception as e:
            self.logger.error(f"提取参数失败: {str(e)}")
            raise
            
    # 使用 2Captcha 服务解决 Turnstile 验证码    
    def solver_captcha(self,apikey, params):  # 使用 2Captcha 服务解决 Turnstile 验证码    
        """
        使用 2Captcha 服务解决 Turnstile 验证码。
        需要安装: pip install 2captcha-python
        """
        if TwoCaptcha is None:
            self.logger.warning("twocaptcha not installed; captcha solving disabled. Install with: pip install 2captcha-python")
            return None
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

    async def solver_captcha_async(self, apikey, params):
        """异步版本的验证码解决器，避免阻塞事件循环"""
        return await asyncio.to_thread(self.solver_captcha, apikey, params)
        
# 将解码后的 token 传入页面回调函数，从而正式解决验证码问题
    async def send_token_callback(self, page, token):
        """
        注入 Token 并尝试触发验证：先调用拦截到的 callback，若无则填充隐藏 input 并提交表单。
        """
        self.logger.info("注入 Token 并尝试触发验证...")
        await page.evaluate(
            """(token) => {
            if (window.__cf_callback) {
                try { window.__cf_callback(token); return true; } catch (e) {}
            }
            let input = document.querySelector('input[name="cf-turnstile-response"]');
            if (input) {
                input.value = token;
                let form = input.closest('form');
                if (form) form.submit();
                return true;
            }
            return false;
        }""",
            token,
        )
        self.logger.info("Token 已发送至回调或表单")

# ==================== 通用 Cloudflare 验证码自动处理 ====================
    async def solve_cloudflare_if_needed(self, page, filepath: str = None,
                                          max_retries: int = 2,
                                          wait_after_solve: bool = True,
                                          task_download_dir: Optional[str] = None,
                                          request_context: Optional[DownloadRequestContext] = None) -> Tuple[bool, Optional[str]]:
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
        ctx = request_context or self._get_request_context()
        phase_timeout = self._start_phase(
            "challenge_handling",
            base_seconds=self.timeouts.get("captcha_timeout", 60),
            request_context=ctx,
        )
        if ctx and ctx.budget_controller is not None:
            ctx.budget_controller.promote_challenge_window()

        for attempt in range(max_retries):
            try:
                cf_detected = await self.is_cloudflare_verifying(page)
                if not cf_detected:
                    self.logger.debug("未检测到 Cloudflare 验证码")
                    return (True, None)

                self._emit_progress_event(
                    "blocker_detected",
                    request_context=ctx,
                    blocker="cloudflare",
                    attempt=attempt + 1,
                )
                
                # 给扩展程序（CapSolver）或纯 JS 盾 5–8 秒自动通过的缓冲时间
                self.logger.info("检测到 Cloudflare，等待 5-8 秒观察是否能自动通过...")
                try:
                    auto_pass_timeout_ms = int(max(1000, min(phase_timeout, self.timeouts.get("cloudflare_timeout", 60), 12) * 1000))
                    await page.wait_for_function(
                        "() => !document.querySelector('#challenge-running') && !document.querySelector('#cf-spinner-please-wait')",
                        timeout=auto_pass_timeout_ms,
                    )
                    await asyncio.sleep(1)
                    if not await self.is_cloudflare_verifying(page, detection_duration=0.5):
                        self.logger.info("Cloudflare 验证已自动解决 (插件或 JS 盾)")
                        self._grant_progress(
                            "challenge_solved",
                            extra_seconds=10,
                            request_context=ctx,
                            allow_count=1,
                        )
                        self._emit_progress_event("challenge_solved", request_context=ctx, method="auto_pass")
                        return (True, None)
                except PlaywrightTimeoutError:
                    pass
                
                self.logger.info("自动解决超时，启动 2Captcha 手动拦截...")
                self.logger.info(f"检测到 Cloudflare 验证码，尝试解决 (尝试 {attempt + 1}/{max_retries})")
                
                if not self.apikey:
                    self.logger.error("未配置 2Captcha API Key，无法解决验证码")
                    return (False, None)
                
                params = None
                try:
                    params = await self.get_captcha_params(page, self.intercept_script, request_context=ctx)
                except Exception as param_e:
                    self.logger.error(f"提取验证码参数失败 (尝试 {attempt + 1}): {param_e}")
                    failure_class = self._classify_failure(exception=param_e, blocker=Blocker(BlockerType.CAPTCHA))
                    failure_count = self._record_failure(
                        failure_class,
                        request_context=ctx,
                        attempt=attempt + 1,
                    )
                    if failure_class == "anti_bot_challenge" and failure_count >= 2:
                        return (False, None)
                    should_retry, backoff = self._should_retry_failure(failure_class, attempt + 1)
                    if should_retry and attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        continue
                    return (False, None)
                
                if not params:
                    self.logger.error("未能获取验证码参数")
                    failure_class = self._classify_failure(blocker=Blocker(BlockerType.CAPTCHA))
                    self._record_failure(failure_class, request_context=ctx, attempt=attempt + 1)
                    should_retry, backoff = self._should_retry_failure(failure_class, attempt + 1)
                    if should_retry and attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        continue
                    return (False, None)
                
                self.logger.info(f"成功提取验证码参数: sitekey={params.get('sitekey', 'N/A')[:20]}...")
                
                token = None
                try:
                    token = await self.solver_captcha_async(self.apikey, params)
                except Exception as solve_e:
                    self.logger.error(f"验证码解决失败 (尝试 {attempt + 1}): {solve_e}")
                    failure_class = self._classify_failure(exception=solve_e, blocker=Blocker(BlockerType.CAPTCHA))
                    self._record_failure(failure_class, request_context=ctx, attempt=attempt + 1)
                    should_retry, backoff = self._should_retry_failure(failure_class, attempt + 1)
                    if should_retry and attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
                        continue
                    return (False, None)
                
                if not token:
                    self.logger.error("未能获取验证码 token")
                    failure_class = self._classify_failure(blocker=Blocker(BlockerType.CAPTCHA))
                    self._record_failure(failure_class, request_context=ctx, attempt=attempt + 1)
                    should_retry, backoff = self._should_retry_failure(failure_class, attempt + 1)
                    if should_retry and attempt < max_retries - 1:
                        await asyncio.sleep(backoff)
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
                            self._grant_progress(
                                "download_materialized",
                                extra_seconds=15,
                                request_context=ctx,
                                allow_count=1,
                            )
                            self._emit_progress_event("download_materialized", request_context=ctx, method="cloudflare_callback")
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
                    self._grant_progress(
                        "challenge_solved",
                        extra_seconds=10,
                        request_context=ctx,
                        allow_count=1,
                    )
                    self._emit_progress_event("challenge_solved", request_context=ctx, method="captcha_token")
                    return (True, None)
                else:
                    self.logger.warning("验证码发送后仍检测到 Cloudflare，可能需要重试")
                    
            except Exception as e:
                self.logger.error(f"解决 Cloudflare 验证码时出错 (尝试 {attempt + 1}): {e}")
                failure_class = self._classify_failure(exception=e, blocker=Blocker(BlockerType.CLOUDFLARE))
                self._record_failure(failure_class, request_context=ctx, attempt=attempt + 1)
                should_retry, backoff = self._should_retry_failure(failure_class, attempt + 1)
                if should_retry and attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
        
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
                              year: Optional[int] = None, page=None) -> bool:
        """直接下载文件（对于直接提供PDF下载链接的情况）

        当传入 page 时，优先使用 Playwright 浏览器上下文发起请求，100% 继承已通过的
        Cloudflare 验证、Session Cookie 等鉴权信息，避免 aiohttp 裸连被 403 拦截。
        若 Playwright 请求失败，自动降级回 aiohttp。
        
        Args:
            url: 下载链接
            filepath: 保存路径
            headers: 请求头（仅 aiohttp 降级时使用）
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            page: Playwright Page 对象（可选），传入后使用浏览器上下文鉴权下载
            
        Returns:
            bool: 是否下载成功
        """
        self.logger.info(f"尝试直接下载: {url}")

        # 确保文件所在目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        content = None
        status = 0

        # 优先使用 Playwright 浏览器上下文（携带完整 Cookie/CF 验证）
        if page:
            try:
                self.logger.info("使用 Playwright 浏览器上下文抓取直链...")
                req_response = await page.context.request.get(
                    url,
                    timeout=self.download_timeout * 1000,
                    headers={"Accept": "application/pdf,application/epub+zip,application/x-download,*/*"}
                )
                status = req_response.status
                if status == 200:
                    content_type = req_response.headers.get("content-type", "").lower()
                    if "text/html" not in content_type:
                        content = await req_response.body()
                    else:
                        self.logger.warning("Playwright 请求返回了 HTML，可能遇到付费墙或验证码，降级到 aiohttp")
            except Exception as e:
                self.logger.debug(f"Playwright 请求异常，降级到 aiohttp: {e}")

        # aiohttp 降级路径
        if not content:
            if headers is None:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            try:
                session = self._http_session
                owns_session = False
                if session is None or session.closed:
                    timeout = aiohttp.ClientTimeout(total=min(self.download_timeout, 60))
                    session = aiohttp.ClientSession(timeout=timeout)
                    owns_session = True
                ssl_skip_domains = {"sci-hub.st", "sci-hub.se", "libgen.rs"}
                parsed = urlparse(url)
                use_ssl = parsed.hostname not in ssl_skip_domains if parsed.hostname else True
                async with session.get(url, headers=headers, ssl=use_ssl) as response:
                    status = response.status
                    if status == 200:
                        content = await response.read()
                    else:
                        self.logger.warning(f"下载失败，HTTP状态码: {response.status}")
                        return False
            except Exception as e:
                self.logger.error(f"直接下载过程中出错: {str(e)}")
                return False
            finally:
                if owns_session:
                    try:
                        await session.close()
                    except Exception:
                        pass

        try:
            if content and status == 200:
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                if not self.is_valid_pdf_file(filepath):
                    self.logger.warning(f"直接下载的文件验证失败: {filepath}")
                    os.remove(filepath)
                    return False
                
                self.logger.info(f"直接下载的文件验证成功: {filepath}")
                
                # 计算并记录 MD5（用于去重/过滤）
                md5 = self._compute_file_md5(filepath)
                self._record_download_success(
                    filepath,
                    url,
                    title=title,
                    authors=authors,
                    method="direct_download",
                    md5=md5
                )
                
                self.logger.info(f"直接下载成功: {filepath}")
                return True
            else:
                self.logger.warning(f"下载失败，HTTP状态码: {status}")
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
        
        # 仅当 "pdf" 出现在路径/参数关键位置时才判定
        parsed = urlparse(url_lower)
        path = parsed.path
        query = parsed.query
        if any(p in path for p in ['/pdf/', '/pdf.', '.pdf', '/pdfdirect', '/pdfft', '/epdf/']):
            return True
        if any(p in query for p in ['pdf=', 'format=pdf', 'type=pdf']):
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
            async with aiohttp.ClientSession() as session:
                for attempt in range(1, 4):
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
            async with aiohttp.ClientSession() as session:
                for attempt in range(1, 4):
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
                self._record_download_success(
                    filepath,
                    url,
                    title=title,
                    authors=authors,
                    method="brightdata",
                    md5=md5
                )

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
    
    def _write_pdf_preferences(self, user_data_dir: str, custom_download_dir: Optional[str] = None):
        """写入 PDF 下载首选项到用户数据目录
        
        确保 always_open_pdf_externally=True 生效，
        这样 PDF 链接会触发下载事件而不是在浏览器中内联显示。
        若传入 custom_download_dir，则下载/另存为均指向该沙盒目录（并发隔离）。
        
        Args:
            user_data_dir: 浏览器用户数据目录
            custom_download_dir: 可选，专属下载目录（会话沙盒），未传则用 self.download_dir
        """
        default_dir = os.path.join(user_data_dir, "Default")
        os.makedirs(default_dir, exist_ok=True)
        target_dir = custom_download_dir or self.download_dir
        os.makedirs(target_dir, exist_ok=True)
        
        preferences_path = os.path.join(default_dir, "Preferences")
        preferences = {
            "plugins": {
                "always_open_pdf_externally": True
            },
            "download": {
                "prompt_for_download": False,
                "default_directory": target_dir
            },
            "savefile": {
                "default_directory": target_dir
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
        """初始化浏览器会话池。优先使用 resident headed context 池（每 slot 独立 context）；否则单上下文多标签。"""
        self.logger.debug(f"[DEBUG] _init_session_pool called, already_initialized={self._session_pool_initialized}")
        async with self._context_lock:
            if self._session_pool_initialized:
                return

            effective_headless = self.get_pool_headless()
            try:
                from config.settings import settings
                from src.retrieval.context_pool import SharedContextPool
                sb = getattr(settings, "shared_browser", None)
                pool = SharedContextPool.get_instance()
                if (
                    not effective_headless
                    and sb is not None
                    and pool.is_initialized()
                ):
                    pool_size = getattr(sb, "headed_context_pool_size", 2)
                    self._session_pool_size = min(self.max_concurrent, pool_size)
                    self._pool_sem = asyncio.Semaphore(self._session_pool_size)
                    self._use_context_pool = True
                    self._session_pool_initialized = True
                    self._pool_was_headless = False
                    self.logger.info(
                        "会话池使用 resident headed context 池，并发上限=%d",
                        self._session_pool_size,
                    )
                    self._start_cleanup_task()
                    return
            except Exception as e:
                self.logger.debug("context pool not used for downloader: %s", e)

            # 单上下文多标签模式
            self.logger.info(f"正在初始化会话池（单上下文多标签），大小: {self._session_pool_size}")
            user_data_dir = os.path.join(
                os.path.dirname(self.download_dir),
                ".browser_data",
                "shared_session",
            )
            shared_downloads_dir = os.path.join(self.download_dir, ".shared_downloads")
            os.makedirs(user_data_dir, exist_ok=True)
            os.makedirs(shared_downloads_dir, exist_ok=True)
            self._clean_lock_files_in_dir(user_data_dir)
            self._write_pdf_preferences(user_data_dir, custom_download_dir=shared_downloads_dir)

            try:
                self.logger.info(
                    "[headed-diag] _init_session_pool: effective_headless=%s, "
                    "calling launch_persistent_browser(headless=%s, reuse_shared_cdp=True, browser_type=%r, ...)",
                    effective_headless,
                    effective_headless,
                    self.browser_type,
                )
                self._shared_browser_context = await self.browser_manager.launch_persistent_browser(
                    user_data_dir=user_data_dir,
                    browser_type=self.browser_type,
                    headless=effective_headless,
                    stealth_mode=self.stealth_mode,
                    reuse_shared_cdp=True,
                    viewport={'width': 1280, 'height': 720},
                    downloads_path=shared_downloads_dir,
                    timeout=self.download_timeout * 1000,
                    proxy=getattr(self, "_proxy", None),
                    extension_path=getattr(self, "_capsolver_extension_path", None),
                )
            except Exception as e:
                self._shared_browser_context = None
                self.logger.error(f"创建共享浏览器上下文失败: {e}")
                return

            for i in range(self._session_pool_size):
                slot = {
                    'context': self._shared_browser_context,
                    'user_data_dir': user_data_dir,
                    'session_downloads_dir': shared_downloads_dir,
                    'index': i,
                    'last_used': time.time(),
                    'in_use': False,
                }
                self._session_pool_data.append(slot)
                await self._session_pool.put(slot)

            self._session_pool_initialized = True
            # Record the mode actually used to create the shared context. Do not re-read
            # the mutable override here, otherwise a concurrent request can flip the flag
            # mid-init and make a headless context look "headed", preventing rebuild.
            self._pool_was_headless = effective_headless
            self.logger.info(
                "会话池初始化完成（单上下文），可用 slot 数: %d，created_headless=%s, current_effective_headless=%s",
                self._session_pool.qsize(),
                self._pool_was_headless,
                self.get_pool_headless(),
            )
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
        
        # 仅实例级 override 能触发共享 context 重建；单次请求不应影响 pool 生命周期。
        effective = self.get_pool_headless()
        will_rebuild = (
            self._session_pool_initialized
            and self._pool_was_headless is not None
            and effective != self._pool_was_headless
        )
        self.logger.info(
            "[headed-diag] _acquire_session: effective=%s, _pool_was_headless=%s, "
            "pool_initialized=%s -> will_rebuild=%s",
            effective,
            self._pool_was_headless,
            self._session_pool_initialized,
            will_rebuild,
        )
        if will_rebuild:
            self.logger.info("有头/无头设置变更，关闭会话池以便用新模式重建")
            await self._close_session_pool()
        
        # 如果会话池已关闭，重新初始化
        if not self._session_pool_initialized:
            await self._init_session_pool()

        # Resident headed context 池：按需借出
        if self._use_context_pool and self._pool_sem is not None:
            await self._pool_sem.acquire()
            try:
                from config.settings import settings
                from src.retrieval.context_pool import acquire_headed_context
                cfg = getattr(settings, "shared_browser", None)
                timeout = getattr(cfg, "context_acquire_timeout_seconds", 30.0) if cfg else 30.0
                lease = await acquire_headed_context(timeout=timeout, purpose="download")
                if lease is None:
                    self._pool_sem.release()
                    return None
                session_downloads_dir = getattr(lease._slot, "downloads_dir", None) or os.path.join(
                    self.download_dir, ".shared_downloads", lease.slot_id
                )
                os.makedirs(session_downloads_dir, exist_ok=True)
                return {
                    "context": lease.context,
                    "user_data_dir": "",
                    "session_downloads_dir": session_downloads_dir,
                    "index": 0,
                    "last_used": time.time(),
                    "in_use": True,
                    "_lease": lease,
                }
            except Exception as e:
                self._pool_sem.release()
                raise

        if self._session_pool.empty() and not self._session_pool_data:
            self.logger.error("会话池为空且无法初始化")
            return None

        try:
            # 阻塞等待可用会话
            session = await asyncio.wait_for(
                self._session_pool.get(),
                timeout=self.timeouts.get("session_acquire_timeout", 15)
            )
            session["in_use"] = True
            session["last_used"] = time.time()
            self.logger.debug(f"获取会话 [{session['index']}]")
            return session
        except asyncio.TimeoutError:
            self.logger.error("获取会话超时")
            return None

    async def _release_session(self, session: Dict):
        """释放会话回会话池或归还 resident context 池 lease。"""
        if not session:
            return
        self._last_activity_time = time.time()
        if session.get("_lease") is not None:
            try:
                from src.retrieval.context_pool import release_context as release_context_lease
                await release_context_lease(session["_lease"], had_error=False)
            except Exception as e:
                self.logger.warning("release headed context lease: %s", e)
            if self._pool_sem is not None:
                self._pool_sem.release()
            self.logger.debug("释放会话 (pool lease)")
            return
        session["in_use"] = False
        session["last_used"] = time.time()
        await self._session_pool.put(session)
        self.logger.debug(f"释放会话 [{session['index']}]")

    async def _close_session_pool(self):
        """关闭会话池中的所有上下文，立即释放资源。使用 resident pool 时仅重置状态，不关闭池内 context。"""
        async with self._context_lock:
            if not self._session_pool_initialized:
                return

            self.logger.info("正在关闭会话池...")

            cleanup_task = self._cleanup_task
            current_task = asyncio.current_task()
            self._cleanup_task = None
            if cleanup_task and not cleanup_task.done() and cleanup_task is not current_task:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass

            if self._use_context_pool:
                self._use_context_pool = False
                self._pool_sem = None
                self._warmed_domains.clear()
                self._domain_locks.clear()
                self._session_pool_initialized = False
                self._pool_was_headless = None
                self.logger.info("会话池已关闭（resident pool 模式，context 由全局池管理）")
                return

            close_tasks = []
            for session_data in self._session_pool_data:
                close_tasks.append(self._close_single_session(session_data))

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            self._session_pool_data.clear()
            while not self._session_pool.empty():
                try:
                    self._session_pool.get_nowait()
                except Exception:
                    break

            if self._shared_browser_context:
                try:
                    await asyncio.wait_for(self._shared_browser_context.close(), timeout=5)
                except Exception as e:
                    self.logger.warning(f"关闭共享浏览器上下文时出错: {e}")
                self._shared_browser_context = None

            self._warmed_domains.clear()
            self._domain_locks.clear()
            self._session_pool_initialized = False
            self._pool_was_headless = None
            self.logger.info("会话池已关闭，资源已释放")

    async def _close_single_session(self, session_data: Dict):
        """关闭单个会话（辅助方法）。单上下文模式下 slot 不拥有 context，不在此处关闭 context。"""
        try:
            session_data['in_use'] = False
            self.logger.debug(f"已释放 slot [{session_data.get('index', '?')}]")
        except Exception as e:
            self.logger.warning(f"释放 slot [{session_data.get('index', '?')}] 时出错: {e}")

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
        
        # 关闭会话池（唯一的浏览器上下文由会话池管理）
        if self._session_pool_initialized:
            await self._close_session_pool()
        
        self.logger.info("所有浏览器资源已释放")

    # ==================== 会话池相关方法结束 ====================

    async def get_browser_context(self):
        """获取共享浏览器上下文（兼容接口）

        所有浏览器操作统一通过会话池管理。此方法确保会话池已初始化，
        然后返回池内唯一的共享上下文。不应再单独创建 context。
        """
        if not self._session_pool_initialized:
            await self._init_session_pool()
        return self._shared_browser_context

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
                    response = (await asyncio.to_thread(input, "检测到正在进行的下载。是否关闭浏览器？(y/N): ")).strip().lower()
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

        # 关闭会话池（单上下文多页面：统一入口）
        if self._session_pool_initialized:
            try:
                await self._close_session_pool()
            except Exception as e:
                self.logger.warning(f"关闭会话池时出错: {e}")

        if self._http_session is not None:
            try:
                if not self._http_session.closed:
                    await self._http_session.close()
            except Exception as e:
                self.logger.warning(f"关闭 aiohttp 会话失败: {e}")
            finally:
                self._http_session = None

        # 关闭浏览器（通过 BrowserManager 统一管理，清理底层 Playwright 进程）
        if self.browser_manager is not None:
            try:
                self.logger.info("正在关闭 BrowserManager...")
                await self.browser_manager.close()
                self.logger.info("BrowserManager 已关闭")
            except Exception as e:
                self.logger.warning(f"关闭 BrowserManager 时出错: {str(e)}")

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

    async def pdf_download_with_browser(
        self,
        url: str,
        filepath: str,
        request_context: Optional[DownloadRequestContext] = None,
    ) -> bool:
        """使用浏览器下载 PDF 文件
        
        使用会话池获取浏览器上下文，保证 cookie/session 持续性和 PDF 设置生效。
        """
        self.logger.debug(f"[DEBUG] pdf_download_with_browser: url={url}, filepath={filepath}")
        downloads_dir = os.path.dirname(filepath)
        os.makedirs(downloads_dir, exist_ok=True)
        finished = False
        page = None
        session = None
        session_downloads_dir = None
        active_context = request_context or self._get_request_context()
        token = None
        restore_state: Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[bool], Optional[str], Optional[str]]] = None

        if active_context is None:
            active_context = self._build_request_context(
                paper_id=os.path.splitext(os.path.basename(filepath))[0],
                strategy="browser_pdf_url",
                title=os.path.splitext(os.path.basename(filepath))[0],
                url=url,
                filepath=filepath,
            )
        else:
            restore_state = (
                active_context.strategy,
                active_context.url,
                active_context.filepath,
                active_context.show_browser_override,
                active_context.llm_provider_override,
                active_context.llm_model_override,
            )
            active_context.strategy = active_context.strategy or "browser_pdf_url"
            active_context.url = url
            active_context.filepath = filepath
        token = _REQUEST_CONTEXT.set(active_context)
        self._emit_progress_event("request_context_ready", request_context=active_context, url=url[:120])
        
        # 创建任务专属临时目录（并发安全）
        task_download_dir, task_start_time, initial_main_files = self._create_task_download_dir()
        active_context.task_download_dir = task_download_dir
        
        try:
            self._start_phase(
                "session_acquire",
                base_seconds=self.timeouts.get("session_acquire_timeout", 15),
                request_context=active_context,
            )
            # 从会话池获取上下文
            session = await self._acquire_session()
            if session is None:
                self.logger.error("无法从会话池获取浏览器上下文")
                self._record_failure(
                    self._classify_failure(exception=asyncio.TimeoutError()),
                    request_context=active_context,
                    stage="session_acquire",
                )
                return False
            active_context.session_slot_index = session.get("index")
            self._emit_progress_event(
                "session_acquired",
                request_context=active_context,
                slot=session.get("index"),
            )
            
            context = session['context']
            session_downloads_dir = session.get('session_downloads_dir')
            # 任务前清场：彻底排空当前 Session 沙盒，防止上一任务残留污染（含 Chromium 临时目录）
            if session_downloads_dir and os.path.exists(session_downloads_dir):
                shutil.rmtree(session_downloads_dir, ignore_errors=True)
                os.makedirs(session_downloads_dir, exist_ok=True)

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
            
            # 预热步骤：按平台 key 处理（避免对 doi.org 等跳转域做无效预热）
            try:
                await self._warmup_platform_once(
                    page,
                    request_url=url,
                    task_download_dir=task_download_dir,
                    request_context=active_context,
                )
            except Exception as warmup_e:
                self.logger.warning(f"预热访问失败，继续尝试直接下载: {warmup_e}")
            
            # 设置下载事件监听器（列表接收，避免多下载事件覆盖）
            captured_downloads = []
            def _on_download_capture(download):
                captured_downloads.append(download)
                self.logger.debug(f"[下载监听] 捕获下载事件: {download.suggested_filename}")
            page.on("download", _on_download_capture)

            sniffed_pdf_buffers_simple = []
            async def _on_response_sniff_simple(response):
                try:
                    if response.status == 200 and response.request.method not in ("OPTIONS", "HEAD"):
                        ct = (response.headers.get("content-type") or "").lower()
                        ru = response.url.lower()
                        if ("application/pdf" in ct or ru.split('?')[0].endswith(".pdf")) and not ru.startswith("data:"):
                            body = await response.body()
                            if body and len(body) > 10240:
                                self.logger.info(f"[流量嗅探] pdf_download_with_browser 截获 PDF: {response.url[:80]}")
                                sniffed_pdf_buffers_simple.append((response.url, body))
                except Exception:
                    pass
            page.on("response", _on_response_sniff_simple)

            # 尝试初始下载
            self.logger.info("开始尝试初始下载")
            response = None
            try:
                # 初始下载等待时间上限，避免过久阻塞
                self._start_phase(
                    "initial_probe",
                    base_seconds=self.timeouts.get("download_event_timeout", 15),
                    request_context=active_context,
                )
                download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                async with page.expect_download(timeout=download_timeout_ms) as download_info:
                    try: 
                        response = await page.goto(url, wait_until="domcontentloaded") 
                        await self._handle_cookie_consent(page, context_label="跳转后")
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                    except Exception as goto_e:
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                download = await download_info.value
                self._grant_progress(
                    "download_event_captured",
                    extra_seconds=15,
                    request_context=active_context,
                    allow_count=1,
                )
                self._emit_progress_event("download_event_captured", request_context=active_context)
                # 使用统一的保存方法（并发安全）
                if await self._save_and_validate_download(download, filepath, task_download_dir, "initial_download"):
                    self.logger.info(f"初始下载尝试成功保存到: {filepath}")
                    finished = True
                    self._emit_progress_event("artifact_validated", request_context=active_context, method="initial_download")
                    return True
                else:
                    self.logger.warning(f"初始下载的文件无效")
                    finished = False

            except Exception as e:
                self.logger.error(f"初始下载失败：{e}, 检查是否404错误")
                # expect_download 超时后，下载可能已在后台落盘；先等待其稳定，避免过早关闭页面导致 Target closed
                for dl in list(captured_downloads):
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
                                    except Exception:
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
                    page, filepath, task_download_dir=task_download_dir, request_context=active_context
                )
                if cf_success:
                    self._mark_platform_ready(
                        self._resolve_platform_key(url, page.url),
                        reason="browser_pdf_flow",
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
                        session_downloads_dir=session_downloads_dir if session else None,
                        initial_main_files=initial_main_files,
                        task_start_time=task_start_time,
                        request_context=active_context,
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
                    while captured_downloads:
                        dl = captured_downloads.pop(0)
                        if await self._save_and_validate_download(
                            dl, filepath, task_download_dir, "finally-cleanup"
                        ):
                            finished = True
                            break
                    if not finished and sniffed_pdf_buffers_simple:
                        for s_url, s_body in list(sniffed_pdf_buffers_simple):
                            temp = os.path.join(task_download_dir, "sniffed_simple.pdf")
                            try:
                                with open(temp, 'wb') as f:
                                    f.write(s_body)
                                if self.is_valid_pdf_file(temp):
                                    if os.path.exists(filepath):
                                        os.remove(filepath)
                                    shutil.move(temp, filepath)
                                    self.logger.info(f"⚡ [极速落盘-simple] 内存流兜底落盘成功 -> {os.path.basename(filepath)}")
                                    finished = True
                                    break
                                else:
                                    try:
                                        os.remove(temp)
                                    except Exception:
                                        pass
                            except Exception as e:
                                self.logger.debug(f"内存流兜底落盘失败: {e}")
                    if not finished and await self._check_download_in_progress(
                        filepath, task_download_dir=task_download_dir
                    ):
                        await self._wait_for_download_complete(
                            filepath,
                            timeout=min(self.timeouts.get("download_complete_timeout", 20), 10),
                            task_download_dir=task_download_dir,
                            session_downloads_dir=session_downloads_dir if session else None,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time,
                            request_context=active_context,
                        )
            except Exception as wait_e:
                self.logger.debug(f"等待下载完成失败（清理前）: {wait_e}")

            # 清理任务专属临时目录
            self._cleanup_task_download_dir(task_download_dir)
            
            # 移除下载监听器
            if page:
                try:
                    page.remove_listener("download", _on_download_capture)
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
            if active_context is not None:
                active_context.trace.final_outcome = "success" if finished else "failed"
                self._emit_progress_event(
                    "request_finished",
                    request_context=active_context,
                    success=finished,
                )
            if restore_state is not None and active_context is not None:
                (
                    active_context.strategy,
                    active_context.url,
                    active_context.filepath,
                    active_context.show_browser_override,
                    active_context.llm_provider_override,
                    active_context.llm_model_override,
                ) = restore_state
            if token is not None:
                _REQUEST_CONTEXT.reset(token)

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
            # 缩短标题（英文优先按词截断；中文/无空格场景按字符截断）
            if len(clean_title) > 50:
                words = clean_title.split()
                if len(words) > 1:
                    clean_title = "_".join(words[:6])
                if len(clean_title) > 50:
                    clean_title = clean_title[:50]
            
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
        """检查文件是否是有效的PDF文件；严格校验魔法头位置并排除 HTML 伪装。
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否是有效的PDF文件
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                return False
            file_size = os.path.getsize(file_path)
            if file_size < 1000:
                self.logger.warning(f"文件过小 ({file_size} 字节)，不可能是有效的PDF")
                return False
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            # 严格检查 %PDF- 位置（允许最多 30 字节偏移，应对 BOM 等）
            pdf_idx = header.find(b"%PDF-")
            if pdf_idx < 0 or pdf_idx > 30:
                self.logger.warning(f"文件不是PDF格式或头部异常: {file_path}")
                return False
            # 排除 HTML 伪装（如 Cloudflare 验证页中恰好包含 PDF 字符串）
            lower_header = header[:500].lower()
            if b"<!doctype html>" in lower_header or b"<html" in lower_header:
                self.logger.warning(f"文件实为 HTML 伪装: {file_path}")
                return False
            if not (b'obj' in header or b'stream' in header or b'/Type' in header or b'/Pages' in header):
                self.logger.warning(f"文件缺少基本PDF结构: {file_path}")
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
        """保存 MD5 索引到文件 (并发写安全：锁 + UUID 临时文件)"""
        md5_index_file = os.path.join(self.download_dir, ".md5_index.json")
        with self._io_lock:
            temp_file = None
            try:
                temp_file = md5_index_file + f".{uuid.uuid4().hex[:8]}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self.file_md5_index.copy(), f, indent=2)
                if sys.platform == 'win32' and os.path.exists(md5_index_file):
                    os.remove(md5_index_file)
                os.rename(temp_file, md5_index_file)
                temp_file = None
            except Exception as e:
                self.logger.warning(f"保存 MD5 索引失败: {e}")
                if temp_file is not None and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
    
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
        
        doi_pattern = re.compile(r'10\.\d{4,9}/\S+', re.IGNORECASE)
        doi_match = doi_pattern.search(query or "")
        if not doi_match:
            self.logger.info("Anna's Archive MD5 搜索跳过：query 不是 DOI")
            return None

        target_doi = doi_match.group(0).lower().rstrip(').,;')
        encoded_query = quote(target_doi, safe='')

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

        if not query:
            return []

        encoded_query = quote(query, safe='')
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
        
        async def download_with_timeout(paper):
            """带超时保护的单篇下载"""
            delay = random.uniform(0.5, 1.0)
            await asyncio.sleep(delay)
            paper_title = paper.get('title', 'Unknown')[:50]
            hard_timeout = self._get_request_budget_settings()["hard"]
            try:
                await asyncio.wait_for(
                    self._download_paper(paper, semaphore, stats),
                    timeout=hard_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"论文下载超时（{hard_timeout:.0f}秒）: {paper_title}"
                )
                stats['failed'] += 1
            except Exception as e:
                self.logger.error(f"论文下载异常: {paper_title} - {e}")
                stats['failed'] += 1

        await asyncio.gather(
            *(download_with_timeout(paper) for paper in papers),
            return_exceptions=True
        )
        
        return stats

    async def _strategy_direct_pdf(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.pdf_url:
            return False
        self.logger.info(f"尝试直接下载PDF链接: {ctx.pdf_url}")
        return await self.download_direct(
            ctx.pdf_url,
            ctx.filepath,
            title=ctx.title,
            authors=ctx.authors,
            year=ctx.year,
        )

    async def _strategy_browser_pdf_url(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.pdf_url:
            return False
        self.logger.info(f"尝试通过浏览器下载PDF链接: {ctx.pdf_url}")
        return await self.pdf_download_with_browser(ctx.pdf_url, ctx.filepath)

    async def _strategy_brightdata_pdf_url(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.pdf_url:
            return False
        max_retries = 3 if "researchgate.net" in (ctx.pdf_url or "").lower() else 1
        for attempt in range(max_retries):
            if attempt > 0:
                self.logger.info(f"ResearchGate 重试 ({attempt + 1}/{max_retries}): {ctx.pdf_url[:60]}...")
                await asyncio.sleep(2)
            else:
                self.logger.info(f"尝试BrightData下载pdf_url: {ctx.pdf_url}")

            if await self.download_with_solver(
                ctx.pdf_url,
                ctx.filepath,
                title=ctx.title,
                authors=ctx.authors,
                year=ctx.year,
            ):
                return True
        return False

    async def _strategy_ssrn(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.pdf_url:
            return False
        self.logger.info(f"检测到 SSRN pdf_url，使用页面查找 PDF: {ctx.pdf_url}")
        return await self.find_and_download_pdf_with_browser(
            ctx.pdf_url,
            ctx.filepath,
            title=ctx.title,
            authors=ctx.authors,
            year=ctx.year,
            source=ctx.source,
        )

    async def _strategy_semantic_doi(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.doi:
            return False
        doi_str = str(ctx.doi).strip()
        doi_url = doi_str if doi_str.lower().startswith("http") else f"https://doi.org/{doi_str}"
        self.logger.info(f"Semantic Scholar：尝试通过 DOI 解析下载: {doi_url}")
        return await self.find_and_download_pdf_with_browser(
            doi_url,
            ctx.filepath,
            title=ctx.title,
            authors=ctx.authors,
            year=ctx.year,
            source=ctx.source,
        )

    async def _strategy_browser_url(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.url:
            return False
        if ctx.is_semantic_source and "semanticscholar.org" in ctx.url.lower():
            self.logger.info("检测到Semantic Scholar链接，跳过页面抓取，继续其他策略")
            return False
        self.logger.info(f"尝试从论文页面查找PDF: {ctx.url}")
        return await self.find_and_download_pdf_with_browser(
            ctx.url,
            ctx.filepath,
            title=ctx.title,
            authors=ctx.authors,
            year=ctx.year,
            source=ctx.source,
        )

    async def _strategy_scihub(self, ctx: _PaperDownloadContext) -> bool:
        scihub_input = ctx.doi or ctx.url
        if not scihub_input:
            return False

        annas_md5 = ctx.annas_md5
        if not annas_md5:
            try:
                annas_query = ctx.doi or ctx.title
                if annas_query:
                    md5 = await self._annas_search_md5(annas_query)
                    if md5:
                        annas_md5 = md5
                        ctx.paper["annas_md5"] = md5
                        self.logger.info(f"Anna's Archive 找到 MD5: {md5}（已保存，供后续手动重试）")
                    else:
                        self.logger.info("Anna's Archive 未匹配 MD5，跳过 Sci-Hub 以节省资源")
                else:
                    self.logger.info("缺少 DOI/标题，无法查询 MD5，跳过 Sci-Hub 以节省资源")
            except Exception as e:
                self.logger.warning(f"Anna's MD5 搜索失败: {e}")
                annas_md5 = None

        if not annas_md5:
            self.logger.info("已跳过 Sci-Hub（需要 MD5 才尝试）")
            return False

        self.logger.info(f"尝试BrightData + Sci-Hub下载: {scihub_input}")
        return await self.download_with_sci_hub(
            scihub_input,
            ctx.filepath,
            title=ctx.title,
            authors=ctx.authors,
            year=ctx.year,
        )

    async def _strategy_academia(self, ctx: _PaperDownloadContext) -> bool:
        if not ctx.pdf_url:
            return False
        short_url = ctx.pdf_url[:60] + "..." if len(ctx.pdf_url) > 60 else ctx.pdf_url
        self.logger.info("Academia 兜底：尝试通过浏览器下载: %s", short_url)
        return await self.pdf_download_with_browser(ctx.pdf_url, ctx.filepath)

    async def _download_paper(self, paper: Dict, semaphore: asyncio.Semaphore, stats: Dict[str, int]) -> None:
        """下载单篇论文"""
        title = paper.get('title', '').strip()
        authors = paper.get('authors', [])
        year = paper.get('year')
        url = paper.get('url') or paper.get('link')
        pdf_url = paper.get('pdf_url') or paper.get('pdf_link')
        doi = paper.get('doi')
        source = paper.get('source', '')
        
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
            async with semaphore:
                # ==========================================================
                # ==== 预印本及开放获取平台 URL 智能推断（绕过渲染，提速核心）====
                # ==========================================================
                if not pdf_url and url:
                    url_lower = url.lower()
                    
                    # 1. arXiv: /abs/ -> /pdf/ + .pdf
                    if 'arxiv.org/abs/' in url_lower:
                        arxiv_match = re.search(r'arxiv\.org/abs/([a-z\-]+/\d+(?:v\d+)?|\d+\.\d+(?:v\d+)?)', url_lower)
                        if arxiv_match:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_match.group(1)}.pdf"
                            self.logger.info(f"[智能推断] arXiv PDF直链: {pdf_url}")
                            
                    # 2. bioRxiv / medRxiv: 加上 .full.pdf 后缀
                    elif 'biorxiv.org/content/' in url_lower or 'medrxiv.org/content/' in url_lower:
                        if not url_lower.endswith('.full.pdf') and not url_lower.endswith('.pdf'):
                            clean_url = url.split('?')[0].rstrip('/')
                            pdf_url = f"{clean_url}.full.pdf"
                            self.logger.info(f"[智能推断] bioRxiv/medRxiv PDF直链: {pdf_url}")
                            
                    # 3. OSF Preprints: 提取 ID 后接 /download
                    elif 'osf.io/' in url_lower and '/download' not in url_lower:
                        osf_match = re.search(r'osf\.io/(?:preprints/[^/]+/)?([a-zA-Z0-9]{4,6})/?$', url.split('?')[0])
                        if osf_match:
                            pdf_url = f"https://osf.io/{osf_match.group(1)}/download"
                            self.logger.info(f"[智能推断] OSF Preprints PDF直链: {pdf_url}")
                            
                    # 4. Research Square: 直接加 .pdf 后缀
                    elif 'researchsquare.com/article/' in url_lower:
                        if not url_lower.endswith('.pdf'):
                            clean_url = url.split('?')[0].rstrip('/')
                            pdf_url = f"{clean_url}.pdf"
                            self.logger.info(f"[智能推断] Research Square PDF直链: {pdf_url}")
                # ==========================================================
                
                # 确保至少有URL或DOI
                if not url and not pdf_url and not doi:
                    self.logger.warning(f"跳过没有URL、PDF链接和DOI的论文: {title}")
                    stats['skipped'] += 1
                    return
                
                annas_md5 = paper.get('annas_md5')
                source_flags = self._infer_source_flags(source, url=url, pdf_url=pdf_url)
                is_semantic_source = source_flags["is_semantic_source"]
                is_semantic_link = bool(url and "semanticscholar.org" in url.lower())
                if is_semantic_source and is_semantic_link:
                    self.logger.info("Semantic Scholar 来源链接不作为下载入口，将仅使用 pdf_url/doi")
                    url = None
                is_academia_pdf = bool(pdf_url and source_flags["is_academia_source"])
                is_likely_pdf = bool(pdf_url and self._is_likely_pdf_url(pdf_url))
                is_ssrn = bool(pdf_url and source_flags["is_ssrn_source"])
                force_pdf_attempt = bool(is_semantic_source and pdf_url and not is_likely_pdf)
                
                # 确保会话池已就绪
                if not self._session_pool_initialized:
                    self.logger.error("浏览器会话池未初始化")
                    stats['failed'] += 1
                    return

                ctx = _PaperDownloadContext(
                    paper=paper,
                    filepath=filepath,
                    title=title,
                    authors=authors,
                    year=year,
                    url=url,
                    pdf_url=pdf_url,
                    doi=doi,
                    source=source,
                    is_likely_pdf=is_likely_pdf,
                    is_ssrn=is_ssrn,
                    is_academia_pdf=is_academia_pdf,
                    is_semantic_source=is_semantic_source,
                    force_pdf_attempt=force_pdf_attempt,
                    annas_md5=annas_md5,
                )

                strategies: List[Tuple[str, Any]] = []
                if ctx.pdf_url and not ctx.is_academia_pdf:
                    self.logger.debug(f"检测到 pdf_url: {ctx.pdf_url}")
                    self.logger.debug(
                        f"URL 识别结果: is_pdf={ctx.is_likely_pdf}, is_ssrn={ctx.is_ssrn}"
                    )
                    if ctx.is_likely_pdf or ctx.force_pdf_attempt:
                        strategies.extend([
                            ("direct_pdf_url", self._strategy_direct_pdf),
                            ("browser_pdf_url", self._strategy_browser_pdf_url),
                            ("brightdata_pdf_url", self._strategy_brightdata_pdf_url),
                        ])
                    if ctx.is_ssrn and not ctx.is_likely_pdf:
                        strategies.append(("ssrn", self._strategy_ssrn))
                if ctx.is_semantic_source and ctx.doi:
                    strategies.append(("semantic_doi", self._strategy_semantic_doi))
                if ctx.url:
                    strategies.append(("browser_url", self._strategy_browser_url))
                if ctx.doi or ctx.url:
                    strategies.append(("scihub", self._strategy_scihub))
                if ctx.pdf_url and ctx.is_academia_pdf:
                    strategies.append(("academia", self._strategy_academia))

                for name, strategy_fn in strategies:
                    try:
                        if await strategy_fn(ctx):
                            stats['success'] += 1
                            return
                    except Exception as e:
                        self.logger.warning("策略 %s 失败: %s", name, e)

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
                except Exception:
                    pass
                return {}
            except Exception as e:
                self.logger.warning(f"加载下载历史失败: {str(e)}，创建新的历史记录")
                return {}
        else:
            self.logger.info("下载历史文件不存在，创建新的历史记录")
            return {}

    def save_download_history(self, blocking: bool = True):
        """安全地保存下载历史到文件（同步版本，兼容信号处理器和非async上下文）。"""
        temp_file = None
        try:
            temp_file = f"{self.history_file}.{uuid.uuid4().hex[:8]}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_history.copy(), f, ensure_ascii=False, indent=2)

            if sys.platform == 'win32' and os.path.exists(self.history_file):
                os.remove(self.history_file)

            os.rename(temp_file, self.history_file)
            temp_file = None
            self.logger.debug(f"下载历史已保存到 {self.history_file}")
        except Exception as e:
            self.logger.error(f"保存下载历史失败: {str(e)}")
            if temp_file is not None and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

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

    def _record_download_success(self, filepath: str, url: str,
                                 title: Optional[str] = None,
                                 authors: Optional[List[str]] = None,
                                 method: str = "unknown",
                                 md5: Optional[str] = None):
        """统一记录下载成功到历史（去重逻辑 + MD5 索引 + 写入历史）"""
        try:
            if not md5:
                md5 = self._compute_file_md5(filepath)

            if md5:
                existing_path = self.file_md5_index.get(md5)
                if existing_path and existing_path != filepath and os.path.exists(existing_path):
                    self.logger.info(f"检测到重复文件 (MD5: {md5[:8]}...): {filepath}")
                else:
                    self.file_md5_index[md5] = filepath
                    self._save_md5_index()

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
                'method': method,
                'md5': md5
            }
        except Exception as e:
            self.logger.warning(f"记录下载成功失败（不影响已保存的文件）: {e}")

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

    async def _check_and_save_all(
        self,
        sniffed_pdf_buffers: list,
        captured_downloads: list,
        filepath: str,
        task_download_dir: str,
        reason: str,
        wait_time: float = 0.0,
    ) -> bool:
        """统一落盘接管：内存嗅探流优先，其次 Download 事件，可选轮询等待。"""
        start_t = time.time()
        while True:
            # 1. Memory-sniffed stream (zero-wait, highest priority)
            while sniffed_pdf_buffers:
                s_url, s_body = sniffed_pdf_buffers.pop(0)
                self.logger.info(f"[网络嗅探-优先] 提取内存流落盘 ({reason}): {s_url[:80]}")
                temp = os.path.join(task_download_dir, "sniffed_memory.pdf")
                try:
                    with open(temp, 'wb') as f:
                        f.write(s_body)
                    if self.is_valid_pdf_file(temp):
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        shutil.move(temp, filepath)
                        self.logger.info(f"⚡ [极速落盘] 内存流落盘成功 -> {os.path.basename(filepath)}")
                        return True
                    else:
                        try:
                            os.remove(temp)
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.debug(f"内存流落盘失败: {e}")
            # 2. Download event
            while captured_downloads:
                dl = captured_downloads.pop(0)
                if await self._save_and_validate_download(dl, filepath, task_download_dir, f"event_{reason}"):
                    return True
            if wait_time <= 0 or (time.time() - start_t) >= wait_time:
                break
            await asyncio.sleep(0.3)
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

    async def initialize(self):
        """初始化 HTTP 会话和浏览器会话池（单上下文多页面模式）"""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=min(self.download_timeout, 60))
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        if not self._session_pool_initialized:
            await self._init_session_pool()

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
                                   session_downloads_dir: Optional[str] = None,
                                   initial_main_files: Optional[set] = None,
                                   task_start_time: Optional[float] = None,
                                   sniffed_pdf_buffers: Optional[list] = None,
                                   captured_downloads: Optional[list] = None,
                                   request_context: Optional[DownloadRequestContext] = None) -> bool:
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
            session_downloads_dir: 会话沙盒目录（隐式下载落点）
            initial_main_files: 主目录初始文件集合（用于回退过滤）
            task_start_time: 任务开始时间（用于回退过滤）
        
        Returns:
            bool: 是否成功下载
        """
        visited_urls = set()
        last_action = None
        ctx = request_context or self._get_request_context()
        if max_total_time is None:
            max_total_time = self.timeouts.get("smart_loop_total_timeout", 60)
        start_time = time.time()
        action_history = []  # 新增：记录操作历史
        failed_selectors = []  # 新增：记录失败的选择器
        self._start_phase(
            "smart_loop",
            base_seconds=max_total_time,
            request_context=ctx,
            url=url,
        )
        
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
                                    session_downloads_dir=session_downloads_dir,
                                    initial_main_files=initial_main_files,
                                    task_start_time=task_start_time
                                ):
                                    # 成功！记录这次成功
                                    await self.experience_store.record_success(domain, url, learned_actions)
                                    return True
                    except Exception as e:
                        self.logger.debug(f"[智能循环] 经验操作执行失败: {e}")
            self.logger.info(f"[智能循环] 经验操作未成功，继续常规流程")
        # === 经验查询结束 ===
        
        for iteration in range(max_iterations):
            # 检查总时间限制
            elapsed = time.time() - start_time
            if elapsed > max_total_time or (ctx and ctx.budget_controller is not None and ctx.budget_controller.should_abort()):
                self.logger.warning(f"[智能循环] 超过总时间限制 {max_total_time}秒，退出")
                self._record_failure(
                    self._classify_failure(download_stalled=True),
                    request_context=ctx,
                    iteration=iteration + 1,
                )
                return False
            
            current_url = page.url
            self.logger.info(f"[智能循环] 迭代 {iteration + 1}/{max_iterations}, "
                            f"已用时 {elapsed:.1f}s, URL: {current_url[:60]}...")
            
            # 防止URL死循环
            url_key = current_url.split('#')[0].split('?')[0]  # 去掉锚点和查询参数
            if url_key in visited_urls and iteration > 0:
                self.logger.debug(f"[智能循环] URL重复访问，尝试不同策略")
            visited_urls.add(url_key)

            # Drain any buffers captured by the sniffer or download handler before expensive page analysis
            if (sniffed_pdf_buffers is not None or captured_downloads is not None) and task_download_dir:
                if await self._check_and_save_all(
                    sniffed_pdf_buffers or [], captured_downloads or [],
                    filepath, task_download_dir, f"smart_loop_iter_{iteration}"
                ):
                    self.logger.info("[智能循环] 嗅探/下载缓冲区中已有 PDF，极速落盘成功")
                    return True

            # 1. 分析当前页面
            analysis = await self.page_analyzer.analyze(page)
            self._emit_progress_event(
                "page_analyzed",
                request_context=ctx,
                blocker_count=len(analysis.blockers),
                actionable_count=len(analysis.actionable_elements),
                content_type=analysis.content_type.value,
            )
            
            # === 阶段3：应用经验库选择器加成 ===
            if analysis.actionable_elements:
                self._grant_progress(
                    "actionable_found",
                    extra_seconds=8,
                    request_context=ctx,
                    allow_count=2,
                    actionable_count=len(analysis.actionable_elements),
                )
                analysis.actionable_elements = self.experience_store.boost_element_scores(
                    analysis.actionable_elements
                )
            # === 加成结束 ===

            # === 单轮 LLM 候选精排（启发式粗排后）===
            if analysis.actionable_elements and not (ctx and ctx.metadata.get("disable_assist_llm")):
                try:
                    # 传入已按启发式分数排好序的 top_n，避免在 rerank_candidates 内部重排导致
                    # 索引错位；idx 搜索覆盖全列表以防元素出现在 top_n 之外。
                    top_n = min(5, len(analysis.actionable_elements))
                    assist_mode = ((ctx.metadata.get("assist_llm_mode") if ctx else None) or "ultra-lite").strip().lower()
                    rerank_result = await self.pdf_extractor.rerank_candidates(
                        [self.pdf_extractor.to_candidate(item) for item in analysis.actionable_elements[:top_n]],
                        top_n=top_n,
                        mode=assist_mode,
                        provider=(ctx.llm_provider_override if ctx else None),
                        model_override=(ctx.llm_model_override if ctx else None),
                    )
                    if rerank_result and rerank_result.get("best_candidate_id"):
                        best_id = str(rerank_result.get("best_candidate_id"))
                        idx = next(
                            (
                                i
                                for i, item in enumerate(analysis.actionable_elements)
                                if (getattr(item, "element_id", "") or item.selector) == best_id
                            ),
                            -1,
                        )
                        if idx >= 0:
                            best = analysis.actionable_elements[idx]
                            analysis.actionable_elements = [best] + [
                                e for i, e in enumerate(analysis.actionable_elements) if i != idx
                            ]
                            self.logger.info(
                                "[智能循环] LLM精排选择 id=%s confidence=%.2f",
                                best_id[:80],
                                float(rerank_result.get("confidence", 0.0) or 0.0),
                            )
                except Exception as rerank_e:
                    self.logger.debug(f"[智能循环] 单轮 LLM 精排异常，保持启发式顺序: {rerank_e}")
            
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
                    session_downloads_dir=session_downloads_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time,
                    request_context=ctx,
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
                            self._grant_progress(
                                "high_confidence_element_clicked",
                                extra_seconds=10,
                                request_context=ctx,
                                allow_count=2,
                                selector=best.selector[:80],
                            )
                            await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                            if await self._wait_for_download_complete(
                                filepath,
                                timeout=self.timeouts.get("download_complete_timeout", 20),
                                task_download_dir=task_download_dir,
                                session_downloads_dir=session_downloads_dir,
                                initial_main_files=initial_main_files,
                                task_start_time=task_start_time,
                                request_context=ctx,
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
                        self._grant_progress(
                            "high_confidence_element_clicked",
                            extra_seconds=10,
                            request_context=ctx,
                            allow_count=2,
                            selector=best.selector[:80],
                        )
                        
                        # 优先检查：如果最终文件已存在且有效，直接返回成功
                        if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                            self.logger.info(f"[智能循环] 下载已完成（点击后文件直接可用）")
                            await self.experience_store.record_success(domain, url, action_history)
                            return True
                        
                        await asyncio.sleep(self.timeouts.get("page_stable_wait", 2))
                        
                        # 再次检查最终文件
                        if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                            self.logger.info(f"[智能循环] 下载已完成（等待后文件可用）")
                            await self.experience_store.record_success(domain, url, action_history)
                            return True
                        
                        # 检查是否触发了下载
                        if await self._wait_for_download_complete(
                            filepath,
                            timeout=self.timeouts.get("download_complete_timeout", 20),
                            task_download_dir=task_download_dir,
                            session_downloads_dir=session_downloads_dir,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time,
                            request_context=ctx,
                        ):
                            # === 阶段3：记录成功经验 ===
                            await self.experience_store.record_success(domain, url, action_history)
                            # === 记录结束 ===
                            return True
                        
                        # 检查页面是否变化（可能跳转到新页面）
                        if page.url != current_url:
                            self.logger.info(f"[智能循环] 页面已跳转到: {page.url[:70]}...")
                            self._grant_progress(
                                "navigation_advanced",
                                extra_seconds=10,
                                request_context=ctx,
                                allow_count=2,
                                url=page.url[:120],
                            )
                            continue  # 重新分析新页面
                    else:
                        self.logger.info(f"[智能循环] 点击失败，尝试下一个元素")
                        if ctx:
                            ctx.failed_selectors.append(best.selector)
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
                        session_downloads_dir=session_downloads_dir,
                        initial_main_files=initial_main_files,
                        task_start_time=task_start_time,
                        request_context=ctx,
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
        
        # 统一策略：仅保留“启发式 + 单轮候选精排”，不再执行页面级动作型 LLM 兜底。
        
        # 记录失败的选择器
        if failed_selectors:
            await self.experience_store.record_failure(domain, failed_selectors)
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
        request_context = self._get_request_context()
        for blocker in blockers:
            self._emit_progress_event(
                "blocker_detected",
                request_context=request_context,
                blocker=blocker.type.value,
            )
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
                    self._record_failure(
                        self._classify_failure(blocker=blocker),
                        request_context=request_context,
                    )
                    await asyncio.sleep(self.timeouts.get("cloudflare_retry_wait", 10))
                    # 检查是否自动通过了
                    new_title = await page.title()
                    if '请稍候' not in new_title and 'moment' not in new_title.lower():
                        return 'solved'
                    return 'continue'  # 让循环继续尝试
            
            elif blocker.type == BlockerType.CAPTCHA:
                self.logger.info("[阻断处理] 检测到验证码，尝试通用验证码路由求解...")
                success, _ = await self.solve_captcha_if_needed(
                    page,
                    task_download_dir=task_download_dir,
                    request_context=request_context,
                )
                if not success:
                    self._record_failure(
                        self._classify_failure(blocker=blocker),
                        request_context=request_context,
                    )
                return 'solved' if success else 'continue'
            
            elif blocker.type in [BlockerType.PAYWALL, BlockerType.LOGIN_REQUIRED, BlockerType.GEO_BLOCKED]:
                self.logger.warning(f"[阻断处理] 不可解决的阻断: {blocker.type.value}")
                self._record_failure(
                    self._classify_failure(blocker=blocker),
                    request_context=request_context,
                )
                return 'unsolvable'
            
            elif blocker.type == BlockerType.NOT_FOUND:
                self.logger.error(f"[阻断处理] 页面不存在")
                self._record_failure(
                    self._classify_failure(blocker=blocker),
                    request_context=request_context,
                )
                return 'unsolvable'
            
            elif blocker.type == BlockerType.SERVER_ERROR:
                # 服务器错误可能是误报，继续尝试
                self.logger.warning(f"[阻断处理] 检测到服务器错误（可能误报），继续尝试")
                return 'continue'
            
            elif blocker.type == BlockerType.RATE_LIMITED:
                self.logger.warning(f"[阻断处理] 频率限制，等待30秒...")
                self._record_failure(
                    self._classify_failure(blocker=blocker),
                    request_context=request_context,
                )
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
            
            if locator is None and element.selector and 'data-dl-uid' in element.selector:
                self.logger.info(f"[点击] 常规定位失败，启动 Shadow DOM 穿透打捞: {element.selector}")
                uid_match = re.search(r'data-dl-uid="([^"]+)"', element.selector)
                if uid_match:
                    uid_str = uid_match.group(1)
                    try:
                        async with page.expect_download(timeout=self.timeouts.get("download_event_timeout", 15) * 1000) as dl_info:
                            clicked_via_js = await page.evaluate('''async (targetUid) => {
                                function findElementByUid(root, uid) {
                                    if (root.querySelector && root.querySelector(`[data-dl-uid="${uid}"]`))
                                        return root.querySelector(`[data-dl-uid="${uid}"]`);
                                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
                                    let node;
                                    while (node = walker.nextNode()) {
                                        if (node.shadowRoot) {
                                            let found = findElementByUid(node.shadowRoot, uid);
                                            if (found) return found;
                                        }
                                    }
                                    return null;
                                }
                                const el = findElementByUid(document.body, targetUid);
                                if (el) {
                                    const ev = new MouseEvent('click', { view: window, bubbles: true, cancelable: true, clientX: window.innerWidth/2, clientY: window.innerHeight/2 });
                                    el.dispatchEvent(ev);
                                    if (el.firstElementChild) el.firstElementChild.dispatchEvent(ev);
                                    el.click();
                                    return true;
                                }
                                return false;
                            }''', uid_str)
                        if clicked_via_js:
                            download = await dl_info.value
                            local_task_dir = task_download_dir or self._create_task_download_dir()[0]
                            if await self._save_and_validate_download(download, filepath, local_task_dir, "shadow_dom_click"):
                                self.logger.info("[点击] Shadow DOM 穿透触发下载成功")
                                return True
                    except Exception as e:
                        self.logger.debug(f"[点击] Shadow DOM 探测异常: {e}")
            
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

            # 物理清除可能遮挡按钮的 Cookie 声明/订阅悬浮窗/模态遮罩
            try:
                await page.evaluate("""() => {
                    const selectors = [
                        '[id*="onetrust"]', '[class*="cookie-banner"]', '[class*="cookie-consent"]',
                        '[class*="cookie-notice"]', '[class*="cookie-bar"]',
                        '[id*="backdrop"]', '[id*="CybotCookiebotDialog"]',
                        '#CybotCookiebotDialog'
                    ];
                    document.querySelectorAll(selectors.join(',')).forEach(el => {
                        const style = window.getComputedStyle(el);
                        const zIndex = parseInt(style.zIndex, 10);
                        if (style.position === 'fixed' || style.position === 'absolute' || zIndex > 50) {
                            el.remove();
                        }
                    });
                }""")
            except Exception:
                pass

            # 防弹窗降维打击：防止智能循环点击时弹出新窗口导致 expect_download 漏接
            try:
                target_attr = await locator.get_attribute("target")
                if target_attr == "_blank":
                    await locator.evaluate("el => { el.removeAttribute('target'); el.target = '_self'; }")
            except Exception:
                pass
            
            # 4. 用短窗口探测是否触发下载
            try:
                async with page.expect_download(
                    timeout=self.timeouts.get("download_event_timeout", 15) * 1000
                ) as download_info:
                    try:
                        await locator.click(
                            timeout=self.timeouts.get("button_click_timeout", 5) * 1000,
                            force=use_force
                        )
                    except Exception as click_e:
                        error_msg_click = str(click_e).lower()
                        if 'intercepted' in error_msg_click or 'outside' in error_msg_click or 'not visible' in error_msg_click:
                            self.logger.info("[点击] Playwright 点击受阻，尝试底层 MouseEvent 派发兜底")
                            await locator.evaluate("""el => {
                                const ev = new MouseEvent('click', {
                                    view: window, bubbles: true, cancelable: true,
                                    clientX: window.innerWidth / 2, clientY: window.innerHeight / 2
                                });
                                el.dispatchEvent(ev);
                                if (el.firstElementChild) el.firstElementChild.dispatchEvent(ev);
                                el.click();
                            }""")
                        else:
                            raise click_e
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
                                page.remove_listener("download", _on_direct_download)
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
                                          session_downloads_dir: Optional[str] = None,
                                          initial_main_files: Optional[set] = None,
                                          task_start_time: Optional[float] = None,
                                          request_context: Optional[DownloadRequestContext] = None) -> bool:
        """等待下载完成：同时轮询并发隔离沙盒，彻底摒弃主目录扫描"""
        start_time = time.time()
        ctx = request_context or self._get_request_context()
        timeout = timeout or self.timeouts.get("download_complete_timeout", 20)
        remaining = self._start_phase(
            "download_materialize",
            base_seconds=timeout,
            request_context=ctx,
            target=os.path.basename(filepath),
        )
        poll_interval = self.timeouts.get("download_poll_interval", 1)
        timeout = max(timeout, remaining)
        growth_limit = int(self.timeouts.get("download_growth_grant_limit", 2))
        observed_sizes: Dict[str, int] = {}

        # 仅扫描本任务专属 task_download_dir，禁止扫描共享目录（避免跨任务抢夺文件）
        safe_dirs = []
        if task_download_dir and os.path.exists(task_download_dir):
            safe_dirs.append(task_download_dir)

        if not safe_dirs:
            return False

        while time.time() - start_time < timeout:
            if ctx and ctx.budget_controller is not None and ctx.budget_controller.should_abort():
                self._record_failure(
                    self._classify_failure(download_stalled=True),
                    request_context=ctx,
                    target=os.path.basename(filepath),
                )
                return False
            if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                self.logger.info(f"[下载等待] 目标文件已就绪: {filepath}")
                self._emit_progress_event("download_materialized", request_context=ctx, target=os.path.basename(filepath))
                return True

            # 同时安全扫描所有沙盒
            for s_dir in safe_dirs:
                try:
                    for f in os.listdir(s_dir):
                        fpath = os.path.join(s_dir, f)
                        if any(f.lower().endswith(ext) for ext in ['.crdownload', '.part', '.download', '.tmp']):
                            try:
                                current_size = os.path.getsize(fpath)
                            except OSError:
                                current_size = 0
                            previous_size = observed_sizes.get(fpath, 0)
                            if current_size > previous_size:
                                observed_sizes[fpath] = current_size
                                self._grant_progress(
                                    "download_file_growing",
                                    extra_seconds=15,
                                    request_context=ctx,
                                    allow_count=growth_limit,
                                    bytes=current_size,
                                    file=os.path.basename(fpath),
                                )
                            continue
                        if os.path.isdir(fpath):
                            continue

                        if self.is_valid_pdf_file(fpath):
                            self.logger.info(f"[下载等待] 在沙盒 {os.path.basename(s_dir)} 发现PDF: {f}")
                            try:
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                                shutil.move(fpath, filepath)
                                self._grant_progress(
                                    "download_materialized",
                                    extra_seconds=8,
                                    request_context=ctx,
                                    allow_count=1,
                                    file=f,
                                )
                                self._emit_progress_event(
                                    "download_materialized",
                                    request_context=ctx,
                                    source_file=f,
                                )
                                return True
                            except Exception as e:
                                self.logger.warning(f"移动文件失败: {e}")
                except Exception:
                    pass

            await asyncio.sleep(poll_interval)
            if ctx and ctx.budget_controller is not None:
                timeout = max(timeout, ctx.budget_controller.remaining_seconds())
        self._record_failure(
            self._classify_failure(download_stalled=True),
            request_context=ctx,
            target=os.path.basename(filepath),
        )
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
                
                // 检查 iframe 元素（通用）
                const iframe = document.querySelector('iframe[src*=".pdf"], iframe[src*="/pdf/"]');
                if (iframe && iframe.src) return iframe.src;
                
                // 检查 object 元素
                const obj = document.querySelector('object[type="application/pdf"], object[data*=".pdf"]');
                if (obj && obj.data) return obj.data;
                
                // 出版商特定 iframe：Elsevier ScienceDirect
                const iframeElsevier = document.querySelector('iframe[src*="pdfft"]');
                if (iframeElsevier && iframeElsevier.src) return iframeElsevier.src;
                // Wiley pdfdirect
                const iframeWiley = document.querySelector('iframe[src*="pdfdirect"]');
                if (iframeWiley && iframeWiley.src) return iframeWiley.src;
                // pdf-frame / pdf_frame 阅读器
                const iframePdfFrame = document.querySelector('iframe[src*="pdf-frame"], iframe[src*="pdf_frame"]');
                if (iframePdfFrame && iframePdfFrame.src) return iframePdfFrame.src;
                // IEEE stamp.jsp
                const iframeStamp = document.querySelector('iframe[src*="stamp.jsp"]');
                if (iframeStamp && iframeStamp.src) return iframeStamp.src;
                // viewer + pdf
                const iframeViewer = document.querySelector('iframe[src*="viewer"][src*="pdf"], iframe[src*="PDF"]');
                if (iframeViewer && iframeViewer.src) return iframeViewer.src;
                
                return null;
            }''')
            
            if pdf_url:
                self.logger.info(f"[内联PDF] 发现嵌入的PDF URL: {pdf_url[:70]}...")
                success = await self.download_direct(pdf_url, filepath, page=page)
                if success and self.is_valid_pdf_file(filepath):
                    self.logger.info(f"[内联PDF] 嵌入PDF下载成功")
                    return True
        except Exception as e:
            self.logger.debug(f"[内联PDF] 查找嵌入PDF失败: {e}")
        
        # 方法3: JS fetch 作为最后兜底 (使用原生 FileReader 防大文件内存溢出)
        try:
            pdf_data = await page.evaluate('''async () => {
                try {
                    const response = await fetch(location.href, {
                        credentials: 'include',
                        headers: { 'Accept': 'application/pdf' }
                    });
                    if (!response.ok) return null;
                    const blob = await response.blob();
                    if (blob.type === 'application/pdf' || blob.type.includes('pdf') || blob.size > 1000) {
                        return await new Promise((resolve, reject) => {
                            const reader = new FileReader();
                            reader.onloadend = () => resolve(reader.result.split(',')[1]);
                            reader.onerror = reject;
                            reader.readAsDataURL(blob);
                        });
                    }
                } catch (e) { console.error('PDF fetch error:', e); }
                return null;
            }''')
            
            if pdf_data:
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(pdf_data))
                if self.is_valid_pdf_file(filepath):
                    self.logger.info(f"[内联PDF] JS fetch (FileReader) 成功")
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
            supp_kw = [
                'supplementary', 'supplement', 'supplemental', 'supporting', 'appendix',
                'esm', 'moesm', '-sup-', '/suppl/', 'suppl_file', '/doi/suppl/',
                '附件', '附录'
            ]
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
                                           source: Optional[str] = None,
                                           llm_provider: Optional[str] = None,
                                           model_override: Optional[str] = None,
                                           assist_llm_mode: Optional[str] = None,
                                           progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                                           show_browser: Optional[bool] = None,
                                           disable_assist_llm: bool = False,
                                           request_context: Optional[DownloadRequestContext] = None) -> bool:
        """使用浏览器查找并下载PDF文件
        
        查找页面中的PDF下载链接按钮，点击后下载PDF文件
        
        Args:
            url: 论文页面URL
            filepath: 保存路径
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            llm_provider: 可选，本任务使用的 LLM 提供商（覆盖默认）
            model_override: 可选，本任务使用的模型（覆盖提供商默认）
            
        Returns:
            bool: 是否下载成功
        """
        active_context = request_context or self._get_request_context()
        token = None
        restore_state: Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[bool], Optional[Callable[[Dict[str, Any]], None]], Optional[str], Optional[str], Optional[str], Dict[str, Any]]] = None
        if active_context is None:
            active_context = self._build_request_context(
                paper_id=os.path.splitext(os.path.basename(filepath))[0],
                strategy="browser_url",
                title=title or os.path.splitext(os.path.basename(filepath))[0],
                url=url,
                filepath=filepath,
                show_browser=show_browser,
                llm_provider=llm_provider,
                model_override=model_override,
                assist_llm_mode=assist_llm_mode,
                progress_callback=progress_callback,
            )
            if disable_assist_llm:
                active_context.metadata["disable_assist_llm"] = True
            active_context.metadata["assist_llm_mode"] = (assist_llm_mode or "ultra-lite")
        else:
            restore_state = (
                active_context.strategy,
                active_context.url,
                active_context.filepath,
                active_context.show_browser_override,
                active_context.progress_callback,
                active_context.llm_provider_override,
                active_context.llm_model_override,
                active_context.title,
                dict(active_context.metadata),
            )
            active_context.strategy = active_context.strategy or "browser_url"
            active_context.url = url
            active_context.filepath = filepath
            active_context.title = title or active_context.title
            if show_browser is not None:
                active_context.show_browser_override = show_browser
            if progress_callback is not None:
                active_context.progress_callback = progress_callback
            if llm_provider or model_override:
                active_context.llm_provider_override = llm_provider
                active_context.llm_model_override = model_override
            if assist_llm_mode:
                active_context.assist_llm_mode = assist_llm_mode
                active_context.metadata["assist_llm_mode"] = assist_llm_mode
            elif "assist_llm_mode" not in active_context.metadata:
                active_context.metadata["assist_llm_mode"] = active_context.assist_llm_mode or "ultra-lite"
            if disable_assist_llm:
                active_context.metadata["disable_assist_llm"] = True
            else:
                active_context.metadata.pop("disable_assist_llm", None)
        token = _REQUEST_CONTEXT.set(active_context)
        try:
            self._emit_progress_event("request_context_ready", request_context=active_context, url=url[:120])
            return await self._find_and_download_pdf_with_browser_impl(
                url, filepath, title=title, authors=authors, year=year, source=source
            )
        finally:
            if restore_state is not None and active_context is not None:
                (
                    active_context.strategy,
                    active_context.url,
                    active_context.filepath,
                    active_context.show_browser_override,
                    active_context.progress_callback,
                    active_context.llm_provider_override,
                    active_context.llm_model_override,
                    active_context.title,
                    active_context.metadata,
                ) = restore_state
            if token is not None:
                _REQUEST_CONTEXT.reset(token)

    async def _find_and_download_pdf_with_browser_impl(self, url: str, filepath: str,
                                           title: Optional[str] = None,
                                           authors: Optional[List[str]] = None,
                                           year: Optional[int] = None,
                                           source: Optional[str] = None) -> bool:
        """Implementation of find_and_download_pdf_with_browser using the active request context."""
        self.logger.info(f"尝试查找并下载PDF: {url}")
        downloads_dir = os.path.dirname(filepath)
        os.makedirs(downloads_dir, exist_ok=True)
        
        task_download_dir, task_start_time, initial_main_files = self._create_task_download_dir()
        
        finished = False
        page = None
        new_page = None  # 用于跟踪可能打开的新页面
        new_page_download_handler = None
        session = None  # 并发池会话，finally 中归还
        session_downloads_dir = None  # 会话沙盒目录，供 salvage/wait 使用
        task_popups = []  # 本次任务衍生的弹窗，finally 中统一关闭防泄漏
        
        try:
            # 从并发池获取隔离会话
            session = await self._acquire_session()
            if session is None:
                self.logger.error("无法从会话池获取浏览器上下文")
                return False
            context = session['context']
            session_downloads_dir = session.get('session_downloads_dir')
            # 任务前清场：彻底排空当前 Session 沙盒，防止上一任务残留污染（含 Chromium 临时目录）
            if session_downloads_dir and os.path.exists(session_downloads_dir):
                shutil.rmtree(session_downloads_dir, ignore_errors=True)
                os.makedirs(session_downloads_dir, exist_ok=True)
            # 基于隔离 context 创建页面
            page = await context.new_page()
            self._active_pages.add(page)

            # 捕获“非常快的下载”：用列表接收，避免多下载事件覆盖（如 .ris 后跟真 PDF）
            captured_downloads = []
            def _on_download(download):
                self.logger.debug(f"[下载监听] 捕获下载事件: {download.suggested_filename}")
                captured_downloads.append(download)
            page.on("download", _on_download)

            # 监听上下文内所有新页面，防止 target="_blank" 极速弹窗漏接下载；收集衍生弹窗便于结束时统一关闭防泄漏
            def _on_page_created(new_page):
                new_page.on("download", _on_download)
                self._active_pages.add(new_page)
                task_popups.append(new_page)
            page.context.on("page", _on_page_created)

            # 全局 XHR/Fetch 底层流量嗅探（直接截取内存二进制流，破解一次性 Token）
            sniffed_pdf_buffers = []
            async def _on_response_sniff(response):
                try:
                    if response.status == 200 and response.request.method not in ("OPTIONS", "HEAD"):
                        ct = (response.headers.get("content-type") or "").lower()
                        req_url = response.url
                        ru = req_url.lower()
                        if (("application/pdf" in ct or "application/epub" in ct or ru.split('?')[0].endswith(".pdf"))
                                and ru != page.url.lower() and not ru.startswith("data:")):
                            self.logger.info(f"[流量嗅探] 发现后台深层 PDF 响应: {req_url[:80]}")
                            try:
                                body = await response.body()
                                if body and len(body) > 10240:
                                    sniffed_pdf_buffers.append((req_url, body))
                                    self.logger.info(f"[流量嗅探] 成功截获内存二进制流，大小: {len(body)} bytes")
                            except Exception as body_e:
                                self.logger.debug(f"[流量嗅探] 提取 Body 失败(流可能未结束或跨域限制): {body_e}")
                except Exception:
                    pass
            page.on("response", _on_response_sniff)

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

                    self._record_download_success(
                        filepath,
                        url,
                        title=title,
                        authors=authors,
                        method=f"browser:{reason}",
                        md5=md5
                    )

                    finished = True
                    return True
                except Exception as e:
                    self.logger.debug(f"finalize_download 失败({reason}): {e}")
                    return False

            async def _consume_download_event(reason: str) -> bool:
                """消费内存嗅探流（优先）或下载事件队列，直到找到合法的 PDF"""
                nonlocal finished
                # 1. 内存嗅探流（最高优先级，立即消费，无需等待磁盘落盘）
                while sniffed_pdf_buffers:
                    s_url, s_body = sniffed_pdf_buffers.pop(0)
                    self.logger.info(f"[网络嗅探-优先] 提取内存流落盘 ({reason}): {s_url[:80]}")
                    temp = os.path.join(task_download_dir, "sniffed_memory.pdf")
                    try:
                        with open(temp, 'wb') as f:
                            f.write(s_body)
                        if self.is_valid_pdf_file(temp):
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            shutil.move(temp, filepath)
                            self._record_download_success(filepath, url, title=title, authors=authors, method=f"memory_sniffing:{reason}")
                            finished = True
                            self.logger.info(f"⚡ [极速落盘] 内存流落盘成功 -> {os.path.basename(filepath)}")
                            return True
                        else:
                            try:
                                os.remove(temp)
                            except Exception:
                                pass
                    except Exception as e:
                        self.logger.debug(f"内存流落盘失败: {e}")
                # 2. Download 事件队列
                success = False
                while captured_downloads:
                    dl = captured_downloads.pop(0)
                    if await _finalize_download_obj(dl, reason):
                        success = True
                        break
                return success

            async def _consume_download_event_with_wait(reason: str, wait_time: float = 2.0) -> bool:
                """Like _consume_download_event, but polls up to wait_time seconds."""
                start_t = time.time()
                while True:
                    if await _consume_download_event(reason):
                        return True
                    if (time.time() - start_t) >= wait_time:
                        break
                    await asyncio.sleep(0.3)
                return False

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
                """并发安全打捞：仅扫描本任务 task_download_dir，严禁扫描共享目录（避免跨任务抢夺）。"""
                nonlocal finished
                if os.path.exists(filepath) and self.is_valid_pdf_file(filepath):
                    return True
                safe_dirs = [task_download_dir] if task_download_dir else []
                task_candidates = []
                for s_dir in safe_dirs:
                    if not s_dir or not os.path.exists(s_dir):
                        continue
                    try:
                        for name in os.listdir(s_dir):
                            if name.lower().endswith(('.crdownload', '.download', '.part', '.tmp')):
                                continue
                            p = os.path.join(s_dir, name)
                            if os.path.isdir(p):
                                continue
                            if self.is_valid_pdf_file(p):
                                mtime = os.path.getmtime(p)
                                md5 = self._compute_file_md5(p)
                                task_candidates.append((mtime, p, name, md5))
                    except Exception:
                        pass
                if task_candidates:
                    task_candidates.sort(key=lambda x: x[0], reverse=True)
                    _, best_path, best_name, md5 = task_candidates[0]
                    self.logger.info(f"[salvage安全回收] 从隔离沙盒打捞: {best_name}")
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        shutil.move(best_path, filepath)
                        if md5:
                            existing_path = self.file_md5_index.get(md5)
                            if not (existing_path and existing_path != filepath and os.path.exists(existing_path)):
                                self.file_md5_index[md5] = filepath
                                self._save_md5_index()
                        self._record_download_success(
                            filepath,
                            url,
                            title=title,
                            authors=authors,
                            method=f"salvage_isolated:{reason}",
                            md5=md5
                        )
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
            
            # 预热步骤：按平台 key 处理（防惊群 + DOI 入口跳过）
            try:
                await self._warmup_platform_once(
                    page,
                    request_url=url,
                    task_download_dir=task_download_dir,
                    consume_download_event=_consume_download_event,
                )
            except Exception as warmup_e:
                self.logger.warning(f"预热访问失败，继续尝试直接访问: {warmup_e}")
            
            # 检测是否为DOI链接（含出版商 /doi/ 页面，使 ASM 等 preferred_pdf_entrypoints 直通车生效）
            source_flags = self._infer_source_flags(source, url=url)
            is_doi_link = self._is_doi_like_url(url)
            is_semantic_scholar = source_flags["is_semantic_source"]
            is_wiley = source_flags["is_wiley_source"]

            # ScienceDirect (Elsevier) PII 协议级直通车
            if 'sciencedirect.com' in url.lower() and '/pii/' in url.lower():
                try:
                    pii_match = re.search(r'pii/([a-zA-Z0-9]+)', url, re.IGNORECASE)
                    if pii_match:
                        pii = pii_match.group(1)
                        sd_url = f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true"
                        self.logger.info(f"[特攻] ScienceDirect PII 直达: {sd_url[:80]}")
                        try:
                            async with page.expect_download(timeout=15000) as sd_dl_info:
                                await page.goto(sd_url, wait_until="domcontentloaded")
                                await self.solve_cloudflare_if_needed(page, task_download_dir=task_download_dir)
                            if await self._save_and_validate_download(
                                await sd_dl_info.value, filepath, task_download_dir, "elsevier_pii"
                            ):
                                finished = True
                                return True
                        except Exception:
                            if await self.download_direct(sd_url, filepath, page=page):
                                if self.is_valid_pdf_file(filepath):
                                    finished = True
                                    return True
                except Exception as sd_e:
                    self.logger.debug(f"ScienceDirect PII 特攻失败: {sd_e}")

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
                        self._record_download_success(
                            filepath,
                            url,
                            title=title,
                            authors=authors,
                            method="direct_download_url"
                        )
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
                    parsed_wiley = urlparse(url)
                    # 兼容 /doi/full/{doi}、/doi/abs/{doi}、/doi/pdf/{doi}、/doi/epdf/{doi}
                    m = re.search(r'/doi/(?:full|abs|pdf|epdf)/([^?]+)', parsed_wiley.path, flags=re.IGNORECASE)
                    wiley_doi = m.group(1) if m else None
                    if wiley_doi:
                        pdfdirect_url = f"{parsed_wiley.scheme}://{parsed_wiley.netloc}/doi/pdfdirect/{wiley_doi}?download=true"

                        download_timeout_ms = self.timeouts.get("download_event_timeout", 15) * 1000
                        async with page.expect_download(timeout=download_timeout_ms) as download_info:
                            try:
                                await page.goto(pdfdirect_url, wait_until="domcontentloaded")
                            except Exception as goto_err:
                                err_lower = str(goto_err).lower()
                                if "download is starting" in err_lower or "net::err_aborted" in err_lower:
                                    self.logger.info("Wiley pdfdirect 导航触发下载流，正常接收中...")
                                else:
                                    raise
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
                except PlaywrightTimeoutError:
                    # expect_download timed out: no native download event, but sniffer might have captured the body
                    if await _consume_download_event("wiley-pdfdirect-sniffed"):
                        return True
                    self.logger.debug("Wiley pdfdirect: expect_download 超时且嗅探器未截获")
                except Exception as wiley_direct_e:
                    self.logger.warning(f"Wiley pdfdirect 尝试失败（继续走原有流程）: {wiley_direct_e}")
                    # Still try draining in case the download/sniff completed during the exception
                    await asyncio.sleep(0.5)
                    if await _consume_download_event("wiley-pdfdirect-fallback"):
                        return True
            
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
                            self._record_download_success(
                                filepath,
                                url,
                                title=title,
                                authors=authors,
                                method="doi_direct"
                            )
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
                        if cf_success:
                            self._mark_platform_ready(
                                self._resolve_platform_key(url, page.url),
                                reason="doi_navigation",
                            )
                        if cf_file and self.is_valid_pdf_file(cf_file):
                            self.logger.info(f"DOI 导航后 Cloudflare 验证触发下载: {cf_file}")
                            return True
                        if await _consume_download_event("goto-doi-parse"):
                            return True
                        # citation_pdf_url 隐藏通道（相对路径自动转绝对路径）
                        try:
                            citation_pdf = await page.evaluate("""() => {
                                const tags = [
                                    'citation_pdf_url',
                                    'eprints.document_url',
                                    'wkhealth_pdf_url',
                                    'bepress_citation_pdf_url',
                                    'prism.url'
                                ];
                                for (const tag of tags) {
                                    const meta = document.querySelector(`meta[name="${tag}"]`);
                                    if (meta && meta.content && meta.content.toLowerCase().endsWith('.pdf')
                                        && !meta.content.includes('researchgate.net/publication/')) {
                                        return new URL(meta.content, document.baseURI).href;
                                    }
                                }
                                return null;
                            }""")
                            if citation_pdf:
                                self.logger.info(f"[Meta] 发现 citation_pdf_url: {citation_pdf[:100]}")
                                try:
                                    async with page.expect_download(timeout=15000) as meta_dl_info:
                                        try:
                                            await page.goto(citation_pdf, wait_until="domcontentloaded")
                                        except Exception as goto_e:
                                            if any(x in str(goto_e).lower() for x in ["download is starting", "net::err_aborted"]):
                                                self.logger.info("[Meta] citation_pdf 导航触发下载流")
                                            else:
                                                raise
                                    if await self._save_and_validate_download(
                                        await meta_dl_info.value, filepath, task_download_dir, "meta_citation"
                                    ):
                                        self.logger.info("[Meta] citation_pdf_url 导航下载成功")
                                        finished = True
                                        return True
                                except Exception as _nav_e:
                                    self.logger.debug(f"[Meta] 导航未触发下载，降级: {_nav_e}")
                                    ok = await self.download_direct(citation_pdf, filepath, page=page)
                                    if ok and self.is_valid_pdf_file(filepath):
                                        self.logger.info("[Meta] citation_pdf_url 直链下载成功")
                                        finished = True
                                        return True
                        except Exception as _meta_e:
                            self.logger.debug(f"[Meta] citation_pdf_url 探测异常: {_meta_e}")
                else:
                    response = await page.goto(url, wait_until="domcontentloaded")
                    await self._handle_cookie_consent(page, context_label="页面解析")
                    await _log_analysis("导航后")
                    # 导航后立即检测并处理 Cloudflare 验证
                    cf_success, cf_file = await self.solve_cloudflare_if_needed(
                        page, filepath, task_download_dir=task_download_dir
                    )
                    if cf_success:
                        self._mark_platform_ready(
                            self._resolve_platform_key(url, page.url),
                            reason="page_navigation",
                        )
                    if cf_file and self.is_valid_pdf_file(cf_file):
                        self.logger.info(f"导航后 Cloudflare 验证触发下载: {cf_file}")
                        return True
                    if await _consume_download_event_with_wait("goto-article", 1.5):
                        return True
                    # citation_pdf_url 隐藏通道（相对路径自动转绝对路径）
                    try:
                        citation_pdf = await page.evaluate("""() => {
                            const tags = [
                                'citation_pdf_url',
                                'eprints.document_url',
                                'wkhealth_pdf_url',
                                'bepress_citation_pdf_url',
                                'prism.url'
                            ];
                            for (const tag of tags) {
                                const meta = document.querySelector(`meta[name="${tag}"]`);
                                if (meta && meta.content && meta.content.toLowerCase().endsWith('.pdf')
                                    && !meta.content.includes('researchgate.net/publication/')) {
                                    return new URL(meta.content, document.baseURI).href;
                                }
                            }
                            return null;
                        }""")
                        if citation_pdf:
                            self.logger.info(f"[Meta] 发现 citation_pdf_url: {citation_pdf[:100]}")
                            try:
                                async with page.expect_download(timeout=15000) as meta_dl_info:
                                    try:
                                        await page.goto(citation_pdf, wait_until="domcontentloaded")
                                    except Exception as goto_e:
                                        if any(x in str(goto_e).lower() for x in ["download is starting", "net::err_aborted"]):
                                            self.logger.info("[Meta] citation_pdf 导航触发下载流")
                                        else:
                                            raise
                                if await self._save_and_validate_download(
                                    await meta_dl_info.value, filepath, task_download_dir, "meta_citation"
                                ):
                                    self.logger.info("[Meta] citation_pdf_url 导航下载成功")
                                    finished = True
                                    return True
                            except Exception as _nav_e:
                                self.logger.debug(f"[Meta] 导航未触发下载，降级: {_nav_e}")
                                ok = await self.download_direct(citation_pdf, filepath, page=page)
                                if ok and self.is_valid_pdf_file(filepath):
                                    self.logger.info("[Meta] citation_pdf_url 直链下载成功")
                                    finished = True
                                    return True
                    except Exception as _meta_e:
                        self.logger.debug(f"[Meta] citation_pdf_url 探测异常: {_meta_e}")
            except Exception as e:
                error_msg = str(e)
                if "Download is starting" in error_msg:
                    # 这意味着URL是直接的PDF下载链接
                    self.logger.info("检测到URL直接触发下载，尝试捕获")
                    direct_download_triggered = True
                    
                    # 【修复】等待 download 事件触发（异步延迟问题）
                    # page.goto() 抛出异常比 download 事件触发更快，需要等待
                    for wait_attempt in range(10):  # 最多等待 2 秒
                        if captured_downloads:
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

                preferred_pdf_urls = self._build_preferred_pdf_entrypoints(page.url)
                for preferred_pdf_url in preferred_pdf_urls:
                    current_url_no_query = page.url.split("?", 1)[0].rstrip("/")
                    if preferred_pdf_url.rstrip("/") == current_url_no_query:
                        continue
                    try:
                        self.logger.info(f"优先尝试正文 PDF 入口: {preferred_pdf_url}")
                        async with page.expect_download(timeout=15000) as preferred_dl_info:
                            try:
                                await page.goto(preferred_pdf_url, wait_until="domcontentloaded")
                            except Exception as goto_e:
                                if any(x in str(goto_e).lower() for x in ["download is starting", "net::err_aborted"]):
                                    self.logger.info("正文 PDF 入口导航触发下载流")
                                else:
                                    raise
                            await self._handle_cookie_consent(page, context_label="正文PDF入口")
                        if await self._save_and_validate_download(
                            await preferred_dl_info.value, filepath, task_download_dir, "preferred_pdf_entrypoint"
                        ):
                            self.logger.info("正文 PDF 入口下载成功")
                            return True
                    except Exception as preferred_e:
                        self.logger.debug(f"正文 PDF 入口未直接触发下载，继续页面流程: {preferred_e}")
                
                # ========== 阶段2：智能循环优先 ==========
                smart_success = await self._smart_download_loop(
                    page=page, filepath=filepath, url=url,
                    title=title, authors=authors, max_iterations=3,
                    task_download_dir=task_download_dir,
                    session_downloads_dir=session_downloads_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time,
                    sniffed_pdf_buffers=sniffed_pdf_buffers,
                    captured_downloads=captured_downloads,
                )
                if smart_success:
                    self._record_download_success(
                        filepath,
                        url,
                        title=title,
                        authors=authors,
                        method="smart_loop"
                    )
                    return True
                self.logger.info(f"[兜底] 智能循环未成功，继续传统流程...")
                # ========== 阶段2结束 ==========
            else:
                await asyncio.sleep(2)
                await self._handle_cookie_consent(page, context_label="页面稳定后")
                if await _consume_download_event_with_wait("after-stabilize", 1.5):
                    return True
                
                # ========== 阶段2：智能循环优先（非DOI链接）==========
                smart_success = await self._smart_download_loop(
                    page=page, filepath=filepath, url=url,
                    title=title, authors=authors, max_iterations=3,
                    task_download_dir=task_download_dir,
                    session_downloads_dir=session_downloads_dir,
                    initial_main_files=initial_main_files,
                    task_start_time=task_start_time,
                    sniffed_pdf_buffers=sniffed_pdf_buffers,
                    captured_downloads=captured_downloads,
                )
                if smart_success:
                    self._record_download_success(
                        filepath,
                        url,
                        title=title,
                        authors=authors,
                        method="smart_loop"
                    )
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
            if cf_success:
                self._mark_platform_ready(
                    self._resolve_platform_key(url, page.url),
                    reason="browser_find_flow",
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
            for site_id, site_config in self.multi_step_sites.items():
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
                            if await _consume_download_event("fast-download-fallback"):
                                return True

                            # Wiley 专属兜底：进入 /doi/(e)pdf 后，构造 pdfdirect 直达下载并捕获下载事件
                            try:
                                current_url = page.url
                                if (site_id == 'wiley'
                                    and 'onlinelibrary.wiley.com' in current_url.lower()
                                    and ('/doi/pdf/' in current_url.lower() or '/doi/epdf/' in current_url.lower())):
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
                    'supplementary', 'supplement', 'supplemental', 'supporting', 'appendix',
                    'esm', 'moesm', 'mediaobject', '-sup-', '-si-', '/si/',
                    '/suppl/', 'suppl_file', '/doi/suppl/',
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
                        # 强制改为 _self，避免极速弹窗导致下载漏接
                        await valid_pdf_link_element.evaluate("el => { el.removeAttribute('target'); el.target = '_self'; }")
                        self.logger.info("已将 target='_blank' 强制转换为 '_self'，防弹窗漏接")
                        will_open_new_page = False
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

                try:
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
                                if cf_success:
                                    self._mark_platform_ready(
                                        self._resolve_platform_key(url, new_page.url),
                                        reason="new_page_find_flow",
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
                            err_msg = str(e).lower()
                            if ("target" in err_msg and "closed" in err_msg) and captured_downloads:
                                self.logger.info("弹窗已关闭，使用上下文监听到的下载句柄兜底")
                                if await _consume_download_event("popup_closed_fallback"):
                                    finished = True
                                    return True
                            self.logger.error(f"处理新页面时出错: {str(e)}")

                    if not finished:
                        await _try_external_pdf_fallback("popup失败", href_attr)
                except Exception as popup_e:
                    err_msg = str(popup_e).lower()
                    if ("target" in err_msg and "closed" in err_msg) and captured_downloads:
                        self.logger.info("弹窗已关闭，使用上下文监听到的下载句柄兜底")
                        if await _consume_download_event("popup_closed_fallback"):
                            finished = True
                            return True
                    finished = False
                    self.logger.error(f"处理新页面时出错: {popup_e}")
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
                        except Exception:
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
                if sniffed_pdf_buffers:
                    for s_url, s_body in sniffed_pdf_buffers:
                        self.logger.info(f"[网络嗅探] 释放内存截获的 PDF 数据: {s_url[:80]}")
                        temp_filepath = os.path.join(task_download_dir, "sniffed_memory.pdf")
                        with open(temp_filepath, 'wb') as f:
                            f.write(s_body)
                        if self.is_valid_pdf_file(temp_filepath):
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            shutil.move(temp_filepath, filepath)
                            self._record_download_success(filepath, url, title=title, authors=authors, method="memory_sniffing")
                            self.logger.info("[网络嗅探] 内存二进制流兜底落盘成功!")
                            finished = True
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
                    if captured_downloads:
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
                            session_downloads_dir=session_downloads_dir if session else None,
                            initial_main_files=initial_main_files,
                            task_start_time=task_start_time
                        )
            except Exception as wait_e:
                self.logger.debug(f"等待下载完成失败（关闭前）: {wait_e}")

            self._cleanup_task_download_dir(task_download_dir)
            
            # 移除上下文级新页面监听，避免泄漏
            if page and page.context:
                try:
                    page.context.remove_listener("page", _on_page_created)
                except Exception:
                    pass

            # 关闭本次任务所有衍生弹窗，防止幽灵窗口导致内存泄漏
            for popup in task_popups:
                try:
                    if not popup.is_closed():
                        await asyncio.wait_for(popup.close(), timeout=2.0)
                    self._active_pages.discard(popup)
                except Exception:
                    pass

            if page:
                try:
                    try:
                        page.remove_listener("download", _on_download)
                    except Exception:
                        pass
                    try:
                        page.remove_listener("response", _on_response_sniff)
                    except Exception:
                        pass
                    await page.close()
                    self._active_pages.discard(page)
                except Exception as e:
                    self.logger.warning(f"关闭页面时出错: {str(e)}")

            # 将隔离会话归还并发池（必须在所有清理完成后最后执行）
            if session:
                try:
                    await self._release_session(session)
                except Exception as release_e:
                    self.logger.warning(f"归还会话失败: {release_e}")
            
        return finished

def setup_signal_handlers(downloader):
    """设置信号处理器以捕获中断信号
    
    Args:
        downloader: PaperDownloader实例
    """
    def signal_handler(sig, frame):
        logger.info("接收到中断信号，保存历史记录...")
        downloader.save_download_history(blocking=False)
        logger.info("历史记录已保存，程序退出")
        sys.exit(0)
    
    # 注册SIGINT (Ctrl+C) 和 SIGTERM (终止信号)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _run_pdf_selector_mock_demo() -> None:
    """Mock demo: print heuristic scores for three representative candidates."""
    demo_logger = get_logger("scholar_downloader.PDFSelectorDemo")
    extractor = PDFExtractor(demo_logger)
    candidates = [
        Candidate(
            element_id="candidate_a_nature_sidebar_fixed",
            text="open pdf",
            href="https://www.nature.com/articles/s41586-2026-00001.pdf",
            is_fixed=True,
            rect_top=88.0,
            position_ratio=0.12,
            ancestor_features="sidebar article-tools header",
            is_visible=True,
        ),
        Candidate(
            element_id="candidate_b_elsevier_related",
            text="pdf",
            href="https://www.sciencedirect.com/science/article/pii/S0000000000000012/pdfft?isDTMRedir=true",
            is_fixed=False,
            rect_top=980.0,
            position_ratio=0.92,
            ancestor_features="sidebar related recommendation footer",
            is_visible=True,
        ),
        Candidate(
            element_id="candidate_c_supplementary",
            text="supplementary material (2.1 mb)",
            href="https://example.org/article/supplementary.pdf",
            is_fixed=False,
            rect_top=760.0,
            position_ratio=0.68,
            ancestor_features="main article-content",
            is_visible=True,
        ),
    ]
    print("\n=== PDF Selector Mock Demo ===")
    for item in candidates:
        score = extractor._calculate_score(item)
        print(f"{item.element_id}: score={score:.1f} text='{item.text}' href='{item.href[:80]}'")

async def main():
    """主函数，处理命令行参数并执行下载任务"""
    parser = argparse.ArgumentParser(description="从JSON搜索结果下载论文PDF文件")
    parser.add_argument("input", nargs="?", default="", help="输入JSON文件路径或包含JSON文件的目录")
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
    parser.add_argument(
        "--mock-pdf-selector-demo",
        action="store_true",
        help="运行 PDF 候选打分 Mock 演示并退出",
    )
    

    args = parser.parse_args()
    if args.mock_pdf_selector_demo:
        _run_pdf_selector_mock_demo()
        return 0
    if not args.input:
        parser.error("input is required unless --mock-pdf-selector-demo is set")

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
                finally:
                    if downloader is not None:
                        downloader.stop_auto_save()
                        downloader.save_download_history()
            
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
            finally:
                if downloader is not None:
                    downloader.stop_auto_save()
                    downloader.save_download_history()
        
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
    if sys.platform == 'win32':
        # Windows平台需要特殊处理asyncio事件循环
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)