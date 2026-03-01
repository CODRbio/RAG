"""
统一网络搜索聚合器

整合 Tavily、Google Scholar、Google、Semantic Scholar、NCBI 五种搜索来源，按来源权重去重。
输出格式与 hybrid_retriever 兼容，可直接送入 reranker。

使用方法:
---------
from src.retrieval.unified_web_search import unified_web_searcher

# 异步搜索（默认启用所有已配置的来源）
results = await unified_web_searcher.search("deep learning")

# 指定来源（前端手动选择，绝对尊重）
results = await unified_web_searcher.search("machine learning", providers=["scholar", "tavily", "semantic"])

# 智能自动路由（优化器选最优引擎）
results = await unified_web_searcher.search("deep sea cold seep", providers=["auto"])

# 同步搜索
results = unified_web_searcher.search_sync("deep learning")

来源权重（去重时优先保留权重高的）:
- ncbi:    0.98 (生物医学专库，最高)
- scholar: 1.0  (Google Scholar，最高)
- semantic: 0.95
- web/tavily: 0.8
- google: 0.6
"""

import asyncio
import atexit
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.log import get_logger
from src.retrieval.smart_query_optimizer import RoutingPlan, get_smart_query_optimizer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Persistent background event loop for sync→async bridging.
#
# Using asyncio.run() in worker threads creates a temporary loop that is
# destroyed immediately after the coroutine finishes.  Any lingering SSL
# connections (aiohttp, httpx) still reference the closed loop, causing
# "Event loop is closed" / "Bad file descriptor" cascading errors.
#
# Instead we keep a single daemon-thread loop alive for the process lifetime
# and schedule coroutines on it via run_coroutine_threadsafe().
# ---------------------------------------------------------------------------
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread] = None
_bg_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily create) a persistent background event loop."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed():
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(
                target=_bg_loop.run_forever,
                name="unified-search-bg-loop",
                daemon=True,
            )
            _bg_thread.start()
        return _bg_loop


def _shutdown_bg_loop() -> None:
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is not None and not _bg_loop.is_closed():
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)
            if _bg_thread is not None:
                _bg_thread.join(timeout=5)


atexit.register(_shutdown_bg_loop)

_SCHOLAR_MAX_TOKENS = 12
_SCHOLAR_MAX_CHARS = 120
_SEMANTIC_MAX_CHARS = 300
_SEMANTIC_MAX_TOKENS = 20


def _sanitize_scholar_query(q: str) -> str:
    """Truncate and clean a query for keyword-based engines (Scholar/Google).

    Scholar performs poorly with long sentence-like queries.  This keeps at
    most ``_SCHOLAR_MAX_TOKENS`` keyword tokens and strips sentence-ending
    punctuation.  Handles both English and CJK input.
    """
    q = (q or "").strip()
    if not q:
        return q
    q = re.sub(r"[.!?。！？；;]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) <= _SCHOLAR_MAX_CHARS:
        return q
    en_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+/]*", q)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    if en_tokens or zh_tokens:
        parts = []
        if zh_tokens:
            parts.append(" ".join(zh_tokens[:_SCHOLAR_MAX_TOKENS // 2]))
        if en_tokens:
            budget = _SCHOLAR_MAX_TOKENS - len(zh_tokens[:_SCHOLAR_MAX_TOKENS // 2])
            parts.append(" ".join(en_tokens[:max(budget, 4)]))
        short = " ".join(parts).strip()
        if short:
            return short
    return q[:_SCHOLAR_MAX_CHARS]


def _sanitize_semantic_query(q: str) -> str:
    """Truncate and clean a query for Semantic Scholar.

    Semantic Scholar works best with concise keyword queries rather than
    long natural language sentences. This keeps at most
    ``_SEMANTIC_MAX_TOKENS`` tokens and strips sentence-ending punctuation.
    """
    q = (q or "").strip()
    if not q:
        return q
    # Remove sentence-ending punctuation and normalize whitespace
    q = re.sub(r"[.!?。！？；;]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) <= _SEMANTIC_MAX_CHARS:
        return q
    # Extract keyword tokens (same pattern as scholar)
    en_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\+/]*", q)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", q)
    if en_tokens or zh_tokens:
        parts = []
        if zh_tokens:
            parts.append(" ".join(zh_tokens[:_SEMANTIC_MAX_TOKENS // 2]))
        if en_tokens:
            budget = _SEMANTIC_MAX_TOKENS - len(zh_tokens[:_SEMANTIC_MAX_TOKENS // 2])
            parts.append(" ".join(en_tokens[:max(budget, 6)]))
        short = " ".join(parts).strip()
        if short:
            return short
    return q[:_SEMANTIC_MAX_CHARS]


# 来源权重（去重时保留权重高的）
SOURCE_WEIGHTS = {
    "ncbi": 0.98,             # NCBI PubMed - 生物医学专库
    "scholar": 1.0,           # Google Scholar - 最高
    "semantic_snippet": 0.96, # Semantic Scholar 正文片段（体文比摘要更丰富，略高于 abstract）
    "semantic": 0.95,         # Semantic Scholar abstract
    "semantic_bulk": 0.85,    # Semantic Scholar bulk（布尔托底，无相关性排序）
    "web": 0.8,               # Tavily
    "tavily": 0.8,            # Tavily（别名）
    "serpapi_scholar": 0.98,  # SerpAPI Google Scholar
    "serpapi_google": 0.55,   # SerpAPI Google Web
    "google": 0.6,            # Google 普通搜索
}


def _get_source_weight(source: str) -> float:
    """获取来源权重"""
    return SOURCE_WEIGHTS.get(source, 0.5)


def _merge_and_dedup(all_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并并去重搜索结果
    
    规则：
    - 按 URL 去重
    - 同一 URL 被多个来源返回时，保留权重高的来源版本
    - 不做排序（由后续 reranker 处理）
    
    Args:
        all_hits: 所有来源的搜索结果
    
    Returns:
        去重后的结果列表
    """
    # url -> (hit, source_weight)
    seen_urls: Dict[str, Tuple[Dict, float]] = {}
    # 无 URL 的结果单独收集
    no_url_hits: List[Dict] = []
    
    for hit in all_hits:
        metadata = hit.get("metadata", {}) or {}
        url = (metadata.get("url") or "").strip()
        source = metadata.get("source", "")
        weight = _get_source_weight(source)
        
        if not url:
            no_url_hits.append(hit)
            continue
        
        # URL 去重：保留权重高的
        if url not in seen_urls or weight > seen_urls[url][1]:
            seen_urls[url] = (hit, weight)
    
    # 合并结果（先有 URL 的，再无 URL 的）
    results = [item[0] for item in seen_urls.values()]
    results.extend(no_url_hits)
    
    return results


@dataclass
class UnifiedWebSearcher:
    """
    统一网络搜索聚合器

    整合 Tavily、Google Scholar、Google、Semantic Scholar、NCBI 五种搜索来源。
    """

    _tavily_searcher: Any = field(default=None, repr=False)
    _google_searcher: Any = field(default=None, repr=False)
    _serpapi_searcher: Any = field(default=None, repr=False)
    _semantic_searcher: Any = field(default=None, repr=False)
    _ncbi_searcher: Any = field(default=None, repr=False)
    _rr_position: int = field(default=0, repr=False)

    def __post_init__(self):
        # 延迟加载搜索器
        self._content_fetcher = None
    
    def _get_content_fetcher(self):
        """获取 WebContentFetcher（延迟加载）"""
        if self._content_fetcher is None:
            try:
                from src.retrieval.web_content_fetcher import WebContentFetcher
                self._content_fetcher = WebContentFetcher.from_settings()
            except Exception as e:
                logger.warning(f"WebContentFetcher not available: {e}")
                self._content_fetcher = False
        return self._content_fetcher if self._content_fetcher else None

    def _get_tavily_searcher(self):
        """获取 Tavily 搜索器（延迟加载）"""
        if self._tavily_searcher is None:
            try:
                from src.retrieval.web_search import TavilySearcher
                self._tavily_searcher = TavilySearcher()
            except Exception as e:
                logger.warning(f"Tavily searcher not available: {e}")
                self._tavily_searcher = False  # 标记为不可用
        return self._tavily_searcher if self._tavily_searcher else None
    
    def _get_google_searcher(self):
        """获取 Google 搜索器（延迟加载）"""
        if self._google_searcher is None:
            try:
                from src.retrieval.google_search import GoogleSearcher
                self._google_searcher = GoogleSearcher()
            except Exception as e:
                logger.warning(f"Google searcher not available: {e}")
                self._google_searcher = False  # 标记为不可用
        return self._google_searcher if self._google_searcher else None

    def _get_serpapi_searcher(self):
        """获取 SerpAPI 搜索器（延迟加载）"""
        if self._serpapi_searcher is None:
            try:
                from src.retrieval.serpapi_search import SerpAPISearcher
                self._serpapi_searcher = SerpAPISearcher()
            except Exception as e:
                logger.warning(f"SerpAPI searcher not available: {e}")
                self._serpapi_searcher = False
        return self._serpapi_searcher if self._serpapi_searcher else None

    def _get_semantic_searcher(self):
        """获取 Semantic Scholar 搜索器（延迟加载）"""
        if self._semantic_searcher is None:
            try:
                from src.retrieval.semantic_scholar import semantic_scholar_searcher
                self._semantic_searcher = semantic_scholar_searcher
            except Exception as e:
                logger.warning(f"Semantic Scholar searcher not available: {e}")
                self._semantic_searcher = False
        return self._semantic_searcher if self._semantic_searcher else None

    def _get_ncbi_searcher(self):
        """获取 NCBI PubMed 搜索器（延迟加载，无需 API Key）"""
        if self._ncbi_searcher is None:
            try:
                from src.retrieval.ncbi_search import get_ncbi_searcher
                self._ncbi_searcher = get_ncbi_searcher()
            except Exception as e:
                logger.warning(f"NCBI searcher not available: {e}")
                self._ncbi_searcher = False
        return self._ncbi_searcher if self._ncbi_searcher else None

    def _resolve_providers(self, providers: Optional[List[str]] = None) -> List[str]:
        """
        解析要使用的搜索来源。

        - providers=None 或含 "auto"：自动检测所有已启用的来源（交由优化器路由）
        - 其他显式列表：前端手动选择，绝对尊重，不允许优化器增减
        """
        if providers is not None:
            cleaned = [str(p).strip().lower() for p in providers if str(p).strip()]
            # 不含 "auto"：严格尊重前端选择
            if "auto" not in cleaned:
                return cleaned

        # auto 或 None：检测所有可用来源，作为优化器的候选池
        result = []

        # NCBI（免费 API，默认启用；可通过配置 ncbi.enabled=false 关闭）
        try:
            from config.settings import settings as _s
            ncbi_enabled = bool(getattr(getattr(_s, "ncbi", None), "enabled", True))
        except Exception:
            ncbi_enabled = True
        ncbi = self._get_ncbi_searcher()
        if ncbi and ncbi_enabled:
            result.append("ncbi")

        # Tavily
        tavily = self._get_tavily_searcher()
        if tavily and getattr(tavily, "enabled", False):
            result.append("tavily")

        # Google Scholar / Google
        google = self._get_google_searcher()
        if google:
            if getattr(google, "scholar_enabled", False):
                result.append("scholar")
            if getattr(google, "google_enabled", False):
                result.append("google")

        # Semantic Scholar
        semantic = self._get_semantic_searcher()
        if semantic and getattr(semantic, "enabled", False):
            result.append("semantic")

        # SerpAPI
        serpapi = self._get_serpapi_searcher()
        if serpapi and getattr(serpapi, "enabled", False):
            result.append("serpapi")

        return result
    
    async def search(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        source_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_results_per_provider: int = 5,
        use_query_expansion: Optional[bool] = None,
        use_query_optimizer: Optional[bool] = None,
        query_optimizer_max_queries: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
        use_content_fetcher: Optional[str] = None,
        llm_client: Optional[Any] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        queries_per_provider: Optional[Dict[str, List[str]]] = None,
        semantic_query_map: Optional[Dict[str, str]] = None,
        serpapi_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        异步搜索多个来源并合并去重
        
        Args:
            query: 搜索查询（同时作为 default_query 兜底）
            providers: 要使用的来源列表 ["scholar", "tavily", "google", "semantic"]，
                      None 表示使用所有已启用的来源
            source_configs: 每个来源的独立配置 {provider_id: {topK: int, threshold: float}}
            max_results_per_provider: 每个来源的最大结果数（当 source_configs 未指定时使用）
            use_query_expansion: 是否使用查询扩展（仅 Tavily 支持）
            use_query_optimizer: 是否启用智能查询优化器
            query_optimizer_max_queries: 优化器每个来源生成的最大查询数
            llm_client: LLMManager 客户端，用于 Lazy Fetching 的 LLM 预判
            queries_per_provider: 调用方预构建的 per-provider 查询，
                      优先级高于 default_query 但低于 SmartQueryOptimizer。
                      当 optimizer 关闭时直接使用；optimizer 开启时忽略。
        
        Returns:
            合并去重后的结果列表
        """
        # 检测是否允许智能自动路由（providers=None 或含 "auto"）
        raw_providers = providers
        is_auto_route = raw_providers is None or (
            raw_providers is not None
            and "auto" in [p.strip().lower() for p in raw_providers if p]
        )
        providers = self._resolve_providers(raw_providers)

        if not providers:
            return []

        t0 = time.perf_counter()
        logger.info(
            "[retrieval] unified_web_search start query_len=%d providers=%s max_per_provider=%d",
            len(query), providers, max_results_per_provider,
        )
        logger.info(
            f"统一搜索: query={query!r}, providers={providers}, auto_route={is_auto_route}"
        )

        source_configs = source_configs or {}

        optimizer_enabled = use_query_optimizer
        if optimizer_enabled is None:
            optimizer_enabled = bool(
                getattr(getattr(settings, "web_search", None), "enable_query_optimizer", True)
            )

        smart_optimizer = get_smart_query_optimizer()
        use_smart = optimizer_enabled and smart_optimizer.enabled

        all_hits: List[Dict[str, Any]] = []

        # ── 代价感知路由模式（auto） ────────────────────────────────────────────
        if is_auto_route and use_smart:
            logger.info("[retrieval] model_call query_optimizer get_routing_plan")
            plan: RoutingPlan = smart_optimizer.get_routing_plan(
                query,
                providers,
                max_queries_per_provider=query_optimizer_max_queries,
                llm_provider=llm_provider,
                model_override=model_override,
            )
            logger.info(
                f"路由计划执行: primary={plan.primary}, fallback={plan.fallback}, "
                f"is_fresh={plan.is_fresh}"
            )

            # 执行 primary 引擎
            primary_hits = await self._run_providers(
                plan.primary,
                plan.queries,
                query,
                source_configs,
                max_results_per_provider,
                year_start=year_start,
                year_end=year_end,
                semantic_query_map=semantic_query_map,
                serpapi_ratio=serpapi_ratio,
                use_query_expansion=use_query_expansion,
                llm_provider=llm_provider,
            )
            primary_unique_hits = _merge_and_dedup(primary_hits)
            all_hits.extend(primary_unique_hits)

            # 结果不足时自动启动 fallback 引擎
            if len(primary_unique_hits) < plan.min_results and plan.fallback:
                logger.info(
                    f"Primary 结果不足 ({len(primary_unique_hits)}/{plan.min_results})，"
                    f"启动 fallback: {plan.fallback}"
                )
                fallback_hits = await self._run_providers(
                    plan.fallback,
                    plan.queries,
                    query,
                    source_configs,
                    max_results_per_provider,
                    year_start=year_start,
                    year_end=year_end,
                    semantic_query_map=semantic_query_map,
                    serpapi_ratio=serpapi_ratio,
                    use_query_expansion=use_query_expansion,
                    llm_provider=llm_provider,
                )
                all_hits.extend(fallback_hits)
                logger.info(f"Fallback 补充 {len(fallback_hits)} 条，合计 {len(all_hits)} 条")

        # ── 普通模式（前端手动指定引擎）──────────────────────────────────────────
        else:
            resolved_qpp: Dict[str, List[str]] = {}
            if use_smart:
                logger.info("[retrieval] model_call query_optimizer optimize")
                resolved_qpp = smart_optimizer.optimize(
                    query,
                    providers,
                    max_queries_per_provider=query_optimizer_max_queries,
                    llm_provider=llm_provider,
                    model_override=model_override,
                    auto_route=False,
                )
                logger.info(f"Smart optimizer 生成多组查询: {list(resolved_qpp.keys())}")
            elif queries_per_provider:
                resolved_qpp = queries_per_provider
                logger.info(f"使用调用方预构建查询: {list(resolved_qpp.keys())}")

            all_hits = await self._run_providers(
                providers,
                resolved_qpp,
                query,
                source_configs,
                max_results_per_provider,
                year_start=year_start,
                year_end=year_end,
                semantic_query_map=semantic_query_map,
                serpapi_ratio=serpapi_ratio,
                use_query_expansion=use_query_expansion,
                llm_provider=llm_provider,
            )

        # 合并去重
        merged = _merge_and_dedup(all_hits)
        logger.info(f"统一搜索完成: 合并前 {len(all_hits)} 条, 去重后 {len(merged)} 条")

        # 全文抓取增强（UI 优先：请求显式传参时以 UI 为准，未传时由 config 决定，便于命令行）
        #   'force' / True  → 硬强制，全量抓取，不做 LLM 预判
        #   'off' / False  → 跳过
        #   'auto'          → 智能模式（UI 显式选择）：只要 fetcher 存在就执行
        #   None / 未传    → 无 UI 场景（如 CLI）：由 config content_fetcher.enabled 决定
        fetcher = self._get_content_fetcher()
        if use_content_fetcher == "off" or use_content_fetcher is False:
            do_enrich = False
        elif use_content_fetcher == "force" or use_content_fetcher is True:
            do_enrich = fetcher is not None
        elif use_content_fetcher == "auto":
            do_enrich = fetcher is not None
        else:
            # None / 未传：命令行等无 UI 场景，以 config 为准
            do_enrich = fetcher is not None and getattr(fetcher, "enabled", False)

        if do_enrich and fetcher:
            try:
                logger.info("[retrieval] content_fetcher start hits=%d", len(merged))
                merged = await fetcher.enrich_results(
                    merged,
                    query=query,
                    llm_client=llm_client,
                    use_content_fetcher=use_content_fetcher,
                )
                full_count = sum(
                    1 for h in merged
                    if (h.get("metadata") or {}).get("content_type") == "full_text"
                )
                sufficient_count = sum(
                    1 for h in merged
                    if (h.get("metadata") or {}).get("content_type") == "snippet_sufficient"
                )
                logger.info(
                    "[retrieval] content_fetcher done full=%d sufficient=%d total=%d",
                    full_count, sufficient_count, len(merged),
                )
                logger.info(
                    f"全文抓取完成: {full_count} 条全文 / {sufficient_count} 条摘要已足够 / "
                    f"{len(merged)} 条总计"
                )
            except Exception as e:
                logger.warning(f"全文抓取失败，使用原始片段: {e}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[retrieval] unified_web_search done total=%d elapsed_ms=%.0f",
            len(merged), elapsed_ms,
        )
        return merged
    
    async def _search_tavily(
        self,
        searcher,
        query: str,
        limit: int,
        use_query_expansion: bool,
        llm_provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Tavily 搜索。扩展时优先使用 llm_provider（UI），无则用 config。"""
        try:
            results = await searcher.async_search(
                query,
                use_query_expansion=use_query_expansion,
                max_results=limit,
                llm_provider=llm_provider,
            )
            return results[:limit]
        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return []
    
    async def _search_scholar(
        self,
        searcher,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Google Scholar 搜索"""
        try:
            return await searcher.search_scholar(
                query,
                limit=limit,
                year_start=year_start,
                year_end=year_end,
            )
        except Exception as e:
            logger.error(f"Google Scholar 搜索失败: {e}")
            return []
    
    async def _search_google(
        self,
        searcher,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Google 搜索"""
        try:
            return await searcher.search_google(query, limit=limit)
        except Exception as e:
            logger.error(f"Google 搜索失败: {e}")
            return []

    async def _search_serpapi_scholar(
        self,
        searcher,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """SerpAPI Google Scholar 搜索"""
        try:
            return await searcher.search_scholar(
                query,
                limit=limit,
                year_start=year_start,
                year_end=year_end,
            )
        except Exception as e:
            logger.error(f"SerpAPI Scholar 搜索失败: {e}")
            return []

    async def _search_serpapi_google(
        self,
        searcher,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """SerpAPI Google 搜索"""
        try:
            return await searcher.search_google(query, limit=limit)
        except Exception as e:
            logger.error(f"SerpAPI Google 搜索失败: {e}")
            return []

    async def _search_scholar_batch(
        self,
        searcher,
        queries: List[str],
        limit_per_query: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Google Scholar 批量搜索（串行执行）"""
        try:
            logger.info(f"Scholar 批量搜索：{len(queries)} 个查询，每个最多 {limit_per_query} 条")
            return await searcher.search_scholar_batch(
                queries,
                limit_per_query=limit_per_query,
                year_start=year_start,
                year_end=year_end,
            )
        except Exception as e:
            logger.error(f"Google Scholar 批量搜索失败: {e}")
            return []

    async def _search_google_batch(
        self,
        searcher,
        queries: List[str],
        limit_per_query: int,
    ) -> List[Dict[str, Any]]:
        """Google 批量搜索（串行执行）"""
        try:
            logger.info(f"Google 批量搜索：{len(queries)} 个查询，每个最多 {limit_per_query} 条")
            return await searcher.search_google_batch(queries, limit_per_query=limit_per_query)
        except Exception as e:
            logger.error(f"Google 批量搜索失败: {e}")
            return []

    async def _search_semantic(
        self,
        searcher,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        semantic_query_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar 三模式搜索：

        1. relevance (/paper/search) + snippet (/snippet/search) 并发执行 —
           两者互补：relevance 提供摘要，snippet 提供正文片段。
        2. relevance 无结果时，启动 bulk (/paper/search/bulk) 布尔托底搜索。
        上层 _merge_and_dedup 按 URL 去重，semantic_snippet 权重略高于 semantic，
        同一篇论文的正文片段会覆盖其摘要版本。
        """
        try:
            query_map = semantic_query_map or {}
            relevance_query = str(query_map.get("relevance_query") or query).strip() or query
            bulk_query = str(query_map.get("bulk_query") or query).strip() or query
            relevance_results, snippet_results = await asyncio.gather(
                searcher.search(relevance_query, limit=limit, year_start=year_start, year_end=year_end),
                searcher.search_snippets(relevance_query, limit=limit, year_start=year_start, year_end=year_end),
                return_exceptions=True,
            )

            hits: List[Dict[str, Any]] = []

            relevance_count = 0
            if isinstance(relevance_results, list):
                relevance_count = len(relevance_results)
                hits.extend(relevance_results)
            else:
                logger.warning("Semantic Scholar relevance 搜索失败: %s", relevance_results)

            snippet_count = 0
            if isinstance(snippet_results, list):
                snippet_count = len(snippet_results)
                hits.extend(snippet_results)
            else:
                logger.warning("Semantic Scholar snippet 搜索失败: %s", snippet_results)

            logger.info(
                "Semantic Scholar 结果: relevance=%d, snippet=%d",
                relevance_count, snippet_count,
            )

            # relevance + snippet 都无结果时，才用 bulk 布尔托底（按引用量排序）。
            # 托底场景下两路贡献均为零，故将 limit 翻倍以维持整体结果量。
            if relevance_count == 0 and snippet_count == 0:
                logger.info(
                    "Semantic Scholar relevance+snippet 均无结果，启动 bulk 托底（limit×2=%d）: %r",
                    limit * 2, bulk_query,
                )
                bulk_results = await searcher.search_bulk(
                    bulk_query, limit=limit * 2, year_start=year_start, year_end=year_end
                )
                hits.extend(bulk_results)

            return hits
        except Exception as e:
            logger.error("Semantic Scholar 搜索失败: %s", e)
            return []

    async def _search_ncbi(
        self,
        searcher,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """NCBI PubMed 搜索"""
        try:
            return await searcher.search(query, limit=limit, year_start=year_start, year_end=year_end)
        except Exception as e:
            logger.error(f"NCBI 搜索失败: {e}")
            return []

    async def _run_providers(
        self,
        providers: List[str],
        queries_per_provider: Dict[str, List[str]],
        default_query: str,
        source_configs: Dict[str, Any],
        max_results_per_provider: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        semantic_query_map: Optional[Dict[str, str]] = None,
        serpapi_ratio: Optional[float] = None,
        use_query_expansion: Optional[bool] = None,
        llm_provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        并发执行指定引擎列表，返回所有命中结果（未去重）。

        - providers: 要执行的引擎列表
        - queries_per_provider: {engine: [query, ...]}；缺失时回退到 default_query
        - use_query_expansion / llm_provider: 仅 Tavily 单查询时生效（多查询由 optimizer 提供，不扩展）
        - Scholar/Google 走批量浏览器方法（串行，防风控）
        - 其他引擎全并发
        """
        if not providers:
            return []

        t_run = time.perf_counter()
        logger.info(
            "[retrieval] web_search _run_providers start providers=%s query_keys=%s",
            providers, list(queries_per_provider.keys()) if queries_per_provider else [],
        )

        # Chat 时每个 search engine provider 在各组查询下均使用该 provider 的 topK（来自 source_configs）
        def _get_max(p: str) -> int:
            cfg = source_configs.get(p) or {}
            # 兼容前端 topK / 后端 top_k
            return cfg.get("topK") or cfg.get("top_k") or max_results_per_provider

        if source_configs:
            per_provider_k = {p: _get_max(p) for p in providers}
            logger.info("web search per-provider topK (from source_configs): %s", per_provider_k)

        def _queries_for(p: str) -> List[str]:
            qs = queries_per_provider.get(p)
            return qs if qs else [default_query]

        tasks_with_flags: List[tuple] = []
        scholar_queries: List[str] = []
        google_queries: List[str] = []
        serpapi_scholar_queries: List[str] = []
        serpapi_google_queries: List[str] = []
        scholar_max = max_results_per_provider
        google_max = max_results_per_provider
        serpapi_scholar_max = max_results_per_provider
        serpapi_google_max = max_results_per_provider

        # UI selection is authoritative — if a provider is in the list,
        # we only check that the searcher object can be loaded, NOT its
        # config-level "enabled" flag.  The enabled flag is only used by
        # _resolve_providers() for auto-discovery.
        for provider in providers:
            qs = _queries_for(provider)
            pmax = _get_max(provider)

            if provider == "tavily":
                tavily = self._get_tavily_searcher()
                if tavily:
                    # 仅当单查询时使用 Tavily 侧扩展（None 表示由 Tavily config 决定）；多查询由 optimizer 提供，不再扩展
                    expand = use_query_expansion if len(qs) == 1 else False
                    for q in qs:
                        tasks_with_flags.append(
                            (self._search_tavily(tavily, q, pmax, expand, llm_provider), False)
                        )
            elif provider == "scholar":
                google = self._get_google_searcher()
                if google:
                    scholar_queries.extend(qs)
                    scholar_max = pmax
                use_serpapi_for_scholar = bool(
                    (source_configs.get("scholar") or {}).get("useSerpapi")
                )
                if use_serpapi_for_scholar:
                    serpapi_check = self._get_serpapi_searcher()
                    if serpapi_check:
                        serpapi_scholar_queries.extend(qs)
                        serpapi_scholar_max = pmax
            elif provider == "google":
                google = self._get_google_searcher()
                if google:
                    google_queries.extend(qs)
                    google_max = pmax
                use_serpapi_for_google = bool(
                    (source_configs.get("google") or {}).get("useSerpapi")
                )
                if use_serpapi_for_google:
                    serpapi_check = self._get_serpapi_searcher()
                    if serpapi_check:
                        serpapi_google_queries.extend(qs)
                        serpapi_google_max = pmax
            elif provider == "serpapi":
                serpapi = self._get_serpapi_searcher()
                if serpapi:
                    serpapi_scholar_queries.extend(qs)
                    serpapi_google_queries.extend(qs)
                    serpapi_scholar_max = pmax
                    serpapi_google_max = pmax
            elif provider == "semantic":
                semantic = self._get_semantic_searcher()
                if semantic:
                    raw_semantic = qs
                    semantic_queries = [_sanitize_semantic_query(q) for q in qs]
                    semantic_queries = [q for q in semantic_queries if q]
                    if raw_semantic != semantic_queries:
                        logger.info(
                            "Semantic query sanitized: %s → %s", raw_semantic, semantic_queries
                        )
                    for q in semantic_queries:
                        tasks_with_flags.append(
                            (
                                self._search_semantic(
                                    semantic,
                                    q,
                                    pmax,
                                    year_start=year_start,
                                    year_end=year_end,
                                    semantic_query_map=semantic_query_map,
                                ),
                                False,
                            )
                        )
            elif provider == "ncbi":
                ncbi = self._get_ncbi_searcher()
                if ncbi:
                    for q in qs:
                        tasks_with_flags.append(
                            (
                                self._search_ncbi(
                                    ncbi,
                                    q,
                                    pmax,
                                    year_start=year_start,
                                    year_end=year_end,
                                ),
                                False,
                            )
                        )

        def _sanitize_queries(raw_queries: List[str], label: str) -> List[str]:
            sanitized = [_sanitize_scholar_query(q) for q in raw_queries]
            sanitized = [q for q in sanitized if q]
            if raw_queries != sanitized:
                logger.info("%s query sanitized: %s → %s", label, raw_queries, sanitized)
            return sanitized

        def _merge_preserve_order(primary: List[str], secondary: List[str]) -> List[str]:
            merged: List[str] = []
            seen = set()
            for q in primary + secondary:
                if q and q not in seen:
                    merged.append(q)
                    seen.add(q)
            return merged

        scholar_queries = _sanitize_queries(scholar_queries, "Scholar")
        google_queries = _sanitize_queries(google_queries, "Google")
        serpapi_scholar_queries = _sanitize_queries(serpapi_scholar_queries, "SerpAPI Scholar")
        serpapi_google_queries = _sanitize_queries(serpapi_google_queries, "SerpAPI Google")

        gsearcher = self._get_google_searcher()
        serpapi_searcher = self._get_serpapi_searcher()

        use_rr_scholar = bool(
            gsearcher and serpapi_searcher and scholar_queries and serpapi_scholar_queries
        )
        use_rr_google = bool(
            gsearcher and serpapi_searcher and google_queries and serpapi_google_queries
        )

        # ── SerpAPI 轮询比例（离散档位）──────────────────────────────────────────
        # 每个档位对应 (n_serp, cycle)，含义：每 cycle 次查询中有 n_serp 次走 SerpAPI。
        # 例：(2, 3) = SSP SSP … → 0.67；(1, 4) = SPPP SPPP … → 0.25
        _RATIO_PATTERNS: List[Tuple[float, int, int]] = [
            (0.0,  0, 1),   # 全部 Playwright
            (0.25, 1, 4),   # SPPP
            (0.33, 1, 3),   # SPP
            (0.50, 1, 2),   # SP
            (0.67, 2, 3),   # SSP
            (0.75, 3, 4),   # SSSP
            (1.0,  1, 1),   # 全部 SerpAPI
        ]

        def _snap_ratio(r: float) -> Tuple[int, int]:
            """Snap r (0-1) to nearest discrete step and return (n_serp, cycle)."""
            best = min(_RATIO_PATTERNS, key=lambda t: abs(t[0] - r))
            return best[1], best[2]

        def _round_robin_split(
            queries: List[str], n_serp: int, cycle: int, offset: int = 0
        ) -> Tuple[List[str], List[str]]:
            """
            Interleave queries between SerpAPI and Playwright in a fixed rhythm.
            Position (offset + i): SerpAPI if (offset + i) % cycle < n_serp, else Playwright.
            """
            if n_serp == 0:
                return [], queries
            if n_serp == cycle:
                return queries, []
            serp = [q for i, q in enumerate(queries) if (offset + i) % cycle < n_serp]
            browser = [q for i, q in enumerate(queries) if (offset + i) % cycle >= n_serp]
            return serp, browser

        raw_ratio = max(0.0, min(1.0, serpapi_ratio)) if serpapi_ratio is not None else 0.5
        _n_serp, _cycle = _snap_ratio(raw_ratio)
        snapped = _n_serp / _cycle if _cycle else 0.0
        if use_rr_scholar or use_rr_google:
            logger.info(
                "SerpAPI round-robin: requested=%.2f snapped=%.2f pattern=%d/%d "
                "rr_scholar=%s rr_google=%s rr_offset=%d",
                raw_ratio, snapped, _n_serp, _cycle, use_rr_scholar, use_rr_google, self._rr_position,
            )

        if use_rr_scholar:
            merged_scholar_queries = _merge_preserve_order(scholar_queries, serpapi_scholar_queries)
            scholar_serp_queries, scholar_browser_queries = _round_robin_split(
                merged_scholar_queries, _n_serp, _cycle, self._rr_position
            )
            self._rr_position += len(merged_scholar_queries)
        else:
            scholar_browser_queries = scholar_queries
            scholar_serp_queries = serpapi_scholar_queries

        if use_rr_google:
            merged_google_queries = _merge_preserve_order(google_queries, serpapi_google_queries)
            google_serp_queries, google_browser_queries = _round_robin_split(
                merged_google_queries, _n_serp, _cycle, self._rr_position
            )
            self._rr_position += len(merged_google_queries)
        else:
            google_browser_queries = google_queries
            google_serp_queries = serpapi_google_queries

        if serpapi_searcher:
            for q in scholar_serp_queries:
                tasks_with_flags.append(
                    (
                        self._search_serpapi_scholar(
                            serpapi_searcher,
                            q,
                            serpapi_scholar_max,
                            year_start=year_start,
                            year_end=year_end,
                        ),
                        False,
                    )
                )
            for q in google_serp_queries:
                tasks_with_flags.append(
                    (self._search_serpapi_google(serpapi_searcher, q, serpapi_google_max), False)
                )

        if scholar_browser_queries:
            if gsearcher and scholar_browser_queries:
                tasks_with_flags.append(
                    (
                        self._search_scholar_batch(
                            gsearcher,
                            scholar_browser_queries,
                            scholar_max,
                            year_start=year_start,
                            year_end=year_end,
                        ),
                        True,
                    )
                )

        if google_browser_queries:
            if gsearcher and google_browser_queries:
                tasks_with_flags.append(
                    (self._search_google_batch(gsearcher, google_browser_queries, google_max), True)
                )

        if not tasks_with_flags:
            return []

        perf = getattr(settings, "perf_unified_web", None)
        max_parallel = getattr(perf, "max_parallel_providers", 3) or 3
        timeout_s = getattr(perf, "per_provider_timeout_seconds", 30) or 30
        browser_max = getattr(perf, "browser_providers_max_parallel", 1) or 1
        max_browser_q = max(len(scholar_browser_queries), len(google_browser_queries), 1)
        browser_timeout = max(timeout_s * max_browser_q, 120)

        sem = asyncio.Semaphore(max_parallel)
        sem_browser = asyncio.Semaphore(browser_max)

        async def _run_one(coro, is_browser: bool):
            if is_browser:
                async with sem_browser:
                    async with sem:
                        return await asyncio.wait_for(coro, timeout=float(browser_timeout))
            else:
                async with sem:
                    return await asyncio.wait_for(coro, timeout=float(timeout_s))

        hits: List[Dict[str, Any]] = []
        wrapped = [_run_one(coro, is_b) for coro, is_b in tasks_with_flags]
        results = await asyncio.gather(*wrapped, return_exceptions=True)
        for r in results:
            if isinstance(r, (asyncio.TimeoutError, asyncio.CancelledError)):
                logger.warning("搜索任务超时或取消")
            elif isinstance(r, Exception):
                logger.error(f"搜索任务出错: {r}")
            elif isinstance(r, list):
                hits.extend(r)
        elapsed_ms = (time.perf_counter() - t_run) * 1000
        logger.info(
            "[retrieval] web_search _run_providers done total_hits=%d elapsed_ms=%.0f",
            len(hits), elapsed_ms,
        )
        return hits

    def search_sync(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        source_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_results_per_provider: int = 5,
        use_query_expansion: Optional[bool] = None,
        use_query_optimizer: Optional[bool] = None,
        query_optimizer_max_queries: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
        use_content_fetcher: Optional[str] = None,
        llm_client: Optional[Any] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        queries_per_provider: Optional[Dict[str, List[str]]] = None,
        semantic_query_map: Optional[Dict[str, str]] = None,
        serpapi_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """同步搜索 — 在持久后台事件循环上运行异步版本。

        All coroutines are scheduled on a single persistent background loop
        via ``run_coroutine_threadsafe``.  This avoids the old pattern of
        ``asyncio.run()`` in worker threads which creates/destroys temporary
        loops and orphans SSL connections (causing "Event loop is closed"
        cascading errors).
        """
        _search_kwargs = dict(
            query=query,
            providers=providers,
            source_configs=source_configs,
            max_results_per_provider=max_results_per_provider,
            use_query_expansion=use_query_expansion,
            use_query_optimizer=use_query_optimizer,
            query_optimizer_max_queries=query_optimizer_max_queries,
            llm_provider=llm_provider,
            model_override=model_override,
            use_content_fetcher=use_content_fetcher,
            llm_client=llm_client,
            year_start=year_start,
            year_end=year_end,
            queries_per_provider=queries_per_provider,
            semantic_query_map=semantic_query_map,
            serpapi_ratio=serpapi_ratio,
        )
        bg = _get_bg_loop()
        future = asyncio.run_coroutine_threadsafe(self.search(**_search_kwargs), bg)
        return future.result()


# 全局单例
unified_web_searcher = UnifiedWebSearcher()
