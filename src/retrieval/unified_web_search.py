"""
统一网络搜索聚合器

整合 Tavily、Google Scholar、Google、Semantic Scholar 四种搜索来源，按来源权重去重。
输出格式与 hybrid_retriever 兼容，可直接送入 reranker。

使用方法:
---------
from src.retrieval.unified_web_search import unified_web_searcher

# 异步搜索（默认启用所有已配置的来源）
results = await unified_web_searcher.search("deep learning")

# 指定来源
results = await unified_web_searcher.search("machine learning", providers=["scholar", "tavily", "semantic"])

# 同步搜索
results = unified_web_searcher.search_sync("deep learning")

来源权重（去重时优先保留权重高的）:
- scholar: 1.0 (最高)
- semantic: 0.95
- web/tavily: 0.8
- google: 0.6
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.log import get_logger
from src.retrieval.smart_query_optimizer import get_smart_query_optimizer

logger = get_logger(__name__)


# 来源权重（去重时保留权重高的）
SOURCE_WEIGHTS = {
    "scholar": 1.0,  # Google Scholar - 最高
    "semantic": 0.95,  # Semantic Scholar
    "web": 0.8,      # Tavily
    "google": 0.6,   # Google 普通搜索
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
    
    整合 Tavily、Google Scholar、Google 三种搜索来源。
    """
    
    _tavily_searcher: Any = field(default=None, repr=False)
    _google_searcher: Any = field(default=None, repr=False)
    _semantic_searcher: Any = field(default=None, repr=False)
    
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
    
    def _resolve_providers(self, providers: Optional[List[str]] = None) -> List[str]:
        """
        解析要使用的搜索来源
        
        Args:
            providers: 指定的来源列表，None 表示使用所有已启用的来源
        
        Returns:
            实际要使用的来源列表
        """
        if providers is not None:
            return [str(p).strip().lower() for p in providers if str(p).strip()]
        
        # 自动检测已启用的来源
        result = []
        
        # 检查 Tavily
        tavily = self._get_tavily_searcher()
        if tavily and getattr(tavily, 'enabled', False):
            result.append("tavily")
        
        # 检查 Google Scholar / Google
        google = self._get_google_searcher()
        if google:
            if getattr(google, 'scholar_enabled', False):
                result.append("scholar")
            if getattr(google, 'google_enabled', False):
                result.append("google")

        # 检查 Semantic Scholar
        semantic = self._get_semantic_searcher()
        if semantic and getattr(semantic, "enabled", False):
            result.append("semantic")
        
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
        use_content_fetcher: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        异步搜索多个来源并合并去重
        
        Args:
            query: 搜索查询
            providers: 要使用的来源列表 ["scholar", "tavily", "google", "semantic"]，
                      None 表示使用所有已启用的来源
            source_configs: 每个来源的独立配置 {provider_id: {topK: int, threshold: float}}
            max_results_per_provider: 每个来源的最大结果数（当 source_configs 未指定时使用）
            use_query_expansion: 是否使用查询扩展（仅 Tavily 支持）
            use_query_optimizer: 是否启用智能查询优化器
            query_optimizer_max_queries: 优化器每个来源生成的最大查询数
        
        Returns:
            合并去重后的结果列表
        """
        providers = self._resolve_providers(providers)
        
        if not providers:
            return []
        
        logger.info(f"统一搜索: query={query!r}, providers={providers}")

        # 准备每个来源的配置
        source_configs = source_configs or {}

        def _get_max_results(provider: str) -> int:
            """获取指定来源的最大结果数"""
            cfg = source_configs.get(provider, {})
            return cfg.get("topK", max_results_per_provider)

        # query_expansion 已合并到统一优化器逻辑，不再使用 Tavily 内部扩展
        expansion_enabled = False

        optimizer_enabled = use_query_optimizer
        if optimizer_enabled is None:
            optimizer_enabled = bool(getattr(getattr(settings, "web_search", None), "enable_query_optimizer", True))

        smart_optimizer = get_smart_query_optimizer()
        use_smart = optimizer_enabled and smart_optimizer.enabled
        queries_per_provider: Dict[str, List[str]] = {}
        if use_smart:
            queries_per_provider = smart_optimizer.optimize(
                query, providers,
                max_queries_per_provider=query_optimizer_max_queries,
                llm_provider=llm_provider,
                model_override=model_override,
            )
            logger.info(f"Smart optimizer 生成多组查询: {list(queries_per_provider.keys())}")

        def _queries_for(provider: str) -> List[str]:
            if use_smart and provider in queries_per_provider and queries_per_provider[provider]:
                return queries_per_provider[provider]
            return [query]

        all_hits: List[Dict[str, Any]] = []
        tasks_with_flags: List[tuple] = []  # (coro, is_browser)
        
        # 收集 Scholar/Google 的所有查询，用于批量执行
        scholar_queries: List[str] = []
        google_queries: List[str] = []
        scholar_max_results = max_results_per_provider
        google_max_results = max_results_per_provider

        for provider in providers:
            queries = _queries_for(provider)
            provider_max_results = _get_max_results(provider)
            # Tavily 使用 smart 时不再启用内部 query_expansion，避免重复扩展
            tavily_expand = expansion_enabled and not use_smart

            if provider == "tavily":
                tavily = self._get_tavily_searcher()
                if tavily and getattr(tavily, 'enabled', False):
                    for q in queries:
                        tasks_with_flags.append((self._search_tavily(tavily, q, provider_max_results, tavily_expand), False))
            elif provider == "scholar":
                google = self._get_google_searcher()
                if google and getattr(google, 'scholar_enabled', False):
                    scholar_queries.extend(queries)
                    scholar_max_results = provider_max_results
            elif provider == "google":
                google = self._get_google_searcher()
                if google and getattr(google, 'google_enabled', False):
                    google_queries.extend(queries)
                    google_max_results = provider_max_results
            elif provider == "semantic":
                semantic = self._get_semantic_searcher()
                if semantic and getattr(semantic, "enabled", False):
                    for q in queries:
                        tasks_with_flags.append((self._search_semantic(semantic, q, provider_max_results), False))
        
        # Scholar/Google 使用批量方法（串行执行，避免浏览器冲突）
        if scholar_queries:
            google_searcher = self._get_google_searcher()
            if google_searcher:
                tasks_with_flags.append((
                    self._search_scholar_batch(google_searcher, scholar_queries, scholar_max_results), 
                    True
                ))
        
        if google_queries:
            google_searcher = self._get_google_searcher()
            if google_searcher:
                tasks_with_flags.append((
                    self._search_google_batch(google_searcher, google_queries, google_max_results), 
                    True
                ))

        # 并发执行：总并发 + 单任务超时；Scholar/Google 单独限流（风控默认 1，最大 2）
        perf = getattr(settings, "perf_unified_web", None)
        max_parallel = getattr(perf, "max_parallel_providers", 3) or 3
        timeout_s = getattr(perf, "per_provider_timeout_seconds", 30) or 30
        browser_max = getattr(perf, "browser_providers_max_parallel", 1) or 1
        
        # 批量浏览器搜索超时：每个查询约 30 秒
        max_browser_queries = max(len(scholar_queries), len(google_queries))
        browser_batch_timeout_s = max(timeout_s * max_browser_queries, 120)  # 至少 2 分钟
        
        sem = asyncio.Semaphore(max_parallel)
        sem_browser = asyncio.Semaphore(browser_max)

        async def _run_one(coro, is_browser: bool):
            effective_timeout = browser_batch_timeout_s if is_browser else timeout_s
            if is_browser:
                async with sem_browser:
                    async with sem:
                        return await asyncio.wait_for(coro, timeout=float(effective_timeout))
            else:
                async with sem:
                    return await asyncio.wait_for(coro, timeout=float(timeout_s))

        if tasks_with_flags:
            wrapped = [_run_one(coro, is_browser) for coro, is_browser in tasks_with_flags]
            results = await asyncio.gather(*wrapped, return_exceptions=True)
            for r in results:
                if isinstance(r, (asyncio.TimeoutError, asyncio.CancelledError)):
                    logger.warning("搜索任务超时或取消")
                elif isinstance(r, Exception):
                    logger.error(f"搜索任务出错: {r}")
                elif isinstance(r, list):
                    all_hits.extend(r)
        
        # 合并去重
        merged = _merge_and_dedup(all_hits)
        logger.info(f"统一搜索完成: 合并前 {len(all_hits)} 条, 去重后 {len(merged)} 条")

        # 全文抓取增强：True 强制启用，False 强制跳过，None 用后端配置
        fetcher = self._get_content_fetcher()
        do_enrich = use_content_fetcher is True or (
            use_content_fetcher is None and fetcher and getattr(fetcher, "enabled", False)
        )
        if do_enrich and fetcher:
            try:
                merged = await fetcher.enrich_results(merged, query=query)
                full_count = sum(
                    1 for h in merged
                    if (h.get("metadata") or {}).get("content_type") == "full_text"
                )
                logger.info(f"全文抓取完成: {full_count}/{len(merged)} 条")
            except Exception as e:
                logger.warning(f"全文抓取失败，使用原始片段: {e}")

        return merged
    
    async def _search_tavily(
        self,
        searcher,
        query: str,
        limit: int,
        use_query_expansion: bool,
    ) -> List[Dict[str, Any]]:
        """Tavily 搜索"""
        try:
            results = await searcher.async_search(query, use_query_expansion=use_query_expansion)
            return results[:limit]
        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return []
    
    async def _search_scholar(
        self,
        searcher,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Google Scholar 搜索"""
        try:
            return await searcher.search_scholar(query, limit=limit)
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

    async def _search_scholar_batch(
        self,
        searcher,
        queries: List[str],
        limit_per_query: int,
    ) -> List[Dict[str, Any]]:
        """Google Scholar 批量搜索（串行执行）"""
        try:
            logger.info(f"Scholar 批量搜索：{len(queries)} 个查询，每个最多 {limit_per_query} 条")
            return await searcher.search_scholar_batch(queries, limit_per_query=limit_per_query)
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
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar 搜索"""
        try:
            return await searcher.search(query, limit=limit)
        except Exception as e:
            logger.error(f"Semantic Scholar 搜索失败: {e}")
            return []
    
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
        use_content_fetcher: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        同步搜索（在 executor 中运行异步版本）
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
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中，使用 run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.search(**_search_kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.search(**_search_kwargs))
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(self.search(**_search_kwargs))


# 全局单例
unified_web_searcher = UnifiedWebSearcher()
