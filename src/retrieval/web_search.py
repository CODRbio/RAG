"""
Tavily 网络搜索模块

与 hybrid_retriever 输出格式兼容，可作为 RAG 的补充素材提交给 LangGraph/LLM。
支持同步/异步搜索、可选的 LLM 多查询扩展。
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from config.settings import settings
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

logger = get_logger(__name__)


class _QueryExpansionResponse(BaseModel):
    queries: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_array(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"queries": data}
        return data


def _get_web_search_config() -> Dict[str, Any]:
    """从 config 或 settings 读取 web_search 配置"""
    try:
        from config.settings import settings
        ws = getattr(settings, "web_search", None)
        if ws is not None:
            return {
                "enabled": getattr(ws, "enabled", True),
                "api_key": getattr(ws, "api_key", "") or "",
                "search_depth": getattr(ws, "search_depth", "advanced"),
                "max_results": getattr(ws, "max_results", 5),
                "include_answer": getattr(ws, "include_answer", True),
                "include_domains": getattr(ws, "include_domains", []) or [],
                "exclude_domains": getattr(ws, "exclude_domains", []) or [],
                "enable_query_optimizer": getattr(ws, "enable_query_optimizer", True),
                "enable_query_expansion": getattr(ws, "enable_query_expansion", False),
                "query_expansion_llm": getattr(ws, "query_expansion_llm", "deepseek"),
                "max_queries": getattr(ws, "max_queries", 4),
            }
    except Exception:
        pass
    config_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    ws = raw.get("web_search") or {}
    include = ws.get("include_domains")
    exclude = ws.get("exclude_domains")
    if isinstance(include, str):
        include = [x.strip() for x in include.split(",") if x.strip()]
    if isinstance(exclude, str):
        exclude = [x.strip() for x in exclude.split(",") if x.strip()]
    return {
        "enabled": ws.get("enabled", True),
        "api_key": (ws.get("api_key") or "").strip(),
        "search_depth": ws.get("search_depth", "advanced"),
        "max_results": min(int(ws.get("max_results", 5)), 10),
        "include_answer": ws.get("include_answer", True),
        "include_domains": include or [],
        "exclude_domains": exclude or [],
        "enable_query_optimizer": ws.get("enable_query_optimizer", True),
        "enable_query_expansion": ws.get("enable_query_expansion", False),
        "query_expansion_llm": (ws.get("query_expansion_llm") or "deepseek").strip(),
        "max_queries": min(int(ws.get("max_queries", 4)), 8),
    }


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url or "")
        netloc = (parsed.netloc or "").strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or ""
    except Exception:
        return ""


def _normalize_hit(
    title: str,
    url: str,
    content: str,
    score: float,
    search_query: str,
) -> Dict[str, Any]:
    """转为与 hybrid_retriever 兼容的 hit 格式"""
    domain = _domain_from_url(url)
    # content 用于 context_packer，保留摘要
    text = (content or "").strip() or title
    return {
        "content": text,
        "score": float(score),
        "metadata": {
            "source": "tavily",
            "doc_id": title or url or "web",
            "title": title or "无标题",
            "url": url or "",
            "domain": domain,
            "search_query": search_query or "",
        },
    }


@dataclass
class TavilySearcher:
    """
    Tavily 搜索器，输出与 RAG 检索结果统一格式，便于与 context_packer / LangGraph 集成。
    """

    _config: Dict[str, Any] = field(default_factory=dict)
    _cache: Optional[TTLCache] = field(default=None, repr=False)

    def __post_init__(self):
        if not self._config:
            self._config = _get_web_search_config()
        perf = getattr(settings, "perf_web_search", None)
        self._cache = (
            get_cache(
                getattr(perf, "cache_enabled", False),
                getattr(perf, "cache_ttl_seconds", 3600),
                prefix="tavily",
            )
            if perf else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled") and (self._config.get("api_key") or "").strip())

    def _generate_queries_sync(self, user_query: str) -> List[str]:
        """同步：使用 LLM 生成多查询（可选）"""
        if not self._config.get("enable_query_expansion"):
            return [user_query]
        try:
            from src.llm import LLMManager
            cfg_path = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
            manager = LLMManager.from_json(str(cfg_path))
            provider = (self._config.get("query_expansion_llm") or "deepseek").strip()
            if not manager.is_available(provider):
                return [user_query]
            client = manager.get_client(provider)
            from datetime import datetime
            year = datetime.now().strftime("%Y")
            prompt = _pm.render("web_search_optimize.txt", user_query=user_query, year=year)
            resp = client.chat(
                messages=[
                    {"role": "system", "content": _pm.render("web_search_optimize_system.txt")},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                response_model=_QueryExpansionResponse,
            )
            parsed: Optional[_QueryExpansionResponse] = resp.get("parsed_object")
            if parsed is None:
                raw_text = (resp.get("final_text") or "").strip()
                if raw_text:
                    parsed = _QueryExpansionResponse.model_validate_json(raw_text)
            queries = [str(q).strip() for q in (parsed.queries if parsed else []) if str(q).strip()]
            if queries:
                return queries[: self._config.get("max_queries", 4)]
        except Exception:
            pass
        return [user_query]

    async def _generate_queries_async(self, user_query: str) -> List[str]:
        """异步：LLM 生成多查询（在 executor 中跑同步）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_queries_sync, user_query)

    def _search_tavily_sync(self, queries: List[str]) -> List[Dict[str, Any]]:
        """同步调用 Tavily API，返回标准化 hit 列表"""
        try:
            from tavily import TavilyClient
        except ImportError as e:
            logger.warning(f"Tavily search skipped: tavily-python not installed ({e}).")
            return []
        api_key = (self._config.get("api_key") or "").strip()
        if not api_key:
            logger.warning("Tavily search skipped: api_key empty in config.")
            return []
        try:
            client = TavilyClient(api_key=api_key)
        except Exception as e:
            logger.error(f"Tavily search failed (client init): {type(e).__name__}: {e}")
            return []
        max_results = min(self._config.get("max_results", 5), 10)
        search_depth = self._config.get("search_depth", "advanced")
        include_answer = self._config.get("include_answer", True)
        include_domains = self._config.get("include_domains") or []
        exclude_domains = self._config.get("exclude_domains") or []
        seen_urls: set = set()
        out: List[Dict[str, Any]] = []
        for q in queries:
            try:
                params = {
                    "query": q,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "include_answer": include_answer,
                }
                if include_domains:
                    params["include_domains"] = include_domains
                if exclude_domains:
                    params["exclude_domains"] = exclude_domains
                response = client.search(**params)
                for item in response.get("results") or []:
                    url = item.get("url") or ""
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        hit = _normalize_hit(
                            title=item.get("title") or "无标题",
                            url=url,
                            content=item.get("content") or "",
                            score=float(item.get("score", 0)),
                            search_query=q,
                        )
                        out.append(hit)
            except Exception as e:
                logger.error(f"Tavily search failed (query={q!r}): {type(e).__name__}: {e}")
                continue
        if not out:
            logger.warning("Tavily search returned 0 results (all queries failed or no hits).")
        out.sort(key=lambda x: x.get("score", 0), reverse=True)
        return out

    def search(
        self,
        query: str,
        *,
        use_query_expansion: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        同步搜索。返回与 hybrid_retriever 兼容的 hit 列表。
        """
        if not self.enabled:
            logger.warning("Tavily search skipped: disabled or api_key not set.")
            return []
        expand = use_query_expansion if use_query_expansion is not None else self._config.get("enable_query_expansion", False)
        cache_key = _make_key("tavily", query, expand)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        queries = self._generate_queries_sync(query) if expand else [query]
        if not queries:
            logger.warning("Tavily search skipped: no queries (query expansion may have failed).")
            return []
        timeout_s = getattr(getattr(settings, "perf_web_search", None), "timeout_seconds", 30) or 30
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._search_tavily_sync, queries)
                out = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning("Tavily search timeout: %ss", timeout_s)
            out = []
        if self._cache and out is not None:
            self._cache.set(cache_key, out)
        return out

    async def async_search(
        self,
        query: str,
        *,
        use_query_expansion: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """异步搜索。"""
        if not self.enabled:
            return []
        expand = use_query_expansion if use_query_expansion is not None else self._config.get("enable_query_expansion", False)
        cache_key = _make_key("tavily", query, expand)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        queries = await self._generate_queries_async(query) if expand else [query]
        timeout_s = getattr(getattr(settings, "perf_web_search", None), "timeout_seconds", 30) or 30
        loop = asyncio.get_event_loop()
        try:
            out = await asyncio.wait_for(
                loop.run_in_executor(None, self._search_tavily_sync, queries),
                timeout=float(timeout_s),
            )
        except asyncio.TimeoutError:
            logger.warning("Tavily async search timeout: %ss", timeout_s)
            out = []
        if self._cache and out is not None:
            self._cache.set(cache_key, out)
        return out


# 全局单例（懒加载配置）
web_searcher = TavilySearcher()
