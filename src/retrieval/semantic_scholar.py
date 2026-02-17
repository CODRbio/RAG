"""
Semantic Scholar API 搜索（ai4scholar 代理）。
输出与 unified_web_search 兼容的 RAG 格式。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from config.settings import settings
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)


def _get_semantic_scholar_config() -> Dict[str, Any]:
    try:
        ss = getattr(settings, "semantic_scholar", None)
        if ss is not None:
            return {
                "enabled": getattr(ss, "enabled", False),
                "api_key": getattr(ss, "api_key", "") or "",
                "base_url": getattr(ss, "base_url", "https://ai4scholar.net/graph/v1"),
                "max_results": getattr(ss, "max_results", 5),
                "timeout_seconds": getattr(ss, "timeout_seconds", 30),
            }
    except Exception:
        pass
    return {}


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url or "")
        netloc = (parsed.netloc or "").strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or ""
    except Exception:
        return ""


@dataclass
class SemanticScholarSearcher:
    _config: Dict[str, Any] = field(default_factory=dict)
    _cache: Optional[TTLCache] = field(default=None, repr=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)

    BASE_URL = "https://ai4scholar.net/graph/v1"
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount",
        "authors",
        "url",
        "openAccessPdf",
        "publicationVenue",
        "externalIds",
    ]

    def __post_init__(self):
        if not self._config:
            self._config = _get_semantic_scholar_config()
        perf = getattr(settings, "perf_web_search", None)
        self._cache = (
            get_cache(
                getattr(perf, "cache_enabled", False),
                getattr(perf, "cache_ttl_seconds", 3600),
                prefix="semantic_scholar",
            )
            if perf else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled") and (self._config.get("api_key") or "").strip())

    async def _ensure_session(self) -> None:
        if self._session and not self._session.closed:
            return
        timeout_s = int(self._config.get("timeout_seconds", 30))
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def _fetch_json(self, url: str, params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        await self._ensure_session()
        assert self._session is not None
        async with self._session.get(url, params=params, headers=headers) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            raise RuntimeError(f"Semantic Scholar API error: {resp.status} - {text[:200]}")

    def _parse_paper(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
        open_pdf = (paper.get("openAccessPdf") or {}).get("url")
        external_ids = paper.get("externalIds") or {}
        doi = external_ids.get("DOI")
        arxiv = external_ids.get("ArXiv")
        url = paper.get("url") or ""
        if not url and doi:
            url = f"https://doi.org/{doi}"
        if not url and arxiv:
            url = f"https://arxiv.org/abs/{arxiv}"
        domain = _domain_from_url(url)
        venue = (paper.get("publicationVenue") or {}).get("name")

        metadata = {
            "source": "semantic",
            "doc_id": paper.get("title") or paper.get("paperId") or url or "semantic",
            "title": paper.get("title") or "",
            "url": url,
            "domain": domain,
            "search_query": query,
        }
        if authors:
            metadata["authors"] = authors
        if paper.get("year"):
            metadata["year"] = paper.get("year")
        if paper.get("citationCount") is not None:
            metadata["cited_by"] = paper.get("citationCount")
        if open_pdf:
            metadata["pdf_url"] = open_pdf
        if doi:
            metadata["doi"] = doi
        if arxiv:
            metadata["arxiv_id"] = arxiv
        if venue:
            metadata["venue"] = venue
        if paper.get("paperId"):
            metadata["paper_id"] = paper.get("paperId")

        return {
            "content": (paper.get("abstract") or paper.get("title") or "").strip(),
            "score": 0.9,
            "metadata": metadata,
        }

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        q = (query or "").strip()
        if not q:
            return []
        limit = limit or int(self._config.get("max_results", 5))
        cache_key = _make_key("semantic_scholar", q, limit, year_start, year_end)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        base_url = (self._config.get("base_url") or self.BASE_URL).rstrip("/")
        url = f"{base_url}/paper/search"
        params: Dict[str, Any] = {
            "query": q,
            "limit": min(int(limit), 100),
            "fields": ",".join(self.PAPER_FIELDS),
        }
        if year_start or year_end:
            if year_start and year_end:
                params["year"] = f"{year_start}-{year_end}"
            elif year_start:
                params["year"] = f"{year_start}-"
            else:
                params["year"] = f"-{year_end}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "DeepSea-RAG/1.0",
        }
        api_key = (self._config.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            data = await self._fetch_json(url, params, headers)
            papers = data.get("data") or []
            results = [self._parse_paper(p, q) for p in papers[:limit]]
            if self._cache:
                self._cache.set(cache_key, results)
            return results
        except asyncio.TimeoutError:
            logger.warning("Semantic Scholar search timeout")
            return []
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None


# 全局单例
semantic_scholar_searcher = SemanticScholarSearcher()
