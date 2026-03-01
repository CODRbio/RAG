"""
SerpAPI 搜索（Google Scholar + Google Web）。
输出与 unified_web_search 兼容的 RAG 格式。
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from config.settings import settings
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url or "")
        netloc = (parsed.netloc or "").strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _get_serpapi_config() -> Dict[str, Any]:
    try:
        cfg = getattr(settings, "serpapi", None)
        if cfg is not None:
            return {
                "enabled": bool(getattr(cfg, "enabled", False)),
                "api_key": (getattr(cfg, "api_key", "") or "").strip(),
                "max_results": int(getattr(cfg, "max_results", 10) or 10),
                "timeout_seconds": int(getattr(cfg, "timeout_seconds", 30) or 30),
            }
    except Exception:
        pass
    return {}


def _parse_authors_from_summary(summary: str) -> List[str]:
    text = (summary or "").strip()
    if not text:
        return []
    # Typical shape: "J Smith, A Doe - Nature, 2022"
    head = text.split(" - ", 1)[0]
    if not head:
        return []
    parts = [p.strip() for p in re.split(r",|，", head) if p.strip()]
    return parts[:8]


def _extract_year_from_summary(summary: str) -> Optional[int]:
    text = summary or ""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if not m:
        return None
    y = int(m.group(1))
    return y if 1900 <= y <= 2100 else None


def _extract_doi_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.I)
    if not m:
        return None
    doi = m.group(1).rstrip(").,; ")
    return doi if "/" in doi else None


def _extract_doi_from_url(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        return None
    m = re.search(r"(?:doi\.org|dx\.doi\.org)/+(10\.\d{4,9}/[^\s&?#]+)", u, re.I)
    if not m:
        return None
    doi = m.group(1).rstrip("/).,; ")
    return doi if "/" in doi else None


@dataclass
class SerpAPISearcher:
    _config: Dict[str, Any] = field(default_factory=dict)
    _cache: Optional[TTLCache] = field(default=None, repr=False)
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)

    BASE_URL = "https://serpapi.com/search.json"

    def __post_init__(self):
        if not self._config:
            self._config = _get_serpapi_config()
        perf = getattr(settings, "perf_web_search", None)
        self._cache = (
            get_cache(
                getattr(perf, "cache_enabled", False),
                getattr(perf, "cache_ttl_seconds", 3600),
                prefix="serpapi",
            )
            if perf
            else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled")) and bool(self._config.get("api_key"))

    async def _ensure_session(self) -> None:
        current_loop = asyncio.get_running_loop()
        if self._session and not self._session.closed:
            session_loop = getattr(self._session, "_loop", None)
            if session_loop is current_loop and not current_loop.is_closed():
                return
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        timeout_s = int(self._config.get("timeout_seconds", 30))
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s))

    async def _fetch_results(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        await self._ensure_session()
        assert self._session is not None
        async with self._session.get(self.BASE_URL, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"SerpAPI error: {resp.status} - {text[:200]}")
            payload = await resp.json()
        return payload.get("organic_results") or []

    async def search_scholar(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        if not self.enabled:
            logger.info("[retrieval] serpapi scholar skip enabled=False")
            return []
        k = int(limit or self._config.get("max_results", 10))
        cache_key = _make_key("serpapi_scholar", q, k, year_start, year_end)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("[retrieval] serpapi scholar cache_hit query=%r hits=%d", q[:80], len(cached))
                return cached

        logger.info("[retrieval] serpapi scholar start query=%r limit=%s year=%s~%s", q[:80], k, year_start, year_end)
        params: Dict[str, Any] = {
            "engine": "google_scholar",
            "q": q,
            "api_key": self._config.get("api_key"),
            "num": min(max(k, 1), 20),
        }
        if year_start:
            params["as_ylo"] = int(year_start)
        if year_end:
            params["as_yhi"] = int(year_end)

        try:
            rows = await self._fetch_results(params)
            results: List[Dict[str, Any]] = []
            for idx, item in enumerate(rows[:k], start=1):
                title = (item.get("title") or "").strip()
                url = (item.get("link") or "").strip()
                snippet = (item.get("snippet") or "").strip()
                pub_info = item.get("publication_info") or {}
                summary = pub_info.get("summary") or ""
                cited_total = ((item.get("inline_links") or {}).get("cited_by") or {}).get("total")
                pdf_url = ""
                resources = item.get("resources") or []
                if resources and isinstance(resources[0], dict):
                    pdf_url = (resources[0].get("link") or "").strip()

                metadata: Dict[str, Any] = {
                    "source": "serpapi_scholar",
                    "provider": "serpapi",
                    "doc_id": title or url or f"serpapi_scholar_{idx}",
                    "title": title,
                    "url": url,
                    "domain": _domain_from_url(url),
                    "search_query": q,
                }
                authors = _parse_authors_from_summary(summary)
                if authors:
                    metadata["authors"] = authors
                if summary:
                    metadata["venue"] = summary
                year = _extract_year_from_summary(summary)
                if year is not None:
                    metadata["year"] = year
                if cited_total is not None:
                    metadata["cited_by"] = cited_total
                if pdf_url:
                    metadata["pdf_url"] = pdf_url
                doi = (
                    _extract_doi_from_url(url)
                    or _extract_doi_from_url(pdf_url)
                    or _extract_doi_from_text(snippet)
                    or _extract_doi_from_text(summary)
                )
                if doi:
                    metadata["doi"] = doi

                results.append(
                    {
                        "content": snippet or title,
                        "score": max(0.0, 1.0 - (idx - 1) * 0.05),
                        "metadata": metadata,
                    }
                )
            if self._cache:
                self._cache.set(cache_key, results)
            logger.info("[retrieval] serpapi scholar done query=%r hits=%d", q[:80], len(results))
            return results
        except asyncio.TimeoutError:
            logger.warning("[retrieval] serpapi scholar timeout query=%r", q[:80])
            return []
        except Exception as e:
            logger.error("[retrieval] serpapi scholar failed query=%r error=%s", q[:80], e)
            return []

    async def search_google(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        if not self.enabled:
            logger.info("[retrieval] serpapi google skip enabled=False")
            return []
        k = int(limit or self._config.get("max_results", 10))
        cache_key = _make_key("serpapi_google", q, k)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("[retrieval] serpapi google cache_hit query=%r hits=%d", q[:80], len(cached))
                return cached

        logger.info("[retrieval] serpapi google start query=%r limit=%s", q[:80], k)
        params: Dict[str, Any] = {
            "engine": "google",
            "q": q,
            "api_key": self._config.get("api_key"),
            "num": min(max(k, 1), 20),
        }

        try:
            rows = await self._fetch_results(params)
            results: List[Dict[str, Any]] = []
            for idx, item in enumerate(rows[:k], start=1):
                title = (item.get("title") or "").strip()
                url = (item.get("link") or "").strip()
                snippet = (item.get("snippet") or "").strip()
                metadata: Dict[str, Any] = {
                    "source": "serpapi_google",
                    "provider": "serpapi",
                    "doc_id": title or url or f"serpapi_google_{idx}",
                    "title": title,
                    "url": url,
                    "domain": _domain_from_url(url),
                    "search_query": q,
                }
                results.append(
                    {
                        "content": snippet or title,
                        "score": max(0.0, 0.8 - (idx - 1) * 0.04),
                        "metadata": metadata,
                    }
                )
            if self._cache:
                self._cache.set(cache_key, results)
            logger.info("[retrieval] serpapi google done query=%r hits=%d", q[:80], len(results))
            return results
        except asyncio.TimeoutError:
            logger.warning("[retrieval] serpapi google timeout query=%r", q[:80])
            return []
        except Exception as e:
            logger.error("[retrieval] serpapi google failed query=%r error=%s", q[:80], e)
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

