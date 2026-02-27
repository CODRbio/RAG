"""
Semantic Scholar API 搜索（ai4scholar 代理）。
输出与 unified_web_search 兼容的 RAG 格式。

支持三种搜索模式：
  - Paper Relevance Search  (/paper/search)       – 按相关性排序的论文搜索
  - Text Snippet Search     (/snippet/search)      – 论文正文片段搜索
  - Paper Bulk Search       (/paper/search/bulk)   – 布尔语法批量搜索（托底）
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

_SEMANTIC_QUERY_MAX_CHARS = 200


def _truncate_query(q: str) -> str:
    """Shorten an overly long query for the Semantic Scholar search API.

    The API returns poor (often zero) results for sentence-length queries.
    We keep at most ``_SEMANTIC_QUERY_MAX_CHARS`` characters, truncating at
    the last word boundary.  We also strip the common "topic preamble"
    pattern used by the deep-research agent (everything before the first
    sentence-ending period, if the result is still meaningful).
    """
    q = (q or "").strip()
    if len(q) <= _SEMANTIC_QUERY_MAX_CHARS:
        return q
    # If the query has "Topic description. Actual question", keep only the
    # part after the first period (which is usually the specific sub-query).
    dot_idx = q.find(". ")
    if dot_idx > 0:
        after_dot = q[dot_idx + 2:].strip()
        if len(after_dot) >= 20:
            q = after_dot
    if len(q) <= _SEMANTIC_QUERY_MAX_CHARS:
        return q
    # Hard truncate at word boundary
    truncated = q[:_SEMANTIC_QUERY_MAX_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > _SEMANTIC_QUERY_MAX_CHARS // 2:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:")


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
    SNIPPET_FIELDS = [
        "snippet.text",
        "snippet.snippetKind",
        "snippet.section",
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
        """Config-level enabled flag (for auto-discovery). API key is optional for public API."""
        return bool(self._config.get("enabled"))

    # ── HTTP helpers ──────────────────────────────────────────────

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

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "DeepSea-RAG/1.0",
        }
        api_key = (self._config.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _year_param(year_start: Optional[int], year_end: Optional[int]) -> Optional[str]:
        """Build the ``year`` query-string value accepted by all S2 endpoints."""
        if year_start and year_end:
            return f"{year_start}-{year_end}"
        if year_start:
            return f"{year_start}-"
        if year_end:
            return f"-{year_end}"
        return None

    # ── Result parsers ────────────────────────────────────────────

    def _parse_paper(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Parse a /paper/search or /paper/search/bulk result."""
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

        metadata: Dict[str, Any] = {
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

    @staticmethod
    def _extract_url_from_disclaimer(disclaimer: str) -> str:
        """Extract the open-access URL from openAccessInfo.disclaimer.

        /snippet/search returns no standalone PDF URL field.  The URL is
        embedded in the disclaimer text:
          "...available at https://arxiv.org/abs/1805.02262, which is subject..."
        """
        m = re.search(r"available at (https?://\S+?)(?:[,\s]|$)", disclaimer or "")
        if m:
            return m.group(1).rstrip(".,)")
        return ""

    def _parse_snippet(self, item: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Parse a /snippet/search result into RAG-compatible format.

        Snippet responses carry ~500-word excerpts from body text alongside
        minimal paper metadata (corpusId, title, authors, openAccessInfo).

        Note: the paper object in snippet responses has NO ``url`` field.
        The S2 paper page URL is constructed from ``corpusId``.
        The open-access PDF URL (if any) is embedded in
        ``openAccessInfo.disclaimer`` and extracted via regex.
        """
        snippet = item.get("snippet") or {}
        paper = item.get("paper") or {}
        score = item.get("score", 0.0)

        # The snippet API returns authors as [[name1, name2, …]] (list of
        # name-lists) rather than the [{authorId, name}] shape of /paper/search.
        authors_raw = paper.get("authors") or []
        authors: List[str] = []
        for a in authors_raw:
            if isinstance(a, list):
                authors.extend(str(n) for n in a if n)
            elif isinstance(a, str):
                authors.append(a)
            elif isinstance(a, dict):
                name = a.get("name", "")
                if name:
                    authors.append(name)

        corpus_id = paper.get("corpusId")
        title = paper.get("title") or ""
        # Construct the S2 paper page URL from corpusId (no url field in response)
        url = f"https://www.semanticscholar.org/paper/{corpus_id}" if corpus_id else ""
        domain = _domain_from_url(url)

        snippet_text = (snippet.get("text") or "").strip()

        metadata: Dict[str, Any] = {
            "source": "semantic_snippet",
            "doc_id": title or str(corpus_id) or "semantic_snippet",
            "title": title,
            "url": url,
            "domain": domain,
            "search_query": query,
        }
        if authors:
            metadata["authors"] = authors
        if corpus_id:
            metadata["corpus_id"] = corpus_id
        if snippet.get("snippetKind"):
            metadata["snippet_kind"] = snippet["snippetKind"]
        if snippet.get("section"):
            metadata["section"] = snippet["section"]

        open_info = paper.get("openAccessInfo") or {}
        pdf_url = self._extract_url_from_disclaimer(open_info.get("disclaimer", ""))
        if pdf_url:
            metadata["pdf_url"] = pdf_url

        return {
            "content": snippet_text or title,
            "score": score,
            "metadata": metadata,
        }

    # ── Public search methods ─────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Paper relevance search (/paper/search).

        Returns up to 1 000 relevance-ranked results.  Long queries are
        automatically truncated via ``_truncate_query``.
        """
        raw_q = (query or "").strip()
        if not raw_q:
            return []
        q = _truncate_query(raw_q)
        if q != raw_q:
            logger.debug("Semantic Scholar query truncated: %d→%d chars", len(raw_q), len(q))
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
        year_val = self._year_param(year_start, year_end)
        if year_val:
            params["year"] = year_val
        headers = self._build_headers()

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

    async def search_snippets(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Text snippet search (/snippet/search).

        Returns body-text excerpts (~500 words) drawn from title, abstract
        and full paper text, ranked by relevance to *query*.  Unlike
        ``search()`` which only returns abstracts, this surfaces passages
        from the paper body – useful for finding specific evidence.
        """
        raw_q = (query or "").strip()
        if not raw_q:
            return []
        limit = limit or int(self._config.get("max_results", 5))
        cache_key = _make_key("semantic_snippet", raw_q, limit, year_start, year_end)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        base_url = (self._config.get("base_url") or self.BASE_URL).rstrip("/")
        url = f"{base_url}/snippet/search"
        params: Dict[str, Any] = {
            "query": raw_q,
            "limit": min(int(limit), 100),
            "fields": ",".join(self.SNIPPET_FIELDS),
        }
        year_val = self._year_param(year_start, year_end)
        if year_val:
            params["year"] = year_val
        headers = self._build_headers()

        try:
            data = await self._fetch_json(url, params, headers)
            items = data.get("data") or []
            results = [self._parse_snippet(it, raw_q) for it in items[:limit]]
            if self._cache:
                self._cache.set(cache_key, results)
            return results
        except asyncio.TimeoutError:
            logger.warning("Semantic Scholar snippet search timeout")
            return []
        except Exception as e:
            logger.error("Semantic Scholar snippet search failed: %s", e)
            return []

    async def search_bulk(
        self,
        query: str,
        limit: Optional[int] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        sort: str = "citationCount:desc",
    ) -> List[Dict[str, Any]]:
        """Paper bulk search (/paper/search/bulk).

        No relevance ranking; supports boolean query syntax
        (``|``, ``+``, ``-``, ``""``, ``~``).  Sort defaults to
        ``citationCount:desc`` so the most-cited matches come first.
        Intended as a fallback when relevance search returns too few results.
        Each call returns up to 1 000 papers; pagination via continuation
        token is *not* exposed here – we only take the first batch.
        """
        raw_q = (query or "").strip()
        if not raw_q:
            return []
        limit = limit or int(self._config.get("max_results", 5))
        cache_key = _make_key("semantic_bulk", raw_q, limit, year_start, year_end, sort)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        base_url = (self._config.get("base_url") or self.BASE_URL).rstrip("/")
        url = f"{base_url}/paper/search/bulk"
        params: Dict[str, Any] = {
            "query": raw_q,
            "fields": ",".join(self.PAPER_FIELDS),
        }
        if sort:
            params["sort"] = sort
        year_val = self._year_param(year_start, year_end)
        if year_val:
            params["year"] = year_val
        headers = self._build_headers()

        try:
            data = await self._fetch_json(url, params, headers)
            papers = data.get("data") or []
            results = [self._parse_paper(p, raw_q) for p in papers[:limit]]
            for r in results:
                r["metadata"]["source"] = "semantic_bulk"
                r["score"] = 0.7
            if self._cache:
                self._cache.set(cache_key, results)
            return results
        except asyncio.TimeoutError:
            logger.warning("Semantic Scholar bulk search timeout")
            return []
        except Exception as e:
            logger.error("Semantic Scholar bulk search failed: %s", e)
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None


# 全局单例
semantic_scholar_searcher = SemanticScholarSearcher()
