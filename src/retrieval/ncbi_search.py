"""
NCBI PubMed 生物医学文献搜索（E-Utilities 免费 API）。

使用 esearch 获取 PubMed ID 列表，再用 esummary 获取元数据并通过 efetch 拉取摘要，
输出与 unified_web_search 兼容的 RAG 格式（content / score / metadata）。
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import aiohttp

from src.log import get_logger
from src.utils.cache import TTLCache, _make_key

logger = get_logger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class NCBISearcher:
    """
    PubMed E-Utilities 搜索器。

    先调 esearch 获取相关 PubMed ID，
    再调 esummary 批量拉取元数据（标题、作者、年份、DOI），
    最后用 efetch 拉取摘要文本。
    结果带 TTL 缓存，避免重复请求。
    """

    def __init__(
        self,
        api_key: str = "",
        timeout_seconds: int = 20,
        cache_ttl: int = 3600,
        cache_maxsize: int = 256,
    ):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._cache = TTLCache(maxsize=cache_maxsize, ttl_seconds=cache_ttl)

    async def close(self) -> None:
        # 保持兼容：当前实现为短生命周期 session，无需显式关闭。
        return None

    # ── 内部请求 ─────────────────────────────────────────────────────────────

    def _base_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {"retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def _esearch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[str]:
        """返回相关度排序的 PubMed ID 列表。"""
        term = query
        if year_start is not None or year_end is not None:
            if year_start is not None and year_end is not None:
                y0, y1 = sorted((int(year_start), int(year_end)))
                term += f' AND ("{y0}"[Date - Publication] : "{y1}"[Date - Publication])'
            elif year_start is not None:
                term += f' AND ("{int(year_start)}"[Date - Publication] : "3000"[Date - Publication])'
            elif year_end is not None:
                term += f' AND ("1000"[Date - Publication] : "{int(year_end)}"[Date - Publication])'
        params = self._base_params()
        params.update(
            {
                "db": "pubmed",
                "term": term,
                "retmax": str(limit),
                "sort": "relevance",
            }
        )
        async with session.get(ESEARCH_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
        return data.get("esearchresult", {}).get("idlist", [])

    async def _esummary(
        self,
        session: aiohttp.ClientSession,
        ids: List[str],
    ) -> Dict[str, Any]:
        """批量获取 PubMed 元数据记录，返回 {pmid: doc_dict}。"""
        if not ids:
            return {}
        params = self._base_params()
        params.update(
            {
                "db": "pubmed",
                "id": ",".join(ids),
                "version": "2.0",
            }
        )
        async with session.get(ESUMMARY_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
        return data.get("result", {})

    async def _efetch_abstracts(
        self,
        session: aiohttp.ClientSession,
        ids: List[str],
    ) -> Dict[str, str]:
        """批量获取摘要文本，返回 {pmid: abstract}。"""
        if not ids:
            return {}
        params = self._base_params()
        params.update(
            {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            }
        )
        async with session.get(EFETCH_URL, params=params) as resp:
            resp.raise_for_status()
            xml_text = await resp.text()

        out: Dict[str, str] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return out

        for article in root.findall(".//PubmedArticle"):
            pmid = (
                article.findtext(".//MedlineCitation/PMID")
                or article.findtext(".//PMID")
                or ""
            ).strip()
            if not pmid:
                continue

            parts: List[str] = []
            for node in article.findall(".//Abstract/AbstractText"):
                text = "".join(node.itertext()).strip()
                if not text:
                    continue
                label = (node.attrib.get("Label") or "").strip()
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)

            abstract = " ".join(parts).strip()
            if abstract:
                out[pmid] = abstract
        return out

    # ── 格式转换 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_rag_hit(pmid: str, doc: Dict[str, Any], abstract: str = "") -> Dict[str, Any]:
        """将 esummary 记录转换为 RAG-compatible hit。"""
        title = (doc.get("title") or "").strip().rstrip(".")

        authors_raw = doc.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if a.get("name")]

        pub_date = doc.get("pubdate") or doc.get("epubdate") or ""
        year = pub_date[:4] if pub_date else ""

        doi = ""
        for aid in doc.get("articleids") or []:
            if aid.get("idtype") == "doi":
                doi = (aid.get("value") or "").strip()
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        content_parts = [title] if title else [f"PubMed:{pmid}"]
        if authors:
            content_parts.append(f"Authors: {', '.join(authors[:5])}")
        if year:
            content_parts.append(f"Year: {year}")
        if doi:
            content_parts.append(f"DOI: {doi}")
        if abstract:
            content_parts.append(f"Abstract: {abstract[:900]}")

        return {
            "content": " | ".join(content_parts),
            "score": 0.98,
            "metadata": {
                "source": "ncbi",
                "title": title,
                "url": url,
                "doi": doi,
                "authors": authors,
                "year": year,
                "pmid": pmid,
                "abstract": abstract,
            },
        }

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        limit: int = 5,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        异步搜索 PubMed，返回最多 `limit` 条 RAG-compatible 结果。
        相同 query+limit 命中缓存时直接返回，不重复请求。
        """
        cache_key = _make_key("ncbi", query, limit, year_start, year_end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"NCBI cache hit: {query!r}")
            return cached

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            connector = aiohttp.TCPConnector(limit=5)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                ids = await self._esearch(
                    session,
                    query,
                    limit,
                    year_start=year_start,
                    year_end=year_end,
                )
                if not ids:
                    logger.info(f"NCBI esearch 无结果: {query!r}")
                    self._cache.set(cache_key, [])
                    return []

                result_map = await self._esummary(session, ids)
                abstracts = await self._efetch_abstracts(session, ids)

                hits: List[Dict[str, Any]] = []
                for pmid in ids:
                    doc = result_map.get(pmid) or result_map.get(str(pmid))
                    if not isinstance(doc, dict):
                        continue
                    hits.append(self._to_rag_hit(pmid, doc, abstracts.get(str(pmid), "")))

            self._cache.set(cache_key, hits)
            logger.info(f"NCBI 搜索完成: query={query!r}, 返回 {len(hits)} 条")
            return hits

        except asyncio.TimeoutError:
            logger.warning(f"NCBI 搜索超时: {query!r}")
            return []
        except Exception as e:
            logger.error(f"NCBI 搜索失败: {e}")
            return []


# ── 全局单例（无需 API key，免费端点）─────────────────────────────────────────

_ncbi_searcher_instance: Optional[NCBISearcher] = None


def get_ncbi_searcher() -> NCBISearcher:
    global _ncbi_searcher_instance
    if _ncbi_searcher_instance is None:
        try:
            from config.settings import settings
            cfg = getattr(settings, "ncbi", None)
            _ncbi_searcher_instance = NCBISearcher(
                api_key=getattr(cfg, "api_key", "") or "",
                timeout_seconds=int(getattr(cfg, "timeout_seconds", 20)),
                cache_ttl=int(getattr(cfg, "cache_ttl_seconds", 3600)),
                cache_maxsize=int(getattr(cfg, "cache_maxsize", 256)),
            )
        except Exception:
            _ncbi_searcher_instance = NCBISearcher()
    return _ncbi_searcher_instance
