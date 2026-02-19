"""
去重与多样性模块
- 指纹去重（避免相邻 chunk 高度相似）
- 单文档上限（per_doc_cap，防止单 doc 垄断 top N）
- 跨源去重（拦截网络搜索中与本地文库重叠的文献）
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_LRU_MAX = 512


class _LRUCache:
    """Minimal thread-unsafe LRU; fine for per-request hot path."""
    __slots__ = ("_data", "_max")

    def __init__(self, maxsize: int = _LRU_MAX):
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._max = maxsize

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max:
            self._data.popitem(last=False)


_crossref_lru = _LRUCache()

_DOI_RE = re.compile(
    r"(?:doi[:\s]*|(?:https?://)?(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s,;)\]\"'<>]+[^\s,;)\]\"'<>.:])",
    re.IGNORECASE,
)
_CROSSREF_API = "https://api.crossref.org/works"
_CROSSREF_TIMEOUT_SECONDS = 4


def _get_paper_meta_store():
    """延迟加载 PaperMetadataStore，避免循环导入。"""
    from src.indexing.paper_metadata_store import paper_meta_store
    return paper_meta_store


def normalize_doi(doi: Optional[str]) -> str:
    """
    归一化 DOI：小写、去空白、去 URL 前缀。
    '10.1038/ismej.2016.124' / 'https://doi.org/10.1038/ISMEJ.2016.124' → '10.1038/ismej.2016.124'
    """
    if not doi or not isinstance(doi, str):
        return ""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
    return d.rstrip("/.")


def normalize_title(title: str) -> str:
    """标题归一化：小写 + 去除非字母数字字符。"""
    if not title or not isinstance(title, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _fingerprint(text: str) -> str:
    """简单 hash 指纹（用于去重）"""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()[:512]
    return str(hash(t))


# ── 跨源去重 ──────────────────────────────────────────

def _extract_local_fields(chunk: Any) -> Tuple[str, str, str]:
    """兼容 EvidenceChunk / dict，提取 (doi, title, doc_id)。"""
    if isinstance(chunk, dict):
        meta = chunk.get("metadata") or {}
        doi = meta.get("doi") or chunk.get("doi") or ""
        title = meta.get("doc_title") or meta.get("title") or chunk.get("doc_title") or chunk.get("title") or ""
        doc_id = meta.get("doc_id") or meta.get("paper_id") or chunk.get("doc_id") or chunk.get("paper_id") or ""
        return str(doi), str(title), str(doc_id)

    doi = getattr(chunk, "doi", "") or ""
    title = getattr(chunk, "doc_title", "") or getattr(chunk, "title", "") or ""
    doc_id = getattr(chunk, "doc_id", "") or getattr(chunk, "paper_id", "") or ""
    return str(doi), str(title), str(doc_id)


def _extract_web_title(hit: Dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    return str(meta.get("title") or hit.get("title") or "")


def _extract_web_url(hit: Dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    return str(meta.get("url") or hit.get("url") or "")


def _extract_doi_from_text(text: str) -> str:
    if not text:
        return ""
    m = _DOI_RE.search(text)
    return normalize_doi(m.group(1)) if m else ""


def _token_set(text: str) -> Set[str]:
    if not text:
        return set()
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {w for w in t.split() if len(w) >= 3}


def _title_overlap(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), 1)


def _crossref_lookup_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    用标题查 CrossRef，返回 {doi,title,authors,year,venue} 或 None。
    两级缓存：SQLite 持久化 + 进程内 LRU（热路径零 I/O）。
    """
    key = normalize_title(title)
    if not key:
        return None

    # L1: 进程内 LRU（capacity=512，覆盖一次检索的热词）
    if key in _crossref_lru:
        return _crossref_lru[key]

    # L2: SQLite 持久化缓存
    try:
        store = _get_paper_meta_store()
        cached = store.crossref_get(title)
        if cached is not None:
            _crossref_lru[key] = cached
            return cached
        if store.crossref_has(title):
            _crossref_lru[key] = None
            return None
    except Exception:
        pass

    # L3: 实际 API 请求
    def _request() -> Dict[str, Any]:
        qs = urlencode({"query.bibliographic": title, "rows": 3})
        req = Request(
            f"{_CROSSREF_API}?{qs}",
            headers={"User-Agent": "DeepSea-RAG/1.0", "Accept": "application/json"},
            method="GET",
        )
        with urlopen(req, timeout=_CROSSREF_TIMEOUT_SECONDS) as resp:
            if getattr(resp, "status", 200) != 200:
                return {}
            return json.loads(resp.read().decode("utf-8"))

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_request)
            data = fut.result(timeout=float(_CROSSREF_TIMEOUT_SECONDS + 1))
        items = (data.get("message") or {}).get("items") or []
    except (FuturesTimeoutError, Exception):
        _crossref_lru[key] = None
        try:
            _get_paper_meta_store().crossref_put(title, None)
        except Exception:
            pass
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for it in items:
        cr_title = ((it.get("title") or [""])[0] or "").strip()
        score = _title_overlap(title, cr_title)
        if score > best_score:
            best_score = score
            best = it

    if not best or best_score < 0.45:
        _crossref_lru[key] = None
        try:
            _get_paper_meta_store().crossref_put(title, None)
        except Exception:
            pass
        return None

    doi = normalize_doi(best.get("DOI"))
    if not doi:
        _crossref_lru[key] = None
        try:
            _get_paper_meta_store().crossref_put(title, None)
        except Exception:
            pass
        return None

    authors = []
    for a in best.get("author") or []:
        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if name:
            authors.append(name)

    year = None
    for df in ("published-print", "published-online", "issued", "created"):
        dp = (best.get(df) or {}).get("date-parts", [[]])
        if dp and dp[0] and dp[0][0]:
            year = dp[0][0]
            break

    venue = ((best.get("container-title") or [""])[0] or "").strip()
    parsed = {
        "doi": doi,
        "title": ((best.get("title") or [""])[0] or "").strip() or title,
        "authors": authors or None,
        "year": year,
        "venue": venue or None,
    }
    _crossref_lru[key] = parsed
    try:
        _get_paper_meta_store().crossref_put(title, parsed)
    except Exception:
        pass
    return parsed


def _enrich_web_hits_missing_doi(
    web_hits: List[Dict[str, Any]],
    local_title_blacklist: Set[str],
) -> Tuple[int, int]:
    """
    尝试为无 DOI 的 web hit 回填 DOI（先 URL 正则，再 CrossRef 标题检索）。
    返回 (crossref_lookups, resolved_with_crossref)。
    """
    lookups = 0
    resolved = 0
    for hit in web_hits:
        meta = hit.get("metadata") or {}
        if normalize_doi(meta.get("doi")):
            continue

        # 先试 URL 直接抽 DOI（零成本）
        url_doi = _extract_doi_from_text(_extract_web_url(hit))
        if url_doi:
            meta["doi"] = url_doi
            hit["metadata"] = meta
            continue

        # 再试标题走 CrossRef
        title = _extract_web_title(hit)
        nt = normalize_title(title)
        if not nt:
            continue
        # 标题已命中本地黑名单，无需再打 CrossRef
        if nt in local_title_blacklist:
            continue

        lookups += 1
        crossref = _crossref_lookup_by_title(title)
        if not crossref:
            continue

        meta["doi"] = crossref["doi"]
        # 补充 semantic / ncbi 等来源常缺的字段
        if not meta.get("title") and crossref.get("title"):
            meta["title"] = crossref["title"]
        if not meta.get("authors") and crossref.get("authors"):
            meta["authors"] = crossref["authors"]
        if not meta.get("year") and crossref.get("year"):
            meta["year"] = crossref["year"]
        if not meta.get("venue") and crossref.get("venue"):
            meta["venue"] = crossref["venue"]
        meta["doi_source"] = "crossref"
        hit["metadata"] = meta
        resolved += 1

    return lookups, resolved

def cross_source_dedup(
    local_chunks: List[Any],
    web_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    拦截网络搜索中与本地文库重叠的文献。

    通过 DOI 匹配构建"本地黑名单"，过滤掉 web_hits 中已存在于本地的结果，
    把上下文 Token 留给真正的增量信息。

    Args:
        local_chunks: 本地检索结果（通常是 EvidenceChunk）
        web_hits: 网络搜索结果（raw dicts from UnifiedWebSearch）

    Returns:
        过滤后的 web_hits（仅保留本地不存在的文献）
    """
    if not local_chunks or not web_hits:
        return web_hits

    store = _get_paper_meta_store()

    # 构建本地 DOI + Title 黑名单
    # 基础集合：从 SQLite 全量拉取（已有索引，毫秒级）
    local_dois: Set[str] = store.all_dois()
    local_titles: Set[str] = store.all_normalized_titles()

    # 再从当前检索结果中补充（可能有新入库但还未写 SQLite 的）
    for c in local_chunks:
        doi_raw, title_raw, doc_id = _extract_local_fields(c)
        doi = normalize_doi(doi_raw)
        if doi:
            local_dois.add(doi)
        nt = normalize_title(title_raw)
        if nt:
            local_titles.add(nt)

    if not local_dois and not local_titles:
        return web_hits

    # 先为无 DOI 的 web hit 尝试补 DOI（URL / CrossRef）
    crossref_lookups, crossref_resolved = _enrich_web_hits_missing_doi(web_hits, local_titles)

    # 过滤 web_hits
    before = len(web_hits)
    filtered: List[Dict[str, Any]] = []
    for hit in web_hits:
        meta = hit.get("metadata") or {}
        web_doi = normalize_doi(meta.get("doi"))
        web_title = normalize_title(meta.get("title") or hit.get("title") or "")
        if (web_doi and web_doi in local_dois) or (web_title and web_title in local_titles):
            continue
        filtered.append(hit)

    removed = before - len(filtered)
    if removed > 0:
        logger.info(
            "cross_source_dedup: removed=%d, local_doi=%d, local_title=%d, crossref_lookups=%d, crossref_resolved=%d",
            removed,
            len(local_dois),
            len(local_titles),
            crossref_lookups,
            crossref_resolved,
        )
    return filtered


# ── 指纹去重 + 单文档上限 ─────────────────────────────

def dedup_and_diversify(
    candidates: List[Dict],
    per_doc_cap: int = 3,
) -> List[Dict]:
    """
    去重 + 单文档上限
    
    Args:
        candidates: 已按 rerank score 排序的候选列表
        per_doc_cap: 单文档最多保留条数
    
    Returns:
        去重且满足 per_doc_cap 的结果
    """
    seen_fingerprints: set = set()
    per_doc_count: Dict[str, int] = defaultdict(int)
    output: List[Dict] = []

    for c in candidates:
        text = c.get("content") or c.get("raw_content") or ""
        fp = _fingerprint(text)
        if fp and fp in seen_fingerprints:
            continue

        meta = c.get("metadata", {}) or {}
        doc_id = meta.get("doc_id") or meta.get("paper_id") or ""
        if per_doc_cap > 0 and doc_id and per_doc_count[doc_id] >= per_doc_cap:
            continue

        output.append(c)
        if fp:
            seen_fingerprints.add(fp)
        if doc_id:
            per_doc_count[doc_id] += 1

    return output
