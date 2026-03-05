"""
去重与多样性模块
- 指纹去重（避免相邻 chunk 高度相似）
- 单文档上限（per_doc_cap，防止单 doc 垄断 top N）
- 跨源去重（拦截网络搜索中与本地文库重叠的文献）
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import socket
from urllib.error import URLError
from urllib.parse import unquote, urlparse, urlencode
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
    归一化 DOI：小写、去空白、去 URL 前缀、解码 %2F 等。
    '10.1038/ismej.2016.124' / 'https://doi.org/10.1038%2FISMEJ.2016.124' → '10.1038/ismej.2016.124'
    """
    if not doi or not isinstance(doi, str):
        return ""
    d = unquote(doi.strip())
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
    m = _DOI_RE.search(d)
    return (m.group(1).lower().rstrip("/.") if m else d.lower().rstrip("/."))


def normalize_title(title: str) -> str:
    """标题归一化：小写 + 去除非字母数字字符。"""
    if not title or not isinstance(title, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


# arXiv ID: YYMM.NNNNN or YYMM.NNNNNvN
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_doi_from_pdf_tiered(pdf_path: "Path") -> Tuple[Optional[str], Optional[str]]:
    """
    从 PDF 分层提取 DOI（及可选 title）。供入库前去重、DOI 回填等使用。
    Tier 1: 原生 PDF 元数据；Tier 2: 首页正文 DOI 正则；Tier 3: 标题 → CrossRef。
    返回 (doi, title)，无则为 None。
    """
    from pathlib import Path
    path = Path(pdf_path) if not isinstance(pdf_path, Path) else pdf_path
    if not path.exists():
        return None, None
    # Tier 1: native metadata
    try:
        from src.parser.pdf_parser import extract_native_metadata
        meta = extract_native_metadata(path)
        if meta.get("doi"):
            return meta.get("doi"), meta.get("title") or None
    except Exception:
        pass
    # Tier 2: first-page text scan
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            if len(doc) > 0:
                text = doc[0].get_text() or ""
                m = _DOI_RE.search(text)
                if m:
                    return m.group(1).rstrip(".:"), None
    except Exception:
        pass
    # Tier 3: title -> CrossRef
    title = None
    try:
        from src.parser.pdf_parser import extract_native_metadata
        meta = extract_native_metadata(path)
        title = (meta.get("title") or "").strip() or None
    except Exception:
        pass
    if not title or len(title) < 10:
        return None, None
    try:
        cr = _crossref_lookup_by_title(title)
        if cr and cr.get("doi"):
            return cr["doi"], title
    except Exception:
        pass
    return None, None


def extract_arxiv_id(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    从 metadata 或 URL 中提取规范化 arXiv ID（YYMM.NNNNN 或 YYMM.NNNNNvN）。
    返回小写字符串，无则返回 None。
    """
    if not meta:
        return None
    raw = meta.get("arxiv_id") or meta.get("arxiv")
    if raw and isinstance(raw, str):
        m = _ARXIV_ID_RE.search(raw.strip())
        return m.group(1).lower() if m else raw.strip().lower()
    url = meta.get("url") or ""
    if url:
        m = _ARXIV_ID_RE.search(url)
        return m.group(1).lower() if m else None
    return None


def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    规范化 URL：去掉 www.、末尾 /、UTM 等追踪参数、锚点。
    返回 netloc+path 小写，无则返回 None。
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").replace("www.", "").lower()
    path = (parsed.path or "").rstrip("/")
    if not netloc and not path:
        return None
    return f"{netloc}{path}".lower()


# 用于 merge 时判定来源优先级（整数越大越优先）
_SOURCE_PRIORITY_MAP = {
    "scholar": 10,
    "serpapi_scholar": 10,
    "semantic": 7,
    "semantic_bulk": 6,
    "ncbi": 5,
    "tavily": 3,
    "web": 3,
    "google": 2,
    "serpapi_google": 2,
}


def _source_priority(source: str) -> int:
    """返回来源优先级（越大越优先）。semantic_snippet 不参与 merge，不在此列。"""
    return _SOURCE_PRIORITY_MAP.get(source or "", 1)


# 合并时从 loser 补入 winner 的空字段
_MERGE_FIELDS = [
    "pdf_url", "doi", "abstract", "authors", "year",
    "pmid", "arxiv_id", "corpus_id", "venue", "cited_by",
]


def merge_metadata_by_priority(winner_hit: Dict[str, Any], loser_hit: Dict[str, Any]) -> None:
    """
    将 loser_hit 中有价值且 winner 中为空的字段补入 winner（原地修改 winner）。
    url 仅在 winner 完全无 url 时补入。
    """
    w_meta = winner_hit.get("metadata") or {}
    l_meta = loser_hit.get("metadata") or {}
    for key in _MERGE_FIELDS:
        if not w_meta.get(key) and l_meta.get(key) is not None:
            w_meta[key] = l_meta[key]
    if not w_meta.get("url") and l_meta.get("url"):
        w_meta["url"] = l_meta["url"]
    winner_hit["metadata"] = w_meta
    # 顶层字段（部分 provider 把 url/title 放在顶层）
    for key in ("url", "title", "doi", "abstract"):
        if not winner_hit.get(key) and loser_hit.get(key):
            winner_hit[key] = loser_hit[key]


def _fingerprint(text: str) -> str:
    """稳定且抗干扰的 hash 指纹（确定性，跨进程一致；抗空格/标点差异）"""
    if not text or not isinstance(text, str):
        return ""
    t = re.sub(r"[\W_]+", "", text.lower())[:1024]
    if not t:
        return ""
    return hashlib.md5(t.encode("utf-8")).hexdigest()


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

    # L3: 实际 API 请求（urlopen 自带 timeout，无需线程池）
    def _request() -> Dict[str, Any]:
        qs = urlencode({"query.bibliographic": title, "rows": 3})
        req = Request(
            f"{_CROSSREF_API}?{qs}",
            headers={"User-Agent": "DeepSea-RAG/1.0", "Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(req, timeout=_CROSSREF_TIMEOUT_SECONDS) as resp:
                if getattr(resp, "status", 200) != 200:
                    return {}
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, socket.timeout, Exception):
            return {}

    data = _request()
    items = (data.get("message") or {}).get("items") or []
    if not items:
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
        if meta.get("source") == "semantic_snippet":
            continue
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

def _extract_local_arxiv(chunk: Any) -> Optional[str]:
    """从本地 chunk（dict 或 EvidenceChunk）提取 arxiv_id。"""
    if isinstance(chunk, dict):
        return extract_arxiv_id(chunk.get("metadata"))
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    return extract_arxiv_id(meta)


def cross_source_dedup(
    local_chunks: List[Any],
    web_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    拦截网络搜索中与本地文库重叠的文献。

    通过 DOI / arXiv ID / Title 黑名单过滤 web_hits。snippet 旁路放行。
    单遍遍历保持 Reranker 原始相关性顺序，避免 snippet 被挪到末尾遭截断。
    """
    if not web_hits:
        return web_hits

    non_snippets = [h for h in web_hits if (h.get("metadata") or {}).get("source", "") != "semantic_snippet"]
    if not local_chunks or not non_snippets:
        return web_hits

    store = _get_paper_meta_store()
    local_dois: Set[str] = set(store.all_dois())
    local_titles: Set[str] = set(store.all_normalized_titles())
    local_arxivs: Set[str] = set()

    for c in local_chunks:
        doi_raw, title_raw, _ = _extract_local_fields(c)
        doi = normalize_doi(doi_raw)
        if doi:
            local_dois.add(doi)
        nt = normalize_title(title_raw)
        if nt:
            local_titles.add(nt)
        arxiv = _extract_local_arxiv(c)
        if arxiv:
            local_arxivs.add(arxiv)

    crossref_lookups, crossref_resolved = _enrich_web_hits_missing_doi(non_snippets, local_titles)

    filtered_hits: List[Dict[str, Any]] = []
    removed = 0
    for hit in web_hits:
        meta = hit.get("metadata") or {}
        if meta.get("source") == "semantic_snippet":
            filtered_hits.append(hit)
            continue
        web_doi = normalize_doi(meta.get("doi"))
        web_title = normalize_title(meta.get("title") or hit.get("title") or "")
        web_arxiv = extract_arxiv_id(meta)
        if (web_doi and web_doi in local_dois) or (web_title and web_title in local_titles) or (web_arxiv and web_arxiv in local_arxivs):
            removed += 1
            continue
        filtered_hits.append(hit)

    if removed > 0:
        logger.info(
            "cross_source_dedup: removed=%d, local_doi=%d, local_title=%d, local_arxiv=%d, crossref_lookups=%d, crossref_resolved=%d",
            removed,
            len(local_dois),
            len(local_titles),
            len(local_arxivs),
            crossref_lookups,
            crossref_resolved,
        )
    return filtered_hits


# ── 指纹去重 + 单文档上限 ─────────────────────────────

def dedup_and_diversify(
    candidates: List[Dict],
    per_doc_cap: int = 3,
    top_doc_cap: int = 8,
) -> List[Dict]:
    """
    去重 + 单文档上限 + 重复时融合元数据（merge_metadata_by_priority）。
    doc_id 兜底 url/doi，防止网页切片绕过 per_doc_cap。
    榜首特权：排名第一的文档（首个出现的非空 doc_id）允许最多 top_doc_cap 条，
    其余文档受 per_doc_cap 限制，兼顾核心证据深度与多源交叉验证。
    """
    seen_fingerprints: Dict[str, Dict] = {}
    per_doc_count: Dict[str, int] = defaultdict(int)
    output: List[Dict] = []
    best_doc_id: Optional[str] = None

    for c in candidates:
        text = c.get("content") or c.get("raw_content") or ""
        fp = _fingerprint(text)
        meta = c.get("metadata", {}) or {}
        doc_id = (
            meta.get("doc_id")
            or meta.get("paper_id")
            or normalize_url(meta.get("url"))
            or normalize_doi(meta.get("doi"))
            or ""
        )

        if doc_id and best_doc_id is None:
            best_doc_id = doc_id

        if fp and fp in seen_fingerprints:
            winner = seen_fingerprints[fp]
            merge_metadata_by_priority(winner_hit=winner, loser_hit=c)
            continue

        current_cap = top_doc_cap if (doc_id and doc_id == best_doc_id) else per_doc_cap
        if current_cap > 0 and doc_id and per_doc_count[doc_id] >= current_cap:
            continue

        output.append(c)
        if fp:
            seen_fingerprints[fp] = c
        if doc_id:
            per_doc_count[doc_id] += 1

    return output
