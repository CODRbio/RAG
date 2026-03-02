"""
Perplexity Sonar API 引用解析：将 citations / search_results 转为 EvidenceChunk，
纳入统一引文管理（resolve_response_citations、canvas、provider 统计等）。
"""

import hashlib
from typing import Any, Dict, List, Optional

from src.retrieval.evidence import EvidenceChunk


def _parse_year_from_date(date_val: Any) -> Optional[int]:
    """从 date 字符串或 None 解析年份。"""
    if date_val is None:
        return None
    if isinstance(date_val, int) and 1900 <= date_val <= 2100:
        return date_val
    s = str(date_val).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def parse_sonar_citations(
    citations: Optional[List[str]] = None,
    search_results: Optional[List[Dict[str, Any]]] = None,
    response_text: Optional[str] = None,
    query: str = "",
) -> List[EvidenceChunk]:
    """
    将 Perplexity API 返回的 citations 和 search_results 转为 EvidenceChunk 列表。

    - citations: URL 数组（0-indexed，正文中 [1] 对应 citations[0]）
    - search_results: 每项含 title, url, date, snippet 等
    - response_text: 可选，用于增强 snippet（当前仅用 search_results 的 snippet）
    - query: 用于日志/调试，不影响 chunk 内容

    返回的 chunk 使用 source_type="web", provider="sonar"，可直接参与
    fuse_pools_with_gap_protection 的 gap_candidates 或 agent chunk collector。
    """
    out: List[EvidenceChunk] = []
    citations = citations or []
    search_results = search_results or []

    # 优先用 search_results（含 title/snippet/date），缺项时用 citations 补 URL
    url_to_meta: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(search_results):
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        url_to_meta[url] = {
            "title": item.get("title") or "",
            "snippet": item.get("snippet") or "",
            "date": item.get("date"),
            "index": i,
        }

    # 合并：citations 顺序即 [1],[2] 顺序，用 URL 去重并保留第一次出现顺序
    seen_urls: set = set()
    ordered_urls: List[str] = []
    for u in citations:
        u = (u or "").strip()
        if u and u not in seen_urls:
            seen_urls.add(u)
            ordered_urls.append(u)
    for u in url_to_meta:
        if u not in seen_urls:
            seen_urls.add(u)
            ordered_urls.append(u)

    for idx, url in enumerate(ordered_urls):
        meta = url_to_meta.get(url) or {}
        title = meta.get("title") or ""
        snippet = (meta.get("snippet") or "").strip()
        date_val = meta.get("date")
        year = _parse_year_from_date(date_val)
        text = snippet or title or url
        if not text:
            text = url
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        chunk_id = f"sonar_{idx}_{url_hash}"
        # 稳定 doc_id 用于 doc_group_key（同 URL 同文档）
        doc_id = url
        score = 0.9
        out.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                score=score,
                source_type="web",
                doc_title=title or None,
                authors=None,
                year=year,
                url=url,
                doi=None,
                provider="sonar",
            )
        )
    return out
