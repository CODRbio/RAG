"""
去重与多样性模块
- 指纹去重（避免相邻 chunk 高度相似）
- 单文档上限（per_doc_cap，防止单 doc 垄断 top N）
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict


def _fingerprint(text: str) -> str:
    """简单 hash 指纹（用于去重）"""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()[:512]
    return str(hash(t))


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
