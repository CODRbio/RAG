"""
Context Packing 模块
- QA 模式：短答场景，top chunks + 引用头
- Longform 模式：报告/综述场景，按 section 聚类，每节选 top N
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Any


def pack_qa_context(
    chunks: List[Dict],
    top_n: int = 6,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    QA 模式：top chunks + 引用头
    
    Args:
        chunks: 检索结果（含 content 和 metadata）
        top_n: 取前 N 个 chunk
        separator: chunk 间分隔符
    
    Returns:
        拼接后的 context 字符串
    """
    context_parts = []
    for c in chunks[:top_n]:
        content = c.get("content") or c.get("raw_content") or ""
        if not content:
            continue
        meta = c.get("metadata", {}) or {}
        doc_id = meta.get("doc_id") or meta.get("paper_id") or "N/A"
        page = meta.get("page_range") or [meta.get("page", 0)]
        page_str = f"p.{page[0]}" if isinstance(page, (list, tuple)) and page else f"p.{meta.get('page', 0)}"
        section_path = meta.get("section_path") or ""
        header = f"[{doc_id} | {page_str} | {section_path}]"
        context_parts.append(f"{header}\n{content}")
    return separator.join(context_parts)


def pack_longform_context(
    chunks: List[Dict],
    chunks_per_section: int = 3,
    max_bundles: int = 8,
) -> List[Dict[str, Any]]:
    """
    Longform 模式：按 section 聚类，每节选 top N
    
    Args:
        chunks: 检索结果（含 content 和 metadata）
        chunks_per_section: 每 section 保留的 chunk 数
        max_bundles: 最多保留的 section 数
    
    Returns:
        [{"section": str, "evidence": [str, ...]}, ...]
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        section = meta.get("section_path") or "(root)"
        groups[section].append(c)

    bundles = []
    for section, section_chunks in groups.items():
        if len(bundles) >= max_bundles:
            break
        top_in_section = section_chunks[:chunks_per_section]
        evidence = [
            (c.get("content") or c.get("raw_content") or "").strip()
            for c in top_in_section
        ]
        evidence = [e for e in evidence if e]
        if evidence:
            bundles.append({"section": section, "evidence": evidence})
    return bundles
