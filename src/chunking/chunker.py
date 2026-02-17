"""
结构化切块模块
基于 block + section_path + 长度约束
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# Block 可以是 dict（JSON）或带 heading_path/text/table_data/figure_data 的对象


@dataclass
class ChunkConfig:
    target_chars: int = 1000
    min_chars: int = 200
    max_chars: int = 1800
    overlap_sentences: int = 2
    table_rows_per_chunk: int = 10


@dataclass
class Chunk:
    chunk_id: str
    text: str
    content_type: str  # text | table | image_caption
    meta: dict = field(default_factory=dict)


def _normalize_text(t: Optional[str]) -> str:
    if t is None:
        return ""
    s = str(t).strip()
    return s


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


def _section_path(block: dict | Any) -> str:
    hp = getattr(block, "heading_path", None) or block.get("heading_path", []) or []
    return " > ".join(str(h) for h in hp)


def _block_text(block: dict | Any) -> str:
    return _normalize_text(getattr(block, "text", None) or block.get("text"))


def _block_type(block: dict | Any) -> str:
    bt = getattr(block, "block_type", None)
    if bt is not None:
        return getattr(bt, "value", str(bt)) if hasattr(bt, "value") else str(bt)
    return str(block.get("block_type", "text"))


def _page_index(block: dict | Any) -> int:
    return getattr(block, "page_index", None) or block.get("page_index", 0)


def _block_id(block: dict | Any) -> str:
    return getattr(block, "block_id", None) or block.get("block_id", "")


def _block_bbox(block: dict | Any) -> Optional[list]:
    bbox = getattr(block, "bbox", None) or block.get("bbox")
    if bbox is not None:
        return list(bbox) if isinstance(bbox, (list, tuple)) else None
    return None


def _sentence_tokenize(text: str) -> list[str]:
    """简单句子切分：按 . ! ? 。！？；; 分割"""
    if not text.strip():
        return []
    pattern = r'(?<=[.!?。！？；;])\s+'
    parts = re.split(pattern, text)
    result = []
    for p in parts:
        p = p.strip()
        if p:
            result.append(p)
    if not result:
        result = [text.strip()] if text.strip() else []
    return result


def _generate_stable_id(doc_id: str, section_path: str, page_start: int, content_prefix: str, chunk_index: int = 0) -> str:
    raw = f"{doc_id}:{section_path}:{page_start}:{content_prefix[:24]}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _merge_blocks_text(blocks: list) -> str:
    parts = []
    for b in blocks:
        t = _block_text(b)
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _merge_metadata(blocks: list, doc_id: str, content_type: str = "text") -> dict:
    if not blocks:
        return {"doc_id": doc_id, "page_range": [0, 0], "section_path": "", "block_types": [], "source_uri": ""}
    pages = [_page_index(b) for b in blocks]
    section = _section_path(blocks[0])
    block_types = list(dict.fromkeys(_block_type(b) for b in blocks))
    bboxes = []
    for b in blocks:
        bb = _block_bbox(b)
        if bb:
            bboxes.append(bb)
    meta = {
        "doc_id": doc_id,
        "page_range": [min(pages), max(pages)],
        "section_path": section,
        "block_types": block_types,
        "source_uri": f"{doc_id}#{_block_id(blocks[0])}",
    }
    if bboxes:
        meta["bbox"] = bboxes
    return meta


def _split_long_block_by_sentences(
    text: str, meta: dict, doc_id: str, target: int, overlap_sent: int
) -> list[Chunk]:
    sentences = _sentence_tokenize(text)
    if not sentences:
        if text.strip():
            return [Chunk(chunk_id=_generate_stable_id(doc_id, meta.get("section_path", ""), meta.get("page_range", [0, 0])[0], text[:32]), text=text.strip(), content_type="text", meta=meta)]
        return []
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(sentences):
        chunk_sents = []
        chunk_len = 0
        while i < len(sentences) and chunk_len + len(sentences[i]) <= target:
            chunk_sents.append(sentences[i])
            chunk_len += len(sentences[i])
            i += 1
        if not chunk_sents:
            chunk_sents = [sentences[i]]
            chunk_len = len(sentences[i])
            i += 1
        chunk_text_val = " ".join(chunk_sents)
        m = meta.copy()
        cid = _generate_stable_id(doc_id, m.get("section_path", ""), m.get("page_range", [0, 0])[0], chunk_text_val[:32], chunk_idx)
        chunks.append(Chunk(chunk_id=cid, text=chunk_text_val, content_type="text", meta=m))
        chunk_idx += 1
        i = max(i - overlap_sent, i - len(chunk_sents) + 1)
    return chunks


def _finalize_buffer(
    buffer: list, doc_id: str, target: int, max_c: int, overlap_sent: int
) -> list[Chunk]:
    text = _merge_blocks_text(buffer)
    if _is_blank(text):
        return []
    meta = _merge_metadata(buffer, doc_id, "text")
    if len(text) > max_c:
        return _split_long_block_by_sentences(text, meta, doc_id, target, overlap_sent)
    cid = _generate_stable_id(doc_id, meta.get("section_path", ""), meta.get("page_range", [0, 0])[0], text[:32])
    return [Chunk(chunk_id=cid, text=text, content_type="text", meta=meta)]


def chunk_table_block(block: dict | Any, doc_id: str, max_chars: int, rows_per_chunk: int = 10) -> list[Chunk]:
    td = getattr(block, "table_data", None) or block.get("table_data")
    if not td:
        return []
    md = td.get("markdown", "") if isinstance(td, dict) else getattr(td, "markdown", "") or ""
    if not md:
        structured = td.get("structured", []) if isinstance(td, dict) else getattr(td, "structured", [])
        if structured:
            header = structured[0] if structured else []
            rows = structured[1:]
            lines = ["| " + " | ".join(str(c) for c in header) + " |", "|" + "---|" * len(header)]
            for r in rows[:50]:
                lines.append("| " + " | ".join(str(c) for c in r[: len(header)]) + " |")
            md = "\n".join(lines)
    if not md.strip():
        return []
    section_path = _section_path(block)
    page = _page_index(block)
    meta = {"doc_id": doc_id, "page_range": [page, page], "section_path": section_path, "block_types": ["table"], "source_uri": f"{doc_id}#{_block_id(block)}"}
    if len(md) <= max_chars:
        cid = _generate_stable_id(doc_id, section_path, page, md[:32])
        return [Chunk(chunk_id=cid, text=md, content_type="table", meta={**meta, "table_id": _block_id(block)})]
    lines = md.split("\n")
    header_lines = lines[:2] if len(lines) >= 2 else lines
    data_lines = lines[2:] if len(lines) > 2 else []
    chunks = []
    for start in range(0, len(data_lines), rows_per_chunk):
        chunk_lines = header_lines + data_lines[start : start + rows_per_chunk]
        chunk_text_val = "\n".join(chunk_lines)
        cid = _generate_stable_id(doc_id, section_path, page, chunk_text_val[:32], start // rows_per_chunk)
        chunks.append(Chunk(chunk_id=cid, text=chunk_text_val, content_type="table", meta={**meta, "table_id": _block_id(block)}))
    return chunks


def chunk_image_block(block: dict | Any, doc_id: str) -> list[Chunk]:
    fd = getattr(block, "figure_data", None) or block.get("figure_data")
    enrichment = getattr(block, "enrichment", None) or block.get("enrichment") or {}
    caption = None
    if fd:
        caption = fd.get("caption") if isinstance(fd, dict) else getattr(fd, "caption", None)
    if not caption:
        interp = enrichment.get("interpretation") if isinstance(enrichment, dict) else getattr(enrichment, "interpretation", None)
        if interp:
            caption = interp.get("description") if isinstance(interp, dict) else getattr(interp, "description", None)
    if not caption or not str(caption).strip():
        return []
    section_path = _section_path(block)
    page = _page_index(block)
    meta = {"doc_id": doc_id, "page_range": [page, page], "section_path": section_path, "block_types": ["figure"], "source_uri": f"{doc_id}#{_block_id(block)}", "figure_id": _block_id(block)}
    cid = _generate_stable_id(doc_id, section_path, page, str(caption)[:32])
    return [Chunk(chunk_id=cid, text=str(caption).strip(), content_type="image_caption", meta=meta)]


def _match_claims_to_chunk(chunk: Chunk, claims: list[dict]) -> list[str]:
    """
    将文档级 claims 匹配到 chunk，返回匹配的 claim_id 列表。

    匹配规则：claim 的 source_block_ids 与 chunk 的 source_uri 有交集。
    """
    if not claims:
        return []
    # 从 chunk meta 提取 block_id（source_uri 格式: doc_id#block_id）
    source_uri = chunk.meta.get("source_uri", "")
    chunk_block_id = source_uri.split("#", 1)[1] if "#" in source_uri else ""
    if not chunk_block_id:
        return []

    matched = []
    for claim in claims:
        block_ids = claim.get("source_block_ids", [])
        if chunk_block_id in block_ids:
            matched.append(claim.get("claim_id", ""))
    return [m for m in matched if m]


def chunk_blocks(
    blocks: list,
    doc_id: str,
    config: Optional[ChunkConfig] = None,
    claims: Optional[list[dict]] = None,
) -> list[Chunk]:
    cfg = config or ChunkConfig()
    target, min_c, max_c, overlap_sent = cfg.target_chars, cfg.min_chars, cfg.max_chars, cfg.overlap_sentences
    chunks = []
    buffer = []
    buffer_len = 0
    current_section = None

    for block in blocks:
        bt = _block_type(block)

        if bt == "table":
            if buffer_len > 0:
                chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)
                buffer, buffer_len, current_section = [], 0, None
            chunks += chunk_table_block(block, doc_id, max_c, cfg.table_rows_per_chunk)
            continue

        if bt == "figure":
            if buffer_len > 0:
                chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)
                buffer, buffer_len, current_section = [], 0, None
            chunks += chunk_image_block(block, doc_id)
            continue

        if bt in ("caption", "footnote", "formula"):
            continue

        text = _block_text(block)
        if _is_blank(text):
            continue

        section = _section_path(block)

        if section != current_section and buffer_len >= min_c:
            chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)
            buffer, buffer_len = [], 0

        current_section = section

        if buffer_len + len(text) <= max_c:
            buffer.append(block)
            buffer_len += len(text)
            if buffer_len >= target:
                chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)
                buffer, buffer_len, current_section = [], 0, None
        else:
            if buffer_len > 0:
                chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)
                buffer, buffer_len = [], 0
            meta = _merge_metadata([block], doc_id)
            chunks += _split_long_block_by_sentences(text, meta, doc_id, target, overlap_sent)
            current_section = None

    if buffer_len > 0:
        chunks += _finalize_buffer(buffer, doc_id, target, max_c, overlap_sent)

    # 注入 claims 到 chunk.meta
    if claims:
        for chunk in chunks:
            matched = _match_claims_to_chunk(chunk, claims)
            if matched:
                chunk.meta["claims"] = matched

    return chunks
