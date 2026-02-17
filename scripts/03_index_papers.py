#!/usr/bin/env python
"""步骤3: 切块 + 向量化 + 入库（结构化切块，enriched.json）"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)
from src.indexing.milvus_ops import milvus
from src.indexing.embedder import embedder
from src.chunking.chunker import Chunk, ChunkConfig, chunk_blocks


def truncate_content(content: str, max_length: int = 65000) -> str:
    """截断超长内容"""
    if len(content) > max_length:
        return content[:max_length]
    return content


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - 向量入库（结构化切块）")
    logger.info("=" * 60)

    parsed_dir = settings.path.parsed
    enriched_files = list(parsed_dir.rglob("enriched.json"))

    logger.info(f"解析目录: {parsed_dir}")
    logger.info(f"找到 enriched.json: {len(enriched_files)} 个")

    if not enriched_files:
        logger.warning("未找到 enriched.json")
        logger.warning("请先运行: python scripts/02_parse_papers.py")
        return

    cs = getattr(settings, "chunk", None)
    chunk_cfg = ChunkConfig(
        target_chars=getattr(cs, "target_chars", 1000) if cs else 1000,
        min_chars=getattr(cs, "min_chars", 200) if cs else 200,
        max_chars=getattr(cs, "max_chars", 1800) if cs else 1800,
        overlap_sentences=getattr(cs, "overlap_sentences", 2) if cs else 2,
        table_rows_per_chunk=getattr(cs, "table_rows_per_chunk", 10) if cs else 10,
    )

    artifact = {
        "run_id": run_id,
        "input_count": len(enriched_files),
        "total_chunks": 0,
        "inserted_count": 0,
        "truncated_count": 0,
        "collections": {},
    }

    collection_name = settings.collection.global_
    all_data: List[Dict] = []

    logger.info("处理文件并生成向量...")
    for i, json_path in enumerate(enriched_files, 1):
        logger.info(f"[{i}/{len(enriched_files)}] {json_path.parent.name}")

        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_id = doc.get("doc_id", json_path.parent.name)
        content_flow = doc.get("content_flow", [])

        chunks = chunk_blocks(content_flow, doc_id=doc_id, config=chunk_cfg)
        logger.info(f"chunks: {len(chunks)}")

        for c in chunks:
            text = truncate_content(c.text)
            if len(c.text) > 65000:
                artifact["truncated_count"] += 1

            meta = c.meta or {}
            page_range = meta.get("page_range", [0, 0])
            page = page_range[0] if isinstance(page_range, (list, tuple)) else meta.get("page", 0)

            all_data.append({
                "paper_id": doc_id,
                "chunk_id": c.chunk_id,
                "content": text,
                "raw_content": text,
                "domain": "global",
                "content_type": c.content_type,
                "chunk_type": ",".join(meta.get("block_types", []))[:64] or "paragraph",
                "section_path": str(meta.get("section_path", ""))[:512],
                "page": int(page) if isinstance(page, (int, float)) else 0,
                "_text_for_embed": text,
            })

        artifact["total_chunks"] += len(chunks)

    if not all_data:
        logger.warning("无有效内容")
        return

    # 批量生成向量
    logger.info(f"生成向量 (共 {len(all_data)} 个 chunks)...")
    texts = [d["_text_for_embed"] for d in all_data]

    batch_size = 32
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]

        embeddings = embedder.encode(batch_texts)

        for k, idx in enumerate(range(batch_start, batch_end)):
            all_data[idx]["dense_vector"] = embeddings["dense"][k].tolist()
            sparse_row = embeddings["sparse"]._getrow(k).tocoo()
            all_data[idx]["sparse_vector"] = {int(col): float(val) for col, val in zip(sparse_row.col, sparse_row.data)}

        logger.info(f"向量化: {batch_end}/{len(texts)}")

    for d in all_data:
        del d["_text_for_embed"]

    # 批量入库
    logger.info(f"入库到 {collection_name}...")
    insert_batch_size = 100
    for batch_start in range(0, len(all_data), insert_batch_size):
        batch_end = min(batch_start + insert_batch_size, len(all_data))
        batch = all_data[batch_start:batch_end]

        milvus.insert(collection_name, batch)
        artifact["inserted_count"] += len(batch)
        logger.info(f"插入: {batch_end}/{len(all_data)}")

    count = milvus.count(collection_name)
    artifact["collections"][collection_name] = count

    artifact_path = settings.path.artifacts / f"03_index_{run_id}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"入库完成: {artifact['inserted_count']} 条")
    logger.info(f"Collection '{collection_name}' 总数: {count}")
    if artifact["truncated_count"] > 0:
        logger.warning(f"截断: {artifact['truncated_count']} 条")
    logger.info(f"产物已保存: {artifact_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
