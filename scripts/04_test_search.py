#!/usr/bin/env python
"""步骤4: 检索测试（调试版）"""

import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)
from src.indexing.milvus_ops import milvus
from src.indexing.embedder import embedder
from pymilvus import AnnSearchRequest, RRFRanker


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - 检索测试")
    logger.info("=" * 60)

    collection_name = settings.collection.global_
    test_queries = [
        "深海热液喷口",
        "海洋生物多样性",
        "深海探测技术",
    ]

    artifact = {
        "run_id": run_id,
        "collection": collection_name,
        "queries": []
    }

    # 检查 collection
    count = milvus.count(collection_name)
    logger.info("Collection: %s, 文档数量: %s", collection_name, count)

    if count == 0:
        logger.warning("Collection 为空")
        logger.warning("请先运行: python scripts/03_index_papers.py")
        return

    for query in test_queries:
        logger.info("Query: %s", query)

        query_result = {"query": query, "levels": {}}

        # Level 1: Embedding 检查
        logger.info("[Level 1] Embedding 输出")
        emb = embedder.encode([query])
        dense_vec = emb["dense"][0]
        sparse_vec = emb["sparse"]._getrow(0)

        dense_dim = len(dense_vec)
        sparse_nnz = sparse_vec.nnz

        logger.info("dense_dim: %s, sparse_nnz: %s", dense_dim, sparse_nnz)

        query_result["levels"]["embedding"] = {
            "dense_dim": dense_dim,
            "sparse_nnz": sparse_nnz,
            "status": "ok" if dense_dim == 1024 and sparse_nnz > 0 else "warn"
        }

        # Level 2: Milvus hybrid_search
        logger.info("[Level 2] Hybrid Search")

        dense_req = AnnSearchRequest(
            data=[dense_vec.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=10,
        )

        # 转换 sparse vector 格式
        sparse_coo = sparse_vec.tocoo()
        sparse_dict = {int(col): float(val) for col, val in zip(sparse_coo.col, sparse_coo.data)}

        sparse_req = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=10,
        )

        results = milvus.hybrid_search(
            collection=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=settings.search.rrf_k),
            limit=5,
            output_fields=["paper_id", "chunk_id", "content", "section_path", "page"]
        )

        hits = results[0] if results else []
        logger.info("返回结果: %s 条", len(hits))

        query_result["levels"]["hybrid_search"] = {
            "hit_count": len(hits),
            "status": "ok" if len(hits) >= 3 else "warn"
        }

        # 显示结果
        hit_details = []
        for j, hit in enumerate(hits[:5], 1):
            paper_id = hit.entity.get("paper_id", "N/A")
            chunk_id = hit.entity.get("chunk_id", "N/A")
            content = hit.entity.get("content", "")[:100]
            score = hit.distance

            print(f"\n  [{j}] score={score:.4f}")
            print(f"      paper_id: {paper_id}")
            print(f"      chunk_id: {chunk_id}")
            print(f"      content: {content}...")

            hit_details.append({
                "rank": j,
                "score": score,
                "paper_id": paper_id,
                "chunk_id": chunk_id
            })

        query_result["hits"] = hit_details

        # Level 3: Rerank（可选）
        logger.info("[Level 3] Rerank")
        if hits:
            docs = [hit.entity.get("content", "") for hit in hits]
            try:
                reranked = embedder.rerank(query, docs, top_k=3)
                logger.info("Rerank 结果: %s 条", len(reranked))

                rerank_details = []
                for r in reranked:
                    logger.info("原排名 %s -> 新分数 %s", r.index + 1, f"{r.score:.4f}")
                    rerank_details.append({
                        "original_rank": r.index + 1,
                        "rerank_score": r.score
                    })

                query_result["levels"]["rerank"] = {
                    "status": "ok",
                    "results": rerank_details
                }
            except Exception as e:
                logger.warning("Rerank 失败: %s", e)
                query_result["levels"]["rerank"] = {"status": "fail", "error": str(e)}
        else:
            query_result["levels"]["rerank"] = {"status": "skip", "reason": "no hits"}

        artifact["queries"].append(query_result)

    # 保存 artifact
    artifact_path = settings.path.artifacts / f"04_search_{run_id}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("检索测试完成")
    logger.info("产物已保存: %s", artifact_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
