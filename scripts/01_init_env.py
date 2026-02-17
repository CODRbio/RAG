#!/usr/bin/env python
"""步骤1: 初始化环境"""

import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)
from src.indexing.milvus_ops import milvus
from src.indexing.embedder import embedder


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - 环境初始化")
    logger.info("=" * 60)

    settings.print_info()

    artifact = {"run_id": run_id, "steps": []}

    # 1. 测试 Milvus
    logger.info("[1/3] 测试 Milvus 连接...")
    try:
        collections = milvus.client.list_collections()
        logger.info(f"Milvus 连接成功: {settings.milvus.uri}")
        artifact["steps"].append({"step": "milvus_connect", "status": "ok"})
    except Exception as e:
        logger.error(f"连接失败: {e}")
        logger.error("请运行: docker compose --profile dev up -d")
        logger.error("然后执行: bash scripts/00_healthcheck_docker.sh")
        artifact["steps"].append({"step": "milvus_connect", "status": "fail", "error": str(e)})
        _save_artifact(artifact, run_id)
        return

    # 2. 创建 Collections
    logger.info("[2/3] 初始化 Collections...")
    milvus.init_all_collections()
    artifact["steps"].append({
        "step": "init_collections",
        "status": "ok",
        "collections": settings.collection.all()
    })

    # 3. 测试 Embedding
    logger.info("[3/3] 测试 Embedding 模型...")
    try:
        result = embedder.encode(["测试文本"])
        dense_dim = len(result["dense"][0])
        sparse_nnz = result["sparse"]._getrow(0).nnz
        logger.info(f"BGE-M3 正常 (dense_dim={dense_dim}, sparse_nnz={sparse_nnz})")

        if dense_dim != 1024:
            logger.warning(f"dense_dim 应为 1024，实际为 {dense_dim}")

        artifact["steps"].append({
            "step": "embedding_test",
            "status": "ok",
            "dense_dim": dense_dim,
            "sparse_nnz": sparse_nnz
        })
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        artifact["steps"].append({"step": "embedding_test", "status": "fail", "error": str(e)})
        _save_artifact(artifact, run_id)
        return

    _save_artifact(artifact, run_id)

    logger.info("=" * 60)
    logger.info("环境初始化完成！")
    logger.info("=" * 60)


def _save_artifact(data: dict, run_id: str):
    path = settings.path.artifacts / f"01_init_env_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"产物已保存: {path}")


if __name__ == "__main__":
    main()
