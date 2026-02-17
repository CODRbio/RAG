#!/usr/bin/env python
"""步骤3b: 构建 HippoRAG 知识图谱"""

import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger
from src.graph.hippo_rag import HippoRAG

logger = get_logger(__name__)


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - HippoRAG 图谱构建")
    logger.info("=" * 60)

    parsed_dir = settings.path.parsed
    graph_path = settings.path.data / "hippo_graph.json"

    # 检查解析结果
    json_files = list(parsed_dir.rglob("enriched.json"))
    if not json_files:
        logger.warning(f"未找到解析结果: {parsed_dir}")
        logger.warning("请先运行: python scripts/02_parse_papers.py")
        return

    logger.info(f"解析目录: {parsed_dir}")
    logger.info(f"找到文档: {len(json_files)} 个")

    # 构建图谱
    hippo = HippoRAG()
    hippo.build_from_parsed_docs(parsed_dir, use_llm=False)

    # 统计
    stats = hippo.stats()
    logger.info("图谱统计: 总节点=%s, 总边数=%s, 实体数=%s, Chunk数=%s",
                stats['total_nodes'], stats['total_edges'], stats['entity_count'], stats['chunk_count'])

    # 保存
    hippo.save(graph_path)

    # 保存 artifact
    artifact = {
        "run_id": run_id,
        "graph_path": str(graph_path),
        "stats": stats
    }
    artifact_path = settings.path.artifacts / f"03b_graph_{run_id}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("图谱构建完成")
    logger.info(f"图谱文件: {graph_path}")
    logger.info(f"产物已保存: {artifact_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
