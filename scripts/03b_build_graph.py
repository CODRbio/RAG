#!/usr/bin/env python
"""步骤3b: 构建 HippoRAG 知识图谱"""

import sys
import json
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger
from src.graph.entity_extractor import ExtractorConfig
from src.graph.hippo_rag import HippoRAG

logger = get_logger(__name__)


def _build_extractor_config() -> ExtractorConfig:
    cfg = settings.graph_entity_extraction
    return ExtractorConfig(
        strategy=cfg.strategy,
        fallback=cfg.fallback,
        ontology_path=cfg.ontology_path,
        gliner_model=cfg.gliner_model,
        gliner_threshold=cfg.gliner_threshold,
        gliner_device=cfg.gliner_device,
        llm_provider=cfg.llm_provider,
        llm_max_tokens=cfg.llm_max_tokens,
    )


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("知识库 - HippoRAG 图谱构建")
    logger.info("=" * 60)

    parsed_dir = settings.path.parsed
    graph_path = settings.path.data / "hippo_graph.json"

    json_files = list(parsed_dir.rglob("enriched.json"))
    if not json_files:
        logger.warning("未找到解析结果: %s", parsed_dir)
        logger.warning("请先运行: python scripts/02_parse_papers.py")
        return

    logger.info("解析目录: %s", parsed_dir)
    logger.info("找到文档: %d 个", len(json_files))

    ext_cfg = _build_extractor_config()
    logger.info("实体抽取策略: %s (fallback=%s)", ext_cfg.strategy, ext_cfg.fallback)

    hippo = HippoRAG(extractor_config=ext_cfg)
    hippo.build_from_parsed_docs(parsed_dir)

    stats = hippo.stats()
    logger.info(
        "图谱统计: 总节点=%s, 总边数=%s, 实体数=%s, Chunk数=%s, 策略=%s",
        stats["total_nodes"], stats["total_edges"],
        stats["entity_count"], stats["chunk_count"],
        stats["extraction_strategy"],
    )

    hippo.save(graph_path)

    artifact = {
        "run_id": run_id,
        "graph_path": str(graph_path),
        "extraction_strategy": ext_cfg.strategy,
        "stats": stats,
    }
    artifact_path = settings.path.artifacts / f"03b_graph_{run_id}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("图谱构建完成")
    logger.info("图谱文件: %s", graph_path)
    logger.info("产物已保存: %s", artifact_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
