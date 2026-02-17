#!/usr/bin/env python
"""LangGraph 离线入库：解析 PDF → 切块 → 向量化 → Milvus upsert（chunk_id 主键）→ HippoRAG 建图"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.log import get_logger
from src.indexing.milvus_ops import milvus

logger = get_logger(__name__)
from src.indexing.embedder import embedder
from src.pipelines.ingestion_graph import build_ingestion_graph
from langgraph.checkpoint.sqlite import SqliteSaver


def main():
    parser = argparse.ArgumentParser(description="LangGraph 离线入库（upsert + HippoRAG）")
    parser.add_argument("--skip-enrichment", action="store_true", help="跳过 LLM 增强")
    parser.add_argument("--skip-table-enrichment", action="store_true", help="跳过表格 LLM 增强")
    parser.add_argument("--skip-figure-enrichment", action="store_true", help="跳过图像 LLM 增强")
    parser.add_argument("--build-graph", action="store_true", help="入库后构建 HippoRAG 图谱")
    parser.add_argument("--max-docs", type=int, default=None, help="最多处理 PDF 数量")
    parser.add_argument("--recreate-collection", action="store_true", help="删除并重建 Collection（v2 chunk_id 主键）")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - LangGraph 离线入库")
    logger.info("=" * 60)

    collection_name = settings.collection.global_
    milvus.create_collection(collection_name, recreate=args.recreate_collection, schema_version="v2")
    raw_papers = settings.path.raw_papers
    parsed_dir = settings.path.parsed
    graph_path = settings.path.data / "hippo_graph.json"
    artifacts_dir = settings.path.artifacts
    checkpoint_dir = settings.path.data / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ingestion.db"

    config_path = Path(__file__).resolve().parent.parent / "config" / "rag_config.json"
    try:
        from src.parser.pdf_parser import PDFProcessor, ParserConfig
    except ImportError as e:
        logger.error("pdf_parser 导入失败: %s", e)
        return 1

    parser_cfg = ParserConfig.from_json(config_path) if config_path.exists() else ParserConfig()
    if args.skip_table_enrichment:
        parser_cfg.enrich_tables = False
    if args.skip_figure_enrichment:
        parser_cfg.enrich_figures = False

    llm_manager = None
    if not args.skip_enrichment:
        try:
            from src.llm import LLMManager
            llm_manager = LLMManager.from_json(str(config_path))
        except Exception as e:
            logger.warning("LLM 未加载，将跳过增强: %s", e)
    processor = PDFProcessor(config=parser_cfg, llm_manager=llm_manager)

    chunk_config = {
        "target_chars": settings.chunk.target_chars,
        "min_chars": settings.chunk.min_chars,
        "max_chars": settings.chunk.max_chars,
        "overlap_sentences": settings.chunk.overlap_sentences,
        "table_rows_per_chunk": settings.chunk.table_rows_per_chunk,
    }

    configurable = {
        "thread_id": run_id,
        "processor": processor,
        "embedder": embedder,
        "milvus": milvus,
        "chunk_config": chunk_config,
        "collection_name": collection_name,
        "skip_enrichment": args.skip_enrichment,
        "parsed_dir": str(parsed_dir),
        "raw_papers_path": str(raw_papers),
        "artifacts_dir": str(artifacts_dir),
        "run_id": run_id,
        "graph_output_path": str(graph_path),
        "max_docs": args.max_docs,
    }

    initial_state = {
        "run_id": run_id,
        "build_graph": args.build_graph,
    }

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_ingestion_graph(checkpointer=checkpointer)
        result = graph.invoke(initial_state, config={"configurable": configurable})

    logger.info("=" * 60)
    logger.info("入库完成: %s 条 upsert, 总 chunks: %s",
                result.get('total_upserted', 0), result.get('total_chunks', 0))
    if result.get("errors"):
        logger.warning("错误: %s 条", len(result['errors']))
    if result.get("artifact_path"):
        logger.info("产物: %s", result['artifact_path'])
    if args.build_graph:
        logger.info("图谱: %s", graph_path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
