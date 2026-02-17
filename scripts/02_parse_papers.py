#!/usr/bin/env python
"""步骤2: 解析 PDF 文件（EnrichedDoc 完整 schema）"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="解析 PDF 为 EnrichedDoc")
    parser.add_argument("--skip-enrichment", action="store_true", help="跳过 LLM 增强（表格摘要、图表解读）")
    parser.add_argument("--skip-table-enrichment", action="store_true", help="跳过表格 LLM 增强")
    parser.add_argument("--skip-figure-enrichment", action="store_true", help="跳过图像 LLM 增强")
    parser.add_argument("--llm-text-provider", type=str, help="覆盖文本 LLM provider")
    parser.add_argument("--llm-vision-provider", type=str, help="覆盖图像 LLM provider")
    parser.add_argument("--llm-text-model", type=str, help="覆盖文本 LLM model")
    parser.add_argument("--llm-vision-model", type=str, help="覆盖图像 LLM model")
    parser.add_argument("--llm-vision-concurrency", type=int, help="图像 LLM 并发数")
    parser.add_argument("--llm-text-max-tokens", type=int, help="覆盖文本 LLM max_tokens")
    parser.add_argument("--llm-vision-max-tokens", type=int, help="覆盖图像 LLM max_tokens")
    parser.add_argument("--llm-json-repair-max-tokens", type=int, help="覆盖 JSON 修复 max_tokens")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings.path.ensure_dirs()

    logger.info("=" * 60)
    logger.info("深海科研知识库 - PDF 解析（EnrichedDoc）")
    logger.info("=" * 60)

    input_dir = settings.path.raw_papers
    output_base = settings.path.parsed

    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"找到 PDF: {len(pdf_files)} 个")

    if not pdf_files:
        logger.warning("未找到 PDF 文件")
        logger.warning(f"请将 PDF 放入: {input_dir}")
        return

    # 加载 parser 与 LLM
    try:
        from src.parser.pdf_parser import PDFProcessor, ParserConfig
    except ImportError as e:
        logger.error(f"pdf_parser 导入失败: {e}")
        return

    config_path = Path(__file__).parent.parent / "config" / "rag_config.json"
    if config_path.exists():
        try:
            config = ParserConfig.from_json(config_path)
        except Exception:
            config = ParserConfig()
    else:
        config = ParserConfig()
    if args.llm_text_provider:
        config.llm_text_provider = args.llm_text_provider
    if args.llm_vision_provider:
        config.llm_vision_provider = args.llm_vision_provider
    if args.llm_text_model:
        config.llm_text_model = args.llm_text_model
    if args.llm_vision_model:
        config.llm_vision_model = args.llm_vision_model
    if args.llm_vision_concurrency is not None:
        config.llm_vision_concurrency = max(1, int(args.llm_vision_concurrency))
    if args.llm_text_max_tokens is not None:
        config.llm_text_max_tokens = max(1, int(args.llm_text_max_tokens))
    if args.llm_vision_max_tokens is not None:
        config.llm_vision_max_tokens = max(1, int(args.llm_vision_max_tokens))
    if args.llm_json_repair_max_tokens is not None:
        config.llm_json_repair_max_tokens = max(1, int(args.llm_json_repair_max_tokens))
    if args.skip_table_enrichment:
        config.enrich_tables = False
    if args.skip_figure_enrichment:
        config.enrich_figures = False

    llm_manager = None
    if not args.skip_enrichment:
        try:
            from src.llm import LLMManager
            llm_manager = LLMManager.from_json(str(config_path))
        except Exception as e:
            logger.warning(f"LLM 未加载，将跳过增强: {e}")

    processor = PDFProcessor(config=config, llm_manager=llm_manager)

    artifact = {
        "run_id": run_id,
        "input_count": len(pdf_files),
        "success_count": 0,
        "fail_count": 0,
        "results": [],
        "badcases": [],
    }

    logger.info(f"开始解析... (skip_enrichment={args.skip_enrichment})")
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_path.name}")

        try:
            output_dir = output_base / pdf_path.stem
            doc = processor.process(
                pdf_path,
                output_dir=output_dir,
                skip_enrichment=args.skip_enrichment,
            )

            n_blocks = len(doc.content_flow)
            n_tables = sum(1 for b in doc.content_flow if b.block_type.value == "table")
            n_figures = sum(1 for b in doc.content_flow if b.block_type.value == "figure")

            status = "ok"
            text_len = sum(len(b.text or "") for b in doc.content_flow)
            if text_len < 2000:
                status = "badcase"
                artifact["badcases"].append({
                    "paper_id": pdf_path.stem,
                    "reason": f"text_chars={text_len} < 2000",
                })

            artifact["results"].append({
                "paper_id": pdf_path.stem,
                "status": status,
                "blocks": n_blocks,
                "tables": n_tables,
                "figures": n_figures,
                "output": str(output_dir / "enriched.json"),
            })
            artifact["success_count"] += 1

            logger.info(f"blocks={n_blocks}, tables={n_tables}, figures={n_figures}")

        except Exception as e:
            logger.error(f"{e}")
            artifact["fail_count"] += 1
            artifact["results"].append({
                "paper_id": pdf_path.stem,
                "status": "fail",
                "error": str(e),
            })

    artifact_path = settings.path.artifacts / f"02_parse_{run_id}.json"
    import json
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"解析完成: {artifact['success_count']}/{artifact['input_count']} 成功")
    if artifact["badcases"]:
        logger.warning(f"Badcases: {len(artifact['badcases'])} 个")
    logger.info(f"产物已保存: {artifact_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
