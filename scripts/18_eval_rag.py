#!/usr/bin/env python
"""RAG 量化评测：检索召回 / 生成相关性 / 引用准确性"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from config.settings import settings
from src.evaluation.dataset import load_dataset
from src.evaluation.runner import evaluate_dataset
from src.llm.llm_manager import get_manager
from src.log import get_logger
from src.retrieval.service import get_retrieval_service

logger = get_logger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_mini.json", help="评测数据集路径")
    parser.add_argument("--mode", default=None, help="覆盖检索模式：local/web/hybrid")
    parser.add_argument("--top-k", type=int, default=10, help="检索返回条数")
    parser.add_argument("--max-context-chunks", type=int, default=8, help="用于生成的证据条数")
    parser.add_argument("--max-context-chars", type=int, default=800, help="每条证据最大字符数")
    parser.add_argument("--llm", type=str, default=None, help="LLM provider（默认 config 中 default）")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名或别名")
    parser.add_argument("--no-generate", action="store_true", help="仅评测检索")
    parser.add_argument("--max-cases", type=int, default=None, help="限制评测条数")
    parser.add_argument("--config", type=str, default="config/rag_config.json", help="配置文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 artifact 路径")
    args = parser.parse_args()

    settings.path.ensure_dirs()
    cases, meta = load_dataset(args.dataset)
    if args.max_cases:
        cases = cases[: args.max_cases]

    retrieval = get_retrieval_service(top_k=args.top_k)

    llm_client = None
    if not args.no_generate:
        if settings.llm.dry_run:
            logger.warning("LLM_DRY_RUN=true，跳过生成评测")
        else:
            manager = get_manager(args.config)
            llm_client = manager.get_client(args.llm)

    logger.info("评测样本数: %s", len(cases))
    logger.info("检索模式: %s", args.mode or "dataset/default")
    logger.info("top_k=%s, max_context_chunks=%s", args.top_k, args.max_context_chunks)

    report = evaluate_dataset(
        cases=cases,
        retrieval=retrieval,
        llm_client=llm_client,
        mode_override=args.mode,
        top_k=args.top_k,
        max_context_chunks=args.max_context_chunks,
        max_context_chars=args.max_context_chars,
        model_override=args.model,
    )

    artifact = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": str(args.dataset),
        "meta": meta,
        "config_path": args.config,
        "params": {
            "mode_override": args.mode,
            "top_k": args.top_k,
            "max_context_chunks": args.max_context_chunks,
            "max_context_chars": args.max_context_chars,
            "llm": args.llm,
            "model": args.model,
            "no_generate": args.no_generate or llm_client is None,
        },
        "summary": report["summary"],
        "results": report["results"],
    }

    output_path = args.output
    if not output_path:
        output_path = settings.path.artifacts / f"eval_{artifact['run_id']}.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("评测完成，结果写入: %s", output_path)
    logger.info("Summary: %s", artifact["summary"])


if __name__ == "__main__":
    main()
