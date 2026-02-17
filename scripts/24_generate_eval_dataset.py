#!/usr/bin/env python
"""
LLM 辅助生成评估数据集

读取所有 data/parsed/*/enriched.json，从每篇论文的 Abstract + 前几个 content blocks
提取要点，调用 LLM 生成 3-5 个 QA pairs / 论文，输出标准 eval_mini.json 格式。

使用方法:
---------
# 预览（不写入文件）
python scripts/24_generate_eval_dataset.py --dry-run

# 生成并写入
python scripts/24_generate_eval_dataset.py --output data/eval_generated.json

# 指定 LLM provider / model
python scripts/24_generate_eval_dataset.py --llm deepseek --model deepseek-chat

# 追加到已有数据集
python scripts/24_generate_eval_dataset.py --append-to data/eval_mini.json

# 限制论文数
python scripts/24_generate_eval_dataset.py --max-papers 5

# 每篇论文的 QA 数量
python scripts/24_generate_eval_dataset.py --qa-per-paper 5
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

from config.settings import settings
from src.llm.llm_manager import get_manager
from src.log import get_logger

logger = get_logger(__name__)

# 每篇论文提取的最大 content block 数
MAX_BLOCKS_PER_PAPER = 15
# 每个 block 的最大字符数
MAX_BLOCK_CHARS = 500


def _load_enriched(path: Path) -> Optional[Dict[str, Any]]:
    """加载 enriched.json"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载失败 {path}: {e}")
        return None


def _extract_paper_summary(doc: Dict[str, Any]) -> str:
    """从 enriched.json 提取论文摘要文本（Abstract + 前几个 content blocks）"""
    doc_id = doc.get("doc_id", "unknown")
    blocks = doc.get("content_flow", [])

    text_parts = [f"Paper ID: {doc_id}"]

    count = 0
    for block in blocks:
        if count >= MAX_BLOCKS_PER_PAPER:
            break

        block_type = block.get("type", "")
        content = ""

        if block_type == "heading":
            content = block.get("text", "").strip()
            if content:
                text_parts.append(f"\n## {content}")
                count += 1
        elif block_type == "paragraph":
            content = block.get("text", "").strip()
            if content:
                if len(content) > MAX_BLOCK_CHARS:
                    content = content[:MAX_BLOCK_CHARS] + "..."
                text_parts.append(content)
                count += 1
        elif block_type == "table":
            desc = block.get("llm_description", "")
            if desc:
                if len(desc) > MAX_BLOCK_CHARS:
                    desc = desc[:MAX_BLOCK_CHARS] + "..."
                text_parts.append(f"[Table] {desc}")
                count += 1

    return "\n\n".join(text_parts)


def _build_generation_prompt(paper_summary: str, doc_id: str, qa_count: int = 3) -> str:
    """构建 LLM 生成 QA 对的 prompt"""
    return f"""Based on the following scientific paper content, generate {qa_count} high-quality question-answer pairs for evaluating a retrieval-augmented generation (RAG) system.

Requirements for the QA pairs:
1. Questions should be answerable from the paper content provided
2. Include a mix of:
   - Factual questions (specific findings, data, measurements)
   - Methodology questions (experimental design, techniques used)
   - Data questions (specific numbers, statistics, quantities)
3. Answers should be concise but complete (2-4 sentences)
4. Each question should be standalone (understandable without seeing other questions)

Paper content:
{paper_summary}

Return a JSON array (no other text):
[
  {{
    "query": "The question text",
    "tags": ["factual" or "methodology" or "data_query"],
    "reference_answer": "The concise answer based on the paper"
  }}
]

IMPORTANT: Return ONLY the JSON array, no markdown fences or other text."""


def _parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """解析 LLM 返回的 JSON"""
    # 去掉 markdown code fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "cases" in data:
            return data["cases"]
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        # 尝试修复常见问题
        try:
            # 寻找第一个 [ 和最后一个 ]
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            return []


def generate_for_paper(
    doc: Dict[str, Any],
    llm_client: Any,
    qa_count: int = 3,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """为单篇论文生成 QA 对"""
    doc_id = doc.get("doc_id", "unknown")
    summary = _extract_paper_summary(doc)

    if len(summary) < 100:
        logger.warning(f"论文内容过短，跳过: {doc_id}")
        return []

    prompt = _build_generation_prompt(summary, doc_id, qa_count)

    try:
        resp = llm_client.chat(
            [
                {"role": "system", "content": "You are a scientific QA pair generator. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=model_override,
            max_tokens=2000,
        )
        text = (resp.get("final_text") or "").strip()
        raw_cases = _parse_llm_response(text)
    except Exception as e:
        logger.error(f"LLM 调用失败 ({doc_id}): {e}")
        return []

    # 标准化输出
    cases = []
    for i, raw in enumerate(raw_cases):
        query = (raw.get("query") or "").strip()
        answer = (raw.get("reference_answer") or "").strip()
        tags = raw.get("tags", ["factual"])
        if isinstance(tags, str):
            tags = [tags]

        if not query or not answer:
            continue

        cases.append({
            "query": query,
            "tags": tags,
            "expected_doc_ids": [doc_id],
            "expected_citations": [],
            "reference_answer": answer,
        })

    logger.info(f"  {doc_id}: 生成 {len(cases)}/{qa_count} 条 QA")
    return cases


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM 辅助生成评估 QA 数据集")
    parser.add_argument("--output", type=str, default="data/eval_generated.json", help="输出文件路径")
    parser.add_argument("--append-to", type=str, default=None, help="追加到已有数据集（合并 cases）")
    parser.add_argument("--parsed-dir", type=str, default="data/parsed", help="enriched.json 所在目录")
    parser.add_argument("--qa-per-paper", type=int, default=3, help="每篇论文生成的 QA 数量")
    parser.add_argument("--max-papers", type=int, default=None, help="最多处理的论文数")
    parser.add_argument("--llm", type=str, default=None, help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="模型覆盖")
    parser.add_argument("--config", type=str, default="config/rag_config.json", help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入文件")
    args = parser.parse_args()

    # 扫描所有 enriched.json
    parsed_dir = Path(args.parsed_dir)
    enriched_files = sorted(parsed_dir.glob("*/enriched.json"))

    if not enriched_files:
        logger.error(f"未找到 enriched.json 文件: {parsed_dir}")
        sys.exit(1)

    if args.max_papers:
        enriched_files = enriched_files[: args.max_papers]

    logger.info(f"找到 {len(enriched_files)} 篇论文")

    # 初始化 LLM
    if settings.llm.dry_run:
        logger.warning("LLM_DRY_RUN=true，无法生成 QA 对")
        sys.exit(1)

    manager = get_manager(args.config)
    llm_client = manager.get_client(args.llm)

    # 生成 QA 对
    all_cases: List[Dict[str, Any]] = []
    id_counter = 1

    for enriched_path in enriched_files:
        doc = _load_enriched(enriched_path)
        if not doc:
            continue

        doc_id = doc.get("doc_id", enriched_path.parent.name)
        logger.info(f"处理: {doc_id}")

        cases = generate_for_paper(
            doc=doc,
            llm_client=llm_client,
            qa_count=args.qa_per_paper,
            model_override=args.model,
        )

        for case in cases:
            case["id"] = f"gen_{id_counter:03d}"
            case["mode"] = "local"
            id_counter += 1
            all_cases.append(case)

    logger.info(f"总计生成 {len(all_cases)} 条 QA 对")

    if args.dry_run:
        print(json.dumps(all_cases[:5], ensure_ascii=False, indent=2))
        logger.info("(dry-run 模式，不写入文件)")
        return

    # 构建输出
    output_data = {
        "name": "deepsea_eval_generated",
        "description": f"LLM-generated evaluation dataset ({len(all_cases)} cases from {len(enriched_files)} papers)",
        "version": "1.0.0",
        "cases": all_cases,
    }

    # 追加模式
    if args.append_to:
        append_path = Path(args.append_to)
        if append_path.exists():
            with open(append_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_cases = existing.get("cases", [])
            existing_ids = {c["id"] for c in existing_cases}

            # 去重并追加
            new_cases = [c for c in all_cases if c["id"] not in existing_ids]
            existing_cases.extend(new_cases)
            existing["cases"] = existing_cases
            existing["description"] = (
                existing.get("description", "") +
                f" + {len(new_cases)} generated cases"
            )

            output_path = append_path
            output_data = existing
            logger.info(f"追加 {len(new_cases)} 条到 {append_path}")
        else:
            output_path = append_path
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"写入: {output_path} ({len(output_data['cases'])} 条)")


if __name__ == "__main__":
    main()
