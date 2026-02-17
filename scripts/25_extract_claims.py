#!/usr/bin/env python
"""
批量 Claim 提取脚本

对已解析的论文（data/parsed/*/enriched.json）批量提取核心 claims，
结果写回 enriched.json 的 `claims` 字段（不需要重新解析 PDF）。

使用方法:
---------
# 提取所有论文的 claims
python scripts/25_extract_claims.py

# 仅处理指定论文
python scripts/25_extract_claims.py --paper-id "2026_Botté_et_al_Artificial_Light"

# 指定 LLM provider
python scripts/25_extract_claims.py --llm deepseek

# 跳过已有 claims 的论文
python scripts/25_extract_claims.py --skip-existing

# 预览（不写入文件）
python scripts/25_extract_claims.py --dry-run

# 限制论文数
python scripts/25_extract_claims.py --max-papers 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

from config.settings import settings
from src.llm.llm_manager import get_manager
from src.log import get_logger
from src.parser.claim_extractor import ClaimExtractor

logger = get_logger(__name__)

_CONFIG_PATH = Path("config/rag_config.json")


def _find_enriched_files(data_dir: Optional[Path] = None) -> List[Path]:
    """查找所有 enriched.json 文件"""
    base = data_dir or settings.path.data / "parsed"
    if not base.exists():
        logger.warning(f"目录不存在: {base}")
        return []
    return sorted(base.glob("*/enriched.json"))


def _load_enriched(path: Path) -> Optional[Dict[str, Any]]:
    """加载 enriched.json"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载失败 {path}: {e}")
        return None


def _save_enriched(path: Path, data: Dict[str, Any]) -> None:
    """保存 enriched.json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class _SimpleDoc:
    """轻量 EnrichedDoc 代理，用于 ClaimExtractor"""

    def __init__(self, doc_id: str, content_flow: List[Any]):
        self.doc_id = doc_id
        self.content_flow = content_flow


class _SimpleBlock:
    """轻量 ContentBlock 代理"""

    def __init__(self, block_id: str, heading_path: List[str], text: str):
        self.block_id = block_id
        self.heading_path = heading_path
        self.text = text


def _build_doc_proxy(data: Dict[str, Any]) -> _SimpleDoc:
    """从 enriched.json 的 dict 构建轻量 doc 对象"""
    doc_id = data.get("doc_id", "unknown")
    blocks = []
    for block in data.get("content_flow", []):
        blocks.append(_SimpleBlock(
            block_id=block.get("block_id", ""),
            heading_path=block.get("heading_path", []),
            text=block.get("text", ""),
        ))
    return _SimpleDoc(doc_id, blocks)


def main():
    parser = argparse.ArgumentParser(description="批量提取论文核心 Claims")
    parser.add_argument("--paper-id", type=str, default=None, help="仅处理指定 paper_id")
    parser.add_argument("--llm", type=str, default=None, help="LLM provider (如 deepseek)")
    parser.add_argument("--model", type=str, default=None, help="覆盖默认模型")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已有 claims 的论文")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入文件")
    parser.add_argument("--max-papers", type=int, default=None, help="最多处理的论文数")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录（默认 data/parsed）")
    args = parser.parse_args()

    # 初始化 LLM
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(args.llm or None)
    if args.model:
        # 存储 model override
        _original_chat = client.chat

        def _chat_with_model(**kwargs):
            kwargs.setdefault("model", args.model)
            return _original_chat(**kwargs)

        client.chat = _chat_with_model

    extractor = ClaimExtractor()

    # 查找文件
    data_dir = Path(args.data_dir) if args.data_dir else None
    files = _find_enriched_files(data_dir)
    logger.info(f"找到 {len(files)} 个 enriched.json 文件")

    if args.paper_id:
        files = [f for f in files if args.paper_id in f.parent.name]
        logger.info(f"过滤后: {len(files)} 个文件 (paper_id={args.paper_id})")

    if args.max_papers:
        files = files[:args.max_papers]

    # 处理
    total_claims = 0
    processed = 0
    errors = 0

    for fpath in files:
        data = _load_enriched(fpath)
        if data is None:
            errors += 1
            continue

        doc_id = data.get("doc_id", fpath.parent.name)

        # 跳过已有 claims
        if args.skip_existing and data.get("claims"):
            logger.info(f"[{doc_id}] 已有 {len(data['claims'])} claims，跳过")
            continue

        logger.info(f"[{doc_id}] 提取 claims...")

        try:
            doc_proxy = _build_doc_proxy(data)
            claims = extractor.extract(doc_proxy, client)
            claims_dicts = [c.to_dict() for c in claims]

            total_claims += len(claims_dicts)
            processed += 1

            if args.dry_run:
                print(f"\n{'='*60}")
                print(f"Paper: {doc_id}")
                print(f"Claims ({len(claims_dicts)}):")
                for c in claims_dicts:
                    print(f"  [{c['confidence']}] {c['text'][:100]}...")
            else:
                data["claims"] = claims_dicts
                _save_enriched(fpath, data)
                logger.info(f"[{doc_id}] 写入 {len(claims_dicts)} claims")

        except Exception as e:
            logger.error(f"[{doc_id}] 提取失败: {e}")
            errors += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成: 处理 {processed} 篇论文, 提取 {total_claims} 个 claims, {errors} 个错误")
    if args.dry_run:
        print("（预览模式，未写入文件）")


if __name__ == "__main__":
    main()
