#!/usr/bin/env python
"""
步骤29: paper_uid 框架一致性自动升级

对现有文献库（关系数据库）执行全套 paper_uid 回填，使历史数据符合
ref_tools 零号规范（docs/paper_uid_invariants.md）。

升级内容：
  1. paper_metadata 表   → 按 doi/title/authors/year 回填 paper_uid
  2. scholar_library_papers 表 → 按 doi/title/authors/year/url 回填 paper_uid
  3. papers 表           → 从 paper_metadata 继承 paper_uid

向量库（Milvus）说明：
  - 历史 chunk 不需要改写：retrieval/service.py 已在查询时通过 paper_id
    查 paper_metadata 来动态补全 paper_uid（零感知，运行时透明）。
  - 新入库的 chunk 会在 _build_rows() / _chunk_embed_upsert_one_doc()
    中自动写入 paper_uid 字段（dynamic field）。

用法：
    python scripts/29_upgrade_framework_consistency.py             # 标准升级
    python scripts/29_upgrade_framework_consistency.py --dry-run   # 只统计，不写入
    python scripts/29_upgrade_framework_consistency.py --force     # 强制重算全部行
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.log import get_logger

logger = get_logger(__name__)

BACKFILL_SCRIPT = Path(__file__).parent / "28_backfill_paper_uid.py"
TABLES = ["paper_metadata", "scholar_library_papers", "papers"]


def run_backfill(table: str, dry_run: bool, force: bool, batch_size: int) -> bool:
    """Run 28_backfill_paper_uid.py for one table. Returns True on success."""
    cmd = [sys.executable, str(BACKFILL_SCRIPT), "--table", table, "--batch-size", str(batch_size)]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")
    logger.info("▶  %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("✗  %s 回填失败 (exit=%d)", table, result.returncode)
        return False
    logger.info("✓  %s 完成", table)
    return True


def check_milvus_status() -> None:
    """Print Milvus compatibility note — no action needed."""
    logger.info("")
    logger.info("── Milvus 向量库状态 ──────────────────────────────")
    logger.info("  历史 chunk：无需改写。retrieval/service.py 查询时通过 paper_id")
    logger.info("             → paper_metadata 动态补全 paper_uid（已由 GPT 实现）。")
    logger.info("  新入库 chunk：_build_rows() 和 _chunk_embed_upsert_one_doc()")
    logger.info("               已自动写入 paper_uid 动态字段。")
    logger.info("───────────────────────────────────────────────────")
    logger.info("")


def verify_coverage() -> None:
    """Print coverage statistics for paper_uid fields."""
    try:
        from sqlmodel import Session, func, select
        from src.db.engine import get_engine
        from src.db.models import Paper, PaperMetadata, ScholarLibraryPaper

        with Session(get_engine()) as session:
            stats = []
            for model, label in [
                (PaperMetadata, "paper_metadata"),
                (ScholarLibraryPaper, "scholar_library_papers"),
                (Paper, "papers"),
            ]:
                total = session.exec(select(func.count()).select_from(model)).one()
                empty = session.exec(
                    select(func.count()).select_from(model).where(
                        (model.paper_uid == None) | (model.paper_uid == "")
                    )
                ).one()
                filled = total - empty
                pct = f"{filled / total * 100:.1f}%" if total else "N/A"
                stats.append((label, total, filled, empty, pct))

        logger.info("── 覆盖率统计 ──────────────────────────────────────")
        logger.info("  %-30s  总行数    已填充    空值    覆盖率", "表")
        logger.info("  %s", "─" * 62)
        for label, total, filled, empty, pct in stats:
            logger.info("  %-30s  %-8d  %-8d  %-6d  %s", label, total, filled, empty, pct)
        logger.info("───────────────────────────────────────────────────")
    except Exception as e:
        logger.warning("覆盖率统计失败（数据库可能未启动）: %s", e)


def main():
    parser = argparse.ArgumentParser(description="paper_uid 框架一致性自动升级")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计，不写入数据库")
    parser.add_argument("--force", action="store_true",
                        help="强制全量重算（覆盖已有 paper_uid，用于规则更新后）")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="每批提交行数（默认 200）")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过升级后覆盖率统计")
    args = parser.parse_args()

    mode = "[DRY-RUN] " if args.dry_run else ""
    logger.info("%s======= paper_uid 框架一致性升级 =======", mode)
    logger.info("  force=%s  batch_size=%d", args.force, args.batch_size)
    logger.info("")

    t0 = time.time()
    failed = []

    # ── Step 1–3：关系数据库三张表 ──────────────────────────────────────────
    for table in TABLES:
        ok = run_backfill(table, args.dry_run, args.force, args.batch_size)
        if not ok:
            failed.append(table)

    # ── Step 4：Milvus 向量库说明 ───────────────────────────────────────────
    check_milvus_status()

    # ── Step 5：覆盖率验证 ──────────────────────────────────────────────────
    if not args.skip_verify and not args.dry_run:
        verify_coverage()

    elapsed = time.time() - t0
    logger.info("")
    if failed:
        logger.error("升级完成，但以下表失败：%s  (%.1fs)", failed, elapsed)
        sys.exit(1)
    else:
        if args.dry_run:
            logger.info("[DRY-RUN] 扫描完成（%.1fs），未写入任何数据", elapsed)
        else:
            logger.info("升级完成 (%.1fs)  所有表回填成功 ✓", elapsed)


if __name__ == "__main__":
    main()
