#!/usr/bin/env python
"""
步骤28: 为现有数据库记录回填 paper_uid

将已存在于数据库中的三张表的历史记录补全 paper_uid 字段：
  - paper_metadata   （PaperMetadata）
  - scholar_library_papers （ScholarLibraryPaper）
  - papers           （Paper）

策略：
  - 对每条 paper_uid 为空的行，从现有 doi/title/authors/year/url 字段调用
    compute_paper_uid() 计算，然后 UPDATE 写回。
  - 对 papers 表，优先从 paper_metadata 表已有的 paper_uid 继承。
  - 幂等：已有 paper_uid 的行不会被覆盖（除非 --force）。

用法:
    python scripts/28_backfill_paper_uid.py                  # 仅补全空值行
    python scripts/28_backfill_paper_uid.py --dry-run        # 仅统计，不写入
    python scripts/28_backfill_paper_uid.py --force          # 全量重算（覆盖已有值）
    python scripts/28_backfill_paper_uid.py --table paper_metadata   # 只处理单表
    python scripts/28_backfill_paper_uid.py --batch-size 500         # 调整批次大小
"""

import argparse
import json
import sys
from typing import List, Optional

sys.path.insert(0, ".")

import sqlalchemy as sa
from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import Paper, PaperMetadata, ScholarLibraryPaper
from src.log import get_logger
from src.retrieval.dedup import compute_paper_uid

logger = get_logger(__name__)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _parse_authors(raw: Optional[str]) -> List[str]:
    """将数据库中 JSON 字符串形式的 authors 反序列化为列表。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# ── 各表回填逻辑 ──────────────────────────────────────────────────────────────

def backfill_paper_metadata(session: Session, dry_run: bool, force: bool, batch_size: int) -> dict:
    """回填 paper_metadata 表的 paper_uid。"""
    stmt = select(PaperMetadata)
    if not force:
        stmt = stmt.where(
            (PaperMetadata.paper_uid == None) | (PaperMetadata.paper_uid == "")
        )
    rows = session.exec(stmt).all()
    total = len(rows)
    updated = 0

    for i, row in enumerate(rows):
        authors = _parse_authors(row.authors)
        uid = compute_paper_uid(
            doi=row.doi or None,
            title=row.title or None,
            authors=authors or None,
            year=row.year,
        )
        if dry_run:
            logger.info("[DRY-RUN] paper_metadata paper_id=%s  →  %s", row.paper_id, uid)
        else:
            row.paper_uid = uid
            session.add(row)
            updated += 1
        if (i + 1) % batch_size == 0:
            if not dry_run:
                session.commit()
            logger.info("  paper_metadata: 已处理 %d / %d ...", i + 1, total)

    if not dry_run and updated % batch_size != 0:
        session.commit()

    return {"table": "paper_metadata", "total": total, "updated": updated}


def backfill_scholar_library_papers(session: Session, dry_run: bool, force: bool, batch_size: int) -> dict:
    """回填 scholar_library_papers 表的 paper_uid。"""
    stmt = select(ScholarLibraryPaper)
    if not force:
        stmt = stmt.where(
            (ScholarLibraryPaper.paper_uid == None) | (ScholarLibraryPaper.paper_uid == "")
        )
    rows = session.exec(stmt).all()
    total = len(rows)
    updated = 0

    for i, row in enumerate(rows):
        authors = _parse_authors(row.authors)
        uid = compute_paper_uid(
            doi=row.doi or None,
            title=row.title or None,
            authors=authors or None,
            year=row.year,
            url=row.url or None,
        )
        if dry_run:
            logger.info("[DRY-RUN] scholar_library_papers id=%s  →  %s", row.id, uid)
        else:
            row.paper_uid = uid
            session.add(row)
            updated += 1
        if (i + 1) % batch_size == 0:
            if not dry_run:
                session.commit()
            logger.info("  scholar_library_papers: 已处理 %d / %d ...", i + 1, total)

    if not dry_run and updated % batch_size != 0:
        session.commit()

    return {"table": "scholar_library_papers", "total": total, "updated": updated}


def backfill_papers(session: Session, dry_run: bool, force: bool, batch_size: int) -> dict:
    """回填 papers 表的 paper_uid。优先继承 paper_metadata 已有 paper_uid。"""
    # 构建 paper_id → paper_uid 索引（仅从 paper_metadata 中已有 uid 的行）
    pm_rows = session.exec(
        select(PaperMetadata).where(
            PaperMetadata.paper_uid != None,
            PaperMetadata.paper_uid != "",
        )
    ).all()
    pm_uid_index = {r.paper_id: r.paper_uid for r in pm_rows}

    stmt = select(Paper)
    if not force:
        stmt = stmt.where(
            (Paper.paper_uid == None) | (Paper.paper_uid == "")
        )
    rows = session.exec(stmt).all()
    total = len(rows)
    updated = 0

    for i, row in enumerate(rows):
        # 优先用 paper_metadata 已有的 paper_uid
        uid = pm_uid_index.get(row.paper_id) or ""
        if not uid:
            # 没有可继承的，只有文件名，无元数据可用，留空（元数据补全后会再 upsert）
            pass

        if dry_run:
            status = "inherited" if uid else "skipped(no_meta)"
            logger.info("[DRY-RUN] papers paper_id=%s  →  %s  [%s]", row.paper_id, uid or "(empty)", status)
        elif uid:
            row.paper_uid = uid
            session.add(row)
            updated += 1
        if (i + 1) % batch_size == 0:
            if not dry_run:
                session.commit()
            logger.info("  papers: 已处理 %d / %d ...", i + 1, total)

    if not dry_run and updated % batch_size != 0:
        session.commit()

    return {"table": "papers", "total": total, "updated": updated}


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="回填数据库中的 paper_uid 字段")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅扫描统计，不写入数据库")
    parser.add_argument("--force", action="store_true",
                        help="强制全量重算，覆盖已有 paper_uid")
    parser.add_argument("--table", choices=["paper_metadata", "scholar_library_papers", "papers", "all"],
                        default="all", help="指定只处理哪张表（默认 all）")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="每批提交的行数（默认 200）")
    args = parser.parse_args()

    mode = "[DRY-RUN] " if args.dry_run else ""
    logger.info("%s开始回填 paper_uid  force=%s  table=%s  batch=%d",
                mode, args.force, args.table, args.batch_size)

    results = []
    with Session(get_engine()) as session:
        if args.table in ("paper_metadata", "all"):
            r = backfill_paper_metadata(session, args.dry_run, args.force, args.batch_size)
            results.append(r)

        if args.table in ("scholar_library_papers", "all"):
            r = backfill_scholar_library_papers(session, args.dry_run, args.force, args.batch_size)
            results.append(r)

        if args.table in ("papers", "all"):
            r = backfill_papers(session, args.dry_run, args.force, args.batch_size)
            results.append(r)

    logger.info("── 回填结果 ──────────────────────────────────")
    for r in results:
        logger.info("  %-30s  total=%-6d  updated=%d", r["table"], r["total"], r["updated"])
    logger.info("──────────────────────────────────────────────")
    if args.dry_run:
        logger.info("[DRY-RUN] 未写入任何数据")


if __name__ == "__main__":
    main()
