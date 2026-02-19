"""
Paper 元数据持久化：记录每个集合中入库的文件信息，支持文件级查询和删除。
底层存储已迁移至 data/rag.db (papers 表)，通过 SQLModel 访问。
"""

import time
from typing import List, Optional

from sqlmodel import Session, select, and_

from src.db.engine import get_engine
from src.db.models import Paper
from src.log import get_logger

logger = get_logger(__name__)


# ── 写入 ──────────────────────────────────────────────────────────────────────

def upsert_paper(
    collection: str,
    paper_id: str,
    filename: str = "",
    file_path: str = "",
    file_size: int = 0,
    chunk_count: int = 0,
    row_count: int = 0,
    enrich_tables_enabled: bool = False,
    enrich_figures_enabled: bool = False,
    table_count: int = 0,
    figure_count: int = 0,
    table_success: int = 0,
    figure_success: int = 0,
    status: str = "done",
    error_message: str = "",
    content_hash: str = "",
) -> None:
    """插入或更新 paper 记录。"""
    now = time.time()
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(Paper.collection == collection, Paper.paper_id == paper_id)
        )
        row = session.exec(stmt).first()
        if row is None:
            row = Paper(
                collection=collection,
                paper_id=paper_id,
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                chunk_count=chunk_count,
                row_count=row_count,
                enrich_tables_enabled=int(bool(enrich_tables_enabled)),
                enrich_figures_enabled=int(bool(enrich_figures_enabled)),
                table_count=int(table_count),
                figure_count=int(figure_count),
                table_success=int(table_success),
                figure_success=int(figure_success),
                status=status,
                error_message=error_message,
                content_hash=content_hash or None,
                created_at=now,
            )
            session.add(row)
        else:
            row.filename = filename
            row.file_path = file_path
            if file_size > 0:
                row.file_size = file_size
            if chunk_count > 0:
                row.chunk_count = chunk_count
            if row_count > 0:
                row.row_count = row_count
            row.enrich_tables_enabled = int(bool(enrich_tables_enabled))
            row.enrich_figures_enabled = int(bool(enrich_figures_enabled))
            row.table_count = int(table_count)
            row.figure_count = int(figure_count)
            row.table_success = int(table_success)
            row.figure_success = int(figure_success)
            row.status = status
            row.error_message = error_message
            row.content_hash = content_hash or None
            session.add(row)
        session.commit()


# ── 查询 ──────────────────────────────────────────────────────────────────────

def list_papers(collection: str) -> List[dict]:
    """列出指定集合中的所有 paper。"""
    with Session(get_engine()) as session:
        stmt = (
            select(Paper)
            .where(Paper.collection == collection)
            .order_by(Paper.created_at.desc())
        )
        rows = session.exec(stmt).all()
    return [
        {
            "paper_id": r.paper_id,
            "filename": r.filename,
            "file_size": r.file_size,
            "chunk_count": r.chunk_count,
            "row_count": r.row_count,
            "enrich_tables_enabled": r.enrich_tables_enabled,
            "enrich_figures_enabled": r.enrich_figures_enabled,
            "table_count": r.table_count,
            "figure_count": r.figure_count,
            "table_success": r.table_success,
            "figure_success": r.figure_success,
            "status": r.status,
            "error_message": r.error_message,
            "created_at": r.created_at,
            "content_hash": r.content_hash,
        }
        for r in rows
    ]


def get_paper(collection: str, paper_id: str) -> Optional[dict]:
    """获取单个 paper 信息。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(Paper.collection == collection, Paper.paper_id == paper_id)
        )
        row = session.exec(stmt).first()
    if not row:
        return None
    return {
        "paper_id": row.paper_id,
        "filename": row.filename,
        "file_size": row.file_size,
        "chunk_count": row.chunk_count,
        "row_count": row.row_count,
        "enrich_tables_enabled": row.enrich_tables_enabled,
        "enrich_figures_enabled": row.enrich_figures_enabled,
        "table_count": row.table_count,
        "figure_count": row.figure_count,
        "table_success": row.table_success,
        "figure_success": row.figure_success,
        "status": row.status,
        "error_message": row.error_message,
        "created_at": row.created_at,
        "content_hash": row.content_hash,
    }


# ── 删除 ──────────────────────────────────────────────────────────────────────

def delete_paper(collection: str, paper_id: str) -> bool:
    """删除 paper 记录。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(Paper.collection == collection, Paper.paper_id == paper_id)
        )
        row = session.exec(stmt).first()
        if not row:
            return False
        session.delete(row)
        session.commit()
    return True


def delete_collection_papers(collection: str) -> int:
    """删除整个集合的所有 paper 记录。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(Paper.collection == collection)
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    return count
