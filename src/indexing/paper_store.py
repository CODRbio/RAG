"""
Paper 元数据持久化：记录每个集合中入库的文件信息，支持文件级查询和删除。
底层存储已迁移至 data/rag.db (papers 表)，通过 SQLModel 访问。
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from sqlmodel import Session, select, and_

from src.db.engine import get_engine
from src.db.models import Paper
from src.log import get_logger

logger = get_logger(__name__)


def _with_persisted_metadata(payload: dict, paper_id: str) -> dict:
    """Merge persisted paper_metadata fields so file list/detail can read back rich metadata."""
    try:
        from src.indexing.paper_metadata_store import paper_meta_store

        meta = paper_meta_store.get(paper_id)
        if not meta:
            return payload
        extra = {}
        extra_raw = meta.get("extra_raw")
        if extra_raw:
            try:
                extra = json.loads(extra_raw)
            except Exception:
                extra = {}
        payload.update(
            {
                "doi": meta.get("doi") or None,
                "title": meta.get("title") or None,
                "authors": meta.get("authors") or None,
                "year": meta.get("year"),
                "venue": extra.get("venue"),
                "url": extra.get("url"),
                "pdf_url": extra.get("pdf_url"),
                "arxiv_id": extra.get("arxiv_id"),
            }
        )
    except Exception as e:
        logger.debug("paper_store metadata merge failed for %s: %s", paper_id, e)
    return payload


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
    user_id: str = "default",
    library_id: Optional[int] = None,
    library_paper_id: Optional[int] = None,
    source: Optional[str] = None,
) -> None:
    """插入或更新 paper 记录。可选 library_id/library_paper_id/source 用于关联文献库。"""
    now = time.time()
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(Paper.collection == collection, Paper.paper_id == paper_id)
        )
        row = session.exec(stmt).first()
        if row is None:
            row = Paper(
                user_id=user_id,
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
                library_id=library_id,
                library_paper_id=library_paper_id,
                source=source or "",
            )
            session.add(row)
        else:
            if hasattr(row, "user_id"):
                row.user_id = user_id
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
            if library_id is not None:
                row.library_id = library_id
            if library_paper_id is not None:
                row.library_paper_id = library_paper_id
            if source is not None:
                row.source = source
            session.add(row)
        session.commit()


# ── 查询 ──────────────────────────────────────────────────────────────────────

def list_papers(collection: str, user_id: Optional[str] = None) -> List[dict]:
    """列出指定集合中的所有 paper；若提供 user_id 则仅返回该用户的记录。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(Paper.collection == collection)
        if user_id is not None:
            stmt = stmt.where(Paper.user_id == user_id)
        stmt = stmt.order_by(Paper.created_at.desc())
        rows = session.exec(stmt).all()
    return [
        _with_persisted_metadata({
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
            "library_id": getattr(r, "library_id", None),
            "library_paper_id": getattr(r, "library_paper_id", None),
            "source": getattr(r, "source", "") or "",
        }, r.paper_id)
        for r in rows
    ]


def get_paper_by_id(paper_id: str) -> Optional[dict]:
    """按 paper_id 查找（跨所有 collection），用于 PDF 端点 fallback。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(Paper.paper_id == paper_id).limit(1)
        row = session.exec(stmt).first()
    if not row:
        return None
    return {
        "paper_id": row.paper_id,
        "file_path": row.file_path or "",
        "collection": row.collection or "",
        "filename": row.filename or "",
    }


def get_paper_by_library_paper_id(library_paper_id: int) -> Optional[dict]:
    """按 library_paper_id 查找已入库的 Paper 记录（用于 PDF 删除时同步清理向量）。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(Paper.library_paper_id == library_paper_id).limit(1)
        row = session.exec(stmt).first()
    if not row:
        return None
    return {
        "paper_id": row.paper_id,
        "collection": row.collection or "",
        "file_path": row.file_path or "",
    }


def list_papers_linked_to_library(
    collection: str,
    library_id: int,
    user_id: Optional[str] = None,
) -> List[dict]:
    """List papers in the collection that are linked to the given scholar library (by library_id).
    Used to sync-remove from vector store when papers are removed from the library."""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(
                Paper.collection == collection,
                Paper.library_id == library_id,
            )
        )
        if user_id is not None:
            stmt = stmt.where(Paper.user_id == user_id)
        stmt = stmt.order_by(Paper.created_at.desc())
        rows = session.exec(stmt).all()
    return [
        {
            "paper_id": r.paper_id,
            "library_id": getattr(r, "library_id", None),
            "library_paper_id": getattr(r, "library_paper_id", None),
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
    return _with_persisted_metadata({
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
        "library_id": getattr(row, "library_id", None),
        "library_paper_id": getattr(row, "library_paper_id", None),
        "source": getattr(row, "source", "") or "",
    }, row.paper_id)


# ── 删除 ──────────────────────────────────────────────────────────────────────

def _delete_paper_files(file_path: str) -> None:
    """删除 paper 对应的磁盘文件（PDF + 解析中间产物），尽力而为，失败只记日志。"""
    if not file_path:
        return
    p = Path(file_path)
    # 删除 PDF
    try:
        if p.exists():
            p.unlink()
            logger.debug("Deleted PDF file: %s", file_path)
    except Exception as e:
        logger.warning("Failed to delete PDF file %s: %s", file_path, e)
    # 删除解析中间产物（同级目录 parsed_data/{stem}.json）
    try:
        parsed = p.parent.parent / "parsed_data" / (p.stem + ".json")
        if parsed.exists():
            parsed.unlink()
            logger.debug("Deleted parsed data: %s", parsed)
    except Exception as e:
        logger.warning("Failed to delete parsed data for %s: %s", file_path, e)


def delete_paper(collection: str, paper_id: str) -> bool:
    """删除 paper 记录及其磁盘文件。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            and_(Paper.collection == collection, Paper.paper_id == paper_id)
        )
        row = session.exec(stmt).first()
        if not row:
            return False
        _delete_paper_files(row.file_path)
        session.delete(row)
        session.commit()
    return True


def delete_collection_papers(collection: str) -> int:
    """删除整个集合的所有 paper 记录及其磁盘文件。"""
    with Session(get_engine()) as session:
        stmt = select(Paper).where(Paper.collection == collection)
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            _delete_paper_files(row.file_path)
            session.delete(row)
        session.commit()
    return count
