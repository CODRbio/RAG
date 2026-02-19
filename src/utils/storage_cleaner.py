"""
持久化存储清理工具：按生命周期和总大小限制清理过期/超额数据。
操作单一的 data/rag.db，通过 SQLModel 访问。
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from sqlmodel import Session, select

from src.db.engine import get_engine, _make_absolute_sqlite_url, _resolve_db_url
from src.db.models import Canvas, CanvasCitation, CanvasDraftBlock, CanvasOutlineSection, CanvasVersion, ChatSession, Turn, UserProject
from src.log import get_logger

logger = get_logger(__name__)


def _get_rag_db_path() -> Path:
    """Return the absolute path to data/rag.db."""
    url = _make_absolute_sqlite_url(_resolve_db_url())
    # url is like "sqlite:////abs/path/to/rag.db"
    path_str = url[len("sqlite:///"):]
    return Path(path_str)


def _get_total_size_bytes() -> int:
    """Size of data/rag.db in bytes (0 if not found)."""
    p = _get_rag_db_path()
    return p.stat().st_size if p.exists() else 0


def _cleanup_by_age_canvas(cutoff: datetime, batch_size: int) -> int:
    """Delete canvases older than cutoff that are not archived (cascade deletes children)."""
    cutoff_str = cutoff.isoformat()
    with Session(get_engine()) as session:
        stmt = (
            select(Canvas)
            .where(Canvas.updated_at < cutoff_str)
            .where((Canvas.archived == 0) | (Canvas.archived == None))
            .limit(batch_size)
        )
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)  # cascades to outline_sections, draft_blocks, etc.
        session.commit()
    return count


def _cleanup_by_age_sessions(cutoff: datetime, batch_size: int) -> int:
    """Delete sessions older than cutoff (cascade deletes turns)."""
    cutoff_str = cutoff.isoformat()
    with Session(get_engine()) as session:
        stmt = (
            select(ChatSession)
            .where(ChatSession.updated_at < cutoff_str)
            .limit(batch_size)
        )
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)  # cascades to turns
        session.commit()
    return count


def _cleanup_by_age_persistent(cutoff: datetime, batch_size: int) -> int:
    """Delete user_projects older than cutoff."""
    cutoff_str = cutoff.isoformat()
    with Session(get_engine()) as session:
        stmt = (
            select(UserProject)
            .where(UserProject.updated_at < cutoff_str)
            .limit(batch_size)
        )
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    return count


def cleanup_by_age(max_age_days: int, batch_size: int = 100) -> Tuple[int, int, int]:
    """
    按时间清理所有数据库中超过 max_age_days 的记录。
    返回 (canvas_removed, session_removed, project_removed)
    """
    cutoff = datetime.now() - timedelta(days=max_age_days)
    c = _cleanup_by_age_canvas(cutoff, batch_size)
    s = _cleanup_by_age_sessions(cutoff, batch_size)
    p = _cleanup_by_age_persistent(cutoff, batch_size)
    if c or s or p:
        logger.info(
            "[storage] age cleanup: canvas=%d, session=%d, project=%d (cutoff=%s)",
            c, s, p, cutoff.date(),
        )
    return c, s, p


def _get_oldest_canvas() -> list:
    """Return [(id, updated_at)] of oldest non-archived canvases."""
    with Session(get_engine()) as session:
        stmt = (
            select(Canvas)
            .where((Canvas.archived == 0) | (Canvas.archived == None))
            .order_by(Canvas.updated_at.asc())
            .limit(50)
        )
        rows = session.exec(stmt).all()
    return [(r.id, r.updated_at) for r in rows]


def _get_oldest_sessions() -> list:
    """Return [(session_id, updated_at)] of oldest sessions."""
    with Session(get_engine()) as session:
        stmt = (
            select(ChatSession)
            .order_by(ChatSession.updated_at.asc())
            .limit(50)
        )
        rows = session.exec(stmt).all()
    return [(r.session_id, r.updated_at) for r in rows]


def cleanup_by_size(max_size_gb: float, batch_size: int = 100) -> int:
    """
    按总大小清理：若数据库超过 max_size_gb，则删除最旧记录。
    优先删除 canvas（含快照），再删 session。
    返回删除的记录总数。
    """
    max_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    removed_total = 0

    while True:
        current_size = _get_total_size_bytes()
        if current_size <= max_bytes:
            break

        oldest = _get_oldest_canvas()
        if oldest:
            cid = oldest[0][0]
            with Session(get_engine()) as session:
                row = session.get(Canvas, cid)
                if row:
                    session.delete(row)
                    session.commit()
            removed_total += 1
            continue

        oldest_sessions = _get_oldest_sessions()
        if oldest_sessions:
            sid = oldest_sessions[0][0]
            with Session(get_engine()) as session:
                row = session.get(ChatSession, sid)
                if row:
                    session.delete(row)
                    session.commit()
            removed_total += 1
            continue

        break

    if removed_total:
        final_size = _get_total_size_bytes()
        logger.info(
            "[storage] size cleanup: removed=%d, size=%.1fMB (limit=%sGB)",
            removed_total, final_size / 1024 / 1024, max_size_gb,
        )
    return removed_total


def run_cleanup(max_age_days: int = 30, max_size_gb: float = 5.0, batch_size: int = 100) -> dict:
    """
    执行完整清理流程：先按时间清理，再按大小清理。
    返回清理统计。
    """
    c, s, p = cleanup_by_age(max_age_days, batch_size)
    size_removed = cleanup_by_size(max_size_gb, batch_size)
    final_size = _get_total_size_bytes()

    return {
        "age_cleanup": {"canvas": c, "session": s, "project": p},
        "size_cleanup": size_removed,
        "final_size_mb": round(final_size / 1024 / 1024, 2),
    }


def vacuum_databases() -> None:
    """对 rag.db 执行 VACUUM，回收空间。"""
    p = _get_rag_db_path()
    if p.exists():
        try:
            with sqlite3.connect(str(p)) as conn:
                conn.execute("VACUUM")
            logger.debug("[storage] vacuumed rag.db")
        except Exception as e:
            logger.warning("[storage] vacuum failed for rag.db: %s", e)


def get_storage_stats() -> dict:
    """获取存储统计信息。"""
    p = _get_rag_db_path()
    size_mb = 0.0
    if p.exists():
        size_mb = round(p.stat().st_size / 1024 / 1024, 2)
    return {
        "total_size_mb": size_mb,
        "databases": {"rag.db": size_mb},
    }
