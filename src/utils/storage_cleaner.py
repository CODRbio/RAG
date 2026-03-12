"""
持久化存储清理工具：按生命周期和总大小限制清理过期/超额数据。
通过 SQLModel + PostgreSQL 访问。
"""

import sqlalchemy as sa
from datetime import datetime, timedelta
from typing import Tuple

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import Canvas, CanvasCitation, CanvasDraftBlock, CanvasOutlineSection, CanvasVersion, ChatSession, Turn, UserProject, IngestJob, DeepResearchJob
from src.log import get_logger

logger = get_logger(__name__)


def _get_total_size_bytes() -> int:
    """Current size of the database in bytes (via pg_database_size)."""
    try:
        with get_engine().connect() as conn:
            result = conn.execute(sa.text("SELECT pg_database_size(current_database())")).scalar()
            return result or 0
    except Exception:
        return 0


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


def _cleanup_temporary_collections(cutoff: datetime) -> int:
    """Delete temporary job collections (job_xxx) older than cutoff."""
    try:
        from src.indexing.milvus_ops import milvus
        from src.indexing.paper_store import delete_collection_papers
        from src.db.models import DeepResearchJob
    except ImportError:
        return 0

    cutoff_ts = cutoff.timestamp()
    removed = 0
    try:
        collections = milvus.client.list_collections()
        for col in collections:
            col_name = str(col)
            if col_name.startswith("job_"):
                job_id = col_name[4:].replace("_", "-")
                with Session(get_engine()) as session:
                    job = session.get(DeepResearchJob, job_id)
                    should_delete = False
                    if job:
                        if job.updated_at < cutoff_ts:
                            should_delete = True
                    else:
                        # If job doesn't exist, check papers to ensure it's not newly created but job commit delayed
                        from src.db.models import Paper
                        oldest_paper = session.exec(
                            select(Paper).where(Paper.collection == col_name).order_by(Paper.created_at.asc()).limit(1)
                        ).first()
                        if oldest_paper:
                            if oldest_paper.created_at < cutoff_ts:
                                should_delete = True
                        else:
                            should_delete = True
                    
                    if should_delete:
                        milvus.client.drop_collection(col_name)
                        try:
                            delete_collection_papers(col_name)
                        except Exception as pe:
                            logger.warning(f"[storage] paper_store cleanup failed for temp collection {col_name}: {pe}")
                        
                        logger.info(f"[storage] cleaned up temporary collection: {col_name}")
                        removed += 1
    except Exception as e:
        logger.warning(f"[storage] temp collection cleanup failed: {e}")
        
    return removed


_INGEST_TERMINAL = ("done", "error", "cancelled")
_DR_TERMINAL = ("done", "error", "cancelled")


def _cleanup_by_age_ingest_jobs(cutoff: datetime, batch_size: int) -> int:
    """Delete terminal IngestJobs older than cutoff (cascade deletes events and checkpoints)."""
    cutoff_ts = cutoff.timestamp()
    with Session(get_engine()) as session:
        stmt = (
            select(IngestJob)
            .where(IngestJob.status.in_(_INGEST_TERMINAL))
            .where(IngestJob.updated_at < cutoff_ts)
            .limit(batch_size)
        )
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    return count


def _cleanup_by_age_dr_jobs(cutoff: datetime, batch_size: int) -> int:
    """Delete terminal DeepResearchJobs older than cutoff (cascade deletes all sub-tables).
    Also drops the associated Milvus job_* collection if it still exists."""
    cutoff_ts = cutoff.timestamp()
    with Session(get_engine()) as session:
        stmt = (
            select(DeepResearchJob)
            .where(DeepResearchJob.status.in_(_DR_TERMINAL))
            .where(DeepResearchJob.updated_at < cutoff_ts)
            .limit(batch_size)
        )
        rows = session.exec(stmt).all()
        job_ids = [r.job_id for r in rows]
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    # Drop associated Milvus collections after DB commit
    if job_ids:
        try:
            from src.indexing.milvus_ops import milvus
            for job_id in job_ids:
                col_name = "job_" + job_id.replace("-", "_")
                try:
                    if milvus.client.has_collection(col_name):
                        milvus.client.drop_collection(col_name)
                        logger.debug("[storage] dropped Milvus collection %s for aged-out DR job", col_name)
                except Exception as e:
                    logger.warning("[storage] failed to drop Milvus collection %s: %s", col_name, e)
        except Exception as e:
            logger.warning("[storage] Milvus cleanup for aged DR jobs failed: %s", e)
    return count


def cleanup_by_age(max_age_days: int, batch_size: int = 100) -> Tuple[int, int, int, int, int, int]:
    """
    按时间清理所有数据库中超过 max_age_days 的记录。
    返回 (canvas_removed, session_removed, project_removed, temp_collections_removed, ingest_jobs_removed, dr_jobs_removed)
    """
    temp_cutoff = datetime.now() - timedelta(days=1)
    cutoff = datetime.now() - timedelta(days=max_age_days)

    c = _cleanup_by_age_canvas(cutoff, batch_size)
    s = _cleanup_by_age_sessions(cutoff, batch_size)
    p = _cleanup_by_age_persistent(cutoff, batch_size)
    t = _cleanup_temporary_collections(temp_cutoff)
    ij = _cleanup_by_age_ingest_jobs(cutoff, batch_size)
    dr = _cleanup_by_age_dr_jobs(cutoff, batch_size)

    if c or s or p or t or ij or dr:
        logger.info(
            "[storage] age cleanup: canvas=%d, session=%d, project=%d, temp_col=%d, ingest_jobs=%d, dr_jobs=%d (cutoff=%s)",
            c, s, p, t, ij, dr, cutoff.date()
        )
    return c, s, p, t, ij, dr


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
    c, s, p, t, ij, dr = cleanup_by_age(max_age_days, batch_size)
    size_removed = cleanup_by_size(max_size_gb, batch_size)
    final_size = _get_total_size_bytes()

    return {
        "age_cleanup": {"canvas": c, "session": s, "project": p, "temp_col": t, "ingest_jobs": ij, "dr_jobs": dr},
        "size_cleanup": size_removed,
        "final_size_mb": round(final_size / 1024 / 1024, 2),
    }


def vacuum_databases() -> None:
    """对数据库执行 VACUUM ANALYZE，回收空间并更新查询统计。"""
    try:
        with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(sa.text("VACUUM ANALYZE"))
        logger.debug("[storage] VACUUM ANALYZE completed")
    except Exception as e:
        logger.warning("[storage] VACUUM ANALYZE failed: %s", e)


def get_storage_stats() -> dict:
    """获取存储统计信息。"""
    size_bytes = _get_total_size_bytes()
    size_mb = round(size_bytes / 1024 / 1024, 2)
    return {
        "total_size_mb": size_mb,
        "databases": {"rag": size_mb},
    }
