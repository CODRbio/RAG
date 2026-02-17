"""
持久化存储清理工具：按生命周期和总大小限制清理过期/超额数据。
支持 Canvas、Session、Persistent 三个 SQLite 数据库。
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from src.log import get_logger

logger = get_logger(__name__)


def _get_db_paths() -> List[Path]:
    """获取所有需要管理的 SQLite 数据库路径"""
    data_dir = Path(__file__).resolve().parents[2] / "data"
    return [
        data_dir / "canvas.db",
        data_dir / "sessions.db",
        data_dir / "persistent.db",
    ]


def _ensure_canvas_archive_column(conn: sqlite3.Connection) -> None:
    """Ensure canvases has archived column (migration for cleanup)."""
    cur = conn.execute("PRAGMA table_info(canvases)")
    columns = {row[1] for row in cur.fetchall()}
    if "archived" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")


def _get_total_size_bytes(paths: List[Path]) -> int:
    """计算所有数据库文件的总大小（字节）"""
    total = 0
    for p in paths:
        if p.exists():
            total += p.stat().st_size
    return total


def _cleanup_by_age_canvas(db_path: Path, cutoff: datetime, batch_size: int) -> int:
    """按时间清理 canvas.db 中过期的 canvas（及关联表），跳过已存档。"""
    if not db_path.exists():
        return 0
    removed = 0
    cutoff_str = cutoff.isoformat()
    with sqlite3.connect(db_path) as conn:
        _ensure_canvas_archive_column(conn)
        # 查找过期且未存档的 canvas
        rows = conn.execute(
            "SELECT id FROM canvases WHERE updated_at < ? AND (archived = 0 OR archived IS NULL) LIMIT ?",
            (cutoff_str, batch_size),
        ).fetchall()
        canvas_ids = [r[0] for r in rows]
        for cid in canvas_ids:
            conn.execute("DELETE FROM canvas_versions WHERE canvas_id = ?", (cid,))
            conn.execute("DELETE FROM canvas_citations WHERE canvas_id = ?", (cid,))
            conn.execute("DELETE FROM draft_blocks WHERE canvas_id = ?", (cid,))
            conn.execute("DELETE FROM outline_sections WHERE canvas_id = ?", (cid,))
            conn.execute("DELETE FROM canvases WHERE id = ?", (cid,))
            removed += 1
        conn.commit()
    return removed


def _cleanup_by_age_sessions(db_path: Path, cutoff: datetime, batch_size: int) -> int:
    """按时间清理 sessions.db 中过期的 session（及关联 turns）"""
    if not db_path.exists():
        return 0
    removed = 0
    cutoff_str = cutoff.isoformat()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE updated_at < ? LIMIT ?",
            (cutoff_str, batch_size),
        ).fetchall()
        session_ids = [r[0] for r in rows]
        for sid in session_ids:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            removed += 1
        conn.commit()
    return removed


def _cleanup_by_age_persistent(db_path: Path, cutoff: datetime, batch_size: int) -> int:
    """按时间清理 persistent.db 中过期的用户项目记录"""
    if not db_path.exists():
        return 0
    removed = 0
    cutoff_str = cutoff.isoformat()
    with sqlite3.connect(db_path) as conn:
        # user_projects 有 updated_at
        cur = conn.execute(
            "DELETE FROM user_projects WHERE updated_at < ? LIMIT ?",
            (cutoff_str, batch_size),
        )
        removed += cur.rowcount
        conn.commit()
    return removed


def cleanup_by_age(max_age_days: int, batch_size: int = 100) -> Tuple[int, int, int]:
    """
    按时间清理所有数据库中超过 max_age_days 的记录。
    返回 (canvas_removed, session_removed, project_removed)
    """
    cutoff = datetime.now() - timedelta(days=max_age_days)
    data_dir = Path(__file__).resolve().parents[2] / "data"
    c = _cleanup_by_age_canvas(data_dir / "canvas.db", cutoff, batch_size)
    s = _cleanup_by_age_sessions(data_dir / "sessions.db", cutoff, batch_size)
    p = _cleanup_by_age_persistent(data_dir / "persistent.db", cutoff, batch_size)
    if c or s or p:
        logger.info(f"[storage] age cleanup: canvas={c}, session={s}, project={p} (cutoff={cutoff.date()})")
    return c, s, p


def _get_oldest_canvas(db_path: Path) -> List[Tuple[str, str]]:
    """获取最旧的未存档 canvas (id, updated_at)，按 updated_at 升序"""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        _ensure_canvas_archive_column(conn)
        rows = conn.execute(
            "SELECT id, updated_at FROM canvases WHERE (archived = 0 OR archived IS NULL) ORDER BY updated_at ASC LIMIT 50"
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _get_oldest_sessions(db_path: Path) -> List[Tuple[str, str]]:
    """获取最旧的 session (session_id, updated_at)，按 updated_at 升序"""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, updated_at FROM sessions ORDER BY updated_at ASC LIMIT 50"
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def cleanup_by_size(max_size_gb: float, batch_size: int = 100) -> int:
    """
    按总大小清理：若数据库总大小超过 max_size_gb，则删除最旧的记录，
    优先删除 canvas（含快照），再删 session。
    返回删除的记录总数。
    """
    max_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    db_paths = _get_db_paths()
    removed_total = 0
    
    while True:
        current_size = _get_total_size_bytes(db_paths)
        if current_size <= max_bytes:
            break
        
        # 优先删 canvas
        data_dir = Path(__file__).resolve().parents[2] / "data"
        canvas_db = data_dir / "canvas.db"
        oldest = _get_oldest_canvas(canvas_db)
        if oldest:
            cid = oldest[0][0]
            with sqlite3.connect(canvas_db) as conn:
                conn.execute("DELETE FROM canvas_versions WHERE canvas_id = ?", (cid,))
                conn.execute("DELETE FROM canvas_citations WHERE canvas_id = ?", (cid,))
                conn.execute("DELETE FROM draft_blocks WHERE canvas_id = ?", (cid,))
                conn.execute("DELETE FROM outline_sections WHERE canvas_id = ?", (cid,))
                conn.execute("DELETE FROM canvases WHERE id = ?", (cid,))
                conn.commit()
            removed_total += 1
            continue
        
        # canvas 空了，删 session
        sessions_db = data_dir / "sessions.db"
        oldest_sessions = _get_oldest_sessions(sessions_db)
        if oldest_sessions:
            sid = oldest_sessions[0][0]
            with sqlite3.connect(sessions_db) as conn:
                conn.execute("DELETE FROM turns WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
                conn.commit()
            removed_total += 1
            continue
        
        # 都空了，退出
        break
    
    if removed_total:
        final_size = _get_total_size_bytes(db_paths)
        logger.info(f"[storage] size cleanup: removed={removed_total}, size={final_size / 1024 / 1024:.1f}MB (limit={max_size_gb}GB)")
    return removed_total


def run_cleanup(max_age_days: int = 30, max_size_gb: float = 5.0, batch_size: int = 100) -> dict:
    """
    执行完整清理流程：先按时间清理，再按大小清理。
    返回清理统计。
    """
    c, s, p = cleanup_by_age(max_age_days, batch_size)
    size_removed = cleanup_by_size(max_size_gb, batch_size)
    
    db_paths = _get_db_paths()
    final_size = _get_total_size_bytes(db_paths)
    
    return {
        "age_cleanup": {"canvas": c, "session": s, "project": p},
        "size_cleanup": size_removed,
        "final_size_mb": round(final_size / 1024 / 1024, 2),
    }


def vacuum_databases() -> None:
    """对所有数据库执行 VACUUM，回收空间"""
    for db_path in _get_db_paths():
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.execute("VACUUM")
                logger.debug(f"[storage] vacuumed {db_path.name}")
            except Exception as e:
                logger.warning(f"[storage] vacuum failed for {db_path.name}: {e}")


def get_storage_stats() -> dict:
    """获取存储统计信息"""
    db_paths = _get_db_paths()
    stats = {
        "total_size_mb": 0.0,
        "databases": {},
    }
    for p in db_paths:
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            stats["databases"][p.name] = round(size_mb, 2)
            stats["total_size_mb"] += size_mb
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    return stats
