"""
Paper 元数据持久化：记录每个集合中入库的文件信息，支持文件级查询和删除。
存储位置：src/data/papers.db
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from src.log import get_logger

logger = get_logger(__name__)

_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "papers.db"


def _db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            collection    TEXT    NOT NULL,
            paper_id      TEXT    NOT NULL,
            filename      TEXT    NOT NULL DEFAULT '',
            file_path     TEXT    NOT NULL DEFAULT '',
            file_size     INTEGER NOT NULL DEFAULT 0,
            chunk_count   INTEGER NOT NULL DEFAULT 0,
            row_count     INTEGER NOT NULL DEFAULT 0,
            enrich_tables_enabled INTEGER NOT NULL DEFAULT 0,
            enrich_figures_enabled INTEGER NOT NULL DEFAULT 0,
            table_count   INTEGER NOT NULL DEFAULT 0,
            figure_count  INTEGER NOT NULL DEFAULT 0,
            table_success INTEGER NOT NULL DEFAULT 0,
            figure_success INTEGER NOT NULL DEFAULT 0,
            status        TEXT    NOT NULL DEFAULT 'done',
            error_message TEXT    DEFAULT '',
            created_at    REAL    NOT NULL,
            UNIQUE(collection, paper_id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_papers_collection
        ON papers (collection)
    """)
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN content_hash TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    for col_sql in [
        "ALTER TABLE papers ADD COLUMN enrich_tables_enabled INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE papers ADD COLUMN enrich_figures_enabled INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE papers ADD COLUMN table_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE papers ADD COLUMN figure_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE papers ADD COLUMN table_success INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE papers ADD COLUMN figure_success INTEGER NOT NULL DEFAULT 0",
    ]:
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass
    conn.commit()


# ── 写入 ──

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
):
    """插入或更新 paper 记录"""
    now = time.time()
    with _db() as conn:
        conn.execute("""
            INSERT INTO papers (collection, paper_id, filename, file_path, file_size,
                                chunk_count, row_count,
                                enrich_tables_enabled, enrich_figures_enabled,
                                table_count, figure_count, table_success, figure_success,
                                status, error_message, created_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collection, paper_id) DO UPDATE SET
                filename      = excluded.filename,
                file_path     = excluded.file_path,
                file_size     = CASE WHEN excluded.file_size > 0 THEN excluded.file_size ELSE file_size END,
                chunk_count   = CASE WHEN excluded.chunk_count > 0 THEN excluded.chunk_count ELSE chunk_count END,
                row_count     = CASE WHEN excluded.row_count > 0 THEN excluded.row_count ELSE row_count END,
                enrich_tables_enabled = excluded.enrich_tables_enabled,
                enrich_figures_enabled = excluded.enrich_figures_enabled,
                table_count   = excluded.table_count,
                figure_count  = excluded.figure_count,
                table_success = excluded.table_success,
                figure_success = excluded.figure_success,
                status        = excluded.status,
                error_message = excluded.error_message,
                created_at    = CASE WHEN status = 'done' THEN created_at ELSE excluded.created_at END,
                content_hash  = excluded.content_hash
        """, (
            collection, paper_id, filename, file_path, file_size,
            chunk_count, row_count,
            int(bool(enrich_tables_enabled)), int(bool(enrich_figures_enabled)),
            int(table_count), int(figure_count), int(table_success), int(figure_success),
            status, error_message, now, content_hash or None
        ))
        conn.commit()


# ── 查询 ──

def list_papers(collection: str) -> List[dict]:
    """列出指定集合中的所有 paper"""
    with _db() as conn:
        rows = conn.execute("""
            SELECT paper_id, filename, file_size, chunk_count, row_count,
                   enrich_tables_enabled, enrich_figures_enabled,
                   table_count, figure_count, table_success, figure_success,
                   status, error_message, created_at, content_hash
            FROM papers
            WHERE collection = ?
            ORDER BY created_at DESC
        """, (collection,)).fetchall()
        return [dict(r) for r in rows]


def get_paper(collection: str, paper_id: str) -> Optional[dict]:
    """获取单个 paper 信息"""
    with _db() as conn:
        row = conn.execute("""
            SELECT paper_id, filename, file_size, chunk_count, row_count,
                   enrich_tables_enabled, enrich_figures_enabled,
                   table_count, figure_count, table_success, figure_success,
                   status, error_message, created_at, content_hash
            FROM papers
            WHERE collection = ? AND paper_id = ?
        """, (collection, paper_id)).fetchone()
        return dict(row) if row else None


# ── 删除 ──

def delete_paper(collection: str, paper_id: str) -> bool:
    """删除 paper 记录"""
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM papers WHERE collection = ? AND paper_id = ?",
            (collection, paper_id),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_collection_papers(collection: str) -> int:
    """删除整个集合的所有 paper 记录"""
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM papers WHERE collection = ?",
            (collection,),
        )
        conn.commit()
        return cur.rowcount
