"""
Ingest 任务状态持久化（支持前端断连后重连查看进度）。
存储位置：src/data/ingest_jobs.db
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "ingest_jobs.db"


def _db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            job_id          TEXT PRIMARY KEY,
            collection      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            total_files     INTEGER NOT NULL DEFAULT 0,
            processed_files INTEGER NOT NULL DEFAULT 0,
            failed_files    INTEGER NOT NULL DEFAULT 0,
            total_chunks    INTEGER NOT NULL DEFAULT 0,
            total_upserted  INTEGER NOT NULL DEFAULT 0,
            current_file    TEXT NOT NULL DEFAULT '',
            current_stage   TEXT NOT NULL DEFAULT '',
            message         TEXT NOT NULL DEFAULT '',
            error_message   TEXT NOT NULL DEFAULT '',
            payload_json    TEXT NOT NULL DEFAULT '{}',
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL,
            finished_at     REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_job_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id     TEXT NOT NULL,
            event      TEXT NOT NULL,
            data_json  TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_job_events_job_id_id
        ON ingest_job_events(job_id, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ingest_jobs_created_at
        ON ingest_jobs(created_at DESC)
        """
    )
    conn.commit()


def create_job(collection: str, payload: Dict[str, Any], total_files: int) -> Dict[str, Any]:
    now = time.time()
    job_id = uuid.uuid4().hex
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO ingest_jobs (
                job_id, collection, status, total_files, processed_files, failed_files,
                total_chunks, total_upserted, payload_json, created_at, updated_at
            ) VALUES (?, ?, 'pending', ?, 0, 0, 0, 0, ?, ?, ?)
            """,
            (job_id, collection, int(total_files), json.dumps(payload, ensure_ascii=False), now, now),
        )
        conn.commit()
    return get_job(job_id) or {}


def update_job(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_job(job_id)
    fields["updated_at"] = time.time()
    keys = list(fields.keys())
    set_expr = ", ".join([f"{k} = ?" for k in keys])
    vals = [fields[k] for k in keys]
    vals.append(job_id)
    with _db() as conn:
        conn.execute(f"UPDATE ingest_jobs SET {set_expr} WHERE job_id = ?", vals)
        conn.commit()
    return get_job(job_id)


def append_event(job_id: str, event: str, data: Dict[str, Any]) -> int:
    now = time.time()
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO ingest_job_events (job_id, event, data_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, event, json.dumps(data, ensure_ascii=False), now),
        )
        conn.execute("UPDATE ingest_jobs SET updated_at = ? WHERE job_id = ?", (now, job_id))
        conn.commit()
        return int(cur.lastrowid or 0)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["payload"] = json.loads(out.get("payload_json") or "{}")
        except Exception:
            out["payload"] = {}
        out.pop("payload_json", None)
        return out


def list_jobs(limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with _db() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM ingest_jobs
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM ingest_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except Exception:
                item["payload"] = {}
            item.pop("payload_json", None)
            result.append(item)
        return result


def list_events(job_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    after_id = max(0, int(after_id))
    limit = max(1, min(int(limit), 2000))
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, event, data_json, created_at
            FROM ingest_job_events
            WHERE job_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (job_id, after_id, limit),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        data: Dict[str, Any]
        try:
            data = json.loads(r["data_json"] or "{}")
        except Exception:
            data = {}
        data["event_id"] = int(r["id"])
        out.append(
            {
                "event_id": int(r["id"]),
                "event": str(r["event"]),
                "created_at": float(r["created_at"]),
                "data": data,
            }
        )
    return out
