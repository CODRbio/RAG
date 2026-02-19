"""
Deep Research 任务状态持久化（支持前端断连后恢复查看）。
存储位置：src/data/deep_research_jobs.db
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "deep_research_jobs.db"


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
        CREATE TABLE IF NOT EXISTS deep_research_jobs (
            job_id              TEXT PRIMARY KEY,
            topic               TEXT NOT NULL,
            session_id          TEXT NOT NULL DEFAULT '',
            canvas_id           TEXT NOT NULL DEFAULT '',
            status              TEXT NOT NULL DEFAULT 'pending',
            current_stage       TEXT NOT NULL DEFAULT '',
            message             TEXT NOT NULL DEFAULT '',
            error_message       TEXT NOT NULL DEFAULT '',
            request_json        TEXT NOT NULL DEFAULT '{}',
            result_markdown     TEXT NOT NULL DEFAULT '',
            result_citations    TEXT NOT NULL DEFAULT '[]',
            result_dashboard    TEXT NOT NULL DEFAULT '{}',
            total_time_ms       REAL NOT NULL DEFAULT 0,
            created_at          REAL NOT NULL,
            updated_at          REAL NOT NULL,
            finished_at         REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_research_job_events (
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
        CREATE INDEX IF NOT EXISTS idx_dr_job_events_job_id_id
        ON deep_research_job_events(job_id, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dr_jobs_created_at
        ON deep_research_jobs(created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_research_section_reviews (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id     TEXT NOT NULL,
            section_id TEXT NOT NULL,
            action     TEXT NOT NULL DEFAULT 'approve',
            feedback   TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            UNIQUE(job_id, section_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_research_resume_queue (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL,
            owner_instance  TEXT NOT NULL DEFAULT '',
            source          TEXT NOT NULL DEFAULT 'review',
            status          TEXT NOT NULL DEFAULT 'pending',
            message         TEXT NOT NULL DEFAULT '',
            created_at      REAL NOT NULL,
            updated_at      REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dr_resume_queue_status_created
        ON deep_research_resume_queue(status, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dr_resume_queue_owner_status
        ON deep_research_resume_queue(owner_instance, status, created_at)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_dr_resume_queue_job_status_unique
        ON deep_research_resume_queue(job_id, status)
        """
    )
    # ── Gap Supplements (section-scoped user supplements for information gaps) ──
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_research_gap_supplements (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL,
            section_id      TEXT NOT NULL,
            gap_text        TEXT NOT NULL DEFAULT '',
            supplement_type TEXT NOT NULL DEFAULT 'material',
            content_json    TEXT NOT NULL DEFAULT '{}',
            status          TEXT NOT NULL DEFAULT 'pending',
            created_at      REAL NOT NULL,
            consumed_at     REAL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dr_gap_supplements_job_section
        ON deep_research_gap_supplements(job_id, section_id)
        """
    )
    # ── Research Insights Ledger (accumulated gaps/conflicts/limitations for future directions) ──
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_research_insights (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL,
            section_id      TEXT NOT NULL DEFAULT '',
            insight_type    TEXT NOT NULL DEFAULT 'gap',
            text            TEXT NOT NULL DEFAULT '',
            source_context  TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dr_insights_job_id
        ON deep_research_insights(job_id)
        """
    )
    conn.commit()


def create_job(
    *,
    topic: str,
    session_id: str = "",
    canvas_id: str = "",
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    job_id = uuid.uuid4().hex
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO deep_research_jobs (
                job_id, topic, session_id, canvas_id, status, request_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                job_id,
                topic,
                session_id,
                canvas_id,
                json.dumps(request_payload or {}, ensure_ascii=False),
                now,
                now,
            ),
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
        conn.execute(f"UPDATE deep_research_jobs SET {set_expr} WHERE job_id = ?", vals)
        conn.commit()
    return get_job(job_id)


def append_event(job_id: str, event: str, data: Dict[str, Any]) -> int:
    now = time.time()
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO deep_research_job_events (job_id, event, data_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, event, json.dumps(data, ensure_ascii=False, default=str), now),
        )
        conn.execute("UPDATE deep_research_jobs SET updated_at = ? WHERE job_id = ?", (now, job_id))
        conn.commit()
        return int(cur.lastrowid or 0)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM deep_research_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
    out = dict(row)
    try:
        out["request"] = json.loads(out.get("request_json") or "{}")
    except Exception:
        out["request"] = {}
    try:
        out["result_citations"] = json.loads(out.get("result_citations") or "[]")
    except Exception:
        out["result_citations"] = []
    try:
        out["result_dashboard"] = json.loads(out.get("result_dashboard") or "{}")
    except Exception:
        out["result_dashboard"] = {}
    out.pop("request_json", None)
    return out


def list_jobs(limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with _db() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM deep_research_jobs
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM deep_research_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["request"] = json.loads(item.get("request_json") or "{}")
        except Exception:
            item["request"] = {}
        try:
            item["result_citations"] = json.loads(item.get("result_citations") or "[]")
        except Exception:
            item["result_citations"] = []
        try:
            item["result_dashboard"] = json.loads(item.get("result_dashboard") or "{}")
        except Exception:
            item["result_dashboard"] = {}
        item.pop("request_json", None)
        result.append(item)
    return result


def list_events(job_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    after_id = max(0, int(after_id))
    limit = max(1, min(int(limit), 2000))
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, event, data_json, created_at
            FROM deep_research_job_events
            WHERE job_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (job_id, after_id, limit),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row["data_json"] or "{}")
        except Exception:
            data = {}
        data["event_id"] = int(row["id"])
        out.append(
            {
                "event_id": int(row["id"]),
                "event": str(row["event"]),
                "created_at": float(row["created_at"]),
                "data": data,
            }
        )
    return out


def submit_review(job_id: str, section_id: str, action: str = "approve", feedback: str = "") -> Dict[str, Any]:
    now = time.time()
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO deep_research_section_reviews (job_id, section_id, action, feedback, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id, section_id) DO UPDATE SET
                action = excluded.action,
                feedback = excluded.feedback,
                created_at = excluded.created_at
            """,
            (job_id, section_id, action, feedback, now),
        )
        conn.commit()
    return {"job_id": job_id, "section_id": section_id, "action": action}


def get_pending_review(job_id: str, section_id: str) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            """
            SELECT job_id, section_id, action, feedback, created_at
            FROM deep_research_section_reviews
            WHERE job_id = ? AND section_id = ?
            """,
            (job_id, section_id),
        ).fetchone()
    return dict(row) if row else None


def list_reviews(job_id: str) -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT job_id, section_id, action, feedback, created_at
            FROM deep_research_section_reviews
            WHERE job_id = ?
            ORDER BY created_at ASC
            """,
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def enqueue_resume_request(
    job_id: str,
    owner_instance: str,
    source: str = "review",
    message: str = "",
) -> Dict[str, Any]:
    """
    Enqueue one resume request for a job.
    Idempotent when an existing pending/running row already exists.
    """
    now = time.time()
    with _db() as conn:
        existing = conn.execute(
            """
            SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
            FROM deep_research_resume_queue
            WHERE job_id = ? AND status IN ('pending', 'running')
            ORDER BY id DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if existing:
            row = dict(existing)
            if row.get("owner_instance") != owner_instance:
                conn.execute(
                    """
                    UPDATE deep_research_resume_queue
                    SET owner_instance = ?, source = ?, message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (owner_instance, source, message, now, row["id"]),
                )
                conn.commit()
                row["owner_instance"] = owner_instance
                row["source"] = source
                row["message"] = message
                row["updated_at"] = now
            return row

        cur = conn.execute(
            """
            INSERT INTO deep_research_resume_queue
                (job_id, owner_instance, source, status, message, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?, ?)
            """,
            (job_id, owner_instance, source, message, now, now),
        )
        conn.commit()
        rid = int(cur.lastrowid or 0)
        row = conn.execute(
            """
            SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
            FROM deep_research_resume_queue
            WHERE id = ?
            """,
            (rid,),
        ).fetchone()
    return dict(row) if row else {}


def claim_resume_requests(owner_instance: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Atomically claim pending resume requests for this instance."""
    limit = max(1, min(int(limit), 100))
    now = time.time()
    claimed: List[Dict[str, Any]] = []
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
            FROM deep_research_resume_queue
            WHERE status = 'pending' AND owner_instance = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (owner_instance, limit),
        ).fetchall()
        for row in rows:
            rid = int(row["id"])
            cur = conn.execute(
                """
                UPDATE deep_research_resume_queue
                SET status = 'running', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, rid),
            )
            if cur.rowcount <= 0:
                continue
            updated = conn.execute(
                """
                SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
                FROM deep_research_resume_queue
                WHERE id = ?
                """,
                (rid,),
            ).fetchone()
            if updated:
                claimed.append(dict(updated))
        conn.commit()
    return claimed


def complete_resume_request(resume_id: int, status: str, message: str = "") -> None:
    """Mark a running resume request as done/error/cancelled."""
    if status not in {"done", "error", "cancelled"}:
        status = "error"
    now = time.time()
    with _db() as conn:
        conn.execute(
            """
            UPDATE deep_research_resume_queue
            SET status = ?, message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, message, now, int(resume_id)),
        )
        conn.commit()


def list_resume_requests(
    limit: int = 50,
    status: Optional[str] = None,
    owner_instance: Optional[str] = None,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List resume queue rows with optional filters."""
    limit = max(1, min(int(limit), 500))
    sql = """
        SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
        FROM deep_research_resume_queue
        WHERE 1 = 1
    """
    params: List[Any] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if owner_instance:
        sql += " AND owner_instance = ?"
        params.append(owner_instance)
    if job_id:
        sql += " AND job_id = ?"
        params.append(job_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def cleanup_resume_requests(
    *,
    statuses: Optional[List[str]] = None,
    before_ts: Optional[float] = None,
    owner_instance: Optional[str] = None,
    job_id: Optional[str] = None,
) -> int:
    """
    Delete resume queue rows by filters, defaulting to terminal statuses only.
    Returns deleted row count.
    """
    effective_statuses = statuses or ["done", "error", "cancelled"]
    allowed = {"pending", "running", "done", "error", "cancelled"}
    effective_statuses = [s for s in effective_statuses if s in allowed]
    if not effective_statuses:
        return 0

    placeholders = ", ".join(["?"] * len(effective_statuses))
    sql = f"DELETE FROM deep_research_resume_queue WHERE status IN ({placeholders})"
    params: List[Any] = list(effective_statuses)
    if before_ts is not None:
        sql += " AND updated_at <= ?"
        params.append(float(before_ts))
    if owner_instance:
        sql += " AND owner_instance = ?"
        params.append(owner_instance)
    if job_id:
        sql += " AND job_id = ?"
        params.append(job_id)

    with _db() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return int(cur.rowcount or 0)


def retry_resume_request(
    *,
    resume_id: int,
    owner_instance: str,
    message: str = "manual retry",
) -> Optional[Dict[str, Any]]:
    """
    Retry a terminal resume request by setting it back to pending.
    Raises ValueError when retry is unsafe/conflicting.
    """
    now = time.time()
    with _db() as conn:
        row = conn.execute(
            """
            SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
            FROM deep_research_resume_queue
            WHERE id = ?
            """,
            (int(resume_id),),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        status = str(item.get("status") or "")
        if status in {"pending", "running"}:
            raise ValueError("resume request is already active")

        conflict = conn.execute(
            """
            SELECT id
            FROM deep_research_resume_queue
            WHERE job_id = ? AND status IN ('pending', 'running') AND id != ?
            LIMIT 1
            """,
            (item["job_id"], int(resume_id)),
        ).fetchone()
        if conflict:
            raise ValueError("job already has active resume request")

        conn.execute(
            """
            UPDATE deep_research_resume_queue
            SET owner_instance = ?,
                source = 'manual_retry',
                status = 'pending',
                message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (owner_instance, message, now, int(resume_id)),
        )
        conn.commit()
        updated = conn.execute(
            """
            SELECT id, job_id, owner_instance, source, status, message, created_at, updated_at
            FROM deep_research_resume_queue
            WHERE id = ?
            """,
            (int(resume_id),),
        ).fetchone()
    return dict(updated) if updated else None


# ────────────────────────────────────────────────
# Gap Supplements CRUD
# ────────────────────────────────────────────────

def submit_gap_supplement(
    job_id: str,
    section_id: str,
    gap_text: str,
    supplement_type: str = "material",
    content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Submit a section-scoped supplement for a specific information gap."""
    now = time.time()
    content_json = json.dumps(content or {}, ensure_ascii=False)
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO deep_research_gap_supplements
                (job_id, section_id, gap_text, supplement_type, content_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (job_id, section_id, gap_text, supplement_type, content_json, now),
        )
        conn.commit()
        return {
            "id": int(cur.lastrowid or 0),
            "job_id": job_id,
            "section_id": section_id,
            "gap_text": gap_text,
            "supplement_type": supplement_type,
            "status": "pending",
        }


def list_gap_supplements(
    job_id: str,
    section_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List gap supplements, optionally filtered by section_id and/or status."""
    with _db() as conn:
        sql = "SELECT * FROM deep_research_gap_supplements WHERE job_id = ?"
        params: List[Any] = [job_id]
        if section_id:
            sql += " AND section_id = ?"
            params.append(section_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC"
        rows = conn.execute(sql, params).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["content"] = json.loads(item.get("content_json") or "{}")
        except Exception:
            item["content"] = {}
        item.pop("content_json", None)
        out.append(item)
    return out


def mark_gap_supplement_consumed(supplement_id: int) -> None:
    """Mark a gap supplement as consumed (used in research/write)."""
    now = time.time()
    with _db() as conn:
        conn.execute(
            "UPDATE deep_research_gap_supplements SET status = 'consumed', consumed_at = ? WHERE id = ?",
            (now, supplement_id),
        )
        conn.commit()


# ────────────────────────────────────────────────
# Research Insights Ledger CRUD
# ────────────────────────────────────────────────

def append_insight(
    job_id: str,
    insight_type: str,
    text: str,
    section_id: str = "",
    source_context: str = "",
) -> int:
    """Append a research insight (gap, conflict, limitation, future_direction)."""
    now = time.time()
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO deep_research_insights
                (job_id, section_id, insight_type, text, source_context, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'open', ?)
            """,
            (job_id, section_id, insight_type, text, source_context, now),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_insights(
    job_id: str,
    insight_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List research insights, optionally filtered by type and/or status."""
    with _db() as conn:
        sql = "SELECT * FROM deep_research_insights WHERE job_id = ?"
        params: List[Any] = [job_id]
        if insight_type:
            sql += " AND insight_type = ?"
            params.append(insight_type)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC"
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_insight_status(insight_id: int, status: str) -> None:
    """Update the status of a research insight (open, addressed, deferred)."""
    with _db() as conn:
        conn.execute(
            "UPDATE deep_research_insights SET status = ? WHERE id = ?",
            (status, insight_id),
        )
        conn.commit()


def bulk_mark_insights_addressed(job_id: str, insight_type: Optional[str] = None) -> int:
    """Mark all open insights for a job as addressed. Returns count of updated rows."""
    with _db() as conn:
        sql = "UPDATE deep_research_insights SET status = 'addressed' WHERE job_id = ? AND status = 'open'"
        params: List[Any] = [job_id]
        if insight_type:
            sql += " AND insight_type = ?"
            params.append(insight_type)
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


def get_latest_job_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    """按 session_id 获取最近一条 Deep Research 任务（用于刷新后恢复 dashboard）。"""
    if not session_id:
        return None
    with _db() as conn:
        row = conn.execute(
            """
            SELECT * FROM deep_research_jobs
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    try:
        out["request"] = json.loads(out.get("request_json") or "{}")
    except Exception:
        out["request"] = {}
    try:
        out["result_citations"] = json.loads(out.get("result_citations") or "[]")
    except Exception:
        out["result_citations"] = []
    try:
        out["result_dashboard"] = json.loads(out.get("result_dashboard") or "{}")
    except Exception:
        out["result_dashboard"] = {}
    out.pop("request_json", None)
    return out
