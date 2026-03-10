"""
Ingest 任务状态持久化（支持前端断连后重连查看进度）。
底层存储已迁移至 data/rag.db (ingest_jobs / ingest_job_events 表)，通过 SQLModel 访问。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import IngestJob, IngestJobEvent, IngestCheckpoint


def create_job(collection: str, payload: Dict[str, Any], total_files: int) -> Dict[str, Any]:
    now = time.time()
    job_id = uuid.uuid4().hex
    user_id = payload.get("user_id", "default")
    with Session(get_engine()) as session:
        row = IngestJob(
            job_id=job_id,
            user_id=user_id,
            collection=collection,
            status="pending",
            total_files=int(total_files),
            payload_json=json.dumps(payload, ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.commit()
    return get_job(job_id) or {}


def update_job(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_job(job_id)
    fields["updated_at"] = time.time()
    with Session(get_engine()) as session:
        row = session.get(IngestJob, job_id)
        if not row:
            return None
        for k, v in fields.items():
            if hasattr(row, k):
                setattr(row, k, v)
        session.add(row)
        session.commit()
    return get_job(job_id)


def append_event(job_id: str, event: str, data: Dict[str, Any]) -> int:
    now = time.time()
    with Session(get_engine()) as session:
        ev = IngestJobEvent(
            job_id=job_id,
            event=event,
            data_json=json.dumps(data, ensure_ascii=False),
            created_at=now,
        )
        session.add(ev)

        # Update parent job's updated_at
        row = session.get(IngestJob, job_id)
        if row:
            row.updated_at = now
            session.add(row)

        session.commit()
        return ev.id or 0


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with Session(get_engine()) as session:
        row = session.get(IngestJob, job_id)
    if not row:
        return None
    return row.to_dict()


def count_jobs_by_status(status: str) -> int:
    """Return the number of ingest jobs with the given status (e.g. 'running', 'pending')."""
    with Session(get_engine()) as session:
        stmt = select(IngestJob).where(IngestJob.status == status)
        rows = session.exec(stmt).all()
    return len(rows)


def list_jobs(limit: int = 20, status: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with Session(get_engine()) as session:
        stmt = select(IngestJob)
        if user_id is not None:
            stmt = stmt.where(IngestJob.user_id == user_id)
        if status:
            stmt = stmt.where(IngestJob.status == status)
        stmt = stmt.order_by(IngestJob.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
    return [r.to_dict() for r in rows]


_STAGE_ORDER = ["parsed", "chunked", "embedded", "indexed"]


def list_events(job_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    after_id = max(0, int(after_id))
    limit = max(1, min(int(limit), 2000))
    with Session(get_engine()) as session:
        stmt = (
            select(IngestJobEvent)
            .where(IngestJobEvent.job_id == job_id)
            .where(IngestJobEvent.id > after_id)
            .order_by(IngestJobEvent.id.asc())
            .limit(limit)
        )
        rows = session.exec(stmt).all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            data = json.loads(r.data_json or "{}")
        except Exception:
            data = {}
        data["event_id"] = int(r.id or 0)
        out.append(
            {
                "event_id": int(r.id or 0),
                "event": str(r.event),
                "created_at": float(r.created_at),
                "data": data,
            }
        )
    return out


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ingest_checkpoint(
    job_id: str,
    file_name: str,
    stage: str,
    status: str = "done",
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upsert a per-file stage checkpoint. Safe to call multiple times (idempotent)."""
    now = time.time()
    detail_json = json.dumps(detail or {}, ensure_ascii=False)
    with Session(get_engine()) as session:
        row = session.get(IngestCheckpoint, (job_id, file_name, stage))
        if row is None:
            row = IngestCheckpoint(
                job_id=job_id,
                file_name=file_name,
                stage=stage,
                status=status,
                detail_json=detail_json,
                created_at=now,
            )
        else:
            row.status = status
            row.detail_json = detail_json
            row.created_at = now
        session.add(row)
        session.commit()
    return {"job_id": job_id, "file_name": file_name, "stage": stage, "status": status}


def load_ingest_checkpoints(job_id: str, file_name: str) -> List[Dict[str, Any]]:
    """Return all completed-stage checkpoints for one file within a job."""
    with Session(get_engine()) as session:
        stmt = (
            select(IngestCheckpoint)
            .where(IngestCheckpoint.job_id == job_id)
            .where(IngestCheckpoint.file_name == file_name)
            .order_by(IngestCheckpoint.created_at.asc())
        )
        rows = session.exec(stmt).all()
    return [
        {
            "job_id": r.job_id,
            "file_name": r.file_name,
            "stage": r.stage,
            "status": r.status,
            "detail": r.get_detail(),
            "created_at": r.created_at,
        }
        for r in rows
    ]


def get_last_completed_stage(job_id: str, file_name: str) -> Optional[str]:
    """Return the last successfully completed stage name for a file, or None."""
    checkpoints = load_ingest_checkpoints(job_id, file_name)
    done_stages = {c["stage"] for c in checkpoints if c["status"] == "done"}
    for stage in reversed(_STAGE_ORDER):
        if stage in done_stages:
            return stage
    return None


def list_all_checkpoints(job_id: str) -> List[Dict[str, Any]]:
    """Return all checkpoint rows for a job, ordered by file then stage."""
    with Session(get_engine()) as session:
        stmt = (
            select(IngestCheckpoint)
            .where(IngestCheckpoint.job_id == job_id)
            .order_by(IngestCheckpoint.file_name.asc(), IngestCheckpoint.created_at.asc())
        )
        rows = session.exec(stmt).all()
    return [
        {
            "job_id": r.job_id,
            "file_name": r.file_name,
            "stage": r.stage,
            "status": r.status,
            "detail": r.get_detail(),
            "created_at": r.created_at,
        }
        for r in rows
    ]


def purge_ingest_checkpoints(job_id: str) -> int:
    """Delete all checkpoint rows for a job. Called on job done or cancel."""
    with Session(get_engine()) as session:
        stmt = select(IngestCheckpoint).where(IngestCheckpoint.job_id == job_id)
        rows = session.exec(stmt).all()
        count = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    return count


def purge_stale_ingest_checkpoints(max_age_days: int = 7) -> int:
    """TTL safety net: delete checkpoints for terminal jobs or older than max_age_days.
    Called at startup alongside storage cleanup."""
    import time as _time
    cutoff = _time.time() - max_age_days * 86400
    with Session(get_engine()) as session:
        # Delete checkpoints whose parent job is in a terminal state
        terminal_job_ids_stmt = (
            select(IngestJob.job_id)
            .where(IngestJob.status.in_(["done", "error", "cancelled"]))
        )
        terminal_ids = [r for r in session.exec(terminal_job_ids_stmt).all()]
        count = 0
        if terminal_ids:
            stmt = select(IngestCheckpoint).where(IngestCheckpoint.job_id.in_(terminal_ids))
            rows = session.exec(stmt).all()
            count += len(rows)
            for row in rows:
                session.delete(row)
        # Also delete any orphaned checkpoints older than TTL (belt-and-suspenders)
        old_stmt = (
            select(IngestCheckpoint)
            .where(IngestCheckpoint.created_at < cutoff)
        )
        old_rows = session.exec(old_stmt).all()
        for row in old_rows:
            if row not in session.identity_map.values():
                count += 1
                session.delete(row)
        session.commit()
    return count
