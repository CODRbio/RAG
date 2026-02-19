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
from src.db.models import IngestJob, IngestJobEvent


def create_job(collection: str, payload: Dict[str, Any], total_files: int) -> Dict[str, Any]:
    now = time.time()
    job_id = uuid.uuid4().hex
    with Session(get_engine()) as session:
        row = IngestJob(
            job_id=job_id,
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


def list_jobs(limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with Session(get_engine()) as session:
        stmt = select(IngestJob).order_by(IngestJob.created_at.desc()).limit(limit)
        if status:
            stmt = select(IngestJob).where(IngestJob.status == status).order_by(IngestJob.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
    return [r.to_dict() for r in rows]


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
