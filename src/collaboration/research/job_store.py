"""
Deep Research 任务状态持久化（支持前端断连后恢复查看）。
底层存储已迁移至 data/rag.db，通过 SQLModel 访问。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import (
    DeepResearchJob,
    DRCheckpoint,
    DRGapSupplement,
    DRInsight,
    DRJobEvent,
    DRResumeQueue,
    DRSectionReview,
)


def create_job(
    *,
    topic: str,
    session_id: str = "",
    canvas_id: str = "",
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    job_id = uuid.uuid4().hex
    with Session(get_engine()) as session:
        row = DeepResearchJob(
            job_id=job_id,
            topic=topic,
            session_id=session_id,
            canvas_id=canvas_id,
            status="pending",
            request_json=json.dumps(request_payload or {}, ensure_ascii=False),
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
        row = session.get(DeepResearchJob, job_id)
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
        ev = DRJobEvent(
            job_id=job_id,
            event=event,
            data_json=json.dumps(data, ensure_ascii=False, default=str),
            created_at=now,
        )
        session.add(ev)
        row = session.get(DeepResearchJob, job_id)
        if row:
            row.updated_at = now
            session.add(row)
        session.commit()
        return ev.id or 0


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with Session(get_engine()) as session:
        row = session.get(DeepResearchJob, job_id)
    if not row:
        return None
    return row.to_dict()


def list_jobs(limit: int = 20, status: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with Session(get_engine()) as session:
        if status:
            stmt = (
                select(DeepResearchJob)
                .where(DeepResearchJob.status == status)
                .order_by(DeepResearchJob.created_at.desc())
                .limit(limit)
            )
        else:
            stmt = select(DeepResearchJob).order_by(DeepResearchJob.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
    return [r.to_dict() for r in rows]


# ── Checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(
    job_id: str,
    phase: str,
    state_dict: Dict[str, Any],
    section_title: str = "",
) -> Dict[str, Any]:
    now = time.time()
    normalized_phase = (phase or "").strip().lower()
    normalized_section = (section_title or "").strip()
    payload = json.dumps(state_dict or {}, ensure_ascii=False, default=str)
    with Session(get_engine()) as session:
        row = session.get(DRCheckpoint, (job_id, normalized_phase, normalized_section))
        if row is None:
            row = DRCheckpoint(
                job_id=job_id,
                phase=normalized_phase,
                section_title=normalized_section,
                state_json=payload,
                created_at=now,
            )
        else:
            row.state_json = payload
            row.created_at = now
        session.add(row)
        session.commit()
        session.refresh(row)
    return {
        "job_id": row.job_id,
        "phase": row.phase,
        "section_title": row.section_title,
        "state": row.get_state(),
        "created_at": row.created_at,
    }


def load_checkpoint(
    job_id: str,
    phase: Optional[str] = None,
    section_title: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with Session(get_engine()) as session:
        stmt = select(DRCheckpoint).where(DRCheckpoint.job_id == job_id)
        if phase is not None:
            stmt = stmt.where(DRCheckpoint.phase == phase.strip().lower())
        if section_title is not None:
            stmt = stmt.where(DRCheckpoint.section_title == section_title.strip())
        stmt = stmt.order_by(DRCheckpoint.created_at.desc()).limit(1)
        row = session.exec(stmt).first()
    if row is None:
        return None
    return {
        "job_id": row.job_id,
        "phase": row.phase,
        "section_title": row.section_title,
        "state": row.get_state(),
        "created_at": row.created_at,
    }


def list_checkpoints(job_id: str) -> List[Dict[str, Any]]:
    with Session(get_engine()) as session:
        stmt = (
            select(DRCheckpoint)
            .where(DRCheckpoint.job_id == job_id)
            .order_by(DRCheckpoint.created_at.desc())
        )
        rows = session.exec(stmt).all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "job_id": row.job_id,
                "phase": row.phase,
                "section_title": row.section_title,
                "state": row.get_state(),
                "created_at": row.created_at,
            }
        )
    return out


def list_events(job_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    after_id = max(0, int(after_id))
    limit = max(1, min(int(limit), 2000))
    with Session(get_engine()) as session:
        stmt = (
            select(DRJobEvent)
            .where(DRJobEvent.job_id == job_id)
            .where(DRJobEvent.id > after_id)
            .order_by(DRJobEvent.id.asc())
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


# ── Section Reviews ───────────────────────────────────────────────────────────

def submit_review(job_id: str, section_id: str, action: str = "approve", feedback: str = "") -> Dict[str, Any]:
    now = time.time()
    with Session(get_engine()) as session:
        stmt = select(DRSectionReview).where(
            DRSectionReview.job_id == job_id,
            DRSectionReview.section_id == section_id,
        )
        row = session.exec(stmt).first()
        if row is None:
            row = DRSectionReview(
                job_id=job_id,
                section_id=section_id,
                action=action,
                feedback=feedback,
                created_at=now,
            )
            session.add(row)
        else:
            row.action = action
            row.feedback = feedback
            row.created_at = now
            session.add(row)
        session.commit()
    return {"job_id": job_id, "section_id": section_id, "action": action}


def get_pending_review(job_id: str, section_id: str) -> Optional[Dict[str, Any]]:
    with Session(get_engine()) as session:
        stmt = select(DRSectionReview).where(
            DRSectionReview.job_id == job_id,
            DRSectionReview.section_id == section_id,
        )
        row = session.exec(stmt).first()
    return row.model_dump() if row else None


def list_reviews(job_id: str) -> List[Dict[str, Any]]:
    with Session(get_engine()) as session:
        stmt = (
            select(DRSectionReview)
            .where(DRSectionReview.job_id == job_id)
            .order_by(DRSectionReview.created_at.asc())
        )
        rows = session.exec(stmt).all()
    return [r.model_dump() for r in rows]


# ── Resume Queue ──────────────────────────────────────────────────────────────

def enqueue_resume_request(
    job_id: str,
    owner_instance: str,
    source: str = "review",
    message: str = "",
) -> Dict[str, Any]:
    """Enqueue one resume request. Idempotent when pending/running already exists."""
    now = time.time()
    with Session(get_engine()) as session:
        existing = session.exec(
            select(DRResumeQueue)
            .where(DRResumeQueue.job_id == job_id)
            .where(DRResumeQueue.status.in_(["pending", "running"]))
            .order_by(DRResumeQueue.id.desc())
            .limit(1)
        ).first()
        if existing:
            row = existing
            if row.owner_instance != owner_instance:
                row.owner_instance = owner_instance
                row.source = source
                row.message = message
                row.updated_at = now
                session.add(row)
                session.commit()
            return row.model_dump()

        row = DRResumeQueue(
            job_id=job_id,
            owner_instance=owner_instance,
            source=source,
            status="pending",
            message=message,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.model_dump()


def claim_resume_requests(owner_instance: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Atomically claim pending resume requests for this instance."""
    limit = max(1, min(int(limit), 100))
    now = time.time()
    claimed: List[Dict[str, Any]] = []
    with Session(get_engine()) as session:
        rows = session.exec(
            select(DRResumeQueue)
            .where(DRResumeQueue.status == "pending")
            .where(DRResumeQueue.owner_instance == owner_instance)
            .order_by(DRResumeQueue.created_at.asc())
            .limit(limit)
        ).all()
        for row in rows:
            # Re-check status inside transaction
            fresh = session.get(DRResumeQueue, row.id)
            if not fresh or fresh.status != "pending":
                continue
            fresh.status = "running"
            fresh.updated_at = now
            session.add(fresh)
            session.commit()
            session.refresh(fresh)
            claimed.append(fresh.model_dump())
    return claimed


def complete_resume_request(resume_id: int, status: str, message: str = "") -> None:
    """Mark a running resume request as done/error/cancelled."""
    if status not in {"done", "error", "cancelled"}:
        status = "error"
    now = time.time()
    with Session(get_engine()) as session:
        row = session.get(DRResumeQueue, int(resume_id))
        if row:
            row.status = status
            row.message = message
            row.updated_at = now
            session.add(row)
            session.commit()


def list_resume_requests(
    limit: int = 50,
    status: Optional[str] = None,
    owner_instance: Optional[str] = None,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List resume queue rows with optional filters."""
    limit = max(1, min(int(limit), 500))
    with Session(get_engine()) as session:
        stmt = select(DRResumeQueue)
        if status:
            stmt = stmt.where(DRResumeQueue.status == status)
        if owner_instance:
            stmt = stmt.where(DRResumeQueue.owner_instance == owner_instance)
        if job_id:
            stmt = stmt.where(DRResumeQueue.job_id == job_id)
        stmt = stmt.order_by(DRResumeQueue.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
    return [r.model_dump() for r in rows]


def cleanup_resume_requests(
    *,
    statuses: Optional[List[str]] = None,
    before_ts: Optional[float] = None,
    owner_instance: Optional[str] = None,
    job_id: Optional[str] = None,
) -> int:
    """Delete resume queue rows by filters."""
    effective_statuses = statuses or ["done", "error", "cancelled"]
    allowed = {"pending", "running", "done", "error", "cancelled"}
    effective_statuses = [s for s in effective_statuses if s in allowed]
    if not effective_statuses:
        return 0
    with Session(get_engine()) as session:
        stmt = select(DRResumeQueue).where(DRResumeQueue.status.in_(effective_statuses))
        if before_ts is not None:
            stmt = stmt.where(DRResumeQueue.updated_at <= float(before_ts))
        if owner_instance:
            stmt = stmt.where(DRResumeQueue.owner_instance == owner_instance)
        if job_id:
            stmt = stmt.where(DRResumeQueue.job_id == job_id)
        rows = session.exec(stmt).all()
        count = len(rows)
        for r in rows:
            session.delete(r)
        session.commit()
    return count


def retry_resume_request(
    *,
    resume_id: int,
    owner_instance: str,
    message: str = "manual retry",
) -> Optional[Dict[str, Any]]:
    """Retry a terminal resume request by setting it back to pending."""
    now = time.time()
    with Session(get_engine()) as session:
        row = session.get(DRResumeQueue, int(resume_id))
        if not row:
            return None
        if row.status in {"pending", "running"}:
            raise ValueError("resume request is already active")
        conflict = session.exec(
            select(DRResumeQueue)
            .where(DRResumeQueue.job_id == row.job_id)
            .where(DRResumeQueue.status.in_(["pending", "running"]))
            .where(DRResumeQueue.id != int(resume_id))
            .limit(1)
        ).first()
        if conflict:
            raise ValueError("job already has active resume request")
        row.owner_instance = owner_instance
        row.source = "manual_retry"
        row.status = "pending"
        row.message = message
        row.updated_at = now
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.model_dump()


# ── Gap Supplements ───────────────────────────────────────────────────────────

def submit_gap_supplement(
    job_id: str,
    section_id: str,
    gap_text: str,
    supplement_type: str = "material",
    content: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    with Session(get_engine()) as session:
        row = DRGapSupplement(
            job_id=job_id,
            section_id=section_id,
            gap_text=gap_text,
            supplement_type=supplement_type,
            content_json=json.dumps(content or {}, ensure_ascii=False),
            status="pending",
            created_at=now,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return {
            "id": row.id,
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
    with Session(get_engine()) as session:
        stmt = select(DRGapSupplement).where(DRGapSupplement.job_id == job_id)
        if section_id:
            stmt = stmt.where(DRGapSupplement.section_id == section_id)
        if status:
            stmt = stmt.where(DRGapSupplement.status == status)
        stmt = stmt.order_by(DRGapSupplement.created_at.asc())
        rows = session.exec(stmt).all()
    out = []
    for r in rows:
        item = r.model_dump()
        item["content"] = r.get_content()
        item.pop("content_json", None)
        out.append(item)
    return out


def mark_gap_supplement_consumed(supplement_id: int) -> None:
    now = time.time()
    with Session(get_engine()) as session:
        row = session.get(DRGapSupplement, int(supplement_id))
        if row:
            row.status = "consumed"
            row.consumed_at = now
            session.add(row)
            session.commit()


# ── Research Insights ─────────────────────────────────────────────────────────

def append_insight(
    job_id: str,
    insight_type: str,
    text: str,
    section_id: str = "",
    source_context: str = "",
) -> int:
    now = time.time()
    with Session(get_engine()) as session:
        row = DRInsight(
            job_id=job_id,
            section_id=section_id,
            insight_type=insight_type,
            text=text,
            source_context=source_context,
            status="open",
            created_at=now,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id or 0


def list_insights(
    job_id: str,
    insight_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with Session(get_engine()) as session:
        stmt = select(DRInsight).where(DRInsight.job_id == job_id)
        if insight_type:
            stmt = stmt.where(DRInsight.insight_type == insight_type)
        if status:
            stmt = stmt.where(DRInsight.status == status)
        stmt = stmt.order_by(DRInsight.created_at.asc())
        rows = session.exec(stmt).all()
    return [r.model_dump() for r in rows]


def update_insight_status(insight_id: int, status: str) -> None:
    with Session(get_engine()) as session:
        row = session.get(DRInsight, int(insight_id))
        if row:
            row.status = status
            session.add(row)
            session.commit()


def bulk_mark_insights_addressed(job_id: str, insight_type: Optional[str] = None) -> int:
    with Session(get_engine()) as session:
        stmt = select(DRInsight).where(
            DRInsight.job_id == job_id,
            DRInsight.status == "open",
        )
        if insight_type:
            stmt = stmt.where(DRInsight.insight_type == insight_type)
        rows = session.exec(stmt).all()
        count = len(rows)
        for r in rows:
            r.status = "addressed"
            session.add(r)
        session.commit()
    return count


def get_latest_job_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    """按 session_id 获取最近一条 Deep Research 任务。"""
    if not session_id:
        return None
    with Session(get_engine()) as session:
        stmt = (
            select(DeepResearchJob)
            .where(DeepResearchJob.session_id == session_id)
            .order_by(DeepResearchJob.updated_at.desc())
            .limit(1)
        )
        row = session.exec(stmt).first()
    if not row:
        return None
    return row.to_dict()
