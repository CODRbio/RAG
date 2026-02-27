"""
Background task worker: polls the consolidated rag.db job queues and executes
tasks via asyncio.to_thread.

Previously used direct sqlite3.connect() against two separate .db files.
Now uses SQLModel sessions via the shared engine in src/db/engine.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import DeepResearchJob, DRResumeQueue, IngestJob
from src.log import get_logger

logger = get_logger(__name__)

# ── Concurrency caps ──────────────────────────────────────────────────────────
_DR_MAX_CONCURRENT = 2
_INGEST_MAX_CONCURRENT = 2
_POLL_INTERVAL = 5  # seconds

# ── Worker instance id (used by db resume-queue routing) ─────────────────────
_WORKER_INSTANCE_ID = f"{os.getenv('HOSTNAME', 'local')}:{os.getpid()}"


def get_worker_instance_id() -> str:
    return _WORKER_INSTANCE_ID


# ──────────────────────────────────────────────────────────────────────────────
# Startup cleanup
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_stale_jobs() -> None:
    """Reset running/cancelling jobs to error — they were interrupted by a process restart."""
    now = time.time()
    try:
        with Session(get_engine()) as session:
            dr_rows = session.exec(
                select(DeepResearchJob).where(
                    DeepResearchJob.status.in_(["running", "cancelling", "waiting_review"])
                )
            ).all()
            count_dr = len(dr_rows)
            for row in dr_rows:
                row.status = "error"
                row.error_message = "服务重启，任务中断"
                row.message = "服务重启，任务中断"
                row.updated_at = now
                row.finished_at = now
                session.add(row)

            # Reset stale resume queue entries
            rq_rows = session.exec(
                select(DRResumeQueue).where(DRResumeQueue.status == "running")
            ).all()
            for rq in rq_rows:
                rq.status = "error"
                rq.message = "服务重启，恢复请求失效"
                rq.updated_at = now
                session.add(rq)

            ingest_rows = session.exec(
                select(IngestJob).where(IngestJob.status.in_(["running", "cancelling"]))
            ).all()
            count_ingest = len(ingest_rows)
            for row in ingest_rows:
                row.status = "error"
                row.error_message = "服务重启，任务中断"
                row.message = "服务重启，任务中断"
                row.updated_at = now
                row.finished_at = now
                session.add(row)

            session.commit()

        if count_dr > 0:
            logger.warning("[task_runner] reset %d stale deep-research job(s) to error", count_dr)
        if count_ingest > 0:
            logger.warning("[task_runner] reset %d stale ingest job(s) to error", count_ingest)

    except Exception as e:
        logger.warning("[task_runner] cleanup_stale_jobs failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Pending-job fetch + atomic claim
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_pending_dr(limit: int, exclude: set) -> List[Dict[str, Any]]:
    """Read pending deep-research jobs, skipping already-tracked IDs."""
    if limit <= 0:
        return []
    try:
        with Session(get_engine()) as session:
            rows = session.exec(
                select(DeepResearchJob)
                .where(DeepResearchJob.status == "pending")
                .order_by(DeepResearchJob.created_at.asc())
                .limit(limit + len(exclude))
            ).all()
        result = []
        for row in rows:
            if row.job_id in exclude:
                continue
            try:
                data = json.loads(row.request_json or "{}")
            except Exception:
                data = {}
            result.append({"job_id": row.job_id, "data": data})
            if len(result) >= limit:
                break
        return result
    except Exception as e:
        logger.debug("[task_runner] fetch pending DR failed: %s", e)
        return []


def _fetch_pending_ingest(limit: int, exclude: set) -> List[Dict[str, Any]]:
    """Read pending ingest jobs, skipping already-tracked IDs."""
    if limit <= 0:
        return []
    try:
        with Session(get_engine()) as session:
            rows = session.exec(
                select(IngestJob)
                .where(IngestJob.status == "pending")
                .order_by(IngestJob.created_at.asc())
                .limit(limit + len(exclude))
            ).all()
        result = []
        for row in rows:
            if row.job_id in exclude:
                continue
            try:
                data = json.loads(row.payload_json or "{}")
            except Exception:
                data = {}
            result.append({"job_id": row.job_id, "data": data})
            if len(result) >= limit:
                break
        return result
    except Exception as e:
        logger.debug("[task_runner] fetch pending ingest failed: %s", e)
        return []


def _claim_dr_job(job_id: str) -> bool:
    """Atomically claim a pending DR job: pending → running. Returns True on success."""
    now = time.time()
    try:
        with Session(get_engine()) as session:
            row = session.get(DeepResearchJob, job_id)
            if not row or row.status != "pending":
                return False
            row.status = "running"
            row.message = "Worker 已领取任务，准备执行"
            row.updated_at = now
            session.add(row)
            session.commit()
        return True
    except Exception as e:
        logger.warning("[task_runner] claim DR job failed job_id=%s err=%s", job_id, e)
        return False


def _claim_ingest_job(job_id: str) -> bool:
    """Atomically claim a pending ingest job: pending → running. Returns True on success."""
    now = time.time()
    try:
        with Session(get_engine()) as session:
            row = session.get(IngestJob, job_id)
            if not row or row.status != "pending":
                return False
            row.status = "running"
            row.message = "Worker 已领取任务，准备执行"
            row.updated_at = now
            session.add(row)
            session.commit()
        return True
    except Exception as e:
        logger.warning("[task_runner] claim ingest job failed job_id=%s err=%s", job_id, e)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Per-job executors
# ──────────────────────────────────────────────────────────────────────────────

async def _exec_dr_job(job_id: str, request: dict) -> None:
    """Execute a deep-research job; wraps the existing sync business logic."""
    try:
        from src.api.routes_chat import _run_deep_research_job_safe
        from src.api.schemas import DeepResearchConfirmRequest

        optional_user_id = request.get("_worker_user_id")
        restart_spec = request.get("_restart")
        body_data = {k: v for k, v in request.items() if k != "_worker_user_id"}
        body_data.pop("_restart", None)
        body = DeepResearchConfirmRequest(**body_data)

        await asyncio.to_thread(
            _run_deep_research_job_safe,
            job_id=job_id,
            body=body,
            optional_user_id=optional_user_id,
            restart_spec=restart_spec if isinstance(restart_spec, dict) else None,
        )
    except Exception as e:
        logger.error("[task_runner] DR job %s failed: %s", job_id, e, exc_info=True)
        try:
            from src.collaboration.research.job_store import update_job
            update_job(
                job_id,
                status="error",
                error_message=str(e),
                message=f"Worker 执行失败: {e}",
                finished_at=time.time(),
            )
        except Exception:
            pass


async def _exec_resume(job_id: str) -> None:
    """Resume a suspended deep-research job (in-memory langgraph state)."""
    try:
        from src.api.routes_chat import _dr_get_suspended_runtime, _resume_suspended_job

        if not _dr_get_suspended_runtime(job_id):
            raise RuntimeError("当前实例不存在挂起上下文，无法恢复")

        await asyncio.to_thread(_resume_suspended_job, job_id)
    except Exception as e:
        logger.error("[task_runner] DR resume %s failed: %s", job_id, e, exc_info=True)
        raise


async def _exec_ingest_job(job_id: str, cfg: dict) -> None:
    """Execute an ingest job; wraps the existing sync business logic."""
    try:
        from src.api.routes_ingest import _run_ingest_job_safe

        await asyncio.to_thread(_run_ingest_job_safe, job_id, cfg)
    except Exception as e:
        logger.error("[task_runner] ingest job %s failed: %s", job_id, e, exc_info=True)
        try:
            from src.indexing.ingest_job_store import update_job
            update_job(
                job_id,
                status="error",
                error_message=str(e),
                message=f"Worker 执行失败: {e}",
                finished_at=time.time(),
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main polling loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_background_worker() -> None:
    """Poll rag.db for pending tasks every POLL_INTERVAL seconds and dispatch them."""
    logger.info(
        "[task_runner] background worker started (instance=%s, poll=%ds, dr_max=%d, ingest_max=%d)",
        _WORKER_INSTANCE_ID,
        _POLL_INTERVAL,
        _DR_MAX_CONCURRENT,
        _INGEST_MAX_CONCURRENT,
    )

    dr_tasks: dict[str, asyncio.Task] = {}
    ingest_tasks: dict[str, asyncio.Task] = {}

    while True:
        try:
            # ── Prune completed tasks ──
            for jid in [k for k, t in dr_tasks.items() if t.done()]:
                dr_tasks.pop(jid, None)
            for jid in [k for k, t in ingest_tasks.items() if t.done()]:
                ingest_tasks.pop(jid, None)

            # ── Deep Research: process resume queue first (same DR slots) ──
            dr_available = _DR_MAX_CONCURRENT - len(dr_tasks)
            if dr_available > 0:
                try:
                    from src.collaboration.research.job_store import (
                        claim_resume_requests,
                        complete_resume_request,
                    )
                    resume_rows = claim_resume_requests(_WORKER_INSTANCE_ID, limit=dr_available)
                    for row in resume_rows:
                        rid = int(row.get("id") or 0)
                        jid = str(row.get("job_id") or "")
                        if not jid or rid <= 0:
                            complete_resume_request(rid, "error", "恢复请求缺少 job_id")
                            continue
                        existing = dr_tasks.get(jid)
                        if existing and not existing.done():
                            complete_resume_request(rid, "error", "任务正在执行，无法恢复")
                            continue

                        async def _run_resume_request(resume_id: int, job_id: str) -> None:
                            try:
                                await _exec_resume(job_id)
                                complete_resume_request(resume_id, "done", "恢复请求执行完成")
                            except Exception as e:
                                complete_resume_request(resume_id, "error", f"恢复请求失败: {e}")

                        logger.info("[task_runner] claimed resume request id=%s job=%s", rid, jid)
                        dr_tasks[jid] = asyncio.create_task(_run_resume_request(rid, jid))
                except Exception as e:
                    logger.warning("[task_runner] resume queue poll failed: %s", e)

            # ── Deep Research: claim pending jobs ──
            dr_available = _DR_MAX_CONCURRENT - len(dr_tasks)
            if dr_available > 0:
                pending = _fetch_pending_dr(limit=dr_available, exclude=set(dr_tasks.keys()))
                for job in pending:
                    jid = job["job_id"]
                    if not _claim_dr_job(jid):
                        continue
                    logger.info("[task_runner] claimed DR job %s", jid)
                    dr_tasks[jid] = asyncio.create_task(_exec_dr_job(jid, job["data"]))

            # ── Ingest: claim pending jobs ──
            ingest_available = _INGEST_MAX_CONCURRENT - len(ingest_tasks)
            if ingest_available > 0:
                pending = _fetch_pending_ingest(limit=ingest_available, exclude=set(ingest_tasks.keys()))
                for job in pending:
                    jid = job["job_id"]
                    if not _claim_ingest_job(jid):
                        continue
                    logger.info("[task_runner] claimed ingest job %s", jid)
                    ingest_tasks[jid] = asyncio.create_task(_exec_ingest_job(jid, job["data"]))

        except Exception as e:
            logger.error("[task_runner] poll cycle error: %s", e, exc_info=True)

        await asyncio.sleep(_POLL_INTERVAL)
