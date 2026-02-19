"""
Background task worker: polls SQLite job queues and executes tasks via asyncio.to_thread.

Replaces in-memory ThreadPoolExecutor so that pending tasks survive process restarts.
Worker is started once inside the FastAPI lifespan hook.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from src.log import get_logger

logger = get_logger(__name__)

# ── DB paths (mirror the paths used by job_store / ingest_job_store) ──
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DR_DB = _DATA_DIR / "deep_research_jobs.db"
_INGEST_DB = _DATA_DIR / "ingest_jobs.db"

# ── Concurrency caps (same as the old ThreadPoolExecutor max_workers) ──
_DR_MAX_CONCURRENT = 2
_INGEST_MAX_CONCURRENT = 2
_POLL_INTERVAL = 5  # seconds

# ── worker instance id (used by db resume-queue routing) ──
_WORKER_INSTANCE_ID = f"{os.getenv('HOSTNAME', 'local')}:{os.getpid()}"


def get_worker_instance_id() -> str:
    return _WORKER_INSTANCE_ID


# ────────────────────────────────────────────────
# Startup cleanup
# ────────────────────────────────────────────────

def cleanup_stale_jobs() -> None:
    """Reset running/cancelling jobs to error — they were interrupted by a process restart."""
    now = time.time()
    if _DR_DB.exists():
        try:
            conn = sqlite3.connect(str(_DR_DB), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.execute(
                """
                UPDATE deep_research_jobs
                SET status = 'error',
                    error_message = '服务重启，任务中断',
                    message = '服务重启，任务中断',
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ('running', 'cancelling', 'waiting_review')
                """,
                (now, now),
            )
            if cur.rowcount > 0:
                logger.warning(
                    "[task_runner] reset %d stale deep-research job(s) to error",
                    cur.rowcount,
                )
            conn.commit()
            # Reset stale resume-queue entries for current process crash.
            try:
                conn.execute(
                    """
                    UPDATE deep_research_resume_queue
                    SET status = 'error',
                        message = '服务重启，恢复请求失效',
                        updated_at = ?
                    WHERE status = 'running'
                    """,
                    (now,),
                )
                conn.commit()
            except Exception:
                # resume queue table may not exist before first migration; ignore safely
                pass
            conn.close()
        except Exception as e:
            logger.warning("[task_runner] cleanup deep-research stale jobs failed: %s", e)
    if _INGEST_DB.exists():
        try:
            conn = sqlite3.connect(str(_INGEST_DB), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = 'error',
                    error_message = '服务重启，任务中断',
                    message = '服务重启，任务中断',
                    updated_at = ?,
                    finished_at = ?
                WHERE status IN ('running', 'cancelling')
                """,
                (now, now),
            )
            if cur.rowcount > 0:
                logger.warning(
                    "[task_runner] reset %d stale ingest job(s) to error",
                    cur.rowcount,
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("[task_runner] cleanup ingest stale jobs failed: %s", e)


# ────────────────────────────────────────────────
# Pending-job fetch (lightweight, no schema init)
# ────────────────────────────────────────────────

def _fetch_pending(
    db_path: Path,
    table: str,
    json_col: str,
    limit: int,
    exclude: set[str],
) -> List[Dict[str, Any]]:
    """Read pending rows from *table*, skipping IDs already tracked in *exclude*."""
    if not db_path.exists() or limit <= 0:
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            f"""
            SELECT job_id, {json_col} FROM {table}
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit + len(exclude),),
        ).fetchall()
        conn.close()

        result: list[Dict[str, Any]] = []
        for row in rows:
            jid = row["job_id"]
            if jid in exclude:
                continue
            try:
                data = json.loads(row[json_col] or "{}")
            except Exception:
                data = {}
            result.append({"job_id": jid, "data": data})
            if len(result) >= limit:
                break
        return result
    except Exception as e:
        logger.debug("[task_runner] fetch pending from %s failed: %s", table, e)
        return []


def _claim_pending_job(db_path: Path, table: str, job_id: str) -> bool:
    """
    Atomically claim a pending job by switching pending -> running.
    Returns True only when the row was still pending and got claimed.
    """
    if not db_path.exists():
        return False
    now = time.time()
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        cur = conn.execute(
            f"""
            UPDATE {table}
            SET status = 'running',
                message = 'Worker 已领取任务，准备执行',
                updated_at = ?
            WHERE job_id = ? AND status = 'pending'
            """,
            (now, job_id),
        )
        conn.commit()
        conn.close()
        return cur.rowcount > 0
    except Exception as e:
        logger.warning("[task_runner] claim job failed table=%s job_id=%s err=%s", table, job_id, e)
        return False


# ────────────────────────────────────────────────
# Per-job executors (each runs in a worker thread)
# ────────────────────────────────────────────────

async def _exec_dr_job(job_id: str, request: dict) -> None:
    """Execute a deep-research job; wraps the existing sync business logic."""
    try:
        from src.api.routes_chat import _run_deep_research_job_safe
        from src.api.schemas import DeepResearchConfirmRequest

        optional_user_id = request.get("_worker_user_id")
        body_data = {k: v for k, v in request.items() if k != "_worker_user_id"}
        body = DeepResearchConfirmRequest(**body_data)

        await asyncio.to_thread(
            _run_deep_research_job_safe,
            job_id=job_id,
            body=body,
            optional_user_id=optional_user_id,
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


# ────────────────────────────────────────────────
# Main polling loop
# ────────────────────────────────────────────────

async def run_background_worker() -> None:
    """Poll SQLite for pending tasks every POLL_INTERVAL seconds and dispatch them."""
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
                pending = _fetch_pending(
                    _DR_DB,
                    "deep_research_jobs",
                    "request_json",
                    limit=dr_available,
                    exclude=set(dr_tasks.keys()),
                )
                for job in pending:
                    jid = job["job_id"]
                    if not _claim_pending_job(_DR_DB, "deep_research_jobs", jid):
                        continue
                    logger.info("[task_runner] claimed DR job %s", jid)
                    dr_tasks[jid] = asyncio.create_task(_exec_dr_job(jid, job["data"]))

            # ── Ingest: claim pending jobs ──
            ingest_available = _INGEST_MAX_CONCURRENT - len(ingest_tasks)
            if ingest_available > 0:
                pending = _fetch_pending(
                    _INGEST_DB,
                    "ingest_jobs",
                    "payload_json",
                    limit=ingest_available,
                    exclude=set(ingest_tasks.keys()),
                )
                for job in pending:
                    jid = job["job_id"]
                    if not _claim_pending_job(_INGEST_DB, "ingest_jobs", jid):
                        continue
                    logger.info("[task_runner] claimed ingest job %s", jid)
                    ingest_tasks[jid] = asyncio.create_task(
                        _exec_ingest_job(jid, job["data"])
                    )

        except Exception as e:
            logger.error("[task_runner] poll cycle error: %s", e, exc_info=True)

        await asyncio.sleep(_POLL_INTERVAL)
