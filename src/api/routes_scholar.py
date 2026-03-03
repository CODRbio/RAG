"""
Scholar API: search + download + ingest.
Prefix: /scholar
"""

import asyncio
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings
from src.log import get_logger
from src.tasks.dispatcher import process_download_and_ingest
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState

logger = get_logger(__name__)
router = APIRouter(prefix="/scholar", tags=["scholar"])


class ScholarSearchRequest(BaseModel):
    query: str
    source: str = "google_scholar"  # google_scholar | semantic | ncbi | annas_archive
    limit: int = 10
    year_start: Optional[int] = None
    year_end: Optional[int] = None


class DownloadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    doi: Optional[str] = Field(None, pattern=r"^10\.\d{4,}/\S+")
    pdf_url: Optional[str] = None
    annas_md5: Optional[str] = Field(None, pattern=r"^[0-9a-f]{32}$")
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    collection: Optional[str] = None
    auto_ingest: Optional[bool] = None


class BatchDownloadRequest(BaseModel):
    papers: List[DownloadRequest]
    collection: Optional[str] = None
    max_concurrent: int = 3


@router.post("/search")
async def scholar_search(req: ScholarSearchRequest):
    """Unified search; all sources use RAG modules except annas_archive (adapter)."""
    if not getattr(settings, "scholar_downloader", None) or not settings.scholar_downloader.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")
    from src.retrieval.downloader.adapter import get_adapter

    adapter = get_adapter()

    if req.source == "google_scholar":
        results = await adapter.search_google_scholar(
            query=req.query,
            limit=req.limit,
            year_start=req.year_start,
            year_end=req.year_end,
        )
    elif req.source == "semantic":
        results = await adapter.search_semantic_scholar(query=req.query, limit=req.limit)
    elif req.source == "ncbi":
        results = await adapter.search_ncbi(query=req.query, limit=req.limit)
    elif req.source == "annas_archive":
        results = await adapter.search_annas_archive(query=req.query, limit=req.limit)
    else:
        raise HTTPException(status_code=400, detail=f"未知搜索源: {req.source}")

    return {"results": results, "count": len(results)}


@router.post("/download")
async def scholar_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Download one paper PDF.
    If auto_ingest=True (default from config), runs download then ingest in background and returns task_id.
    Otherwise runs download synchronously and returns result.
    """
    cfg = getattr(settings, "scholar_downloader", None)
    if not cfg or not cfg.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    should_ingest = (
        req.auto_ingest if req.auto_ingest is not None else cfg.auto_ingest_after_download
    )

    if should_ingest:
        task_id = f"dl_ingest_{uuid.uuid4().hex[:8]}"
        from src.tasks.redis_queue import get_task_queue
        from src.tasks.task_state import TaskKind, TaskState

        q = get_task_queue()
        state = TaskState(
            task_id=task_id,
            kind=TaskKind.scholar,
            status=TaskStatus.queued,
            payload={
                "paper_info": req.model_dump(),
                "collection": req.collection or "",
            },
        )
        q.set_state(state)
        background_tasks.add_task(
            process_download_and_ingest,
            task_id=task_id,
            paper_info=req.model_dump(),
            collection=req.collection,
        )
        return {
            "status": "submitted",
            "task_id": task_id,
            "message": "下载并入库任务已投递至后台",
        }

    from src.retrieval.downloader.adapter import get_adapter

    result = await get_adapter().download_paper(
        title=req.title,
        doi=req.doi,
        pdf_url=req.pdf_url,
        annas_md5=req.annas_md5,
        authors=req.authors,
        year=req.year,
    )
    return result


@router.post("/download/batch")
async def scholar_batch_download(req: BatchDownloadRequest, background_tasks: BackgroundTasks):
    """Batch download + ingest; concurrent with semaphore; parent task state for task_id."""
    cfg = getattr(settings, "scholar_downloader", None)
    if not cfg or not cfg.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    task_id = f"batch_dl_{uuid.uuid4().hex[:8]}"
    total = len(req.papers)
    q = get_task_queue()
    parent = TaskState(
        task_id=task_id,
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={
            "total": total,
            "completed": 0,
            "failed": 0,
            "collection": req.collection or "",
        },
    )
    q.set_state(parent)

    async def _batch_job():
        sem = asyncio.Semaphore(req.max_concurrent)
        completed = 0
        failed = 0
        progress_lock = asyncio.Lock()

        async def _one(paper: DownloadRequest):
            nonlocal completed, failed
            sub_id = f"{task_id}_{uuid.uuid4().hex[:4]}"
            result = await process_download_and_ingest(
                task_id=sub_id,
                paper_info=paper.model_dump(),
                collection=req.collection,
            )
            async with progress_lock:
                if result.get("ingest_triggered") or result.get("success"):
                    completed += 1
                else:
                    failed += 1
                state = q.get_state(task_id)
                if state:
                    state.payload["completed"] = completed
                    state.payload["failed"] = failed
                    state.payload["total"] = total
                    q.set_state(state)
                    q.push_event(
                        task_id,
                        "progress",
                        {"completed": completed, "failed": failed, "total": total},
                    )

        await asyncio.gather(*[sem_wrap(sem, _one, p) for p in req.papers])

        state = q.get_state(task_id)
        if state:
            state.finished_at = time.time()
            state.payload["completed"] = completed
            state.payload["failed"] = failed
            if failed == total:
                state.status = TaskStatus.error
                state.error_message = "所有论文下载均失败"
            else:
                state.status = TaskStatus.completed
            q.set_state(state)
            q.push_event(
                task_id,
                "done",
                {"completed": completed, "failed": failed, "total": total},
            )

    async def sem_wrap(sem, coro_fn, paper):
        async with sem:
            return await coro_fn(paper)

    background_tasks.add_task(lambda: asyncio.run(_batch_job()))
    return {
        "status": "submitted",
        "task_id": task_id,
        "total": total,
        "message": f"批量下载任务已投递（{total} 篇）",
    }


@router.get("/task/{task_id}")
async def scholar_task_status(task_id: str):
    """Poll download+ingest task status (Redis task state + payload)."""
    try:
        q = get_task_queue()
        state = q.get_state(task_id)
    except Exception as e:
        logger.warning("scholar task status get_task_queue failed: %s", e)
        raise HTTPException(status_code=503, detail="Task store unavailable")

    if not state:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    if state.kind != TaskKind.scholar:
        raise HTTPException(status_code=400, detail="Not a scholar task")

    return {
        "task_id": task_id,
        "status": state.status.value,
        "error_message": state.error_message,
        "payload": state.payload,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
    }


@router.get("/health")
async def scholar_health():
    from src.retrieval.downloader.adapter import is_adapter_ready

    cfg = getattr(settings, "scholar_downloader", None)
    return {
        "enabled": bool(cfg and cfg.enabled),
        "adapter_ready": is_adapter_ready(),
        "download_dir": getattr(cfg, "download_dir", "") if cfg else "",
    }
