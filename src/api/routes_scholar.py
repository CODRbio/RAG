"""
Scholar API: search + download + ingest.
Prefix: /scholar
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from config.settings import settings
from src.log import get_logger
from src.tasks.dispatcher import process_download_and_ingest
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus

logger = get_logger(__name__)
router = APIRouter(prefix="/scholar", tags=["scholar"])


class ScholarSearchRequest(BaseModel):
    query: str
    source: str = "google_scholar"  # google_scholar | semantic | ncbi | annas_archive
    limit: int = 10
    year_start: Optional[int] = None
    year_end: Optional[int] = None


class DownloadRequest(BaseModel):
    title: str
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    annas_md5: Optional[str] = None
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
    """Batch download + ingest; each paper is a separate background task."""
    cfg = getattr(settings, "scholar_downloader", None)
    if not cfg or not cfg.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    task_id = f"batch_dl_{uuid.uuid4().hex[:8]}"

    async def _batch_job():
        for paper in req.papers:
            sub_id = f"{task_id}_{uuid.uuid4().hex[:4]}"
            await process_download_and_ingest(
                task_id=sub_id,
                paper_info=paper.model_dump(),
                collection=req.collection,
            )

    background_tasks.add_task(_batch_job)
    return {
        "status": "submitted",
        "task_id": task_id,
        "total": len(req.papers),
        "message": f"批量下载任务已投递（{len(req.papers)} 篇）",
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
    from src.retrieval.downloader.adapter import _adapter_instance

    cfg = getattr(settings, "scholar_downloader", None)
    return {
        "enabled": bool(cfg and cfg.enabled),
        "adapter_ready": _adapter_instance is not None,
        "download_dir": getattr(cfg, "download_dir", "") if cfg else "",
    }
