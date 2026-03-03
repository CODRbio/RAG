"""
Scholar API: search + download + ingest.
Prefix: /scholar

Google Scholar / Google search use RAG unified_web_search (SerpAPI + Playwright + serpapi_ratio).
"""

import asyncio
import json
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from config.settings import settings
from src.db.engine import get_engine
from src.db.models import ScholarLibrary, ScholarLibraryPaper
from src.log import get_logger
from src.tasks.dispatcher import process_download_and_ingest
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState

logger = get_logger(__name__)
router = APIRouter(prefix="/scholar", tags=["scholar"])


class ScholarSearchRequest(BaseModel):
    query: str
    source: str = "google_scholar"  # google_scholar | google | semantic | semantic_relevance | semantic_bulk | ncbi | annas_archive
    limit: int = 30
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    optimize: bool = False  # when True, use ultra_lite to rewrite query per source before search
    use_serpapi: bool = False  # when True for google_scholar/google, part of queries go via SerpAPI (see serpapi_ratio)
    serpapi_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)  # 0–1, share of queries to SerpAPI; default 0.5 when use_serpapi=True


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


# ─── Scholar sub-libraries ───────────────────────────────────────────────────

class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=500)


class AddPapersToLibraryRequest(BaseModel):
    papers: List[Dict[str, Any]]  # each item: { content?, score?, metadata: { title, authors?, year?, doi?, pdf_url?, url?, source?, annas_md5? } }


def _normalize_web_hit_for_scholar(hit: Dict[str, Any], canonical_source: str) -> Dict[str, Any]:
    """Map unified_web_search hit to Scholar API response shape (content, score, metadata)."""
    meta = (hit.get("metadata") or {}).copy()
    meta.setdefault("source", canonical_source)
    return {
        "content": hit.get("content") or "",
        "score": hit.get("score") or 0.0,
        "metadata": meta,
    }


@router.post("/search")
async def scholar_search(req: ScholarSearchRequest):
    """Unified search; Google Scholar/Google use RAG unified_web_search (SerpAPI + Playwright + serpapi_ratio)."""
    if not getattr(settings, "scholar_downloader", None) or not settings.scholar_downloader.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")
    from src.retrieval.downloader.adapter import get_adapter
    from src.retrieval.scholar_query_optimizer import optimize_scholar_query

    query = req.query
    if req.optimize:
        query = optimize_scholar_query(query, req.source)
        logger.info("scholar search: query optimized for source=%s", req.source)

    # Google Scholar / Google: use RAG unified_web_search (SerpAPI + Playwright + serpapi_ratio)
    if req.source == "google_scholar":
        from src.retrieval.unified_web_search import unified_web_searcher

        source_configs: Dict[str, Any] = {
            "scholar": {
                "topK": req.limit,
                "useSerpapi": req.use_serpapi,
            }
        }
        serpapi_ratio = req.serpapi_ratio if req.use_serpapi else None
        if req.use_serpapi and serpapi_ratio is None:
            serpapi_ratio = 0.5
        raw_hits = await unified_web_searcher.search(
            query,
            providers=["scholar"],
            source_configs=source_configs,
            max_results_per_provider=req.limit,
            year_start=req.year_start,
            year_end=req.year_end,
            serpapi_ratio=serpapi_ratio,
            use_content_fetcher="off",
        )
        results = [_normalize_web_hit_for_scholar(h, "google_scholar") for h in raw_hits]
        return {"results": results, "count": len(results)}
    if req.source == "google":
        from src.retrieval.unified_web_search import unified_web_searcher

        source_configs = {
            "google": {
                "topK": req.limit,
                "useSerpapi": req.use_serpapi,
            }
        }
        serpapi_ratio = req.serpapi_ratio if req.use_serpapi else None
        if req.use_serpapi and serpapi_ratio is None:
            serpapi_ratio = 0.5
        raw_hits = await unified_web_searcher.search(
            query,
            providers=["google"],
            source_configs=source_configs,
            max_results_per_provider=req.limit,
            serpapi_ratio=serpapi_ratio,
            use_content_fetcher="off",
        )
        results = [_normalize_web_hit_for_scholar(h, "google") for h in raw_hits]
        return {"results": results, "count": len(results)}

    # Semantic Scholar relevance: relevance + snippet 并发（不做 bulk 托底，用户若需要 bulk 请选 semantic_bulk）
    if req.source == "semantic" or req.source == "semantic_relevance":
        import asyncio as _asyncio
        from src.retrieval.semantic_scholar import semantic_scholar_searcher

        relevance_res, snippet_res = await _asyncio.gather(
            semantic_scholar_searcher.search(
                query, limit=req.limit,
                year_start=req.year_start, year_end=req.year_end,
            ),
            semantic_scholar_searcher.search_snippets(
                query, limit=req.limit,
                year_start=req.year_start, year_end=req.year_end,
            ),
            return_exceptions=True,
        )
        raw_hits: List[Dict[str, Any]] = []
        if isinstance(relevance_res, list):
            raw_hits.extend(relevance_res)
        else:
            logger.warning("semantic relevance search failed: %s", relevance_res)
        if isinstance(snippet_res, list):
            raw_hits.extend(snippet_res)
        else:
            logger.warning("semantic snippet search failed: %s", snippet_res)
        results = [_normalize_web_hit_for_scholar(h, "semantic") for h in raw_hits]
        return {"results": results, "count": len(results)}

    # NCBI PubMed (年份过滤) via unified_web_searcher
    if req.source == "ncbi":
        from src.retrieval.unified_web_search import unified_web_searcher

        logger.info("scholar search ncbi: year_start=%s year_end=%s", req.year_start, req.year_end)
        raw_hits = await unified_web_searcher.search(
            query,
            providers=["ncbi"],
            source_configs={"ncbi": {"topK": req.limit}},
            max_results_per_provider=req.limit,
            year_start=req.year_start,
            year_end=req.year_end,
            use_content_fetcher="off",
        )
        results = [_normalize_web_hit_for_scholar(h, "ncbi") for h in raw_hits]
        return {"results": results, "count": len(results)}

    # Semantic Scholar bulk (布尔/关键词，年份过滤) — bulk API 无 snippet，直接走 adapter
    adapter = get_adapter()
    if req.source == "semantic_bulk":
        results = await adapter.search_semantic_scholar_bulk(
            query=query, limit=req.limit,
            year_start=req.year_start, year_end=req.year_end,
        )
    elif req.source == "annas_archive":
        results = await adapter.search_annas_archive(query=query, limit=req.limit)
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
                    logger.warning(
                        "文献下载/入库失败: %s - %s",
                        paper.title,
                        result.get("message", "未知错误"),
                    )
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


# ─── Scholar sub-libraries ───────────────────────────────────────────────────

@router.get("/libraries")
def list_scholar_libraries():
    """List all scholar libraries with paper count."""
    with Session(get_engine()) as session:
        libs = session.exec(select(ScholarLibrary).order_by(ScholarLibrary.created_at)).all()
        # Count papers per library in one go
        all_papers = session.exec(select(ScholarLibraryPaper.library_id)).all()
        counts = Counter(all_papers)
        return [
            {
                "id": lib.id,
                "name": lib.name,
                "description": lib.description or "",
                "paper_count": counts.get(lib.id, 0),
                "created_at": lib.created_at,
                "updated_at": lib.updated_at,
            }
            for lib in libs
        ]


@router.post("/libraries")
def create_scholar_library(body: CreateLibraryRequest):
    """Create a new scholar library."""
    with Session(get_engine()) as session:
        existing = session.exec(select(ScholarLibrary).where(ScholarLibrary.name == body.name)).first()
        if existing:
            raise HTTPException(status_code=400, detail="库名称已存在")
        lib = ScholarLibrary(name=body.name.strip(), description=(body.description or "").strip())
        session.add(lib)
        session.commit()
        session.refresh(lib)
        return {"id": lib.id, "name": lib.name, "description": lib.description, "created_at": lib.created_at}


@router.delete("/libraries/{lib_id:int}")
def delete_scholar_library(lib_id: int):
    """Delete a scholar library and all its papers."""
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib:
            raise HTTPException(status_code=404, detail="库不存在")
        session.delete(lib)
        session.commit()
        return {"ok": True}


def _paper_from_search_item(item: Dict[str, Any]) -> ScholarLibraryPaper:
    meta = item.get("metadata") or {}
    authors = meta.get("authors") or []
    return ScholarLibraryPaper(
        title=(meta.get("title") or "").strip() or "(无标题)",
        authors=json.dumps(authors) if isinstance(authors, list) else "[]",
        year=meta.get("year") if isinstance(meta.get("year"), (int, type(None))) else None,
        doi=(meta.get("doi") or "") or "",
        pdf_url=(meta.get("pdf_url") or "") or "",
        url=(meta.get("url") or "") or "",
        source=(meta.get("source") or "") or "",
        score=float(item.get("score") or 0),
        annas_md5=(meta.get("annas_md5") or "") or "",
    )


@router.get("/libraries/{lib_id:int}/papers")
def list_scholar_library_papers(lib_id: int):
    """List all papers in a scholar library."""
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
        return [
            {
                "id": p.id,
                "library_id": p.library_id,
                "title": p.title,
                "authors": p.get_authors(),
                "year": p.year,
                "doi": p.doi or None,
                "pdf_url": p.pdf_url or None,
                "url": p.url or None,
                "source": p.source,
                "score": p.score,
                "annas_md5": p.annas_md5 or None,
                "added_at": p.added_at,
            }
            for p in papers
        ]


@router.post("/libraries/{lib_id:int}/papers")
def add_papers_to_scholar_library(lib_id: int, body: AddPapersToLibraryRequest):
    """Add search results to a library. Dedup by DOI then by title."""
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib:
            raise HTTPException(status_code=404, detail="库不存在")
        existing = session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
        existing_dois = {p.doi for p in existing if p.doi}
        existing_titles = {p.title.strip().lower() for p in existing}
        added = 0
        for item in body.papers:
            paper = _paper_from_search_item(item)
            paper.library_id = lib_id
            if paper.doi and paper.doi in existing_dois:
                continue
            if not paper.doi and paper.title.strip().lower() in existing_titles:
                continue
            session.add(paper)
            session.flush()
            if paper.doi:
                existing_dois.add(paper.doi)
            existing_titles.add(paper.title.strip().lower())
            added += 1
        session.commit()
        return {"added": added, "total_requested": len(body.papers)}


@router.delete("/libraries/{lib_id:int}/papers/{paper_id:int}")
def remove_paper_from_scholar_library(lib_id: int, paper_id: int):
    """Remove one paper from a scholar library."""
    with Session(get_engine()) as session:
        paper = session.get(ScholarLibraryPaper, paper_id)
        if not paper or paper.library_id != lib_id:
            raise HTTPException(status_code=404, detail="文献不存在")
        session.delete(paper)
        session.commit()
        return {"ok": True}


@router.get("/health")
async def scholar_health():
    from src.retrieval.downloader.adapter import is_adapter_ready

    cfg = getattr(settings, "scholar_downloader", None)
    return {
        "enabled": bool(cfg and cfg.enabled),
        "adapter_ready": is_adapter_ready(),
        "download_dir": getattr(cfg, "download_dir", "") if cfg else "",
    }
