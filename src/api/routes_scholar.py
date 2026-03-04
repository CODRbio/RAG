"""
Scholar API: search + download + ingest.
Prefix: /scholar

Google Scholar / Google search use RAG unified_web_search (SerpAPI + Playwright + serpapi_ratio).
"""

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from config.settings import settings
from src.api.routes_auth import get_current_user_id
from src.db.engine import get_engine
from src.db.models import ScholarLibrary, ScholarLibraryPaper
from src.utils.path_manager import PathManager
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
    library_id: Optional[int] = None  # when set and library has folder_path, download to folder_path/pdfs
    auto_ingest: bool = False  # when False (default), only download; when True, download then ingest per paper


# ─── Scholar sub-libraries ───────────────────────────────────────────────────

# In-memory temporary libraries (24h TTL); key = negative lib_id
_temp_store: Dict[int, Dict[str, Any]] = {}
_temp_id_counter = -1
_temp_paper_id_counter = -1
TEMP_TTL_SECONDS = 86400  # 24h


def _is_temp_library(lib_id: int) -> bool:
    return lib_id < 0


def _purge_expired_temp_libraries() -> None:
    """Remove temp libraries older than TEMP_TTL_SECONDS."""
    now = time.time()
    expired = [k for k, v in _temp_store.items() if (now - v.get("created_at_ts", 0)) > TEMP_TTL_SECONDS]
    for k in expired:
        _temp_store.pop(k, None)


def _temp_library_response(entry: Dict[str, Any]) -> Dict[str, Any]:
    meta = entry["meta"]
    papers = entry.get("papers") or []
    return {
        "id": meta["id"],
        "name": meta["name"],
        "description": meta.get("description") or "",
        "paper_count": len(papers),
        "created_at": meta["created_at"],
        "updated_at": meta["updated_at"],
        "is_temporary": True,
        "folder_path": None,
    }


class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=500)
    folder_path: Optional[str] = Field(default=None, max_length=1024)
    is_temporary: bool = False


class AddPapersToLibraryRequest(BaseModel):
    papers: List[Dict[str, Any]]  # each item: { content?, score?, metadata: { title, authors?, year?, doi?, pdf_url?, url?, source?, annas_md5? } }


class EnrichDoiRequest(BaseModel):
    """Request body for POST /scholar/enrich-doi: current search results to re-enrich with Crossref."""
    results: List[Dict[str, Any]] = Field(default_factory=list)


def _normalize_web_hit_for_scholar(hit: Dict[str, Any], canonical_source: str) -> Dict[str, Any]:
    """Map unified_web_search hit to Scholar API response shape (content, score, metadata)."""
    meta = (hit.get("metadata") or {}).copy()
    meta.setdefault("source", canonical_source)
    return {
        "content": hit.get("content") or "",
        "score": hit.get("score") or 0.0,
        "metadata": meta,
    }


def _enrich_scholar_results_with_crossref(results: List[Dict[str, Any]]) -> None:
    """In-place: supplement DOI for items missing it via Crossref (title lookup). Default after each search."""
    if not results:
        return
    try:
        from src.retrieval.dedup import _enrich_web_hits_missing_doi
        _enrich_web_hits_missing_doi(results, set())
    except Exception as e:
        logger.debug("Crossref enrich for scholar results failed (non-fatal): %s", e)


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
        _enrich_scholar_results_with_crossref(results)
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
        _enrich_scholar_results_with_crossref(results)
        return {"results": results, "count": len(results)}

    # Semantic Scholar relevance only: /paper/search, no snippet, no fallback
    if req.source == "semantic" or req.source == "semantic_relevance":
        from src.retrieval.semantic_scholar import semantic_scholar_searcher

        relevance_res = await semantic_scholar_searcher.search(
            query, limit=req.limit,
            year_start=req.year_start, year_end=req.year_end,
        )
        raw_hits: List[Dict[str, Any]] = (
            relevance_res if isinstance(relevance_res, list) else []
        )
        if not isinstance(relevance_res, list):
            logger.warning("semantic relevance search failed: %s", relevance_res)
        results = [_normalize_web_hit_for_scholar(h, "semantic") for h in raw_hits]
        _enrich_scholar_results_with_crossref(results)
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
        _enrich_scholar_results_with_crossref(results)
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

    _enrich_scholar_results_with_crossref(results)
    return {"results": results, "count": len(results)}


@router.post("/download")
async def scholar_download(req: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Download one paper PDF (download only; no vectorization/ingest by default).
    If auto_ingest=True, runs download then ingest in background and returns task_id.
    Otherwise (default) runs download only and returns result.
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


def _resolve_library_download_dir(library_id: Optional[int]) -> Optional[str]:
    """If library_id is a permanent library with folder_path, return folder_path/pdfs else None."""
    if library_id is None or _is_temp_library(library_id):
        return None
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, library_id)
        if not lib or not getattr(lib, "folder_path", None):
            return None
        return str(Path(lib.folder_path) / "pdfs")


def _resolve_permanent_library_folder(user_id: str, name: str) -> Path:
    """Resolve absolute path for a permanent library (base dir containing pdfs/)."""
    return PathManager.get_user_library_path(user_id, name)


@router.post("/download/batch")
async def scholar_batch_download(
    req: BatchDownloadRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """Batch download; when auto_ingest=True also runs ingest per paper. Default: download only."""
    cfg = getattr(settings, "scholar_downloader", None)
    if not cfg or not cfg.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    batch_download_dir = _resolve_library_download_dir(req.library_id)

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
            "auto_ingest": req.auto_ingest,
        },
    )
    q.set_state(parent)

    async def _batch_job():
        from src.retrieval.downloader.adapter import get_adapter

        sem = asyncio.Semaphore(req.max_concurrent)
        completed = 0
        failed = 0
        progress_lock = asyncio.Lock()
        adapter = get_adapter()

        async def _one(paper: DownloadRequest):
            nonlocal completed, failed
            if req.auto_ingest:
                sub_id = f"{task_id}_{uuid.uuid4().hex[:4]}"
                result = await process_download_and_ingest(
                    task_id=sub_id,
                    paper_info=paper.model_dump(),
                    collection=req.collection,
                    download_dir=batch_download_dir,
                )
            else:
                result = await adapter.download_paper(
                    title=paper.title,
                    doi=paper.doi,
                    pdf_url=paper.pdf_url,
                    annas_md5=paper.annas_md5,
                    authors=paper.authors,
                    year=paper.year,
                    download_dir=batch_download_dir,
                )
                result = {**result, "ingest_triggered": False}
            async with progress_lock:
                if result.get("ingest_triggered") or result.get("success"):
                    completed += 1
                else:
                    failed += 1
                    logger.warning(
                        "文献下载失败: %s - %s",
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
def list_scholar_libraries(user_id: str = Depends(get_current_user_id)):
    """List all scholar libraries (DB + in-memory temp) with paper count; DB libs filtered by user_id."""
    _purge_expired_temp_libraries()
    result: List[Dict[str, Any]] = []
    with Session(get_engine()) as session:
        libs = session.exec(
            select(ScholarLibrary).where(ScholarLibrary.user_id == user_id).order_by(ScholarLibrary.created_at)
        ).all()
        all_papers = session.exec(select(ScholarLibraryPaper.library_id)).all()
        counts = Counter(all_papers)
        for lib in libs:
            result.append({
                "id": lib.id,
                "name": lib.name,
                "description": lib.description or "",
                "paper_count": counts.get(lib.id, 0),
                "created_at": lib.created_at,
                "updated_at": lib.updated_at,
                "is_temporary": False,
                "folder_path": getattr(lib, "folder_path", None) or None,
            })
    for lib_id, entry in list(_temp_store.items()):
        result.append(_temp_library_response(entry))
    return result


@router.post("/libraries")
def create_scholar_library(
    body: CreateLibraryRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new scholar library (temporary in-memory or permanent; folder_path is optional, computed from user_id + name when omitted)."""
    name = body.name.strip()
    description = (body.description or "").strip()
    if body.is_temporary:
        global _temp_id_counter, _temp_paper_id_counter
        _temp_id_counter -= 1
        lib_id = _temp_id_counter
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        _temp_store[lib_id] = {
            "meta": {
                "id": lib_id,
                "name": name,
                "description": description,
                "created_at": now_iso,
                "updated_at": now_iso,
            },
            "papers": [],
            "created_at_ts": time.time(),
        }
        return {
            "id": lib_id,
            "name": name,
            "description": description,
            "created_at": now_iso,
            "is_temporary": True,
            "folder_path": None,
        }
    # Permanent: use PathManager when folder_path not provided
    if (body.folder_path or "").strip():
        abspath = Path((body.folder_path or "").strip())
        if not abspath.is_absolute():
            abspath = Path(os.getcwd()) / abspath
    else:
        abspath = _resolve_permanent_library_folder(user_id, name)
    pdfs_path = abspath / "pdfs"
    try:
        abspath.mkdir(parents=True, exist_ok=True)
        pdfs_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("create_scholar_library: 无法创建目录 path=%s err=%s", abspath, e)
        raise HTTPException(status_code=400, detail=f"无法创建目录: {e}")
    with Session(get_engine()) as session:
        existing = session.exec(
            select(ScholarLibrary).where(ScholarLibrary.user_id == user_id, ScholarLibrary.name == name)
        ).first()
        if existing:
            logger.info("create_scholar_library: 库名称已存在 user_id=%s name=%s", user_id, name)
            raise HTTPException(status_code=400, detail="库名称已存在")
        lib = ScholarLibrary(
            user_id=user_id,
            name=name,
            description=description,
            folder_path=str(abspath.resolve()),
        )
        session.add(lib)
        session.commit()
        session.refresh(lib)
        return {
            "id": lib.id,
            "name": lib.name,
            "description": lib.description or "",
            "created_at": lib.created_at,
            "is_temporary": False,
            "folder_path": lib.folder_path,
        }


@router.delete("/libraries/{lib_id:int}")
def delete_scholar_library(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """Delete a scholar library and all its papers (temp or permanent)."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        _temp_store.pop(lib_id, None)
        return {"ok": True}
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        session.delete(lib)
        session.commit()
        return {"ok": True}


def _scholar_source_priority(source: str) -> int:
    """Lower = better. Same DOI: prefer google_scholar > google > semantic > ncbi."""
    s = (source or "").strip().lower()
    if s in ("google_scholar", "scholar"):
        return 0
    if s == "google":
        return 1
    if s in ("semantic", "semantic_relevance", "semantic_bulk"):
        return 2
    if s == "ncbi":
        return 3
    return 4


def _dedupe_papers_by_doi_keep_best_source(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each DOI keep one paper: the one with best source priority (google_scholar > google > semantic > ncbi). No-DOI items are kept as-is."""
    if not items:
        return items
    no_doi: List[Dict[str, Any]] = []
    by_doi: Dict[str, Dict[str, Any]] = {}
    for item in items:
        meta = item.get("metadata") or {}
        doi = (meta.get("doi") or "").strip()
        pri = _scholar_source_priority(meta.get("source") or "")
        if not doi:
            no_doi.append(item)
        elif doi not in by_doi or pri < _scholar_source_priority((by_doi[doi].get("metadata") or {}).get("source") or ""):
            by_doi[doi] = item
    return list(by_doi.values()) + no_doi


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


def _temp_paper_from_search_item(item: Dict[str, Any], lib_id: int) -> Dict[str, Any]:
    """Build a temp-library paper dict with a negative paper id."""
    global _temp_paper_id_counter
    _temp_paper_id_counter -= 1
    meta = item.get("metadata") or {}
    authors = meta.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    return {
        "id": _temp_paper_id_counter,
        "library_id": lib_id,
        "title": (meta.get("title") or "").strip() or "(无标题)",
        "authors": authors,
        "year": meta.get("year") if isinstance(meta.get("year"), (int, type(None))) else None,
        "doi": (meta.get("doi") or "") or "",
        "pdf_url": (meta.get("pdf_url") or "") or "",
        "url": (meta.get("url") or "") or "",
        "source": (meta.get("source") or "") or "",
        "score": float(item.get("score") or 0),
        "annas_md5": (meta.get("annas_md5") or "") or "",
        "added_at": now_iso,
        "downloaded_at": None,
    }


@router.get("/libraries/{lib_id:int}/papers")
def list_scholar_library_papers(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """List all papers in a scholar library (temp or permanent)."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        return list(_temp_store[lib_id].get("papers") or [])
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
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
                "source": p.source or "",
                "score": p.score,
                "annas_md5": p.annas_md5 or None,
                "added_at": p.added_at,
                "downloaded_at": getattr(p, "downloaded_at", None),
            }
            for p in papers
        ]


@router.post("/libraries/{lib_id:int}/papers")
def add_papers_to_scholar_library(lib_id: int, body: AddPapersToLibraryRequest, user_id: str = Depends(get_current_user_id)):
    """Add search results to a library. Dedup by DOI (keep best source: google_scholar > google > semantic > ncbi), then by title."""
    papers_to_add = _dedupe_papers_by_doi_keep_best_source(body.papers)
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        entry = _temp_store[lib_id]
        papers_list = entry.setdefault("papers", [])
        existing_dois = {p.get("doi") for p in papers_list if p.get("doi")}
        existing_by_doi = {p.get("doi"): p for p in papers_list if p.get("doi")}
        existing_titles = {str(p.get("title") or "").strip().lower() for p in papers_list}
        added = 0
        for item in papers_to_add:
            paper_d = _temp_paper_from_search_item(item, lib_id)
            if paper_d["doi"] and paper_d["doi"] in existing_dois:
                existing = existing_by_doi.get(paper_d["doi"])
                if existing and _scholar_source_priority(paper_d["source"]) < _scholar_source_priority(existing.get("source") or ""):
                    idx = next(i for i, p in enumerate(papers_list) if p.get("doi") == paper_d["doi"])
                    papers_list[idx] = {**paper_d, "id": papers_list[idx]["id"]}
                    added += 1
                continue
            if not paper_d["doi"] and paper_d["title"].strip().lower() in existing_titles:
                continue
            papers_list.append(paper_d)
            if paper_d["doi"]:
                existing_dois.add(paper_d["doi"])
                existing_by_doi[paper_d["doi"]] = paper_d
            existing_titles.add(paper_d["title"].strip().lower())
            added += 1
        entry["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return {"added": added, "total_requested": len(body.papers)}
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        existing = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        existing_dois = {p.doi for p in existing if p.doi}
        existing_by_doi = {p.doi: p for p in existing if p.doi}
        existing_titles = {p.title.strip().lower() for p in existing}
        added = 0
        for item in papers_to_add:
            paper = _paper_from_search_item(item)
            paper.library_id = lib_id
            if paper.doi and paper.doi in existing_dois:
                existing_row = existing_by_doi.get(paper.doi)
                if existing_row and _scholar_source_priority(paper.source) < _scholar_source_priority(existing_row.source or ""):
                    existing_row.title = paper.title
                    existing_row.authors = paper.authors
                    existing_row.year = paper.year
                    existing_row.doi = paper.doi
                    existing_row.pdf_url = paper.pdf_url or ""
                    existing_row.url = paper.url or ""
                    existing_row.source = paper.source
                    existing_row.score = paper.score
                    existing_row.annas_md5 = paper.annas_md5 or ""
                    session.add(existing_row)
                    added += 1
                continue
            if not paper.doi and paper.title.strip().lower() in existing_titles:
                continue
            session.add(paper)
            session.flush()
            if paper.doi:
                existing_dois.add(paper.doi)
                existing_by_doi[paper.doi] = paper
            existing_titles.add(paper.title.strip().lower())
            added += 1
        session.commit()
        return {"added": added, "total_requested": len(body.papers)}


@router.delete("/libraries/{lib_id:int}/papers/{paper_id:int}")
def remove_paper_from_scholar_library(lib_id: int, paper_id: int, user_id: str = Depends(get_current_user_id)):
    """Remove one paper from a scholar library."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        papers_list = _temp_store[lib_id].get("papers") or []
        for i, p in enumerate(papers_list):
            if p.get("id") == paper_id:
                papers_list.pop(i)
                return {"ok": True}
        raise HTTPException(status_code=404, detail="文献不存在")
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
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
