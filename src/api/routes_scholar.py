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
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from config.settings import settings
from src.api.routes_auth import get_current_user_id
from src.db.engine import get_engine
from src.db.models import ScholarLibrary, ScholarLibraryPaper
from src.log import get_logger
from src.services.collection_library_binding_service import (
    ensure_collection_binding,
    resolve_bound_library_for_collection,
)
from src.tasks.dispatcher import process_download_and_ingest
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState
from src.utils.path_manager import PathManager

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
    url: Optional[str] = None
    annas_md5: Optional[str] = Field(None, pattern=r"^[0-9a-f]{32}$")
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    collection: Optional[str] = None
    auto_ingest: Optional[bool] = None
    library_paper_id: Optional[int] = Field(None, description="ScholarLibraryPaper.id; when set, downloaded_at is written on success")
    library_id: Optional[int] = Field(None, description="When set with library_paper_id, download to library pdfs folder")
    llm_provider: Optional[str] = Field(None, description="LLM provider for downloader assist (e.g. qwen-thinking); uses config default if not set")
    model_override: Optional[str] = Field(None, description="Model override for downloader LLM; uses provider default if not set")
    assist_llm_enabled: Optional[bool] = Field(None, description="显式启停下载辅助 LLM；False 时彻底关闭，不回退到服务端默认")
    show_browser: Optional[bool] = Field(None, description="有头(True)/无头(False)；不传则用配置默认")
    include_academia: bool = Field(False, description="是否允许尝试 Academia.edu 下载（默认跳过，避免卡住）")
    strategy_order: Optional[List[str]] = Field(None, description="Scholar 下载策略顺序（UI 策略项）")


class BatchDownloadRequest(BaseModel):
    papers: List[DownloadRequest]
    collection: Optional[str] = None
    max_concurrent: int = 3
    library_id: Optional[int] = None  # when set and library has folder_path, download to folder_path/pdfs
    auto_ingest: bool = False  # when False (default), only download; when True, download then ingest per paper
    llm_provider: Optional[str] = Field(None, description="LLM provider for downloader assist; applies to all papers in batch")
    model_override: Optional[str] = Field(None, description="Model override for downloader LLM; applies to all papers in batch")
    assist_llm_enabled: Optional[bool] = Field(None, description="显式启停下载辅助 LLM；False 时彻底关闭")
    show_browser: Optional[bool] = Field(None, description="有头(True)/无头(False)；不传则用配置默认")
    include_academia: bool = Field(False, description="是否允许尝试 Academia.edu 下载（默认跳过，避免卡住）")
    strategy_order: Optional[List[str]] = Field(None, description="Scholar 下载策略顺序（UI 策略项）")


class HeadedBrowserWindowState(BaseModel):
    available: bool
    running: bool
    mode: str
    cdp_url: Optional[str] = None
    bounds: Optional[Dict[str, int]] = None


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


class ExtractDoiDedupPapersRequest(BaseModel):
    """Request body for POST /scholar/papers/extract-doi-dedup (client temp library)."""
    papers: List[Dict[str, Any]] = Field(default_factory=list)


class EnrichDoiRequest(BaseModel):
    """Request body for POST /scholar/enrich-doi: current search results to re-enrich with Crossref."""
    results: List[Dict[str, Any]] = Field(default_factory=list)


class LibraryIngestRequest(BaseModel):
    """Request body for POST /scholar/libraries/{lib_id}/ingest."""
    collection: Optional[str] = None
    skip_duplicate_doi: bool = True
    skip_unchanged: bool = True
    auto_download_missing: bool = False
    max_auto_download: int = Field(30, ge=1, le=200)
    llm_provider: Optional[str] = None
    model_override: Optional[str] = None
    assist_llm_enabled: Optional[bool] = None
    show_browser: Optional[bool] = None
    include_academia: bool = False
    strategy_order: Optional[List[str]] = None


class LibraryImportPdfSummary(BaseModel):
    total_files: int
    imported: int
    linked_existing: int
    renamed: int
    skipped_duplicates: int
    invalid_pdf: int
    no_doi: int
    errors: List[str] = Field(default_factory=list)


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


def _normalize_and_enrich_venue(results: List[Dict[str, Any]]) -> None:
    """In-place: clean venue, add normalized_journal_name, batch lookup IF, add impact_factor/jif_quartile/jif_5year/journal_name_matched."""
    if not results:
        return
    try:
        from src.retrieval.venue_utils import extract_clean_venue, normalize_journal_name
        from src.services.impact_factor_service import lookup_many

        venues_to_lookup: List[str] = []
        for r in results:
            meta = r.get("metadata") or {}
            venue = (meta.get("venue") or "").strip()
            if not venue and meta.get("venue_raw"):
                venue = extract_clean_venue(meta.get("venue_raw") or "")
            if venue:
                meta["venue"] = venue
                norm = normalize_journal_name(venue)
                meta["normalized_journal_name"] = norm
                venues_to_lookup.append(venue)
            else:
                meta["normalized_journal_name"] = ""

        if not venues_to_lookup:
            return
        if_map = lookup_many(venues_to_lookup, ensure_current=True)
        for r in results:
            meta = r.get("metadata") or {}
            norm = meta.get("normalized_journal_name") or ""
            venue = meta.get("venue") or ""
            data = None
            if norm and norm in if_map:
                data = if_map[norm]
            elif venue and venue in if_map:
                data = if_map[venue]
            if data:
                meta["impact_factor"] = data.get("impact_factor")
                meta["jif_quartile"] = data.get("jif_quartile") or ""
                meta["jif_5year"] = data.get("jif_5year")
                meta["journal_name_matched"] = data.get("journal_name") or ""
            else:
                meta["impact_factor"] = None
                meta["jif_quartile"] = None
                meta["jif_5year"] = None
                meta["journal_name_matched"] = None
    except Exception as e:
        logger.debug("Impact factor enrich for scholar results failed (non-fatal): %s", e)


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
        _normalize_and_enrich_venue(results)
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
        _normalize_and_enrich_venue(results)
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
        _normalize_and_enrich_venue(results)
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
        _normalize_and_enrich_venue(results)
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
    _normalize_and_enrich_venue(results)
    return {"results": results, "count": len(results)}


@router.post("/download")
async def scholar_download(
    req: DownloadRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
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

    effective_library_id = getattr(req, "library_id", None)
    if effective_library_id is None and (req.collection or "").strip():
        bound = resolve_bound_library_for_collection(user_id, req.collection, auto_create=True)
        if bound is not None and bound.id is not None:
            effective_library_id = int(bound.id)

    download_dir = _resolve_library_download_dir(effective_library_id)

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
            download_dir=download_dir,
            user_id=user_id,
            library_id=effective_library_id,
            library_paper_id=getattr(req, "library_paper_id", None),
            llm_provider=getattr(req, "llm_provider", None),
            model_override=getattr(req, "model_override", None),
            assist_llm_enabled=getattr(req, "assist_llm_enabled", None),
            show_browser=getattr(req, "show_browser", None),
            include_academia=getattr(req, "include_academia", False),
            strategy_order=getattr(req, "strategy_order", None),
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
        url=req.url,
        authors=req.authors,
        year=req.year,
        download_dir=download_dir,
        llm_provider=getattr(req, "llm_provider", None),
        model_override=getattr(req, "model_override", None),
        assist_llm_enabled=getattr(req, "assist_llm_enabled", None),
        show_browser=getattr(req, "show_browser", None),
        include_academia=getattr(req, "include_academia", False),
        strategy_order=getattr(req, "strategy_order", None),
    )
    if result.get("success") and req.library_paper_id is not None:
        from src.tasks.dispatcher import _mark_library_paper_downloaded

        _mark_library_paper_downloaded(req.library_paper_id)
    if effective_library_id is not None:
        result["library_id"] = effective_library_id
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


def _pdf_rename_dedup_library_folder(pdfs_dir: str) -> dict:
    """Scan pdfs in folder: extract DOI, rename to DOI-based stem, remove duplicate PDFs. Returns {renamed, removed, no_doi}."""
    from src.indexing.paper_metadata_store import paper_meta_store
    from src.retrieval.dedup import extract_doi_from_pdf_tiered, normalize_doi
    from src.retrieval.downloader.adapter import _doi_to_paper_id
    from src.retrieval.downloader.utils import is_valid_pdf

    pdfs_path = Path(pdfs_dir)
    seen_dois: Dict[str, Path] = {}
    renamed = removed = no_doi = 0

    for pdf in sorted(pdfs_path.glob("*.pdf")):
        if not is_valid_pdf(str(pdf)):
            continue
        doi, _title = extract_doi_from_pdf_tiered(pdf)
        if not doi:
            no_doi += 1
            continue
        new_stem = _doi_to_paper_id(doi)
        new_path = pdfs_path / f"{new_stem}.pdf"
        ndoi = normalize_doi(doi)
        if not ndoi:
            no_doi += 1
            continue
        if ndoi in seen_dois:
            # Only remove if this is a different file from the one we kept (same path = already kept, do not delete)
            kept = seen_dois[ndoi]
            try:
                if pdf.resolve() != kept.resolve():
                    pdf.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
            continue
        seen_dois[ndoi] = new_path
        if pdf != new_path:
            if new_path.exists():
                try:
                    pdf.unlink(missing_ok=True)
                except OSError:
                    pass
                removed += 1
            else:
                try:
                    pdf.rename(new_path)
                    renamed += 1
                except OSError:
                    continue
        try:
            paper_meta_store.upsert(new_stem, doi=doi, source="pdf_rename_dedup")
        except Exception:
            pass

    return {"renamed": renamed, "removed": removed, "no_doi": no_doi}


def _sync_library_paper_downloaded_at(lib_id: int, pdfs_dir: str) -> int:
    """Set downloaded_at for ScholarLibraryPaper rows where the PDF exists on disk (by DOI → paper_id). Returns count updated."""
    from src.retrieval.downloader.adapter import _doi_to_paper_id
    from src.retrieval.downloader.utils import is_valid_pdf

    pdfs_path = Path(pdfs_dir)
    if not pdfs_path.is_dir():
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    updated = 0
    with Session(get_engine()) as session:
        rows = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        for row in rows:
            doi = (row.doi or "").strip()
            if not doi or getattr(row, "downloaded_at", None):
                continue
            paper_id = _doi_to_paper_id(doi)
            candidate = pdfs_path / f"{paper_id}.pdf"
            if candidate.is_file() and is_valid_pdf(str(candidate)):
                row.downloaded_at = now
                session.add(row)
                updated += 1
        if updated:
            session.commit()
    return updated


@router.post("/download/batch")
async def scholar_batch_download(
    req: BatchDownloadRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Batch download; when auto_ingest=True also runs ingest per paper. Default: download only."""
    cfg = getattr(settings, "scholar_downloader", None)
    if not cfg or not cfg.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    effective_library_id = req.library_id
    if effective_library_id is None and (req.collection or "").strip():
        bound = resolve_bound_library_for_collection(user_id, req.collection, auto_create=True)
        if bound is not None and bound.id is not None:
            effective_library_id = int(bound.id)
    batch_download_dir = _resolve_library_download_dir(effective_library_id)

    task_id = f"batch_dl_{uuid.uuid4().hex[:8]}"
    total = len(req.papers)
    q = get_task_queue()
    parent = TaskState(
        task_id=task_id,
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        started_at=time.time(),
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

        batch_start = time.time()
        heartbeat_stop = asyncio.Event()
        _BATCH_HEARTBEAT_INTERVAL = 5

        async def _batch_heartbeat() -> None:
            while not heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=_BATCH_HEARTBEAT_INTERVAL)
                except asyncio.TimeoutError:
                    pass
                if heartbeat_stop.is_set():
                    break
                s = q.get_state(task_id)
                if s and s.status in (TaskStatus.completed, TaskStatus.error):
                    break
                elapsed = time.time() - batch_start
                payload = s.payload if s else {}
                q.push_event(
                    task_id,
                    "heartbeat",
                    {
                        "completed": payload.get("completed", 0),
                        "failed": payload.get("failed", 0),
                        "total": total,
                        "elapsed_s": round(elapsed, 1),
                    },
                )

        heartbeat_task = asyncio.create_task(_batch_heartbeat())
        sem = asyncio.Semaphore(req.max_concurrent)
        completed = 0
        failed = 0
        progress_lock = asyncio.Lock()
        adapter = get_adapter()

        async def _one(paper: DownloadRequest):
            nonlocal completed, failed
            try:
                include_academia = getattr(req, "include_academia", False)
                if req.auto_ingest:
                    sub_id = f"{task_id}_{uuid.uuid4().hex[:4]}"
                    result = await process_download_and_ingest(
                        task_id=sub_id,
                        paper_info=paper.model_dump(),
                        collection=req.collection,
                        download_dir=batch_download_dir,
                        user_id=user_id,
                        library_id=effective_library_id,
                        library_paper_id=getattr(paper, "library_paper_id", None),
                        llm_provider=getattr(req, "llm_provider", None) or getattr(paper, "llm_provider", None),
                        model_override=getattr(req, "model_override", None) or getattr(paper, "model_override", None),
                        assist_llm_enabled=(
                            getattr(req, "assist_llm_enabled", None)
                            if getattr(req, "assist_llm_enabled", None) is not None
                            else getattr(paper, "assist_llm_enabled", None)
                        ),
                        show_browser=getattr(req, "show_browser", None),
                        include_academia=include_academia,
                        strategy_order=getattr(req, "strategy_order", None) or getattr(paper, "strategy_order", None),
                    )
                else:
                    result = await adapter.download_paper(
                        title=paper.title,
                        doi=paper.doi,
                        pdf_url=paper.pdf_url,
                        url=paper.url,
                        annas_md5=paper.annas_md5,
                        authors=paper.authors,
                        year=paper.year,
                        download_dir=batch_download_dir,
                        llm_provider=getattr(req, "llm_provider", None) or getattr(paper, "llm_provider", None),
                        model_override=getattr(req, "model_override", None) or getattr(paper, "model_override", None),
                        assist_llm_enabled=(
                            getattr(req, "assist_llm_enabled", None)
                            if getattr(req, "assist_llm_enabled", None) is not None
                            else getattr(paper, "assist_llm_enabled", None)
                        ),
                        show_browser=getattr(req, "show_browser", None),
                        include_academia=include_academia,
                        strategy_order=getattr(req, "strategy_order", None) or getattr(paper, "strategy_order", None),
                    )
                    result = {**result, "ingest_triggered": False}
                    # Update downloaded_at for the library paper row when download succeeds
                    if result.get("success") and paper.library_paper_id is not None:
                        from src.tasks.dispatcher import _mark_library_paper_downloaded
                        _mark_library_paper_downloaded(paper.library_paper_id)
            except Exception as e:
                logger.exception("[scholar-batch] paper failed task_id=%s title=%s: %s", task_id, paper.title, e)
                result = {
                    "success": False,
                    "ingest_triggered": False,
                    "message": str(e) or e.__class__.__name__,
                }
            async with progress_lock:
                if result.get("ingest_triggered") or result.get("success"):
                    completed += 1
                else:
                    failed += 1
                    route_diag = result.get("route_diag") if isinstance(result, dict) else None
                    logger.warning(
                        "文献下载失败: %s - %s | route_attempted=%s route_skipped=%s",
                        paper.title,
                        result.get("message", "未知错误"),
                        (route_diag or {}).get("attempted", []),
                        list(((route_diag or {}).get("skipped", {}) or {}).keys()),
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

        try:
            await asyncio.gather(*[sem_wrap(sem, _one, p) for p in req.papers])
            if effective_library_id is not None and not _is_temp_library(effective_library_id) and batch_download_dir:
                try:
                    _sync_library_paper_downloaded_at(effective_library_id, batch_download_dir)
                except Exception as e:
                    logger.warning("batch post-sync downloaded_at failed: %s", e)

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
                    state.error_message = None
                q.set_state(state)
                q.push_event(
                    task_id,
                    "done",
                    {"completed": completed, "failed": failed, "total": total},
                )
        except Exception as e:
            logger.exception("[scholar-batch] task_id=%s failed: %s", task_id, e)
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.error
                state.finished_at = time.time()
                state.error_message = str(e) or e.__class__.__name__
                state.payload["completed"] = completed
                state.payload["failed"] = failed
                state.payload["total"] = total
                q.set_state(state)
                q.push_event(
                    task_id,
                    "error",
                    {
                        "message": state.error_message,
                        "completed": completed,
                        "failed": failed,
                        "total": total,
                    },
                )
        finally:
            heartbeat_stop.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def sem_wrap(sem, coro_fn, paper):
        async with sem:
            return await coro_fn(paper)

    # Schedule on the app's event loop (same as context pool). BackgroundTasks would run in
    # thread pool and expects a callable, not a coroutine.
    batch_task = asyncio.create_task(_batch_job(), name=f"scholar-batch:{task_id}")

    def _log_batch_task_result(task: asyncio.Task) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.warning("[scholar-batch] task_id=%s was cancelled", task_id)
            return
        if exc is not None:
            logger.exception("[scholar-batch] task_id=%s escaped task wrapper: %s", task_id, exc)

    batch_task.add_done_callback(_log_batch_task_result)
    return {
        "status": "submitted",
        "task_id": task_id,
        "total": total,
        "message": f"批量下载任务已投递（{total} 篇）",
    }


_SCHOLAR_SSE_HEARTBEAT_INTERVAL = 5  # seconds: keep SSE connection alive during long downloads


def _scholar_event_stream(task_id: str, q, after_id: str = "-"):
    """SSE event stream for scholar tasks: read events from Redis, emit heartbeat when idle."""
    last_id = after_id
    last_sent_at = time.monotonic()
    yield ": stream-init\n\n"
    last_sent_at = time.monotonic()
    while True:
        events = q.read_events(task_id, after_id=last_id)
        for ev in events:
            last_id = ev.get("id", last_id)
            typ = ev.get("type", "message")
            data = ev.get("data", {})
            yield f"id: {last_id}\nevent: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            last_sent_at = time.monotonic()
            if typ in ("done", "error", "cancelled", "timeout"):
                return

        state = q.get_state(task_id)
        if state and state.is_terminal():
            events = q.read_events(task_id, after_id=last_id)
            for ev in events:
                last_id = ev.get("id", last_id)
                typ = ev.get("type", "message")
                data = ev.get("data", {})
                yield f"id: {last_id}\nevent: {typ}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
                if typ in ("done", "error", "cancelled", "timeout"):
                    return
            yield f"event: {state.status.value}\ndata: {json.dumps({'status': state.status.value}, ensure_ascii=False)}\n\n"
            return

        if time.monotonic() - last_sent_at >= _SCHOLAR_SSE_HEARTBEAT_INTERVAL:
            yield ": heartbeat\n\n"
            last_sent_at = time.monotonic()

        time.sleep(0.3)


@router.get("/task/{task_id}/stream")
def scholar_task_stream(task_id: str, request: Request):
    """SSE stream for download+ingest task; supports Last-Event-ID / after_id for resume."""
    try:
        q = get_task_queue()
    except Exception as e:
        logger.warning("scholar stream get_task_queue failed: %s", e)
        raise HTTPException(status_code=503, detail="Task store unavailable")
    state = q.get_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if state.kind != TaskKind.scholar:
        raise HTTPException(status_code=400, detail="Not a scholar task")
    after_id = (
        request.headers.get("Last-Event-ID")
        or request.query_params.get("after_id")
        or "-"
    )
    return StreamingResponse(
        _scholar_event_stream(task_id, q, after_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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


@router.get("/libraries/by-collection/{collection_name:path}")
def get_scholar_library_by_collection(
    collection_name: str,
    auto_create: bool = False,
    user_id: str = Depends(get_current_user_id),
):
    """Resolve scholar library bound to collection."""
    collection = (collection_name or "").strip()
    if not collection:
        raise HTTPException(status_code=400, detail="集合名称不能为空")

    lib = resolve_bound_library_for_collection(user_id, collection, auto_create=bool(auto_create))
    if lib is None and auto_create:
        with Session(get_engine()) as session:
            _binding, lib, _binding_created, _library_created = ensure_collection_binding(session, user_id, collection)
            session.commit()
    if lib is None:
        return {"collection": collection, "bound": False, "library": None}
    return {
        "collection": collection,
        "bound": True,
        "library": {
            "id": lib.id,
            "name": lib.name,
            "description": lib.description or "",
            "paper_count": 0,
            "created_at": lib.created_at,
            "updated_at": lib.updated_at,
            "is_temporary": False,
            "folder_path": getattr(lib, "folder_path", None) or None,
        },
    }


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


def _is_academia_pdf_url(url: Optional[str]) -> bool:
    """True if pdf_url points to Academia.edu (deprioritize when merging by DOI)."""
    return bool(url and "academia.edu" in (url or "").lower())


def _dedupe_papers_by_doi_keep_best_source(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each DOI keep one paper: best source priority; when equal, prefer non-Academia pdf_url (Academia often paywalled). No-DOI items kept as-is."""
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
            continue
        existing = by_doi.get(doi)
        if existing is None:
            by_doi[doi] = item
            continue
        existing_meta = existing.get("metadata") or {}
        existing_pri = _scholar_source_priority(existing_meta.get("source") or "")
        item_pdf = (meta.get("pdf_url") or "").strip()
        existing_pdf = (existing_meta.get("pdf_url") or "").strip()
        # Prefer higher source priority; when equal, prefer non-Academia pdf_url
        take_item = (
            pri < existing_pri
            or (
                pri == existing_pri
                and _is_academia_pdf_url(existing_pdf)
                and not _is_academia_pdf_url(item_pdf)
            )
        )
        if take_item:
            by_doi[doi] = item
    return list(by_doi.values()) + no_doi


def _extract_doi_and_dedup_library_papers(
    papers: List[Dict[str, Any]], use_crossref: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    In-place fill missing DOIs (from url/title, then optionally Crossref), normalize all DOIs,
    then dedupe by normalized DOI keeping one per DOI by source priority.
    papers: list of library-paper dicts (flat: id, title, doi, url, source, ...).
    Returns (deduped_list, {"extracted_count": int, "removed_count": int}).
    """
    from src.retrieval.dedup import (
        _crossref_lookup_by_title,
        _extract_doi_from_text,
        normalize_doi,
    )

    extracted_count = 0
    for p in papers:
        doi_raw = (p.get("doi") or "").strip()
        ndoi = normalize_doi(doi_raw)
        if ndoi:
            p["doi"] = ndoi
            continue
        url = (p.get("url") or "").strip()
        title = (p.get("title") or "").strip()
        ndoi = _extract_doi_from_text(url or title)
        if ndoi:
            p["doi"] = ndoi
            extracted_count += 1
            continue
        if use_crossref and title:
            try:
                cr = _crossref_lookup_by_title(title)
                if cr and cr.get("doi"):
                    p["doi"] = cr["doi"]
                    extracted_count += 1
            except Exception as e:
                logger.debug("Crossref lookup for library paper failed: %s", e)

    by_ndoi: Dict[str, Dict[str, Any]] = {}
    no_doi_list: List[Dict[str, Any]] = []
    for p in papers:
        ndoi = normalize_doi(p.get("doi") or "")
        if not ndoi:
            no_doi_list.append(p)
        else:
            src = (p.get("source") or "").strip()
            existing = by_ndoi.get(ndoi)
            if existing is None:
                by_ndoi[ndoi] = p
            else:
                existing_pri = _scholar_source_priority((existing.get("source") or "").strip())
                pri = _scholar_source_priority(src)
                take_p = (
                    pri < existing_pri
                    or (
                        pri == existing_pri
                        and _is_academia_pdf_url(existing.get("pdf_url"))
                        and not _is_academia_pdf_url(p.get("pdf_url"))
                    )
                )
                if take_p:
                    by_ndoi[ndoi] = p
    deduped = list(by_ndoi.values()) + no_doi_list
    removed_count = len(papers) - len(deduped)
    return deduped, {"extracted_count": extracted_count, "removed_count": removed_count}


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
        venue=(meta.get("venue") or "") or "",
        normalized_journal_name=(meta.get("normalized_journal_name") or "") or "",
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
    out: Dict[str, Any] = {
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
        "venue": (meta.get("venue") or "") or "",
        "normalized_journal_name": (meta.get("normalized_journal_name") or "") or "",
    }
    if meta.get("impact_factor") is not None:
        out["impact_factor"] = meta["impact_factor"]
    if meta.get("jif_quartile"):
        out["jif_quartile"] = meta["jif_quartile"]
    if meta.get("jif_5year") is not None:
        out["jif_5year"] = meta["jif_5year"]
    return out


def _doi_to_paper_id_for_api(doi: str) -> str:
    """DOI to paper_id stem for API (delegate to adapter)."""
    from src.retrieval.downloader.adapter import _doi_to_paper_id
    return _doi_to_paper_id(doi)


def _attach_impact_factor_metadata(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """In-place: add impact_factor, jif_quartile, jif_5year to each paper that has venue/normalized_journal_name."""
    if not papers:
        return papers
    try:
        from src.services.impact_factor_service import lookup_many

        venues = []
        for p in papers:
            v = (p.get("venue") or "").strip() or (p.get("normalized_journal_name") or "").strip()
            if v:
                venues.append(v)
        if not venues:
            return papers
        if_map = lookup_many(venues, ensure_current=True)
        for p in papers:
            v = (p.get("venue") or "").strip()
            norm = (p.get("normalized_journal_name") or "").strip() or v
            data = if_map.get(norm) or (if_map.get(v) if v else None)
            if data:
                p["impact_factor"] = data.get("impact_factor")
                p["jif_quartile"] = data.get("jif_quartile") or ""
                p["jif_5year"] = data.get("jif_5year")
            else:
                p.setdefault("impact_factor", None)
                p.setdefault("jif_quartile", None)
                p.setdefault("jif_5year", None)
    except Exception as e:
        logger.debug("Attach impact factor to library papers failed (non-fatal): %s", e)
    return papers


def _build_library_ingest_cfg(
    *,
    file_paths: List[str],
    collection_name: str,
    user_id: str,
    skip_duplicate_doi: bool,
    skip_unchanged: bool,
) -> Dict[str, Any]:
    """Build ingest config payload for library -> ingest bridge."""
    return {
        "file_paths": file_paths,
        "collection_name": collection_name,
        "content_hashes": {},
        "enrich_tables": False,
        "enrich_figures": False,
        "actual_skip": True,
        "skip_duplicate_doi": bool(skip_duplicate_doi),
        "skip_unchanged": bool(skip_unchanged),
        "llm_text_provider": None,
        "llm_text_model": None,
        "llm_text_concurrency": None,
        "llm_vision_provider": None,
        "llm_vision_model": None,
        "llm_vision_concurrency": None,
        "user_id": user_id,
    }


def _library_paper_id(doi: Optional[str], title: str, authors: List[str], year: Optional[int]) -> Optional[str]:
    """Return paper_id for a library paper: from DOI if present, else from title/authors/year so download button is enabled when only pdf_url exists."""
    if doi and (doi or "").strip():
        return _doi_to_paper_id_for_api(doi)
    if not (title or (authors and authors[0])):
        return None
    from src.retrieval.downloader.adapter import _normalize_to_paper_id
    return _normalize_to_paper_id(title or "", list(authors) if authors else [], year)


def _build_library_row_index(rows: List[ScholarLibraryPaper]) -> Tuple[Dict[str, ScholarLibraryPaper], Dict[str, ScholarLibraryPaper]]:
    """Build in-memory DOI/title lookup for fast dedup/linking."""
    from src.retrieval.dedup import normalize_doi, normalize_title

    by_doi: Dict[str, ScholarLibraryPaper] = {}
    by_title: Dict[str, ScholarLibraryPaper] = {}
    for row in rows:
        ndoi = normalize_doi((row.doi or "").strip())
        if ndoi and ndoi not in by_doi:
            by_doi[ndoi] = row
        ntitle = normalize_title((row.title or "").strip())
        if ntitle and ntitle not in by_title:
            by_title[ntitle] = row
    return by_doi, by_title


@router.get("/libraries/{lib_id:int}/papers")
def list_scholar_library_papers(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """List all papers in a scholar library (temp or permanent)."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = list(_temp_store[lib_id].get("papers") or [])
        out = []
        for p in papers:
            d = dict(p)
            d["paper_id"] = _library_paper_id(
                (d.get("doi") or "").strip() or None,
                d.get("title") or "",
                d.get("authors") or [],
                d.get("year"),
            )
            d["venue"] = d.get("venue") or None
            d["normalized_journal_name"] = d.get("normalized_journal_name") or None
            d["is_downloaded"] = bool(d.get("downloaded_at"))
            out.append(d)
        return _attach_impact_factor_metadata(out)
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
        out = [
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
                "venue": getattr(p, "venue", None) or None,
                "normalized_journal_name": getattr(p, "normalized_journal_name", None) or None,
                "paper_id": _library_paper_id(
                    (p.doi or "").strip() or None,
                    p.title or "",
                    p.get_authors(),
                    p.year,
                ),
                "is_downloaded": bool(getattr(p, "downloaded_at", None)),
            }
            for p in papers
        ]
        return _attach_impact_factor_metadata(out)


@router.post("/libraries/{lib_id:int}/ingest")
async def ingest_scholar_library(
    lib_id: int,
    body: LibraryIngestRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create one ingest job from a scholar library's PDFs (incremental by DOI/content hash)."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库不支持直接增量建库")

    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        rows = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())

    download_dir = _resolve_library_download_dir(lib_id)
    if not download_dir:
        raise HTTPException(status_code=400, detail="该库未配置可用目录")
    pdfs_dir = Path(download_dir)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    from src.retrieval.downloader.utils import is_valid_pdf

    file_paths: List[str] = []
    missing_rows: List[ScholarLibraryPaper] = []
    for row in rows:
        stem = _library_paper_id(
            (row.doi or "").strip() or None,
            row.title or "",
            row.get_authors(),
            row.year,
        )
        if not stem:
            continue
        candidate = pdfs_dir / f"{stem}.pdf"
        if candidate.is_file() and is_valid_pdf(str(candidate)):
            file_paths.append(str(candidate))
        else:
            missing_rows.append(row)

    downloaded_now = 0
    failed_downloads = 0
    attempted_downloads = 0

    if body.auto_download_missing and missing_rows:
        from src.retrieval.downloader.adapter import get_adapter

        adapter = get_adapter()
        for row in missing_rows[: body.max_auto_download]:
            attempted_downloads += 1
            result = await adapter.download_paper(
                title=row.title,
                doi=(row.doi or "").strip() or None,
                pdf_url=(row.pdf_url or "").strip() or None,
                annas_md5=(row.annas_md5 or "").strip() or None,
                authors=row.get_authors(),
                year=row.year,
                download_dir=download_dir,
                llm_provider=body.llm_provider,
                model_override=body.model_override,
                assist_llm_enabled=body.assist_llm_enabled,
                show_browser=body.show_browser,
                include_academia=body.include_academia,
                strategy_order=body.strategy_order,
            )
            if result.get("success") and result.get("filepath"):
                fp = str(result["filepath"])
                if fp not in file_paths:
                    file_paths.append(fp)
                downloaded_now += 1
                try:
                    from src.tasks.dispatcher import _mark_library_paper_downloaded

                    _mark_library_paper_downloaded(row.id)
                except Exception as e:
                    logger.warning("mark downloaded_at failed for library paper %s: %s", row.id, e)
            else:
                failed_downloads += 1

    unique_paths = list(dict.fromkeys(file_paths))
    if not unique_paths:
        raise HTTPException(status_code=400, detail="该文献库没有可入库的 PDF，请先下载文献")

    from src.indexing.ingest_job_store import create_job

    collection_name = (body.collection or "").strip() or settings.collection.global_
    ingest_cfg = _build_library_ingest_cfg(
        file_paths=unique_paths,
        collection_name=collection_name,
        user_id=user_id,
        skip_duplicate_doi=body.skip_duplicate_doi,
        skip_unchanged=body.skip_unchanged,
    )
    job = create_job(collection_name, ingest_cfg, total_files=len(unique_paths))
    job_id = job.get("job_id")
    if not job_id:
        raise HTTPException(status_code=500, detail="创建入库任务失败")

    return {
        "ok": True,
        "job_id": job_id,
        "collection": collection_name,
        "library_id": lib_id,
        "total_library_papers": len(rows),
        "pdf_ready_count": len(unique_paths),
        "missing_pdf_count": max(0, len(missing_rows) - downloaded_now),
        "attempted_downloads": attempted_downloads,
        "downloaded_now": downloaded_now,
        "failed_downloads": failed_downloads,
    }


@router.get("/libraries/{lib_id:int}/pdf/{paper_id:path}")
def get_library_pdf(lib_id: int, paper_id: str, user_id: str = Depends(get_current_user_id)):
    """Serve PDF from a permanent library's pdfs folder. paper_id is the filename stem (e.g. DOI-based)."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库无本地 PDF")
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
    if not lib or getattr(lib, "user_id", None) != user_id:
        raise HTTPException(status_code=404, detail="库不存在")
    pdfs_dir = _resolve_library_download_dir(lib_id)
    if not pdfs_dir:
        raise HTTPException(status_code=400, detail="该库无 PDF 目录")
    # Sanitize: only allow safe filename stem (no path segments)
    if ".." in paper_id or "/" in paper_id or "\\" in paper_id:
        raise HTTPException(status_code=400, detail="无效 paper_id")
    pdf_path = Path(pdfs_dir) / f"{paper_id}.pdf"
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF 未找到")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{paper_id}.pdf",
    )


@router.post("/libraries/{lib_id:int}/papers/{record_id:int}/upload-pdf")
async def upload_library_paper_pdf(
    lib_id: int,
    record_id: int,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Upload a PDF for a library paper; store as {paper_id}.pdf, set downloaded_at."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库不支持上传 PDF")
    pdfs_dir = _resolve_library_download_dir(lib_id)
    if not pdfs_dir:
        raise HTTPException(status_code=400, detail="该库无 PDF 目录")
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        row = session.get(ScholarLibraryPaper, record_id)
        if not row or row.library_id != lib_id:
            raise HTTPException(status_code=404, detail="文献不存在")
        paper_id = _library_paper_id(
            (row.doi or "").strip() or None,
            row.title or "",
            row.get_authors(),
            row.year,
        )
        if not paper_id:
            raise HTTPException(status_code=400, detail="无法生成 paper_id（需 DOI 或标题）")
    content = await file.read()
    if not content or content[:4] != b"%PDF":
        raise HTTPException(status_code=400, detail="无效的 PDF 文件")
    pdfs_path = Path(pdfs_dir)
    pdfs_path.mkdir(parents=True, exist_ok=True)
    target = pdfs_path / f"{paper_id}.pdf"
    target.write_bytes(content)
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with Session(get_engine()) as session:
        row = session.get(ScholarLibraryPaper, record_id)
        if row:
            row.downloaded_at = now
            session.add(row)
            session.commit()
    return {"success": True, "paper_id": paper_id, "filename": f"{paper_id}.pdf"}


@router.post("/libraries/{lib_id:int}/import-pdfs", response_model=LibraryImportPdfSummary)
async def import_library_pdfs(
    lib_id: int,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Bulk import local PDF files into one permanent library and auto-link records."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库不支持该操作")
    if not files:
        raise HTTPException(status_code=400, detail="未提供 PDF 文件")

    pdfs_dir = _resolve_library_download_dir(lib_id)
    if not pdfs_dir:
        raise HTTPException(status_code=400, detail="该库无 PDF 目录")
    pdfs_path = Path(pdfs_dir)
    pdfs_path.mkdir(parents=True, exist_ok=True)

    from src.indexing.paper_metadata_store import paper_meta_store
    from src.parser.pdf_parser import extract_native_metadata
    from src.retrieval.dedup import extract_doi_from_pdf_tiered, normalize_doi, normalize_title
    from src.retrieval.downloader.adapter import _normalize_to_paper_id

    summary: Dict[str, Any] = {
        "total_files": len(files),
        "imported": 0,
        "linked_existing": 0,
        "renamed": 0,
        "skipped_duplicates": 0,
        "invalid_pdf": 0,
        "no_doi": 0,
        "errors": [],
    }
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")

        existing_rows = list(
            session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
        )
        existing_by_doi, existing_by_title = _build_library_row_index(existing_rows)

        for idx, upload in enumerate(files):
            raw_name = (upload.filename or f"import_{idx}.pdf").strip()
            original_stem = Path(raw_name).stem
            temp_path = pdfs_path / f".__import_{uuid.uuid4().hex}.pdf"
            target_path: Optional[Path] = None
            try:
                payload = await upload.read()
                if not payload or payload[:4] != b"%PDF":
                    summary["invalid_pdf"] += 1
                    summary["errors"].append(f"{raw_name}: invalid PDF")
                    continue

                temp_path.write_bytes(payload)

                extracted_doi, extracted_title = extract_doi_from_pdf_tiered(temp_path)
                native_meta = extract_native_metadata(temp_path)
                title = (
                    (extracted_title or native_meta.get("title") or original_stem or "").strip()
                    or "(无标题)"
                )
                ntitle = normalize_title(title)
                ndoi = normalize_doi(extracted_doi)
                if not ndoi:
                    summary["no_doi"] += 1

                if ndoi:
                    paper_id = _doi_to_paper_id_for_api(ndoi)
                    existing = existing_by_doi.get(ndoi)
                else:
                    paper_id = _normalize_to_paper_id(title or "", [], None)
                    existing = existing_by_title.get(ntitle) if ntitle else None
                target_path = pdfs_path / f"{paper_id}.pdf"

                if existing is not None:
                    summary["linked_existing"] += 1
                    if not target_path.exists():
                        temp_path.rename(target_path)
                        summary["imported"] += 1
                        if target_path.stem != original_stem:
                            summary["renamed"] += 1
                    else:
                        summary["skipped_duplicates"] += 1

                    changed = False
                    if ndoi and not (existing.doi or "").strip():
                        existing.doi = ndoi
                        changed = True
                    if title and ((existing.title or "").strip() in ("", "(无标题)")):
                        existing.title = title
                        changed = True
                    if not (existing.downloaded_at or "").strip():
                        existing.downloaded_at = now_iso
                        changed = True
                    if not (existing.source or "").strip():
                        existing.source = "folder_import"
                        changed = True
                    if changed:
                        session.add(existing)

                    if ndoi:
                        existing_by_doi[ndoi] = existing
                    if ntitle and ntitle not in existing_by_title:
                        existing_by_title[ntitle] = existing

                    try:
                        paper_meta_store.upsert(
                            target_path.stem,
                            doi=ndoi or None,
                            title=title,
                            source="library_folder_import",
                        )
                    except Exception:
                        pass
                    continue

                if target_path.exists():
                    summary["skipped_duplicates"] += 1
                    continue

                temp_path.rename(target_path)
                summary["imported"] += 1
                if target_path.stem != original_stem:
                    summary["renamed"] += 1

                new_row = ScholarLibraryPaper(
                    library_id=lib_id,
                    title=title,
                    authors="[]",
                    year=None,
                    doi=ndoi or "",
                    pdf_url="",
                    url="",
                    source="folder_import",
                    score=0.0,
                    annas_md5="",
                    added_at=now_iso,
                    downloaded_at=now_iso,
                )
                session.add(new_row)
                session.flush()

                if ndoi:
                    existing_by_doi[ndoi] = new_row
                if ntitle and ntitle not in existing_by_title:
                    existing_by_title[ntitle] = new_row

                try:
                    paper_meta_store.upsert(
                        target_path.stem,
                        doi=ndoi or None,
                        title=title,
                        source="library_folder_import",
                    )
                except Exception:
                    pass
            except Exception as e:
                summary["errors"].append(f"{raw_name}: {e}")
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        session.commit()

    # Keep payload bounded for UI toast/readability.
    if len(summary["errors"]) > 20:
        extra = len(summary["errors"]) - 20
        summary["errors"] = summary["errors"][:20] + [f"... and {extra} more errors"]
    return summary


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
                    existing_row.venue = paper.venue or ""
                    existing_row.normalized_journal_name = paper.normalized_journal_name or ""
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


@router.delete("/libraries/{lib_id:int}/papers/{record_id:int}/pdf")
def delete_library_paper_pdf(lib_id: int, record_id: int, user_id: str = Depends(get_current_user_id)):
    """Delete the downloaded PDF file for a library paper and clear downloaded_at. Paper stays in the library so user can re-download."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库无本地 PDF")
    pdfs_dir = _resolve_library_download_dir(lib_id)
    if not pdfs_dir or not Path(pdfs_dir).is_dir():
        raise HTTPException(status_code=400, detail="该库无 PDF 目录")
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        paper = session.get(ScholarLibraryPaper, record_id)
        if not paper or paper.library_id != lib_id:
            raise HTTPException(status_code=404, detail="文献不存在")
        stem = _library_paper_id(
            (paper.doi or "").strip() or None,
            paper.title or "",
            paper.get_authors(),
            paper.year,
        )
        if not stem:
            raise HTTPException(status_code=400, detail="无法解析 paper_id（需 DOI 或标题）")
        pdf_path = Path(pdfs_dir) / f"{stem}.pdf"
        if pdf_path.is_file():
            try:
                pdf_path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("delete_library_paper_pdf unlink failed: %s", e)
                raise HTTPException(status_code=500, detail="删除文件失败")
        paper.downloaded_at = None
        session.add(paper)
        session.commit()
    return {"ok": True, "paper_id": stem}


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


@router.post("/libraries/{lib_id:int}/extract-doi-dedup")
def extract_doi_dedup_library(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """Extract missing DOIs (from url/title + optional Crossref), normalize, dedupe by DOI; update library in place."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = list(_temp_store[lib_id].get("papers") or [])
        # Copy so we can mutate; preserve structure (authors may be list)
        papers = [dict(p) for p in papers]
        deduped, stats = _extract_doi_and_dedup_library_papers(papers)
        _temp_store[lib_id]["papers"] = deduped
        _temp_store[lib_id]["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return {"extracted_count": stats["extracted_count"], "removed_count": stats["removed_count"]}
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        rows = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        papers = [
            {
                "id": p.id,
                "library_id": p.library_id,
                "title": p.title,
                "authors": p.get_authors(),
                "year": p.year,
                "doi": p.doi or "",
                "pdf_url": p.pdf_url or "",
                "url": p.url or "",
                "source": p.source or "",
                "score": p.score,
                "annas_md5": p.annas_md5 or "",
                "added_at": p.added_at,
                "downloaded_at": getattr(p, "downloaded_at", None),
            }
            for p in rows
        ]
        deduped, stats = _extract_doi_and_dedup_library_papers(papers)
        ids_to_keep = {p["id"] for p in deduped}
        by_id = {p["id"]: p for p in deduped}
        for row in rows:
            if row.id not in ids_to_keep:
                session.delete(row)
            else:
                d = by_id.get(row.id)
                if d and (d.get("doi") or "") != (row.doi or ""):
                    row.doi = d.get("doi") or ""
        session.commit()
        return {"extracted_count": stats["extracted_count"], "removed_count": stats["removed_count"]}


@router.post("/libraries/{lib_id:int}/pdf-rename-dedup")
def pdf_rename_dedup_library(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """Scan pdfs/ folder: extract DOI, rename to DOI-based stem, remove duplicate PDFs."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="仅永久库支持此操作")
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
    if not lib or getattr(lib, "user_id", None) != user_id:
        raise HTTPException(status_code=404, detail="库不存在")
    pdfs_dir = _resolve_library_download_dir(lib_id)
    if not pdfs_dir or not Path(pdfs_dir).is_dir():
        raise HTTPException(status_code=400, detail="该库无 PDF 文件夹")
    stats = _pdf_rename_dedup_library_folder(pdfs_dir)
    # Sync ScholarLibraryPaper.downloaded_at so frontend "已下载" badges update after rename
    synced = _sync_library_paper_downloaded_at(lib_id, pdfs_dir)
    stats["synced_downloaded"] = synced
    return stats


@router.post("/papers/extract-doi-dedup")
def extract_doi_dedup_papers(
    body: ExtractDoiDedupPapersRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Return papers with DOIs filled and deduped by DOI (for client-side temp libraries). No DB mutation."""
    papers = [dict(p) for p in (body.papers or [])]
    deduped, stats = _extract_doi_and_dedup_library_papers(papers)
    return {
        "papers": deduped,
        "extracted_count": stats["extracted_count"],
        "removed_count": stats["removed_count"],
    }


@router.get("/health")
async def scholar_health():
    from src.retrieval.downloader.adapter import is_adapter_ready

    cfg = getattr(settings, "scholar_downloader", None)
    allowed = [
        "direct_download",
        "playwright_download",
        "browser_lookup",
        "sci_hub",
        "brightdata",
        "anna",
    ]
    configured = list(getattr(cfg, "default_strategy_order", [])) if cfg else []
    normalized = [item for item in configured if item in allowed]
    for item in allowed:
        if item not in normalized:
            normalized.append(item)
    return {
        "enabled": bool(cfg and cfg.enabled),
        "adapter_ready": is_adapter_ready(),
        "download_dir": getattr(cfg, "download_dir", "") if cfg else "",
        "default_strategy_order": normalized,
    }


@router.get("/browser/headed", response_model=HeadedBrowserWindowState)
async def scholar_headed_browser_state():
    from src.retrieval.browser_service import SharedBrowserService

    state = SharedBrowserService.get_headed_window_state()
    state["bounds"] = (
        SharedBrowserService._get_headed_window_bounds(state["mode"])
        if state.get("mode") in {"parked", "visible"}
        else None
    )
    return state


@router.post("/browser/headed/show", response_model=HeadedBrowserWindowState)
async def scholar_show_headed_browser():
    from src.retrieval.browser_service import SharedBrowserService

    try:
        return await SharedBrowserService.show_headed()
    except Exception as e:
        logger.warning("[scholar] show headed browser failed: %s", e)
        raise HTTPException(status_code=503, detail=f"无法召回有头浏览器: {e}")


@router.post("/browser/headed/park", response_model=HeadedBrowserWindowState)
async def scholar_park_headed_browser():
    from src.retrieval.browser_service import SharedBrowserService

    try:
        return await SharedBrowserService.park_headed()
    except Exception as e:
        logger.warning("[scholar] park headed browser failed: %s", e)
        raise HTTPException(status_code=503, detail=f"无法停靠有头浏览器: {e}")
