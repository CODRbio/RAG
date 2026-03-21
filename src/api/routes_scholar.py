"""
Scholar API: search + download + ingest.
Prefix: /scholar

Google Scholar / Google search use RAG unified_web_search (SerpAPI + Playwright + serpapi_ratio).
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
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
from src.tasks.dispatcher import process_download_and_ingest, _scholar_is_cancelled
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskStatus, TaskState
from src.utils.path_manager import PathManager

logger = get_logger(__name__)
router = APIRouter(prefix="/scholar", tags=["scholar"])

# Registry of background tasks (batch, recommend) for graceful shutdown
_scholar_bg_tasks: set = set()


def _register_scholar_bg_task(t: asyncio.Task) -> None:
    _scholar_bg_tasks.add(t)
    t.add_done_callback(_scholar_bg_tasks.discard)


async def wait_scholar_background_tasks_or_timeout(timeout_seconds: float) -> None:
    """Wait for in-flight Scholar tasks up to timeout_seconds; then cancel any remaining."""
    if not _scholar_bg_tasks:
        return
    try:
        done, pending = await asyncio.wait(
            _scholar_bg_tasks.copy(),
            timeout=max(0.1, timeout_seconds),
            return_when=asyncio.ALL_COMPLETED,
        )
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info("[scholar] graceful shutdown: waited %.1fs, cancelled %d task(s)", timeout_seconds, len(pending))
    except Exception as e:
        logger.warning("[scholar] shutdown wait failed: %s", e)


def _mark_graph_scopes_stale_for_library(
    user_id: str,
    lib_id: int,
    *,
    collection_name: Optional[str] = None,
    reason: str,
) -> None:
    try:
        from src.services.global_graph_service import mark_graph_scope_stale

        mark_graph_scope_stale(
            user_id=user_id,
            scope_type="library",
            scope_key=str(int(lib_id)),
            reason=reason,
        )
        if (collection_name or "").strip():
            mark_graph_scope_stale(
                user_id=user_id,
                scope_type="collection",
                scope_key=str(collection_name).strip(),
                reason=reason,
            )
    except Exception as e:
        logger.debug(
            "mark graph scopes stale for library failed user=%s lib_id=%s collection=%s err=%s",
            user_id,
            lib_id,
            collection_name,
            e,
        )


class ScholarSearchRequest(BaseModel):
    query: str
    source: str = "google_scholar"  # google_scholar | google | semantic | semantic_relevance | semantic_bulk | ncbi | annas_archive
    limit: int = 30
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    optimize: bool = False  # when True, use ultra_lite to rewrite query per source before search
    use_serpapi: bool = False  # when True for google_scholar/google, part of queries go via SerpAPI (see serpapi_ratio)
    serpapi_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)  # 0–1, share of queries to SerpAPI; default 0.5 when use_serpapi=True


# Sources included in batch search (all except plain Google)
BATCH_SEARCH_DEFAULT_SOURCES: List[str] = [
    "google_scholar",
    "semantic_relevance",
    "semantic_bulk",
    "ncbi",
    "annas_archive",
]


class ScholarBatchSearchRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None  # defaults to BATCH_SEARCH_DEFAULT_SOURCES when None
    limit_per_source: int = 20
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    optimize: bool = True  # each source gets its own LLM-optimized query


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
    assist_llm_mode: Optional[str] = Field(
        None,
        description="下载辅助 LLM 模式: ultra-lite | lite | auto-upgrade",
    )
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
    assist_llm_mode: Optional[str] = Field(
        None,
        description="下载辅助 LLM 模式: ultra-lite | lite | auto-upgrade; applies to all papers in batch",
    )
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
    enrich_tables: bool = False
    enrich_figures: bool = False
    llm_text_provider: Optional[str] = None
    llm_text_model: Optional[str] = None
    llm_text_concurrency: Optional[int] = None
    llm_vision_provider: Optional[str] = None
    llm_vision_model: Optional[str] = None
    llm_vision_concurrency: Optional[int] = None
    llm_provider: Optional[str] = None
    model_override: Optional[str] = None
    assist_llm_mode: Optional[str] = None
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


class LibraryRefreshMetadataSummary(BaseModel):
    updated: int
    skipped_no_doi: int
    failed: int


class LibraryRecommendRequest(BaseModel):
    question: str
    collection: Optional[str] = None
    candidate_library_paper_ids: Optional[List[int]] = None
    top_k: int = Field(default=10, ge=1, le=50)


class LibraryRecommendItem(BaseModel):
    library_paper_id: int
    collection_paper_id: str
    paper_uid: Optional[str] = None
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    impact_factor: Optional[float] = None
    score: float
    matched_chunks: int
    best_chunk_score: float
    snippets: List[str]


class LibraryRecommendResponse(BaseModel):
    question: str
    library_id: int
    collection: str
    total_candidates: int
    eligible_candidates: int
    excluded_not_ingested: int
    recommendations: List[LibraryRecommendItem]


class RecommendStartResponse(BaseModel):
    task_id: str
    status: str = "submitted"
    message: Optional[str] = None


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


@router.post("/search/batch")
async def scholar_search_batch(req: ScholarBatchSearchRequest):
    """
    Search multiple sources simultaneously; each source gets its own LLM-optimized query
    when optimize=True. Results are merged and deduplicated by DOI.
    Excluded: plain Google (use single-source /search for that).
    """
    if not getattr(settings, "scholar_downloader", None) or not settings.scholar_downloader.enabled:
        raise HTTPException(status_code=503, detail="Scholar downloader is disabled")

    from src.retrieval.downloader.adapter import get_adapter
    from src.retrieval.scholar_query_optimizer import optimize_scholar_query
    from src.retrieval.unified_web_search import unified_web_searcher
    from src.retrieval.semantic_scholar import semantic_scholar_searcher

    valid_sources = {"google_scholar", "semantic_relevance", "semantic", "semantic_bulk", "ncbi", "annas_archive"}
    requested = req.sources if req.sources else BATCH_SEARCH_DEFAULT_SOURCES
    sources = [s for s in requested if s in valid_sources]
    if not sources:
        return {"results": [], "count": 0, "source_counts": {}}

    adapter = get_adapter()

    async def _search_one(source: str) -> List[Dict[str, Any]]:
        query = req.query
        if req.optimize:
            try:
                query = optimize_scholar_query(query, source)
                logger.info("batch search: optimized for source=%s -> %r", source, query[:80])
            except Exception as e:
                logger.warning("batch search: optimize failed for source=%s: %s", source, e)
        try:
            if source == "google_scholar":
                raw_hits = await unified_web_searcher.search(
                    query,
                    providers=["scholar"],
                    source_configs={"scholar": {"topK": req.limit_per_source, "useSerpapi": False}},
                    max_results_per_provider=req.limit_per_source,
                    year_start=req.year_start,
                    year_end=req.year_end,
                    serpapi_ratio=None,
                    use_content_fetcher="off",
                )
                return [_normalize_web_hit_for_scholar(h, "google_scholar") for h in raw_hits]
            elif source in ("semantic", "semantic_relevance"):
                relevance_res = await semantic_scholar_searcher.search(
                    query, limit=req.limit_per_source,
                    year_start=req.year_start, year_end=req.year_end,
                )
                raw_hits = relevance_res if isinstance(relevance_res, list) else []
                return [_normalize_web_hit_for_scholar(h, "semantic") for h in raw_hits]
            elif source == "ncbi":
                raw_hits = await unified_web_searcher.search(
                    query,
                    providers=["ncbi"],
                    source_configs={"ncbi": {"topK": req.limit_per_source}},
                    max_results_per_provider=req.limit_per_source,
                    year_start=req.year_start,
                    year_end=req.year_end,
                    use_content_fetcher="off",
                )
                return [_normalize_web_hit_for_scholar(h, "ncbi") for h in raw_hits]
            elif source == "semantic_bulk":
                return await adapter.search_semantic_scholar_bulk(
                    query=query, limit=req.limit_per_source,
                    year_start=req.year_start, year_end=req.year_end,
                )
            elif source == "annas_archive":
                return await adapter.search_annas_archive(query=query, limit=req.limit_per_source)
            else:
                return []
        except Exception as e:
            logger.warning("batch search source=%s failed: %s", source, e)
            return []

    all_results_lists = await asyncio.gather(*[_search_one(src) for src in sources])

    source_counts: Dict[str, int] = {}
    all_merged: List[Dict[str, Any]] = []
    for src, results_list in zip(sources, all_results_lists):
        source_counts[src] = len(results_list)
        all_merged.extend(results_list)

    deduped = _dedupe_papers_by_doi_keep_best_source(all_merged)
    _enrich_scholar_results_with_crossref(deduped)
    _normalize_and_enrich_venue(deduped)

    logger.info(
        "batch search done: sources=%s source_counts=%s total_after_dedup=%d",
        sources, source_counts, len(deduped),
    )
    return {"results": deduped, "count": len(deduped), "source_counts": source_counts}


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
            assist_llm_mode=getattr(req, "assist_llm_mode", None),
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

    result = await get_adapter(show_browser=getattr(req, "show_browser", None)).download_paper(
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
        assist_llm_mode=getattr(req, "assist_llm_mode", None),
        assist_llm_enabled=getattr(req, "assist_llm_enabled", None),
        show_browser=getattr(req, "show_browser", None),
        include_academia=getattr(req, "include_academia", False),
        strategy_order=getattr(req, "strategy_order", None),
    )
    if result.get("success") and result.get("should_mark_downloaded", True) and req.library_paper_id is not None:
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
        ndoi = normalize_doi(doi) if doi else ""
        if not ndoi:
            no_doi += 1
            continue
        new_stem = _doi_to_paper_id(ndoi)
        new_path = pdfs_path / f"{new_stem}.pdf"
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
    """Set downloaded_at for ScholarLibraryPaper rows where the PDF exists on disk.
    Uses _library_paper_id(doi, title, authors, year) so both DOI-based and fallback (title/authors/year) papers are recognized."""
    from src.retrieval.downloader.utils import is_valid_pdf

    pdfs_path = Path(pdfs_dir)
    if not pdfs_path.is_dir():
        return 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    updated = 0
    with Session(get_engine()) as session:
        rows = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        for row in rows:
            if getattr(row, "downloaded_at", None):
                continue
            paper_id = _library_paper_id(
                (row.doi or "").strip() or None,
                (row.title or "").strip(),
                row.get_authors() if hasattr(row, "get_authors") else [],
                row.year,
            )
            if not paper_id:
                continue
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

        batch_show_browser = getattr(req, "show_browser", None)
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
                if s and s.status in (TaskStatus.completed, TaskStatus.error, TaskStatus.cancelled):
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
        adapter = get_adapter(show_browser=batch_show_browser)

        async def _one(paper: DownloadRequest):
            nonlocal completed, failed
            if _scholar_is_cancelled(task_id, q):
                return {"cancelled": True, "success": False}
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
                        assist_llm_mode=getattr(req, "assist_llm_mode", None) or getattr(paper, "assist_llm_mode", None),
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
                        assist_llm_mode=getattr(req, "assist_llm_mode", None) or getattr(paper, "assist_llm_mode", None),
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
                    if result.get("success") and result.get("should_mark_downloaded", True) and paper.library_paper_id is not None:
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
                if result.get("cancelled"):
                    pass
                elif result.get("ingest_triggered") or result.get("success"):
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
                if state.status == TaskStatus.cancelled:
                    q.set_state(state)
                    logger.info(
                        '{"task_type":"scholar_batch","task_id":"%s","event":"task_cancelled","duration_ms":%.0f}',
                        task_id, (state.finished_at - (state.started_at or state.created_at or 0)) * 1000,
                    )
                elif failed == total:
                    state.status = TaskStatus.error
                    state.error_message = "所有论文下载均失败"
                    q.set_state(state)
                    q.push_event(
                        task_id,
                        "done",
                        {"completed": completed, "failed": failed, "total": total},
                    )
                    logger.info(
                        '{"task_type":"scholar_batch","task_id":"%s","event":"task_failed","duration_ms":%.0f,"error_message":"%s"}',
                        task_id, (time.time() - batch_start) * 1000, state.error_message or "all_failed",
                    )
                else:
                    state.status = TaskStatus.completed
                    state.error_message = None
                    q.set_state(state)
                    q.push_event(
                        task_id,
                        "done",
                        {"completed": completed, "failed": failed, "total": total},
                    )
                    logger.info(
                        '{"task_type":"scholar_batch","task_id":"%s","event":"task_completed","duration_ms":%.0f}',
                        task_id, (time.time() - batch_start) * 1000,
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
    _register_scholar_bg_task(batch_task)

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
    """Delete a scholar library and all its papers (temp or permanent). Deletes folder_path directory if set."""
    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        _temp_store.pop(lib_id, None)
        return {"ok": True}
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        folder_path = getattr(lib, "folder_path", None)
        paper_ids = [
            int(p.id)
            for p in session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
            if getattr(p, "id", None) is not None
        ]
        session.delete(lib)
        session.commit()
    try:
        from src.indexing.assistant_artifact_store import delete_resource_annotations_for_resource
        from src.services.resource_state_service import get_resource_state_service

        resource_state_service = get_resource_state_service()
        for paper_id in paper_ids:
            delete_resource_annotations_for_resource(
                user_id=user_id,
                resource_type="scholar_library_paper",
                resource_id=str(paper_id),
            )
            resource_state_service.delete_resource_overlays(
                user_id=user_id,
                resource_type="scholar_library_paper",
                resource_id=str(paper_id),
            )
    except Exception as e:
        logger.warning("delete scholar library overlay cleanup failed lib_id=%s: %s", lib_id, e)
    if folder_path:
        import shutil
        try:
            p = Path(folder_path)
            if p.exists() and p.is_dir():
                shutil.rmtree(p)
                logger.info("Deleted library folder: %s", folder_path)
        except Exception as e:
            logger.warning("Failed to delete library folder %s: %s", folder_path, e)
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
    """For each normalized DOI keep one paper: best source priority; when equal, prefer non-Academia pdf_url. No-DOI items kept as-is."""
    from src.retrieval.dedup import normalize_doi
    if not items:
        return items
    no_doi: List[Dict[str, Any]] = []
    by_doi: Dict[str, Dict[str, Any]] = {}
    for item in items:
        meta = item.get("metadata") or {}
        doi = (meta.get("doi") or "").strip()
        ndoi = normalize_doi(doi)
        pri = _scholar_source_priority(meta.get("source") or "")
        if not ndoi:
            no_doi.append(item)
            continue
        existing = by_doi.get(ndoi)
        if existing is None:
            by_doi[ndoi] = item
            continue
        existing_meta = existing.get("metadata") or {}
        existing_pri = _scholar_source_priority(existing_meta.get("source") or "")
        item_pdf = (meta.get("pdf_url") or "").strip()
        existing_pdf = (existing_meta.get("pdf_url") or "").strip()
        take_item = (
            pri < existing_pri
            or (
                pri == existing_pri
                and _is_academia_pdf_url(existing_pdf)
                and not _is_academia_pdf_url(item_pdf)
            )
        )
        if take_item:
            by_doi[ndoi] = item
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
        ndoi = _extract_doi_from_text(url) if url else None
        if not ndoi and title:
            ndoi = _extract_doi_from_text(title)
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
    from src.retrieval.dedup import compute_paper_uid
    meta = item.get("metadata") or {}
    authors = meta.get("authors") or []
    doi = (meta.get("doi") or "") or ""
    title = (meta.get("title") or "").strip() or "(无标题)"
    year = meta.get("year") if isinstance(meta.get("year"), (int, type(None))) else None
    authors_list = authors if isinstance(authors, list) else []
    return ScholarLibraryPaper(
        title=title,
        authors=json.dumps(authors_list) if isinstance(authors, list) else "[]",
        year=year,
        doi=doi,
        pdf_url=(meta.get("pdf_url") or "") or "",
        url=(meta.get("url") or "") or "",
        source=(meta.get("source") or "") or "",
        score=float(item.get("score") or 0),
        annas_md5=(meta.get("annas_md5") or "") or "",
        venue=(meta.get("venue") or "") or "",
        normalized_journal_name=(meta.get("normalized_journal_name") or "") or "",
        paper_uid=compute_paper_uid(
            doi=doi,
            title=title,
            authors=authors_list,
            year=year,
            url=meta.get("url") or "",
            pmid=meta.get("pmid") or "",
        ),
    )


def _temp_paper_from_search_item(item: Dict[str, Any], lib_id: int) -> Dict[str, Any]:
    """Build a temp-library paper dict with a negative paper id."""
    from src.retrieval.dedup import compute_paper_uid

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
        "paper_uid": compute_paper_uid(
            doi=(meta.get("doi") or "") or "",
            title=(meta.get("title") or "").strip() or "(无标题)",
            authors=authors,
            year=meta.get("year") if isinstance(meta.get("year"), (int, type(None))) else None,
            url=(meta.get("url") or "") or "",
            pmid=(meta.get("pmid") or "") or "",
        ),
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
    library_name: str,
    skip_duplicate_doi: bool,
    skip_unchanged: bool,
    file_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build ingest config payload for library -> ingest bridge."""
    return {
        "file_paths": file_paths,
        "file_metadata_map": file_metadata_map or {},
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
        "library_name": library_name,
    }


def _sync_remove_library_deleted_papers(
    *,
    collection_name: str,
    lib_id: int,
    user_id: str,
    library_name: str,
    current_library_paper_ids: set[int],
) -> int:
    """Delete papers from vector/Paper/parsed_raw when they are no longer in the library."""
    removed_papers = 0
    parsed_root = PathManager.get_user_library_parsed_path(user_id, library_name)
    try:
        from src.indexing.paper_store import list_papers_linked_to_library, delete_paper
        from src.indexing.milvus_ops import milvus

        linked = list_papers_linked_to_library(collection_name, lib_id, user_id=user_id)
        for p in linked:
            lp_id = p.get("library_paper_id")
            if lp_id is None:
                continue
            try:
                lp_id_int = int(lp_id)
            except Exception:
                continue
            if lp_id_int in current_library_paper_ids:
                continue

            paper_id = (p.get("paper_id") or "").strip()
            if not paper_id:
                continue
            try:
                if milvus.client.has_collection(collection_name):
                    milvus.client.delete(
                        collection_name=collection_name,
                        filter=f'paper_id == "{paper_id}"',
                    )
                delete_paper(collection_name, paper_id)
                parsed_paper_dir = parsed_root / paper_id
                if parsed_paper_dir.exists():
                    shutil.rmtree(parsed_paper_dir)
                removed_papers += 1
            except Exception as e:
                logger.warning("sync-remove paper from collection failed paper_id=%s: %s", paper_id, e)
    except Exception as e:
        logger.warning("library ingest sync-removal (remove deleted papers from vector) failed: %s", e)
    return removed_papers


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


def _normalize_authors_for_paper_uid(authors: Any) -> List[str]:
    if isinstance(authors, list):
        return [str(a) for a in authors if str(a).strip()]
    if isinstance(authors, str) and authors.strip():
        return [authors.strip()]
    return []


def _resolve_paper_uid_aux(collection_paper_id: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    if not collection_paper_id:
        return None, None, None
    try:
        from src.indexing.paper_metadata_store import paper_meta_store

        meta = paper_meta_store.get(collection_paper_id)
    except Exception:
        meta = None
    if not meta:
        return None, None, None
    extra = meta.get("extra") or {}
    url = (extra.get("url") or extra.get("pdf_url") or "").strip() or None
    pmid = (extra.get("pmid") or "").strip() or None
    return url, pmid, meta


def _resolve_library_paper_uid(paper: Any) -> Optional[str]:
    from src.retrieval.dedup import compute_paper_uid

    raw_uid = getattr(paper, "paper_uid", None) if hasattr(paper, "paper_uid") else paper.get("paper_uid")
    uid = (raw_uid or "").strip()
    if uid:
        return uid
    authors = paper.get_authors() if hasattr(paper, "get_authors") else _normalize_authors_for_paper_uid(paper.get("authors"))
    coll_paper_id = getattr(paper, "collection_paper_id", None) if hasattr(paper, "collection_paper_id") else paper.get("collection_paper_id")
    url, pmid, meta = _resolve_paper_uid_aux((coll_paper_id or "").strip() or None)
    doi = getattr(paper, "doi", None) if hasattr(paper, "doi") else paper.get("doi")
    title = getattr(paper, "title", None) if hasattr(paper, "title") else paper.get("title")
    year = getattr(paper, "year", None) if hasattr(paper, "year") else paper.get("year")
    if meta:
        doi = doi or meta.get("doi")
        title = title or meta.get("title")
        year = year if year is not None else meta.get("year")
        if not authors:
            authors = _normalize_authors_for_paper_uid(meta.get("authors"))
    if not (doi or title or pmid):
        return None
    return compute_paper_uid(
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        url=url or (getattr(paper, "url", None) if hasattr(paper, "url") else paper.get("url")),
        pmid=pmid,
    )


def _chunk_lookup_key(chunk: Any) -> str:
    from src.retrieval.dedup import compute_paper_uid

    paper_uid = (getattr(chunk, "paper_uid", None) or "").strip()
    if paper_uid:
        return paper_uid
    doi = getattr(chunk, "doi", None)
    title = getattr(chunk, "doc_title", None)
    authors = getattr(chunk, "authors", None)
    year = getattr(chunk, "year", None)
    url = getattr(chunk, "url", None)
    pmid = getattr(chunk, "pmid", None)
    if doi or title or pmid:
        return compute_paper_uid(
            doi=doi,
            title=title,
            authors=authors,
            year=year,
            url=url,
            pmid=pmid,
        )
    return (getattr(chunk, "doc_id", None) or getattr(chunk, "chunk_id", None) or "").strip()


@router.get("/libraries/{lib_id:int}/papers")
def list_scholar_library_papers(
    lib_id: int,
    collection: Optional[str] = Query(None, description="If set, each paper includes in_collection and collection_paper_id for this collection"),
    user_id: str = Depends(get_current_user_id),
):
    """List all papers in a scholar library (temp or permanent). Optional collection param for in_collection/collection_paper_id."""
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
            d["in_collection"] = False
            d["collection_paper_id"] = None
            d["paper_uid"] = _resolve_library_paper_uid(d)
            out.append(d)
        return _attach_impact_factor_metadata(out)
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all()
        out = []
        for p in papers:
            coll_name = getattr(p, "collection_name", None) or ""
            coll_paper_id = getattr(p, "collection_paper_id", None) or ""
            in_coll = bool(collection and coll_name == collection)
            d = {
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
                "paper_uid": _resolve_library_paper_uid(p),
                "paper_id": _library_paper_id(
                    (p.doi or "").strip() or None,
                    p.title or "",
                    p.get_authors(),
                    p.year,
                ),
                "is_downloaded": bool(getattr(p, "downloaded_at", None)),
                "in_collection": in_coll,
                "collection_paper_id": coll_paper_id if in_coll else None,
            }
            out.append(d)
        return _attach_impact_factor_metadata(out)


def _aggregate_chunks_to_library_papers(
    chunks: List[Any],
    eligible_lookup: Dict[str, Any],
    top_k: int,
    max_snippets: int = 3,
) -> List[Dict[str, Any]]:
    """
    Group EvidenceChunk objects by paper_uid-first lookup and join them to library papers.

    eligible_lookup: dict mapping paper_uid -> ScholarLibraryPaper (or dict with same keys).
    Returns list of aggregated dicts sorted by (best_chunk_score desc, matched_chunks desc), capped at top_k.
    """
    from collections import defaultdict

    doc_chunks: Dict[str, List[Any]] = defaultdict(list)
    for chunk in chunks:
        lookup_key = _chunk_lookup_key(chunk)
        if lookup_key and lookup_key in eligible_lookup:
            doc_chunks[lookup_key].append(chunk)

    results: List[Dict[str, Any]] = []
    for lookup_key, chunk_list in doc_chunks.items():
        paper = eligible_lookup[lookup_key]
        scores = [getattr(c, "score", 0.0) for c in chunk_list]
        best_score = max(scores) if scores else 0.0
        # Sort chunks by score desc for snippet selection
        sorted_chunks = sorted(chunk_list, key=lambda c: getattr(c, "score", 0.0), reverse=True)
        snippets = [
            (getattr(c, "text", "") or "").strip()
            for c in sorted_chunks[:max_snippets]
            if (getattr(c, "text", "") or "").strip()
        ]

        p_id = getattr(paper, "id", None) if hasattr(paper, "id") else paper.get("id")
        p_title = getattr(paper, "title", None) if hasattr(paper, "title") else paper.get("title", "")
        p_doi = getattr(paper, "doi", None) if hasattr(paper, "doi") else paper.get("doi")
        p_year = getattr(paper, "year", None) if hasattr(paper, "year") else paper.get("year")
        p_venue = getattr(paper, "venue", None) if hasattr(paper, "venue") else paper.get("venue")
        p_if = getattr(paper, "impact_factor", None) if hasattr(paper, "impact_factor") else paper.get("impact_factor")
        p_coll_pid = (
            getattr(paper, "collection_paper_id", None)
            if hasattr(paper, "collection_paper_id")
            else paper.get("collection_paper_id")
            if isinstance(paper, dict)
            else None
        )
        p_uid = (
            getattr(paper, "paper_uid", None)
            if hasattr(paper, "paper_uid")
            else paper.get("paper_uid")
            if isinstance(paper, dict)
            else None
        )

        results.append({
            "library_paper_id": p_id,
            "collection_paper_id": (p_coll_pid or lookup_key),
            "paper_uid": (p_uid or lookup_key),
            "title": p_title or "",
            "doi": (p_doi or "").strip() or None,
            "year": p_year,
            "venue": (p_venue or "").strip() or None,
            "impact_factor": p_if,
            "score": best_score,
            "matched_chunks": len(chunk_list),
            "best_chunk_score": best_score,
            "snippets": snippets,
        })

    results.sort(key=lambda r: (r["best_chunk_score"], r["matched_chunks"]), reverse=True)
    return results[:top_k]


_RECOMMEND_JOB_TIMEOUT_S = 120  # hard timeout for recommend retrieval + aggregate


def _recommend_sync_step(
    collection: str,
    question: str,
    retrieval_top_k: int,
    eligible_lookup: Dict[str, Dict[str, Any]],
    top_k: int,
    lib_id: int,
    total_candidates: int,
    eligible_count: int,
    excluded_count: int,
) -> Dict[str, Any]:
    """Run retrieval and aggregate in a thread; returns LibraryRecommendResponse as dict."""
    from src.retrieval.service import get_retrieval_service

    svc = get_retrieval_service(collection=collection)
    pack = svc.search(question, mode="local", top_k=retrieval_top_k)
    chunks = pack.chunks
    aggregated = _aggregate_chunks_to_library_papers(
        chunks=chunks,
        eligible_lookup=eligible_lookup,
        top_k=top_k,
        max_snippets=3,
    )
    raw_rows = [
        {
            "id": r["library_paper_id"],
            "library_paper_id": r["library_paper_id"],
            "collection_paper_id": r["collection_paper_id"],
            "paper_uid": r.get("paper_uid"),
            "title": r["title"],
            "doi": r["doi"],
            "year": r["year"],
            "venue": r["venue"],
            "impact_factor": r["impact_factor"],
            "score": r["score"],
            "matched_chunks": r["matched_chunks"],
            "best_chunk_score": r["best_chunk_score"],
            "snippets": r["snippets"],
        }
        for r in aggregated
    ]
    enriched = _attach_impact_factor_metadata(raw_rows)
    recommendations = [
        {
            "library_paper_id": row["library_paper_id"],
            "collection_paper_id": row["collection_paper_id"],
            "paper_uid": row.get("paper_uid"),
            "title": row["title"],
            "doi": row.get("doi"),
            "year": row.get("year"),
            "venue": row.get("venue"),
            "impact_factor": row.get("impact_factor"),
            "score": row["score"],
            "matched_chunks": row["matched_chunks"],
            "best_chunk_score": row["best_chunk_score"],
            "snippets": row["snippets"],
        }
        for row in enriched
    ]
    return {
        "question": question,
        "library_id": lib_id,
        "collection": collection,
        "total_candidates": total_candidates,
        "eligible_candidates": eligible_count,
        "excluded_not_ingested": excluded_count,
        "recommendations": recommendations,
    }


async def _run_recommend_job(
    task_id: str,
    lib_id: int,
    body: LibraryRecommendRequest,
    collection: str,
    eligible_lookup: Dict[str, Dict[str, Any]],
    total_candidates: int,
    eligible_count: int,
    excluded_count: int,
) -> None:
    q = get_task_queue()
    heartbeat_stop = asyncio.Event()
    job_start = time.time()
    _HEARTBEAT_INTERVAL = 5

    async def _heartbeat() -> None:
        while not heartbeat_stop.is_set():
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                pass
            if heartbeat_stop.is_set():
                break
            s = q.get_state(task_id)
            if s and s.status in (TaskStatus.completed, TaskStatus.error, TaskStatus.timeout, TaskStatus.cancelled):
                break
            elapsed = time.time() - job_start
            q.push_event(task_id, "heartbeat", {"stage": "recommend", "elapsed_s": round(elapsed, 1)})

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        if eligible_count == 0:
            q.push_event(
                task_id,
                "progress",
                {"stage": "no_eligible", "message": "无已入库候选，跳过检索"},
            )
            response_dict = {
                "question": body.question,
                "library_id": lib_id,
                "collection": collection,
                "total_candidates": total_candidates,
                "eligible_candidates": 0,
                "excluded_not_ingested": excluded_count,
                "recommendations": [],
            }
            q.push_event(task_id, "done", response_dict)
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.completed
                state.finished_at = time.time()
                state.payload["recommend_result"] = response_dict
                q.set_state(state)
            return

        if _scholar_is_cancelled(task_id, q):
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.cancelled
                state.finished_at = time.time()
                q.set_state(state)
            q.push_event(task_id, "cancelled", {"status": "cancelled"})
            return

        q.push_event(
            task_id,
            "progress",
            {"stage": "retrieval_started", "message": "正在检索..."},
        )
        retrieval_top_k = min(body.top_k * 5, 100)
        loop = asyncio.get_event_loop()
        try:
            future = loop.run_in_executor(
                None,
                lambda: _recommend_sync_step(
                    collection=collection,
                    question=body.question,
                    retrieval_top_k=retrieval_top_k,
                    eligible_lookup=eligible_lookup,
                    top_k=body.top_k,
                    lib_id=lib_id,
                    total_candidates=total_candidates,
                    eligible_count=eligible_count,
                    excluded_count=excluded_count,
                ),
            )
            response_dict = await asyncio.wait_for(future, timeout=_RECOMMEND_JOB_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            q.push_event(
                task_id,
                "error",
                {"message": f"推荐检索超时（{_RECOMMEND_JOB_TIMEOUT_S}s）"},
            )
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.timeout
                state.finished_at = time.time()
                state.error_message = "推荐检索超时"
                q.set_state(state)
            return
        except Exception as exc:
            logger.exception("[recommend] job failed task_id=%s: %s", task_id, exc)
            q.push_event(task_id, "error", {"message": str(exc) or exc.__class__.__name__})
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.error
                state.finished_at = time.time()
                state.error_message = str(exc) or exc.__class__.__name__
                q.set_state(state)
            return

        if _scholar_is_cancelled(task_id, q):
            state = q.get_state(task_id)
            if state:
                state.status = TaskStatus.cancelled
                state.finished_at = time.time()
                q.set_state(state)
            q.push_event(task_id, "cancelled", {"status": "cancelled"})
            return

        q.push_event(
            task_id,
            "progress",
            {"stage": "aggregating", "message": "正在汇总..."},
        )
        q.push_event(task_id, "done", response_dict)
        state = q.get_state(task_id)
        if state:
            state.status = TaskStatus.completed
            state.finished_at = time.time()
            state.payload["recommend_result"] = response_dict
            q.set_state(state)
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


@router.post("/libraries/{lib_id:int}/recommend/start", response_model=RecommendStartResponse)
async def recommend_library_papers_start(
    lib_id: int,
    body: LibraryRecommendRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Start an async recommend task; client subscribes to GET /scholar/task/{task_id}/stream for progress and result."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库不支持推荐")

    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")

        all_papers = session.exec(
            select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)
        ).all()

    collection = (body.collection or "").strip()
    if not collection:
        from config.settings import settings as _settings
        collection = (_settings.collection.global_ or "").strip()
    if not collection:
        raise HTTPException(status_code=400, detail="未指定知识库集合（collection），请先绑定子库或传入 collection 参数")

    candidate_set: Optional[set] = None
    if body.candidate_library_paper_ids is not None:
        candidate_set = set(body.candidate_library_paper_ids)

    total_candidates = len(all_papers) if candidate_set is None else len(candidate_set)
    eligible_lookup: Dict[str, Any] = {}
    excluded_count = 0
    for p in all_papers:
        if candidate_set is not None and p.id not in candidate_set:
            continue
        coll_name = (getattr(p, "collection_name", None) or "").strip()
        coll_paper_id = (getattr(p, "collection_paper_id", None) or "").strip()
        paper_uid = _resolve_library_paper_uid(p)
        if coll_name == collection and coll_paper_id and paper_uid:
            eligible_lookup[paper_uid] = {
                "id": p.id,
                "collection_paper_id": coll_paper_id,
                "paper_uid": paper_uid,
                "title": p.title or "",
                "doi": p.doi,
                "year": p.year,
                "venue": (getattr(p, "venue", None) or "").strip() or None,
                "impact_factor": getattr(p, "impact_factor", None),
            }
        else:
            excluded_count += 1

    eligible_count = len(eligible_lookup)
    task_id = f"recommend_{uuid.uuid4().hex[:12]}"
    q = get_task_queue()
    parent = TaskState(
        task_id=task_id,
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        started_at=time.time(),
        payload={
            "type": "recommend",
            "library_id": lib_id,
            "question": body.question,
            "collection": collection,
            "total_candidates": total_candidates,
            "eligible_candidates": eligible_count,
            "excluded_not_ingested": excluded_count,
        },
    )
    q.set_state(parent)
    recommend_task = asyncio.create_task(
        _run_recommend_job(
            task_id=task_id,
            lib_id=lib_id,
            body=body,
            collection=collection,
            eligible_lookup=eligible_lookup,
            total_candidates=total_candidates,
            eligible_count=eligible_count,
            excluded_count=excluded_count,
        ),
        name=f"recommend:{task_id}",
    )
    _register_scholar_bg_task(recommend_task)

    def _log_recommend_task(t: asyncio.Task) -> None:
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            logger.warning("[recommend] task_id=%s was cancelled", task_id)
            return
        if exc is not None:
            logger.exception("[recommend] task_id=%s escaped: %s", task_id, exc)

    recommend_task.add_done_callback(_log_recommend_task)
    return RecommendStartResponse(
        task_id=task_id,
        status="submitted",
        message="推荐任务已提交",
    )


@router.post("/libraries/{lib_id:int}/recommend", response_model=LibraryRecommendResponse)
async def recommend_library_papers(
    lib_id: int,
    body: LibraryRecommendRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Recommend the most relevant papers in a library for a given question using local RAG retrieval."""
    if _is_temp_library(lib_id):
        raise HTTPException(status_code=400, detail="临时库不支持推荐")

    from src.retrieval.service import get_retrieval_service

    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")

        all_papers = session.exec(
            select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)
        ).all()

    # Determine effective collection
    collection = (body.collection or "").strip()
    if not collection:
        from config.settings import settings as _settings
        collection = (_settings.collection.global_ or "").strip()
    if not collection:
        raise HTTPException(status_code=400, detail="未指定知识库集合（collection），请先绑定子库或传入 collection 参数")

    # Optionally restrict to candidate IDs from frontend filter
    candidate_set: Optional[set] = None
    if body.candidate_library_paper_ids is not None:
        candidate_set = set(body.candidate_library_paper_ids)

    total_candidates = len(all_papers) if candidate_set is None else len(candidate_set)

    # Partition: eligible = ingested into this collection
    eligible_lookup: Dict[str, Any] = {}
    excluded_count = 0
    for p in all_papers:
        if candidate_set is not None and p.id not in candidate_set:
            continue
        coll_name = (getattr(p, "collection_name", None) or "").strip()
        coll_paper_id = (getattr(p, "collection_paper_id", None) or "").strip()
        paper_uid = _resolve_library_paper_uid(p)
        if coll_name == collection and coll_paper_id and paper_uid:
            eligible_lookup[paper_uid] = p
        else:
            excluded_count += 1

    eligible_count = len(eligible_lookup)

    if eligible_count == 0:
        return LibraryRecommendResponse(
            question=body.question,
            library_id=lib_id,
            collection=collection,
            total_candidates=total_candidates,
            eligible_candidates=0,
            excluded_not_ingested=excluded_count,
            recommendations=[],
        )

    # Run local RAG retrieval (over-retrieve to get enough distinct docs)
    retrieval_top_k = min(body.top_k * 5, 100)
    try:
        svc = get_retrieval_service(collection=collection)
        pack = svc.search(body.question, mode="local", top_k=retrieval_top_k)
        chunks = pack.chunks
    except Exception as exc:
        logger.exception("[recommend] retrieval failed lib_id=%s: %s", lib_id, exc)
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}")

    # Aggregate chunks -> library papers
    aggregated = _aggregate_chunks_to_library_papers(
        chunks=chunks,
        eligible_lookup=eligible_lookup,
        top_k=body.top_k,
        max_snippets=3,
    )

    # Attach IF metadata by building minimal dicts compatible with _attach_impact_factor_metadata
    raw_rows = [
        {
            "id": r["library_paper_id"],
            "library_paper_id": r["library_paper_id"],
            "collection_paper_id": r["collection_paper_id"],
            "paper_uid": r.get("paper_uid"),
            "title": r["title"],
            "doi": r["doi"],
            "year": r["year"],
            "venue": r["venue"],
            "impact_factor": r["impact_factor"],
            "score": r["score"],
            "matched_chunks": r["matched_chunks"],
            "best_chunk_score": r["best_chunk_score"],
            "snippets": r["snippets"],
        }
        for r in aggregated
    ]
    enriched = _attach_impact_factor_metadata(raw_rows)

    recommendations = [
        LibraryRecommendItem(
            library_paper_id=row["library_paper_id"],
            collection_paper_id=row["collection_paper_id"],
            paper_uid=row.get("paper_uid"),
            title=row["title"],
            doi=row.get("doi"),
            year=row.get("year"),
            venue=row.get("venue"),
            impact_factor=row.get("impact_factor"),
            score=row["score"],
            matched_chunks=row["matched_chunks"],
            best_chunk_score=row["best_chunk_score"],
            snippets=row["snippets"],
        )
        for row in enriched
    ]

    return LibraryRecommendResponse(
        question=body.question,
        library_id=lib_id,
        collection=collection,
        total_candidates=total_candidates,
        eligible_candidates=eligible_count,
        excluded_not_ingested=excluded_count,
        recommendations=recommendations,
    )


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
    file_metadata_map: Dict[str, Dict[str, Any]] = {}
    missing_rows: List[ScholarLibraryPaper] = []

    def _record_file_metadata(path: Path, row: ScholarLibraryPaper) -> None:
        file_metadata_map[str(path)] = {
            "library_id": lib_id,
            "library_paper_id": int(row.id) if row.id is not None else None,
            "source": (row.source or "").strip() or "scholar_library",
            "metadata": {
                "doi": (row.doi or "").strip() or None,
                "title": (row.title or "").strip() or None,
                "authors": row.get_authors() or None,
                "year": row.year,
                "venue": (row.venue or "").strip() or None,
                "url": (row.url or "").strip() or None,
                "pdf_url": (row.pdf_url or "").strip() or None,
            },
        }

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
            _record_file_metadata(candidate, row)
        else:
            missing_rows.append(row)

    downloaded_now = 0
    failed_downloads = 0
    attempted_downloads = 0

    if body.auto_download_missing and missing_rows:
        from src.retrieval.downloader.adapter import get_adapter

        adapter = get_adapter(show_browser=body.show_browser)
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
                assist_llm_mode=body.assist_llm_mode,
                assist_llm_enabled=body.assist_llm_enabled,
                show_browser=body.show_browser,
                include_academia=body.include_academia,
                strategy_order=body.strategy_order,
            )
            if result.get("success") and result.get("filepath"):
                fp = str(result["filepath"])
                if fp not in file_paths:
                    file_paths.append(fp)
                _record_file_metadata(Path(fp), row)
                downloaded_now += 1
                if result.get("should_mark_downloaded", True):
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

    collection_name = (body.collection or "").strip() or settings.collection.global_
    current_library_paper_ids = {int(r.id) for r in rows if r.id is not None}
    removed_papers = _sync_remove_library_deleted_papers(
        collection_name=collection_name,
        lib_id=lib_id,
        user_id=user_id,
        library_name=lib.name,
        current_library_paper_ids=current_library_paper_ids,
    )

    from src.indexing.ingest_job_store import create_job

    ingest_cfg = _build_library_ingest_cfg(
        file_paths=unique_paths,
        collection_name=collection_name,
        user_id=user_id,
        library_name=lib.name,
        skip_duplicate_doi=body.skip_duplicate_doi,
        skip_unchanged=body.skip_unchanged,
        file_metadata_map={p: file_metadata_map[p] for p in unique_paths if p in file_metadata_map},
    )
    ingest_cfg["enrich_tables"] = body.enrich_tables
    ingest_cfg["enrich_figures"] = body.enrich_figures
    ingest_cfg["actual_skip"] = not (body.enrich_tables or body.enrich_figures)
    # Table enrichment defaults to qwen (qwen3.5, no thinking) when not specified
    if body.enrich_tables and not (body.llm_text_provider or "").strip():
        ingest_cfg["llm_text_provider"] = "qwen"
    elif body.llm_text_provider:
        ingest_cfg["llm_text_provider"] = body.llm_text_provider
    if body.llm_text_model:
        ingest_cfg["llm_text_model"] = body.llm_text_model
    if body.llm_text_concurrency:
        ingest_cfg["llm_text_concurrency"] = body.llm_text_concurrency
    if body.llm_vision_provider:
        ingest_cfg["llm_vision_provider"] = body.llm_vision_provider
    if body.llm_vision_model:
        ingest_cfg["llm_vision_model"] = body.llm_vision_model
    if body.llm_vision_concurrency:
        ingest_cfg["llm_vision_concurrency"] = body.llm_vision_concurrency

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
        "removed_papers": removed_papers,
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
    collection_name = ""
    with Session(get_engine()) as session:
        row = session.get(ScholarLibraryPaper, record_id)
        if row:
            collection_name = getattr(row, "collection_name", "") or ""
            row.downloaded_at = now
            session.add(row)
            session.commit()
    _mark_graph_scopes_stale_for_library(
        user_id=user_id,
        lib_id=lib_id,
        collection_name=collection_name,
        reason="library_paper_pdf_uploaded",
    )
    return {"success": True, "paper_id": paper_id, "filename": f"{paper_id}.pdf"}


def _extract_pdf_metadata_from_text(pdf_path: Path) -> Dict[str, Any]:
    """
    Lightweight full-text metadata extraction for library import.
    Best effort only: title/doi/authors/year/venue; crossref fallback happens later.
    """
    out: Dict[str, Any] = {}
    try:
        import fitz
        import re

        lines: List[str] = []
        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
            for page_idx in range(page_count):
                text = (doc[page_idx].get_text() or "").strip()
                if not text:
                    continue
                lines.extend([ln.strip() for ln in text.splitlines() if ln.strip()])
                # Keep extraction bounded for very large files.
                if len(lines) >= 600:
                    break

        if not lines:
            return out

        from src.retrieval.dedup import _extract_doi_from_text

        head_text = "\n".join(lines[:120])
        full_text = "\n".join(lines[:600])

        doi = _extract_doi_from_text(head_text) or _extract_doi_from_text(full_text)
        if doi:
            out["doi"] = doi

        # Title: prefer an early long, non-noisy line.
        for line in lines[:40]:
            ll = line.lower()
            if len(line) < 12 or len(line) > 260:
                continue
            if ll.startswith(("doi", "http://", "https://", "arxiv:", "copyright", "abstract")):
                continue
            if sum(ch.isdigit() for ch in line) / max(len(line), 1) > 0.08:
                continue
            out["title"] = line
            break

        # Authors: try first page-ish region, comma/and separated names.
        for line in lines[:30]:
            if len(line) < 8 or len(line) > 220:
                continue
            ll = line.lower()
            if any(k in ll for k in ("university", "department", "journal", "abstract", "doi")):
                continue
            parts = [p.strip() for p in line.replace(" and ", ",").split(",") if p.strip()]
            if not (1 <= len(parts) <= 12):
                continue
            if all(2 <= len(p.split()) <= 4 for p in parts) and sum(ch.isdigit() for ch in line) == 0:
                out["authors"] = parts
                break

        # Year: prefer a modern publication year candidate.
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", head_text)]
        if years:
            plausible = [y for y in years if 1900 <= y <= 2100]
            if plausible:
                out["year"] = max(plausible)

        # Venue: simple first-match heuristic.
        for line in lines[:120]:
            ll = line.lower()
            if any(k in ll for k in ("journal", "proceedings", "conference", "transactions on")):
                out["venue"] = line[:300]
                break
    except Exception as e:
        logger.debug("full-text metadata extraction failed for %s: %s", pdf_path, e)
    return out


def _resolve_library_import_metadata(raw_name: str, temp_path: Path) -> Dict[str, Any]:
    from src.parser.pdf_parser import extract_native_metadata
    from src.retrieval.dedup import (
        crossref_fetch_by_doi,
        _crossref_lookup_by_title,
        extract_doi_from_filename,
        extract_doi_from_pdf_tiered,
        normalize_doi,
    )

    original_stem = Path(raw_name).stem
    filename_doi = extract_doi_from_filename(raw_name)
    extracted_doi, extracted_title = extract_doi_from_pdf_tiered(temp_path)
    native_meta = extract_native_metadata(temp_path)
    text_meta = _extract_pdf_metadata_from_text(temp_path)

    ndoi = normalize_doi(
        filename_doi
        or extracted_doi
        or text_meta.get("doi")
        or native_meta.get("doi")
    )
    title = (
        (
            extracted_title
            or text_meta.get("title")
            or native_meta.get("title")
            or original_stem
            or ""
        ).strip()
        or "(无标题)"
    )
    authors: List[str] = (
        text_meta.get("authors")
        if isinstance(text_meta.get("authors"), list)
        else []
    )
    year = text_meta.get("year") if isinstance(text_meta.get("year"), int) else None
    venue = (text_meta.get("venue") or "").strip() if isinstance(text_meta.get("venue"), str) else ""

    cr_meta = crossref_fetch_by_doi(ndoi) if ndoi else None
    if not cr_meta and title:
        cr_meta = _crossref_lookup_by_title(title)
    if cr_meta:
        # CrossRef-first: trust DOI metadata when present; keep PDF-derived values as fallback.
        ndoi = normalize_doi(cr_meta.get("doi") or ndoi)
        cr_title = str(cr_meta.get("title") or "").strip()
        if cr_title:
            title = cr_title
        cr_authors = (
            [str(a).strip() for a in (cr_meta.get("authors") or []) if str(a).strip()]
            if isinstance(cr_meta.get("authors"), list)
            else []
        )
        if cr_authors:
            authors = cr_authors
        cr_year = cr_meta.get("year")
        if isinstance(cr_year, int):
            year = int(cr_year)
        elif isinstance(cr_year, str) and cr_year.isdigit():
            year = int(cr_year)
        cr_venue = str(cr_meta.get("venue") or "").strip()
        if cr_venue:
            venue = cr_venue

    return {
        "doi": ndoi,
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
    }


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
    from src.retrieval.dedup import normalize_doi, normalize_title
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

                resolved = _resolve_library_import_metadata(raw_name, temp_path)
                ndoi = normalize_doi(resolved.get("doi"))
                title = (resolved.get("title") or original_stem or "").strip() or "(无标题)"
                authors: List[str] = resolved.get("authors") if isinstance(resolved.get("authors"), list) else []
                year = resolved.get("year") if isinstance(resolved.get("year"), int) else None
                venue = (resolved.get("venue") or "").strip() if isinstance(resolved.get("venue"), str) else ""

                ntitle = normalize_title(title)
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
                    if authors and (existing.authors or "").strip() in ("", "[]"):
                        existing.authors = json.dumps(authors, ensure_ascii=False)
                        changed = True
                    if year is not None and existing.year is None:
                        existing.year = year
                        changed = True
                    if venue and not (existing.venue or "").strip():
                        existing.venue = venue
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
                            authors=authors or None,
                            year=year,
                            source="library_folder_import",
                        )
                    except Exception:
                        pass
                    continue

                if target_path.exists():
                    # File already on disk but no DB row matched (orphan file). Create library row and link so directory stays in sync.
                    from src.retrieval.dedup import compute_paper_uid
                    new_row = ScholarLibraryPaper(
                        library_id=lib_id,
                        title=title,
                        authors=json.dumps(authors, ensure_ascii=False) if authors else "[]",
                        year=year,
                        doi=ndoi or "",
                        pdf_url="",
                        url="",
                        source="folder_import",
                        score=0.0,
                        annas_md5="",
                        added_at=now_iso,
                        downloaded_at=now_iso,
                        venue=venue or "",
                        paper_uid=compute_paper_uid(doi=ndoi or "", title=title, authors=authors, year=year),
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
                            authors=authors or None,
                            year=year,
                            source="library_folder_import",
                        )
                    except Exception:
                        pass
                    summary["imported"] += 1
                    summary["linked_existing"] += 1
                    continue

                temp_path.rename(target_path)
                summary["imported"] += 1
                if target_path.stem != original_stem:
                    summary["renamed"] += 1

                new_row = ScholarLibraryPaper(
                    library_id=lib_id,
                    title=title,
                    authors=json.dumps(authors, ensure_ascii=False) if authors else "[]",
                    year=year,
                    doi=ndoi or "",
                    pdf_url="",
                    url="",
                    source="folder_import",
                    score=0.0,
                    annas_md5="",
                    added_at=now_iso,
                    downloaded_at=now_iso,
                    venue=venue or "",
                    paper_uid=compute_paper_uid(doi=ndoi or "", title=title, authors=authors, year=year),
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
                        authors=authors or None,
                        year=year,
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
    from src.retrieval.dedup import normalize_doi
    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        existing = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        existing_dois = {normalize_doi(p.doi or "") for p in existing if (p.doi or "").strip()}
        existing_by_doi = {normalize_doi(p.doi or ""): p for p in existing if (p.doi or "").strip()}
        existing_titles = {p.title.strip().lower() for p in existing}
        added = 0
        for item in papers_to_add:
            paper = _paper_from_search_item(item)
            paper.library_id = lib_id
            ndoi = normalize_doi(paper.doi or "") if (paper.doi or "").strip() else ""
            if ndoi and ndoi in existing_dois:
                existing_row = existing_by_doi.get(ndoi)
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
                ndoi_add = normalize_doi(paper.doi)
                if ndoi_add:
                    existing_dois.add(ndoi_add)
                    existing_by_doi[ndoi_add] = paper
            existing_titles.add(paper.title.strip().lower())
            added += 1
        session.commit()
        _mark_graph_scopes_stale_for_library(
            user_id=user_id,
            lib_id=lib_id,
            reason="scholar_library_papers_changed",
        )
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
        collection_name = getattr(paper, "collection_name", "") or ""
        paper.downloaded_at = None
        session.add(paper)
        session.commit()
    _mark_graph_scopes_stale_for_library(
        user_id=user_id,
        lib_id=lib_id,
        collection_name=collection_name,
        reason="library_paper_pdf_deleted",
    )

    # Remove the ingested Paper record and its Milvus vectors if this PDF was previously ingested
    removed_from_collection = False
    try:
        from src.indexing.paper_store import get_paper_by_library_paper_id, delete_paper
        from src.indexing.milvus_ops import milvus

        ingested = get_paper_by_library_paper_id(record_id)
        if ingested:
            col = ingested["collection"]
            pid = ingested["paper_id"]
            if col and milvus.client.has_collection(col):
                milvus.client.delete(collection_name=col, filter=f'paper_id == "{pid}"')
            delete_paper(col, pid)
            removed_from_collection = True
            logger.info("delete_library_paper_pdf: removed vectors for paper_id=%s collection=%s", pid, col)
    except Exception as e:
        logger.warning("delete_library_paper_pdf: vector cleanup failed for record_id=%s: %s", record_id, e)

    return {"ok": True, "paper_id": stem, "removed_from_collection": removed_from_collection}


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
        collection_name = getattr(paper, "collection_name", "") or ""
        session.delete(paper)
        session.commit()
        try:
            from src.indexing.assistant_artifact_store import delete_resource_annotations_for_resource
            from src.services.resource_state_service import get_resource_state_service

            delete_resource_annotations_for_resource(
                user_id=user_id,
                resource_type="scholar_library_paper",
                resource_id=str(paper_id),
            )
            get_resource_state_service().delete_resource_overlays(
                user_id=user_id,
                resource_type="scholar_library_paper",
                resource_id=str(paper_id),
            )
        except Exception as e:
            logger.warning("remove scholar library paper overlay cleanup failed paper_id=%s: %s", paper_id, e)
        _mark_graph_scopes_stale_for_library(
            user_id=user_id,
            lib_id=lib_id,
            collection_name=collection_name,
            reason="scholar_library_paper_deleted",
        )
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


@router.post("/libraries/{lib_id:int}/refresh-metadata", response_model=LibraryRefreshMetadataSummary)
def refresh_library_metadata(lib_id: int, user_id: str = Depends(get_current_user_id)):
    """Refresh title/authors/year/venue by DOI via CrossRef for all papers in a library."""
    from src.retrieval.dedup import crossref_fetch_by_doi, normalize_doi
    from src.retrieval.venue_utils import extract_clean_venue, normalize_journal_name

    updated = 0
    skipped_no_doi = 0
    failed = 0

    def _parse_crossref_authors(meta: Dict[str, Any]) -> List[str]:
        raw = meta.get("authors")
        if not isinstance(raw, list):
            return []
        return [str(a).strip() for a in raw if str(a).strip()]

    def _parse_crossref_year(meta: Dict[str, Any]) -> Optional[int]:
        y = meta.get("year")
        if isinstance(y, int):
            return int(y)
        if isinstance(y, str) and y.isdigit():
            return int(y)
        return None

    if _is_temp_library(lib_id):
        if lib_id not in _temp_store:
            raise HTTPException(status_code=404, detail="库不存在")
        papers = list(_temp_store[lib_id].get("papers") or [])
        for p in papers:
            doi = normalize_doi((p.get("doi") or "").strip())
            if not doi:
                skipped_no_doi += 1
                continue
            try:
                cr = crossref_fetch_by_doi(doi)
            except Exception:
                cr = None
            if not cr:
                failed += 1
                continue
            venue_raw = str(cr.get("venue") or "").strip()
            venue = extract_clean_venue(venue_raw) if venue_raw else ""
            p["doi"] = normalize_doi(cr.get("doi") or doi) or doi
            if cr.get("title"):
                p["title"] = str(cr.get("title")).strip() or p.get("title") or "(无标题)"
            cr_authors = _parse_crossref_authors(cr)
            if cr_authors:
                p["authors"] = cr_authors
            cr_year = _parse_crossref_year(cr)
            if cr_year is not None:
                p["year"] = cr_year
            if venue:
                p["venue"] = venue
                p["normalized_journal_name"] = normalize_journal_name(venue)
            updated += 1
        _temp_store[lib_id]["papers"] = papers
        _temp_store[lib_id]["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        return {"updated": updated, "skipped_no_doi": skipped_no_doi, "failed": failed}

    with Session(get_engine()) as session:
        lib = session.get(ScholarLibrary, lib_id)
        if not lib or getattr(lib, "user_id", None) != user_id:
            raise HTTPException(status_code=404, detail="库不存在")
        rows = list(session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == lib_id)).all())
        for row in rows:
            doi = normalize_doi((row.doi or "").strip())
            if not doi:
                skipped_no_doi += 1
                continue
            try:
                cr = crossref_fetch_by_doi(doi)
            except Exception:
                cr = None
            if not cr:
                failed += 1
                continue

            cr_doi = normalize_doi(cr.get("doi") or doi) or doi
            cr_title = str(cr.get("title") or "").strip()
            cr_authors = _parse_crossref_authors(cr)
            cr_year = _parse_crossref_year(cr)
            cr_venue_raw = str(cr.get("venue") or "").strip()
            cr_venue = extract_clean_venue(cr_venue_raw) if cr_venue_raw else ""

            row.doi = cr_doi
            if cr_title:
                row.title = cr_title
            if cr_authors:
                row.authors = json.dumps(cr_authors, ensure_ascii=False)
            if cr_year is not None:
                row.year = cr_year
            if cr_venue:
                row.venue = cr_venue
                row.normalized_journal_name = normalize_journal_name(cr_venue)
            updated += 1
        session.commit()
    return {"updated": updated, "skipped_no_doi": skipped_no_doi, "failed": failed}


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
