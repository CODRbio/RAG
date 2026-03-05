#!/usr/bin/env python
"""
Incremental ingest CLI: scan PDFs, deduplicate by DOI (primary) and content_hash (secondary),
then run the existing ingest pipeline only on new or changed files.

Usage:
  conda run -n deepsea-rag python scripts/incremental_ingest.py [--collection NAME] [--dir DIR] [--dry-run]
  conda run -n deepsea-rag python scripts/incremental_ingest.py --files a.pdf b.pdf

Requires: target collection already exists. Run with project root on sys.path.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Project root on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incremental ingest: add only new/changed PDFs to a collection (DOI + hash dedup).",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Collection name (default: from settings.collection.global_)",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Directory to scan for *.pdf (default: settings.path.raw_papers)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit PDF paths (overrides --dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which files would be ingested; do not run ingest",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        default=True,
        help="Skip LLM enrichment (default: True)",
    )
    parser.add_argument(
        "--no-skip-enrichment",
        action="store_false",
        dest="skip_enrichment",
        help="Enable LLM enrichment (tables/figures)",
    )
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="Skip files already in collection with same content_hash (secondary dedup)",
    )
    parser.add_argument(
        "--reingest-if-doi-changed",
        action="store_true",
        help="When DOI matches but file hash differs, still ingest (updated PDF)",
    )
    args = parser.parse_args()

    collection_name = (args.collection or "").strip() or settings.collection.global_
    if args.files is not None and len(args.files) > 0:
        candidate_paths = [Path(p).resolve() for p in args.files]
        for p in candidate_paths:
            if not p.exists():
                logger.error("File not found: %s", p)
                return 1
            if p.suffix.lower() != ".pdf":
                logger.warning("Skipping non-PDF: %s", p)
        candidate_paths = [p for p in candidate_paths if p.exists() and p.suffix.lower() == ".pdf"]
    else:
        raw_dir = Path(args.dir or str(settings.path.raw_papers))
        if not raw_dir.exists():
            logger.error("Directory not found: %s", raw_dir)
            return 1
        candidate_paths = sorted(raw_dir.glob("*.pdf"))

    if not candidate_paths:
        logger.info("No candidate PDFs found.")
        return 0

    # Load existing state for this collection
    from src.indexing.paper_store import list_papers
    from src.indexing.paper_metadata_store import paper_meta_store

    papers_in_collection = list_papers(collection_name)
    existing_paper_ids = {p["paper_id"] for p in papers_in_collection}
    existing_hashes: dict[str, str] = {}
    for p in papers_in_collection:
        pid = p.get("paper_id") or ""
        if pid:
            existing_hashes[pid] = (p.get("content_hash") or "") or ""

    from src.retrieval.dedup import normalize_doi as _normalize_doi

    existing_dois: set[str] = set()
    for pid in existing_paper_ids:
        meta = paper_meta_store.get(pid)
        if meta and meta.get("doi"):
            nd = _normalize_doi(meta["doi"])
            if nd:
                existing_dois.add(nd)

    # Check collection exists
    try:
        from src.indexing.milvus_ops import milvus
        if not milvus.client.has_collection(collection_name):
            logger.error("Collection '%s' does not exist. Create it first (e.g. via API).", collection_name)
            return 1
    except Exception as e:
        logger.error("Could not check collection: %s", e)
        return 1

    to_ingest: list[Path] = []
    content_hashes: dict[str, str] = {}

    for pdf_path in candidate_paths:
        paper_id = pdf_path.stem
        try:
            file_hash = _file_hash(pdf_path)
        except Exception as e:
            logger.warning("Could not hash %s: %s", pdf_path, e)
            file_hash = ""

        content_hashes[str(pdf_path)] = file_hash

        # DOI: fast path from paper_meta_store (e.g. written by adapter at download time)
        from src.retrieval.dedup import extract_doi_from_pdf_tiered
        candidate_doi: str | None = None
        meta = paper_meta_store.get(paper_id)
        if meta and meta.get("doi"):
            candidate_doi = meta["doi"]
        if not candidate_doi:
            candidate_doi, _ = extract_doi_from_pdf_tiered(pdf_path)

        norm_doi = _normalize_doi(candidate_doi) if candidate_doi else ""

        # Primary dedup: DOI already in collection
        if norm_doi and norm_doi in existing_dois:
            if not args.reingest_if_doi_changed:
                logger.debug("Skip (DOI in collection): %s", pdf_path.name)
                continue
            # reingest-if-doi-changed: only skip if hash also matches
            stored_hash = existing_hashes.get(paper_id) or ""
            if stored_hash and stored_hash == file_hash:
                logger.debug("Skip (DOI + same hash): %s", pdf_path.name)
                continue

        # Secondary dedup: same paper_id and same content_hash
        if paper_id in existing_paper_ids and args.skip_unchanged:
            stored_hash = existing_hashes.get(paper_id) or ""
            if stored_hash and stored_hash == file_hash:
                logger.debug("Skip (unchanged): %s", pdf_path.name)
                continue

        to_ingest.append(pdf_path)

    if not to_ingest:
        logger.info("No new or changed files to ingest.")
        return 0

    if args.dry_run:
        logger.info("Dry-run: would ingest %d file(s):", len(to_ingest))
        for p in to_ingest:
            logger.info("  %s", p)
        return 0

    file_paths = [str(p) for p in to_ingest]
    content_hashes_cfg = {p: content_hashes.get(p, "") for p in file_paths}

    skip_enrichment = bool(args.skip_enrichment)
    enrich_tables = not skip_enrichment
    enrich_figures = not skip_enrichment
    actual_skip = skip_enrichment and (not enrich_tables) and (not enrich_figures)

    cfg = {
        "file_paths": file_paths,
        "collection_name": collection_name,
        "content_hashes": content_hashes_cfg,
        "enrich_tables": enrich_tables,
        "enrich_figures": enrich_figures,
        "actual_skip": actual_skip,
        "llm_text_provider": None,
        "llm_text_model": None,
        "llm_text_concurrency": None,
        "llm_vision_provider": None,
        "llm_vision_model": None,
        "llm_vision_concurrency": None,
    }

    from src.indexing.ingest_job_store import create_job
    from src.api.routes_ingest import _run_ingest_job_safe

    job = create_job(collection_name, cfg, total_files=len(to_ingest))
    job_id = job.get("job_id") or ""
    if not job_id:
        logger.error("Failed to create ingest job")
        return 1

    logger.info("Starting ingest job %s for %d file(s)...", job_id, len(to_ingest))
    _run_ingest_job_safe(job_id, cfg)
    logger.info("Ingest job %s finished.", job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
