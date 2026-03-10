from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import CollectionLibraryBinding, Paper, ScholarLibrary, ScholarLibraryPaper
from src.log import get_logger
from src.retrieval.dedup import extract_doi_from_pdf_tiered, normalize_doi, normalize_title
from src.retrieval.downloader.adapter import _doi_to_paper_id, _normalize_to_paper_id
from src.utils.path_manager import PathManager

logger = get_logger(__name__)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _resolve_library_folder(user_id: str, library_name: str) -> Path:
    base = PathManager.get_user_library_path(user_id, library_name)
    pdfs = base / "pdfs"
    base.mkdir(parents=True, exist_ok=True)
    pdfs.mkdir(parents=True, exist_ok=True)
    return base


def _find_existing_library_for_collection(session: Session, user_id: str, collection_name: str) -> Optional[ScholarLibrary]:
    return session.exec(
        select(ScholarLibrary).where(
            ScholarLibrary.user_id == user_id,
            ScholarLibrary.name == collection_name,
        )
    ).first()


def get_collection_binding(
    session: Session, user_id: str, collection_name: str
) -> Tuple[Optional[CollectionLibraryBinding], Optional[ScholarLibrary]]:
    binding = session.exec(
        select(CollectionLibraryBinding).where(
            CollectionLibraryBinding.user_id == user_id,
            CollectionLibraryBinding.collection_name == collection_name,
        )
    ).first()
    if not binding:
        return None, None
    lib = session.get(ScholarLibrary, binding.library_id)
    if not lib or lib.user_id != user_id:
        return binding, None
    return binding, lib


def ensure_collection_binding(
    session: Session,
    user_id: str,
    collection_name: str,
) -> Tuple[CollectionLibraryBinding, ScholarLibrary, bool, bool]:
    """
    Ensure the collection has a permanent bound scholar library.
    Returns: (binding, library, binding_created, library_created)
    """
    collection_name = (collection_name or "").strip()
    if not collection_name:
        raise ValueError("collection_name is empty")

    existing_binding, existing_lib = get_collection_binding(session, user_id, collection_name)
    if existing_binding and existing_lib:
        return existing_binding, existing_lib, False, False

    library_created = False
    if existing_lib is None:
        existing_lib = _find_existing_library_for_collection(session, user_id, collection_name)
    if existing_lib is None:
        folder = _resolve_library_folder(user_id, collection_name)
        existing_lib = ScholarLibrary(
            user_id=user_id,
            name=collection_name,
            description=f"Auto-bound library for collection '{collection_name}'",
            folder_path=str(folder.resolve()),
        )
        session.add(existing_lib)
        session.flush()
        library_created = True

    if existing_binding is None:
        existing_binding = CollectionLibraryBinding(
            user_id=user_id,
            collection_name=collection_name,
            library_id=int(existing_lib.id),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        session.add(existing_binding)
        session.flush()
        return existing_binding, existing_lib, True, library_created

    existing_binding.library_id = int(existing_lib.id)
    existing_binding.updated_at = _now_iso()
    session.add(existing_binding)
    session.flush()
    return existing_binding, existing_lib, False, library_created


def resolve_bound_library_for_collection(
    user_id: str, collection_name: Optional[str], auto_create: bool = False
) -> Optional[ScholarLibrary]:
    collection = (collection_name or "").strip()
    if not collection:
        return None
    with Session(get_engine()) as session:
        binding, lib = get_collection_binding(session, user_id, collection)
        if lib:
            return lib
        if auto_create:
            _, created_lib, _, _ = ensure_collection_binding(session, user_id, collection)
            session.commit()
            return created_lib
        if binding and not lib:
            logger.warning(
                "collection binding points to missing library: user=%s collection=%s library_id=%s",
                user_id,
                collection,
                binding.library_id,
            )
        return None


def delete_collection_binding(user_id: str, collection_name: str, delete_library: bool = True) -> Dict[str, Any]:
    """Delete binding and optionally its bound permanent library."""
    with Session(get_engine()) as session:
        binding, lib = get_collection_binding(session, user_id, collection_name)
        if not binding:
            return {"deleted_binding": False, "deleted_library": False}
        deleted_library = False
        if delete_library and lib is not None:
            session.delete(lib)
            deleted_library = True
        session.delete(binding)
        session.commit()
        return {"deleted_binding": True, "deleted_library": deleted_library}


def _library_paper_id(doi: Optional[str], title: str, authors: List[str], year: Optional[int]) -> Optional[str]:
    if doi and doi.strip():
        return _doi_to_paper_id(doi)
    if not (title or (authors and authors[0])):
        return None
    return _normalize_to_paper_id(title or "", list(authors) if authors else [], year)


def repair_collection_library_links(
    user_id: str,
    collection_name: str,
    auto_create_binding: bool = True,
    max_scan: int = 200,
) -> Dict[str, Any]:
    """
    Repair links between collection papers and bound library records by DOI/title.
    Also ensures expected library PDF filenames exist by copying collection PDFs.
    """
    collection_name = (collection_name or "").strip()
    if not collection_name:
        raise ValueError("collection_name is empty")

    with Session(get_engine()) as session:
        binding, lib = get_collection_binding(session, user_id, collection_name)
        if (binding is None or lib is None) and auto_create_binding:
            binding, lib, _b_created, _l_created = ensure_collection_binding(session, user_id, collection_name)
            session.commit()
        if not binding or not lib:
            return {
                "ok": False,
                "collection": collection_name,
                "reason": "binding_missing",
            }

        rows = list(
            session.exec(
                select(Paper).where(
                    Paper.user_id == user_id,
                    Paper.collection == collection_name,
                )
            ).all()
        )
        rows = rows[: max(1, int(max_scan))]

        lib_papers = list(
            session.exec(
                select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == int(lib.id))
            ).all()
        )
        doi_map: Dict[str, List[ScholarLibraryPaper]] = {}
        title_map: Dict[str, List[ScholarLibraryPaper]] = {}
        for p in lib_papers:
            d = normalize_doi(p.doi or "")
            if d:
                doi_map.setdefault(d, []).append(p)
            nt = normalize_title(p.title or "")
            if nt:
                title_map.setdefault(nt, []).append(p)

        created = matched_doi = matched_title = copied_pdf = conflict = 0
        skipped = 0
        now = _now_iso()
        pdf_root = Path(lib.folder_path or "") / "pdfs" if (lib.folder_path or "").strip() else None
        if pdf_root:
            pdf_root.mkdir(parents=True, exist_ok=True)

        for row in rows:
            source_pdf = Path(row.file_path or "")
            if not source_pdf.exists():
                source_pdf = PathManager.get_user_raw_papers_path(user_id) / f"{row.paper_id}.pdf"
            if not source_pdf.exists():
                skipped += 1
                continue

            doi, extracted_title = extract_doi_from_pdf_tiered(source_pdf)
            normalized_doi = normalize_doi(doi)
            inferred_title = (extracted_title or row.filename or row.paper_id or "").strip()
            normalized_name = normalize_title(inferred_title)

            target: Optional[ScholarLibraryPaper] = None
            if normalized_doi:
                candidates = doi_map.get(normalized_doi, [])
                if len(candidates) == 1:
                    target = candidates[0]
                    matched_doi += 1
                elif len(candidates) > 1:
                    conflict += 1
                    continue
            if target is None and normalized_name:
                candidates = title_map.get(normalized_name, [])
                if len(candidates) == 1:
                    target = candidates[0]
                    matched_title += 1
                elif len(candidates) > 1:
                    conflict += 1
                    continue

            if target is None:
                target = ScholarLibraryPaper(
                    library_id=int(lib.id),
                    title=inferred_title or row.paper_id,
                    authors=json.dumps([], ensure_ascii=False),
                    year=None,
                    doi=(normalized_doi or ""),
                    pdf_url="",
                    url="",
                    source="ingest_repair",
                    score=0.0,
                    annas_md5="",
                    added_at=now,
                    downloaded_at=now,
                )
                session.add(target)
                session.flush()
                created += 1
                if normalized_doi:
                    doi_map.setdefault(normalized_doi, []).append(target)
                if normalized_name:
                    title_map.setdefault(normalized_name, []).append(target)
            elif not target.downloaded_at:
                target.downloaded_at = now
                session.add(target)

            if getattr(target, "id", None) is not None:
                row.library_id = int(lib.id)
                row.library_paper_id = int(target.id)
                if not (row.source or "").strip():
                    row.source = "ingest_repair"
                target.collection_name = collection_name
                target.collection_paper_id = row.paper_id
                session.add(row)
                session.add(target)

            if pdf_root:
                paper_id = _library_paper_id(
                    (target.doi or "").strip() or None,
                    target.title or "",
                    target.get_authors(),
                    target.year,
                )
                if paper_id:
                    destination = pdf_root / f"{paper_id}.pdf"
                    if not destination.exists():
                        try:
                            shutil.copy2(source_pdf, destination)
                            copied_pdf += 1
                        except OSError:
                            logger.warning("repair copy pdf failed src=%s dst=%s", source_pdf, destination)

        session.commit()
        return {
            "ok": True,
            "collection": collection_name,
            "library_id": int(lib.id),
            "library_name": lib.name,
            "scanned": len(rows),
            "matched_by_doi": matched_doi,
            "matched_by_title": matched_title,
            "created_library_records": created,
            "copied_pdfs": copied_pdf,
            "conflicts": conflict,
            "skipped": skipped,
        }
