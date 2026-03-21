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
from src.retrieval.dedup import compute_paper_uid, extract_doi_from_pdf_tiered, normalize_doi, normalize_title
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


def _decode_extra(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return {}


def _compute_row_paper_uid(
    doi: Optional[str],
    title: Optional[str],
    authors: Optional[List[str]],
    year: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    extra = extra or {}
    url = (extra.get("url") or extra.get("pdf_url") or "").strip() or None
    arxiv_id = (extra.get("arxiv_id") or "").strip()
    if not url and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    pmid = (extra.get("pmid") or "").strip() or None
    return compute_paper_uid(
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        url=url,
        pmid=pmid,
    )


def _mark_collection_binding_scopes_stale(user_id: str, collection_name: str, library_id: Optional[int], reason: str) -> None:
    try:
        from src.services.global_graph_service import mark_graph_scope_stale

        mark_graph_scope_stale(
            user_id=user_id,
            scope_type="collection",
            scope_key=collection_name,
            reason=reason,
        )
        if library_id is not None:
            mark_graph_scope_stale(
                user_id=user_id,
                scope_type="library",
                scope_key=str(int(library_id)),
                reason=reason,
            )
    except Exception as e:
        logger.debug(
            "mark collection binding scopes stale failed user=%s collection=%s library_id=%s err=%s",
            user_id,
            collection_name,
            library_id,
            e,
        )


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
        _mark_collection_binding_scopes_stale(
            user_id=user_id,
            collection_name=collection_name,
            library_id=int(existing_lib.id),
            reason="collection_binding_created",
        )
        return existing_binding, existing_lib, True, library_created

    existing_binding.library_id = int(existing_lib.id)
    existing_binding.updated_at = _now_iso()
    session.add(existing_binding)
    session.flush()
    _mark_collection_binding_scopes_stale(
        user_id=user_id,
        collection_name=collection_name,
        library_id=int(existing_lib.id),
        reason="collection_binding_updated",
    )
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
        try:
            from src.indexing.paper_metadata_store import paper_meta_store
        except Exception:
            paper_meta_store = None

        binding, lib = get_collection_binding(session, user_id, collection_name)
        if not binding:
            return {"deleted_binding": False, "deleted_library": False}
        library_id = int(lib.id) if lib is not None and getattr(lib, "id", None) is not None else binding.library_id
        library_paper_ids = []
        if delete_library and lib is not None:
            library_paper_ids = [
                int(row.id)
                for row in session.exec(select(ScholarLibraryPaper).where(ScholarLibraryPaper.library_id == int(lib.id))).all()
                if getattr(row, "id", None) is not None
            ]
        deleted_library = False
        if delete_library and lib is not None:
            session.delete(lib)
            deleted_library = True
        session.delete(binding)
        session.commit()
        if library_paper_ids:
            try:
                from src.indexing.assistant_artifact_store import delete_resource_annotations_for_resource
                from src.services.resource_state_service import get_resource_state_service

                resource_state_service = get_resource_state_service()
                for paper_id in library_paper_ids:
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
                logger.warning(
                    "delete collection binding overlay cleanup failed user=%s collection=%s err=%s",
                    user_id,
                    collection_name,
                    e,
                )
        _mark_collection_binding_scopes_stale(
            user_id=user_id,
            collection_name=collection_name,
            library_id=library_id,
            reason="collection_binding_deleted",
        )
        return {"deleted_binding": True, "deleted_library": deleted_library}


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
        uid_map: Dict[str, List[ScholarLibraryPaper]] = {}
        doi_map: Dict[str, List[ScholarLibraryPaper]] = {}
        title_map: Dict[str, List[ScholarLibraryPaper]] = {}
        for p in lib_papers:
            p_uid = (p.paper_uid or "").strip()
            if not p_uid:
                p_uid = _compute_row_paper_uid(
                    doi=(p.doi or "").strip() or None,
                    title=(p.title or "").strip() or None,
                    authors=p.get_authors(),
                    year=p.year,
                )
                if p_uid:
                    p.paper_uid = p_uid
                    session.add(p)
            if p_uid:
                uid_map.setdefault(p_uid, []).append(p)
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

            meta = paper_meta_store.get(row.paper_id) if paper_meta_store else None
            meta_extra = _decode_extra((meta or {}).get("extra_raw"))
            doi, extracted_title = extract_doi_from_pdf_tiered(source_pdf)
            effective_doi = (meta or {}).get("doi") or doi or ""
            normalized_doi = normalize_doi(effective_doi)
            inferred_title = ((meta or {}).get("title") or extracted_title or row.filename or row.paper_id or "").strip()
            inferred_authors = (meta or {}).get("authors") or []
            inferred_year = (meta or {}).get("year")
            paper_uid = (getattr(row, "paper_uid", None) or "").strip() or _compute_row_paper_uid(
                doi=effective_doi or None,
                title=inferred_title or None,
                authors=inferred_authors,
                year=inferred_year,
                extra=meta_extra,
            )
            if paper_uid:
                row.paper_uid = paper_uid
            normalized_name = normalize_title(inferred_title)

            target: Optional[ScholarLibraryPaper] = None
            if paper_uid:
                candidates = uid_map.get(paper_uid, [])
                if len(candidates) == 1:
                    target = candidates[0]
                elif len(candidates) > 1:
                    conflict += 1
                    continue
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
                    authors=json.dumps(inferred_authors, ensure_ascii=False) if inferred_authors else "[]",
                    year=inferred_year,
                    doi=(normalized_doi or ""),
                    pdf_url="",
                    url=(meta_extra.get("url") or ""),
                    source="ingest_repair",
                    score=0.0,
                    annas_md5="",
                    added_at=now,
                    downloaded_at=now,
                    paper_uid=paper_uid,
                )
                session.add(target)
                session.flush()
                created += 1
                if paper_uid:
                    uid_map.setdefault(paper_uid, []).append(target)
                if normalized_doi:
                    doi_map.setdefault(normalized_doi, []).append(target)
                if normalized_name:
                    title_map.setdefault(normalized_name, []).append(target)
            elif not target.downloaded_at:
                target.downloaded_at = now
                if paper_uid and (target.paper_uid or "").strip() != paper_uid:
                    target.paper_uid = paper_uid
                session.add(target)

            if getattr(target, "id", None) is not None:
                row.library_id = int(lib.id)
                row.library_paper_id = int(target.id)
                if not (row.source or "").strip():
                    row.source = "ingest_repair"
                target.collection_name = collection_name
                target.collection_paper_id = row.paper_id
                if paper_uid and (target.paper_uid or "").strip() != paper_uid:
                    target.paper_uid = paper_uid
                session.add(row)
                session.add(target)

            if pdf_root:
                _tdoi = (target.doi or "").strip() or None
                _tauthors = target.get_authors()
                paper_id = (
                    _doi_to_paper_id(_tdoi)
                    if _tdoi
                    else _normalize_to_paper_id(target.title or "", _tauthors, target.year)
                    if (target.title or (_tauthors and _tauthors[0]))
                    else None
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
