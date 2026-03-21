from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import Paper, ResourceAnnotation
from src.indexing.embedder import embedder
from src.indexing.milvus_ops import milvus
from src.log import get_logger
from src.parser.pdf_parser import EnrichedDoc

logger = get_logger(__name__)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def _truncate(text: Any, limit: int = 65500) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _stable_chunk_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p or "").strip() for p in parts if str(p or "").strip())
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _get_field_max_lengths(collection_name: str) -> Dict[str, int]:
    defaults = {
        "paper_id": 250,
        "chunk_id": 250,
        "domain": 60,
        "content_type": 60,
        "chunk_type": 120,
        "section_path": 4000,
    }
    if not collection_name:
        return defaults
    try:
        info = milvus.client.describe_collection(collection_name)
        for field in info.get("fields", []):
            name = field.get("name", "")
            params = field.get("params", {})
            if name in defaults and "max_length" in params:
                defaults[name] = max(16, int(params["max_length"]) - 2)
    except Exception:
        pass
    return defaults


def _delete_with_filter(collection_name: str, filters: Iterable[str]) -> int:
    if not collection_name:
        return 0
    try:
        if not milvus.client.has_collection(collection_name):
            return 0
    except Exception:
        return 0
    deleted = 0
    for flt in filters:
        try:
            result = milvus.client.delete(collection_name=collection_name, filter=flt)
            if isinstance(result, dict):
                deleted += int(result.get("delete_count", 0) or 0)
        except Exception as exc:
            logger.debug("milvus delete skipped collection=%s filter=%s err=%s", collection_name, flt, exc)
    return deleted


def _embed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    texts = [row.pop("_text_for_embed") for row in rows]
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        emb = embedder.encode(batch)
        dense = emb["dense"]
        sparse = emb["sparse"]
        for offset, idx in enumerate(range(start, min(start + batch_size, len(rows)))):
            rows[idx]["dense_vector"] = dense[offset].tolist()
            sp = sparse._getrow(offset).tocoo()
            rows[idx]["sparse_vector"] = {
                int(col): float(val)
                for col, val in zip(sp.col, sp.data)
            }
    return rows


def _row_base(
    *,
    collection_name: str,
    paper_id: str,
    chunk_id: str,
    text: str,
    content_type: str,
    chunk_type: str,
    section_path: str,
    page: int,
    doc_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    limits = _get_field_max_lengths(collection_name)
    row = {
        "paper_id": str(paper_id)[: limits["paper_id"]],
        "chunk_id": str(chunk_id)[: limits["chunk_id"]],
        "content": _truncate(text),
        "raw_content": _truncate(text),
        "domain": "global"[: limits["domain"]],
        "content_type": str(content_type)[: limits["content_type"]],
        "chunk_type": str(chunk_type)[: limits["chunk_type"]],
        "section_path": _truncate(section_path, limits["section_path"]),
        "page": int(page if isinstance(page, int) else 0),
        "_text_for_embed": _truncate(text),
    }
    meta = doc_metadata or {}
    if meta.get("doi"):
        row["doi"] = str(meta["doi"]).strip()
    if meta.get("title"):
        row["doc_title"] = str(meta["title"]).strip()
    if meta.get("paper_uid"):
        row["paper_uid"] = str(meta["paper_uid"]).strip()
    return row


def _figure_analysis_text(block: Any) -> str:
    figure_data = getattr(block, "figure_data", None)
    enrichment = getattr(block, "enrichment", None)
    interpretation = getattr(enrichment, "interpretation", None) if enrichment is not None else None
    parts: List[str] = []
    caption = getattr(figure_data, "caption", None)
    if caption:
        parts.append(f"Caption: {caption}")
    if interpretation is not None:
        if getattr(interpretation, "figure_type", None):
            parts.append(f"Figure type: {interpretation.figure_type}")
        if getattr(interpretation, "description", None):
            parts.append(f"Description: {interpretation.description}")
        if getattr(interpretation, "key_findings", None):
            parts.append("Key findings: " + "; ".join(str(item) for item in interpretation.key_findings if str(item).strip()))
        if getattr(interpretation, "evidence", None):
            parts.append("Evidence: " + "; ".join(str(item) for item in interpretation.evidence if str(item).strip()))
    ocr_text = getattr(figure_data, "ocr_text", None)
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    return "\n".join(part for part in parts if part.strip()).strip()


def build_media_vector_rows(
    collection_name: str,
    *,
    doc: EnrichedDoc,
    paper_id: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = doc_metadata or {}
    for idx, block in enumerate(getattr(doc, "content_flow", []) or []):
        figure_data = getattr(block, "figure_data", None)
        if figure_data is None:
            continue
        section_path = " > ".join(getattr(block, "heading_path", []) or [])
        page = int(getattr(block, "page_index", 0) or 0)
        block_id = getattr(block, "block_id", "") or f"figure-{idx}"
        if getattr(figure_data, "caption", None):
            caption_text = str(figure_data.caption).strip()
            if caption_text:
                row = _row_base(
                    collection_name=collection_name,
                    paper_id=paper_id,
                    chunk_id=_stable_chunk_id("media-caption", meta.get("paper_uid"), block_id, "caption"),
                    text=caption_text,
                    content_type="image_caption",
                    chunk_type="figure",
                    section_path=section_path,
                    page=page,
                    doc_metadata=meta,
                )
                row["figure_id"] = block_id
                row["image_path"] = getattr(figure_data, "image_path", "") or ""
                row["bbox"] = list(getattr(block, "bbox", []) or [])
                rows.append(row)
        analysis_text = _figure_analysis_text(block)
        if analysis_text:
            row = _row_base(
                collection_name=collection_name,
                paper_id=paper_id,
                chunk_id=_stable_chunk_id("media-analysis", meta.get("paper_uid"), block_id, "analysis"),
                text=analysis_text,
                content_type="image_analysis",
                chunk_type="figure",
                section_path=section_path,
                page=page,
                doc_metadata=meta,
            )
            row["figure_id"] = block_id
            row["image_path"] = getattr(figure_data, "image_path", "") or ""
            row["bbox"] = list(getattr(block, "bbox", []) or [])
            rows.append(row)
    return rows


def upsert_media_vectors(
    collection_name: str,
    *,
    doc: EnrichedDoc,
    paper_id: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not collection_name:
        return {"deleted_count": 0, "upserted_count": 0}
    deleted_count = _delete_with_filter(
        collection_name,
        [
            f'paper_id == "{paper_id}" and content_type == "image_caption"',
            f'paper_id == "{paper_id}" and content_type == "image_analysis"',
        ],
    )
    rows = build_media_vector_rows(
        collection_name,
        doc=doc,
        paper_id=paper_id,
        doc_metadata=doc_metadata,
    )
    if not rows:
        return {"deleted_count": deleted_count, "upserted_count": 0}
    _embed_rows(rows)
    milvus.upsert(collection_name, rows)
    return {"deleted_count": deleted_count, "upserted_count": len(rows)}


def _resolve_paper_for_annotation(
    *,
    user_id: str,
    paper_uid: str,
    collection: Optional[str] = None,
) -> Optional[Paper]:
    if not paper_uid:
        return None
    with Session(get_engine()) as session:
        stmt = select(Paper).where(
            Paper.user_id == user_id,
            Paper.paper_uid == paper_uid,
        )
        if collection:
            stmt = stmt.where(Paper.collection == collection)
        stmt = stmt.order_by(Paper.created_at.desc())
        return session.exec(stmt).first()


def _annotation_text(row: ResourceAnnotation) -> str:
    locator = row.get_target_locator()
    parts: List[str] = []
    if row.target_text:
        parts.append(f"Target: {row.target_text}")
    if row.directive:
        parts.append(f"Annotation: {row.directive}")
    if locator:
        if locator.get("figure_id"):
            parts.append(f"Figure: {locator['figure_id']}")
        if locator.get("page") is not None:
            parts.append(f"Page: {locator['page']}")
    return "\n".join(part for part in parts if str(part).strip()).strip()


def build_annotation_vector_row(
    collection_name: str,
    *,
    annotation: ResourceAnnotation,
    paper: Paper,
) -> Optional[Dict[str, Any]]:
    if not collection_name:
        return None
    text = _annotation_text(annotation)
    if not text:
        return None
    locator = annotation.get_target_locator()
    page = locator.get("page")
    page_val = int(page) if isinstance(page, int) else 0
    row = _row_base(
        collection_name=collection_name,
        paper_id=paper.paper_id,
        chunk_id=_stable_chunk_id("annotation", annotation.id, annotation.paper_uid, annotation.resource_type),
        text=text,
        content_type="annotation",
        chunk_type=str(annotation.target_kind or "annotation"),
        section_path=str(locator.get("section_path") or ""),
        page=page_val,
        doc_metadata={"paper_uid": annotation.paper_uid},
    )
    row["annotation_id"] = int(annotation.id or 0)
    row["resource_type"] = annotation.resource_type
    row["resource_id"] = annotation.resource_id
    row["target_kind"] = annotation.target_kind
    row["target_locator_json"] = annotation.target_locator_json
    if "bbox" in locator:
        row["bbox"] = locator.get("bbox")
    if "figure_id" in locator:
        row["figure_id"] = locator.get("figure_id")
    return row


def sync_annotation_vector(
    annotation: ResourceAnnotation,
    *,
    collection: Optional[str] = None,
) -> bool:
    if not annotation.paper_uid or annotation.status in {"deleted", "archived"}:
        return False
    paper = _resolve_paper_for_annotation(
        user_id=annotation.user_id,
        paper_uid=annotation.paper_uid,
        collection=collection,
    )
    if paper is None:
        return False
    row = build_annotation_vector_row(
        collection or paper.collection,
        annotation=annotation,
        paper=paper,
    )
    if row is None:
        return False
    _embed_rows([row])
    milvus.upsert(collection or paper.collection, [row])
    return True


def delete_annotation_vectors(
    collection_name: str,
    annotation_ids: Iterable[int],
) -> int:
    return _delete_with_filter(
        collection_name,
        [f'annotation_id == {int(annotation_id)}' for annotation_id in annotation_ids],
    )


def list_resource_annotations(
    *,
    user_id: str,
    paper_uid: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    target_kind: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[ResourceAnnotation]:
    with Session(get_engine()) as session:
        stmt = select(ResourceAnnotation).where(ResourceAnnotation.user_id == user_id)
        if paper_uid:
            stmt = stmt.where(ResourceAnnotation.paper_uid == paper_uid)
        if resource_type:
            stmt = stmt.where(ResourceAnnotation.resource_type == resource_type)
        if resource_id:
            stmt = stmt.where(ResourceAnnotation.resource_id == resource_id)
        if target_kind:
            stmt = stmt.where(ResourceAnnotation.target_kind == target_kind)
        if status:
            stmt = stmt.where(ResourceAnnotation.status == status)
        stmt = stmt.order_by(ResourceAnnotation.updated_at.desc()).offset(max(0, offset)).limit(max(1, min(limit, 200)))
        return list(session.exec(stmt).all())


def get_resource_annotation(annotation_id: int, *, user_id: str) -> Optional[ResourceAnnotation]:
    with Session(get_engine()) as session:
        row = session.get(ResourceAnnotation, annotation_id)
        if row is None or row.user_id != user_id:
            return None
        return row


def upsert_resource_annotation(
    *,
    user_id: str,
    resource_type: str,
    resource_id: str,
    paper_uid: str = "",
    target_kind: str = "chunk",
    target_locator: Optional[Dict[str, Any]] = None,
    target_text: str = "",
    directive: str = "",
    status: str = "active",
    annotation_id: Optional[int] = None,
    collection: Optional[str] = None,
) -> ResourceAnnotation:
    old_row: Optional[ResourceAnnotation] = None
    with Session(get_engine()) as session:
        row: Optional[ResourceAnnotation] = None
        if annotation_id is not None:
            row = session.get(ResourceAnnotation, annotation_id)
            if row is not None and row.user_id != user_id:
                raise ValueError("annotation does not belong to user")
        if row is None:
            row = ResourceAnnotation(
                user_id=user_id,
                resource_type=resource_type.strip(),
                resource_id=str(resource_id).strip(),
                paper_uid=(paper_uid or "").strip(),
                target_kind=(target_kind or "chunk").strip(),
                target_locator_json=_safe_json_dumps(target_locator),
                target_text=target_text or "",
                directive=directive or "",
                status=(status or "active").strip(),
                created_at=_now_iso(),
                updated_at=_now_iso(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
        else:
            old_row = ResourceAnnotation.model_validate(row.model_dump())
            row.resource_type = resource_type.strip()
            row.resource_id = str(resource_id).strip()
            row.paper_uid = (paper_uid or "").strip()
            row.target_kind = (target_kind or "chunk").strip()
            row.target_locator_json = _safe_json_dumps(target_locator)
            row.target_text = target_text or ""
            row.directive = directive or ""
            row.status = (status or "active").strip()
            row.updated_at = _now_iso()
            session.add(row)
            session.commit()
            session.refresh(row)
        saved = ResourceAnnotation.model_validate(row.model_dump())

    if old_row is not None and old_row.paper_uid and old_row.id is not None:
        old_paper = _resolve_paper_for_annotation(
            user_id=user_id,
            paper_uid=old_row.paper_uid,
            collection=collection,
        )
        if old_paper is not None:
            delete_annotation_vectors(old_paper.collection, [int(old_row.id)])

    if saved.id is not None:
        if saved.status in {"deleted", "archived"}:
            paper = _resolve_paper_for_annotation(
                user_id=user_id,
                paper_uid=saved.paper_uid,
                collection=collection,
            )
            if paper is not None:
                delete_annotation_vectors(paper.collection, [int(saved.id)])
            if saved.status == "deleted":
                try:
                    from src.services.resource_state_service import get_resource_state_service

                    get_resource_state_service().delete_resource_overlays(
                        user_id=user_id,
                        resource_type="resource_annotation",
                        resource_id=str(saved.id),
                        include_state=False,
                    )
                except Exception as exc:
                    logger.warning("resource state cleanup failed for annotation=%s: %s", saved.id, exc)
        else:
            sync_annotation_vector(saved, collection=collection)
    return saved


def delete_resource_annotations_for_resource(
    *,
    user_id: str,
    resource_type: str,
    resource_id: str,
) -> int:
    with Session(get_engine()) as session:
        rows = list(
            session.exec(
                select(ResourceAnnotation).where(
                    ResourceAnnotation.user_id == user_id,
                    ResourceAnnotation.resource_type == resource_type,
                    ResourceAnnotation.resource_id == str(resource_id),
                )
            ).all()
        )
        for row in rows:
            session.delete(row)
        session.commit()
    deleted = 0
    by_collection: Dict[str, List[int]] = {}
    for row in rows:
        if row.id is None:
            continue
        try:
            from src.services.resource_state_service import get_resource_state_service

            get_resource_state_service().delete_resource_overlays(
                user_id=user_id,
                resource_type="resource_annotation",
                resource_id=str(int(row.id)),
                include_state=False,
            )
        except Exception as exc:
            logger.warning("resource state cleanup failed for annotation=%s: %s", row.id, exc)
        paper = _resolve_paper_for_annotation(user_id=user_id, paper_uid=row.paper_uid)
        if paper is None:
            continue
        by_collection.setdefault(paper.collection, []).append(int(row.id))
    for collection_name, annotation_ids in by_collection.items():
        deleted += delete_annotation_vectors(collection_name, annotation_ids)
    return deleted


def cleanup_assistant_artifacts_for_paper(
    *,
    user_id: str,
    collection: str,
    paper_id: str,
    paper_uid: str = "",
) -> Dict[str, int]:
    deleted_vectors = _delete_with_filter(
        collection,
        [
            f'paper_id == "{paper_id}" and content_type == "image_caption"',
            f'paper_id == "{paper_id}" and content_type == "image_analysis"',
            f'paper_id == "{paper_id}" and content_type == "annotation"',
        ],
    )
    with Session(get_engine()) as session:
        stmt = select(ResourceAnnotation).where(ResourceAnnotation.user_id == user_id)
        if paper_uid:
            stmt = stmt.where(ResourceAnnotation.paper_uid == paper_uid)
        else:
            stmt = stmt.where(
                ResourceAnnotation.resource_type == "paper",
                ResourceAnnotation.resource_id == paper_id,
            )
        rows = list(session.exec(stmt).all())
        deleted_annotations = len(rows)
        for row in rows:
            session.delete(row)
        session.commit()
    for row in rows:
        if row.id is None:
            continue
        try:
            from src.services.resource_state_service import get_resource_state_service

            get_resource_state_service().delete_resource_overlays(
                user_id=user_id,
                resource_type="resource_annotation",
                resource_id=str(int(row.id)),
                include_state=False,
            )
        except Exception as exc:
            logger.warning("resource state cleanup failed for annotation=%s: %s", row.id, exc)
    return {
        "deleted_vectors": deleted_vectors,
        "deleted_annotations": deleted_annotations,
    }
