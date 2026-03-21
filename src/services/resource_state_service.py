from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import (
    Canvas,
    Paper,
    ResourceAnnotation,
    ResourceNote,
    ResourceTag,
    ResourceUserState,
    ScholarLibrary,
    ScholarLibraryPaper,
)

READ_STATUS_VALUES = {"unread", "reading", "read"}
_RESOURCE_TYPE_ALIASES = {"project": "canvas"}
_USER_STATE_RESOURCE_TYPES = {"canvas", "paper", "scholar_library_paper"}
_TAG_NOTE_RESOURCE_TYPES = {"canvas", "paper", "scholar_library_paper", "resource_annotation"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _normalize_resource_type(resource_type: str) -> str:
    value = str(resource_type or "").strip().lower()
    if not value:
        raise ValueError("resource_type is required")
    return _RESOURCE_TYPE_ALIASES.get(value, value)


def _normalize_resource_id(resource_id: Any) -> str:
    value = str(resource_id or "").strip()
    if not value:
        raise ValueError("resource_id is required")
    return value


def _normalize_tag(tag: str) -> tuple[str, str]:
    display = " ".join(str(tag or "").strip().split())
    normalized = display.lower()
    if not normalized:
        raise ValueError("tag is required")
    return display, normalized


def _state_payload(resource_type: str, resource_id: str, row: Optional[ResourceUserState]) -> Dict[str, Any]:
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "favorite": bool(getattr(row, "favorite", 0)) if row is not None else False,
        "archived": bool(getattr(row, "archived", 0)) if row is not None else False,
        "read_status": (getattr(row, "read_status", None) or "unread") if row is not None else "unread",
        "last_opened_at": getattr(row, "last_opened_at", None) if row is not None else None,
        "created_at": getattr(row, "created_at", None),
        "updated_at": getattr(row, "updated_at", None),
    }


def _tag_payload(row: ResourceTag) -> Dict[str, Any]:
    return {
        "id": row.id,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "tag": row.tag,
        "normalized_tag": row.normalized_tag,
        "created_at": row.created_at,
    }


def _note_payload(row: ResourceNote) -> Dict[str, Any]:
    return {
        "id": row.id,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "note_md": row.note_md,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@dataclass(frozen=True)
class _ResolvedResource:
    resource_type: str
    resource_id: str


class ResourceStateService:
    def _resolve_canvas(self, session: Session, user_id: str, resource_id: str) -> _ResolvedResource:
        row = session.get(Canvas, resource_id)
        if row is None:
            raise LookupError("resource not found")
        if (row.user_id or "").strip() != user_id:
            raise PermissionError("resource does not belong to user")
        return _ResolvedResource(resource_type="canvas", resource_id=str(row.id))

    def _resolve_paper(self, session: Session, user_id: str, resource_id: str) -> _ResolvedResource:
        stmt = (
            select(Paper)
            .where(Paper.user_id == user_id, Paper.paper_uid == resource_id)
            .order_by(Paper.created_at.desc())
        )
        row = session.exec(stmt).first()
        if row is None:
            raise LookupError("resource not found")
        paper_uid = (row.paper_uid or "").strip() or resource_id
        return _ResolvedResource(resource_type="paper", resource_id=paper_uid)

    def _resolve_scholar_library_paper(self, session: Session, user_id: str, resource_id: str) -> _ResolvedResource:
        try:
            record_id = int(resource_id)
        except Exception as exc:
            raise ValueError("scholar_library_paper resource_id must be an integer string") from exc
        row = session.get(ScholarLibraryPaper, record_id)
        if row is None:
            raise LookupError("resource not found")
        lib = session.get(ScholarLibrary, row.library_id)
        if lib is None:
            raise LookupError("resource not found")
        if (lib.user_id or "").strip() != user_id:
            raise PermissionError("resource does not belong to user")
        return _ResolvedResource(resource_type="scholar_library_paper", resource_id=str(record_id))

    def _resolve_resource_annotation(self, session: Session, user_id: str, resource_id: str) -> _ResolvedResource:
        try:
            annotation_id = int(resource_id)
        except Exception as exc:
            raise ValueError("resource_annotation resource_id must be an integer string") from exc
        row = session.get(ResourceAnnotation, annotation_id)
        if row is None:
            raise LookupError("resource not found")
        if (row.user_id or "").strip() != user_id:
            raise PermissionError("resource does not belong to user")
        return _ResolvedResource(resource_type="resource_annotation", resource_id=str(annotation_id))

    def resolve_resource(self, *, user_id: str, resource_type: str, resource_id: Any, capability: str) -> _ResolvedResource:
        canonical_type = _normalize_resource_type(resource_type)
        canonical_id = _normalize_resource_id(resource_id)
        if capability == "state" and canonical_type not in _USER_STATE_RESOURCE_TYPES:
            raise ValueError(f"resource_type={canonical_type} does not support user_state")
        if capability in {"tag", "note"} and canonical_type not in _TAG_NOTE_RESOURCE_TYPES:
            raise ValueError(f"resource_type={canonical_type} does not support {capability}")

        with Session(get_engine()) as session:
            if canonical_type == "canvas":
                return self._resolve_canvas(session, user_id, canonical_id)
            if canonical_type == "paper":
                return self._resolve_paper(session, user_id, canonical_id)
            if canonical_type == "scholar_library_paper":
                return self._resolve_scholar_library_paper(session, user_id, canonical_id)
            if canonical_type == "resource_annotation":
                return self._resolve_resource_annotation(session, user_id, canonical_id)
        raise ValueError(f"unsupported resource_type={canonical_type}")

    def get_user_state(self, *, user_id: str, resource_type: str, resource_id: Any) -> Dict[str, Any]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="state")
        with Session(get_engine()) as session:
            stmt = select(ResourceUserState).where(
                ResourceUserState.user_id == user_id,
                ResourceUserState.resource_type == resolved.resource_type,
                ResourceUserState.resource_id == resolved.resource_id,
            )
            row = session.exec(stmt).first()
        return _state_payload(resolved.resource_type, resolved.resource_id, row)

    def get_user_state_map(
        self,
        *,
        user_id: str,
        resource_type: str,
        resource_ids: Iterable[Any],
    ) -> Dict[str, Dict[str, Any]]:
        canonical_type = _normalize_resource_type(resource_type)
        if canonical_type not in _USER_STATE_RESOURCE_TYPES:
            raise ValueError(f"resource_type={canonical_type} does not support user_state")
        ids = [_normalize_resource_id(item) for item in resource_ids if str(item or "").strip()]
        if not ids:
            return {}
        with Session(get_engine()) as session:
            stmt = select(ResourceUserState).where(
                ResourceUserState.user_id == user_id,
                ResourceUserState.resource_type == canonical_type,
                ResourceUserState.resource_id.in_(ids),
            )
            rows = session.exec(stmt).all()
        by_id = {row.resource_id: row for row in rows}
        return {resource_id: _state_payload(canonical_type, resource_id, by_id.get(resource_id)) for resource_id in ids}

    def upsert_user_state(
        self,
        *,
        user_id: str,
        resource_type: str,
        resource_id: Any,
        favorite: Optional[bool] = None,
        archived: Optional[bool] = None,
        read_status: Optional[str] = None,
        last_opened_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="state")
        if read_status is not None:
            read_status = str(read_status).strip().lower()
            if read_status not in READ_STATUS_VALUES:
                raise ValueError("read_status must be one of unread|reading|read")
        if all(value is None for value in (favorite, archived, read_status, last_opened_at)):
            return self.get_user_state(
                user_id=user_id,
                resource_type=resolved.resource_type,
                resource_id=resolved.resource_id,
            )

        now = _now_iso()
        with Session(get_engine()) as session:
            stmt = select(ResourceUserState).where(
                ResourceUserState.user_id == user_id,
                ResourceUserState.resource_type == resolved.resource_type,
                ResourceUserState.resource_id == resolved.resource_id,
            )
            row = session.exec(stmt).first()
            if row is None:
                row = ResourceUserState(
                    user_id=user_id,
                    resource_type=resolved.resource_type,
                    resource_id=resolved.resource_id,
                    favorite=0,
                    archived=0,
                    read_status="unread",
                    created_at=now,
                    updated_at=now,
                )
            if favorite is not None:
                row.favorite = 1 if bool(favorite) else 0
            if archived is not None:
                row.archived = 1 if bool(archived) else 0
            if read_status is not None:
                row.read_status = read_status
            if last_opened_at is not None:
                row.last_opened_at = last_opened_at or None
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return _state_payload(resolved.resource_type, resolved.resource_id, row)

    def list_tags(self, *, user_id: str, resource_type: str, resource_id: Any) -> List[Dict[str, Any]]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="tag")
        with Session(get_engine()) as session:
            stmt = (
                select(ResourceTag)
                .where(
                    ResourceTag.user_id == user_id,
                    ResourceTag.resource_type == resolved.resource_type,
                    ResourceTag.resource_id == resolved.resource_id,
                )
                .order_by(ResourceTag.created_at.asc(), ResourceTag.id.asc())
            )
            rows = session.exec(stmt).all()
        return [_tag_payload(row) for row in rows]

    def add_tag(self, *, user_id: str, resource_type: str, resource_id: Any, tag: str) -> Dict[str, Any]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="tag")
        display_tag, normalized_tag = _normalize_tag(tag)
        with Session(get_engine()) as session:
            stmt = select(ResourceTag).where(
                ResourceTag.user_id == user_id,
                ResourceTag.resource_type == resolved.resource_type,
                ResourceTag.resource_id == resolved.resource_id,
                ResourceTag.normalized_tag == normalized_tag,
            )
            row = session.exec(stmt).first()
            if row is None:
                row = ResourceTag(
                    user_id=user_id,
                    resource_type=resolved.resource_type,
                    resource_id=resolved.resource_id,
                    tag=display_tag,
                    normalized_tag=normalized_tag,
                    created_at=_now_iso(),
                )
                session.add(row)
                session.commit()
                session.refresh(row)
            return _tag_payload(row)

    def delete_tag(self, *, user_id: str, resource_type: str, resource_id: Any, tag: str) -> bool:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="tag")
        _display_tag, normalized_tag = _normalize_tag(tag)
        with Session(get_engine()) as session:
            stmt = select(ResourceTag).where(
                ResourceTag.user_id == user_id,
                ResourceTag.resource_type == resolved.resource_type,
                ResourceTag.resource_id == resolved.resource_id,
                ResourceTag.normalized_tag == normalized_tag,
            )
            row = session.exec(stmt).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list_notes(self, *, user_id: str, resource_type: str, resource_id: Any) -> List[Dict[str, Any]]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="note")
        with Session(get_engine()) as session:
            stmt = (
                select(ResourceNote)
                .where(
                    ResourceNote.user_id == user_id,
                    ResourceNote.resource_type == resolved.resource_type,
                    ResourceNote.resource_id == resolved.resource_id,
                )
                .order_by(ResourceNote.updated_at.desc(), ResourceNote.id.desc())
            )
            rows = session.exec(stmt).all()
        return [_note_payload(row) for row in rows]

    def create_note(self, *, user_id: str, resource_type: str, resource_id: Any, note_md: str) -> Dict[str, Any]:
        resolved = self.resolve_resource(user_id=user_id, resource_type=resource_type, resource_id=resource_id, capability="note")
        content = str(note_md or "").strip()
        if not content:
            raise ValueError("note_md is required")
        now = _now_iso()
        with Session(get_engine()) as session:
            row = ResourceNote(
                user_id=user_id,
                resource_type=resolved.resource_type,
                resource_id=resolved.resource_id,
                note_md=content,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return _note_payload(row)

    def update_note(self, *, user_id: str, note_id: int, note_md: str) -> Dict[str, Any]:
        content = str(note_md or "").strip()
        if not content:
            raise ValueError("note_md is required")
        with Session(get_engine()) as session:
            row = session.get(ResourceNote, note_id)
            if row is None:
                raise LookupError("note not found")
            if (row.user_id or "").strip() != user_id:
                raise PermissionError("note does not belong to user")
            self.resolve_resource(
                user_id=user_id,
                resource_type=row.resource_type,
                resource_id=row.resource_id,
                capability="note",
            )
            row.note_md = content
            row.updated_at = _now_iso()
            session.add(row)
            session.commit()
            session.refresh(row)
            return _note_payload(row)

    def delete_note(self, *, user_id: str, note_id: int) -> bool:
        with Session(get_engine()) as session:
            row = session.get(ResourceNote, note_id)
            if row is None:
                return False
            if (row.user_id or "").strip() != user_id:
                raise PermissionError("note does not belong to user")
            session.delete(row)
            session.commit()
            return True

    def delete_resource_overlays(
        self,
        *,
        user_id: str,
        resource_type: str,
        resource_id: Any,
        include_state: bool = True,
        include_tags: bool = True,
        include_notes: bool = True,
    ) -> Dict[str, int]:
        canonical_type = _normalize_resource_type(resource_type)
        canonical_id = _normalize_resource_id(resource_id)
        deleted = {"user_state": 0, "tags": 0, "notes": 0}
        with Session(get_engine()) as session:
            if include_state:
                stmt = select(ResourceUserState).where(
                    ResourceUserState.user_id == user_id,
                    ResourceUserState.resource_type == canonical_type,
                    ResourceUserState.resource_id == canonical_id,
                )
                rows = session.exec(stmt).all()
                deleted["user_state"] = len(rows)
                for row in rows:
                    session.delete(row)
            if include_tags:
                stmt = select(ResourceTag).where(
                    ResourceTag.user_id == user_id,
                    ResourceTag.resource_type == canonical_type,
                    ResourceTag.resource_id == canonical_id,
                )
                rows = session.exec(stmt).all()
                deleted["tags"] = len(rows)
                for row in rows:
                    session.delete(row)
            if include_notes:
                stmt = select(ResourceNote).where(
                    ResourceNote.user_id == user_id,
                    ResourceNote.resource_type == canonical_type,
                    ResourceNote.resource_id == canonical_id,
                )
                rows = session.exec(stmt).all()
                deleted["notes"] = len(rows)
                for row in rows:
                    session.delete(row)
            session.commit()
        return deleted


_RESOURCE_STATE_SERVICE = ResourceStateService()


def get_resource_state_service() -> ResourceStateService:
    return _RESOURCE_STATE_SERVICE
