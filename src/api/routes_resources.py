from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.routes_auth import get_current_user_id
from src.services.resource_state_service import get_resource_state_service

router = APIRouter(prefix="/resources", tags=["resources"])


def _raise_resource_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=str(exc))
    raise HTTPException(status_code=500, detail=str(exc))


class ResourceRef(BaseModel):
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)


class ResourceUserStateUpsert(BaseModel):
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    favorite: Optional[bool] = None
    archived: Optional[bool] = None
    read_status: Optional[str] = None
    last_opened_at: Optional[str] = None


class ResourceTagUpsert(BaseModel):
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1)


class ResourceNoteCreate(BaseModel):
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    note_md: str = Field(..., min_length=1)


class ResourceNoteUpdate(BaseModel):
    note_md: str = Field(..., min_length=1)


@router.get("/state")
def get_resource_state(
    resource_type: str = Query(..., min_length=1),
    resource_id: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return service.get_user_state(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
    except Exception as exc:
        _raise_resource_error(exc)


@router.patch("/state")
def patch_resource_state(
    body: ResourceUserStateUpsert,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return service.upsert_user_state(
            user_id=user_id,
            resource_type=body.resource_type,
            resource_id=body.resource_id,
            favorite=body.favorite,
            archived=body.archived,
            read_status=body.read_status,
            last_opened_at=body.last_opened_at,
        )
    except Exception as exc:
        _raise_resource_error(exc)


@router.get("/tags")
def list_resource_tags(
    resource_type: str = Query(..., min_length=1),
    resource_id: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return {
            "items": service.list_tags(
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        }
    except Exception as exc:
        _raise_resource_error(exc)


@router.post("/tags")
def create_resource_tag(
    body: ResourceTagUpsert,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return service.add_tag(
            user_id=user_id,
            resource_type=body.resource_type,
            resource_id=body.resource_id,
            tag=body.tag,
        )
    except Exception as exc:
        _raise_resource_error(exc)


@router.delete("/tags")
def delete_resource_tag(
    resource_type: str = Query(..., min_length=1),
    resource_id: str = Query(..., min_length=1),
    tag: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        deleted = service.delete_tag(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            tag=tag,
        )
        return {"deleted": deleted}
    except Exception as exc:
        _raise_resource_error(exc)


@router.get("/notes")
def list_resource_notes(
    resource_type: str = Query(..., min_length=1),
    resource_id: str = Query(..., min_length=1),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return {
            "items": service.list_notes(
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        }
    except Exception as exc:
        _raise_resource_error(exc)


@router.post("/notes")
def create_resource_note(
    body: ResourceNoteCreate,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return service.create_note(
            user_id=user_id,
            resource_type=body.resource_type,
            resource_id=body.resource_id,
            note_md=body.note_md,
        )
    except Exception as exc:
        _raise_resource_error(exc)


@router.patch("/notes/{note_id}")
def update_resource_note(
    note_id: int,
    body: ResourceNoteUpdate,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        return service.update_note(
            user_id=user_id,
            note_id=note_id,
            note_md=body.note_md,
        )
    except Exception as exc:
        _raise_resource_error(exc)


@router.delete("/notes/{note_id}")
def delete_resource_note(
    note_id: int,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_resource_state_service()
    try:
        deleted = service.delete_note(user_id=user_id, note_id=note_id)
        return {"deleted": deleted}
    except Exception as exc:
        _raise_resource_error(exc)
