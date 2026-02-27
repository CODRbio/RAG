"""
项目管理 API：列出当前用户项目、存档/取消存档、删除。
删除项目时会一并清理该画布下的会话、后台调研任务、working memory、用户项目关联。
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.routes_auth import get_current_user_id
from src.collaboration.canvas.canvas_manager import get_canvas_store
from src.collaboration.memory.session_memory import get_session_store
from src.collaboration.memory.working_memory import delete_working_memory
from src.collaboration.memory.persistent_store import delete_user_project
from src.collaboration.research.job_store import delete_jobs_by_canvas_id

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("")
def list_projects(
    include_archived: bool = False,
    user_id: str = Depends(get_current_user_id),
) -> list[dict]:
    """列出当前用户的项目（含存档状态）。"""
    store = get_canvas_store()
    items = store.list_by_user(user_id, include_archived=include_archived)
    return items


@router.post("/{canvas_id}/archive")
def archive_project(
    canvas_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """将项目标记为已存档（不受生命周期清理）。"""
    store = get_canvas_store()
    owner = store.get_canvas_owner(canvas_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="project not found")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="not your project")
    if not store.archive(canvas_id):
        raise HTTPException(status_code=404, detail="project not found")
    return {"canvas_id": canvas_id, "archived": True}


@router.post("/{canvas_id}/unarchive")
def unarchive_project(
    canvas_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """取消存档。"""
    store = get_canvas_store()
    owner = store.get_canvas_owner(canvas_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="project not found")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="not your project")
    if not store.unarchive(canvas_id):
        raise HTTPException(status_code=404, detail="project not found")
    return {"canvas_id": canvas_id, "archived": False}


@router.delete("/{canvas_id}")
def delete_project(
    canvas_id: str,
    user_id: str = Depends(get_current_user_id),
) -> dict:
    """删除项目。已存档的项目需先取消存档再删除。"""
    store = get_canvas_store()
    owner = store.get_canvas_owner(canvas_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="project not found")
    if owner != user_id:
        raise HTTPException(status_code=403, detail="not your project")
    # Check if archived - require unarchive first
    items = store.list_by_user(user_id, include_archived=True)
    current = next((x for x in items if x["id"] == canvas_id), None)
    if current and current.get("archived"):
        raise HTTPException(
            status_code=400,
            detail="archived project must be unarchived before deletion",
        )
    # 清理该画布关联资源：会话、后台调研任务、working memory、用户项目关联
    get_session_store().delete_sessions_by_canvas_id(canvas_id)
    delete_jobs_by_canvas_id(canvas_id)
    delete_working_memory(canvas_id)
    delete_user_project(user_id, canvas_id)
    if not store.delete(canvas_id):
        raise HTTPException(status_code=404, detail="project not found")
    return {"canvas_id": canvas_id, "deleted": True}
