"""
任务队列 API：GET /tasks/queue, POST /tasks/{task_id}/cancel|pause|resume
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.routes_auth import get_current_user_id
from src.api.schemas import TaskCancelResponse, TaskControlResponse, TaskQueueResponse, TaskStateItem
from src.collaboration.memory.persistent_store import get_user_profile
from src.tasks import get_task_queue
from src.tasks.task_state import TaskStatus

try:
    from src.observability import metrics as _obs_metrics
except Exception:
    _obs_metrics = None

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _is_admin_user(user_id: str) -> bool:
    profile = get_user_profile(user_id)
    return bool(profile and profile.get("role") == "admin")


def _state_to_item(state_dict: dict) -> TaskStateItem:
    return TaskStateItem(
        task_id=state_dict.get("task_id", ""),
        kind=state_dict.get("kind", "chat"),
        status=state_dict.get("status", "queued"),
        session_id=state_dict.get("session_id", ""),
        user_id=state_dict.get("user_id", ""),
        queue_position=int(state_dict.get("queue_position", 0)),
        created_at=state_dict.get("created_at"),
        started_at=state_dict.get("started_at"),
        finished_at=state_dict.get("finished_at"),
        pause_started_at=state_dict.get("pause_started_at"),
        paused_total_seconds=state_dict.get("paused_total_seconds"),
        error_message=state_dict.get("error_message"),
        payload=state_dict.get("payload"),
    )


@router.get("/queue", response_model=TaskQueueResponse)
def get_queue(current_user_id: str = Depends(get_current_user_id)):
    """排队区快照：当前活跃任务与排队列表。"""
    try:
        q = get_task_queue()
        snap = q.get_queue_snapshot()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    is_admin = _is_admin_user(current_user_id)
    visible_active = [
        s for s in snap["active"]
        if is_admin or str(s.get("user_id") or "") == current_user_id
    ]
    active = [_state_to_item(s) for s in visible_active]
    queued = []
    for item in snap["queued"]:
        owner_id = str(item.get("user_id") or "")
        if not is_admin and owner_id != current_user_id:
            continue
        visible_item = dict(item)
        visible_item["queue_position"] = len(queued) + 1
        queued.append(visible_item)
    return TaskQueueResponse(
        active_count=len(active),
        max_slots=snap["max_slots"],
        active=active,
        queued=queued,
    )


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task(task_id: str, current_user_id: str = Depends(get_current_user_id)):
    """取消任务：若在排队则移出队列，若在运行则标记为取消（调度器会停止）。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    state = q.get_state(task_id)
    if not state:
        return TaskCancelResponse(success=False, message="task not found")
    if not _is_admin_user(current_user_id):
        owner_id = str(getattr(state, "user_id", "") or "")
        if owner_id != current_user_id:
            raise HTTPException(status_code=403, detail="forbidden")
    if state.status == TaskStatus.queued:
        ok = q.cancel_queued(task_id)
        if ok and _obs_metrics and hasattr(_obs_metrics, "task_queue_cancelled_total"):
            _obs_metrics.task_queue_cancelled_total.labels(kind=state.kind.value).inc()
        return TaskCancelResponse(success=ok, message="cancelled from queue" if ok else "not in queue")
    if state.status in (TaskStatus.running, TaskStatus.pausing, TaskStatus.paused):
        ok = q.cancel_running(task_id)
        if ok and _obs_metrics and hasattr(_obs_metrics, "task_queue_cancelled_total"):
            _obs_metrics.task_queue_cancelled_total.labels(kind=state.kind.value).inc()
        return TaskCancelResponse(success=ok, message="cancelled" if ok else "not running")
    return TaskCancelResponse(success=False, message=f"task already {state.status.value}")


@router.post("/{task_id}/pause", response_model=TaskControlResponse)
def pause_task(task_id: str, current_user_id: str = Depends(get_current_user_id)):
    """暂停运行中的任务。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    state = q.get_state(task_id)
    if not state:
        return TaskControlResponse(success=False, message="task not found")
    if not _is_admin_user(current_user_id):
        owner_id = str(getattr(state, "user_id", "") or "")
        if owner_id != current_user_id:
            raise HTTPException(status_code=403, detail="forbidden")
    ok = q.pause_running(task_id)
    return TaskControlResponse(success=ok, message="pause requested" if ok else f"task not pausable: {state.status.value}")


@router.post("/{task_id}/resume", response_model=TaskControlResponse)
def resume_task(task_id: str, current_user_id: str = Depends(get_current_user_id)):
    """恢复已暂停的任务。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    state = q.get_state(task_id)
    if not state:
        return TaskControlResponse(success=False, message="task not found")
    if not _is_admin_user(current_user_id):
        owner_id = str(getattr(state, "user_id", "") or "")
        if owner_id != current_user_id:
            raise HTTPException(status_code=403, detail="forbidden")
    ok = q.resume_paused(task_id)
    return TaskControlResponse(success=ok, message="resumed" if ok else f"task not resumable: {state.status.value}")
