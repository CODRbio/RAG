"""
任务队列 API：GET /tasks/queue, POST /tasks/{task_id}/cancel
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import TaskCancelResponse, TaskQueueResponse, TaskStateItem
from src.tasks import get_task_queue
from src.tasks.task_state import TaskStatus

try:
    from src.observability import metrics as _obs_metrics
except Exception:
    _obs_metrics = None

router = APIRouter(prefix="/tasks", tags=["tasks"])


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
        error_message=state_dict.get("error_message"),
        payload=state_dict.get("payload"),
    )


@router.get("/queue", response_model=TaskQueueResponse)
def get_queue():
    """排队区快照：当前活跃任务与排队列表。"""
    try:
        q = get_task_queue()
        snap = q.get_queue_snapshot()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    active = [_state_to_item(s) for s in snap["active"]]
    return TaskQueueResponse(
        active_count=snap["active_count"],
        max_slots=snap["max_slots"],
        active=active,
        queued=snap["queued"],
    )


@router.post("/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task(task_id: str):
    """取消任务：若在排队则移出队列，若在运行则标记为取消（调度器会停止）。"""
    try:
        q = get_task_queue()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")
    state = q.get_state(task_id)
    if not state:
        return TaskCancelResponse(success=False, message="task not found")
    if state.status == TaskStatus.queued:
        ok = q.cancel_queued(task_id)
        if ok and _obs_metrics and hasattr(_obs_metrics, "task_queue_cancelled_total"):
            _obs_metrics.task_queue_cancelled_total.labels(kind=state.kind.value).inc()
        return TaskCancelResponse(success=ok, message="cancelled from queue" if ok else "not in queue")
    if state.status == TaskStatus.running:
        ok = q.cancel_running(task_id)
        if ok and _obs_metrics and hasattr(_obs_metrics, "task_queue_cancelled_total"):
            _obs_metrics.task_queue_cancelled_total.labels(kind=state.kind.value).inc()
        return TaskCancelResponse(success=ok, message="cancelled" if ok else "not running")
    return TaskCancelResponse(success=False, message=f"task already {state.status.value}")
