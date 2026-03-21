from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.routes_auth import get_current_user_id
from src.indexing.assistant_artifact_store import (
    get_resource_annotation,
    list_resource_annotations,
    upsert_resource_annotation,
)
from src.log import get_logger
from src.services.reference_assistant_service import get_reference_assistant_service
from src.tasks.redis_queue import get_task_queue
from src.tasks.task_state import TaskKind, TaskState, TaskStatus

logger = get_logger(__name__)
router = APIRouter(prefix="/academic-assistant", tags=["academic-assistant"])

_assistant_bg_tasks: set[asyncio.Task] = set()
_ASSISTANT_SSE_HEARTBEAT_INTERVAL = 5


def _register_assistant_bg_task(task: asyncio.Task) -> None:
    _assistant_bg_tasks.add(task)
    task.add_done_callback(_assistant_bg_tasks.discard)


async def wait_academic_assistant_background_tasks_or_timeout(timeout_seconds: float) -> None:
    if not _assistant_bg_tasks:
        return
    try:
        done, pending = await asyncio.wait(
            _assistant_bg_tasks.copy(),
            timeout=max(0.1, timeout_seconds),
            return_when=asyncio.ALL_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(
                "[academic-assistant] graceful shutdown waited %.1fs, cancelled %d task(s)",
                timeout_seconds,
                len(pending),
            )
    except Exception as exc:
        logger.warning("[academic-assistant] shutdown wait failed: %s", exc)


class AssistantScopePayload(BaseModel):
    scope_type: str = Field("collection")
    scope_key: str = Field("", description="collection name | library id | global")


class PaperLocatorPayload(BaseModel):
    paper_uid: Optional[str] = None
    paper_id: Optional[str] = None
    collection: Optional[str] = None


class PaperSummaryRequest(BaseModel):
    locator: PaperLocatorPayload
    scope: Optional[AssistantScopePayload] = None
    question: Optional[str] = None
    llm_provider: Optional[str] = None
    model_override: Optional[str] = None


class PaperQuestionRequest(BaseModel):
    locator: PaperLocatorPayload
    question: str = Field(..., min_length=1)
    scope: Optional[AssistantScopePayload] = None
    llm_provider: Optional[str] = None
    model_override: Optional[str] = None


class PaperCompareRequest(BaseModel):
    paper_uids: List[str] = Field(..., min_length=2, max_length=5)
    aspects: List[str] = Field(default_factory=lambda: ["objective", "methodology", "key_findings", "limitations"])
    scope: Optional[AssistantScopePayload] = None
    llm_provider: Optional[str] = None
    model_override: Optional[str] = None


class DiscoveryStartRequest(BaseModel):
    paper_uids: List[str] = Field(default_factory=list)
    node_ids: List[str] = Field(default_factory=list)
    scope: Optional[AssistantScopePayload] = None
    question: Optional[str] = None
    limit: int = Field(10, ge=1, le=20)


class MediaAnalysisStartRequest(BaseModel):
    paper_uids: List[str] = Field(..., min_length=1, max_length=20)
    scope: Optional[AssistantScopePayload] = None
    force_reparse: bool = False
    upsert_vectors: bool = True
    llm_text_provider: Optional[str] = None
    llm_vision_provider: Optional[str] = None
    llm_text_model: Optional[str] = None
    llm_vision_model: Optional[str] = None


class ResourceAnnotationUpsertRequest(BaseModel):
    annotation_id: Optional[int] = None
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    paper_uid: Optional[str] = None
    target_kind: str = Field("chunk")
    target_locator: Dict[str, Any] = Field(default_factory=dict)
    target_text: str = Field(default="")
    directive: str = Field(default="")
    status: str = Field(default="active")
    collection: Optional[str] = None


class TaskStartResponse(BaseModel):
    task_id: str
    status: str
    message: str


def _task_event_stream(task_id: str, q, after_id: str = "-"):
    last_id = after_id
    last_sent_at = time.monotonic()
    yield ": stream-init\n\n"
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
            # Fallback: synthesise a terminal event from task state so the client
            # always receives one of the canonical terminal event names.
            status_val = state.status.value
            if status_val == "completed":
                result_payload = (state.payload or {}).get("result") or {}
                yield f"event: done\ndata: {json.dumps(result_payload, ensure_ascii=False, default=str)}\n\n"
            else:
                # error / cancelled / timeout — emit as-is so client recognises them
                yield f"event: {status_val}\ndata: {json.dumps({'status': status_val}, ensure_ascii=False)}\n\n"
            return
        if time.monotonic() - last_sent_at >= _ASSISTANT_SSE_HEARTBEAT_INTERVAL:
            yield ": heartbeat\n\n"
            last_sent_at = time.monotonic()
        time.sleep(0.3)


async def _run_job(task_id: str, task_type: str, fn, *args, **kwargs) -> None:
    q = get_task_queue()
    heartbeat_stop = asyncio.Event()
    started_at = time.time()

    async def _heartbeat() -> None:
        while not heartbeat_stop.is_set():
            try:
                await asyncio.wait_for(heartbeat_stop.wait(), timeout=_ASSISTANT_SSE_HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                pass
            if heartbeat_stop.is_set():
                break
            state = q.get_state(task_id)
            if state and state.is_terminal():
                break
            q.push_event(task_id, "heartbeat", {"task_type": task_type, "elapsed_s": round(time.time() - started_at, 1)})

    heartbeat_task = asyncio.create_task(_heartbeat(), name=f"academic-assistant-heartbeat:{task_id}")
    try:
        q.push_event(task_id, "progress", {"stage": "running", "task_type": task_type})
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
        q.push_event(task_id, "done", result)
        state = q.get_state(task_id)
        if state is not None:
            state.status = TaskStatus.completed
            state.finished_at = time.time()
            state.payload["result"] = result
            q.set_state(state)
    except asyncio.CancelledError:
        state = q.get_state(task_id)
        if state is not None:
            state.status = TaskStatus.cancelled
            state.finished_at = time.time()
            q.set_state(state)
        q.push_event(task_id, "cancelled", {"status": "cancelled"})
        raise
    except Exception as exc:
        logger.exception("[academic-assistant] task_id=%s failed: %s", task_id, exc)
        state = q.get_state(task_id)
        if state is not None:
            state.status = TaskStatus.error
            state.finished_at = time.time()
            state.error_message = str(exc)
            q.set_state(state)
        q.push_event(task_id, "error", {"message": str(exc)})
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


@router.post("/papers/summary")
def summarize_paper(
    body: PaperSummaryRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_reference_assistant_service()
    return service.summarize_paper(
        body.locator.model_dump(),
        user_id=user_id,
        scope=body.scope.model_dump() if body.scope else None,
        question=body.question,
        llm_provider=body.llm_provider,
        model_override=body.model_override,
    )


@router.post("/papers/qa")
def ask_paper(
    body: PaperQuestionRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_reference_assistant_service()
    return service.ask_paper(
        body.locator.model_dump(),
        user_id=user_id,
        question=body.question,
        scope=body.scope.model_dump() if body.scope else None,
        llm_provider=body.llm_provider,
        model_override=body.model_override,
    )


@router.post("/papers/compare")
def compare_papers(
    body: PaperCompareRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    service = get_reference_assistant_service()
    return service.compare_papers(
        body.paper_uids,
        user_id=user_id,
        aspects=body.aspects,
        scope=body.scope.model_dump() if body.scope else None,
        llm_provider=body.llm_provider,
        model_override=body.model_override,
    )


@router.post("/media-analysis/start", response_model=TaskStartResponse)
async def start_media_analysis(
    body: MediaAnalysisStartRequest,
    user_id: str = Depends(get_current_user_id),
) -> TaskStartResponse:
    service = get_reference_assistant_service()
    task_id = f"aa_media_{uuid.uuid4().hex[:12]}"
    q = get_task_queue()
    state = TaskState(
        task_id=task_id,
        kind=TaskKind.academic_assistant,
        status=TaskStatus.running,
        user_id=user_id,
        started_at=time.time(),
        payload={
            "type": "media-analysis",
            "paper_uids": body.paper_uids,
        },
    )
    q.set_state(state)
    task = asyncio.create_task(
        _run_job(
            task_id,
            "media-analysis",
            service.analyze_paper_media,
            body.paper_uids,
            user_id=user_id,
            scope=body.scope.model_dump() if body.scope else None,
            force_reparse=body.force_reparse,
            upsert_vectors_enabled=body.upsert_vectors,
            llm_text_provider=body.llm_text_provider,
            llm_vision_provider=body.llm_vision_provider,
            llm_text_model=body.llm_text_model,
            llm_vision_model=body.llm_vision_model,
        ),
        name=f"academic-assistant-media:{task_id}",
    )
    _register_assistant_bg_task(task)
    return TaskStartResponse(task_id=task_id, status="submitted", message="media-analysis task submitted")


@router.post("/discovery/{mode}/start", response_model=TaskStartResponse)
async def start_discovery(
    mode: str,
    body: DiscoveryStartRequest,
    user_id: str = Depends(get_current_user_id),
) -> TaskStartResponse:
    # Normalise kebab-case coming from the frontend (e.g. "missing-core") to the
    # underscore form that ReferenceAssistantService.discover() expects.
    mode = mode.replace("-", "_")
    service = get_reference_assistant_service()
    task_id = f"aa_discovery_{uuid.uuid4().hex[:12]}"
    q = get_task_queue()
    state = TaskState(
        task_id=task_id,
        kind=TaskKind.academic_assistant,
        status=TaskStatus.running,
        user_id=user_id,
        started_at=time.time(),
        payload={"type": "discovery", "mode": mode, "paper_uids": body.paper_uids},
    )
    q.set_state(state)
    seeds = {
        "paper_uids": body.paper_uids,
        "node_ids": body.node_ids,
    }
    task = asyncio.create_task(
        _run_job(
            task_id,
            f"discovery:{mode}",
            service.discover,
            mode,
            user_id=user_id,
            seeds=seeds,
            scope=body.scope.model_dump() if body.scope else None,
            options={"question": body.question, "limit": body.limit},
        ),
        name=f"academic-assistant-discovery:{task_id}",
    )
    _register_assistant_bg_task(task)
    return TaskStartResponse(task_id=task_id, status="submitted", message=f"{mode} discovery task submitted")


@router.post("/annotations")
def upsert_annotation(
    body: ResourceAnnotationUpsertRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    row = upsert_resource_annotation(
        user_id=user_id,
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        paper_uid=(body.paper_uid or "").strip(),
        target_kind=body.target_kind,
        target_locator=body.target_locator,
        target_text=body.target_text,
        directive=body.directive,
        status=body.status,
        annotation_id=body.annotation_id,
        collection=body.collection,
    )
    return {
        "id": row.id,
        "user_id": row.user_id,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "paper_uid": row.paper_uid,
        "target_kind": row.target_kind,
        "target_locator": row.get_target_locator(),
        "target_text": row.target_text,
        "directive": row.directive,
        "status": row.status,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@router.get("/annotations")
def get_annotations(
    paper_uid: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    target_kind: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    rows = list_resource_annotations(
        user_id=user_id,
        paper_uid=paper_uid,
        resource_type=resource_type,
        resource_id=resource_id,
        target_kind=target_kind,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {
        "items": [
            {
                "id": row.id,
                "user_id": row.user_id,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "paper_uid": row.paper_uid,
                "target_kind": row.target_kind,
                "target_locator": row.get_target_locator(),
                "target_text": row.target_text,
                "directive": row.directive,
                "status": row.status,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]
    }


@router.get("/task/{task_id}/stream")
def assistant_task_stream(task_id: str, request: Request, user_id: str = Depends(get_current_user_id)):
    try:
        q = get_task_queue()
    except Exception as exc:
        logger.warning("academic assistant stream get_task_queue failed: %s", exc)
        raise HTTPException(status_code=503, detail="Task store unavailable")
    state = q.get_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if state.kind != TaskKind.academic_assistant:
        raise HTTPException(status_code=400, detail="Not an academic assistant task")
    if state.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该任务")
    after_id = request.headers.get("Last-Event-ID") or request.query_params.get("after_id") or "-"
    return StreamingResponse(
        _task_event_stream(task_id, q, after_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/task/{task_id}")
def assistant_task_status(task_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        q = get_task_queue()
        state = q.get_state(task_id)
    except Exception as exc:
        logger.warning("academic assistant task status get_task_queue failed: %s", exc)
        raise HTTPException(status_code=503, detail="Task store unavailable")
    if not state:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if state.kind != TaskKind.academic_assistant:
        raise HTTPException(status_code=400, detail="Not an academic assistant task")
    if state.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该任务")
    return {
        "task_id": task_id,
        "status": state.status.value,
        "error_message": state.error_message,
        "payload": state.payload,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
    }
