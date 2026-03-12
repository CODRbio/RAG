"""
Redis-backed task queue: stream for pending, set for active slots, KV for task state, stream per task for SSE events.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.log import get_logger
from src.tasks.task_state import TaskKind, TaskStatus, TaskState

logger = get_logger(__name__)

# Key prefixes
STREAM_QUEUE = "rag:task_queue"
SET_ACTIVE = "rag:active_tasks"
KEY_TASK_PREFIX = "rag:task:"
STREAM_EVENTS_PREFIX = "rag:task_events:"
KEY_ACTIVE_SESSIONS = "rag:active_sessions"  # set of session_id that are currently running


def _task_key(task_id: str) -> str:
    return f"{KEY_TASK_PREFIX}{task_id}"


def _events_stream(task_id: str) -> str:
    return f"{STREAM_EVENTS_PREFIX}{task_id}"


class TaskQueue:
    """Sync Redis client for queue, slots, and task state."""

    def __init__(self, redis_url: Optional[str] = None):
        self._url = redis_url or settings.tasks.redis_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(
                    self._url,
                    decode_responses=True,
                )
                self._client.ping()
            except Exception as e:
                logger.warning("[TaskQueue] Redis connect failed: %s", e)
                raise

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def submit(
        self,
        kind: TaskKind,
        session_id: str,
        user_id: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> str:
        """Enqueue a task; returns task_id."""
        self._ensure_client()
        task_id = task_id or str(uuid.uuid4())
        state = TaskState(
            task_id=task_id,
            kind=kind,
            status=TaskStatus.queued,
            session_id=session_id,
            user_id=user_id,
            queue_position=0,
            payload=payload,
        )
        key = _task_key(task_id)
        self._client.setex(
            key,
            settings.tasks.task_state_ttl_seconds,
            json.dumps(state.to_dict(), ensure_ascii=False),
        )
        # XADD to stream (maxlen to avoid unbounded growth)
        self._client.xadd(
            STREAM_QUEUE,
            {
                "task_id": task_id,
                "kind": kind.value,
                "session_id": session_id,
                "user_id": user_id,
            },
            maxlen=settings.tasks.queue_max_len,
        )
        logger.info("[TaskQueue] submitted task_id=%s kind=%s session_id=%s", task_id, kind.value, session_id)
        return task_id

    def get_state(self, task_id: str) -> Optional[TaskState]:
        self._ensure_client()
        raw = self._client.get(_task_key(task_id))
        if not raw:
            return None
        try:
            return TaskState.from_dict(json.loads(raw))
        except Exception as e:
            logger.warning("[TaskQueue] get_state parse error task_id=%s: %s", task_id, e)
            return None

    def set_state(self, state: TaskState) -> None:
        self._ensure_client()
        key = _task_key(state.task_id)
        self._client.setex(
            key,
            settings.tasks.task_state_ttl_seconds,
            json.dumps(state.to_dict(), ensure_ascii=False),
        )

    def acquire_slot(self, task_id: str, session_id: str) -> bool:
        """Try to mark task as running (consume a slot). Returns True if acquired."""
        self._ensure_client()
        import redis
        for _ in range(5):
            try:
                with self._client.pipeline() as pipe:
                    pipe.watch(SET_ACTIVE, KEY_ACTIVE_SESSIONS)
                    active = pipe.scard(SET_ACTIVE)
                    if active >= settings.tasks.max_active_slots:
                        pipe.unwatch()
                        return False
                    if session_id and pipe.sismember(KEY_ACTIVE_SESSIONS, session_id):
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.sadd(SET_ACTIVE, task_id)
                    if session_id:
                        pipe.sadd(KEY_ACTIVE_SESSIONS, session_id)
                    pipe.execute()
                    return True
            except redis.WatchError:
                continue
        return False

    def release_slot(self, task_id: str, session_id: str) -> None:
        self._ensure_client()
        self._client.srem(SET_ACTIVE, task_id)
        self._client.srem(KEY_ACTIVE_SESSIONS, session_id)

    def active_count(self) -> int:
        self._ensure_client()
        return self._client.scard(SET_ACTIVE)

    def pending_count(self) -> int:
        self._ensure_client()
        return self._client.xlen(STREAM_QUEUE)

    def read_pending(self, count: int = 10) -> List[Dict[str, Any]]:
        """Read oldest entries from queue without removing (for dispatcher to try acquire_slot)."""
        self._ensure_client()
        entries = self._client.xrange(STREAM_QUEUE, count=count)
        out = []
        for eid, data in entries:
            out.append({"stream_id": eid, **data})
        return out

    def claim_next(self) -> Optional[Dict[str, Any]]:
        """Remove and return oldest queue entry; caller must then acquire_slot and run or re-enqueue."""
        self._ensure_client()
        entries = self._client.xrange(STREAM_QUEUE, count=1)
        if not entries:
            return None
        eid, data = entries[0]
        self._client.xdel(STREAM_QUEUE, eid)
        return {"stream_id": eid, **data}

    def remove_from_queue_by_stream_id(self, stream_id: str) -> bool:
        """Remove a specific entry from the queue by its stream id (e.g. after acquire_slot)."""
        self._ensure_client()
        try:
            self._client.xdel(STREAM_QUEUE, stream_id)
            return True
        except Exception:
            return False

    def push_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Append to task event stream for SSE."""
        self._ensure_client()
        stream = _events_stream(task_id)
        self._client.xadd(
            stream,
            {"type": event_type, "data": json.dumps(data, ensure_ascii=False)},
            maxlen=1000,
        )

    def read_events(self, task_id: str, after_id: str = "-", count: int = 100) -> List[Dict[str, Any]]:
        """Read events for SSE (after_id='-' from start). Returns list of {id, type, data}."""
        self._ensure_client()
        stream = _events_stream(task_id)
        try:
            entries = self._client.xrange(stream, min=after_id, max="+", count=count)
        except Exception:
            entries = []
        out = []
        for eid, raw in entries:
            if after_id != "-" and eid == after_id:
                continue
            t = raw.get("type", "")
            d = raw.get("data", "{}")
            try:
                data = json.loads(d)
            except Exception:
                data = d
            out.append({"id": eid, "type": t, "data": data})
        return out

    def get_queue_snapshot(self) -> Dict[str, Any]:
        """Active task_ids and queued items for /tasks/queue API."""
        self._ensure_client()
        active_ids = list(self._client.smembers(SET_ACTIVE))
        active_states = []
        for tid in active_ids:
            s = self.get_state(tid)
            if s:
                active_states.append(s.to_dict())
        queued = self.read_pending(count=settings.tasks.queue_max_len)
        queued_with_position = []
        for i, q in enumerate(queued):
            task_id = q.get("task_id")
            state = self.get_state(task_id) if task_id else None
            queued_with_position.append({
                "task_id": task_id,
                "kind": q.get("kind", "chat"),
                "session_id": q.get("session_id", ""),
                "user_id": q.get("user_id", ""),
                "queue_position": i + 1,
                "state": state.to_dict() if state else None,
            })
        return {
            "active_count": len(active_ids),
            "max_slots": settings.tasks.max_active_slots,
            "active": active_states,
            "queued": queued_with_position,
        }

    def cancel_queued(self, task_id: str) -> bool:
        """Remove task from queue if still queued; update state to cancelled. Returns True if removed."""
        self._ensure_client()
        state = self.get_state(task_id)
        if not state or state.status != TaskStatus.queued:
            return False
        entries = self._client.xrange(STREAM_QUEUE, count=settings.tasks.queue_max_len)
        for eid, data in entries:
            if data.get("task_id") == task_id:
                self._client.xdel(STREAM_QUEUE, eid)
                state.status = TaskStatus.cancelled
                state.finished_at = time.time()
                self.set_state(state)
                return True
        return False

    def cancel_running(self, task_id: str) -> bool:
        """Mark running task as cancelled; dispatcher must check state and stop. Returns True if was running."""
        state = self.get_state(task_id)
        if not state or state.status not in (TaskStatus.running, TaskStatus.pausing, TaskStatus.paused):
            return False
        state.status = TaskStatus.cancelled
        state.finished_at = time.time()
        state.pause_started_at = None
        self.set_state(state)
        self.push_event(task_id, "cancelled", {})
        return True

    def pause_running(self, task_id: str) -> bool:
        """Request cooperative pause for a running task."""
        state = self.get_state(task_id)
        if not state or state.status != TaskStatus.running:
            return False
        state.status = TaskStatus.pausing
        self.set_state(state)
        self.push_event(task_id, "pause_requested", {"status": TaskStatus.pausing.value})
        return True

    def resume_paused(self, task_id: str) -> bool:
        """Resume a paused or pausing task."""
        state = self.get_state(task_id)
        if not state or state.status not in (TaskStatus.paused, TaskStatus.pausing):
            return False
        now = time.time()
        if state.pause_started_at is not None:
            state.paused_total_seconds = float(state.paused_total_seconds or 0.0) + max(
                0.0,
                now - float(state.pause_started_at),
            )
        state.pause_started_at = None
        state.status = TaskStatus.running
        self.set_state(state)
        self.push_event(task_id, "resumed", {"status": TaskStatus.running.value})
        return True

    def repair_stale_scholar_tasks(self, max_age_seconds: int = 300) -> int:
        """Mark Scholar tasks stuck in 'running' longer than max_age_seconds as error (e.g. after restart). Returns count repaired."""
        self._ensure_client()
        repaired = 0
        now = time.time()
        try:
            for key in self._client.scan_iter(match=f"{KEY_TASK_PREFIX}*", count=100):
                task_id = key[len(KEY_TASK_PREFIX):] if key.startswith(KEY_TASK_PREFIX) else None
                if not task_id:
                    continue
                state = self.get_state(task_id)
                if not state or state.kind != TaskKind.scholar or state.status != TaskStatus.running:
                    continue
                started = state.started_at or state.created_at or 0
                if (now - started) < max_age_seconds:
                    continue
                state.status = TaskStatus.error
                state.finished_at = now
                state.error_message = "服务重启，任务中断"
                self.set_state(state)
                self.push_event(task_id, "error", {"message": state.error_message})
                repaired += 1
                logger.info("[TaskQueue] repaired stale scholar task_id=%s (was running >%ds)", task_id, max_age_seconds)
        except Exception as e:
            logger.warning("[TaskQueue] repair_stale_scholar_tasks failed: %s", e)
        return repaired

    def repair_stale_chat_tasks(self, max_age_seconds: int = 60) -> int:
        """Mark Chat tasks stuck in active states as error after restart.

        Chat tasks are short-lived (seconds to minutes); any task still 'running' after a restart
        will never complete because its coroutine died with the process. Marking them as error
        releases the active slot in rag:active_tasks / rag:active_sessions so new tasks
        for the same session are not blocked.

        Returns count repaired.
        """
        self._ensure_client()
        repaired = 0
        now = time.time()
        try:
            for key in self._client.scan_iter(match=f"{KEY_TASK_PREFIX}*", count=100):
                task_id = key[len(KEY_TASK_PREFIX):] if key.startswith(KEY_TASK_PREFIX) else None
                if not task_id:
                    continue
                state = self.get_state(task_id)
                if not state or state.kind != TaskKind.chat or state.status not in (
                    TaskStatus.running,
                    TaskStatus.pausing,
                    TaskStatus.paused,
                ):
                    continue
                started = state.started_at or state.created_at or 0
                if (now - started) < max_age_seconds:
                    continue
                state.status = TaskStatus.error
                state.finished_at = now
                state.error_message = "服务重启，任务中断"
                self.set_state(state)
                self.push_event(task_id, "error", {"message": state.error_message})
                # Release the active slot so new tasks for the same session are not blocked
                self.release_slot(task_id, str(state.session_id or ""))
                repaired += 1
                logger.info("[TaskQueue] repaired stale chat task_id=%s (was active >%ds)", task_id, max_age_seconds)
        except Exception as e:
            logger.warning("[TaskQueue] repair_stale_chat_tasks failed: %s", e)
        return repaired


# Singleton for app lifecycle
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue
