"""
Task state model and status machine for unified Chat + Research queue.
States: queued -> running -> pausing -> paused -> running | completed | error | cancelled | timeout
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TaskKind(str, Enum):
    chat = "chat"
    dr = "dr"
    scholar = "scholar"
    academic_assistant = "academic_assistant"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    pausing = "pausing"
    paused = "paused"
    completed = "completed"
    error = "error"
    cancelled = "cancelled"
    timeout = "timeout"


@dataclass
class TaskState:
    task_id: str
    kind: TaskKind
    status: TaskStatus
    session_id: str = ""
    user_id: str = ""
    queue_position: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error_message: Optional[str] = None
    pause_started_at: Optional[float] = None
    paused_total_seconds: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "queue_position": self.queue_position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_message": self.error_message,
            "pause_started_at": self.pause_started_at,
            "paused_total_seconds": self.paused_total_seconds,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskState:
        kind = data.get("kind", "chat")
        status = data.get("status", "queued")
        return cls(
            task_id=str(data["task_id"]),
            kind=TaskKind(kind) if isinstance(kind, str) else kind,
            status=TaskStatus(status) if isinstance(status, str) else status,
            session_id=str(data.get("session_id", "")),
            user_id=str(data.get("user_id", "")),
            queue_position=int(data.get("queue_position", 0)),
            created_at=float(data.get("created_at", time.time())),
            started_at=float(data["started_at"]) if data.get("started_at") is not None else None,
            finished_at=float(data["finished_at"]) if data.get("finished_at") is not None else None,
            error_message=data.get("error_message"),
            pause_started_at=float(data["pause_started_at"]) if data.get("pause_started_at") is not None else None,
            paused_total_seconds=float(data.get("paused_total_seconds", 0.0) or 0.0),
            payload=dict(data.get("payload") or {}),
        )

    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.completed,
            TaskStatus.error,
            TaskStatus.cancelled,
            TaskStatus.timeout,
        )

    def is_active(self) -> bool:
        return self.status in (TaskStatus.running, TaskStatus.pausing, TaskStatus.paused)

    def effective_runtime_seconds(self, now: Optional[float] = None) -> float:
        if self.started_at is None:
            return 0.0
        current = float(now if now is not None else time.time())
        paused_total = float(self.paused_total_seconds or 0.0)
        if self.pause_started_at is not None:
            paused_total += max(0.0, current - float(self.pause_started_at))
        return max(0.0, current - float(self.started_at) - paused_total)
