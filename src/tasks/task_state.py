"""
Task state model and status machine for unified Chat + Research queue.
States: queued -> running -> completed | error | cancelled | timeout
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TaskKind(str, Enum):
    chat = "chat"
    dr = "dr"


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
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
        return self.status == TaskStatus.running
