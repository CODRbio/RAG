from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from src.api.routes_canvas import router as canvas_router
from src.api.routes_chat import router as chat_router
from src.api.routes_tasks import router as tasks_router
from src.auth.session import create_token
from src.collaboration.canvas.canvas_manager import create_canvas
from src.collaboration.memory.session_memory import get_session_store
from src.db.engine import get_engine, init_db
from src.db.models import UserProfile
from src.tasks.task_state import TaskKind, TaskState, TaskStatus


def _init_test_db(monkeypatch, tmp_path: Path) -> None:
    import src.db.engine as db_engine

    db_path = tmp_path / "authz.db"
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{db_path}")
    db_engine._engine = None
    init_db()


def _create_user(user_id: str, role: str = "user") -> None:
    with Session(get_engine()) as session:
        row = session.get(UserProfile, user_id)
        if row is None:
            row = UserProfile(user_id=user_id, role=role, password_hash="", preferences_json="{}")
        else:
            row.role = role
        session.add(row)
        session.commit()


def _auth_header(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(tasks_router)
    app.include_router(canvas_router)
    return TestClient(app)


def test_session_endpoints_require_owner(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _create_user("alice")
    _create_user("bob")
    client = _build_client()

    store = get_session_store()
    alice_session = store.create_session(user_id="alice")
    bob_session = store.create_session(user_id="bob")

    response = client.get(f"/sessions/{alice_session}")
    assert response.status_code == 401

    response = client.get(f"/sessions/{alice_session}", headers=_auth_header("bob"))
    assert response.status_code == 403

    response = client.get(f"/sessions/{alice_session}", headers=_auth_header("alice"))
    assert response.status_code == 200
    assert response.json()["session_id"] == alice_session

    response = client.get("/sessions", headers=_auth_header("alice"))
    assert response.status_code == 200
    session_ids = {item["session_id"] for item in response.json()}
    assert alice_session in session_ids
    assert bob_session not in session_ids

    response = client.delete(f"/sessions/{bob_session}", headers=_auth_header("alice"))
    assert response.status_code == 403


def test_canvas_endpoints_require_owner(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _create_user("alice")
    _create_user("bob")
    client = _build_client()

    canvas = create_canvas(topic="secure canvas", user_id="alice")

    response = client.get(f"/canvas/{canvas.id}")
    assert response.status_code == 401

    response = client.get(f"/canvas/{canvas.id}/export", headers=_auth_header("bob"))
    assert response.status_code == 403

    response = client.patch(
        f"/canvas/{canvas.id}",
        json={"working_title": "updated"},
        headers=_auth_header("alice"),
    )
    assert response.status_code == 200
    assert response.json()["working_title"] == "updated"


class _FakeQueue:
    def __init__(self):
        self.states = {
            "task-alice-active": TaskState(
                task_id="task-alice-active",
                kind=TaskKind.chat,
                status=TaskStatus.running,
                session_id="session-a",
                user_id="alice",
            ),
            "task-bob-active": TaskState(
                task_id="task-bob-active",
                kind=TaskKind.dr,
                status=TaskStatus.running,
                session_id="session-b",
                user_id="bob",
            ),
            "task-alice-queued": TaskState(
                task_id="task-alice-queued",
                kind=TaskKind.chat,
                status=TaskStatus.queued,
                session_id="session-a",
                user_id="alice",
            ),
        }
        self.cancelled: list[str] = []

    def get_queue_snapshot(self):
        return {
            "active_count": 2,
            "max_slots": 2,
            "active": [
                self.states["task-alice-active"].to_dict(),
                self.states["task-bob-active"].to_dict(),
            ],
            "queued": [
                {
                    "task_id": "task-bob-active",
                    "kind": "dr",
                    "session_id": "session-b",
                    "user_id": "bob",
                    "queue_position": 1,
                    "state": self.states["task-bob-active"].to_dict(),
                },
                {
                    "task_id": "task-alice-queued",
                    "kind": "chat",
                    "session_id": "session-a",
                    "user_id": "alice",
                    "queue_position": 2,
                    "state": self.states["task-alice-queued"].to_dict(),
                },
            ],
        }

    def get_state(self, task_id: str):
        return self.states.get(task_id)

    def cancel_queued(self, task_id: str) -> bool:
        self.cancelled.append(task_id)
        return True

    def cancel_running(self, task_id: str) -> bool:
        self.cancelled.append(task_id)
        return True


def test_task_queue_and_cancel_are_scoped(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _create_user("alice")
    _create_user("bob")
    client = _build_client()
    fake_queue = _FakeQueue()
    monkeypatch.setattr("src.api.routes_tasks.get_task_queue", lambda: fake_queue)

    response = client.get("/tasks/queue")
    assert response.status_code == 401

    response = client.get("/tasks/queue", headers=_auth_header("alice"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["active_count"] == 1
    assert [item["task_id"] for item in payload["active"]] == ["task-alice-active"]
    assert [item["task_id"] for item in payload["queued"]] == ["task-alice-queued"]
    assert payload["queued"][0]["queue_position"] == 1

    response = client.post("/tasks/task-alice-queued/cancel", headers=_auth_header("bob"))
    assert response.status_code == 403

    response = client.post("/tasks/task-alice-queued/cancel", headers=_auth_header("alice"))
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert fake_queue.cancelled == ["task-alice-queued"]
