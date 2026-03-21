from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

import src.db.engine as db_engine
from src.api.routes_academic_assistant import router as aa_router
from src.auth.session import create_token
from src.db.engine import get_engine, init_db
from src.db.models import Paper
from src.indexing import assistant_artifact_store
from src.llm import tools as llm_tools
from src.retrieval.dedup import compute_paper_uid
from src.tasks.task_state import TaskKind, TaskState, TaskStatus


def _init_test_db(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "assistant.db"
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{db_path}")
    db_engine._engine = None
    init_db()


class _FakeDenseVector(list):
    def tolist(self):
        return list(self)


class _FakeSparseMatrix:
    class _Row:
        def tocoo(self):
            return SimpleNamespace(col=[], data=[])

    def _getrow(self, _idx):
        return self._Row()


class _FakeMilvus:
    def __init__(self):
        self.upserts = []
        self.deletes = []
        self.client = SimpleNamespace(
            describe_collection=lambda _name: {"fields": []},
            has_collection=lambda _name: True,
            delete=self._delete,
        )

    def _delete(self, *, collection_name, filter):
        self.deletes.append((collection_name, filter))
        return {"delete_count": 1}

    def upsert(self, collection_name, data):
        self.upserts.append((collection_name, data))


def test_resource_annotation_store_upserts_and_cleans(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    paper_uid = compute_paper_uid(doi="10.1000/assistant-a")

    with Session(get_engine()) as session:
        session.add(
            Paper(
                user_id="alice",
                collection="reef",
                paper_id="paper-a",
                filename="paper-a.pdf",
                file_path="/tmp/paper-a.pdf",
                paper_uid=paper_uid,
            )
        )
        session.commit()

    fake_milvus = _FakeMilvus()
    monkeypatch.setattr(assistant_artifact_store, "milvus", fake_milvus)
    monkeypatch.setattr(
        assistant_artifact_store,
        "embedder",
        SimpleNamespace(encode=lambda texts: {"dense": [_FakeDenseVector([0.1, 0.2]) for _ in texts], "sparse": _FakeSparseMatrix()}),
    )

    row = assistant_artifact_store.upsert_resource_annotation(
        user_id="alice",
        resource_type="paper",
        resource_id="paper-a",
        paper_uid=paper_uid,
        target_kind="figure",
        target_locator={"figure_id": "fig-1", "page": 3, "bbox": [1, 2, 3, 4]},
        target_text="Figure 1 compares treatments.",
        directive="Highlight the treatment contrast.",
        status="active",
        collection="reef",
    )

    assert row.id is not None
    assert fake_milvus.upserts
    upsert_rows = fake_milvus.upserts[0][1]
    assert upsert_rows[0]["content_type"] == "annotation"
    assert upsert_rows[0]["paper_uid"] == paper_uid

    cleanup = assistant_artifact_store.cleanup_assistant_artifacts_for_paper(
        user_id="alice",
        collection="reef",
        paper_id="paper-a",
        paper_uid=paper_uid,
    )
    assert cleanup["deleted_annotations"] == 1
    assert any("content_type == \"annotation\"" in item[1] for item in fake_milvus.deletes)


class _FakeAssistantService:
    def __init__(self):
        self.calls = []

    def summarize_paper(self, locator, **kwargs):
        self.calls.append(("summarize", locator, kwargs))
        return {"summary_md": "# Summary", "citations": [], "evidence_summary": {"total_chunks": 2}}

    def ask_paper(self, locator, **kwargs):
        self.calls.append(("ask", locator, kwargs))
        return {"answer_md": "Answer body", "citations": [], "evidence_summary": {"total_chunks": 2}}

    def compare_papers(self, paper_uids, **kwargs):
        self.calls.append(("compare", paper_uids, kwargs))
        return {"narrative": "# Compare", "comparison_matrix": {}, "citations": [], "evidence_summary": {"total_chunks": 3}}

    def analyze_paper_media(self, paper_uids, **kwargs):
        self.calls.append(("media", paper_uids, kwargs))
        return {"summary_md": "- paper: ok", "items": []}

    def discover(self, mode, **kwargs):
        self.calls.append(("discover", mode, kwargs))
        return {"summary_md": "# Discovery", "items": [{"label": "Alice"}], "citations": [], "provenance": []}


class _FakeQueue:
    def __init__(self):
        self.states = {}

    def set_state(self, state):
        self.states[state.task_id] = state

    def get_state(self, task_id):
        return self.states.get(task_id)

    def push_event(self, task_id, event_type, data):
        return None

    def read_events(self, task_id, after_id="-", count=100):
        return []


class _DummyTask:
    def add_done_callback(self, callback):
        return None

    def cancel(self):
        return None


def _fake_create_task(coro, name=None):
    coro.close()
    return _DummyTask()


def test_academic_assistant_routes_and_tools(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    import src.api.routes_academic_assistant as routes_academic_assistant

    fake_service = _FakeAssistantService()

    monkeypatch.setattr(routes_academic_assistant, "get_reference_assistant_service", lambda: fake_service)
    monkeypatch.setattr(routes_academic_assistant, "get_task_queue", lambda: _FakeQueue())
    monkeypatch.setattr(routes_academic_assistant.asyncio, "create_task", _fake_create_task)

    response = routes_academic_assistant.summarize_paper(
        routes_academic_assistant.PaperSummaryRequest(locator=routes_academic_assistant.PaperLocatorPayload(paper_uid="doi:10.1000/a")),
        user_id="alice",
    )
    assert response["summary_md"] == "# Summary"

    response = routes_academic_assistant.ask_paper(
        routes_academic_assistant.PaperQuestionRequest(
            locator=routes_academic_assistant.PaperLocatorPayload(paper_uid="doi:10.1000/a"),
            question="What is the main result?",
        ),
        user_id="alice",
    )
    assert response["answer_md"] == "Answer body"

    response = routes_academic_assistant.compare_papers(
        routes_academic_assistant.PaperCompareRequest(paper_uids=["doi:10.1000/a", "doi:10.1000/b"]),
        user_id="alice",
    )
    assert response["narrative"] == "# Compare"

    monkeypatch.setattr("src.services.reference_assistant_service.get_reference_assistant_service", lambda: fake_service)
    llm_tools.set_tool_collection("reef")
    llm_tools.set_tool_user_id("alice")

    assert llm_tools._handle_summarize_paper(paper_uid="doi:10.1000/a") == "# Summary"
    assert llm_tools._handle_ask_paper(question="What changed?", paper_uid="doi:10.1000/a") == "Answer body"
    assert "paper: ok" in llm_tools._handle_analyze_paper_media(paper_uids=["doi:10.1000/a"])
    assert "# Discovery" in llm_tools._handle_discover_academic_resources(mode="experts", paper_uids=["doi:10.1000/a"])


# ---------------------------------------------------------------------------
# Discovery mode normalisation and canonical return value
# ---------------------------------------------------------------------------

class _DiscoveryAssistantService:
    """Minimal stub that echoes the mode it receives so we can assert normalisation."""

    def discover(self, mode, **kwargs):
        return {
            "mode": mode,
            "items": [],
            "summary_md": f"mode={mode}",
            "citations": [],
            "provenance": [],
        }


def test_discover_mode_normalised_in_service():
    """discover() must normalise kebab-case to underscore before matching."""
    from src.services.reference_assistant_service import ReferenceAssistantService

    class _StubGraphService:
        def query_subgraph(self, *a, **kw):
            return {"nodes": [], "edges": []}

        def summarize_subgraph(self, *a, **kw):
            return {"summary": "", "provenance": []}

        def _normalize_scope(self, d):
            from src.services.global_graph_service import GraphScope
            return GraphScope(**d)

        class fact_builder:
            @staticmethod
            def collect_scope_papers(scope):
                return []

    svc = ReferenceAssistantService.__new__(ReferenceAssistantService)
    svc.graph_service = _StubGraphService()

    with pytest.raises(ValueError, match="unsupported"):
        svc.discover("UNKNOWN_MODE", user_id="alice", seeds={}, scope={"scope_type": "global", "scope_key": "global"})


@pytest.mark.parametrize("kebab,underscore", [
    ("missing-core", "missing_core"),
    ("forward-tracking", "forward_tracking"),
    ("experts", "experts"),
    ("institutions", "institutions"),
])
def test_discover_mode_return_is_canonical(monkeypatch, kebab, underscore):
    """discover() must return the canonical (underscore) mode in the result dict."""
    from src.services import reference_assistant_service as ras_module

    class _FakeGraphSvc:
        class fact_builder:
            @staticmethod
            def collect_scope_papers(_):
                return []

        def _normalize_scope(self, d):
            from src.services.global_graph_service import GraphScope
            return GraphScope(**d)

        def query_subgraph(self, *a, **kw):
            return {"nodes": [], "edges": []}

        def summarize_subgraph(self, *a, **kw):
            return {"summary": "", "provenance": []}

        class enricher:
            @staticmethod
            def _resolve_openalex_work(_):
                return {}

            @staticmethod
            def _openalex_fetch_json(_):
                return {"results": []}

    svc = ras_module.ReferenceAssistantService.__new__(ras_module.ReferenceAssistantService)
    svc.graph_service = _FakeGraphSvc()

    result = svc.discover(
        kebab,
        user_id="alice",
        seeds={"paper_uids": [], "node_ids": []},
        scope={"scope_type": "global", "scope_key": "global"},
        options={"limit": 5},
    )
    assert result["mode"] == underscore, f"Expected {underscore!r}, got {result['mode']!r}"


# ---------------------------------------------------------------------------
# Task endpoint user-isolation
# ---------------------------------------------------------------------------

def _build_aa_app(fake_queue):
    app = FastAPI()
    import src.api.routes_academic_assistant as aam
    monkeypatched_module = aam
    app.include_router(aa_router)
    return app


def test_task_stream_rejects_foreign_user(monkeypatch, tmp_path):
    _init_test_db(monkeypatch, tmp_path)
    import src.api.routes_academic_assistant as aam
    import time

    q = _FakeQueue()
    state = TaskState(
        task_id="t-alice-1",
        kind=TaskKind.academic_assistant,
        status=TaskStatus.running,
        user_id="alice",
        started_at=time.time(),
        payload={},
    )
    q.set_state(state)
    monkeypatch.setattr(aam, "get_task_queue", lambda: q)

    app = FastAPI()
    app.include_router(aa_router)
    client = TestClient(app, raise_server_exceptions=False)

    # bob must get 403
    res = client.get(
        "/academic-assistant/task/t-alice-1/stream",
        headers={"Authorization": f"Bearer {create_token('bob')}"},
    )
    assert res.status_code == 403

    # alice must not get 403 (may get other errors from SSE, but not auth)
    res2 = client.get(
        "/academic-assistant/task/t-alice-1/stream",
        headers={"Authorization": f"Bearer {create_token('alice')}"},
    )
    assert res2.status_code != 403


def test_task_status_rejects_foreign_user(monkeypatch, tmp_path):
    _init_test_db(monkeypatch, tmp_path)
    import src.api.routes_academic_assistant as aam
    import time

    q = _FakeQueue()
    state = TaskState(
        task_id="t-alice-2",
        kind=TaskKind.academic_assistant,
        status=TaskStatus.completed,
        user_id="alice",
        started_at=time.time(),
        payload={"result": {"summary_md": "ok"}},
    )
    q.set_state(state)
    monkeypatch.setattr(aam, "get_task_queue", lambda: q)

    app = FastAPI()
    app.include_router(aa_router)
    client = TestClient(app)

    # bob must get 403
    res = client.get(
        "/academic-assistant/task/t-alice-2",
        headers={"Authorization": f"Bearer {create_token('bob')}"},
    )
    assert res.status_code == 403

    # alice must get the task
    res2 = client.get(
        "/academic-assistant/task/t-alice-2",
        headers={"Authorization": f"Bearer {create_token('alice')}"},
    )
    assert res2.status_code == 200
    assert res2.json()["task_id"] == "t-alice-2"


# ---------------------------------------------------------------------------
# SSE fallback: completed state → `done` event
# ---------------------------------------------------------------------------

def test_sse_fallback_completed_emits_done():
    """When the task is terminal/completed and no done event is queued,
    _task_event_stream must synthesise an ``event: done`` frame."""
    import src.api.routes_academic_assistant as aam
    import time

    class _TerminalQueue:
        def get_state(self, task_id):
            s = TaskState(
                task_id=task_id,
                kind=TaskKind.academic_assistant,
                status=TaskStatus.completed,
                user_id="alice",
                started_at=time.time() - 1,
                payload={"result": {"summary_md": "done!"}},
            )
            s.finished_at = time.time()
            return s

        def read_events(self, task_id, after_id="-", count=100):
            return []

    frames = list(aam._task_event_stream("x", _TerminalQueue()))
    joined = "".join(frames)
    assert "event: done" in joined
    assert "completed" not in joined  # must NOT emit the raw status name
