from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes_graph import router as graph_router
from src.auth.session import create_token
from src.llm import tools as llm_tools


class _FakeGraphService:
    def __init__(self):
        self.calls = []

    def graph_stats(self, graph_type, scope):
        return {
            "available": True,
            "graph_type": graph_type,
            "scope": scope,
            "fact_count": 2,
            "total_nodes": 3,
            "total_edges": 2,
            "snapshot_version": 7,
            "snapshot_status": "ready",
        }

    def query_subgraph(self, graph_type, scope, seeds, depth, limit, snapshot_version=None):
        return {
            "nodes": [
                {"id": "paper:doi:10.1000/a", "type": "paper", "label": "Paper A", "pagerank": 0.4, "degree": 2},
                {"id": "author:A1", "type": "author", "label": "Alice", "pagerank": 0.3, "degree": 1, "is_seed": True},
            ],
            "edges": [
                {"source": "author:A1", "target": "paper:doi:10.1000/a", "relation": "authored", "weight": 1.0},
            ],
            "metrics": {"node_count": 2, "edge_count": 1, "top_nodes": [{"id": "paper:doi:10.1000/a", "score": 0.4}], "bridge_nodes": []},
            "snapshot_version": snapshot_version or 7,
            "provenance": [{"source": "local", "kind": "test"}],
        }

    def summarize_subgraph(self, graph_type, scope, seeds, **kwargs):
        self.calls.append({"graph_type": graph_type, "scope": dict(scope), "seeds": dict(seeds), "kwargs": dict(kwargs)})
        return {
            "summary": f"# {graph_type}\n- scope: {scope['scope_type']}:{scope['scope_key']}\n- ok",
            "snapshot_version": 7,
            "provenance": [{"source": "local", "kind": "test"}],
            "subgraph": self.query_subgraph(graph_type, scope, seeds, depth=1, limit=10),
        }

    def list_snapshots(self, graph_type, scope):
        return [{"graph_type": graph_type, "snapshot_version": 7, "status": "ready"}]

    def ensure_snapshot(self, graph_type, scope, refresh=False):
        return {"graph_type": graph_type, "snapshot_version": 8 if refresh else 7, "status": "ready"}


def _auth_header(user_id: str = "alice") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_token(user_id)}"}


def test_graph_typed_routes_use_global_graph_service(monkeypatch):
    app = FastAPI()
    app.include_router(graph_router)
    client = TestClient(app)
    monkeypatch.setattr("src.api.routes_graph.get_global_graph_service", lambda: _FakeGraphService())

    response = client.post(
        "/graph/author/subgraph",
        json={
            "scope": {"scope_type": "collection", "scope_key": "reef"},
            "seed_node_ids": ["author:A1"],
            "depth": 2,
            "limit": 10,
        },
        headers=_auth_header(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["snapshot_version"] == 7
    assert body["nodes"][0]["id"] == "paper:doi:10.1000/a"

    response = client.post(
        "/graph/author/summary",
        json={
            "subgraph_request": {
                "scope": {"scope_type": "collection", "scope_key": "reef"},
                "seed_node_ids": ["author:A1"],
                "depth": 1,
                "limit": 10,
            },
            "question": "Who is central?",
        },
        headers=_auth_header(),
    )
    assert response.status_code == 200
    assert response.json()["summary"].startswith("# author")


def test_explore_academic_graph_tool_uses_collection_scope(monkeypatch):
    fake = _FakeGraphService()
    monkeypatch.setattr("src.services.global_graph_service.get_global_graph_service", lambda: fake)
    monkeypatch.setattr(
        "src.services.collection_library_binding_service.resolve_bound_library_for_collection",
        lambda *args, **kwargs: None,
    )
    llm_tools.set_tool_collection("reef")
    llm_tools.set_tool_user_id("alice")

    result = llm_tools._handle_explore_academic_graph(
        graph_type="citation",
        paper_uid="doi:10.1000/a",
    )

    assert "# citation" in result
    assert "collection:reef" in result
    assert fake.calls[0]["scope"]["user_id"] == "alice"
    assert fake.calls[0]["scope"]["scope_type"] == "collection"
    assert fake.calls[0]["scope"]["scope_key"] == "reef"


def test_explore_academic_graph_tool_falls_back_to_library(monkeypatch):
    class _FallbackGraphService(_FakeGraphService):
        def summarize_subgraph(self, graph_type, scope, seeds, **kwargs):
            self.calls.append({"graph_type": graph_type, "scope": dict(scope)})
            if scope["scope_type"] == "collection":
                return {
                    "summary": "# citation\n- scope: collection:reef\n- empty",
                    "snapshot_version": 7,
                    "provenance": [],
                    "subgraph": {
                        "nodes": [],
                        "edges": [],
                        "metrics": {"node_count": 0, "edge_count": 0, "top_nodes": [], "bridge_nodes": []},
                    },
                }
            return {
                "summary": f"# citation\n- scope: {scope['scope_type']}:{scope['scope_key']}\n- ok",
                "snapshot_version": 8,
                "provenance": [{"source": "local", "kind": "test"}],
                "subgraph": {
                    "nodes": [{"id": "paper:doi:10.1000/a", "type": "paper", "label": "Paper A"}],
                    "edges": [],
                    "metrics": {"node_count": 1, "edge_count": 0, "top_nodes": [], "bridge_nodes": []},
                },
            }

    fake = _FallbackGraphService()

    class _BoundLibrary:
        id = 42

    monkeypatch.setattr("src.services.global_graph_service.get_global_graph_service", lambda: fake)
    monkeypatch.setattr(
        "src.services.collection_library_binding_service.resolve_bound_library_for_collection",
        lambda *args, **kwargs: _BoundLibrary(),
    )
    llm_tools.set_tool_collection("reef")
    llm_tools.set_tool_user_id("alice")

    result = llm_tools._handle_explore_academic_graph(
        graph_type="citation",
        paper_uid="doi:10.1000/a",
    )

    assert "library:42" in result
    assert [call["scope"]["scope_type"] for call in fake.calls[:2]] == ["collection", "library"]


# ---------------------------------------------------------------------------
# snapshot_version strict validation
# ---------------------------------------------------------------------------

def test_query_subgraph_snapshot_version_not_found_raises():
    """When an explicit snapshot_version is requested but not found/ready,
    query_subgraph must raise ValueError (→ HTTP 400 via route handler)."""
    from src.services.global_graph_service import GlobalGraphService

    class _FakeSession:
        def exec(self, stmt):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    import networkx as nx

    svc = GlobalGraphService.__new__(GlobalGraphService)

    # Patch helpers so we can call query_subgraph without a real DB
    from src.services.global_graph_service import GraphScope

    def _fake_normalize_scope(d):
        return GraphScope(user_id=d.get("user_id", "u"), scope_type="global", scope_key="global")

    def _fake_ensure_snapshot(graph_type, scope, refresh=False):
        return {"snapshot_version": 7, "status": "ready"}

    def _fake_query_latest_snapshot_rows(graph_type, scope):
        # Return an empty list — no rows → version 99 cannot be found
        return []

    svc._normalize_scope = lambda d: _fake_normalize_scope(d)
    svc.ensure_snapshot = lambda graph_type, scope, refresh=False: _fake_ensure_snapshot(graph_type, scope, refresh)
    svc._query_latest_snapshot_rows = lambda graph_type, scope: _fake_query_latest_snapshot_rows(graph_type, scope)
    svc.entity_adapter = None

    with pytest.raises(ValueError, match="snapshot_version=99"):
        svc.query_subgraph(
            graph_type="citation",
            scope={"user_id": "alice", "scope_type": "global", "scope_key": "global"},
            seeds={},
            depth=1,
            limit=10,
            snapshot_version=99,
        )


def test_query_subgraph_no_version_falls_back_gracefully(monkeypatch):
    """When no snapshot_version is given and no ready row exists, return empty subgraph."""
    from src.services.global_graph_service import GlobalGraphService, GraphScope

    svc = GlobalGraphService.__new__(GlobalGraphService)
    svc._normalize_scope = lambda d: GraphScope(user_id="u", scope_type="global", scope_key="global")
    svc.ensure_snapshot = lambda *a, **kw: {"snapshot_version": 1}
    svc._query_latest_snapshot_rows = lambda *a: []
    svc.entity_adapter = None

    result = svc.query_subgraph(
        graph_type="citation",
        scope={"user_id": "alice", "scope_type": "global", "scope_key": "global"},
        seeds={},
        depth=1,
        limit=10,
    )
    assert result["nodes"] == []
    assert result["edges"] == []


def test_subgraph_route_returns_400_on_invalid_snapshot_version(monkeypatch):
    """The HTTP route must surface the ValueError as 400."""
    app = FastAPI()
    app.include_router(graph_router)

    class _ErrorGraphService(_FakeGraphService):
        def query_subgraph(self, graph_type, scope, seeds, depth, limit, snapshot_version=None):
            if snapshot_version is not None:
                raise ValueError(f"snapshot_version={snapshot_version} not found or not ready")
            return super().query_subgraph(graph_type, scope, seeds, depth, limit)

    monkeypatch.setattr("src.api.routes_graph.get_global_graph_service", lambda: _ErrorGraphService())
    client = TestClient(app)

    response = client.post(
        "/graph/citation/subgraph",
        json={
            "scope": {"scope_type": "global", "scope_key": "global"},
            "snapshot_version": 99,
            "depth": 1,
            "limit": 10,
        },
        headers=_auth_header(),
    )
    assert response.status_code == 400
    assert "snapshot_version=99" in response.json()["detail"]
