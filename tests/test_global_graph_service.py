from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sqlmodel import Session

import src.db.engine as db_engine
from src.db.engine import get_engine, init_db
from src.db.models import Paper, PaperMetadata, UserProfile
from src.retrieval.dedup import compute_paper_uid, normalize_doi, normalize_title
from src.services import global_graph_service as graph_service_module
from src.services.global_graph_service import GlobalGraphService


def _init_test_db(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "graph.db"
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{db_path}")
    db_engine._engine = None
    graph_service_module._global_graph_service = None
    init_db()


def _seed_scope_data() -> list[tuple[Paper, PaperMetadata]]:
    p1_uid = compute_paper_uid(doi="10.1000/a")
    p2_uid = compute_paper_uid(doi="10.1000/b")
    rows = []
    rows.append(
        (
            Paper(
                user_id="alice",
                collection="reef",
                paper_id="paper-a",
                filename="paper-a.pdf",
                file_path="/tmp/paper-a.pdf",
                paper_uid=p1_uid,
            ),
            PaperMetadata(
                paper_id="paper-a",
                doi="10.1000/a",
                normalized_doi=normalize_doi("10.1000/a"),
                title="Paper A",
                normalized_title=normalize_title("Paper A"),
                authors=json.dumps(["Alice Smith", "Bob Lee"]),
                year=2024,
                source="test",
                extra=json.dumps(
                    {
                        "authorships": [
                            {
                                "author": {"display_name": "Alice Smith", "id": "https://openalex.org/A1"},
                                "institutions": [{"display_name": "Lab A", "id": "https://openalex.org/I1"}],
                            },
                            {
                                "author": {"display_name": "Bob Lee", "id": "https://openalex.org/A2"},
                                "institutions": [{"display_name": "Lab A", "id": "https://openalex.org/I1"}],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                paper_uid=p1_uid,
            ),
        )
    )
    rows.append(
        (
            Paper(
                user_id="alice",
                collection="reef",
                paper_id="paper-b",
                filename="paper-b.pdf",
                file_path="/tmp/paper-b.pdf",
                paper_uid=p2_uid,
            ),
            PaperMetadata(
                paper_id="paper-b",
                doi="10.1000/b",
                normalized_doi=normalize_doi("10.1000/b"),
                title="Paper B",
                normalized_title=normalize_title("Paper B"),
                authors=json.dumps(["Alice Smith", "Carol Tan"]),
                year=2025,
                source="test",
                extra=json.dumps(
                    {
                        "references": [{"doi": "10.1000/a", "title": "Paper A"}],
                        "authorships": [
                            {
                                "author": {"display_name": "Alice Smith", "id": "https://openalex.org/A1"},
                                "institutions": [{"display_name": "Lab A", "id": "https://openalex.org/I1"}],
                            },
                            {
                                "author": {"display_name": "Carol Tan", "id": "https://openalex.org/A3"},
                                "institutions": [{"display_name": "Lab B", "id": "https://openalex.org/I2"}],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                paper_uid=p2_uid,
            ),
        )
    )
    return rows


def test_global_graph_service_builds_author_and_citation_snapshots(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        graph_service_module,
        "settings",
        SimpleNamespace(
            path=SimpleNamespace(data=data_root),
            openalex=SimpleNamespace(enabled=False, api_key="", base_url="https://api.openalex.org", timeout_seconds=10),
        ),
    )

    with Session(get_engine()) as session:
        session.add(UserProfile(user_id="alice", role="user", password_hash="", preferences_json="{}"))
        for paper_row, meta_row in _seed_scope_data():
            session.add(paper_row)
            session.add(meta_row)
        session.commit()

    svc = GlobalGraphService()
    monkeypatch.setattr(svc.enricher, "build_facts", lambda graph_type, scope, papers: [])
    scope = {"user_id": "alice", "scope_type": "collection", "scope_key": "reef"}

    author_snapshot = svc.ensure_snapshot("author", scope, refresh=False)
    assert author_snapshot["status"] == "ready"
    assert author_snapshot["snapshot_version"] == 1

    author_subgraph = svc.query_subgraph(
        "author",
        scope,
        seeds={"node_ids": ["author:A1"]},
        depth=2,
        limit=20,
    )
    assert any(edge["relation"] == "authored" for edge in author_subgraph["edges"])
    assert any(edge["relation"] == "co_author" for edge in author_subgraph["edges"])
    assert any(node["id"] == "paper:doi:10.1000/a" for node in author_subgraph["nodes"])

    svc.mark_scope_stale(scope, "test_stale", graph_types=["author"])
    rebuilt_snapshot = svc.ensure_snapshot("author", scope, refresh=False)
    assert rebuilt_snapshot["snapshot_version"] == 2

    citation_snapshot = svc.ensure_snapshot("citation", scope, refresh=False)
    assert citation_snapshot["status"] == "ready"
    citation_subgraph = svc.query_subgraph(
        "citation",
        scope,
        seeds={"paper_uids": [compute_paper_uid(doi="10.1000/b")]},
        depth=2,
        limit=20,
    )
    assert any(edge["relation"] == "cites" for edge in citation_subgraph["edges"])
    assert any(node["id"] == "paper:doi:10.1000/a" for node in citation_subgraph["nodes"])
