from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

import src.db.engine as db_engine
from src.api import routes_project, routes_resources, routes_scholar
from src.db.engine import get_engine, init_db
from src.db.models import (
    Canvas,
    Paper,
    ResourceAnnotation,
    ResourceNote,
    ResourceTag,
    ResourceUserState,
    ScholarLibrary,
    ScholarLibraryPaper,
)
from src.indexing import paper_store
from src.services.resource_state_service import get_resource_state_service


def _init_test_db(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "resource-state.db"
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{db_path}")
    db_engine._engine = None
    init_db()


def _seed_base_data(tmp_path: Path) -> Tuple[str, int, int]:
    pdf_path = tmp_path / "reef" / "pdfs" / "paper-a.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_text("paper-a")
    parsed_path = pdf_path.parent.parent / "parsed_data" / "paper-a.json"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text("{}")

    with Session(get_engine()) as session:
        session.add(
            Canvas(
                id="canvas-1",
                user_id="alice",
                topic="Vent microbiome",
            )
        )
        session.add(
            Paper(
                user_id="alice",
                collection="reef",
                paper_id="paper-a",
                filename="paper-a.pdf",
                file_path=str(pdf_path),
                paper_uid="doi:10.1000/a",
            )
        )
        lib = ScholarLibrary(user_id="alice", name="reef-lib", description="library")
        session.add(lib)
        session.commit()
        session.refresh(lib)

        lib_paper = ScholarLibraryPaper(
            library_id=int(lib.id),
            title="Paper A",
            paper_uid="doi:10.1000/a",
            collection_name="reef",
            collection_paper_id="paper-a",
        )
        session.add(lib_paper)
        session.commit()
        session.refresh(lib_paper)

        annotation = ResourceAnnotation(
            user_id="alice",
            resource_type="paper",
            resource_id="paper-a",
            paper_uid="doi:10.1000/a",
            target_kind="chunk",
            target_locator_json="{}",
            target_text="Key sentence",
            directive="revise this",
            status="active",
            created_at="2026-03-15T00:00:00",
            updated_at="2026-03-15T00:00:00",
        )
        session.add(annotation)
        session.commit()
        session.refresh(annotation)
        return str(pdf_path), int(lib.id), int(lib_paper.id), int(annotation.id)


def test_project_alias_routes_and_archived_compat(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _pdf_path, _lib_id, _lib_paper_id, _annotation_id = _seed_base_data(tmp_path)

    resp = routes_resources.patch_resource_state(
        routes_resources.ResourceUserStateUpsert(
            resource_type="project",
            resource_id="canvas-1",
            archived=True,
            favorite=True,
            read_status="reading",
        ),
        user_id="alice",
    )
    assert resp["resource_type"] == "canvas"
    assert resp["archived"] is True
    assert resp["favorite"] is True
    assert resp["read_status"] == "reading"

    items = routes_project.list_projects(include_archived=False, user_id="alice")
    assert items == []

    with pytest.raises(HTTPException) as exc:
        routes_project.delete_project("canvas-1", user_id="alice")
    assert exc.value.status_code == 400

    unarchived = routes_project.unarchive_project("canvas-1", user_id="alice")
    assert unarchived["archived"] is False
    items = routes_project.list_projects(include_archived=False, user_id="alice")
    assert len(items) == 1
    assert items[0]["archived"] is False


def test_annotation_supports_tags_and_notes_but_not_user_state(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _pdf_path, _lib_id, _lib_paper_id, annotation_id = _seed_base_data(tmp_path)

    first = routes_resources.create_resource_tag(
        routes_resources.ResourceTagUpsert(
            resource_type="resource_annotation",
            resource_id=str(annotation_id),
            tag=" AI ",
        ),
        user_id="alice",
    )
    second = routes_resources.create_resource_tag(
        routes_resources.ResourceTagUpsert(
            resource_type="resource_annotation",
            resource_id=str(annotation_id),
            tag="ai",
        ),
        user_id="alice",
    )
    assert first["id"] == second["id"]

    note = routes_resources.create_resource_note(
        routes_resources.ResourceNoteCreate(
            resource_type="resource_annotation",
            resource_id=str(annotation_id),
            note_md="This annotation needs follow-up.",
        ),
        user_id="alice",
    )
    listed_tags = routes_resources.list_resource_tags(
        resource_type="resource_annotation",
        resource_id=str(annotation_id),
        user_id="alice",
    )
    listed_notes = routes_resources.list_resource_notes(
        resource_type="resource_annotation",
        resource_id=str(annotation_id),
        user_id="alice",
    )
    assert len(listed_tags["items"]) == 1
    assert listed_notes["items"][0]["id"] == note["id"]

    with pytest.raises(HTTPException) as exc:
        routes_resources.patch_resource_state(
            routes_resources.ResourceUserStateUpsert(
                resource_type="resource_annotation",
                resource_id=str(annotation_id),
                archived=True,
            ),
            user_id="alice",
        )
    assert exc.value.status_code == 400


def test_delete_hooks_clean_paper_and_library_overlays(monkeypatch, tmp_path: Path):
    _init_test_db(monkeypatch, tmp_path)
    _pdf_path, lib_id, lib_paper_id, annotation_id = _seed_base_data(tmp_path)
    service = get_resource_state_service()

    service.upsert_user_state(user_id="alice", resource_type="paper", resource_id="doi:10.1000/a", favorite=True)
    service.add_tag(user_id="alice", resource_type="paper", resource_id="doi:10.1000/a", tag="review")
    service.create_note(user_id="alice", resource_type="paper", resource_id="doi:10.1000/a", note_md="paper note")

    service.upsert_user_state(user_id="alice", resource_type="scholar_library_paper", resource_id=str(lib_paper_id), archived=True)
    service.add_tag(user_id="alice", resource_type="scholar_library_paper", resource_id=str(lib_paper_id), tag="candidate")
    service.create_note(
        user_id="alice",
        resource_type="scholar_library_paper",
        resource_id=str(lib_paper_id),
        note_md="library note",
    )

    service.add_tag(user_id="alice", resource_type="resource_annotation", resource_id=str(annotation_id), tag="todo")
    service.create_note(
        user_id="alice",
        resource_type="resource_annotation",
        resource_id=str(annotation_id),
        note_md="annotation note",
    )

    assert paper_store.delete_paper("reef", "paper-a") is True
    routes_scholar.remove_paper_from_scholar_library(lib_id, lib_paper_id, user_id="alice")

    with Session(get_engine()) as session:
        paper_states = session.exec(
            select(ResourceUserState).where(
                ResourceUserState.user_id == "alice",
                ResourceUserState.resource_type == "paper",
                ResourceUserState.resource_id == "doi:10.1000/a",
            )
        ).all()
        lib_states = session.exec(
            select(ResourceUserState).where(
                ResourceUserState.user_id == "alice",
                ResourceUserState.resource_type == "scholar_library_paper",
                ResourceUserState.resource_id == str(lib_paper_id),
            )
        ).all()
        paper_tags = session.exec(
            select(ResourceTag).where(
                ResourceTag.user_id == "alice",
                ResourceTag.resource_type == "paper",
                ResourceTag.resource_id == "doi:10.1000/a",
            )
        ).all()
        paper_notes = session.exec(
            select(ResourceNote).where(
                ResourceNote.user_id == "alice",
                ResourceNote.resource_type == "paper",
                ResourceNote.resource_id == "doi:10.1000/a",
            )
        ).all()
        lib_tags = session.exec(
            select(ResourceTag).where(
                ResourceTag.user_id == "alice",
                ResourceTag.resource_type == "scholar_library_paper",
                ResourceTag.resource_id == str(lib_paper_id),
            )
        ).all()
        lib_notes = session.exec(
            select(ResourceNote).where(
                ResourceNote.user_id == "alice",
                ResourceNote.resource_type == "scholar_library_paper",
                ResourceNote.resource_id == str(lib_paper_id),
            )
        ).all()
        annotation_tags = session.exec(
            select(ResourceTag).where(
                ResourceTag.user_id == "alice",
                ResourceTag.resource_type == "resource_annotation",
                ResourceTag.resource_id == str(annotation_id),
            )
        ).all()
        annotation_notes = session.exec(
            select(ResourceNote).where(
                ResourceNote.user_id == "alice",
                ResourceNote.resource_type == "resource_annotation",
                ResourceNote.resource_id == str(annotation_id),
            )
        ).all()

    assert paper_states == []
    assert lib_states == []
    assert paper_tags == []
    assert paper_notes == []
    assert lib_tags == []
    assert lib_notes == []
    assert annotation_tags == []
    assert annotation_notes == []


# ---------------------------------------------------------------------------
# Project delete cleans overlays AND canvas annotations
# ---------------------------------------------------------------------------

def test_delete_project_cleans_annotations_and_overlays(monkeypatch, tmp_path):
    """delete_project must remove both resource overlays and ResourceAnnotation rows for the canvas."""
    _init_test_db(monkeypatch, tmp_path)

    from src.collaboration.canvas.canvas_manager import get_canvas_store
    from src.api import routes_project
    from src.indexing.assistant_artifact_store import (
        upsert_resource_annotation,
        delete_resource_annotations_for_resource,
    )
    from src.services.resource_state_service import get_resource_state_service

    canvas_id = "canvas-del-1"

    # Seed canvas and annotation
    with Session(get_engine()) as session:
        session.add(Canvas(id=canvas_id, user_id="alice", topic="Test Delete"))
        session.commit()

    # Add a canvas-level annotation
    annotation_row = upsert_resource_annotation(
        user_id="alice",
        resource_type="canvas",
        resource_id=canvas_id,
        paper_uid="",
        target_kind="canvas_section",
        target_locator={},
        target_text="canvas annotation",
        directive="please fix",
        status="active",
    )
    annotation_id = annotation_row.id

    # Add resource overlays
    svc = get_resource_state_service()
    svc.upsert_user_state(user_id="alice", resource_type="canvas", resource_id=canvas_id, favorite=True)
    svc.add_tag(user_id="alice", resource_type="canvas", resource_id=canvas_id, tag="important")

    # Stub out Canvas store methods so delete_project doesn't fail on missing sessions/jobs
    class _StubStore:
        def get_canvas_owner(self, cid):
            return "alice"

        def list_by_user(self, uid, include_archived=False):
            return [{"id": canvas_id, "archived": False}]

        def delete(self, cid):
            return True

    monkeypatch.setattr("src.api.routes_project.get_canvas_store", lambda: _StubStore())
    monkeypatch.setattr("src.api.routes_project.get_session_store", lambda: SimpleNamespace(delete_sessions_by_canvas_id=lambda *a: None))
    monkeypatch.setattr("src.api.routes_project.delete_jobs_by_canvas_id", lambda *a: None)
    monkeypatch.setattr("src.api.routes_project.delete_working_memory", lambda *a: None)
    monkeypatch.setattr("src.api.routes_project.delete_user_project", lambda *a: None)

    routes_project.delete_project(canvas_id=canvas_id, user_id="alice")

    with Session(get_engine()) as session:
        remaining_annotations = session.exec(
            select(ResourceAnnotation).where(
                ResourceAnnotation.user_id == "alice",
                ResourceAnnotation.resource_type == "canvas",
                ResourceAnnotation.resource_id == canvas_id,
            )
        ).all()
        remaining_states = session.exec(
            select(ResourceUserState).where(
                ResourceUserState.user_id == "alice",
                ResourceUserState.resource_type == "canvas",
                ResourceUserState.resource_id == canvas_id,
            )
        ).all()
        remaining_tags = session.exec(
            select(ResourceTag).where(
                ResourceTag.user_id == "alice",
                ResourceTag.resource_type == "canvas",
                ResourceTag.resource_id == canvas_id,
            )
        ).all()

    assert remaining_annotations == [], "canvas annotations must be deleted on project delete"
    assert remaining_states == [], "canvas user_state must be deleted on project delete"
    assert remaining_tags == [], "canvas tags must be deleted on project delete"
