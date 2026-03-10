"""Tests for scholar auto-ingest: user_id in ingest payload and scholar task lifecycle (ingest watcher)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.tasks.dispatcher import _trigger_ingest_pipeline, _watch_ingest_and_update_scholar_task
from src.tasks.task_state import TaskKind, TaskStatus, TaskState


def test_trigger_ingest_pipeline_includes_user_id_in_payload():
    """_trigger_ingest_pipeline must pass user_id in payload so ingest writes to correct user dir."""
    create_job_calls = []
    thread_args = []

    def fake_create_job(collection: str, payload: dict, total_files: int):
        create_job_calls.append((collection, payload, total_files))
        return {"job_id": "test-job-123"}

    def capture_thread(target=None, args=(), **kwargs):
        thread_args.append((target, args))
        return MagicMock()

    with patch("src.indexing.ingest_job_store.create_job", side_effect=fake_create_job), patch(
        "src.api.routes_ingest._run_ingest_job_safe"
    ), patch("src.tasks.dispatcher.threading.Thread", side_effect=capture_thread):
        out = _trigger_ingest_pipeline(
            filepath="/tmp/paper.pdf",
            paper_id="paper_1",
            collection_name="my_coll",
            metadata={"title": "T"},
            user_id="user_abc",
        )
    assert out == "test-job-123"
    assert len(create_job_calls) == 1
    _, payload, _ = create_job_calls[0]
    assert payload.get("user_id") == "user_abc"
    assert len(thread_args) == 1
    _, (job_id, runner_payload) = thread_args[0]
    assert runner_payload.get("user_id") == "user_abc"


def test_trigger_ingest_pipeline_default_user_id_when_none():
    """When user_id is None, payload gets 'default' for ingest runner fallback."""
    create_job_calls = []

    def fake_create_job(collection: str, payload: dict, total_files: int):
        create_job_calls.append(payload)
        return {"job_id": "job-xyz"}

    with patch("src.indexing.ingest_job_store.create_job", side_effect=fake_create_job), patch(
        "src.api.routes_ingest._run_ingest_job_safe"
    ), patch("src.tasks.dispatcher.threading.Thread"):
        _trigger_ingest_pipeline(
            filepath="/tmp/x.pdf",
            paper_id="x",
            collection_name="c",
            metadata={},
            user_id=None,
        )
    assert create_job_calls[0].get("user_id") == "default"


def test_watch_ingest_sets_scholar_task_completed_when_job_done():
    """When ingest job status is 'done', watcher sets scholar task to completed."""
    task_id = "scholar-task-1"
    ingest_job_id = "ingest-job-1"
    job_states = [{"status": "running"}, {"status": "done"}]
    get_job_call_count = [0]

    def fake_get_job(jid):
        assert jid == ingest_job_id
        get_job_call_count[0] += 1
        idx = min(get_job_call_count[0] - 1, len(job_states) - 1)
        return job_states[idx]

    state = TaskState(
        task_id=task_id,
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={"stage": "INGEST_QUEUED", "ingest_job_id": ingest_job_id},
    )
    q = MagicMock()
    q.get_state.return_value = state

    with patch("src.indexing.ingest_job_store.get_job", side_effect=fake_get_job), patch(
        "src.tasks.dispatcher.get_task_queue", return_value=q
    ):
        _watch_ingest_and_update_scholar_task(task_id, ingest_job_id)

    assert state.status == TaskStatus.completed
    assert state.finished_at is not None
    assert state.payload.get("stage") == "INGEST_DONE"
    q.set_state.assert_called()
    q.push_event.assert_called()


def test_watch_ingest_sets_scholar_task_error_when_job_error():
    """When ingest job status is 'error', watcher sets scholar task to error."""
    task_id = "scholar-task-2"
    ingest_job_id = "ingest-job-2"

    def fake_get_job(jid):
        return {"status": "error", "error_message": "Parse failed"}

    state = TaskState(
        task_id=task_id,
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={"stage": "INGEST_QUEUED"},
    )
    q = MagicMock()
    q.get_state.return_value = state

    with patch("src.indexing.ingest_job_store.get_job", side_effect=fake_get_job), patch(
        "src.tasks.dispatcher.get_task_queue", return_value=q
    ):
        _watch_ingest_and_update_scholar_task(task_id, ingest_job_id)

    assert state.status == TaskStatus.error
    assert "Parse failed" in (state.error_message or "")
    assert state.payload.get("stage") == "INGEST_FAILED"
    q.set_state.assert_called()
