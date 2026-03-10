"""
Tests for dispatcher.process_download_and_ingest when the download result
indicates a supplementary PDF.

Covers:
- No ingest pipeline triggered for supplementary
- No downloaded_at marked for supplementary
- Task stage set to SUPPLEMENTARY_SAVED
- ingest_triggered=False in response
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tasks.task_state import TaskKind, TaskStatus, TaskState


def _make_dl_result_supplementary(paper_id="supplementary_10.1234_test"):
    return {
        "success": True,
        "paper_id": paper_id,
        "filepath": f"/tmp/{paper_id}.pdf",
        "message": "已保存 supplementary PDF，未标记为主文已下载",
        "is_supplementary": True,
        "should_mark_downloaded": False,
        "should_auto_ingest": False,
        "primary_paper_id": "10.1234_test",
        "supplementary_reason": "obvious_phrase:supplementary material",
    }


def _make_dl_result_main(paper_id="10.1234_test"):
    return {
        "success": True,
        "paper_id": paper_id,
        "filepath": f"/tmp/{paper_id}.pdf",
        "message": "Sci-Hub",
        "is_supplementary": False,
        "should_mark_downloaded": True,
        "should_auto_ingest": True,
        "primary_paper_id": None,
        "supplementary_reason": None,
    }


def test_supplementary_download_does_not_trigger_ingest():
    """When dl_result.is_supplementary is True, _trigger_ingest_pipeline must not be called."""
    from src.tasks.dispatcher import process_download_and_ingest

    state = TaskState(
        task_id="t1",
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={},
    )
    q = MagicMock()
    q.get_state.return_value = state

    with (
        patch("src.retrieval.downloader.adapter.get_adapter") as mock_adapter_factory,
        patch("src.tasks.dispatcher.get_task_queue", return_value=q),
        patch("src.tasks.dispatcher._trigger_ingest_pipeline") as mock_ingest,
        patch("src.tasks.dispatcher._mark_library_paper_downloaded") as mock_mark,
    ):
        adapter_instance = MagicMock()
        adapter_instance.download_paper = AsyncMock(return_value=_make_dl_result_supplementary())
        mock_adapter_factory.return_value = adapter_instance

        result = asyncio.run(process_download_and_ingest(
            task_id="t1",
            paper_info={"title": "Test Paper", "doi": "10.1234/test"},
            collection="my_coll",
            user_id="user1",
            library_paper_id=42,
        ))

    mock_ingest.assert_not_called()
    mock_mark.assert_not_called()
    assert result.get("ingest_triggered") is False


def test_supplementary_task_stage_is_supplementary_saved():
    """Task state stage must become SUPPLEMENTARY_SAVED for supplementary downloads."""
    from src.tasks.dispatcher import process_download_and_ingest

    state = TaskState(
        task_id="t2",
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={},
    )
    q = MagicMock()
    q.get_state.return_value = state

    with (
        patch("src.retrieval.downloader.adapter.get_adapter") as mock_adapter_factory,
        patch("src.tasks.dispatcher.get_task_queue", return_value=q),
        patch("src.tasks.dispatcher._trigger_ingest_pipeline"),
        patch("src.tasks.dispatcher._mark_library_paper_downloaded"),
    ):
        adapter_instance = MagicMock()
        adapter_instance.download_paper = AsyncMock(return_value=_make_dl_result_supplementary())
        mock_adapter_factory.return_value = adapter_instance

        asyncio.run(process_download_and_ingest(
            task_id="t2",
            paper_info={"title": "Test Paper", "doi": "10.1234/test"},
            collection="my_coll",
        ))

    assert state.payload.get("stage") == "SUPPLEMENTARY_SAVED"
    assert state.status == TaskStatus.completed
    assert state.payload.get("is_supplementary") is True


def test_main_pdf_still_triggers_ingest():
    """Non-supplementary download should still trigger the normal ingest flow."""
    from src.tasks.dispatcher import process_download_and_ingest

    state = TaskState(
        task_id="t3",
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={},
    )
    q = MagicMock()
    q.get_state.return_value = state

    with (
        patch("src.retrieval.downloader.adapter.get_adapter") as mock_adapter_factory,
        patch("src.tasks.dispatcher.get_task_queue", return_value=q),
        patch("src.tasks.dispatcher._trigger_ingest_pipeline", return_value="job-123") as mock_ingest,
        patch("src.tasks.dispatcher._mark_library_paper_downloaded") as mock_mark,
        patch("src.tasks.dispatcher.threading.Thread"),
    ):
        adapter_instance = MagicMock()
        adapter_instance.download_paper = AsyncMock(return_value=_make_dl_result_main())
        mock_adapter_factory.return_value = adapter_instance

        result = asyncio.run(process_download_and_ingest(
            task_id="t3",
            paper_info={"title": "Main Paper", "doi": "10.1234/main"},
            collection="my_coll",
            library_paper_id=7,
        ))

    mock_ingest.assert_called_once()
    mock_mark.assert_called_once_with(7)
    assert result.get("ingest_triggered") is True


def test_supplementary_result_has_primary_paper_id():
    """Result payload must contain primary_paper_id and supplementary_reason."""
    from src.tasks.dispatcher import process_download_and_ingest

    state = TaskState(
        task_id="t4",
        kind=TaskKind.scholar,
        status=TaskStatus.running,
        payload={},
    )
    q = MagicMock()
    q.get_state.return_value = state
    dl_result = _make_dl_result_supplementary()

    with (
        patch("src.retrieval.downloader.adapter.get_adapter") as mock_adapter_factory,
        patch("src.tasks.dispatcher.get_task_queue", return_value=q),
        patch("src.tasks.dispatcher._trigger_ingest_pipeline"),
        patch("src.tasks.dispatcher._mark_library_paper_downloaded"),
    ):
        adapter_instance = MagicMock()
        adapter_instance.download_paper = AsyncMock(return_value=dl_result)
        mock_adapter_factory.return_value = adapter_instance

        result = asyncio.run(process_download_and_ingest(
            task_id="t4",
            paper_info={"title": "Test Paper", "doi": "10.1234/test"},
        ))

    assert result.get("primary_paper_id") == "10.1234_test"
    assert result.get("supplementary_reason") == "obvious_phrase:supplementary material"
    assert state.payload.get("primary_paper_id") == "10.1234_test"
