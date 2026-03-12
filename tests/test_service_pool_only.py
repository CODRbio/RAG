from __future__ import annotations

from unittest.mock import MagicMock

from src.retrieval.service import RetrievalService


def _web_hit(idx: int) -> dict:
    return {
        "chunk_id": f"w{idx}",
        "content": f"web hit {idx}",
        "score": 0.1 * idx,
        "metadata": {
            "chunk_id": f"w{idx}",
            "doc_id": f"doc_w{idx}",
            "title": f"title {idx}",
            "provider": "semantic",
            "url": f"https://example.com/{idx}",
        },
    }


def test_web_pool_only_returns_full_raw_pool_without_rerank(monkeypatch):
    hits = [_web_hit(1), _web_hit(2), _web_hit(3)]
    monkeypatch.setattr(
        "src.retrieval.service.unified_web_searcher.search_sync",
        lambda *args, **kwargs: list(hits),
    )
    monkeypatch.setattr(
        "src.retrieval.service._compress_web_fulltext",
        lambda web_hits, query, filters: web_hits,
    )

    rerank_called = {"value": False}

    def _fail_rerank(*args, **kwargs):
        rerank_called["value"] = True
        raise AssertionError("pool_only should bypass web rerank")

    monkeypatch.setattr("src.retrieval.service._rerank_candidates", _fail_rerank)
    monkeypatch.setattr("src.retrieval.service._embedding_rerank", _fail_rerank)

    svc = RetrievalService(retriever=MagicMock(), collection="test")
    pack = svc.search(
        query="q",
        mode="web",
        top_k=1,
        filters={
            "pool_only": True,
            "step_top_k": 1,
            "web_providers": ["semantic"],
        },
    )

    assert rerank_called["value"] is False
    assert [c.chunk_id for c in pack.chunks] == ["w1", "w2", "w3"]
