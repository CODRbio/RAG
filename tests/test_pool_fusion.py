"""
Unit tests for fuse_pools_with_gap_protection and related hybrid-retrieval changes.

Coverage:
  1. Global ordering — fused output is sorted by score (descending).
  2. Gap quota protection — gap_min_keep is honoured even when gap items would
     otherwise fall outside the top_k window.
  3. Ranked-tail and unranked backfill paths both work.
  4. Empty-pool edge cases — function returns gracefully with one or both pools empty.
  5. top_k boundary — output never exceeds top_k.
  6. _DR_GAP_POOL_SOURCES constant is consistent with _rerank_section_pool_chunks usage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal raw candidate dict as used by service.py
# ---------------------------------------------------------------------------

def _make_cand(chunk_id: str, score: float, pool_tag: str = "main") -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content": f"Content of chunk {chunk_id}",
        "score": score,
        "source": "dense",
        "metadata": {"chunk_id": chunk_id, "doc_id": chunk_id},
        "_pool_tag": pool_tag,
    }


# ---------------------------------------------------------------------------
# Import target (skip if heavy ML deps absent)
# ---------------------------------------------------------------------------

try:
    from src.retrieval.service import (
        fuse_pools_with_gap_protection,
        _GAP_MIN_KEEP_RATIO,
    )
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="retrieval service not importable")


# ---------------------------------------------------------------------------
# Mock reranker: returns candidates sorted by their *existing* score field
# so tests stay deterministic without loading any ML model.
# ---------------------------------------------------------------------------

def _mock_rerank(query, candidates, top_k, reranker_mode=None):
    """Sort by score desc, return top_k, preserving all dict keys."""
    ranked = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)
    return [{**c} for c in ranked[:top_k]]


def _mock_embedding_rerank(query, candidates, top_k):
    return _mock_rerank(query, candidates, top_k)


@pytest.fixture(autouse=True)
def patch_rerankers():
    """Replace real ML rerankers with deterministic score-sort mocks."""
    with (
        patch("src.retrieval.service._rerank_candidates", side_effect=_mock_rerank),
        patch("src.retrieval.service._embedding_rerank", side_effect=_mock_embedding_rerank),
    ):
        yield


# ===========================================================================
# 1. Global ordering
# ===========================================================================

def test_output_is_sorted_descending():
    main = [_make_cand(f"m{i}", float(i)) for i in range(5)]
    result = fuse_pools_with_gap_protection("q", main, [], top_k=5)
    scores = [c["score"] for c in result]
    assert scores == sorted(scores, reverse=True), "Output must be sorted descending by score"


# ===========================================================================
# 2. Gap quota protection and backfill
# ===========================================================================

def test_gap_quota_enforced_when_gap_would_be_excluded():
    """
    Scenario: top_k=3, gap_min_keep=1.
    Main: scores 1.0, 0.9, 0.8 → would fill all 3 slots.
    Gap: score 0.3 → without quota it's excluded.
    With quota the gap item must appear in output (replacing the 0.8 main).
    """
    main = [_make_cand("m1", 1.0), _make_cand("m2", 0.9), _make_cand("m3", 0.8)]
    gap = [_make_cand("g1", 0.3)]

    result = fuse_pools_with_gap_protection("q", main, gap, top_k=3, gap_min_keep=1)
    ids = {c["chunk_id"] for c in result}
    assert "g1" in ids, "Gap quota should force g1 into output"
    assert len(result) == 3, "Output length must still equal top_k"


def test_gap_min_keep_ratio_is_025():
    """Global default gap quota ratio is 0.25."""
    assert _GAP_MIN_KEEP_RATIO == 0.25


def test_gap_quota_default_ratio():
    """Default gap_min_keep = ceil(top_k * GAP_MIN_KEEP_RATIO)."""
    top_k = 8
    expected_min = math.ceil(top_k * _GAP_MIN_KEEP_RATIO)
    # Create more main candidates than top_k so they'd crowd out gap by default
    main = [_make_cand(f"m{i}", float(10 - i)) for i in range(top_k + 2)]
    gap = [_make_cand(f"g{i}", 0.1 * i) for i in range(expected_min + 1)]

    diag: Dict[str, Any] = {}
    result = fuse_pools_with_gap_protection("q", main, gap, top_k=top_k, diag=diag)
    gap_in = diag["pool_fusion"]["gap_in_output"]
    assert gap_in >= expected_min, (
        f"Expected at least {expected_min} gap items in output, got {gap_in}"
    )
    assert diag["pool_fusion"]["gap_min_keep"] == expected_min


def test_chat_research_ratio_overrides():
    """Mode-level ratio overrides should alter default gap_min_keep deterministically."""
    top_k = 10
    main = [_make_cand(f"m{i}", 0.9 - i * 0.05) for i in range(top_k + 3)]
    gap = [_make_cand(f"g{i}", 0.2 - i * 0.01) for i in range(6)]
    diag_chat: Dict[str, Any] = {}
    diag_research: Dict[str, Any] = {}
    fuse_pools_with_gap_protection("q", main, gap, top_k=top_k, gap_ratio=0.2, diag=diag_chat)
    fuse_pools_with_gap_protection("q", main, gap, top_k=top_k, gap_ratio=0.25, diag=diag_research)
    assert diag_chat["pool_fusion"]["gap_min_keep"] == math.ceil(top_k * 0.2)
    assert diag_research["pool_fusion"]["gap_min_keep"] == math.ceil(top_k * 0.25)


def test_gap_backfill_from_ranked_tail():
    """When rerank_k includes gap in tail, deficit should be filled from ranked tail first."""
    main = [_make_cand(f"m{i}", float(100 - i)) for i in range(12)]
    gap = [_make_cand("g1", 1.0), _make_cand("g2", 0.9)]
    diag: Dict[str, Any] = {}
    result = fuse_pools_with_gap_protection(
        "q", main, gap, top_k=8, gap_min_keep=2, rank_pool_multiplier=2.0, diag=diag
    )
    ids = {c["chunk_id"] for c in result}
    assert "g1" in ids and "g2" in ids
    assert diag["pool_fusion"]["gap_backfill_ranked"] >= 1
    assert diag["pool_fusion"]["gap_backfill_unranked"] == 0


def test_gap_backfill_from_unranked_pool():
    """When rerank_k excludes gaps, deficit is filled from unranked original gap pool."""
    main = [_make_cand(f"m{i}", float(50 - i)) for i in range(8)]
    gap = [_make_cand("g1", -10.0), _make_cand("g2", -11.0)]
    diag: Dict[str, Any] = {}
    result = fuse_pools_with_gap_protection(
        "q", main, gap, top_k=2, gap_min_keep=1, rank_pool_multiplier=1.0, diag=diag
    )
    ids = {c["chunk_id"] for c in result}
    assert "g1" in ids or "g2" in ids
    assert diag["pool_fusion"]["gap_backfill_unranked"] >= 1
    assert diag["pool_fusion"]["gap_deficit_before_fill"] >= 1


def test_rank_pool_multiplier_affects_rank_pool_k():
    main = [_make_cand(f"m{i}", float(100 - i)) for i in range(20)]
    gap = [_make_cand(f"g{i}", float(10 - i)) for i in range(5)]
    diag2: Dict[str, Any] = {}
    diag3: Dict[str, Any] = {}
    fuse_pools_with_gap_protection("q", main, gap, top_k=6, rank_pool_multiplier=2.0, diag=diag2)
    fuse_pools_with_gap_protection("q", main, gap, top_k=6, rank_pool_multiplier=3.0, diag=diag3)
    assert diag3["pool_fusion"]["rank_pool_k"] >= diag2["pool_fusion"]["rank_pool_k"]


# ===========================================================================
# 3. Empty-pool edge cases
# ===========================================================================

def test_both_pools_empty_returns_empty():
    result = fuse_pools_with_gap_protection("q", [], [], top_k=5)
    assert result == []


def test_main_only_no_crash():
    main = [_make_cand("m1", 0.9), _make_cand("m2", 0.7)]
    result = fuse_pools_with_gap_protection("q", main, [], top_k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "m1"


def test_gap_only_no_crash():
    gap = [_make_cand("g1", 0.9), _make_cand("g2", 0.5)]
    result = fuse_pools_with_gap_protection("q", [], gap, top_k=2)
    assert len(result) == 2


# ===========================================================================
# 4. top_k boundary
# ===========================================================================

def test_output_never_exceeds_top_k():
    main = [_make_cand(f"m{i}", float(i)) for i in range(20)]
    gap = [_make_cand(f"g{i}", float(i) + 0.5) for i in range(10)]
    result = fuse_pools_with_gap_protection("q", main, gap, top_k=7)
    assert len(result) <= 7


def test_output_length_with_fewer_candidates_than_top_k():
    main = [_make_cand("m1", 0.9)]
    result = fuse_pools_with_gap_protection("q", main, [], top_k=10)
    assert len(result) == 1  # can't return more than total candidates


# ===========================================================================
# 5. Diagnostics output
# ===========================================================================

def test_diagnostics_populated():
    main = [_make_cand("m1", 0.9)]
    gap = [_make_cand("g1", 0.5)]
    diag: Dict[str, Any] = {}
    fuse_pools_with_gap_protection("q", main, gap, top_k=2, diag=diag)
    pf = diag["pool_fusion"]
    assert pf["main_in"] == 1
    assert pf["gap_in"] == 1
    assert "rank_pool_k" in pf
    assert "rank_pool_multiplier" in pf
    assert "gap_boost_abs" in pf
    assert "gap_min_keep" in pf
    assert "output_count" in pf


# ===========================================================================
# 6. _DR_GAP_POOL_SOURCES constant
# ===========================================================================

def test_dr_gap_pool_sources_contains_eval_supplement():
    """The DR gap pool must include eval_supplement so gap protection activates."""
    from src.collaboration.research.agent import _DR_GAP_POOL_SOURCES
    assert "eval_supplement" in _DR_GAP_POOL_SOURCES


# ===========================================================================
# 7. _pool_tag is stripped from output
# ===========================================================================

def test_pool_tag_stripped_from_output():
    main = [_make_cand("m1", 0.9)]
    gap = [_make_cand("g1", 0.5)]
    result = fuse_pools_with_gap_protection("q", main, gap, top_k=2)
    for item in result:
        assert "_pool_tag" not in item, "_pool_tag internal key must be stripped from output"


# ===========================================================================
# 8. Soft-wait timeout diagnostics keys
# ===========================================================================

def test_hybrid_mode_diag_keys_when_timeout(monkeypatch):
    """
    When web retrieval times out, diag must contain web_timeout key.
    We test this via the service's hybrid path using lightweight mocks.
    """
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    import src.retrieval.service as svc_mod

    # Stub retriever
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "lc1", "content": "local content", "score": 0.8,
         "source": "dense", "metadata": {"chunk_id": "lc1", "doc_id": "d1"}},
    ]

    svc = svc_mod.RetrievalService(retriever=mock_retriever, collection="test", top_k=10)

    # Force web Future to raise FuturesTimeoutError
    class _TimingOutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError()
        def cancel(self):
            pass

    class _LocalFuture:
        def result(self, timeout=None):
            return mock_retriever.retrieve.return_value

    class _MockExecutor:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def submit(self, fn, *args, **kwargs):
            # first submit → local, second → web
            if not hasattr(self, "_called"):
                self._called = True
                return _LocalFuture()
            return _TimingOutFuture()
        def shutdown(self, wait=True): pass

    with (
        patch("src.retrieval.service.ThreadPoolExecutor", _MockExecutor),
        patch("src.retrieval.service.cross_source_dedup", return_value=[]),
        patch("src.retrieval.service._compress_web_fulltext", return_value=[]),
        patch("src.retrieval.service.fuse_pools_with_gap_protection",
              return_value=[{"chunk_id": "lc1", "content": "local content",
                             "score": 0.8, "_source_type": "dense",
                             "source": "dense", "metadata": {"chunk_id": "lc1"}}]),
    ):
        pack = svc.search("test query", mode="hybrid", filters={"step_top_k": 5}, top_k=10)

    assert pack.diagnostics is not None
    assert "web_timeout" in pack.diagnostics
