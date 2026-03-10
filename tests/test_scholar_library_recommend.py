"""
Unit tests for the Scholar library recommend feature.
Tests _aggregate_chunks_to_library_papers() in isolation -- no live DB or Milvus required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from src.api.routes_scholar import _aggregate_chunks_to_library_papers


# ---------------------------------------------------------------------------
# Minimal stand-in for EvidenceChunk (matches attribute access pattern)
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunk:
    doc_id: str
    text: str
    score: float
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self.doc_id + "_chunk"


# ---------------------------------------------------------------------------
# Minimal stand-in for ScholarLibraryPaper (attribute access)
# ---------------------------------------------------------------------------

@dataclass
class _FakePaper:
    id: int
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    impact_factor: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lookup(*papers: _FakePaper, coll_ids: List[str]) -> Dict[str, _FakePaper]:
    """Build eligible_lookup keyed by collection_paper_id."""
    assert len(papers) == len(coll_ids)
    return {cid: p for cid, p in zip(coll_ids, papers)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAggregateChunksBasic:
    """5 chunks across 3 docs -- verify grouping, scoring, snippet order."""

    def test_basic_grouping(self):
        paper_a = _FakePaper(id=1, title="Alpha Paper", doi="10.1/a", year=2023)
        paper_b = _FakePaper(id=2, title="Beta Paper", doi="10.1/b", year=2022)
        paper_c = _FakePaper(id=3, title="Gamma Paper", doi=None, year=2021)

        lookup = _make_lookup(paper_a, paper_b, paper_c, coll_ids=["doc_a", "doc_b", "doc_c"])

        chunks = [
            _FakeChunk(doc_id="doc_a", text="Alpha chunk 1", score=0.9),
            _FakeChunk(doc_id="doc_a", text="Alpha chunk 2", score=0.7),
            _FakeChunk(doc_id="doc_b", text="Beta chunk 1", score=0.85),
            _FakeChunk(doc_id="doc_c", text="Gamma chunk 1", score=0.5),
            _FakeChunk(doc_id="doc_c", text="Gamma chunk 2", score=0.6),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)

        assert len(results) == 3

        # Sorted by best_chunk_score descending: doc_a(0.9) > doc_b(0.85) > doc_c(0.6)
        assert results[0]["collection_paper_id"] == "doc_a"
        assert results[1]["collection_paper_id"] == "doc_b"
        assert results[2]["collection_paper_id"] == "doc_c"

    def test_scores_and_matched_chunks(self):
        paper_a = _FakePaper(id=1, title="Alpha")
        lookup = {"doc_a": paper_a}

        chunks = [
            _FakeChunk(doc_id="doc_a", text="chunk 1", score=0.9),
            _FakeChunk(doc_id="doc_a", text="chunk 2", score=0.7),
            _FakeChunk(doc_id="doc_a", text="chunk 3", score=0.5),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)

        assert len(results) == 1
        r = results[0]
        assert r["best_chunk_score"] == pytest.approx(0.9)
        assert r["score"] == pytest.approx(0.9)
        assert r["matched_chunks"] == 3

    def test_snippets_ordered_by_score_descending(self):
        paper = _FakePaper(id=1, title="Paper")
        lookup = {"doc_x": paper}

        chunks = [
            _FakeChunk(doc_id="doc_x", text="low score text", score=0.3),
            _FakeChunk(doc_id="doc_x", text="high score text", score=0.9),
            _FakeChunk(doc_id="doc_x", text="mid score text", score=0.6),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10, max_snippets=2)
        snippets = results[0]["snippets"]
        assert len(snippets) == 2
        assert snippets[0] == "high score text"
        assert snippets[1] == "mid score text"


class TestAggregateFiltersNonEligible:
    """Chunks whose doc_id is NOT in eligible_lookup must be excluded."""

    def test_non_eligible_excluded(self):
        paper = _FakePaper(id=1, title="Eligible")
        lookup = {"eligible_doc": paper}

        chunks = [
            _FakeChunk(doc_id="eligible_doc", text="relevant", score=0.8),
            _FakeChunk(doc_id="not_in_library", text="irrelevant", score=0.99),
            _FakeChunk(doc_id="also_missing", text="also irrelevant", score=0.95),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)

        assert len(results) == 1
        assert results[0]["collection_paper_id"] == "eligible_doc"

    def test_all_non_eligible_returns_empty(self):
        paper = _FakePaper(id=1, title="Eligible")
        lookup = {"eligible_doc": paper}

        chunks = [
            _FakeChunk(doc_id="foreign_1", text="text", score=0.9),
            _FakeChunk(doc_id="foreign_2", text="text", score=0.8),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)
        assert results == []


class TestAggregateTopKLimit:
    """top_k truncation must be respected."""

    def test_top_k_truncates_results(self):
        lookup = {f"doc_{i}": _FakePaper(id=i, title=f"Paper {i}") for i in range(10)}
        chunks = [
            _FakeChunk(doc_id=f"doc_{i}", text=f"text {i}", score=float(i) / 10)
            for i in range(10)
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=3)
        assert len(results) == 3

    def test_top_k_returns_highest_scores(self):
        lookup = {f"doc_{i}": _FakePaper(id=i, title=f"Paper {i}") for i in range(5)}
        chunks = [
            _FakeChunk(doc_id="doc_0", text="t", score=0.1),
            _FakeChunk(doc_id="doc_1", text="t", score=0.5),
            _FakeChunk(doc_id="doc_2", text="t", score=0.9),
            _FakeChunk(doc_id="doc_3", text="t", score=0.7),
            _FakeChunk(doc_id="doc_4", text="t", score=0.3),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=2)
        assert len(results) == 2
        ids = [r["collection_paper_id"] for r in results]
        assert "doc_2" in ids
        assert "doc_3" in ids


class TestAggregateEmpty:
    """Edge cases with no chunks or no eligible papers."""

    def test_empty_chunks(self):
        lookup = {"doc_a": _FakePaper(id=1, title="A")}
        results = _aggregate_chunks_to_library_papers([], lookup, top_k=10)
        assert results == []

    def test_empty_lookup(self):
        chunks = [_FakeChunk(doc_id="doc_a", text="text", score=0.8)]
        results = _aggregate_chunks_to_library_papers(chunks, {}, top_k=10)
        assert results == []

    def test_both_empty(self):
        results = _aggregate_chunks_to_library_papers([], {}, top_k=10)
        assert results == []


class TestCandidateIdsFilter:
    """
    The endpoint-level candidate filtering (not inside the helper), but we can verify
    the helper respects a pre-filtered lookup (simulating what the endpoint does).
    """

    def test_filtered_lookup_restricts_results(self):
        # Simulate: user passed candidate_library_paper_ids=[1, 2], so only papers 1 and 2
        # ended up in the eligible_lookup (paper 3 was excluded before calling helper).
        paper1 = _FakePaper(id=1, title="Paper 1")
        paper2 = _FakePaper(id=2, title="Paper 2")
        # paper3 intentionally NOT in lookup
        lookup = {"doc_1": paper1, "doc_2": paper2}

        chunks = [
            _FakeChunk(doc_id="doc_1", text="relevant 1", score=0.8),
            _FakeChunk(doc_id="doc_2", text="relevant 2", score=0.75),
            _FakeChunk(doc_id="doc_3", text="excluded paper", score=0.99),
        ]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)
        assert len(results) == 2
        ids = {r["library_paper_id"] for r in results}
        assert ids == {1, 2}

    def test_metadata_fields_propagated(self):
        paper = _FakePaper(id=42, title="Test Paper", doi="10.9/test", year=2024, venue="Nature", impact_factor=50.5)
        lookup = {"test_doc": paper}
        chunks = [_FakeChunk(doc_id="test_doc", text="snippet text", score=0.88)]

        results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=10)
        assert len(results) == 1
        r = results[0]
        assert r["library_paper_id"] == 42
        assert r["collection_paper_id"] == "test_doc"
        assert r["title"] == "Test Paper"
        assert r["doi"] == "10.9/test"
        assert r["year"] == 2024
        assert r["venue"] == "Nature"
        assert r["impact_factor"] == pytest.approx(50.5)
        assert r["snippets"] == ["snippet text"]
