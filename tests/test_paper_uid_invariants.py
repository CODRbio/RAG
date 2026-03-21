from __future__ import annotations

from dataclasses import dataclass

from src.api.routes_scholar import _aggregate_chunks_to_library_papers
from src.retrieval.dedup import compute_paper_uid


def test_compute_paper_uid_uses_first_author_last_name():
    uid_comma = compute_paper_uid(title="Deep Learning", authors=["LeCun, Yann"], year=2015)
    uid_plain = compute_paper_uid(title="Deep Learning", authors=["Yann LeCun"], year=2015)

    assert uid_comma == uid_plain
    assert uid_comma == "sha:2638662f42e7f5e7"


@dataclass
class _Chunk:
    doc_id: str
    text: str
    score: float
    paper_uid: str
    chunk_id: str = "chunk"


@dataclass
class _Paper:
    id: int
    title: str
    collection_paper_id: str
    paper_uid: str
    doi: str | None = None
    year: int | None = None
    venue: str | None = None
    impact_factor: float | None = None


def test_aggregate_chunks_prefers_paper_uid_lookup():
    paper_uid = "doi:10.1000/test"
    lookup = {
        paper_uid: _Paper(
            id=7,
            title="Test Paper",
            collection_paper_id="local_doc_1",
            paper_uid=paper_uid,
            doi="10.1000/test",
            year=2024,
        )
    }
    chunks = [
        _Chunk(doc_id="completely_different_doc_id", text="match me", score=0.9, paper_uid=paper_uid),
    ]

    results = _aggregate_chunks_to_library_papers(chunks, lookup, top_k=5)

    assert len(results) == 1
    assert results[0]["library_paper_id"] == 7
    assert results[0]["collection_paper_id"] == "local_doc_1"
    assert results[0]["paper_uid"] == paper_uid
