"""
Tests for pdf_url and doi propagation through the evidence/citation pipeline.

Ensures retrieval fusion behavior is unchanged and metadata (pdf_url, doi) survives
from hit -> EvidenceChunk -> Citation -> serialization.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def test_evidence_chunk_has_pdf_url_and_doi():
    from src.retrieval.evidence import EvidenceChunk

    c = EvidenceChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="content",
        score=0.9,
        source_type="web",
        doc_title="Title",
        url="https://example.com/page",
        pdf_url="https://example.com/paper.pdf",
        doi="10.1234/example",
    )
    assert c.pdf_url == "https://example.com/paper.pdf"
    assert c.doi == "10.1234/example"


def test_hit_to_chunk_forwards_pdf_url_and_doi():
    from src.retrieval.service import _hit_to_chunk

    hit = {
        "content": "snippet",
        "score": 0.8,
        "metadata": {
            "chunk_id": "ch1",
            "doc_id": "d1",
            "title": "Paper",
            "url": "https://site.com/article",
            "pdf_url": "https://site.com/article.pdf",
            "doi": "10.5678/test",
        },
    }
    chunk = _hit_to_chunk(hit, "web", "query")
    assert chunk.pdf_url == "https://site.com/article.pdf"
    assert chunk.doi == "10.5678/test"
    assert chunk.url == "https://site.com/article"


def test_hit_to_chunk_backfills_local_doi_from_paper_metadata():
    from unittest.mock import patch
    from src.retrieval.service import _hit_to_chunk

    hit = {
        "content": "local snippet",
        "score": 0.7,
        "metadata": {
            "chunk_id": "local-ch1",
            "paper_id": "paper-123",
        },
    }
    with patch("src.retrieval.service._get_paper_meta_store") as m_store:
        m_store.return_value.get.return_value = {
            "paper_id": "paper-123",
            "doi": "10.4242/local-doi",
            "title": "Local Paper",
            "authors": ["Alice", "Bob"],
            "year": 2024,
        }
        chunk = _hit_to_chunk(hit, "dense", "query")

    assert chunk.doi == "10.4242/local-doi"
    assert chunk.doc_title == "Local Paper"
    assert chunk.authors == ["Alice", "Bob"]
    assert chunk.year == 2024


def test_chunk_to_citation_preserves_pdf_url_and_doi():
    from src.retrieval.evidence import EvidenceChunk
    from src.collaboration.citation.manager import chunk_to_citation

    c = EvidenceChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="content",
        score=0.9,
        source_type="web",
        doc_title="Title",
        url="https://example.com",
        pdf_url="https://example.com/paper.pdf",
        doi="10.1234/example",
    )
    citation = chunk_to_citation(c)
    assert citation.pdf_url == "https://example.com/paper.pdf"
    assert citation.doi == "10.1234/example"
    assert citation.url == "https://example.com"


def test_serialize_citation_includes_pdf_url():
    from src.retrieval.evidence import EvidenceChunk
    from src.collaboration.citation.manager import chunk_to_citation
    from src.api.routes_chat import _serialize_citation, _chat_citation_from_dict

    c = EvidenceChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="content",
        score=0.9,
        source_type="web",
        doc_title="Title",
        pdf_url="https://example.com/paper.pdf",
        doi="10.1234/example",
    )
    citation = chunk_to_citation(c)
    d = _serialize_citation(citation)
    assert d.get("pdf_url") == "https://example.com/paper.pdf"
    assert d.get("doi") == "10.1234/example"
    assert d.get("chunk_id") == "c1"
    assert d.get("anchors") == [{
        "chunk_id": "c1",
        "page_num": None,
        "bbox": None,
        "snippet": "content",
    }]

    roundtrip = _chat_citation_from_dict(d)
    assert roundtrip.pdf_url == "https://example.com/paper.pdf"
    assert roundtrip.chunk_id == "c1"
    assert roundtrip.anchors[0].chunk_id == "c1"
