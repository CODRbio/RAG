"""
Tests for Crossref cache by-DOI lookup (local reuse without network).

- crossref_put_by_doi / crossref_get_by_doi in paper_metadata_store
- _crossref_lookup_by_doi in dedup (returns cached result only)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def test_crossref_lookup_by_doi_returns_none_when_empty():
    from src.retrieval.dedup import _crossref_lookup_by_doi, normalize_doi

    # No cache, no network - should return None
    out = _crossref_lookup_by_doi("10.1234/not-cached")
    assert out is None


def test_crossref_lookup_by_doi_returns_cached_when_stored():
    from src.retrieval.dedup import _crossref_lookup_by_doi

    cached = {"doi": "10.1000/cached", "title": "Cached Paper", "authors": ["Author"], "year": 2023, "venue": "Venue"}
    with patch("src.retrieval.dedup._get_paper_meta_store") as m_store:
        m_store.return_value.crossref_get_by_doi.return_value = cached
        out = _crossref_lookup_by_doi("10.1000/cached")
    assert out is not None
    assert out["doi"] == "10.1000/cached"
    assert out["title"] == "Cached Paper"


def test_normalize_doi_used_for_lookup_key():
    from src.retrieval.dedup import normalize_doi

    assert normalize_doi("10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("https://doi.org/10.1000/xyz") == "10.1000/xyz"
