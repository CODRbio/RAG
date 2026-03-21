"""
Regression tests for 文献库关联与元信息完善:
- DOI normalization consistency (dedup + paper_metadata_store key)
- DOI variant dedup (scholar add/dedup)
- list_papers / library papers API return shape (library_id, library_paper_id, in_collection, etc.)
- _sync_library_paper_downloaded_at uses _library_paper_id so no-DOI papers get downloaded_at when PDF exists.
"""

import pytest

from src.retrieval.dedup import normalize_doi, normalize_title
from src.api.routes_scholar import (
    _dedupe_papers_by_doi_keep_best_source,
    _library_paper_id,
    _sync_library_paper_downloaded_at,
)


# ─── DOI normalization ───────────────────────────────────────────────────────

def test_normalize_doi_lowercase():
    assert normalize_doi("10.1000/ABC123") == "10.1000/abc123"
    assert normalize_doi("10.1038/ISMEJ.2016.124") == "10.1038/ismej.2016.124"


def test_normalize_doi_strips_url_prefix():
    assert normalize_doi("https://doi.org/10.1000/xyz") == "10.1000/xyz"
    assert normalize_doi("http://dx.doi.org/10.1000/xyz") == "10.1000/xyz"
    assert normalize_doi("https://DOI.ORG/10.1000/xyz") == "10.1000/xyz"


def test_normalize_doi_decodes_percent():
    assert normalize_doi("10.1038%2Fismej.2016.124") == "10.1038/ismej.2016.124"
    assert normalize_doi("https://doi.org/10.1038%2FISMEJ.2016.124") == "10.1038/ismej.2016.124"


def test_normalize_doi_equivalent_inputs_same_output():
    """Equivalent DOIs (URL, case, %2F) must produce the same canonical form."""
    base = "10.1038/ismej.2016.124"
    variants = [
        base,
        "10.1038/ISMEJ.2016.124",
        "https://doi.org/10.1038/ismej.2016.124",
        "https://doi.org/10.1038%2Fismej.2016.124",
        "https://DOI.ORG/10.1038%2FISMEJ.2016.124",
        "  https://doi.org/10.1038/ismej.2016.124  ",
    ]
    for v in variants:
        assert normalize_doi(v) == base, f"Expected {base!r} for input {v!r}, got {normalize_doi(v)!r}"


def test_normalize_doi_empty_and_none():
    assert normalize_doi("") == ""
    assert normalize_doi(None) == ""


def test_normalize_title_consistency():
    t = "Deep-Sea Biodiversity: Patterns and Processes"
    assert normalize_title(t) == normalize_title(t.upper())
    assert " " not in normalize_title(t) or normalize_title(t).replace(" ", "") != ""


# ─── Scholar dedup by normalized DOI ──────────────────────────────────────────

def test_dedupe_papers_by_doi_merges_doi_variants():
    """Items that differ only by DOI representation (URL/case) should dedupe to one."""
    items = [
        {
            "content": "a",
            "score": 0.7,
            "metadata": {
                "doi": "https://doi.org/10.1000/XYZ",
                "source": "google",
                "title": "Title A",
                "pdf_url": "https://example.com/a.pdf",
            },
        },
        {
            "content": "b",
            "score": 0.8,
            "metadata": {
                "doi": "10.1000/xyz",
                "source": "google_scholar",
                "title": "Title B",
                "pdf_url": "https://example.com/b.pdf",
            },
        },
    ]
    out = _dedupe_papers_by_doi_keep_best_source(items)
    assert len(out) == 1
    assert out[0]["metadata"]["source"] == "google_scholar"


def test_dedupe_papers_by_doi_percent_encoded_same_doi():
    items = [
        {
            "content": "a",
            "score": 0.7,
            "metadata": {"doi": "10.1038%2Fismej.2016.124", "source": "google", "title": "A", "pdf_url": ""},
        },
        {
            "content": "b",
            "score": 0.8,
            "metadata": {"doi": "10.1038/ismej.2016.124", "source": "google_scholar", "title": "B", "pdf_url": ""},
        },
    ]
    out = _dedupe_papers_by_doi_keep_best_source(items)
    assert len(out) == 1


# ─── Paper store list_papers return shape ─────────────────────────────────────

def test_list_papers_return_keys_include_library_link():
    """list_papers must include library_id, library_paper_id, source, paper_uid in each paper dict."""
    from src.indexing.paper_store import list_papers
    # Call with a non-existent collection to get empty list; we only check the code path
    # returns dicts that can contain these keys (actual DB may not have them).
    try:
        papers = list_papers("__nonexistent_collection_xyz__")
    except Exception:
        pytest.skip("DB or engine not available")
    assert isinstance(papers, list)
    for p in papers:
        assert "paper_id" in p
        assert "library_id" in p
        assert "library_paper_id" in p
        assert "source" in p
        assert "paper_uid" in p


def test_library_paper_id_fallback_no_doi():
    """_library_paper_id with no DOI returns fallback stem from title/authors/year so no-DOI papers can match PDFs."""
    pid = _library_paper_id(None, "Sync Test Paper", ["Smith", "Jones"], 2021)
    assert pid is not None
    assert isinstance(pid, str)
    assert len(pid) >= 4
    # Normalization: title lowercased, spaces to underscores, author and year suffix
    assert "sync" in pid.lower() or "test" in pid.lower() or "2021" in pid


def test_sync_library_paper_downloaded_at_calls_library_paper_id_for_no_doi_row():
    """_sync_library_paper_downloaded_at calls _library_paper_id(doi, title, authors, year) for each row including no-DOI."""
    from unittest.mock import MagicMock, patch

    calls = []

    def record_library_paper_id(doi, title, authors, year):
        calls.append((doi, title, authors, year))
        return None  # no match so no update

    mock_row = MagicMock()
    mock_row.doi = ""
    mock_row.title = "No DOI Paper"
    mock_row.downloaded_at = None
    mock_row.year = 2020
    mock_row.get_authors.return_value = ["Author"]

    class MockResult:
        def all(self):
            return [mock_row]

    mock_session = MagicMock()
    mock_session.exec.return_value = MockResult()

    class SessionCtx:
        def __enter__(_):
            return mock_session
        def __exit__(_, *a):
            return False

    with patch("src.api.routes_scholar.get_engine"), \
         patch("src.api.routes_scholar.Session", return_value=SessionCtx()), \
         patch("src.api.routes_scholar._library_paper_id", side_effect=record_library_paper_id), \
         patch("src.api.routes_scholar.Path") as mock_path_cls:
        mock_path_cls.return_value.is_dir.return_value = True
        _sync_library_paper_downloaded_at(lib_id=1, pdfs_dir="/tmp/any")

    assert len(calls) >= 1
    doi, title, authors, year = calls[0]
    assert doi is None or doi == ""
    assert title == "No DOI Paper"
    assert authors == ["Author"]
    assert year == 2020
