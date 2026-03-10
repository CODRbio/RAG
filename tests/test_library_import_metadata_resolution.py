from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_extract_doi_from_filename_variants():
    from src.retrieval.dedup import extract_doi_from_filename

    assert extract_doi_from_filename("10.1000%2FAbC.1.pdf") == "10.1000/abc.1"
    assert extract_doi_from_filename("https___doi.org_10.1000_XYZ-2.pdf") == "10.1000/xyz-2"
    assert extract_doi_from_filename("10.1016_j.cell.2020.01.001.pdf") == "10.1016/j.cell.2020.01.001"


@patch("src.api.routes_scholar._extract_pdf_metadata_from_text")
@patch("src.parser.pdf_parser.extract_native_metadata")
@patch("src.retrieval.dedup.extract_doi_from_pdf_tiered")
@patch("src.retrieval.dedup.extract_doi_from_filename")
@patch("src.retrieval.dedup._crossref_lookup_by_doi")
@patch("src.retrieval.dedup._crossref_lookup_by_title")
def test_resolve_library_import_metadata_prefers_filename_doi(
    m_crossref_title,
    m_crossref_doi,
    m_from_name,
    m_from_pdf,
    m_native,
    m_text,
    tmp_path,
):
    from src.api.routes_scholar import _resolve_library_import_metadata

    m_from_name.return_value = "10.2000/from-file"
    m_from_pdf.return_value = ("10.3000/from-pdf", "PDF Title")
    m_native.return_value = {"title": "Native Title"}
    m_text.return_value = {"title": "Text Title"}
    m_crossref_doi.return_value = {
        "doi": "10.2000/from-file",
        "authors": ["Alice", "Bob"],
        "year": 2022,
        "venue": "Journal X",
    }
    m_crossref_title.return_value = None

    out = _resolve_library_import_metadata("10.2000_from-file.pdf", tmp_path / "a.pdf")
    assert out["doi"] == "10.2000/from-file"
    assert out["title"] == "PDF Title"
    assert out["authors"] == ["Alice", "Bob"]
    assert out["year"] == 2022
    assert out["venue"] == "Journal X"
    m_crossref_title.assert_not_called()


@patch("src.api.routes_scholar._extract_pdf_metadata_from_text")
@patch("src.parser.pdf_parser.extract_native_metadata")
@patch("src.retrieval.dedup.extract_doi_from_pdf_tiered")
@patch("src.retrieval.dedup.extract_doi_from_filename")
@patch("src.retrieval.dedup._crossref_lookup_by_doi")
@patch("src.retrieval.dedup._crossref_lookup_by_title")
def test_resolve_library_import_metadata_crossref_title_fallback(
    m_crossref_title,
    m_crossref_doi,
    m_from_name,
    m_from_pdf,
    m_native,
    m_text,
    tmp_path,
):
    from src.api.routes_scholar import _resolve_library_import_metadata

    m_from_name.return_value = ""
    m_from_pdf.return_value = (None, None)
    m_native.return_value = {}
    m_text.return_value = {"title": "Interesting Paper"}
    m_crossref_doi.return_value = None
    m_crossref_title.return_value = {
        "doi": "10.4000/crossref",
        "title": "Interesting Paper",
        "authors": ["Carol"],
        "year": 2019,
        "venue": "Conference Y",
    }

    out = _resolve_library_import_metadata("unknown.pdf", tmp_path / "b.pdf")
    assert out["doi"] == "10.4000/crossref"
    assert out["title"] == "Interesting Paper"
    assert out["authors"] == ["Carol"]
    assert out["year"] == 2019
    assert out["venue"] == "Conference Y"
