from src.api.routes_scholar import (
    _build_library_ingest_cfg,
    _dedupe_papers_by_doi_keep_best_source,
)


def test_build_library_ingest_cfg_preserves_incremental_flags():
    cfg = _build_library_ingest_cfg(
        file_paths=["/tmp/a.pdf", "/tmp/b.pdf"],
        collection_name="my_collection",
        user_id="u1",
        library_name="DeepSea_symbiosis",
        skip_duplicate_doi=True,
        skip_unchanged=True,
    )

    assert cfg["file_paths"] == ["/tmp/a.pdf", "/tmp/b.pdf"]
    assert cfg["collection_name"] == "my_collection"
    assert cfg["user_id"] == "u1"
    assert cfg["library_name"] == "DeepSea_symbiosis"
    assert cfg["skip_duplicate_doi"] is True
    assert cfg["skip_unchanged"] is True
    assert cfg["actual_skip"] is True
    assert cfg["enrich_tables"] is False
    assert cfg["enrich_figures"] is False
    assert cfg["file_metadata_map"] == {}


def test_build_library_ingest_cfg_keeps_per_file_metadata():
    file_metadata_map = {
        "/tmp/a.pdf": {
            "library_id": 12,
            "library_paper_id": 34,
            "source": "google_scholar",
            "metadata": {
                "doi": "10.1000/xyz",
                "title": "Test Title",
                "authors": ["A", "B"],
                "year": 2024,
                "venue": "Nature",
            },
        }
    }
    cfg = _build_library_ingest_cfg(
        file_paths=["/tmp/a.pdf"],
        collection_name="my_collection",
        user_id="u1",
        library_name="DeepSea_symbiosis",
        skip_duplicate_doi=True,
        skip_unchanged=False,
        file_metadata_map=file_metadata_map,
    )

    assert cfg["file_metadata_map"] == file_metadata_map


def test_dedupe_papers_prefers_scholar_source_for_same_doi():
    items = [
        {
            "content": "a",
            "score": 0.7,
            "metadata": {
                "doi": "10.1000/xyz123",
                "source": "google",
                "title": "Title A",
                "pdf_url": "https://example.com/a.pdf",
            },
        },
        {
            "content": "b",
            "score": 0.8,
            "metadata": {
                "doi": "10.1000/xyz123",
                "source": "google_scholar",
                "title": "Title B",
                "pdf_url": "https://example.com/b.pdf",
            },
        },
    ]

    out = _dedupe_papers_by_doi_keep_best_source(items)
    assert len(out) == 1
    assert out[0]["metadata"]["source"] == "google_scholar"
