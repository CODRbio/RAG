from types import SimpleNamespace

from src.api import routes_scholar


def test_sync_remove_library_deleted_papers_removes_parsed_dir(monkeypatch, tmp_path):
    parsed_root = tmp_path / "libraries" / "DeepSea_symbiosis" / "parsed_data"
    stale_dir = parsed_root / "paper_stale"
    keep_dir = parsed_root / "paper_keep"
    stale_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        routes_scholar.PathManager,
        "get_user_library_parsed_path",
        lambda _user_id, _library_name: parsed_root,
    )

    import src.indexing.paper_store as paper_store
    import src.indexing.milvus_ops as milvus_ops

    monkeypatch.setattr(
        paper_store,
        "list_papers_linked_to_library",
        lambda _collection, _lib_id, user_id=None: [
            {"paper_id": "paper_stale", "library_paper_id": 100},
            {"paper_id": "paper_keep", "library_paper_id": 101},
        ],
    )

    deleted_papers = []
    monkeypatch.setattr(
        paper_store,
        "delete_paper",
        lambda collection, paper_id: deleted_papers.append((collection, paper_id)) or True,
    )

    class _FakeMilvusClient:
        @staticmethod
        def has_collection(_name: str) -> bool:
            return True

        @staticmethod
        def delete(**_kwargs):
            return {"delete_count": 1}

    monkeypatch.setattr(milvus_ops, "milvus", SimpleNamespace(client=_FakeMilvusClient()))

    removed = routes_scholar._sync_remove_library_deleted_papers(
        collection_name="c1",
        lib_id=1,
        user_id="u1",
        library_name="DeepSea_symbiosis",
        current_library_paper_ids={101},
    )

    assert removed == 1
    assert deleted_papers == [("c1", "paper_stale")]
    assert not stale_dir.exists()
    assert keep_dir.exists()
