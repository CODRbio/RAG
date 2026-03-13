import json
from types import SimpleNamespace

from src.api import routes_graph


def test_find_enriched_json_prefers_bound_library(monkeypatch, tmp_path):
    bound_root = tmp_path / "users" / "u1" / "libraries" / "DeepSea_symbiosis" / "parsed_data" / "paper_x"
    other_root = tmp_path / "users" / "u1" / "libraries" / "Other" / "parsed_data" / "paper_x"
    bound_root.mkdir(parents=True, exist_ok=True)
    other_root.mkdir(parents=True, exist_ok=True)

    with open(bound_root / "enriched.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "paper_x"}, f)
    with open(other_root / "enriched.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "paper_x"}, f)

    monkeypatch.setattr(
        routes_graph,
        "resolve_bound_library_for_collection",
        lambda _uid, _collection, auto_create=False: SimpleNamespace(name="DeepSea_symbiosis"),
    )
    monkeypatch.setattr(
        routes_graph.PathManager,
        "get_user_library_parsed_path",
        lambda _uid, _name: tmp_path / "users" / "u1" / "libraries" / _name / "parsed_data",
    )
    monkeypatch.setattr(
        routes_graph.PathManager,
        "get_user_all_library_parsed_paths",
        lambda _uid: [tmp_path / "users" / "u1" / "libraries" / "Other" / "parsed_data"],
    )
    monkeypatch.setattr(
        routes_graph.PathManager,
        "get_user_parsed_path",
        lambda _uid: tmp_path / "users" / "u1" / "library" / "parsed_raw",
    )

    hit = routes_graph._find_enriched_json_by_paper_id("paper_x", user_id="u1", collection="deepsea_ocean")
    assert hit is not None
    assert "DeepSea_symbiosis" in str(hit)


def test_graph_chunk_detail_uses_parsed_fallback_for_missing_bbox(monkeypatch):
    class DummyMilvusClient:
        @staticmethod
        def list_collections():
            return ["deepsea_ocean"]

    monkeypatch.setattr(
        routes_graph,
        "_query_chunk_in_collection",
        lambda collection, chunk_id: {
            "collection": collection,
            "chunk_id": chunk_id,
            "paper_id": "paper_x",
            "content": "milvus content",
            "section_path": "Intro",
            "page": 0,
            "content_type": "text",
            "chunk_type": "paragraph",
            "bbox": None,
        },
    )
    monkeypatch.setattr(
        routes_graph,
        "_query_chunk_from_parsed",
        lambda paper_id, chunk_id, user_id=None, collection=None: {
            "collection": "parsed_fallback",
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "content": "parsed content",
            "section_path": "Intro > Detail",
            "page": 2,
            "content_type": "text",
            "chunk_type": "paragraph",
            "bbox": [10, 20, 30, 40],
        },
    )
    monkeypatch.setattr(routes_graph, "_get_hippo", lambda: None)
    monkeypatch.setattr(routes_graph, "settings", SimpleNamespace(collection=SimpleNamespace(all=lambda: ["deepsea_ocean"])))
    monkeypatch.setattr("src.indexing.milvus_ops.milvus", SimpleNamespace(client=DummyMilvusClient()))

    detail = routes_graph.graph_chunk_detail("chunk-1", collection="deepsea_ocean", paper_id="paper_x", user_id="u1")

    assert detail["chunk_id"] == "chunk-1"
    assert detail["bbox"] == [10, 20, 30, 40]
    assert detail["content"] == "parsed content"
    assert detail["page"] == 3
