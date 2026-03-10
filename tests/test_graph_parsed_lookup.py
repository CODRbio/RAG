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
