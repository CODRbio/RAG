import json

from src.api import routes_compare


def test_parsed_roots_include_library_paths_then_fallback(monkeypatch, tmp_path):
    lib_a = tmp_path / "users" / "u1" / "libraries" / "A" / "parsed_data"
    lib_b = tmp_path / "users" / "u1" / "libraries" / "B" / "parsed_data"
    legacy = tmp_path / "users" / "u1" / "library" / "parsed_raw"
    for p in (lib_a, lib_b, legacy):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(routes_compare.PathManager, "get_user_all_library_parsed_paths", lambda _uid: [lib_a, lib_b])
    monkeypatch.setattr(routes_compare.PathManager, "get_user_parsed_path", lambda _uid: legacy)
    roots = routes_compare._parsed_roots("u1")
    assert roots[0] == lib_a
    assert roots[1] == lib_b
    assert legacy in roots
    assert any(str(p).endswith("/parsed") for p in roots)


def test_list_available_papers_aggregates_across_library_parsed_dirs(monkeypatch, tmp_path):
    lib_a = tmp_path / "users" / "u1" / "libraries" / "A" / "parsed_data" / "paper_a"
    lib_b = tmp_path / "users" / "u1" / "libraries" / "B" / "parsed_data" / "paper_b"
    lib_a.mkdir(parents=True, exist_ok=True)
    lib_b.mkdir(parents=True, exist_ok=True)

    with open(lib_a / "enriched.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "paper_a", "content_flow": []}, f)
    with open(lib_b / "enriched.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": "paper_b", "content_flow": []}, f)

    monkeypatch.setattr(
        routes_compare,
        "_parsed_roots",
        lambda _uid=None: [tmp_path / "users" / "u1" / "libraries" / "A" / "parsed_data", tmp_path / "users" / "u1" / "libraries" / "B" / "parsed_data"],
    )

    out = routes_compare.list_available_papers(limit=50, offset=0, q=None, user_id="u1")
    ids = {p["paper_id"] for p in out["papers"]}
    assert {"paper_a", "paper_b"} <= ids
