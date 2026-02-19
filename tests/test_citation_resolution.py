"""
Citation resolution tests:
- [ref:xxxx] placeholder -> cite_key replacement
- document-level merge (multiple chunks from same doc)
- cross-stage stable cite_key reuse
"""

import sys
from pathlib import Path

# Support direct execution: `python tests/test_citation_resolution.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collaboration.citation.manager import resolve_response_citations
from src.retrieval.evidence import EvidenceChunk


def _mk_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    text: str = "evidence text",
    title: str = "Sample Title",
    authors=None,
    year: int | None = 2024,
    url: str | None = None,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=0.9,
        source_type="web" if url else "dense",
        doc_title=title,
        authors=authors or ["Smith, John"],
        year=year,
        url=url,
    )


def test_same_document_multiple_hashes_resolve_to_one_citation():
    c1 = _mk_chunk(chunk_id="chunk-a", doc_id="doc-1", title="Paper A")
    c2 = _mk_chunk(chunk_id="chunk-b", doc_id="doc-1", title="Paper A")
    c3 = _mk_chunk(chunk_id="chunk-c", doc_id="doc-2", title="Paper B", authors=["Jones, Amy"], year=2023)

    raw = f"Claim X [{c1.ref_hash}] and supporting Y [{c2.ref_hash}]. Compare Z [{c3.ref_hash}]."
    resolved, citations, ref_map = resolve_response_citations(raw, [c1, c2, c3], format="author_date")

    # same doc -> same cite_key
    assert ref_map[c1.ref_hash] == ref_map[c2.ref_hash]
    assert len(citations) == 2

    # both hashes replaced to same key
    assert f"[{c1.ref_hash}]" not in resolved
    assert f"[{c2.ref_hash}]" not in resolved
    assert f"[{c3.ref_hash}]" not in resolved
    assert resolved.count(f"[{ref_map[c1.ref_hash]}]") == 2
    assert resolved.count(f"[{ref_map[c3.ref_hash]}]") == 1


def test_cross_stage_reuse_keeps_stable_numeric_keys():
    # phase 1
    a1 = _mk_chunk(chunk_id="a-1", doc_id="doc-a", title="Doc A")
    b1 = _mk_chunk(chunk_id="b-1", doc_id="doc-b", title="Doc B", authors=["Brown, Kim"], year=2022)
    text1 = f"A [{a1.ref_hash}] B [{b1.ref_hash}]"

    shared_doc_map = {}
    shared_keys = set()
    resolved1, cites1, ref_map1 = resolve_response_citations(
        text1,
        [a1, b1],
        format="numeric",
        doc_key_to_cite_key=shared_doc_map,
        existing_cite_keys=shared_keys,
        include_unreferenced_documents=False,
    )

    assert resolved1 == "A [1] B [2]"
    assert len(cites1) == 2

    # phase 2: doc-a appears again + new doc-c
    a2 = _mk_chunk(chunk_id="a-2", doc_id="doc-a", title="Doc A")
    c1 = _mk_chunk(chunk_id="c-1", doc_id="doc-c", title="Doc C", authors=["Chen, Li"], year=2021)
    text2 = f"A-again [{a2.ref_hash}] and C [{c1.ref_hash}]"
    resolved2, cites2, ref_map2 = resolve_response_citations(
        text2,
        [a2, c1],
        format="numeric",
        doc_key_to_cite_key=shared_doc_map,
        existing_cite_keys=shared_keys,
        include_unreferenced_documents=False,
    )

    # doc-a keeps key [1], new doc-c gets next key [3]
    assert resolved2 == "A-again [1] and C [3]"
    assert ref_map1[a1.ref_hash] == "1"
    assert ref_map2[a2.ref_hash] == "1"
    assert ref_map2[c1.ref_hash] == "3"
    assert len(cites2) == 2


def test_uppercase_hex_in_hash_is_supported_for_replacement():
    """Uppercase hex digits in the [ref:XXXX] placeholder should still be resolved."""
    from src.retrieval.evidence import REF_PREFIX
    c = _mk_chunk(chunk_id="upper-1", doc_id="doc-upper", title="Upper Hash Doc")
    # ref_hash returns "ref:xxxxxxxx"; use uppercase hex digits only (keep prefix lowercase)
    hex_part = c.ref_hash[len(REF_PREFIX):]
    raw = f"Upper hash citation [ref:{hex_part.upper()}]."
    resolved, _citations, _ref_map = resolve_response_citations(raw, [c], format="numeric")
    assert resolved == "Upper hash citation [1]."


def test_can_exclude_unreferenced_documents_from_citation_list():
    c1 = _mk_chunk(chunk_id="r-1", doc_id="doc-ref", title="Referenced Doc")
    c2 = _mk_chunk(chunk_id="u-1", doc_id="doc-unref", title="Unreferenced Doc")
    raw = f"Only this one is used [{c1.ref_hash}]"

    resolved, citations, ref_map = resolve_response_citations(
        raw,
        [c1, c2],
        format="author_date",
        include_unreferenced_documents=False,
    )

    assert f"[{c1.ref_hash}]" not in resolved
    assert c2.ref_hash not in ref_map
    assert len(citations) == 1
    assert citations[0].doc_id == "doc-ref"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
