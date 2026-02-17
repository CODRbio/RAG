"""
Research agent state compaction tests:
- _accumulate_evidence_chunks stores shallow copies
- original chunks are not mutated
- text/raw_content are overwritten by configured empty value
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Support direct execution: `python tests/test_research_agent_state_compaction.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collaboration.research.agent import _accumulate_evidence_chunks
from src.retrieval.evidence import EvidenceChunk


def _mk_chunk(chunk_id: str, text: str = "original text") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        score=0.88,
        source_type="dense",
        doc_title="Doc Title",
        authors=["Alice"],
        year=2024,
    )


def test_accumulate_compacts_text_and_keeps_original_untouched():
    state = {"evidence_chunks": []}
    chunk = _mk_chunk("c-1", text="very long evidence body")
    # Simulate optional payload fields that may exist on some chunk-like objects.
    chunk.raw_content = "raw payload"

    _accumulate_evidence_chunks(state, [chunk])

    assert len(state["evidence_chunks"]) == 1
    stored = state["evidence_chunks"][0]

    # Stored object should be a copy, not the original instance.
    assert stored is not chunk
    assert stored.chunk_id == chunk.chunk_id

    # Stored payload fields are compacted.
    assert stored.text == ""
    assert getattr(stored, "raw_content", None) == ""

    # Original chunk remains unchanged.
    assert chunk.text == "very long evidence body"
    assert chunk.raw_content == "raw payload"


def test_accumulate_uses_configurable_empty_value():
    state = {"evidence_chunks": [], "evidence_chunk_empty_value": None}
    chunk = SimpleNamespace(chunk_id="c-2", text="body", raw_content="raw")

    _accumulate_evidence_chunks(state, [chunk])

    stored = state["evidence_chunks"][0]
    assert stored is not chunk
    assert stored.text is None
    assert stored.raw_content is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
