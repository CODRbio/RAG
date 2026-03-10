from __future__ import annotations

from typing import Any, Dict, List

from src.api.routes_chat import _fuse_chat_main_gap_agent_candidates
from src.retrieval.evidence import EvidenceChunk


def _chunk(chunk_id: str, score: float, source_type: str = "dense", provider: str = "local") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        source_type=source_type,
        doc_title=f"title {chunk_id}",
        provider=provider,
    )


def _gap_hit(chunk_id: str, score: float) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content": f"gap {chunk_id}",
        "score": score,
        "_source_type": "web",
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": f"doc_{chunk_id}",
            "title": f"title {chunk_id}",
            "provider": "web",
        },
    }


def test_chat_agent_refusion_three_pool_soft_targets():
    main_chunks: List[EvidenceChunk] = [_chunk(f"m{i}", 1.0 - i * 0.01) for i in range(8)]
    gap_hits = [_gap_hit("g1", 0.2), _gap_hit("g2", 0.1)]
    agent_chunks = [_chunk("a1", 0.15, source_type="web", provider="web")]

    out_chunks, pool_fusion = _fuse_chat_main_gap_agent_candidates(
        query="q",
        message="q",
        main_chunks=main_chunks,
        gap_candidate_hits=gap_hits,
        agent_chunks=agent_chunks,
        write_k=5,
        filters={},
    )

    out_ids = {c.chunk_id for c in out_chunks}
    assert len(out_chunks) <= 5
    assert "g1" in out_ids or "g2" in out_ids
    assert "a1" in out_ids
    assert pool_fusion.get("gap_in_output", 0) >= 1
    assert pool_fusion.get("agent_in_output", 0) >= 1


def test_chat_agent_refusion_no_agent_chunks():
    main_chunks: List[EvidenceChunk] = [_chunk(f"m{i}", 1.0 - i * 0.01) for i in range(6)]
    gap_hits = [_gap_hit("g1", 0.2)]
    out_chunks, pool_fusion = _fuse_chat_main_gap_agent_candidates(
        query="q",
        message="q",
        main_chunks=main_chunks,
        gap_candidate_hits=gap_hits,
        agent_chunks=[],
        write_k=4,
        filters={},
    )
    assert len(out_chunks) <= 4
    assert pool_fusion.get("agent_in", 0) == 0
    assert pool_fusion.get("agent_in_output", 0) == 0
