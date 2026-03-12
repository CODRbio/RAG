from __future__ import annotations

from typing import Any, Dict, List

from src.llm import tools as tools_mod
from src.retrieval.evidence import EvidenceChunk


class _FakePack:
    def __init__(self, chunks: List[EvidenceChunk]):
        self.chunks = chunks

    def to_context_string(self, max_chunks: int = 10) -> str:
        return f"context({max_chunks})"


class _FakeSvc:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def search(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _FakePack(
            [
                EvidenceChunk(
                    chunk_id=f"c{i}",
                    doc_id=f"d{i}",
                    text=f"x{i}",
                    score=1.0,
                    source_type="dense",
                )
                for i in range(1, 4)
            ]
        )


def test_search_local_respects_request_scoped_step_top_k(monkeypatch):
    fake = _FakeSvc()
    monkeypatch.setattr("src.retrieval.service.get_retrieval_service", lambda collection=None: fake)
    tools_mod.set_tool_collection("test_col")
    tools_mod.set_tool_step_top_k(5)
    tools_mod.start_agent_chunk_collector()

    _ = tools_mod._handle_search_local(query="q", top_k=20)
    collected = tools_mod.drain_agent_chunks()

    assert fake.calls, "search should be called"
    call = fake.calls[-1]
    assert call["top_k"] == 5
    assert call["filters"]["step_top_k"] == 5
    assert call["filters"]["reranker_mode"] == "bge_only"
    assert call["filters"]["pool_only"] is True
    assert [c.chunk_id for c in collected] == ["c1", "c2", "c3"]


def test_search_web_without_step_top_k_uses_tool_top_k(monkeypatch):
    fake = _FakeSvc()
    monkeypatch.setattr("src.retrieval.service.get_retrieval_service", lambda collection=None: fake)
    tools_mod.set_tool_collection("test_col")
    tools_mod.set_tool_step_top_k(None)
    tools_mod.start_agent_chunk_collector()

    _ = tools_mod._handle_search_web(query="q", top_k=7)
    collected = tools_mod.drain_agent_chunks()

    assert fake.calls, "search should be called"
    call = fake.calls[-1]
    assert call["top_k"] == 7
    assert "step_top_k" not in call["filters"]
    assert call["filters"]["pool_only"] is True
    assert [c.chunk_id for c in collected] == ["c1", "c2", "c3"]
