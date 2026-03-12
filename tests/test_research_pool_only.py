from __future__ import annotations

from types import SimpleNamespace

from src.collaboration.research import agent as research_agent
from src.collaboration.research.dashboard import ResearchBrief, ResearchDashboard
from src.collaboration.research.trajectory import ResearchTrajectory
from src.retrieval.evidence import EvidenceChunk, EvidencePack


def _pack(query: str, chunk_id: str = "c1") -> EvidencePack:
    return EvidencePack(
        query=query,
        chunks=[
            EvidenceChunk(
                chunk_id=chunk_id,
                doc_id=f"doc_{chunk_id}",
                text=f"text {chunk_id}",
                score=0.1,
                source_type="web",
                provider="semantic",
            )
        ],
        total_candidates=1,
        retrieval_time_ms=0.0,
        sources_used=["semantic"],
    )


class _RecordingSvc:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _pack(kwargs.get("query", "q"), chunk_id=f"c{len(self.calls)}")


def test_research_node_main_search_uses_pool_only(monkeypatch):
    dashboard = ResearchDashboard(brief=ResearchBrief(topic="DeepSea symbiosis"))
    dashboard.add_section("Mechanisms")
    trajectory = ResearchTrajectory(topic="DeepSea symbiosis")
    svc = _RecordingSvc()

    monkeypatch.setattr(research_agent, "_get_retrieval_svc", lambda state: svc)
    monkeypatch.setattr(research_agent, "_resolve_runtime_llm_client", lambda state: object())
    monkeypatch.setattr(
        "src.retrieval.structured_queries.generate_structured_queries_1plus1plus1",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(research_agent, "_retrieve_temp_snippets", lambda *args, **kwargs: [])
    monkeypatch.setattr(research_agent, "_emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_sync_outline_status_to_canvas", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "compress_trajectory", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_accumulate_evidence_chunks", lambda *args, **kwargs: None)

    state = {
        "dashboard": dashboard,
        "trajectory": trajectory,
        "filters": {"step_top_k": 6, "web_providers": ["semantic"]},
        "search_mode": "hybrid",
    }

    out = research_agent.research_node(state)

    assert out["current_section"] == "Mechanisms"
    assert svc.calls, "Expected main research retrieval to run"
    assert svc.calls[0]["filters"]["pool_only"] is True


def test_evaluate_node_only_supplement_search_uses_pool_only(monkeypatch):
    dashboard = ResearchDashboard(brief=ResearchBrief(topic="DeepSea symbiosis"))
    section = dashboard.add_section("Mechanisms")
    trajectory = ResearchTrajectory(topic="DeepSea symbiosis")
    svc = _RecordingSvc()

    monkeypatch.setattr(research_agent, "_get_retrieval_svc", lambda state: svc)
    monkeypatch.setattr(research_agent, "_emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_accumulate_section_pool", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_accumulate_evidence_chunks", lambda *args, **kwargs: None)

    fake_eval = research_agent._CoverageEvalResponse(
        coverage_score=0.2,
        gaps=["missing sulfur oxidation evidence"],
        sufficient=False,
    )
    fake_client = SimpleNamespace(chat=lambda **kwargs: {"parsed_object": fake_eval})
    monkeypatch.setattr(
        research_agent,
        "_resolve_step_lite_client",
        lambda state, step: (fake_client, None),
    )

    state = {
        "dashboard": dashboard,
        "trajectory": trajectory,
        "current_section": "Mechanisms",
        "filters": {"step_top_k": 5},
        "search_mode": "hybrid",
        "section_evidence_pool": {},
    }

    research_agent.evaluate_node(state)

    assert len(svc.calls) >= 2, "Expected fallback retrieval plus supplement retrieval"
    assert svc.calls[0]["filters"].get("pool_only") is not True
    assert svc.calls[1]["filters"]["pool_only"] is True


def test_review_revise_supplement_search_uses_pool_only(monkeypatch):
    dashboard = ResearchDashboard(brief=ResearchBrief(topic="DeepSea symbiosis"))
    dashboard.add_section("Mechanisms")
    svc = _RecordingSvc()

    monkeypatch.setattr(research_agent, "_get_retrieval_svc", lambda state: svc)
    monkeypatch.setattr(research_agent, "_emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_accumulate_section_pool", lambda *args, **kwargs: None)
    monkeypatch.setattr(research_agent, "_accumulate_evidence_chunks", lambda *args, **kwargs: None)

    state = {
        "dashboard": dashboard,
        "current_section": "Mechanisms",
        "review_revise_feedback": "Need stronger sulfur oxidation evidence",
        "filters": {"step_top_k": 6},
        "search_mode": "hybrid",
    }

    research_agent.review_revise_agent_supplement_node(state)

    assert svc.calls, "Expected review-revise supplement retrieval to run"
    assert svc.calls[0]["filters"]["pool_only"] is True
