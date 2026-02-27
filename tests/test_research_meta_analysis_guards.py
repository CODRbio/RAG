"""
Regression tests for debate/meta-analysis enhancements:
- verifier CoV conflict attribution extraction
- write_node quantitative run_code enforcement path
- synthesize_node Debate & Divergence prompt enrichment
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Support direct execution: `python tests/test_research_meta_analysis_guards.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collaboration.research import agent as research_agent
from src.collaboration.research.dashboard import ResearchBrief, ResearchDashboard
from src.collaboration.research.trajectory import ResearchTrajectory
from src.collaboration.research.verifier import verify_claims


class _SequentialLLM:
    """Simple mock client returning pre-seeded chat responses in order."""

    def __init__(self, payloads):
        self._payloads = list(payloads)

    def chat(self, **_kwargs):
        if not self._payloads:
            raise AssertionError("No more mocked LLM payloads available")
        return self._payloads.pop(0)


def test_verify_claims_extracts_attribution_into_conflict_notes():
    claims_json = json.dumps(
        [
            {
                "claim": "Study A reports higher diversity than Study B.",
                "has_citation": True,
                "citation_keys": ["refA", "refB"],
            }
        ],
        ensure_ascii=False,
    )
    verification_json = json.dumps(
        [
            {
                "claim_index": 0,
                "confidence": "medium",
                "evidence_found": "Both studies examine similar ecosystems but report opposite trends.",
                "needs_revision": True,
                "revision_note": "结论存在冲突，需要解释。",
                "attribution_analysis": "差异可能来自采样深度和测序仪器（NovaSeq vs MiSeq）不同。",
                "supplementary_query": "",
            }
        ],
        ensure_ascii=False,
    )
    llm = _SequentialLLM(
        [
            {"final_text": claims_json},
            {"final_text": verification_json},
        ]
    )

    result = verify_claims(
        section_text="A paragraph containing a conflict claim.",
        citations=[],
        llm_client=llm,
        model=None,
    )

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert "采样深度" in claim.attribution_analysis
    assert "Attribution Analysis:" in claim.revision_note
    assert result.conflict_notes, "Expected conflict attribution notes to be populated"
    assert "测序仪器" in result.conflict_notes[0]


def test_effective_write_k_respects_floor_scaling_and_cap():
    preset = {"search_top_k_write": 12, "search_top_k_write_max": 40}

    # No UI override -> preset floor.
    assert research_agent._compute_effective_write_k(preset, {}) == 12
    # Moderate UI top_k -> 1.5x scaling.
    assert research_agent._compute_effective_write_k(preset, {"step_top_k": 10}) == 15
    # Large UI top_k -> capped.
    assert research_agent._compute_effective_write_k(preset, {"step_top_k": 100}) == 40

    # Cap below floor is normalized to floor.
    preset_bad_cap = {"search_top_k_write": 20, "search_top_k_write_max": 10}
    assert research_agent._compute_effective_write_k(preset_bad_cap, {"step_top_k": 200}) == 20

    # Invalid UI top_k falls back to preset floor.
    assert research_agent._compute_effective_write_k(preset, {"step_top_k": "not-a-number"}) == 12


def test_write_node_uses_run_code_path_when_structured_numeric_data_present(monkeypatch):
    dashboard = ResearchDashboard(brief=ResearchBrief(topic="DeepSea microbiome"))
    dashboard.add_section("Quantitative Comparison")
    trajectory = ResearchTrajectory(topic="DeepSea microbiome")

    class _DummyPack:
        def __init__(self, text):
            self._text = text
            self.chunks = []

        def to_context_string(self, max_chunks=8):
            _ = max_chunks
            return self._text

    class _DummyRetrievalSvc:
        def search(self, query, mode, top_k, filters):
            _ = mode, top_k, filters
            if "verification" in query.lower():
                return _DummyPack("verification table computed_stats mean=10 std=2")
            return _DummyPack("primary table computed_stats mean=12 std=3")

    # Keep LLM client simple; in this branch content comes from react_loop.
    dummy_client = SimpleNamespace(chat=lambda **_kwargs: {"final_text": "fallback text"})
    monkeypatch.setattr(
        research_agent,
        "_resolve_step_client_and_model",
        lambda state, step: (dummy_client, None),
    )
    monkeypatch.setattr(research_agent, "_get_retrieval_svc", lambda state: _DummyRetrievalSvc())

    captured = {"called": False, "tool_names": []}

    def _fake_react_loop(messages, tools, llm_client, max_iterations=10, model=None, **llm_kwargs):
        _ = messages, llm_client, max_iterations, model, llm_kwargs
        captured["called"] = True
        captured["tool_names"] = [getattr(t, "name", "") for t in tools]
        return SimpleNamespace(final_text="Computed differences via run_code.")

    import src.llm.react_loop as react_loop_module

    monkeypatch.setattr(react_loop_module, "react_loop", _fake_react_loop)

    state = {
        "dashboard": dashboard,
        "trajectory": trajectory,
        "current_section": "Quantitative Comparison",
        "markdown_parts": [],
        "output_language": "en",
        "search_mode": "hybrid",
        "filters": {},
    }

    out = research_agent.write_node(state)
    assert captured["called"], "Expected write_node to enter the react_loop path"
    assert captured["tool_names"] == ["run_code"], "Expected run_code-only toolset in quantitative mode"
    assert any("Computed differences via run_code." in part for part in out.get("markdown_parts", []))


def test_synthesize_node_injects_debate_divergence_requirements(monkeypatch):
    dashboard = ResearchDashboard(brief=ResearchBrief(topic="DeepSea symbiosis"))
    sec = dashboard.add_section("Findings")
    sec.status = "done"
    dashboard.conflict_notes = [
        "Study A vs Study B disagree; potential drivers include sampling depth (2000m vs 3500m) and sequencing platform.",
    ]

    captured = {"limitations_prompt": ""}

    class _CaptureLLM:
        def chat(self, messages, model=None, max_tokens=None, **kwargs):
            _ = model, max_tokens, kwargs
            user_prompt = str((messages or [{}])[-1].get("content", ""))
            if "Generate a 150-250 word abstract" in user_prompt:
                return {"final_text": "Abstract body."}
            if "write the BODY content for section" in user_prompt:
                captured["limitations_prompt"] = user_prompt
                return {"final_text": "观点交锋与实验条件差异 (Debate & Divergence): analysis paragraph."}
            if "Rewrite the full review to improve global coherence" in user_prompt:
                body = user_prompt.split("Document:\n", 1)[-1]
                return {"final_text": body}
            return {"final_text": ""}

    monkeypatch.setattr(
        research_agent,
        "_resolve_step_client_and_model",
        lambda state, step: (_CaptureLLM(), None),
    )

    state = {
        "dashboard": dashboard,
        "markdown_parts": ["# DeepSea symbiosis\n\n## Findings\n\nSome evidence text."],
        "sections_completed": ["Findings"],
        "output_language": "en",
        "citations": [],
    }

    research_agent.synthesize_node(state)
    lim_prompt = captured["limitations_prompt"]
    assert lim_prompt, "Expected limitations prompt to be generated"
    assert "观点交锋与实验条件差异 (Debate & Divergence)" in lim_prompt
    assert "sampling depth" in lim_prompt
    assert "Conflicts/Contradictions with Attribution Clues" in lim_prompt
    assert "Study A vs Study B disagree" in lim_prompt


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
