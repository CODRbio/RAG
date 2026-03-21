import json
from types import SimpleNamespace

from src.api.routes_chat import _evaluate_chat_evidence_sufficiency
from src.collaboration.intent.commands import analyze_chat_context
from src.collaboration.research.verifier import verify_claims
from src.retrieval.structured_queries import generate_structured_queries_1plus1plus1


class _RecordingLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages=None, **kwargs):
        self.calls.append({"messages": messages or [], "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("No mocked LLM responses left")
        return self._responses.pop(0)


def test_real_scenario_chat_context_analysis_attaches_conversation_cache_hint():
    llm = _RecordingLLM(
        [
            {
                "final_text": json.dumps(
                    {
                        "action": "rag",
                        "context_status": "resolved",
                        "rewritten_query": "Compare methane oxidation pathways discussed in the previous answer.",
                        "clarification": "",
                        "followup_mode": "reuse_and_search",
                        "topic_relevance": "high",
                        "target_span": "methane oxidation pathways in the previous answer",
                    }
                )
            }
        ]
    )

    result = analyze_chat_context(
        message="继续比较上面说的甲烷氧化路径",
        rolling_summary="上一轮回答了冷泉甲烷氧化菌与宿主互作机制。",
        history=[
            SimpleNamespace(role="user", content="冷泉甲烷氧化菌和宿主怎么互作？"),
            SimpleNamespace(role="assistant", content="我总结了三种主要互作机制。"),
        ],
        llm_client=llm,
    )

    assert result.followup_mode == "reuse_and_search"
    cache = llm.calls[0]["kwargs"]["cache"]
    assert cache["mode"] == "provider_only"
    assert cache["scope"] == "conversation"
    assert cache["key"].startswith("chat_context_analyze:")


def test_real_scenario_structured_query_generation_attaches_global_template_cache_hint():
    llm = _RecordingLLM(
        [
            {
                "final_text": (
                    "RECALL:\n"
                    "deep sea mussel methane oxidation symbiosis\n"
                    "PRECISION:\n"
                    "Bathymodiolus methane oxidation symbiont host interaction 2018-2025\n"
                    "DISCOVERY:\n"
                    "How do deep-sea mussels regulate methane-oxidizing symbionts?\n"
                )
            }
        ]
    )

    result = generate_structured_queries_1plus1plus1(
        query="How do deep-sea mussels regulate methane-oxidizing symbionts?",
        evidence_context="Prior evidence mentions host immune filtering and nutrient exchange.",
        llm_client=llm,
    )

    assert result == {
        "recall": "deep sea mussel methane oxidation symbiosis",
        "precision": "Bathymodiolus methane oxidation symbiont host interaction 2018-2025",
        "discovery": "How do deep-sea mussels regulate methane-oxidizing symbionts?",
    }
    cache = llm.calls[0]["kwargs"]["cache"]
    assert cache["mode"] == "provider_only"
    assert cache["scope"] == "global_template"
    assert cache["key"].startswith("chat_generate_queries:")


def test_real_scenario_chat_evidence_sufficiency_attaches_section_cache_hint():
    llm = _RecordingLLM(
        [
            {
                "final_text": json.dumps(
                    {
                        "sufficient": True,
                        "coverage_score": 0.91,
                        "substantive_support": True,
                        "specificity_ok": True,
                        "consistency_ok": True,
                        "reason": "The evidence directly covers the user question.",
                    }
                )
            }
        ]
    )

    result = _evaluate_chat_evidence_sufficiency(
        query="What evidence supports sulfur metabolism in Bathymodiolus symbionts?",
        evidence_context="Paper A reports sulfur oxidation genes. Paper B shows transcriptomic activation under sulfide-rich conditions.",
        llm_client=llm,
        model_override="gpt-5.4",
    )

    assert result["ok"] is True
    assert result["sufficient"] is True
    assert result["coverage_score"] == 0.91
    call = llm.calls[0]
    assert call["kwargs"]["model"] == "gpt-5.4"
    assert call["kwargs"]["response_model"].__name__ == "_ChatSufficiencyResponse"
    cache = call["kwargs"]["cache"]
    assert cache["scope"] == "section"
    assert cache["key"].startswith("chat_evidence_sufficiency:")


def test_real_scenario_verify_claims_attaches_section_cache_hints_for_both_steps():
    llm = _RecordingLLM(
        [
            {
                "final_text": json.dumps(
                    {
                        "claims": [
                            {
                                "claim": "Study A found sulfur oxidation genes enriched in symbionts.",
                                "has_citation": True,
                                "citation_keys": ["refA"],
                            }
                        ]
                    }
                )
            },
            {
                "final_text": json.dumps(
                    {
                        "verifications": [
                            {
                                "claim_index": 0,
                                "confidence": "high",
                                "evidence_found": "Citation refA explicitly reports sulfur oxidation genes.",
                                "needs_revision": False,
                                "revision_note": "",
                                "attribution_analysis": "",
                                "supplementary_query": "",
                            }
                        ]
                    }
                )
            },
        ]
    )

    result = verify_claims(
        section_text="Study A found sulfur oxidation genes enriched in symbionts [refA].",
        citations=[{"cite_key": "refA", "title": "Sulfur metabolism study", "year": 2024}],
        llm_client=llm,
        model="claude-sonnet-4-6",
    )

    assert result.total_claims == 1
    assert len(result.claims) == 1
    assert result.claims[0].confidence == "high"
    assert len(llm.calls) == 2
    assert llm.calls[0]["kwargs"]["cache"]["key"].startswith("extract_claims:")
    assert llm.calls[0]["kwargs"]["cache"]["scope"] == "section"
    assert llm.calls[1]["kwargs"]["cache"]["key"].startswith("verify_claims:")
    assert llm.calls[1]["kwargs"]["cache"]["scope"] == "section"
