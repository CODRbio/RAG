import json
from typing import Any, List
from unittest.mock import MagicMock

from src.collaboration.intent.commands import analyze_chat_context, build_search_query_from_context
from src.collaboration.intent.parser import ParsedIntent
from src.collaboration.memory.session_memory import SessionStore
from src.retrieval.structured_queries import (
    generate_structured_queries_1plus1plus1,
    web_queries_per_provider_from_1plus1plus1,
)


class _FakeLlmClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def chat(self, messages=None, **kwargs):
        return {"final_text": json.dumps(self._payload, ensure_ascii=False)}


def test_analyze_chat_context_rewrite_only():
    client = _FakeLlmClient(
        {
            "action": "chat",
            "context_status": "resolved",
            "rewritten_query": "Please rewrite the previous answer into 5 bullets.",
            "clarification": "",
            "followup_mode": "reuse_only",
            "topic_relevance": "high",
            "target_span": "previous answer",
        }
    )
    out = analyze_chat_context(
        message="把上面的回答整理成5个要点",
        rolling_summary="",
        history=[],
        llm_client=client,
    )
    assert out.followup_mode == "reuse_only"
    assert out.topic_relevance == "high"
    assert out.target_span == "previous answer"


def test_analyze_chat_context_followup_reuse_and_search():
    client = _FakeLlmClient(
        {
            "action": "rag",
            "context_status": "resolved",
            "rewritten_query": "Compare the second and third papers mentioned above.",
            "clarification": "",
            "followup_mode": "reuse_and_search",
            "topic_relevance": "high",
            "target_span": "the papers discussed above",
        }
    )
    out = analyze_chat_context(
        message="继续对比上面提到的第二篇和第三篇",
        rolling_summary="",
        history=[],
        llm_client=client,
    )
    assert out.action == "rag"
    assert out.followup_mode == "reuse_and_search"
    assert out.topic_relevance == "high"


def test_analyze_chat_context_low_relevance_defaults_fresh():
    client = _FakeLlmClient(
        {
            "action": "rag",
            "context_status": "self_contained",
            "rewritten_query": "",
            "clarification": "",
            "followup_mode": "fresh",
            "topic_relevance": "low",
            "target_span": "",
        }
    )
    out = analyze_chat_context(
        message="顺便问下量子纠缠的应用",
        rolling_summary="",
        history=[],
        llm_client=client,
    )
    assert out.followup_mode == "fresh"
    assert out.topic_relevance == "low"


# ---------------------------------------------------------------------------
# Regression tests for "请重新总结"-style follow-up query drift fix
# ---------------------------------------------------------------------------

def test_analyze_chat_context_resumen_returns_reuse_only():
    """LLM returning reuse_only for '重新总结' is parsed correctly."""
    client = _FakeLlmClient(
        {
            "action": "rag",
            "context_status": "resolved",
            "rewritten_query": '请重新总结"病毒在深海贻贝及其共生体的进化与适应中的作用"。',
            "clarification": "",
            "followup_mode": "reuse_only",
            "topic_relevance": "high",
            "target_span": "上一轮关于深海贻贝病毒共生的回答",
        }
    )
    out = analyze_chat_context(
        message="请重新总结",
        rolling_summary="上一轮回答了病毒在深海贻贝共生体中的作用。",
        history=[],
        llm_client=client,
    )
    assert out.followup_mode == "reuse_only"
    assert out.topic_relevance == "high"
    assert "病毒" in out.rewritten_query


def test_analyze_chat_context_new_summarize_topic_stays_fresh():
    """Asking to summarize a NEW topic should not collapse into reuse_only."""
    client = _FakeLlmClient(
        {
            "action": "rag",
            "context_status": "self_contained",
            "rewritten_query": "",
            "clarification": "",
            "followup_mode": "fresh",
            "topic_relevance": "low",
            "target_span": "",
        }
    )
    out = analyze_chat_context(
        message="总结一下深海热液喷口的地质成因",
        rolling_summary="",
        history=[],
        llm_client=client,
    )
    assert out.followup_mode == "fresh"


def test_build_search_query_preserves_resolved_query():
    """When a resolved query is self-contained (no unresolved refs), it is returned as-is."""
    resolved = '请重新总结"病毒在深海贻贝（bathymodilines）及其共生体的进化与适应中的作用"。'
    parsed = ParsedIntent(
        intent_type=None,  # type: ignore[arg-type]
        confidence=0.9,
        from_command=False,
    )
    result = build_search_query_from_context(
        parsed,
        fallback=resolved,
        history=[],
        llm_client=None,
    )
    assert "bathymodilines" in result or "深海贻贝" in result, (
        f"Resolved query should be preserved; got: {result!r}"
    )


def test_generate_structured_queries_receives_resolved_query():
    """
    Verify that generate_structured_queries_1plus1plus1 passes the full resolved
    query to the LLM, not a short raw message. The mock captures the prompt
    sent to the client and verifies it contains the topic-bearing query.
    """
    captured_messages: List[Any] = []
    full_query = '病毒在深海贻贝（Bathymodiolinae）及其共生体的进化与适应中的作用'

    def _mock_chat(messages=None, **kwargs):
        captured_messages.extend(messages or [])
        return {
            "final_text": (
                "RECALL:\n"
                "deep-sea mussel Bathymodiolinae virus symbiont evolution\n"
                "PRECISION:\n"
                "Bathymodiolinae endosymbiont phage virus adaptive evolution mechanism 2015-2024\n"
                "DISCOVERY:\n"
                "What role do viruses play in the evolution of deep-sea mussel symbionts?\n"
            )
        }

    mock_client = MagicMock()
    mock_client.chat.side_effect = _mock_chat

    result = generate_structured_queries_1plus1plus1(
        query=full_query,
        evidence_context="(none)",
        llm_client=mock_client,
    )

    assert result is not None, "Should parse a valid structured result"
    assert result.get("recall"), "recall key must be present"
    # The prompt sent to the LLM must contain the full topic query
    all_prompt_text = " ".join(
        m.get("content", "") for m in captured_messages if isinstance(m, dict)
    )
    assert "Bathymodilin" in all_prompt_text or "深海贻贝" in all_prompt_text, (
        f"LLM prompt should contain the full topic query; prompt snippet: {all_prompt_text[:300]}"
    )

    # Also verify a short raw message would NOT have produced an on-topic recall
    captured_short: List[Any] = []

    def _mock_short(messages=None, **kwargs):
        captured_short.extend(messages or [])
        return {
            "final_text": (
                "RECALL:\narticle summarization paraphrasing techniques\n"
                "PRECISION:\nsummarization methods review article key points\n"
                "DISCOVERY:\nWhat are the best methods to summarize an article?\n"
            )
        }

    mock_short = MagicMock()
    mock_short.chat.side_effect = _mock_short

    result_short = generate_structured_queries_1plus1plus1(
        query="请重新总结",
        evidence_context="(none)",
        llm_client=mock_short,
    )
    assert result_short is not None
    assert "summarization" in (result_short.get("recall") or ""), (
        "Short raw message leads to off-topic recall — this is the broken behavior the fix prevents"
    )


def test_chinese_query_parses_precision_zh():
    """When query is Chinese and LLM returns PRECISION_ZH, it is parsed and present in result."""
    def _mock_chat(messages=None, **kwargs):
        return {
            "final_text": (
                "RECALL:\n"
                "deep-sea mussel symbiosis immune\n"
                "PRECISION:\n"
                "Bathymodiolus transcriptome immune 2020-2024\n"
                "DISCOVERY:\n"
                "What are key immune mechanisms in deep-sea mussel symbiosis?\n"
                "PRECISION_ZH:\n"
                "深海贻贝 共生体 免疫 转录组 2020-2024\n"
            )
        }

    mock_client = MagicMock()
    mock_client.chat.side_effect = _mock_chat

    result = generate_structured_queries_1plus1plus1(
        query="深海贻贝共生体的免疫机制有哪些？",
        evidence_context="(none)",
        llm_client=mock_client,
    )

    assert result is not None
    assert result.get("recall")
    assert result.get("precision")
    assert result.get("discovery")
    assert result.get("precision_zh") == "深海贻贝 共生体 免疫 转录组 2020-2024"


def test_precision_zh_routed_only_to_google_scholar():
    """precision_zh is appended only for scholar and google; ncbi/semantic/tavily unchanged."""
    structured = {
        "recall": "r",
        "precision": "p",
        "discovery": "d",
        "precision_zh": "中文精准词",
    }
    providers = ["ncbi", "semantic", "scholar", "google", "tavily"]
    qpp = web_queries_per_provider_from_1plus1plus1(structured, providers)

    assert qpp["ncbi"] == ["r", "p"]
    assert qpp["semantic"] == ["r", "p"]
    assert qpp["scholar"] == ["r", "p", "中文精准词"]
    assert qpp["google"] == ["r", "p", "中文精准词"]
    assert qpp["tavily"] == ["d"]


def test_non_chinese_query_no_precision_zh():
    """English query returns only recall/precision/discovery; no precision_zh key."""
    def _mock_chat(messages=None, **kwargs):
        return {
            "final_text": (
                "RECALL:\n"
                "machine learning fairness\n"
                "PRECISION:\n"
                "ML fairness bias mitigation 2019-2024\n"
                "DISCOVERY:\n"
                "How is fairness addressed in machine learning systems?\n"
            )
        }

    mock_client = MagicMock()
    mock_client.chat.side_effect = _mock_chat

    result = generate_structured_queries_1plus1plus1(
        query="How do we ensure fairness in ML systems?",
        evidence_context="(none)",
        llm_client=mock_client,
    )

    assert result is not None
    assert set(result.keys()) == {"recall", "precision", "discovery"}
    assert "precision_zh" not in result

    # Provider mapping without precision_zh: scholar/google get only recall+precision
    qpp = web_queries_per_provider_from_1plus1plus1(result, ["scholar", "google"])
    assert qpp["scholar"] == ["machine learning fairness", "ML fairness bias mitigation 2019-2024"]
    assert qpp["google"] == ["machine learning fairness", "ML fairness bias mitigation 2019-2024"]


def test_session_recent_evidence_cache_is_bounded(monkeypatch):
    store = SessionStore()
    state = {"preferences": {}}

    def _get_meta(_session_id):
        return {
            "session_id": "s1",
            "canvas_id": "",
            "stage": "explore",
            "rolling_summary": "",
            "summary_at_turn": 0,
            "title": "",
            "session_type": "chat",
            "preferences": dict(state["preferences"]),
            "created_at": "",
            "updated_at": "",
        }

    def _update_meta(_session_id, meta):
        prefs = state["preferences"]
        incoming = meta.get("preferences") or {}
        prefs.update(incoming)
        state["preferences"] = prefs

    monkeypatch.setattr(store, "get_session_meta", _get_meta)
    monkeypatch.setattr(store, "update_session_meta", _update_meta)

    chunk = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "x" * 3000,
        "score": 0.9,
        "source_type": "web",
        "provider": "scholar",
    }
    for i in range(5):
        store.append_recent_evidence_cache(
            "s1",
            query=f"q{i}",
            chunks=[{**chunk, "chunk_id": f"c{i}"}],
            max_turns=3,
            max_chunks_per_turn=1,
        )

    cached = store.get_recent_evidence_cache("s1")
    assert len(cached) == 3
    assert [item["query"] for item in cached] == ["q2", "q3", "q4"]
    assert len(cached[-1]["chunks"]) == 1
    assert len(cached[-1]["chunks"][0]["text"]) == 2200
