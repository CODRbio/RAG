import json
from unittest.mock import MagicMock

from src.collaboration.intent.commands import analyze_chat_context
from src.collaboration.intent.parser import resolve_intent_provider_name


def test_analyze_chat_context_defaults_to_medium_when_topic_missing():
    client = MagicMock()
    client.chat.return_value = {
        "final_text": json.dumps(
            {
                "action": "rag",
                "context_status": "self_contained",
                "followup_mode": "fresh",
            }
        )
    }

    result = analyze_chat_context(
        message="继续展开",
        rolling_summary="summary",
        history=[],
        llm_client=client,
    )

    assert result.topic_relevance == "medium"


def test_analyze_chat_context_reuse_modes_are_never_low():
    client = MagicMock()
    client.chat.return_value = {
        "final_text": json.dumps(
            {
                "action": "rag",
                "context_status": "resolved",
                "rewritten_query": "继续展开深海热液口第二点",
                "followup_mode": "reuse_and_search",
                "topic_relevance": "low",
            }
        )
    }

    result = analyze_chat_context(
        message="继续展开第二点",
        rolling_summary="summary",
        history=[],
        llm_client=client,
    )

    assert result.followup_mode == "reuse_and_search"
    assert result.topic_relevance == "medium"


def test_resolve_intent_provider_name_prefers_dedicated_provider():
    provider = resolve_intent_provider_name(
        intent_provider="ollama-intent",
        configured_intent_provider="cfg-intent",
        ultra_lite_provider="openai-mini",
        llm_provider="deepseek",
    )

    assert provider == "ollama-intent"


def test_resolve_intent_provider_name_falls_back_to_ultra_lite():
    provider = resolve_intent_provider_name(
        intent_provider=None,
        configured_intent_provider="",
        ultra_lite_provider="local-mini",
        llm_provider="deepseek",
    )

    assert provider == "deepseek"
