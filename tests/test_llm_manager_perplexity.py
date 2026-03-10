from src.llm.llm_manager import HTTPChatClient, ProviderConfig, normalize_response


def _make_perplexity_client() -> HTTPChatClient:
    cfg = ProviderConfig(
        name="sonar",
        api_key="test-key",
        base_url="https://api.perplexity.ai",
        default_model="sonar-pro",
        platform="perplexity",
        models={},
        params={},
    )
    return HTTPChatClient(cfg, provider=None)


def test_perplexity_payload_drops_openai_only_fields_and_tool_metadata():
    client = _make_perplexity_client()
    payload = client._build_openai_payload(
        messages=[
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_local", "arguments": "{\"query\":\"abc\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "search_local", "content": "{\"ok\": true}"},
        ],
        model="sonar-reasoning-pro",
        params={
            "reasoning_effort": "medium",
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "n": 2,
            "max_tokens": 200_000,
        },
        tools=[{"type": "function", "function": {"name": "search_local", "parameters": {}}}],
    )
    assert "tools" not in payload
    assert "frequency_penalty" not in payload
    assert "presence_penalty" not in payload
    assert "n" not in payload
    assert payload.get("reasoning_effort") == "medium"
    assert payload.get("max_tokens") == 128_000
    assert payload["messages"][1] == {"role": "assistant", "content": ""}
    assert payload["messages"][2] == {"role": "tool", "content": "{\"ok\": true}"}


def test_perplexity_structured_output_uses_json_schema():
    client = _make_perplexity_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="sonar-pro",
        params={"response_format": {"type": "json_object"}},
        tools=None,
    )
    assert payload.get("response_format", {}).get("type") == "json_schema"
    assert payload.get("response_format", {}).get("json_schema", {}).get("schema") == {"type": "object"}


def test_normalize_response_extracts_perplexity_citations_and_search_results():
    raw = {
        "choices": [{"message": {"content": "Answer"}}],
        "citations": ["https://example.com/a", "https://example.com/b"],
        "search_results": [{"title": "A", "url": "https://example.com/a"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    normalized = normalize_response("sonar", raw, is_anthropic=False)
    assert normalized["final_text"] == "Answer"
    assert normalized["citations"] == ["https://example.com/a", "https://example.com/b"]
    assert normalized["search_results"] == [{"title": "A", "url": "https://example.com/a"}]
