from src.llm.llm_manager import HTTPChatClient, ProviderConfig
from src.llm.tools import has_tool_calls, parse_tool_calls


def _make_client(name: str, base_url: str, default_model: str = "gpt-5.2") -> HTTPChatClient:
    cfg = ProviderConfig(
        name=name,
        api_key="test-key",
        base_url=base_url,
        default_model=default_model,
        platform=name,
        models={},
        params={},
    )
    return HTTPChatClient(cfg, provider=None)


def test_openai_gpt5_tools_use_responses_api():
    client = _make_client("openai-thinking", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.4",
        params={"reasoning_effort": "high"},
        tools=[{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
    )
    assert payload.get("_api_mode") == "responses"
    assert "tools" in payload
    assert payload.get("reasoning") == {"effort": "high"}
    assert "input" in payload
    assert "messages" not in payload


def test_openai_gpt5_without_tools_uses_responses_api():
    client = _make_client("openai-thinking", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.2",
        params={"reasoning_effort": "high"},
        tools=None,
    )
    assert payload.get("_api_mode") == "responses"
    assert payload.get("reasoning") == {"effort": "high"}


def test_openai_non_reasoning_model_drops_reasoning_effort():
    client = _make_client("openai-thinking", "https://api.openai.com/v1", default_model="gpt-4o")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
        params={"reasoning_effort": "high"},
        tools=None,
    )
    assert "reasoning_effort" not in payload


def test_gemini_openai_compat_normalizes_minimal_reasoning_effort():
    client = _make_client("gemini-thinking", "https://generativelanguage.googleapis.com/v1beta/openai")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-2.5-flash",
        params={"reasoning_effort": "minimal"},
        tools=None,
    )
    assert payload.get("_api_mode") == "gemini_native"
    assert payload.get("config", {}).get("thinkingConfig") == {"thinkingLevel": "low"}
    assert payload.get("_fallback_payload", {}).get("reasoning_effort") == "low"


def test_openai_responses_payload_converts_tool_loop_messages():
    client = _make_client("openai-thinking", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
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
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true}"},
        ],
        model="gpt-5.4",
        params={"reasoning_effort": "high"},
        tools=[{"type": "function", "function": {"name": "search_local", "parameters": {}}}],
    )
    assert payload.get("_api_mode") == "responses"
    assert payload.get("instructions") == "sys"
    assert any(item.get("type") == "function_call" for item in payload.get("input", []))
    assert any(item.get("type") == "function_call_output" for item in payload.get("input", []))


def test_openai_structured_output_stays_on_chat_completions():
    client = _make_client("openai-thinking", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.4",
        params={"reasoning_effort": "high"},
        tools=None,
        response_model=object(),
    )
    assert payload.get("_api_mode") != "responses"
    assert "messages" in payload


def test_parse_tool_calls_from_openai_responses_output():
    raw = {
        "_api_mode": "responses",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"type": "function_call", "call_id": "call_1", "name": "search_web", "arguments": "{\"query\":\"abc\"}"},
        ],
    }
    assert has_tool_calls(raw, is_anthropic=False) is True
    calls = parse_tool_calls(raw, is_anthropic=False)
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "search_web"
    assert calls[0].arguments == {"query": "abc"}


def test_kimi_keeps_provider_specific_thinking_field():
    client = _make_client("kimi-thinking", "https://api.moonshot.ai/v1", default_model="kimi-k2.5")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="kimi-k2.5",
        params={"thinking": {"type": "enabled"}},
        tools=[{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
    )
    assert payload.get("thinking") == {"type": "enabled"}
    assert "tools" in payload
