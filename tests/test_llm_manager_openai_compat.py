from unittest.mock import patch, MagicMock

from src.llm.llm_manager import HTTPChatClient, OpenAICompatProvider, ProviderConfig
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
    assert payload.get("generationConfig", {}).get("thinkingConfig") == {"thinkingLevel": "low"}
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


class _MockStreamProvider:
    def __init__(self, events, raw_response=None, stream_error=None):
        self._events = events
        self._raw_response = raw_response or {}
        self._stream_error = stream_error

    def request(self, payload, timeout=None):
        _ = payload, timeout
        return self._raw_response

    def request_stream(self, payload, timeout=None):
        _ = payload, timeout
        if self._stream_error:
            raise self._stream_error
        for item in self._events:
            yield item


def test_openai_stream_chat_normalizes_chat_completions_deltas():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        platform="openai",
        models={},
        params={},
    )
    provider = _MockStreamProvider([
        {"event": "message", "data": {"choices": [{"delta": {"content": "Hel"}}]}},
        {"event": "message", "data": {"choices": [{"delta": {"content": "lo"}}]}},
        {"event": "message", "data": {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}},
    ])
    client = HTTPChatClient(cfg, provider)

    events = list(client.stream_chat([{"role": "user", "content": "hi"}]))

    assert events[0] == {"type": "text_delta", "delta": "Hel"}
    assert events[1] == {"type": "text_delta", "delta": "lo"}
    assert events[-1]["type"] == "completed"
    assert events[-1]["response"]["final_text"] == "Hello"
    assert events[-1]["response"]["meta"]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2}


def test_openai_stream_chat_normalizes_responses_api_deltas():
    cfg = ProviderConfig(
        name="openai-thinking",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4",
        platform="openai",
        models={},
        params={},
    )
    provider = _MockStreamProvider([
        {"event": "response.output_text.delta", "data": {"delta": "A"}},
        {"event": "response.output_text.delta", "data": {"delta": "B"}},
        {"event": "response.completed", "data": {"response": {"output": [], "output_text": "AB", "usage": {"prompt_tokens": 3, "completion_tokens": 4}}}},
    ])
    client = HTTPChatClient(cfg, provider)

    events = list(client.stream_chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
        reasoning_effort="high",
    ))

    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["A", "B"]
    assert events[-1]["response"]["final_text"] == "AB"
    assert events[-1]["response"]["meta"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}


def test_stream_chat_falls_back_to_buffered_chunks_when_native_stream_unavailable():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        platform="openai",
        models={},
        params={},
    )
    provider = _MockStreamProvider(
        [],
        raw_response={"choices": [{"message": {"content": "fallback text"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        stream_error=NotImplementedError("no native stream"),
    )
    client = HTTPChatClient(cfg, provider)

    events = list(client.stream_chat([{"role": "user", "content": "hi"}]))

    assert "".join(e["delta"] for e in events if e["type"] == "text_delta") == "fallback text"
    assert events[-1]["response"]["final_text"] == "fallback text"


def test_openai_compat_provider_uses_expected_endpoint_by_api_mode():
    """OpenAICompatProvider must call /responses when _api_mode=responses and /chat/completions otherwise."""
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        platform="openai",
        models={},
        params={},
    )
    provider = OpenAICompatProvider(cfg)
    captured = {}

    def capture_request(session, method, url, timeout, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("src.llm.llm_manager._request_with_retry", side_effect=capture_request):
        provider.request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "_api_mode": "responses"},
            timeout=30,
        )
    assert captured["url"].rstrip("/").endswith("/responses")

    with patch("src.llm.llm_manager._request_with_retry", side_effect=capture_request):
        provider.request(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            timeout=30,
        )
    assert captured["url"].rstrip("/").endswith("/chat/completions")
