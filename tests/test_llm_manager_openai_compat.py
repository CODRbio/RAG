from unittest.mock import patch, MagicMock

from config.settings import settings
from src.llm import llm_manager as llm_manager_mod
from src.llm.llm_manager import HTTPChatClient, OpenAICompatProvider, ProviderConfig
from src.llm.tools import has_tool_calls, parse_tool_calls
from src.utils.cache import TTLCache


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
    assert payload.get("generationConfig", {}).get("thinkingConfig") == {"thinkingBudget": 512}
    assert payload.get("_fallback_payload", {}).get("reasoning_effort") == "low"


def test_openai_prompt_cache_fields_are_injected():
    client = _make_client("openai", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4",
        params={},
        cache_policy=client._resolve_cache_policy(
            {"cache": {"scope": "global_template", "key": "query_family", "retention": "24h"}},
            "gpt-5.4",
        ),
    )
    assert payload["prompt_cache_key"] == "query_family"
    assert payload["prompt_cache_retention"] == "24h"


def test_openai_does_not_inject_prompt_cache_fields_without_explicit_cache_config():
    client = _make_client("openai", "https://api.openai.com/v1")
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.4",
        params={},
        cache_policy=client._resolve_cache_policy({}, "gpt-5.4"),
    )
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


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


class _MockRequestProvider:
    def __init__(self):
        self.calls = 0

    def request(self, payload, timeout=None):
        _ = payload, timeout
        self.calls += 1
        return {
            "choices": [{"message": {"content": "cached answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    def request_stream(self, payload, timeout=None):
        _ = payload, timeout
        raise NotImplementedError


class _CaptureRequestProvider:
    def __init__(self, raw_response=None):
        self.calls = 0
        self.last_payload = None
        self._raw_response = raw_response or {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    def request(self, payload, timeout=None):
        _ = timeout
        self.calls += 1
        self.last_payload = payload
        return self._raw_response

    def request_stream(self, payload, timeout=None):
        _ = payload, timeout
        raise NotImplementedError


class _MetricRecorder:
    def __init__(self):
        self.events = []

    def labels(self, **labels):
        recorder = self

        class _Handle:
            def inc(self, value=1):
                recorder.events.append(("inc", labels, value))

            def observe(self, value):
                recorder.events.append(("observe", labels, value))

        return _Handle()


class _FakeMetrics:
    def __init__(self):
        self.llm_requests_total = _MetricRecorder()
        self.llm_duration_seconds = _MetricRecorder()
        self.llm_errors_total = _MetricRecorder()
        self.llm_tokens_used = _MetricRecorder()
        self.llm_cached_tokens_total = _MetricRecorder()
        self.llm_cache_events_total = _MetricRecorder()


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
    usage = events[-1]["response"]["meta"]["usage"]
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 2
    assert usage["cached_input_tokens"] == 0
    assert usage["cache_hit"] is False


def test_http_chat_client_app_cache_reuses_buffered_response():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        platform="openai",
        models={},
        params={},
    )
    provider = _MockRequestProvider()
    client = HTTPChatClient(cfg, provider, app_cache=TTLCache(maxsize=8, ttl_seconds=60))

    with patch.object(settings.perf_llm, "cache_enabled", True):
        first = client.chat(
            [{"role": "user", "content": "hi"}],
            cache={"mode": "provider_plus_app", "scope": "global_template"},
        )
        second = client.chat(
            [{"role": "user", "content": "hi"}],
            cache={"mode": "provider_plus_app", "scope": "global_template"},
        )

    assert provider.calls == 1
    assert first["final_text"] == "cached answer"
    assert second["meta"]["cache"]["source"] == "app"
    assert second["meta"]["cache"]["hit"] is True


def test_http_chat_client_omits_provider_cache_without_explicit_config():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4",
        platform="openai",
        models={},
        params={},
    )
    provider = _CaptureRequestProvider()
    client = HTTPChatClient(cfg, provider)

    result = client.chat([{"role": "user", "content": "hi"}])

    assert provider.calls == 1
    assert "prompt_cache_key" not in provider.last_payload
    assert "prompt_cache_retention" not in provider.last_payload
    assert result["meta"]["cache"]["provider_enabled"] is False
    assert result["meta"]["cache"]["mode"] == "off"


def test_http_chat_client_strips_internal_cache_keys_from_provider_payload():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4",
        platform="openai",
        models={},
        params={},
    )
    provider = _CaptureRequestProvider()
    client = HTTPChatClient(cfg, provider)

    client.chat(
        [{"role": "user", "content": "hi"}],
        cache={
            "mode": "provider_only",
            "scope": "global_template",
            "key": "stable-key",
            "retention": "24h",
        },
        cache_mode="provider_only",
        cache_scope="section",
        enable_prompt_cache=True,
        gemini_cached_content="cachedContents/abc",
    )

    assert provider.last_payload["prompt_cache_key"] == "stable-key"
    assert provider.last_payload["prompt_cache_retention"] == "24h"
    assert "cache" not in provider.last_payload
    assert "cache_mode" not in provider.last_payload
    assert "cache_scope" not in provider.last_payload
    assert "enable_prompt_cache" not in provider.last_payload
    assert "gemini_cached_content" not in provider.last_payload


def test_http_chat_client_app_cache_respects_request_ttl_override():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        platform="openai",
        models={},
        params={},
    )
    provider = _MockRequestProvider()
    client = HTTPChatClient(cfg, provider, app_cache=TTLCache(maxsize=8, ttl_seconds=60))
    messages = [{"role": "user", "content": "hi"}]
    cache = {"mode": "provider_plus_app", "scope": "global_template", "ttl_seconds": 1}

    with patch.object(settings.perf_llm, "cache_enabled", True):
        client.chat(messages, cache=cache)
        cache_key = client._build_app_cache_key(
            resolved_model="gpt-4o-mini",
            messages=messages,
            params={},
            tools=None,
            response_model=None,
            is_anthropic=False,
        )
        client._app_cache.set(
            cache_key,
            {
                "expires_at": 0.0,
                "result": client._app_cache.get(cache_key)["result"],
            },
        )
        client.chat(messages, cache=cache)

    assert provider.calls == 2


def test_http_chat_client_records_provider_cache_unknown_without_usage():
    cfg = ProviderConfig(
        name="openai",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.4",
        platform="openai",
        models={},
        params={},
    )
    provider = _CaptureRequestProvider(
        raw_response={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    )
    client = HTTPChatClient(cfg, provider)
    fake_metrics = _FakeMetrics()

    with patch.object(llm_manager_mod, "_obs_metrics", fake_metrics):
        client.chat(
            [{"role": "user", "content": "hi"}],
            cache={"mode": "provider_only", "scope": "global_template", "key": "stable-key"},
        )

    cache_events = fake_metrics.llm_cache_events_total.events
    assert ("inc", {"provider": "openai", "model": "gpt-5.4", "source": "provider", "result": "unknown"}, 1) in cache_events


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
    usage = events[-1]["response"]["meta"]["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 4
    assert usage["cached_input_tokens"] == 0
    assert usage["cache_hit"] is False


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
