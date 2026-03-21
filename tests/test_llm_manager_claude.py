from src.llm.llm_manager import HTTPChatClient, ProviderConfig


def _make_claude_client(default_model: str = "claude-sonnet-4-6") -> HTTPChatClient:
    cfg = ProviderConfig(
        name="claude-thinking",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        default_model=default_model,
        platform="claude",
        models={},
        params={},
    )
    return HTTPChatClient(cfg, provider=None)


def test_claude_opus_46_upgrades_manual_thinking_to_adaptive():
    client = _make_claude_client(default_model="claude-opus-4-6")
    payload = client._build_anthropic_payload(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        model="claude-opus-4-6",
        params={"thinking": {"type": "enabled", "budget_tokens": 20000}},
        tools=None,
    )
    assert payload["thinking"] == {"type": "adaptive", "effort": "high"}
    assert payload["max_tokens"] == 128000


def test_claude_sonnet_46_keeps_manual_thinking_and_clamps_budget():
    client = _make_claude_client(default_model="claude-sonnet-4-6")
    payload = client._build_anthropic_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        params={"thinking": {"type": "enabled", "budget_tokens": 70000}, "max_tokens": 64000},
        tools=None,
    )
    assert payload["thinking"]["type"] == "enabled"
    assert payload["thinking"]["budget_tokens"] == 56000
    assert payload["max_tokens"] == 64000


def test_claude_adaptive_thinking_gets_default_effort():
    client = _make_claude_client(default_model="claude-sonnet-4-6")
    payload = client._build_anthropic_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        params={"thinking": {"type": "adaptive"}},
        tools=None,
    )
    assert payload["thinking"] == {"type": "adaptive", "effort": "medium"}
    assert payload["max_tokens"] == 64000


def test_claude_auto_prompt_cache_sets_top_level_cache_control():
    client = _make_claude_client(default_model="claude-sonnet-4-6")
    policy = client._resolve_cache_policy(
        {"cache": {"mode": "provider_only", "strategy": "auto"}},
        "claude-sonnet-4-6",
    )
    payload = client._build_anthropic_payload(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        params={},
        tools=None,
        cache_policy=policy,
    )
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert payload["system"] == "sys"


def test_claude_explicit_prompt_cache_marks_system_and_first_user():
    client = _make_claude_client(default_model="claude-sonnet-4-6")
    policy = client._resolve_cache_policy(
        {"cache": {"mode": "provider_only", "strategy": "explicit", "scope": "section"}},
        "claude-sonnet-4-6",
    )
    payload = client._build_anthropic_payload(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        params={},
        tools=None,
        cache_policy=policy,
    )
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_claude_enable_prompt_cache_alias_keeps_backward_compatible_behavior():
    client = _make_claude_client(default_model="claude-sonnet-4-6")
    policy = client._resolve_cache_policy({"enable_prompt_cache": True}, "claude-sonnet-4-6")
    payload = client._build_anthropic_payload(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
        model="claude-sonnet-4-6",
        params={},
        tools=None,
        cache_policy=policy,
    )
    assert policy.provider_enabled is True
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


class _MockAnthropicStreamProvider:
    def request(self, payload, timeout=None):
        _ = payload, timeout
        return {}

    def request_stream(self, payload, timeout=None):
        _ = payload, timeout
        yield {"event": "message_start", "data": {"message": {"usage": {"input_tokens": 2}}}}
        yield {"event": "content_block_delta", "data": {"delta": {"type": "text_delta", "text": "Hello"}}}
        yield {"event": "content_block_delta", "data": {"delta": {"type": "text_delta", "text": " Claude"}}}
        yield {"event": "message_delta", "data": {"usage": {"input_tokens": 2, "output_tokens": 3}}}
        yield {"event": "message_stop", "data": {}}


def test_claude_stream_chat_normalizes_text_deltas():
    client = HTTPChatClient(_make_claude_client().config, _MockAnthropicStreamProvider())

    events = list(client.stream_chat([{"role": "user", "content": "hello"}]))

    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["Hello", " Claude"]
    assert events[-1]["response"]["final_text"] == "Hello Claude"
