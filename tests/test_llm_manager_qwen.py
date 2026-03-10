from unittest.mock import patch

from src.llm.llm_manager import HTTPChatClient, OpenAICompatProvider, ProviderConfig, _qwen_responses_url


def _make_qwen_client() -> HTTPChatClient:
    cfg = ProviderConfig(
        name="qwen-thinking",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.5-plus",
        platform="qwen",
        models={},
        params={},
    )
    return HTTPChatClient(cfg, provider=None)


def test_qwen_responses_url_rewrite():
    assert _qwen_responses_url("https://dashscope.aliyuncs.com/compatible-mode/v1") == "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/responses"


def test_qwen_enable_thinking_uses_responses_api_and_budget():
    client = _make_qwen_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3.5-plus",
        params={"enable_thinking": True},
        tools=None,
    )
    assert payload.get("_api_mode") == "responses"
    assert payload.get("enable_thinking") is True
    assert payload.get("thinking_budget", 0) >= 25000


def test_qwen_tools_use_responses_api():
    client = _make_qwen_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3.5-plus",
        params={},
        tools=[{"type": "function", "function": {"name": "search_local", "parameters": {}}}],
    )
    assert payload.get("_api_mode") == "responses"
    assert "tools" in payload


def test_qwen_structured_output_stays_chat_completions():
    client = _make_qwen_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3.5-plus",
        params={"enable_thinking": True},
        tools=None,
        response_model=object(),
    )
    assert payload.get("_api_mode") != "responses"
    assert "messages" in payload


def test_qwen_provider_uses_official_responses_endpoint():
    cfg = ProviderConfig(
        name="qwen-thinking",
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.5-plus",
        platform="qwen",
        models={},
        params={},
    )
    provider = OpenAICompatProvider(cfg)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"output": [], "output_text": ""}

    with patch("src.llm.llm_manager._request_with_retry", return_value=_Resp()) as mocked:
        provider.request({"_api_mode": "responses", "model": "qwen3.5-plus", "input": []})
        called_url = mocked.call_args.args[2]
        assert called_url == "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/responses"
