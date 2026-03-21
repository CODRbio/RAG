import pytest
import requests
from unittest.mock import patch, MagicMock

from src.llm.llm_manager import GeminiNativeProvider, HTTPChatClient, ProviderConfig, _gemini_native_base_url


def _make_gemini_client() -> HTTPChatClient:
    cfg = ProviderConfig(
        name="gemini-thinking",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.1-pro-preview",
        platform="gemini",
        models={},
        params={},
    )
    return HTTPChatClient(cfg, provider=None)


def test_gemini_native_base_url_strips_openai_suffix():
    assert _gemini_native_base_url("https://generativelanguage.googleapis.com/v1beta/openai") == "https://generativelanguage.googleapis.com/v1beta"


def test_gemini_simple_text_uses_native_payload():
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        model="gemini-3.1-pro-preview",
        params={"reasoning_effort": "high"},
        tools=None,
    )
    assert payload.get("_api_mode") == "gemini_native"
    assert payload.get("_fallback_payload") is not None
    assert payload.get("systemInstruction", {}).get("parts")
    assert payload.get("contents")[0]["role"] == "user"
    assert payload.get("generationConfig", {}).get("thinkingConfig") == {"thinkingBudget": 24576}


def test_gemini_native_tools_and_tool_response_are_encoded():
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
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
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
        ],
        model="gemini-3.1-pro-preview",
        params={},
        tools=[{"type": "function", "function": {"name": "search_local", "parameters": {"type": "object"}}}],
    )
    assert payload.get("_api_mode") == "gemini_native"
    assert payload.get("tools")[0]["functionDeclarations"][0]["name"] == "search_local"
    parts = [part for item in payload.get("contents", []) for part in item.get("parts", [])]
    assert any("functionCall" in part for part in parts)
    assert any("functionResponse" in part for part in parts)


def test_gemini_native_cached_content_passthrough():
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="gemini-3.1-pro-preview",
        params={},
        tools=None,
        cache_policy=client._resolve_cache_policy(
            {"cache": {"cached_content": "cachedContents/test-cache"}},
            "gemini-3.1-pro-preview",
        ),
    )
    assert payload.get("cachedContent") == "cachedContents/test-cache"


def test_gemini_structured_output_falls_back_to_compat_payload():
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="gemini-3.1-pro-preview",
        params={"reasoning_effort": "high"},
        tools=None,
        response_model=object(),
    )
    assert payload.get("_api_mode") != "gemini_native"
    assert "messages" in payload


def test_gemini_native_response_adapts_to_openai_shape():
    raw = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "hello"},
                        {"functionCall": {"id": "call_1", "name": "search_web", "args": {"query": "abc"}}},
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
            "thoughtsTokenCount": 2,
        },
    }
    adapted = GeminiNativeProvider._adapt_native_response(raw)
    assert adapted["_api_mode"] == "gemini_native"
    assert adapted["choices"][0]["message"]["content"] == "hello"
    assert adapted["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search_web"
    assert adapted["usage"]["prompt_tokens"] == 10


def test_gemini_image_prompt_uses_generate_content_url():
    """Gemini native request for image-style prompt must hit :generateContent, not /chat/completions."""
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "帮我画一个可爱的加菲猫吃意大利面"}],
        model="gemini-3.1-flash-image-preview",
        params={},
        tools=None,
    )
    assert payload.get("_api_mode") == "gemini_native"
    captured = {}

    def capture_request(session, method, url, timeout, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }
        resp.raise_for_status = MagicMock()
        return resp

    cfg = ProviderConfig(
        name="gemini",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.1-flash-image-preview",
        platform="gemini",
        models={},
        params={},
    )
    provider = GeminiNativeProvider(cfg)
    with patch("src.llm.llm_manager._request_with_retry", side_effect=capture_request):
        provider.request(payload, timeout=60)

    assert "/models/gemini-3.1-flash-image-preview:generateContent" in captured["url"]
    assert "/chat/completions" not in captured["url"]


def test_gemini_native_404_raises_no_fallback():
    """When Gemini native returns 404, we must not fall back to OpenAI-compat /chat/completions."""
    client = _make_gemini_client()
    payload = client._build_openai_payload(
        messages=[{"role": "user", "content": "hello"}],
        model="gemini-3.1-pro-preview",
        params={},
        tools=None,
    )
    assert payload.get("_api_mode") == "gemini_native"
    assert payload.get("_fallback_payload") is not None

    err = requests.exceptions.HTTPError("404 Not Found")
    err.response = MagicMock()
    err.response.status_code = 404

    cfg = ProviderConfig(
        name="gemini",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.1-pro-preview",
        platform="gemini",
        models={},
        params={},
    )
    provider = GeminiNativeProvider(cfg)
    with patch("src.llm.llm_manager._request_with_retry", side_effect=err):
        with pytest.raises(RuntimeError) as exc_info:
            provider.request(payload, timeout=60)
    assert "404" in str(exc_info.value)
    assert "no fallback" in str(exc_info.value).lower() or "4xx" in str(exc_info.value)
