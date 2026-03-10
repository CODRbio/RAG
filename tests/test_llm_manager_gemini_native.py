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
    assert payload.get("generationConfig", {}).get("thinkingConfig") == {"thinkingLevel": "high"}


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
