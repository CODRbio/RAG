"""Tests for Graphic Abstract image generation dispatch and Gemini native URL normalization."""
import base64
from unittest.mock import patch, MagicMock

import pytest

from src.llm.llm_manager import PlatformConfig


def test_generate_ga_image_gemini_strips_openai_suffix_and_calls_generate_content():
    """_generate_ga_image_gemini must strip /openai from base_url and request :generateContent."""
    from src.api.routes_chat import _generate_ga_image_gemini

    captured = {}

    def capture_post(url, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"fake_png_bytes").decode()}}
                        ]
                    }
                }
            ]
        }
        return resp

    with patch("src.api.routes_chat._http.post", side_effect=capture_post):
        out = _generate_ga_image_gemini(
            "gemini-3.1-flash-image-preview",
            "draw a cat",
            "test-key",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
    assert out == b"fake_png_bytes"
    assert "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent" == captured["url"]
    assert "/openai" not in captured["url"] or captured["url"].endswith(":generateContent")


def test_generate_ga_image_dispatches_gemini_to_native():
    """_generate_ga_image with provider gemini must call _generate_ga_image_gemini, not openai_compat."""
    from src.api.routes_chat import _generate_ga_image

    fake_platform = PlatformConfig(
        name="gemini",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    mock_manager = MagicMock()
    mock_manager.config.platforms.get.return_value = fake_platform

    gemini_called = []
    openai_called = []

    def track_gemini(model, prompt, api_key, base_url):
        gemini_called.append((model, prompt, api_key, base_url))
        return b"png"

    def track_openai(model, prompt, api_key, base_url):
        openai_called.append((model, prompt, api_key, base_url))
        return b"png"

    with patch("src.api.routes_chat.get_manager", return_value=mock_manager):
        with patch("src.api.routes_chat._generate_ga_image_gemini", side_effect=track_gemini):
            with patch("src.api.routes_chat._generate_ga_image_openai_compat", side_effect=track_openai):
                _generate_ga_image("gemini", "gemini-3.1-flash-image-preview", "draw a cat")

    assert len(gemini_called) == 1
    assert gemini_called[0][0] == "gemini-3.1-flash-image-preview"
    assert gemini_called[0][1] == "draw a cat"
    assert len(openai_called) == 0


def test_generate_ga_image_dispatches_openai_to_compat():
    """_generate_ga_image with provider openai must call _generate_ga_image_openai_compat."""
    from src.api.routes_chat import _generate_ga_image

    fake_platform = PlatformConfig(
        name="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
    )
    mock_manager = MagicMock()
    mock_manager.config.platforms.get.return_value = fake_platform

    openai_called = []
    gemini_called = []

    def track_openai(model, prompt, api_key, base_url):
        openai_called.append((model, prompt))
        return b"png"

    def track_gemini(*args, **kwargs):
        gemini_called.append(1)
        return b"png"

    with patch("src.api.routes_chat.get_manager", return_value=mock_manager):
        with patch("src.api.routes_chat._generate_ga_image_openai_compat", side_effect=track_openai):
            with patch("src.api.routes_chat._generate_ga_image_gemini", side_effect=track_gemini):
                _generate_ga_image("openai", "gpt-image-1.5", "draw a cat")

    assert len(openai_called) == 1
    assert openai_called[0][0] == "gpt-image-1.5"
    assert len(gemini_called) == 0


def test_generate_ga_image_raises_when_platform_missing():
    """_generate_ga_image must raise when provider is not in config platforms."""
    from src.api.routes_chat import _generate_ga_image

    mock_manager = MagicMock()
    mock_manager.config.platforms.get.return_value = None

    with patch("src.api.routes_chat.get_manager", return_value=mock_manager):
        with pytest.raises(ValueError) as exc_info:
            _generate_ga_image("nonexistent", "model", "prompt")
    assert "not found" in str(exc_info.value).lower()
