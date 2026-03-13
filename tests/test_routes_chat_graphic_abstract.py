"""Tests for Graphic Abstract model resolution and llm_manager image generation."""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm.llm_manager import LLMConfig, LLMManager, PlatformConfig, ProviderConfig


def _make_manager() -> LLMManager:
    return LLMManager(
        LLMConfig(
            default="openai",
            dry_run=False,
            platforms={
                "gemini": PlatformConfig(
                    name="gemini",
                    api_key="test-gemini",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                ),
                "openai": PlatformConfig(
                    name="openai",
                    api_key="test-openai",
                    base_url="https://api.openai.com/v1",
                ),
                "qwen": PlatformConfig(
                    name="qwen",
                    api_key="test-qwen",
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                "kimi": PlatformConfig(
                    name="kimi",
                    api_key="test-kimi",
                    base_url="https://api.moonshot.ai/v1",
                ),
            },
            providers={
                "openai": ProviderConfig(
                    name="openai",
                    api_key="test-openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-image-1.5",
                    platform="openai",
                )
            },
        )
    )


def test_resolve_graphic_abstract_model_maps_provider_and_legacy_aliases():
    from src.collaboration.graphic_abstract import resolve_graphic_abstract_model

    assert resolve_graphic_abstract_model("gemini") == ("gemini", "gemini-2.5-flash-image")
    assert resolve_graphic_abstract_model("nanobanana 2") == ("gemini", "gemini-2.5-flash-image")
    assert resolve_graphic_abstract_model("kimi-k2.5") == ("gemini", "gemini-2.5-flash-image")
    assert resolve_graphic_abstract_model("gemini-3.1-flash-image-preview") == (
        "gemini",
        "gemini-2.5-flash-image",
    )


def test_generate_image_gemini_strips_openai_suffix_and_calls_generate_content():
    manager = _make_manager()
    captured = {}

    def capture_request(method, url, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"fake_png_bytes").decode(),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        return resp

    with patch("src.llm.llm_manager.requests.Session.request", side_effect=capture_request):
        out = manager.generate_image(
            provider="gemini",
            model="gemini-2.5-flash-image",
            prompt="draw a cat",
        )

    assert out == b"fake_png_bytes"
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash-image:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "test-gemini"
    assert captured["headers"]["Content-Type"] == "application/json"


def test_generate_image_openai_uses_images_endpoint_and_decodes_b64():
    manager = _make_manager()
    captured = {}

    def capture_request(method, url, timeout=None, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json") or {}
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"openai_png").decode()}]
        }
        return resp

    with patch("src.llm.llm_manager.requests.Session.request", side_effect=capture_request):
        out = manager.generate_image(
            provider="openai",
            model="gpt-image-1.5",
            prompt="draw a cat",
        )

    assert out == b"openai_png"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.openai.com/v1/images/generations"
    assert captured["json"] == {
        "model": "gpt-image-1.5",
        "prompt": "draw a cat",
        "size": "1024x1024",
    }


def test_generate_image_qwen_uses_dashscope_endpoint_and_downloads_image():
    manager = _make_manager()
    calls = []

    def capture_request(method, url, timeout=None, **kwargs):
        calls.append((method, url, kwargs))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if method == "POST":
            resp.json.return_value = {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.com/generated-qwen.png"}
                                ]
                            }
                        }
                    ]
                }
            }
        else:
            resp.content = b"qwen_png"
        return resp

    with patch("src.llm.llm_manager.requests.Session.request", side_effect=capture_request):
        out = manager.generate_image(
            provider="qwen",
            model="qwen-image-2.0",
            prompt="draw a cat",
        )

    assert out == b"qwen_png"
    assert calls[0][0] == "POST"
    assert calls[0][1] == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    assert calls[0][2]["json"]["parameters"]["size"] == "1024*1024"
    assert calls[1][0] == "GET"
    assert calls[1][1] == "https://example.com/generated-qwen.png"


def test_generate_image_kimi_is_explicitly_unsupported():
    manager = _make_manager()

    with pytest.raises(NotImplementedError) as exc_info:
        manager.generate_image(provider="kimi", model="kimi-k2.5", prompt="draw a cat")

    assert "not supported" in str(exc_info.value).lower()


def test_generate_graphic_abstract_image_delegates_to_manager():
    from src.collaboration.graphic_abstract import _generate_graphic_abstract_image

    mock_manager = MagicMock()
    mock_manager.generate_image.return_value = b"png"

    with patch("src.collaboration.graphic_abstract.get_manager", return_value=mock_manager):
        out = _generate_graphic_abstract_image("openai", "gpt-image-1.5", "draw a cat")

    assert out == b"png"
    mock_manager.generate_image.assert_called_once_with(
        provider="openai",
        model="gpt-image-1.5",
        prompt="draw a cat",
    )


def test_detect_image_extension_matches_common_magic_bytes():
    from src.collaboration.graphic_abstract import detect_image_extension

    assert detect_image_extension(b"\x89PNG\r\n\x1a\nrest") == ".png"
    assert detect_image_extension(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert detect_image_extension(b"GIF89arest") == ".gif"
    assert detect_image_extension(b"RIFFxxxxWEBPrest") == ".webp"
    assert detect_image_extension(b"unknown") == ".png"


def test_render_graphic_abstract_markdown_uses_detected_extension(tmp_path):
    from src.collaboration import graphic_abstract as ga
    from src.services.media_store import MediaAsset

    png_like_jpeg = b"\xff\xd8\xff\xe0jpeg-bytes"
    mock_store = MagicMock()
    mock_store.store_bytes.return_value = MediaAsset(
        url="https://cdn.example.com/graphic-abstract/sess-1/turn_3.jpg",
        key="graphic-abstract/sess-1/turn_3.jpg",
        backend="s3",
        content_type="image/jpeg",
        size_bytes=len(png_like_jpeg),
    )

    with patch.object(ga, "_generate_graphic_abstract_image", return_value=png_like_jpeg):
        with patch.object(ga, "get_media_store", return_value=mock_store):
            result = ga.render_graphic_abstract_markdown(
                "summary",
                model_raw="nanobanana pro",
                content_kind="chat",
                heading="### Graphic Abstract",
                session_id="sess-1",
                turn_id="3",
            )

    mock_store.store_bytes.assert_called_once_with(
        category="graphic-abstract",
        content=png_like_jpeg,
        content_type="image/jpeg",
        file_ext=".jpg",
        session_id="sess-1",
        logical_name="turn_3",
    )
    assert result.image_url == "https://cdn.example.com/graphic-abstract/sess-1/turn_3.jpg"
    assert result.asset_key == "graphic-abstract/sess-1/turn_3.jpg"
    assert result.content_type == "image/jpeg"
    assert result.storage_backend == "s3"
    assert "![Graphic Abstract](https://cdn.example.com/graphic-abstract/sess-1/turn_3.jpg)" in result.markdown


def test_local_media_store_returns_media_url_and_writes_file(tmp_path):
    from src.services import media_store as ms

    previous_store = ms._store
    try:
        with patch.object(ms.settings.storage.media, "backend", "local"):
            with patch.object(ms.settings.storage.media, "public_base_url", ""):
                with patch.object(ms.settings.storage.media.local, "root", str(tmp_path)):
                    ms._store = None
                    asset = ms.get_media_store().store_bytes(
                        category="graphic-abstract",
                        content=b"\x89PNG\r\n\x1a\ncontent",
                        content_type="image/png",
                        file_ext=".png",
                        session_id="sess-1",
                        logical_name="turn_3",
                    )
    finally:
        ms._store = previous_store

    assert asset.key == "graphic-abstract/sess-1/turn_3.png"
    assert asset.url == "/media/graphic-abstract/sess-1/turn_3.png"
    assert asset.backend == "local"
    assert asset.content_type == "image/png"
    assert asset.size_bytes == len(b"\x89PNG\r\n\x1a\ncontent")
    assert (tmp_path / "graphic-abstract" / "sess-1" / "turn_3.png").read_bytes() == b"\x89PNG\r\n\x1a\ncontent"


def test_s3_media_store_uploads_with_content_type_and_public_url():
    from src.services import media_store as ms

    fake_client = MagicMock()
    captured = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session_kwargs"] = kwargs

        def client(self, service_name, endpoint_url=None, region_name=None):
            captured["client_args"] = {
                "service_name": service_name,
                "endpoint_url": endpoint_url,
                "region_name": region_name,
            }
            return fake_client

    previous_store = ms._store
    try:
        with patch.object(ms, "_load_boto3_session_cls", return_value=FakeSession):
            with patch.object(ms.settings.storage.media, "backend", "s3"):
                with patch.object(ms.settings.storage.media, "public_base_url", ""):
                    with patch.object(ms.settings.storage.media.s3, "bucket", "ga-bucket"):
                        with patch.object(ms.settings.storage.media.s3, "region", "us-east-1"):
                            with patch.object(ms.settings.storage.media.s3, "endpoint", "https://s3.example.com"):
                                with patch.object(ms.settings.storage.media.s3, "key_prefix", "prod"):
                                    with patch.object(ms.settings.storage.media.s3, "public_base_url", "https://cdn.example.com/assets"):
                                        with patch.dict(
                                            os.environ,
                                            {
                                                "MEDIA_S3_ACCESS_KEY_ID": "test-access",
                                                "MEDIA_S3_SECRET_ACCESS_KEY": "test-secret",
                                            },
                                            clear=False,
                                        ):
                                            ms._store = None
                                            asset = ms.get_media_store().store_bytes(
                                                category="graphic-abstract",
                                                content=b"png-bytes",
                                                content_type="image/png",
                                                file_ext=".png",
                                                logical_name="turn_3",
                                            )
    finally:
        ms._store = previous_store

    assert captured["session_kwargs"]["aws_access_key_id"] == "test-access"
    assert captured["session_kwargs"]["aws_secret_access_key"] == "test-secret"
    assert captured["client_args"] == {
        "service_name": "s3",
        "endpoint_url": "https://s3.example.com",
        "region_name": "us-east-1",
    }
    fake_client.put_object.assert_called_once_with(
        Bucket="ga-bucket",
        Key="prod/graphic-abstract/anon/turn_3.png",
        Body=b"png-bytes",
        ContentType="image/png",
    )
    assert asset.key == "prod/graphic-abstract/anon/turn_3.png"
    assert asset.url == "https://cdn.example.com/assets/prod/graphic-abstract/anon/turn_3.png"
    assert asset.backend == "s3"
