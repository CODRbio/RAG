"""Regression tests for Codex app-server provider (stream contract, model list, thread id)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.api import routes_models
from src.llm import llm_manager as llm_manager_mod
from src.api.routes_chat import (
    _codex_thread_id_from_session_meta,
    _is_codex_app_server_provider,
    _persist_codex_thread_from_response,
)
from src.llm.codex_app_server import (
    CodexAppServerChatClient,
    CodexRpcProcess,
    _summarize_codex_transport_stderr,
)
from src.llm.llm_manager import LLMConfig, LLMManager, PlatformConfig, ProviderConfig
from src.llm.model_registry import ModelInfo, ModelRegistry
from src.llm.react_loop import react_loop
from src.llm.tools import ToolCall, ToolDef


def test_codex_stream_chat_emits_delta_and_completed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield {"method": "item/agentMessage/delta", "params": {"delta": {"text": "Hi"}}}
            yield {"method": "turn/completed", "params": {"turn": {"status": "completed"}}}

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    events = list(
        client.stream_chat([{"role": "user", "content": "hello"}], model="gpt-5.4")
    )
    deltas = [e for e in events if e.get("type") == "text_delta"]
    assert deltas
    assert all(e.get("delta") for e in deltas)

    completed = [e for e in events if e.get("type") == "completed"]
    assert len(completed) == 1
    resp = completed[0].get("response")
    assert isinstance(resp, dict)
    assert resp.get("final_text") == "Hi"
    assert (resp.get("meta") or {}).get("codex_thread_id") == "thr_test"


def test_codex_chat_accepts_string_shaped_event_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield {"method": "item/agentMessage/delta", "params": {"delta": "Hi "}}
            yield {"method": "item/completed", "params": {"item": "Hi from item"}}
            yield {"method": "turn/completed", "params": "completed"}

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    resp = client.chat([{"role": "user", "content": "hello"}], model="gpt-5.4")
    assert resp["final_text"] == "Hi from item"


def test_codex_chat_ignores_completed_user_message_items(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield {
                "method": "item/completed",
                "params": {"item": {"id": "u1", "type": "userMessage", "text": "echoed user input"}},
            }
            yield {"method": "item/agentMessage/delta", "params": {"delta": {"text": "OK"}}}
            yield {"method": "turn/completed", "params": {"turn": {"status": "completed"}}}

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    resp = client.chat([{"role": "user", "content": "hello"}], model="gpt-5.4")
    assert resp["final_text"] == "OK"


def test_codex_chat_ignores_non_object_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield "keepalive"
            yield {"method": "item/agentMessage/delta", "params": {"delta": "Hi"}}
            yield {"method": "turn/completed", "params": {"turn": {"status": "completed"}}}

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    resp = client.chat([{"role": "user", "content": "hello"}], model="gpt-5.4")
    assert resp["final_text"] == "Hi"


def test_codex_transport_stderr_summary_for_connection_reset() -> None:
    summary = _summarize_codex_transport_stderr(
        "2026-03-20T13:57:23Z ERROR codex_api::endpoint::responses_websocket: "
        "failed to connect to websocket: IO error: Connection reset by peer (os error 54), "
        "url: wss://chatgpt.com/backend-api/codex/responses"
    )
    assert summary is not None
    assert "network, proxy, VPN, firewall, or TLS interception" in summary
    assert "wss://chatgpt.com/backend-api/codex/responses" in summary


def test_codex_chat_raises_transport_summary_when_turn_never_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            if False:
                yield None

        def transport_error_summary(self):
            return "Codex transport failed while opening the websocket."

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    with pytest.raises(RuntimeError, match="Codex transport failed while opening the websocket"):
        client.chat([{"role": "user", "content": "hello"}], model="gpt-5.4")


def test_codex_chat_raises_when_failed_turn_only_contains_user_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield {
                "method": "item/completed",
                "params": {
                    "item": {
                        "id": "u1",
                        "type": "userMessage",
                        "text": "[System]: 你是一个简洁助手。\n\n[User]: 请回复 OK",
                    }
                },
            }
            yield {
                "method": "turn/completed",
                "params": {
                    "turn": {
                        "status": "failed",
                        "error": {
                            "message": (
                                "stream disconnected before completion: "
                                "error sending request for url "
                                "(https://chatgpt.com/backend-api/codex/responses)"
                            )
                        },
                    }
                },
            }

        def transport_error_summary(self):
            return "Codex transport failed while opening the websocket."

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    with pytest.raises(RuntimeError, match="Codex transport failed while opening the websocket"):
        client.chat([{"role": "user", "content": "hello"}], model="gpt-5.4")


def test_codex_rpc_request_and_notify_always_include_empty_params() -> None:
    rpc = CodexRpcProcess(cli_path="codex", timeout=5)
    captured: list[dict] = []

    def _fake_write(payload):
        captured.append(dict(payload))
        if "id" in payload:
            rpc._pending[payload["id"]].put({"result": {"ok": True}})

    rpc._write = _fake_write  # type: ignore[method-assign]

    assert rpc.request("model/list") == {"ok": True}
    rpc.notify("initialized")

    assert captured[0]["method"] == "model/list"
    assert captured[0]["params"] == {}
    assert captured[1]["method"] == "initialized"
    assert captured[1]["params"] == {}


def test_codex_does_not_inherit_openai_legacy_env_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg_path = tmp_path / "rag_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "llm": {
                    "default": "codex",
                    "dry_run": False,
                    "platforms": {
                        "openai": {"api_key": "", "base_url": "https://api.openai.com/v1"},
                        "codex_app_server": {"api_key": "", "base_url": ""},
                    },
                    "providers": {
                        "openai": {"platform": "openai", "default_model": "gpt-5.4"},
                        "codex": {"platform": "codex_app_server", "default_model": "gpt-5.4"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-env")

    manager = LLMManager.from_json(cfg_path)

    assert manager.config.platforms["openai"].api_key == "sk-test-openai-env"
    assert manager.config.providers["openai"].api_key == "sk-test-openai-env"
    assert manager.config.platforms["codex_app_server"].api_key == ""
    assert manager.config.providers["codex"].api_key == ""


def test_fetch_provider_models_codex_allows_empty_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class _PCfg:
        api_key: str
        base_url: str = ""

    @dataclass
    class _FakeLLMConfig:
        providers: dict
        platforms: dict

    class _FakeManager:
        config = _FakeLLMConfig(
            providers={"codex": _PCfg(api_key="")},
            platforms={"codex_app_server": _PCfg(api_key="")},
        )

    captured: dict = {}

    def _fake_get_manager(_path: str):
        return _FakeManager()

    def _fake_fetch_codex_live_models(api_key: str = "", timeout: float = 20.0):
        captured["provider_name"] = "codex"
        captured["api_key"] = api_key
        return [ModelInfo(id="gpt-5.4", owned_by="openai")]

    app = FastAPI()
    app.include_router(routes_models.router)
    tclient = TestClient(app)

    monkeypatch.setattr(llm_manager_mod, "get_manager", _fake_get_manager)
    monkeypatch.setattr(routes_models, "_fetch_codex_live_models", _fake_fetch_codex_live_models)

    r = tclient.get("/llm/providers/codex/models")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["models"][0]["id"] == "gpt-5.4"
    assert captured["provider_name"] == "codex"
    assert captured["api_key"] == ""

    @dataclass
    class _EmptyCfg:
        providers: dict
        platforms: dict

    class _NoKeyManager:
        config = _EmptyCfg(
            providers={"openai": _PCfg(api_key="")},
            platforms={"openai": _PCfg(api_key="")},
        )

    monkeypatch.setattr(llm_manager_mod, "get_manager", lambda _p: _NoKeyManager())
    r2 = tclient.get("/llm/providers/openai/models")
    assert r2.status_code == 400


def test_react_loop_updates_codex_thread_kwargs_between_iterations() -> None:
    calls: list[dict] = []

    class _FC:
        config = type("C", (), {"is_anthropic": lambda self: False})()

        def chat(self, messages, model=None, tools=None, **kwargs):
            calls.append(dict(kwargs))
            n = len(calls)
            if n == 1:
                return {
                    "final_text": "",
                    "tool_calls": [ToolCall(id="c1", name="test_noop", arguments={})],
                    "raw": {},
                    "meta": {"codex_thread_id": "tid-step1"},
                }
            return {
                "final_text": "ok",
                "tool_calls": [],
                "raw": {},
                "meta": {"codex_thread_id": f"tid-step{n}"},
            }

    tool = ToolDef(
        name="test_noop",
        description="test",
        parameters={"type": "object", "properties": {}},
        handler=lambda **kwargs: "tool-ok",
    )

    react_loop(
        messages=[{"role": "user", "content": "hi"}],
        tools=[tool],
        llm_client=_FC(),
        max_iterations=3,
        model=None,
        session_id="s1",
    )

    assert len(calls) == 2
    assert calls[0].get("codex_thread_id") is None
    assert calls[1].get("codex_thread_id") == "tid-step1"


def test_react_loop_rejects_provider_without_platform_tool_support() -> None:
    class _NoToolClient:
        supports_platform_tool_calls = False
        config = type("C", (), {"name": "codex", "is_anthropic": lambda self: False})()

        def chat(self, *args, **kwargs):
            raise AssertionError("react_loop should reject unsupported providers before calling chat()")

    tool = ToolDef(
        name="test_noop",
        description="test",
        parameters={"type": "object", "properties": {}},
        handler=lambda **kwargs: "tool-ok",
    )

    with pytest.raises(ValueError, match="does not support the platform ReAct tool loop"):
        react_loop(
            messages=[{"role": "user", "content": "hi"}],
            tools=[tool],
            llm_client=_NoToolClient(),
            max_iterations=2,
            model=None,
            session_id="s1",
        )


def test_codex_chat_parses_response_model_and_injects_output_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StructuredResponse(BaseModel):
        answer: str
        score: int

    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)
    captured: dict = {}

    class FakeRpc:
        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

        def request(self, method, params=None, timeout=None):
            if method == "thread/start":
                return {"thread": {"id": "thr_test"}}
            if method == "turn/start":
                captured["turn_params"] = dict(params or {})
                return {"turn": {"id": "turn1"}}
            return {}

        def notify(self, method, params=None) -> None:
            pass

        def iter_notifications(self, timeout=None):
            yield {
                "method": "item/agentMessage/delta",
                "params": {"delta": {"text": '{"answer":"ok","score":2}'}},
            }
            yield {"method": "turn/completed", "params": {"turn": {"status": "completed"}}}

    monkeypatch.setattr(client, "_spawn_rpc", lambda: FakeRpc())
    monkeypatch.setattr(client, "_handshake", lambda rpc: None)
    monkeypatch.setattr(client, "_authenticate", lambda rpc: None)
    monkeypatch.setattr(client, "_ensure_thread", lambda rpc, tid: tid or "thr_test")

    resp = client.chat(
        [{"role": "user", "content": "hello"}],
        model="gpt-5.4",
        response_model=_StructuredResponse,
    )

    parsed = resp.get("parsed_object")
    assert parsed is not None
    assert parsed.answer == "ok"
    assert parsed.score == 2
    assert "outputSchema" in captured["turn_params"]


def test_codex_chat_rejects_platform_tools() -> None:
    cfg = ProviderConfig(
        name="codex",
        api_key="",
        base_url="",
        default_model="gpt-5.4",
        platform="codex_app_server",
        models={},
        params={"timeout": 5},
    )
    client = CodexAppServerChatClient(cfg, log_store=None)

    with pytest.raises(ValueError, match="does not support the platform ReAct tool loop"):
        client.chat(
            [{"role": "user", "content": "hello"}],
            model="gpt-5.4",
            tools=[{"type": "function", "function": {"name": "noop"}}],
        )


def test_llm_manager_is_available_for_codex_without_api_key_when_cli_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = LLMManager(
        LLMConfig(
            default="codex",
            dry_run=False,
            platforms={
                "codex_app_server": PlatformConfig(
                    name="codex_app_server",
                    api_key="",
                    base_url="",
                )
            },
            providers={
                "codex": ProviderConfig(
                    name="codex",
                    api_key="",
                    base_url="",
                    default_model="gpt-5.4",
                    platform="codex_app_server",
                    models={},
                    params={},
                )
            },
        )
    )

    monkeypatch.setattr(llm_manager_mod.shutil, "which", lambda name: "/usr/local/bin/codex" if name == "codex" else None)
    assert manager.is_available("codex") is True


def test_llm_manager_selects_codex_across_normal_lite_and_ultra_lite_modes() -> None:
    manager = LLMManager(
        LLMConfig(
            default="codex",
            dry_run=False,
            platforms={
                "codex_app_server": PlatformConfig(
                    name="codex_app_server",
                    api_key="",
                    base_url="",
                ),
                "openai": PlatformConfig(
                    name="openai",
                    api_key="sk-test-openai",
                    base_url="https://api.openai.com/v1",
                ),
            },
            providers={
                "codex": ProviderConfig(
                    name="codex",
                    api_key="",
                    base_url="",
                    default_model="gpt-5.4",
                    platform="codex_app_server",
                    models={},
                    params={},
                ),
                "openai": ProviderConfig(
                    name="openai",
                    api_key="sk-test-openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-5.4",
                    platform="openai",
                    models={},
                    params={},
                ),
                "openai-thinking": ProviderConfig(
                    name="openai-thinking",
                    api_key="sk-test-openai",
                    base_url="https://api.openai.com/v1",
                    default_model="gpt-5.4",
                    platform="openai",
                    models={},
                    params={"reasoning_effort": "high"},
                ),
            },
        )
    )

    normal_client = manager.get_client("codex")
    lite_client = manager.get_lite_client("codex")
    ultra_lite_client = manager.get_ultra_lite_client("codex")
    downgraded_lite = manager.get_lite_client("openai-thinking")

    assert isinstance(normal_client, CodexAppServerChatClient)
    assert isinstance(lite_client, CodexAppServerChatClient)
    assert isinstance(ultra_lite_client, CodexAppServerChatClient)
    assert normal_client.config.name == "codex"
    assert lite_client.config.name == "codex"
    assert ultra_lite_client.config.name == "codex"
    assert downgraded_lite.config.name == "openai"


def test_llm_manager_auxiliary_lite_client_avoids_inherited_codex() -> None:
    manager = LLMManager(
        LLMConfig(
            default="codex",
            dry_run=False,
            platforms={
                "codex_app_server": PlatformConfig(
                    name="codex_app_server",
                    api_key="",
                    base_url="",
                ),
                "deepseek": PlatformConfig(
                    name="deepseek",
                    api_key="sk-test-deepseek",
                    base_url="https://api.deepseek.com/v1",
                ),
            },
            providers={
                "codex": ProviderConfig(
                    name="codex",
                    api_key="",
                    base_url="",
                    default_model="gpt-5.4",
                    platform="codex_app_server",
                    models={},
                    params={},
                ),
                "deepseek": ProviderConfig(
                    name="deepseek",
                    api_key="sk-test-deepseek",
                    base_url="https://api.deepseek.com/v1",
                    default_model="deepseek-chat",
                    platform="deepseek",
                    models={},
                    params={},
                ),
            },
        )
    )

    aux_client = manager.get_auxiliary_lite_client(requested_provider="codex")

    assert aux_client.config.name == "deepseek"


def test_llm_manager_auxiliary_lite_client_honors_explicit_codex() -> None:
    manager = LLMManager(
        LLMConfig(
            default="deepseek",
            dry_run=False,
            platforms={
                "codex_app_server": PlatformConfig(
                    name="codex_app_server",
                    api_key="",
                    base_url="",
                ),
                "deepseek": PlatformConfig(
                    name="deepseek",
                    api_key="sk-test-deepseek",
                    base_url="https://api.deepseek.com/v1",
                ),
            },
            providers={
                "codex": ProviderConfig(
                    name="codex",
                    api_key="",
                    base_url="",
                    default_model="gpt-5.4",
                    platform="codex_app_server",
                    models={},
                    params={},
                ),
                "deepseek": ProviderConfig(
                    name="deepseek",
                    api_key="sk-test-deepseek",
                    base_url="https://api.deepseek.com/v1",
                    default_model="deepseek-chat",
                    platform="deepseek",
                    models={},
                    params={},
                ),
            },
        )
    )

    aux_client = manager.get_auxiliary_lite_client(
        explicit_provider="codex",
        requested_provider="codex",
    )

    assert isinstance(aux_client, CodexAppServerChatClient)
    assert aux_client.config.name == "codex"


def test_session_meta_roundtrip_codex_thread_id() -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.prefs: dict = {}

        def get_session_meta(self, session_id: str) -> dict:
            return {"preferences": dict(self.prefs)}

        def update_session_meta(self, session_id: str, meta: dict) -> None:
            pref = meta.get("preferences") or {}
            self.prefs.update(pref)

    @dataclass
    class _MCfg:
        providers: dict
        default: str = "codex"

    @dataclass
    class _P:
        platform: str

    manager = type(
        "M",
        (),
        {
            "config": _MCfg(
                providers={"codex": _P(platform="codex_app_server")},
            ),
        },
    )()

    store = _FakeStore()
    assert _codex_thread_id_from_session_meta(store, "sid") is None
    assert _is_codex_app_server_provider(manager, "codex") is True
    assert _is_codex_app_server_provider(manager, "openai") is False

    _persist_codex_thread_from_response(
        store,
        "sid",
        manager,
        "codex",
        {"meta": {"codex_thread_id": "persisted-t1"}},
    )
    assert _codex_thread_id_from_session_meta(store, "sid") == "persisted-t1"
