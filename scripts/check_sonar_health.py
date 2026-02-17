#!/usr/bin/env python3
"""
Sonar provider health check for DeepSea-RAG.

Checks:
1) Config merge and provider presence
2) Client creation via LLMManager
3) Optional live chat request
4) Optional raw network probe

Usage:
  conda run -n deepsea-rag python scripts/check_sonar_health.py
  conda run -n deepsea-rag python scripts/check_sonar_health.py --skip-chat
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "rag_config.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config() -> Dict[str, Any]:
    base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    local_path = CONFIG_PATH.with_name("rag_config.local.json")
    if local_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        return deep_merge(base, local)
    return base


def check_network_tls(base_url: str, timeout_sec: int = 8) -> Tuple[bool, str]:
    try:
        u = urlparse(base_url)
        host = u.hostname
        port = u.port or 443
        if not host:
            return False, f"Invalid URL host: {base_url}"
        ip = socket.gethostbyname(host)
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout_sec) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                pass
        return True, f"DNS+TLS OK ({host} -> {ip}:{port})"
    except Exception as e:
        return False, f"DNS/TLS failed for {base_url}: {e!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check sonar provider health")
    parser.add_argument("--skip-chat", action="store_true", help="Skip live chat request")
    args = parser.parse_args()

    print("=== Sonar Health Check ===")
    print(f"Config: {CONFIG_PATH}")

    # 1) Merged config sanity
    merged = load_merged_config()
    llm = (merged.get("llm") or {})
    providers = (llm.get("providers") or {})
    sonar = providers.get("sonar")
    if not sonar:
        _fail("Provider 'sonar' missing in merged config")
        return 1
    _ok("Provider 'sonar' exists in merged config")

    base_url = str(sonar.get("base_url") or "")
    default_model = str(sonar.get("default_model") or "")
    models = sonar.get("models") or {}
    api_key = str(sonar.get("api_key") or "")

    if base_url:
        _ok(f"base_url: {base_url}")
    else:
        _fail("base_url is empty")
        return 1

    if default_model:
        _ok(f"default_model: {default_model}")
    else:
        _warn("default_model is empty")

    if isinstance(models, dict) and models:
        _ok(f"models: {sorted(models.keys())}")
    else:
        _warn("models map is empty")

    if api_key.strip():
        _ok("api_key is set (non-empty)")
    else:
        _fail("api_key is empty; please set llm.providers.sonar.api_key in rag_config.local.json")
        return 1

    # 2) TLS reachability probe
    tls_ok, tls_msg = check_network_tls(base_url)
    if tls_ok:
        _ok(tls_msg)
    else:
        _warn(tls_msg)

    # 3) LLMManager client creation + optional live request
    try:
        from src.llm.llm_manager import LLMManager

        mgr = LLMManager.from_json(CONFIG_PATH)
        if "sonar" not in mgr.get_provider_names():
            _fail(f"'sonar' not in manager providers: {mgr.get_provider_names()}")
            return 1
        _ok("LLMManager loads provider list with sonar")

        client = mgr.get_client("sonar")
        _ok(f"get_client('sonar') -> {type(client).__name__}")
    except Exception as e:
        _fail(f"Client creation failed: {e!r}")
        traceback.print_exc()
        return 1

    if args.skip_chat:
        print("Skipping live chat request (--skip-chat).")
        return 0

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply exactly with SONAR_OK"},
            ],
            model="sonar",
            max_tokens=20,
        )
        final_text = (resp.get("final_text") or "").strip()
        provider = resp.get("provider")
        model = resp.get("model")
        latency = (resp.get("meta") or {}).get("latency_ms")

        _ok(f"Live chat succeeded provider={provider} model={model} latency_ms={latency}")
        print(f"final_text: {final_text}")
        if "SONAR_OK" in final_text:
            _ok("Response content check passed")
        else:
            _warn("Response did not exactly contain SONAR_OK (API path still works)")
        return 0
    except Exception as e:
        _fail(f"Live chat failed: {e!r}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
