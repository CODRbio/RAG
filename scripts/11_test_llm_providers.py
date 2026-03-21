#!/usr/bin/env python3
"""
批量测试所有 LLM provider 的最小可用性。

用法（建议在 conda 环境 deepsea-rag 下）:
  conda run -n deepsea-rag python scripts/11_test_llm_providers.py
  conda run -n deepsea-rag python scripts/11_test_llm_providers.py --providers codex --modes normal lite
"""

import argparse
import json
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLIENT_MODES = ("normal", "lite", "ultra-lite")


def _parse_providers(raw_values: list[str]) -> list[str]:
    providers: list[str] = []
    for raw in raw_values:
        for part in raw.split(","):
            name = part.strip()
            if name:
                providers.append(name)
    return providers


def _select_client(manager, provider: str, mode: str):
    if mode == "normal":
        return manager.get_client(provider)
    if mode == "lite":
        return manager.get_lite_client(provider)
    if mode == "ultra-lite":
        return manager.get_ultra_lite_client(provider)
    raise ValueError(f"Unsupported client mode: {mode}")


def _build_overrides(requested_provider: str, resolved_provider: str, mode: str) -> dict:
    overrides = {"max_tokens": 16}
    if resolved_provider.startswith("openai"):
        overrides = {"max_completion_tokens": 16}
    if requested_provider == "claude-thinking" and mode == "normal":
        overrides = {
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 8000},
        }
    return overrides


def _load_messages(messages_file: str | None, system_prompt: str, user_prompt: str) -> list[dict]:
    if not messages_file:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    payload = json.loads(Path(messages_file).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("--messages-file must be a JSON array of OpenAI-style message objects")
    return payload


def _run_single(
    provider: str,
    mode: str,
    config_path: str,
    system_prompt: str,
    user_prompt: str,
    messages_file: str | None,
    skip_chat: bool,
    out: Queue,
) -> None:
    from src.llm import LLMManager

    try:
        manager = LLMManager.from_json(config_path)
        client = _select_client(manager, provider, mode)
        client_cfg = getattr(client, "config", None)
        resolved_provider = getattr(client_cfg, "name", provider)
        platform = getattr(client_cfg, "platform", "")
        result = {
            "requested_provider": provider,
            "requested_mode": mode,
            "resolved_provider": resolved_provider,
            "platform": platform,
            "client_class": type(client).__name__,
        }
        if skip_chat:
            out.put(("ok", result))
            return

        messages = _load_messages(messages_file, system_prompt, user_prompt)
        resp = client.chat(
            messages,
            **_build_overrides(provider, resolved_provider, mode),
        )
        result["final_text"] = (resp.get("final_text") or "").strip()
        result["latency_ms"] = ((resp.get("meta") or {}).get("latency_ms"))
        out.put(("ok", result))
    except Exception as exc:
        detail = str(exc)
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detail = f"{detail} | body: {resp.text[:500]}"
            except Exception:
                pass
        out.put(("fail", {
            "requested_provider": provider,
            "requested_mode": mode,
            "error": detail,
        }))


def main() -> None:
    from src.llm import LLMManager

    parser = argparse.ArgumentParser(description="批量测试 LLM provider 的 normal/lite/ultra-lite 路径")
    parser.add_argument("--config", default=str(ROOT / "config" / "rag_config.json"), help="配置文件路径")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=[],
        help="只测试指定 provider，可传多个值或逗号分隔，例如: --providers codex openai-thinking",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=CLIENT_MODES,
        default=["normal"],
        help="要测试的 client 模式，默认只测 normal",
    )
    parser.add_argument("--system", default="你是一个简洁助手。", help="system prompt")
    parser.add_argument("--message", default="请回复 OK", help="user prompt")
    parser.add_argument("--messages-file", help="从 JSON 文件读取完整 messages 数组，格式为 OpenAI-style messages")
    parser.add_argument("--skip-chat", action="store_true", help="只测试 client 选择与解析，不实际发起对话")
    parser.add_argument("--timeout", type=int, default=12, help="单次测试超时秒数（Codex / Claude-thinking 会自动抬高）")
    args = parser.parse_args()

    manager = LLMManager.from_json(args.config)
    providers = manager.get_provider_names()
    if not providers:
        print("No providers found in config.")
        return
    requested_providers = _parse_providers(args.providers)
    if requested_providers:
        unknown = [name for name in requested_providers if name not in providers]
        if unknown:
            print(f"Unknown providers: {', '.join(unknown)}")
            print(f"Available: {', '.join(providers)}")
            raise SystemExit(2)
        providers = requested_providers

    print("LLM providers:", ", ".join(providers), flush=True)
    print("Modes:", ", ".join(args.modes), flush=True)
    print("-" * 60, flush=True)

    for name in providers:
        for mode in args.modes:
            t0 = time.time()
            q: Queue = Queue()
            p = Process(
                target=_run_single,
                args=(
                    name,
                    mode,
                    args.config,
                    args.system,
                    args.message,
                    args.messages_file,
                    args.skip_chat,
                    q,
                ),
            )
            p.start()
            per_timeout = args.timeout
            if name == "claude-thinking" and mode == "normal":
                per_timeout = max(per_timeout, 25)
            if "codex" in name:
                per_timeout = max(per_timeout, 45)
            p.join(timeout=per_timeout)
            elapsed = int((time.time() - t0) * 1000)
            label = f"{name}/{mode}"
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
                print(f"[TIMEOUT] {label} ({elapsed}ms) -> >{per_timeout}s", flush=True)
                continue
            if q.empty():
                print(f"[FAIL] {label} ({elapsed}ms) -> no result", flush=True)
                continue
            status, payload = q.get()
            if status == "ok":
                summary = (
                    f"client={payload['client_class']} "
                    f"resolved={payload['resolved_provider']} "
                    f"platform={payload['platform'] or '-'}"
                )
                if not args.skip_chat:
                    summary = (
                        f"{summary} "
                        f"latency={payload.get('latency_ms')}ms "
                        f"text={str(payload.get('final_text') or '')[:80]}"
                    )
                print(f"[OK] {label} ({elapsed}ms) -> {summary}", flush=True)
            else:
                print(f"[FAIL] {label} ({elapsed}ms) -> {payload['error']}", flush=True)


if __name__ == "__main__":
    main()
