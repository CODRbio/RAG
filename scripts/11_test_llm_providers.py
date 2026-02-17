#!/usr/bin/env python3
"""
批量测试所有 LLM provider 的最小可用性。

用法（建议在 conda 环境 deepsea-rag 下）:
  conda run -n deepsea-rag python scripts/11_test_llm_providers.py
"""

import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_single(provider: str, out: Queue) -> None:
    from src.llm import LLMManager

    try:
        manager = LLMManager.from_json(ROOT / "config" / "rag_config.json")
        client = manager.get_client(provider)
        overrides = {"max_tokens": 16}
        if provider.startswith("openai"):
            overrides = {"max_completion_tokens": 16}
        if provider == "claude-thinking":
            overrides = {
                "max_tokens": 16000,
                "thinking": {"type": "enabled", "budget_tokens": 8000},
            }
        resp = client.chat(
            [
                {"role": "system", "content": "你是一个简洁助手。"},
                {"role": "user", "content": "请回复 OK"},
            ],
            **overrides,
        )
        text = (resp.get("final_text") or "").strip()
        out.put(("ok", text))
    except Exception as exc:
        detail = str(exc)
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detail = f"{detail} | body: {resp.text[:500]}"
            except Exception:
                pass
        out.put(("fail", detail))


def main() -> None:
    from src.llm import LLMManager

    manager = LLMManager.from_json(ROOT / "config" / "rag_config.json")
    providers = manager.get_provider_names()
    if not providers:
        print("No providers found in config.")
        return

    print("LLM providers:", ", ".join(providers), flush=True)
    print("-" * 60, flush=True)

    timeout_s = 10
    for name in providers:
        t0 = time.time()
        q: Queue = Queue()
        p = Process(target=_run_single, args=(name, q))
        p.start()
        per_timeout = 25 if name == "claude-thinking" else timeout_s
        p.join(timeout=per_timeout)
        elapsed = int((time.time() - t0) * 1000)
        if p.is_alive():
            p.terminate()
            p.join(timeout=3)
            print(f"[TIMEOUT] {name} ({elapsed}ms) -> >{per_timeout}s", flush=True)
            continue
        if q.empty():
            print(f"[FAIL] {name} ({elapsed}ms) -> no result", flush=True)
            continue
        status, msg = q.get()
        if status == "ok":
            print(f"[OK] {name} ({elapsed}ms) -> {msg[:50]}", flush=True)
        else:
            print(f"[FAIL] {name} ({elapsed}ms) -> {msg}", flush=True)


if __name__ == "__main__":
    main()
