#!/usr/bin/env python3
"""Quick test: call gemini-3.1-pro-preview via project LLM manager. Exit 0 = OK."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

def main():
    from src.llm.llm_manager import get_manager
    manager = get_manager()
    client = manager.get_client("gemini")
    model = "gemini-3.1-pro-preview"
    messages = [{"role": "user", "content": "Reply with exactly: OK"}]
    print(f"Calling provider=gemini model={model} ...", flush=True)
    try:
        resp = client.chat(messages, model=model, timeout=45)
        text = (resp.get("final_text") or resp.get("content") or "").strip()
        print(f"Response: {text[:200]}")
        print("SUCCESS: Gemini 3.1 Pro Preview 调用正常")
        return 0
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        try:
            import requests
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                print(f"Response body: {e.response.text[:800]}", file=sys.stderr)
        except Exception:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
