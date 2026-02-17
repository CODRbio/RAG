#!/usr/bin/env python3
"""本地调用 chat 逻辑，复现 500 并打印 traceback。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from src.api.routes_chat import chat_post
    from src.api.schemas import ChatRequest
    body = ChatRequest(message="/search 深海冷泉", search_mode="local")
    try:
        out = chat_post(body)
        print("OK", out.session_id, out.evidence_summary.total_chunks)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
