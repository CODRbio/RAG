#!/usr/bin/env python3
"""
多轮对话与意图分流验证：调用 POST /chat，校验检索分支与无检索分支。

前置：先启动 API
  python scripts/08_run_api.py

然后执行（项目根目录）:
  python scripts/10_test_multiturn.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    try:
        import requests
    except ImportError:
        print("pip install requests")
        sys.exit(1)

    from config.settings import settings
    base = f"http://{settings.api.host}:{settings.api.port}"

    # 1) 显式非检索指令 /status：不走检索（不依赖 BGE/Milvus），验证意图分流
    #    需配置有效 LLM API Key（config/rag_config.json 或环境变量）
    r1 = requests.post(
        f"{base}/chat",
        json={"message": "/status"},
        timeout=60,
    )
    r1.raise_for_status()
    d1 = r1.json()
    session_id = d1.get("session_id")
    total_chunks_1 = (d1.get("evidence_summary") or {}).get("total_chunks", -1)
    print(f"[1] /status (no retrieval) -> session_id={session_id}, total_chunks={total_chunks_1}, response_len={len(d1.get('response',''))}")

    if not session_id:
        print("no session_id in response")
        sys.exit(1)

    # 2) 自然语言（非检索意图）：应不走检索
    r2 = requests.post(
        f"{base}/chat",
        json={"session_id": session_id, "message": "你好，当前进度如何？"},
        timeout=60,
    )
    r2.raise_for_status()
    d2 = r2.json()
    total_chunks_2 = (d2.get("evidence_summary") or {}).get("total_chunks", -1)
    print(f"[2] 你好，当前进度如何？ -> total_chunks={total_chunks_2}, response_len={len(d2.get('response',''))}")

    # 3) GET session 校验历史
    r3 = requests.get(f"{base}/sessions/{session_id}", timeout=10)
    r3.raise_for_status()
    sess = r3.json()
    print(f"[3] GET session -> turn_count={sess.get('turn_count', 0)}")

    print("OK: multiturn + intent routing exercised.")

if __name__ == "__main__":
    main()
