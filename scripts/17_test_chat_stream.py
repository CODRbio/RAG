#!/usr/bin/env python3
"""
流式对话联调：POST /chat/stream，打印 SSE 事件。
前置：先启动 API（python scripts/08_run_api.py）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import requests
    from config.settings import settings

    base = f"http://{settings.api.host}:{settings.api.port}"

    with requests.post(
        f"{base}/chat/stream",
        json={"message": "请简要解释深海冷泉的生态意义。"},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        print("[stream] status ok, start receiving events...")
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(line)


if __name__ == "__main__":
    main()
