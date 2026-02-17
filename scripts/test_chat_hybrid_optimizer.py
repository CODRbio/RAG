#!/usr/bin/env python3
"""
测试 /chat/stream 在 hybrid + 查询优化器 + Scholar 时的请求与响应。
用于验证前端传参与后端是否一致、是否触发 Scholar 有头浏览器。

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
    url = f"{base}/chat/stream"

    # 与前端开启「查询优化」+ 勾选 Scholar 时发送的 body 一致
    body = {
        "message": "深海冷泉的生态意义是什么？",
        "search_mode": "hybrid",
        "web_providers": ["scholar"],
        "use_query_optimizer": True,
        "query_optimizer_max_queries": 3,
    }
    print("POST", url)
    print("Body:", body)
    print()

    with requests.post(url, json=body, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        print("[stream] status ok, receiving events...")
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(line)


if __name__ == "__main__":
    main()
