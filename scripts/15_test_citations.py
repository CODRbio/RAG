#!/usr/bin/env python3
"""
引用管理联调：检索 -> citation_pool 入库 -> 导出 BibTeX + 参考文献段落。
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

    # 1) 创建 Canvas
    r1 = requests.post(
        f"{base}/canvas",
        json={"session_id": "", "topic": "深海冷泉综述"},
        timeout=10,
    )
    r1.raise_for_status()
    canvas_id = r1.json()["id"]
    print(f"[1] Canvas created: {canvas_id}")

    # 2) POST /chat 绑定 canvas，发送触发检索的消息（如 /search 或直接问检索类问题）
    r2 = requests.post(
        f"{base}/chat",
        json={"canvas_id": canvas_id, "message": "/search 冷泉 生态系统", "search_mode": "local"},
        timeout=60,
    )
    r2.raise_for_status()
    data = r2.json()
    session_id = data["session_id"]
    citations = data.get("citations", [])
    print(f"[2] Chat (retrieval) -> session_id={session_id}, citations={len(citations)} cite_keys: {citations[:5]}")

    # 3) GET /canvas/{id}/citations?format=bibtex
    r3 = requests.get(f"{base}/canvas/{canvas_id}/citations", params={"format": "bibtex"}, timeout=10)
    r3.raise_for_status()
    bibtex = r3.json().get("content", "")
    print("[3] GET citations (bibtex):")
    print(bibtex[:500] + "..." if len(bibtex) > 500 else bibtex)

    # 4) GET /canvas/{id}/citations?format=text
    r4 = requests.get(f"{base}/canvas/{canvas_id}/citations", params={"format": "text"}, timeout=10)
    r4.raise_for_status()
    ref_list = r4.json().get("content", "")
    print("[4] GET citations (reference list):")
    print(ref_list[:500] + "..." if len(ref_list) > 500 else ref_list)

    # 5) GET format=both 确认结构
    r5 = requests.get(f"{base}/canvas/{canvas_id}/citations", params={"format": "both"}, timeout=10)
    r5.raise_for_status()
    both = r5.json()
    assert "bibtex" in both and "reference_list" in both and "citations" in both
    print(f"[5] GET format=both -> {len(both.get('citations', []))} citation(s)")

    print("OK: retrieval -> citation_pool -> bibtex + reference list exercised.")


if __name__ == "__main__":
    main()
