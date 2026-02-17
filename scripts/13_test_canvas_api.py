#!/usr/bin/env python3
"""
Canvas API 联调：创建 -> 大纲 -> 草稿 -> 快照 -> 导出。
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

    # 1) POST /canvas 创建
    r1 = requests.post(
        f"{base}/canvas",
        json={"session_id": "", "topic": "深海冷泉综述"},
        timeout=10,
    )
    r1.raise_for_status()
    d1 = r1.json()
    canvas_id = d1["id"]
    print(f"[1] POST /canvas -> canvas_id={canvas_id}")

    # 2) PATCH 更新
    r2 = requests.patch(
        f"{base}/canvas/{canvas_id}",
        json={"working_title": "深海冷泉生态系统研究进展"},
        timeout=10,
    )
    r2.raise_for_status()
    print("[2] PATCH /canvas/{id} -> ok")

    # 3) POST outline
    r3 = requests.post(
        f"{base}/canvas/{canvas_id}/outline",
        json={
            "sections": [
                {"id": "s1", "title": "引言", "level": 1, "order": 0, "status": "done"},
                {"id": "s2", "title": "冷泉分布与形成", "level": 1, "order": 1, "status": "drafting"},
            ]
        },
        timeout=10,
    )
    r3.raise_for_status()
    print("[3] POST /canvas/{id}/outline -> ok")

    # 4) POST draft
    r4 = requests.post(
        f"{base}/canvas/{canvas_id}/drafts",
        json={
            "block": {
                "section_id": "s1",
                "content_md": "深海冷泉是海底渗出的流体系统。",
                "version": 1,
            }
        },
        timeout=10,
    )
    r4.raise_for_status()
    print("[4] POST /canvas/{id}/drafts -> ok")

    # 5) POST snapshot
    r5 = requests.post(f"{base}/canvas/{canvas_id}/snapshot", timeout=10)
    r5.raise_for_status()
    ver = r5.json()["version_number"]
    print(f"[5] POST /canvas/{canvas_id}/snapshot -> version_number={ver}")

    # 6) GET export
    r6 = requests.get(f"{base}/canvas/{canvas_id}/export", timeout=10)
    r6.raise_for_status()
    export = r6.json()
    assert "outline" in export and "drafts" in export
    assert len(export["outline"]) == 2
    assert "s1" in export["drafts"]
    print(f"[6] GET /canvas/{canvas_id}/export -> outline={len(export['outline'])}, drafts keys={list(export['drafts'].keys())}")

    # 7) GET canvas
    r7 = requests.get(f"{base}/canvas/{canvas_id}", timeout=10)
    r7.raise_for_status()
    print(f"[7] GET /canvas/{canvas_id} -> ok")

    print("OK: canvas CRUD + outline + drafts + snapshot + export verified.")


if __name__ == "__main__":
    main()
