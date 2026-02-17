#!/usr/bin/env python3
"""
记忆扩展联调：Working Memory（Canvas 摘要缓存）与 Persistent Store（user_id 偏好）。
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
    from src.collaboration.memory.working_memory import get_working_memory
    from src.collaboration.memory.persistent_store import get_user_profile, upsert_user_profile

    base = f"http://{settings.api.host}:{settings.api.port}"

    # 1) 创建 Canvas 并写入 outline
    r1 = requests.post(
        f"{base}/canvas",
        json={"session_id": "", "topic": "深海冷泉综述"},
        timeout=10,
    )
    r1.raise_for_status()
    canvas_id = r1.json()["id"]
    requests.post(
        f"{base}/canvas/{canvas_id}/outline",
        json={"sections": [{"id": "s1", "title": "引言", "level": 1, "order": 0}]},
        timeout=10,
    ).raise_for_status()
    print(f"[1] Canvas created: {canvas_id}")

    # 2) POST /chat 绑定该 canvas（新 session），触发生成 Working Memory
    r2 = requests.post(
        f"{base}/chat",
        json={"canvas_id": canvas_id, "message": "当前进度如何？"},
        timeout=60,
    )
    r2.raise_for_status()
    session_id = r2.json()["session_id"]
    print(f"[2] Chat with canvas_id -> session_id={session_id}")

    # 3) 检查 Working Memory 缓存
    wm = get_working_memory(canvas_id)
    assert wm is not None and wm.get("summary"), "working memory should be cached"
    print(f"[3] Working memory cached: {wm['summary'][:80]}...")

    # 4) Persistent Store: 写入用户偏好
    user_id = "test_user_14"
    upsert_user_profile(user_id, {"llm_provider": "deepseek", "citation_style": "apa"})
    profile = get_user_profile(user_id)
    assert profile is not None and profile.get("preferences", {}).get("llm_provider") == "deepseek"
    print("[4] User profile upserted and read back ok")

    # 5) /chat 带 user_id，确认不报错（偏好会注入系统提示）
    r5 = requests.post(
        f"{base}/chat",
        json={"session_id": session_id, "user_id": user_id, "message": "谢谢"},
        timeout=60,
    )
    r5.raise_for_status()
    print("[5] POST /chat with user_id -> ok")

    print("OK: working memory + persistent store exercised.")


if __name__ == "__main__":
    main()
