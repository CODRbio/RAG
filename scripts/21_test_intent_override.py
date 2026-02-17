#!/usr/bin/env python3
"""
测试意图覆盖与显式命令优先级。
前置：先启动 API（python scripts/08_run_api.py）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_base_url():
    from config.settings import settings

    return f"http://{settings.api.host}:{settings.api.port}"


def _read_meta_event(resp):
    """读取 SSE 的 meta 事件数据并返回解析后的 dict。"""
    current_event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line.replace("event: ", "").strip()
            continue
        if line.startswith("data: ") and current_event == "meta":
            import json

            return json.loads(line.replace("data: ", "", 1))
    return None


def test_intent_detect():
    import requests

    base = _get_base_url()
    resp = requests.post(
        f"{base}/intent/detect",
        json={"message": "请帮我继续上一段内容。", "current_stage": "drafting"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print("[intent_detect]", data)
    assert "intent_type" in data
    assert "needs_retrieval" in data


def test_intent_override_chat():
    import requests

    base = _get_base_url()
    with requests.post(
        f"{base}/chat/stream",
        json={
            "message": "随便聊聊",
            "search_mode": "none",
            "intent_override": "chitchat",
        },
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        meta = _read_meta_event(resp)
        print("[override_chat meta]", meta)
        assert meta and meta.get("intent", {}).get("intent_type") == "chitchat"


def test_command_priority_over_override():
    import requests

    base = _get_base_url()
    with requests.post(
        f"{base}/chat/stream",
        json={
            "message": "/search deep sea cold seep",
            "search_mode": "local",
            "intent_override": "chitchat",
        },
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        meta = _read_meta_event(resp)
        print("[command_priority meta]", meta)
        assert meta and meta.get("intent", {}).get("intent_type") == "search_targeted"


def main():
    test_intent_detect()
    test_intent_override_chat()
    test_command_priority_over_override()
    print("All intent override tests passed.")


if __name__ == "__main__":
    main()
