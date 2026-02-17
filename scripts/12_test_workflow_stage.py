#!/usr/bin/env python3
"""
工作流阶段切换验证：/outline -> outline, /draft -> drafting, /edit -> refine。

前置：先启动 API
  python scripts/08_run_api.py

然后执行（项目根目录）:
  python scripts/12_test_workflow_stage.py
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

    session_id = None

    # 1) 新会话 /outline -> stage 应为 outline
    r1 = requests.post(
        f"{base}/chat",
        json={"message": "/outline"},
        timeout=60,
    )
    r1.raise_for_status()
    session_id = r1.json().get("session_id")
    if not session_id:
        print("no session_id")
        sys.exit(1)
    s1 = requests.get(f"{base}/sessions/{session_id}", timeout=10)
    s1.raise_for_status()
    stage1 = s1.json().get("stage", "")
    print(f"[1] /outline -> stage={stage1}")
    assert stage1 == "outline", f"expected outline got {stage1}"

    # 2) 同会话 /draft -> stage 应为 drafting
    r2 = requests.post(
        f"{base}/chat",
        json={"session_id": session_id, "message": "/draft 第一节"},
        timeout=60,
    )
    r2.raise_for_status()
    s2 = requests.get(f"{base}/sessions/{session_id}", timeout=10)
    s2.raise_for_status()
    stage2 = s2.json().get("stage", "")
    print(f"[2] /draft -> stage={stage2}")
    assert stage2 == "drafting", f"expected drafting got {stage2}"

    # 3) 同会话 /edit -> stage 应为 refine
    r3 = requests.post(
        f"{base}/chat",
        json={"session_id": session_id, "message": "/edit 润色第一段"},
        timeout=60,
    )
    r3.raise_for_status()
    s3 = requests.get(f"{base}/sessions/{session_id}", timeout=10)
    s3.raise_for_status()
    stage3 = s3.json().get("stage", "")
    print(f"[3] /edit -> stage={stage3}")
    assert stage3 == "refine", f"expected refine got {stage3}"

    print("OK: workflow stage transitions verified.")


if __name__ == "__main__":
    main()
