#!/usr/bin/env python3
"""
API 回归脚本：年份窗口硬过滤 + 引文格式（APA/IEEE）。

覆盖点：
1) /chat 透传并回显 diagnostics.year_window
2) /deep-research/submit 接受 year_start/year_end
3) （可选）/deep-research/start 接受 year_start/year_end
4) format_reference_list 支持 style="apa" / "ieee"

运行示例：
  conda run --no-capture-output -n deepsea-rag python scripts/27_test_year_window_and_citation_style.py
  conda run --no-capture-output -n deepsea-rag python scripts/27_test_year_window_and_citation_style.py --with-start
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _post_json(base: str, path: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    import requests

    url = f"{base}{path}"
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.text[:1200]
        raise RuntimeError(f"{path} failed ({resp.status_code}): {detail}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"{path} did not return JSON: {e}") from e


def _check_chat_year_window(
    base: str,
    timeout: int,
    year_start: int,
    year_end: int,
    expect_window: Tuple[int, int],
    session_id: Optional[str] = None,
) -> str:
    """
    发送 chat 请求并校验 diagnostics.year_window。
    返回 session_id 供下一轮复用。
    """
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "message": "请基于文献总结深海冷泉研究进展，并给出关键证据。",
        "search_mode": "local",
        "year_start": year_start,
        "year_end": year_end,
        "local_top_k": 6,
        "final_top_k": 6,
    }
    data = _post_json(base, "/chat", payload, timeout=timeout)
    sid = data.get("session_id") or ""
    _assert(bool(sid), "chat response missing session_id")
    ev = data.get("evidence_summary") or {}
    diag = ev.get("diagnostics") or {}
    yw = diag.get("year_window") or {}

    got_start = yw.get("year_start")
    got_end = yw.get("year_end")
    exp_start, exp_end = expect_window
    _assert(
        got_start == exp_start and got_end == exp_end,
        (
            "year_window mismatch: "
            f"expect=({exp_start}, {exp_end}), got=({got_start}, {got_end}), "
            f"full_diag={json.dumps(diag, ensure_ascii=False)}"
        ),
    )
    return sid


def _check_submit_year_window(base: str, timeout: int, year_start: int, year_end: int) -> str:
    payload: Dict[str, Any] = {
        "topic": "深海冷泉微生物研究进展",
        "search_mode": "local",
        "confirmed_outline": ["背景与定义", "研究进展", "挑战与展望"],
        "year_start": year_start,
        "year_end": year_end,
    }
    data = _post_json(base, "/deep-research/submit", payload, timeout=timeout)
    job_id = data.get("job_id") or ""
    _assert(bool(job_id), "deep-research/submit response missing job_id")
    return job_id


def _check_start_year_window(base: str, timeout: int, year_start: int, year_end: int) -> None:
    payload: Dict[str, Any] = {
        "topic": "深海冷泉甲烷循环机制",
        "search_mode": "local",
        "year_start": year_start,
        "year_end": year_end,
        "local_top_k": 8,
        "final_top_k": 12,
    }
    data = _post_json(base, "/deep-research/start", payload, timeout=timeout)
    outline = data.get("outline") or []
    _assert(isinstance(outline, list), "deep-research/start response outline must be list")


def _cancel_job(base: str, timeout: int, job_id: str) -> None:
    import requests

    url = f"{base}/deep-research/jobs/{job_id}/cancel"
    resp = requests.post(url, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"/deep-research/jobs/{job_id}/cancel failed: {resp.status_code} {resp.text[:500]}")


def _check_formatter_styles() -> None:
    from src.collaboration.canvas.models import Citation
    from src.collaboration.citation.formatter import format_reference_list

    sample = Citation(
        title="Deep Sea Cold Seep Ecosystems",
        authors=["John Smith", "Alice Brown"],
        year=2023,
        doi="10.1000/example-doi",
    )
    apa = format_reference_list([sample], style="apa")
    ieee = format_reference_list([sample], style="ieee")

    _assert("Smith" in apa and "(2023)." in apa, f"APA output unexpected: {apa}")
    _assert(ieee.strip().startswith("[1]"), f"IEEE output unexpected: {ieee}")


def main() -> None:
    parser = argparse.ArgumentParser(description="回归：年份窗口硬过滤 + APA/IEEE")
    parser.add_argument("--base", default="", help="API base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")
    parser.add_argument(
        "--with-start",
        action="store_true",
        help="额外执行 /deep-research/start（较慢，依赖 LLM 可用）",
    )
    args = parser.parse_args()

    if args.base:
        base = args.base.rstrip("/")
    else:
        from config.settings import settings

        base = f"http://{settings.api.host}:{settings.api.port}"

    print(f"[info] base={base}")

    # 1) Chat: 正常年份窗口
    sid = _check_chat_year_window(
        base=base,
        timeout=args.timeout,
        year_start=2020,
        year_end=2024,
        expect_window=(2020, 2024),
    )
    print("[1] /chat year_window passthrough -> PASS")

    # 2) Chat: 反向输入（校验后端自动归一化）
    _check_chat_year_window(
        base=base,
        timeout=args.timeout,
        year_start=2026,
        year_end=2022,
        expect_window=(2022, 2026),
        session_id=sid,
    )
    print("[2] /chat reversed year_window normalize -> PASS")

    # 3) Deep Research submit: year 字段可被 API 接受
    job_id = _check_submit_year_window(base=base, timeout=args.timeout, year_start=2021, year_end=2025)
    print(f"[3] /deep-research/submit year_window accepted -> PASS (job_id={job_id})")

    # 避免任务继续占资源，提交后立即取消（若 worker 尚未执行则会直接 cancelled）
    _cancel_job(base=base, timeout=args.timeout, job_id=job_id)
    print("[4] /deep-research/jobs/{job_id}/cancel -> PASS")

    # 4) 可选：Deep Research start 合同验证
    if args.with_start:
        _check_start_year_window(base=base, timeout=max(args.timeout, 240), year_start=2019, year_end=2024)
        print("[5] /deep-research/start year_window accepted -> PASS")

    # 5) Formatter regression
    _check_formatter_styles()
    print("[6] formatter style=apa/ieee -> PASS")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

