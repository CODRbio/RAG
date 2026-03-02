"""
Frontend event logger: write UI-side debug events to logs/frontend/*.jsonl.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "frontend"
_lock = threading.Lock()


def _daily_path() -> Path:
    _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"


def log_frontend_event(event: Dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        **(event or {}),
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    path = _daily_path()
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_frontend_events(events: List[Dict[str, Any]]) -> int:
    if not events:
        return 0
    path = _daily_path()
    rows = []
    now = datetime.now().isoformat(timespec="milliseconds")
    for e in events:
        rows.append(json.dumps({"timestamp": now, **(e or {})}, ensure_ascii=False, default=str))
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    return len(rows)
