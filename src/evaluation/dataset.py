"""
评测数据集加载与规范化
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class EvalCase:
    id: str
    query: str
    mode: str = "local"
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_citations: List[str] = field(default_factory=list)
    reference_answer: str = ""
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def load_dataset(path: str | Path) -> Tuple[List[EvalCase], Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        cases = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        raw = {"cases": cases}
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            raw = {"cases": raw}

    cases = raw.get("cases") or []
    meta = {k: v for k, v in raw.items() if k != "cases"}
    normalized: List[EvalCase] = []
    for i, c in enumerate(cases, 1):
        normalized.append(_normalize_case(c, i))
    return normalized, meta


def _normalize_case(raw: Dict[str, Any], index: int) -> EvalCase:
    cid = str(raw.get("id") or f"case_{index:03d}")
    query = (raw.get("query") or "").strip()
    mode = (raw.get("mode") or "local").strip()
    expected_doc_ids = list(raw.get("expected_doc_ids") or [])
    expected_citations = list(raw.get("expected_citations") or [])
    reference_answer = (raw.get("reference_answer") or "").strip()
    tags = list(raw.get("tags") or [])
    meta = dict(raw.get("meta") or {})
    return EvalCase(
        id=cid,
        query=query,
        mode=mode,
        expected_doc_ids=expected_doc_ids,
        expected_citations=expected_citations,
        reference_answer=reference_answer,
        tags=tags,
        meta=meta,
    )
