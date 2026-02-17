#!/usr/bin/env python3
"""
Deep Research E2E validation:
- revise priority queue
- supplement load/consume
- review gate behavior
- insights ledger lifecycle

Run (repo root):
  conda run --no-capture-output -n deepsea-rag python scripts/23_test_deep_research_e2e.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaboration.research import job_store
from src.collaboration.research.agent import (
    _consume_revision_queue,
    _load_section_supplements,
    _mark_section_supplements_consumed,
    _scan_fresh_revise_signals,
    review_gate_node,
    synthesize_node,
)
from src.collaboration.research.dashboard import ResearchDashboard


class FakeClient:
    def chat(self, messages, model=None, max_tokens=None):
        last = ""
        if messages and isinstance(messages[-1], dict):
            last = messages[-1].get("content", "")
        if "Limitations and Future Directions" in last or "不足与未来方向" in last:
            return {
                "final_text": "Limitations are acknowledged. Future work should collect broader longitudinal evidence."
            }
        return {"final_text": "This is a concise abstract."}


def _waiter(job_id: str):
    def _inner(section_id: str) -> Optional[Dict[str, object]]:
        return job_store.get_pending_review(job_id, section_id)

    return _inner


def main() -> None:
    temp_dir = tempfile.TemporaryDirectory()
    job_store._DB_PATH = Path(temp_dir.name) / "deep_research_jobs_test.db"

    job_id = "job-e2e-001"
    section_a = "Section A"
    section_b = "Section B"

    dashboard = ResearchDashboard()
    dashboard.add_section(section_a)
    dashboard.add_section(section_b)

    # 1) revise priority queue
    job_store.submit_review(job_id, section_a, action="revise", feedback="Need more evidence A")
    state = {
        "job_id": job_id,
        "dashboard": dashboard,
        "skip_draft_review": False,
        "review_waiter": _waiter(job_id),
        "review_seen_at": {},
        "revision_queue": [],
    }
    _scan_fresh_revise_signals(state)
    assert state["revision_queue"] == [section_a]
    _scan_fresh_revise_signals(state)  # no duplicate enqueue
    assert state["revision_queue"] == [section_a]
    assert _consume_revision_queue(state) == section_a
    assert state["revision_queue"] == []
    print("[PASS] revise priority queue")

    # 2) supplement load + consume
    job_store.submit_gap_supplement(
        job_id,
        section_a,
        "Missing baseline dataset",
        "direct_info",
        {"text": "Baseline dataset 2019 includes 120 samples."},
    )
    job_store.submit_gap_supplement(
        job_id,
        section_a,
        "Need method details",
        "material",
        {"text": "Method appendix URL and extraction protocol."},
    )
    ctx = _load_section_supplements(state, section_a)
    assert "Section-scoped user supplements" in ctx
    assert "Baseline dataset 2019" in ctx
    assert len(job_store.list_gap_supplements(job_id, section_id=section_a, status="pending")) == 2
    _mark_section_supplements_consumed(state, section_a)
    assert len(job_store.list_gap_supplements(job_id, section_id=section_a, status="pending")) == 0
    assert len(job_store.list_gap_supplements(job_id, section_id=section_a, status="consumed")) == 2
    print("[PASS] supplement load+consume")

    # 3) review gate requeue + limitation insight
    job_store.submit_review(job_id, section_a, action="approve", feedback="")
    job_store.submit_review(
        job_id, section_b, action="revise", feedback="Section B lacks contradiction analysis"
    )
    state_gate = {
        "job_id": job_id,
        "dashboard": dashboard,
        "skip_draft_review": False,
        "review_waiter": _waiter(job_id),
        "review_handled_at": {},
    }
    state_gate = review_gate_node(state_gate)
    assert state_gate.get("review_gate_next") == "research"
    assert state_gate.get("current_section") == section_b
    lim_insights = job_store.list_insights(job_id, insight_type="limitation", status="open")
    assert any("lacks contradiction analysis" in (x.get("text") or "") for x in lim_insights)
    print("[PASS] review gate requeue + limitation insight")

    # 4) review gate all approve -> synthesize
    job_store.submit_review(job_id, section_b, action="approve", feedback="ok now")
    state_gate2 = {
        "job_id": job_id,
        "dashboard": dashboard,
        "skip_draft_review": False,
        "review_waiter": _waiter(job_id),
        "review_handled_at": state_gate.get("review_handled_at", {}),
    }
    state_gate2 = review_gate_node(state_gate2)
    assert state_gate2.get("review_gate_next") == "synthesize"
    print("[PASS] review gate all approve -> synthesize")

    # 5) synthesize consumes open insights -> addressed
    job_store.append_insight(
        job_id, "gap", "Need larger multilingual corpus", section_id=section_b, source_context="test"
    )
    job_store.append_insight(
        job_id, "conflict", "Two studies disagree on causality", section_id=section_b, source_context="test"
    )
    state_syn = {
        "job_id": job_id,
        "llm_client": FakeClient(),
        "model_override": None,
        "output_language": "en",
        "markdown_parts": ["# Demo Review", "## Section A\ntext", "## Section B\ntext"],
        "canvas_id": "",
        "dashboard": dashboard,
    }
    state_syn = synthesize_node(state_syn)
    joined = "\n".join(state_syn.get("markdown_parts", []))
    assert "Limitations and Future Directions" in joined
    assert len(job_store.list_insights(job_id, status="open")) == 0
    assert len(job_store.list_insights(job_id, status="addressed")) >= 3
    print("[PASS] synthesize consumes insights and marks addressed")

    print("E2E validation finished: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
