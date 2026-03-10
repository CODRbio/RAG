"""
Shared 1+1+1 structured query generation for Chat and Research plan.

Used by: Chat Round 2, Research plan_node background retrieval.
"""

import re
from typing import Any, Dict, List, Optional

from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

# Optional PRECISION_ZH block when query is Chinese (injected into prompt).
_PRECISION_ZH_INSTRUCTION = """

Category B2 — PRECISION_ZH (exactly 1 query, in Chinese; for Google/Google Scholar only):
- Purpose: same as PRECISION but in Chinese, to retrieve Chinese-language results from Google/Scholar.
- One precision-style phrase in Chinese (about 6-12 words), with core concepts and optional time/method constraints.
- Example: "深海贻贝 共生体 免疫 转录组 2020-2024"
"""

_PRECISION_ZH_OUTPUT_FORMAT = """
PRECISION_ZH:
<single line query in Chinese>"""

_PRECISION_ZH_RULE = " When the user question is in Chinese, you MUST also output PRECISION_ZH (one precision query in Chinese)."


def _is_chinese(text: str) -> bool:
    """True if text contains at least one CJK character."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def generate_structured_queries_1plus1plus1(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Generate 1 recall + 1 precision + 1 discovery using chat_generate_queries prompt.
    When query is Chinese, also requests optional PRECISION_ZH (routed only to Google/Scholar).
    Returns {"recall": str, "precision": str, "discovery": str, "precision_zh"?: str} or None on parse failure.
    """
    if not (query or "").strip():
        return None
    query_stripped = (query or "").strip()
    query_is_chinese = _is_chinese(query_stripped)
    precision_zh_instruction = _PRECISION_ZH_INSTRUCTION if query_is_chinese else ""
    precision_zh_output_format = _PRECISION_ZH_OUTPUT_FORMAT if query_is_chinese else ""
    precision_zh_rule = _PRECISION_ZH_RULE if query_is_chinese else ""

    prompt = _pm.render(
        "chat_generate_queries.txt",
        query=query_stripped,
        evidence_context=(evidence_context or "").strip() or "(none)",
        precision_zh_instruction=precision_zh_instruction,
        precision_zh_output_format=precision_zh_output_format,
        precision_zh_rule=precision_zh_rule,
    )
    try:
        resp = llm_client.chat(
            [{"role": "user", "content": prompt}],
            model=model_override or None,
        )
        text = (resp.get("final_text") or "").strip()
        if not text:
            return None
        out: Dict[str, str] = {}
        for block in ("RECALL", "PRECISION", "DISCOVERY"):
            marker = block + ":"
            pos = text.find(marker)
            if pos == -1:
                return None
            start = pos + len(marker)
            rest = text[start:].lstrip()
            next_marker = "RECALL:" if block == "DISCOVERY" else ("PRECISION:" if block == "RECALL" else "DISCOVERY:")
            end = rest.find(next_marker)
            line = (rest[:end].strip() if end != -1 else rest.strip()).split("\n")[0].strip()
            if not line:
                return None
            out[block.lower()] = line
        # Optional PRECISION_ZH when we requested it (Chinese query)
        if query_is_chinese and "PRECISION_ZH:" in text:
            pzh_marker = "PRECISION_ZH:"
            pzh_pos = text.find(pzh_marker)
            if pzh_pos != -1:
                pzh_start = pzh_pos + len(pzh_marker)
                pzh_rest = text[pzh_start:].lstrip()
                pzh_line = pzh_rest.split("\n")[0].strip() if pzh_rest else ""
                if pzh_line:
                    out["precision_zh"] = pzh_line
        return out
    except Exception:
        return None


def web_queries_per_provider_from_1plus1plus1(
    structured: Dict[str, str],
    web_providers: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Build web_queries_per_provider: recall+precision for ncbi/semantic/scholar/google, discovery for tavily.
    When precision_zh is present, it is appended only for scholar and google."""
    providers = (web_providers or []) or []
    recall = (structured.get("recall") or "").strip()
    precision = (structured.get("precision") or "").strip()
    discovery = (structured.get("discovery") or "").strip()
    precision_zh = (structured.get("precision_zh") or "").strip()
    if not recall:
        return {}
    qpp: Dict[str, List[str]] = {}
    for p in providers:
        p_lower = (p or "").strip().lower()
        if p_lower in ("ncbi", "semantic"):
            qpp[p] = [q for q in [recall, precision] if q]
            if not qpp[p]:
                qpp[p] = [recall]
        elif p_lower in ("scholar", "google"):
            base = [q for q in [recall, precision] if q]
            if not base:
                base = [recall]
            if precision_zh:
                base = base + [precision_zh]
            qpp[p] = base
        elif p_lower == "tavily":
            qpp[p] = [discovery] if discovery else [recall]
        else:
            qpp[p] = [recall]
    return qpp
