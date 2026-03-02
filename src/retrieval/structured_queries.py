"""
Shared 1+1+1 structured query generation for Chat and Research plan.

Used by: Chat Round 2, Research plan_node background retrieval.
"""

from typing import Any, Dict, List, Optional

from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


def generate_structured_queries_1plus1plus1(
    query: str,
    evidence_context: str,
    llm_client: Any,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Generate 1 recall + 1 precision + 1 discovery using chat_generate_queries prompt.
    Returns {"recall": str, "precision": str, "discovery": str} or None on parse failure.
    """
    if not (query or "").strip():
        return None
    prompt = _pm.render(
        "chat_generate_queries.txt",
        query=(query or "").strip(),
        evidence_context=(evidence_context or "").strip() or "(none)",
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
        return out
    except Exception:
        return None


def web_queries_per_provider_from_1plus1plus1(
    structured: Dict[str, str],
    web_providers: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Build web_queries_per_provider: recall+precision for ncbi/semantic/scholar/google, discovery for tavily."""
    providers = (web_providers or []) or []
    recall = (structured.get("recall") or "").strip()
    precision = (structured.get("precision") or "").strip()
    discovery = (structured.get("discovery") or "").strip()
    if not recall:
        return {}
    qpp: Dict[str, List[str]] = {}
    for p in providers:
        p_lower = (p or "").strip().lower()
        if p_lower in ("ncbi", "semantic", "scholar", "google"):
            qpp[p] = [q for q in [recall, precision] if q]
            if not qpp[p]:
                qpp[p] = [recall]
        elif p_lower == "tavily":
            qpp[p] = [discovery] if discovery else [recall]
        else:
            qpp[p] = [recall]
    return qpp
