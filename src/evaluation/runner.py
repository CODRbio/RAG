"""
评测执行器：检索/生成/引用质量统计
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from src.collaboration.citation.manager import CiteKeyGenerator, chunk_to_citation
from src.evaluation.dataset import EvalCase
from src.evaluation.metrics import (
    extract_citations,
    recall_at_k,
    rouge_l_f1,
    safe_mean,
    token_f1,
)
from src.retrieval.evidence import EvidencePack
from src.retrieval.service import RetrievalService
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


def evaluate_dataset(
    cases: List[EvalCase],
    retrieval: RetrievalService,
    llm_client: Any | None = None,
    mode_override: Optional[str] = None,
    top_k: int = 10,
    max_context_chunks: int = 8,
    max_context_chars: int = 800,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    results = []
    for case in cases:
        result = evaluate_case(
            case=case,
            retrieval=retrieval,
            llm_client=llm_client,
            mode_override=mode_override,
            top_k=top_k,
            max_context_chunks=max_context_chunks,
            max_context_chars=max_context_chars,
            model_override=model_override,
        )
        results.append(result)

    summary = _aggregate_results(results)
    return {"summary": summary, "results": results}


def evaluate_case(
    case: EvalCase,
    retrieval: RetrievalService,
    llm_client: Any | None = None,
    mode_override: Optional[str] = None,
    top_k: int = 10,
    max_context_chunks: int = 8,
    max_context_chars: int = 800,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    mode = mode_override or case.mode or "local"
    pack = retrieval.search(case.query, mode=mode, top_k=top_k)

    retrieved_doc_ids = _unique_doc_ids(pack)
    recall, hit = recall_at_k(retrieved_doc_ids, case.expected_doc_ids, top_k)

    retrieval_metrics = {
        "recall_at_k": recall,
        "hit_at_k": hit,
        "top_k": top_k,
        "retrieval_time_ms": pack.retrieval_time_ms,
        "total_candidates": pack.total_candidates,
        "sources_used": pack.sources_used,
        "retrieved_doc_ids": retrieved_doc_ids[:top_k],
    }

    generation_metrics = None
    citation_metrics = None
    answer = None

    if llm_client and case.reference_answer:
        context, cite_keys = _build_context(pack, max_context_chunks, max_context_chars)
        answer = _generate_answer(
            llm_client=llm_client,
            query=case.query,
            context=context,
            cite_keys=cite_keys,
            model_override=model_override,
        )
        generation_metrics = _score_generation(answer, case.reference_answer)
        citation_metrics = _score_citations(answer, cite_keys, case.expected_citations)

    return {
        "case": asdict(case),
        "mode": mode,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "citation": citation_metrics,
        "answer": answer,
    }


def _unique_doc_ids(pack: EvidencePack) -> List[str]:
    out = []
    seen = set()
    for chunk in pack.chunks:
        doc_id = (chunk.doc_id or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _build_context(
    pack: EvidencePack,
    max_chunks: int,
    max_chars: int,
) -> Tuple[str, List[str]]:
    gen = CiteKeyGenerator()
    lines = []
    keys: List[str] = []
    for chunk in pack.chunks[:max_chunks]:
        citation = chunk_to_citation(chunk, generator=gen)
        cite_key = citation.cite_key or citation.id
        title = citation.title or chunk.doc_id or chunk.chunk_id
        header = f"[{cite_key}] {title}"
        text = (chunk.text or "").strip()
        if not text:
            continue
        if max_chars > 0:
            text = text[:max_chars]
        lines.append(f"{header}\n{text}")
        keys.append(cite_key)
    return "\n\n".join(lines), keys


def _generate_answer(
    llm_client: Any,
    query: str,
    context: str,
    cite_keys: List[str],
    model_override: Optional[str] = None,
) -> str:
    keys_str = ", ".join(cite_keys[:30])
    messages = [
        {"role": "system", "content": _pm.render("evaluation_system.txt")},
        {"role": "user", "content": _pm.render("evaluation_user.txt", context=context, keys_str=keys_str, query=query)},
    ]
    resp = llm_client.chat(messages, model=model_override, max_tokens=800)
    return (resp.get("final_text") or "").strip()


def _score_generation(answer: str, reference: str) -> Dict[str, float]:
    p1, r1, f1 = token_f1(answer, reference)
    p2, r2, f2 = rouge_l_f1(answer, reference)
    return {
        "token_precision": p1,
        "token_recall": r1,
        "token_f1": f1,
        "rouge_l_precision": p2,
        "rouge_l_recall": r2,
        "rouge_l_f1": f2,
    }


def _score_citations(
    answer: str,
    allowed_cite_keys: List[str],
    expected_citations: List[str],
) -> Dict[str, float | int | None]:
    found = extract_citations(answer)
    if not found:
        return {
            "citations_found": 0,
            "valid_ratio": 0.0,
            "precision": 0.0,
            "recall": 0.0 if expected_citations else None,
        }
    allowed = set(allowed_cite_keys)
    valid = [c for c in found if c in allowed]
    precision = len(valid) / max(1, len(found))
    recall = None
    if expected_citations:
        expected = set(expected_citations)
        recall = len(set(found) & expected) / max(1, len(expected))
    return {
        "citations_found": len(found),
        "valid_ratio": precision,
        "precision": precision,
        "recall": recall,
    }


def _aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval_recalls = []
    retrieval_hits = []
    token_f1s = []
    rouge_f1s = []
    citation_precisions = []
    citation_recalls = []

    for r in results:
        retrieval = r.get("retrieval") or {}
        retrieval_recalls.append(retrieval.get("recall_at_k"))
        retrieval_hits.append(retrieval.get("hit_at_k"))

        gen = r.get("generation") or {}
        token_f1s.append(gen.get("token_f1"))
        rouge_f1s.append(gen.get("rouge_l_f1"))

        cite = r.get("citation") or {}
        citation_precisions.append(cite.get("precision"))
        citation_recalls.append(cite.get("recall"))

    return {
        "retrieval_recall_at_k_mean": safe_mean(retrieval_recalls),
        "retrieval_hit_at_k_mean": safe_mean(retrieval_hits),
        "token_f1_mean": safe_mean(token_f1s),
        "rouge_l_f1_mean": safe_mean(rouge_f1s),
        "citation_precision_mean": safe_mean(citation_precisions),
        "citation_recall_mean": safe_mean(citation_recalls),
        "case_count": len(results),
    }
