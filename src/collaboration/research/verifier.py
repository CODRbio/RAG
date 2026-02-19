"""
Chain of Verification (CoV) — 对已生成内容执行事实验证。

流程:
1. 提取所有事实性声明
2. 检查每个声明是否有引文支撑
3. 对无支撑声明触发补充搜索
4. 标记置信度并生成修订建议
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.log import get_logger
from src.utils.prompt_manager import PromptManager

logger = get_logger(__name__)
_pm = PromptManager()


# ============================================================
# Pydantic Response Models (结构化输出)
# ============================================================

class _ExtractedClaimItem(BaseModel):
    claim: str = ""
    has_citation: bool = False
    citation_keys: List[str] = Field(default_factory=list)


class _ExtractedClaimsResponse(BaseModel):
    claims: List[_ExtractedClaimItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_array(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"claims": data}
        return data


class _VerificationItem(BaseModel):
    claim_index: int = 0
    confidence: str = "low"
    evidence_found: str = ""
    needs_revision: bool = False
    revision_note: str = ""
    attribution_analysis: str = ""
    conflict_notes: List[str] = Field(default_factory=list)
    supplementary_query: str = ""


class _VerificationsResponse(BaseModel):
    verifications: List[_VerificationItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_array(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"verifications": data}
        return data


@dataclass
class ClaimVerification:
    """单个声明的验证结果"""
    claim_text: str
    has_citation: bool = False
    citation_keys: List[str] = field(default_factory=list)
    confidence: str = "low"  # low | medium | high
    evidence_found: str = ""
    needs_revision: bool = False
    revision_note: str = ""
    attribution_analysis: str = ""


@dataclass
class VerificationResult:
    """整体验证结果"""
    total_claims: int = 0
    verified_claims: int = 0
    unsupported_claims: int = 0
    claims: List[ClaimVerification] = field(default_factory=list)
    overall_confidence: str = "low"
    revision_suggestions: List[str] = field(default_factory=list)
    supplementary_queries: List[str] = field(default_factory=list)
    conflict_notes: List[str] = field(default_factory=list)



def extract_claims(
    text: str,
    llm_client: Any,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从文本中提取事实性声明"""
    prompt = _pm.render("extract_claims.txt", text=text[:4000])
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "你是事实核查专家，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=2000,
            response_model=_ExtractedClaimsResponse,
        )
        parsed: Optional[_ExtractedClaimsResponse] = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                parsed = _ExtractedClaimsResponse.model_validate_json(raw)
        if parsed is not None:
            return [item.model_dump() for item in parsed.claims]
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
    return []


def verify_claims(
    section_text: str,
    citations: List[Any],
    llm_client: Any,
    model: Optional[str] = None,
    search_callback: Optional[Any] = None,
) -> VerificationResult:
    """
    对已生成内容执行 Chain of Verification。

    Args:
        section_text: 待验证的章节文本
        citations: 引用列表（用于匹配引文标记）
        llm_client: LLM 客户端
        model: 可选模型覆盖
        search_callback: 可选的补充搜索函数 (query) -> str

    Returns:
        VerificationResult
    """
    result = VerificationResult()

    # Step 1: 提取事实性声明
    raw_claims = extract_claims(section_text, llm_client, model)
    if not raw_claims:
        return result

    result.total_claims = len(raw_claims)

    # Step 2: 构建引文文本（供 LLM 验证）
    citation_text = ""
    if citations:
        lines = []
        for c in citations[:20]:
            if hasattr(c, "title"):
                lines.append(f"[{getattr(c, 'cite_key', '')}] {c.title} ({getattr(c, 'year', '')})")
            elif isinstance(c, dict):
                lines.append(f"[{c.get('cite_key', '')}] {c.get('title', '')} ({c.get('year', '')})")
        citation_text = "\n".join(lines)

    # 准备声明列表
    claims_text = "\n".join(
        f"{i}. {c.get('claim', '')}" + (f" [{', '.join(c.get('citation_keys', []))}]" if c.get('citation_keys') else "")
        for i, c in enumerate(raw_claims)
    )

    # Step 3: 用 LLM 验证
    prompt = _pm.render("verify_claims.txt", claims=claims_text, evidence=citation_text[:3000])
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "你是学术事实核查专家，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=2000,
            response_model=_VerificationsResponse,
        )
        parsed_verif: Optional[_VerificationsResponse] = resp.get("parsed_object")
        if parsed_verif is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                parsed_verif = _VerificationsResponse.model_validate_json(raw)
        verifications = (
            [item.model_dump() for item in parsed_verif.verifications]
            if parsed_verif is not None
            else []
        )
    except Exception as e:
        logger.warning(f"Claim verification failed: {e}")
        verifications = []

    # Step 4: 整合结果
    for i, raw_claim in enumerate(raw_claims):
        cv = ClaimVerification(
            claim_text=raw_claim.get("claim", ""),
            has_citation=bool(raw_claim.get("citation_keys")),
            citation_keys=raw_claim.get("citation_keys", []),
        )

        # 匹配验证结果
        v = next((v for v in verifications if v.get("claim_index") == i), None)
        if v:
            cv.confidence = v.get("confidence", "low")
            cv.evidence_found = v.get("evidence_found", "")
            cv.needs_revision = v.get("needs_revision", False)
            cv.revision_note = v.get("revision_note", "")
            cv.attribution_analysis = (v.get("attribution_analysis", "") or "").strip()

            raw_conflicts = v.get("conflict_notes", [])
            if isinstance(raw_conflicts, str):
                raw_conflicts = [raw_conflicts]
            conflict_items = [
                str(item).strip()
                for item in (raw_conflicts or [])
                if str(item).strip()
            ]
            if cv.attribution_analysis and not conflict_items:
                conflict_items = [cv.attribution_analysis]

            if conflict_items:
                if cv.attribution_analysis and cv.attribution_analysis not in cv.revision_note:
                    cv.revision_note = (
                        f"{cv.revision_note}\nAttribution Analysis: {cv.attribution_analysis}".strip()
                    )
                for note in conflict_items:
                    result.conflict_notes.append(
                        f"声明「{cv.claim_text[:50]}...」的冲突归因: {note}"
                    )

            sup_query = v.get("supplementary_query", "")
            if sup_query:
                result.supplementary_queries.append(sup_query)
        else:
            cv.confidence = "medium" if cv.has_citation else "low"
            cv.needs_revision = not cv.has_citation

        if cv.confidence == "high":
            result.verified_claims += 1
        elif cv.needs_revision:
            result.unsupported_claims += 1
            result.revision_suggestions.append(
                f"声明「{cv.claim_text[:50]}...」需要补充证据。{cv.revision_note}"
            )

        result.claims.append(cv)

    # 计算总体置信度
    if result.total_claims > 0:
        ratio = result.verified_claims / result.total_claims
        if ratio >= 0.8:
            result.overall_confidence = "high"
        elif ratio >= 0.5:
            result.overall_confidence = "medium"
        else:
            result.overall_confidence = "low"

    # Step 5: 补充搜索（如果有回调）
    if search_callback and result.supplementary_queries:
        for sq in result.supplementary_queries[:3]:
            try:
                search_callback(sq)
            except Exception:
                pass

    return result
