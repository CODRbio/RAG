"""
Chain of Verification (CoV) — 对已生成内容执行事实验证。

流程:
1. 提取所有事实性声明
2. 检查每个声明是否有引文支撑
3. 对无支撑声明触发补充搜索
4. 标记置信度并生成修订建议
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.log import get_logger

logger = get_logger(__name__)


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


_EXTRACT_CLAIMS_PROMPT = """从以下文本中提取所有事实性声明（factual claims）。

只提取可验证的事实性陈述，忽略观点性表述和过渡句。

文本:
{text}

返回 JSON 数组，每项格式:
{{"claim": "声明文本", "has_citation": true/false, "citation_keys": ["key1"]}}

只返回 JSON 数组。"""


_VERIFY_CLAIMS_PROMPT = """请验证以下声明是否有证据支撑。

声明列表:
{claims}

可用的参考资料:
{evidence}

对每个声明判断：
- 是否有充分证据支撑
- 置信度 (high/medium/low)
- 如果不支撑，建议的修订或补充搜索方向

返回 JSON 数组:
[{{"claim_index": 0, "confidence": "high|medium|low", "evidence_found": "支撑证据概要", "needs_revision": false, "revision_note": "", "supplementary_query": ""}}]

只返回 JSON 数组。"""


def extract_claims(
    text: str,
    llm_client: Any,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从文本中提取事实性声明"""
    prompt = _EXTRACT_CLAIMS_PROMPT.format(text=text[:4000])
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "你是事实核查专家，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=2000,
        )
        raw = resp.get("final_text", "")
        # 提取 JSON 数组
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            return json.loads(match.group(0))
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
    prompt = _VERIFY_CLAIMS_PROMPT.format(claims=claims_text, evidence=citation_text[:3000])
    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "你是学术事实核查专家，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=2000,
        )
        raw = resp.get("final_text", "")
        match = re.search(r"\[[\s\S]*\]", raw)
        verifications = json.loads(match.group(0)) if match else []
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
