"""
Claim Extractor — 从解析后的论文中提取核心科学声明

从 Abstract / Results / Conclusion 等关键章节提取 3-5 个核心 claims。
每个 claim 包含：声明文本、支持证据、方法、置信度、局限性。

Claims 存入 enriched.json 的 `claims` 字段，chunking 时注入 chunk.meta。

使用方法:
---------
from src.parser.claim_extractor import ClaimExtractor

extractor = ClaimExtractor()
claims = extractor.extract(enriched_doc, llm_client)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.log import get_logger

logger = get_logger(__name__)


# ============================================================
# Pydantic Response Models (结构化输出)
# ============================================================

class _ClaimItem(BaseModel):
    text: str = ""
    evidence: str = ""
    methodology: str = ""
    confidence: str = "medium"
    limitations: str = ""
    source_section: str = ""


class _ClaimListResponse(BaseModel):
    claims: List[_ClaimItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_array(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"claims": data}
        return data


# ============================================================
# Claim 数据结构
# ============================================================

@dataclass
class Claim:
    """从论文中提取的核心科学声明"""

    claim_id: str
    text: str  # 核心声明（1-2 句）
    evidence: str  # 支持证据摘要
    methodology: str  # 使用的方法
    confidence: str  # high | medium | low
    limitations: str  # 局限性
    source_section: str  # 来源章节
    source_block_ids: List[str] = field(default_factory=list)  # 关联的 block_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "evidence": self.evidence,
            "methodology": self.methodology,
            "confidence": self.confidence,
            "limitations": self.limitations,
            "source_section": self.source_section,
            "source_block_ids": self.source_block_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Claim":
        return cls(
            claim_id=d.get("claim_id", str(uuid.uuid4())[:8]),
            text=d.get("text", ""),
            evidence=d.get("evidence", ""),
            methodology=d.get("methodology", ""),
            confidence=d.get("confidence", "medium"),
            limitations=d.get("limitations", ""),
            source_section=d.get("source_section", ""),
            source_block_ids=d.get("source_block_ids", []),
        )


# ============================================================
# 关键章节识别
# ============================================================

_KEY_SECTION_PATTERNS = {
    "abstract": [r"abstract", r"摘要"],
    "results": [r"result", r"finding", r"outcome", r"data", r"observation"],
    "conclusion": [r"conclusion", r"summary", r"concluding"],
    "discussion": [r"discussion", r"implication"],
}


def _is_key_section(heading_path: List[str], target_sections: List[str]) -> Optional[str]:
    """判断 heading_path 是否属于关键章节，返回匹配的章节类型"""
    path_str = " > ".join(str(h) for h in heading_path).lower()
    for section_type in target_sections:
        for pattern in _KEY_SECTION_PATTERNS.get(section_type, []):
            if re.search(pattern, path_str, re.IGNORECASE):
                return section_type
    return None


# ============================================================
# LLM Prompt
# ============================================================

_CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extraction expert.

Given the following text from a research paper, extract 3-5 **core scientific claims**.

For each claim, provide:
- **text**: The core claim statement (1-2 sentences, precise and falsifiable)
- **evidence**: Brief summary of supporting evidence from the paper
- **methodology**: What method/approach was used
- **confidence**: "high" (directly supported by data), "medium" (inferred), or "low" (speculative)
- **limitations**: Any caveats or limitations mentioned
- **source_section**: Which section this claim comes from (abstract/results/conclusion/discussion)

Return ONLY a JSON object:
{"claims": [{"text": "...", "evidence": "...", "methodology": "...", "confidence": "high|medium|low", "limitations": "...", "source_section": "..."}]}

Paper text:
{text}"""


# ============================================================
# ClaimExtractor
# ============================================================

class ClaimExtractor:
    """
    从 EnrichedDoc 提取核心 claims。

    使用 LLM 从 Abstract + Results + Conclusion 提取 3-5 个核心 claims。
    """

    def __init__(
        self,
        max_text_chars: int = 6000,
        target_sections: Optional[List[str]] = None,
        max_retries: int = 2,
    ):
        self.max_text_chars = max_text_chars
        self.target_sections = target_sections or ["abstract", "results", "conclusion", "discussion"]
        self.max_retries = max_retries

    def extract(self, doc: Any, llm_client: Any) -> List[Claim]:
        """
        从 EnrichedDoc 提取 claims。

        Args:
            doc: EnrichedDoc 实例（需要 content_flow 和 doc_id）
            llm_client: LLM 客户端（需要 .chat() 方法）

        Returns:
            List[Claim]
        """
        # 1. 提取关键章节文本
        section_texts, block_ids_map = self._extract_key_sections(doc)

        if not section_texts:
            logger.warning(f"[{getattr(doc, 'doc_id', '?')}] 未找到关键章节，跳过 claim 提取")
            return []

        # 2. 组装文本（截断到 max_text_chars）
        combined = self._combine_sections(section_texts)

        # 3. 调用 LLM
        raw_claims = self._call_llm(combined, llm_client)

        if not raw_claims:
            logger.warning(f"[{getattr(doc, 'doc_id', '?')}] LLM 未返回有效 claims")
            return []

        # 4. 构建 Claim 对象
        claims = []
        for i, raw in enumerate(raw_claims):
            section = raw.get("source_section", "")
            claim = Claim(
                claim_id=f"{getattr(doc, 'doc_id', 'doc')}_{i:02d}",
                text=raw.get("text", ""),
                evidence=raw.get("evidence", ""),
                methodology=raw.get("methodology", ""),
                confidence=raw.get("confidence", "medium"),
                limitations=raw.get("limitations", ""),
                source_section=section,
                source_block_ids=block_ids_map.get(section, []),
            )
            if claim.text.strip():
                claims.append(claim)

        logger.info(f"[{getattr(doc, 'doc_id', '?')}] 提取 {len(claims)} 个 claims")
        return claims

    def _extract_key_sections(self, doc: Any) -> tuple:
        """
        从 content_flow 提取关键章节文本。

        Returns:
            (section_texts: dict[str, str], block_ids_map: dict[str, list[str]])
        """
        section_texts: Dict[str, List[str]] = {}
        block_ids_map: Dict[str, List[str]] = {}

        blocks = getattr(doc, "content_flow", [])
        for block in blocks:
            hp = getattr(block, "heading_path", None) or []
            text = getattr(block, "text", None) or ""
            block_id = getattr(block, "block_id", None) or ""

            if not text.strip():
                continue

            matched = _is_key_section(hp, self.target_sections)
            if matched:
                section_texts.setdefault(matched, []).append(text)
                block_ids_map.setdefault(matched, []).append(block_id)

        # 转换为字符串
        return {k: "\n\n".join(v) for k, v in section_texts.items()}, block_ids_map

    def _combine_sections(self, section_texts: Dict[str, str]) -> str:
        """组装各章节文本，带标签，截断到 max_text_chars"""
        parts = []
        # 按优先级排序
        priority = ["abstract", "results", "conclusion", "discussion"]
        total = 0
        for section in priority:
            if section in section_texts:
                text = section_texts[section]
                label = f"[{section.upper()}]"
                remaining = self.max_text_chars - total
                if remaining <= 0:
                    break
                if len(text) > remaining:
                    text = text[:remaining] + "..."
                parts.append(f"{label}\n{text}")
                total += len(text)

        return "\n\n".join(parts)

    def _call_llm(self, text: str, llm_client: Any) -> Optional[List[Dict]]:
        """调用 LLM 提取 claims，使用 Pydantic 结构化输出保障解析稳定性"""
        prompt = _CLAIM_EXTRACTION_PROMPT.replace("{text}", text)

        for attempt in range(self.max_retries + 1):
            try:
                resp = llm_client.chat(
                    messages=[
                        {"role": "system", "content": "Extract scientific claims. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    response_model=_ClaimListResponse,
                )
                parsed: Optional[_ClaimListResponse] = resp.get("parsed_object")
                if parsed is None:
                    raw = (resp.get("final_text") or "").strip()
                    if raw:
                        parsed = _ClaimListResponse.model_validate_json(raw)
                if parsed is not None:
                    return [item.model_dump() for item in parsed.claims]
            except Exception as e:
                logger.warning(f"Claim extraction LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    return None

        return None
