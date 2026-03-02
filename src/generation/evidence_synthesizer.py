"""
证据综合模块 (Evidence Synthesizer)

在检索和生成之间插入结构化的证据综合层：
- 时间线排序：按发表年份排序，标注时间标签
- 来源分组：区分本地文献 vs 网络来源，标注交叉验证
- 证据强度分级：从 section_title 推断 finding/method/interpretation/background/summary
- 矛盾/一致性标注：检测同一 doc_id 的多来源覆盖

核心原则：不额外调用 LLM，通过规则化证据组织 + 增强 system prompt 让 LLM 自身完成综合。
零延迟增加、零额外 token 消耗。

引用流程：
  1. 上下文中每条证据用 chunk 的 ref_hash（格式 ref:xxxxxxxx）作为引用标记，在文本中呈现为 [ref:a1b2c3d4]
  2. LLM 在回答中使用 [ref:xxxx] 引用；`ref:` 命名空间前缀防止误匹配领域文本（内存地址、Git commit 等）
  3. 生成后由 citation/manager.resolve_response_citations() 将 [ref:xxxx] 替换为
     正式 cite_key（numbered / author_date / hash 格式），同时输出文档级引文列表

使用方法:
---------
from src.generation.evidence_synthesizer import EvidenceSynthesizer

synthesizer = EvidenceSynthesizer()
context_str, synthesis_meta = synthesizer.synthesize(pack)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.evidence import EvidenceChunk, EvidencePack
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


# ============================================================
# 证据类型分类规则
# ============================================================

# section_title 关键词 → 证据类型映射
_EVIDENCE_TYPE_RULES: List[Tuple[List[str], str]] = [
    # finding（最强）：实验结果、数据
    (
        ["result", "finding", "data", "observation", "outcome",
         "analysis", "statistical", "measurement"],
        "finding",
    ),
    # method：方法论、材料
    (
        ["method", "material", "procedure", "protocol", "experiment",
         "sample", "sampling", "technique", "approach", "design"],
        "method",
    ),
    # interpretation：讨论、结论
    (
        ["discussion", "conclusion", "implication", "interpretation",
         "perspective", "significance", "limitation", "future"],
        "interpretation",
    ),
    # background：引言、背景
    (
        ["introduction", "background", "overview", "context",
         "literature", "review", "state of the art", "related work"],
        "background",
    ),
    # summary：摘要
    (
        ["abstract", "summary", "highlight", "graphical abstract"],
        "summary",
    ),
]

# 证据类型展示名（中文）
_EVIDENCE_TYPE_LABELS = {
    "finding": "实验发现",
    "method": "方法描述",
    "interpretation": "讨论解读",
    "background": "背景信息",
    "summary": "摘要概述",
}

# 证据强度排序（用于排序，越小越强）
_EVIDENCE_STRENGTH_ORDER = {
    "finding": 0,
    "method": 1,
    "interpretation": 2,
    "summary": 3,
    "background": 4,
}

# 本地来源类型
_LOCAL_SOURCES = {"dense", "sparse", "graph"}

# 定量数据正则：p-value, n=, mean±SD, CI, OR/HR/RR, r/R², fold-change, 浓度单位等
_QUANT_PATTERNS = re.compile(
    r"(?:"
    r"p\s*[<>=≤≥]\s*0?\.\d+"  # p-value
    r"|p\s*=\s*0?\.\d+"
    r"|(?:n|N)\s*=\s*\d+"  # sample size
    r"|\d+\.?\d*\s*[±\+\-]\s*\d+\.?\d*"  # mean ± SD
    r"|\d+%\s*(?:CI|confidence\s*interval)"  # confidence interval
    r"|(?:OR|HR|RR|AOR)\s*[=:]\s*\d+\.?\d*"  # odds/hazard/risk ratio
    r"|r\s*=\s*-?0?\.\d+"  # correlation
    r"|R[²2]\s*=\s*0?\.\d+"  # R-squared
    r"|fold[- ]?change"
    r"|\d+\.?\d*\s*(?:mg|μg|µg|ng|mmol|μmol|μmol|mL|μL|µL)\s*/\s*[Ll]"  # concentrations
    r"|(?:mean|median)\s*[\(\[]?\s*\d+"  # mean/median (value)
    r"|IQR\s*[\(\[]?\s*\d+"  # IQR
    r")",
    re.IGNORECASE,
)

# Snippet context window (chars before/after match)
_QUANT_SNIPPET_CONTEXT = 30


def _has_quantitative_data(text: str) -> bool:
    """Return True if text contains common scientific quantitative expressions."""
    return bool(_QUANT_PATTERNS.search(text or ""))


def _extract_quant_snippets(text: str) -> List[str]:
    """Extract matching quantitative substrings with surrounding context (~30 chars)."""
    if not text:
        return []
    snippets: List[str] = []
    for m in _QUANT_PATTERNS.finditer(text):
        start = max(0, m.start() - _QUANT_SNIPPET_CONTEXT)
        end = min(len(text), m.end() + _QUANT_SNIPPET_CONTEXT)
        snip = text[start:end].strip()
        if snip and snip not in snippets:
            snippets.append(snip)
    return snippets


def _classify_quant_snippet(snippet: str) -> str:
    """Classify a quantitative snippet into a category for grouping."""
    lower = snippet.lower()
    if re.search(r"\b(?:n|N)\s*=\s*\d+", snippet):
        return "sample_size"
    if re.search(r"p\s*[<>=≤≥]|p\s*=\s*0?\.\d+", lower):
        return "p_value"
    if re.search(r"(?:OR|HR|RR|AOR)\s*[=:]|effect\s*size|cohen", lower):
        return "effect_size"
    if re.search(r"\d+\.?\d*\s*(?:mg|μg|µg|ng|mmol|μmol|µmol|mL|μL|µL)\s*/\s*[Ll]", lower):
        return "concentration"
    if re.search(r"r\s*=\s*-?0?\.\d+|R[²2]\s*=\s*0?\.\d+|correlation", lower):
        return "correlation"
    if re.search(r"\d+%\s*(?:CI|confidence)", lower):
        return "confidence_interval"
    if re.search(r"[±\+\-]\s*\d+\.?\d*|mean|median|IQR", lower):
        return "mean_sd_median"
    return "other"


def _build_quant_summary(
    sorted_chunks: List[EvidenceChunk],
    cross_validated_docs: set,
) -> str:
    """
    Build a structured block of key quantitative data points from chunks,
    grouped by variable type, with ref tags for citation.
    """
    # (category -> list of "(year, ref, snippet)")
    by_category: Dict[str, List[Tuple[Optional[int], str, str]]] = defaultdict(list)

    for chunk in sorted_chunks:
        text = chunk.text or ""
        if not _has_quantitative_data(text):
            continue
        ref = chunk.ref_hash
        year = chunk.year
        for snip in _extract_quant_snippets(text):
            cat = _classify_quant_snippet(snip)
            by_category[cat].append((year, ref, snip))

    if not by_category:
        return ""

    _CAT_LABELS: Dict[str, str] = {
        "sample_size": "Sample sizes",
        "p_value": "Statistical significance",
        "effect_size": "Effect sizes",
        "concentration": "Concentrations",
        "correlation": "Correlations",
        "confidence_interval": "Confidence intervals",
        "mean_sd_median": "Means / medians",
        "other": "Other quantitative",
    }
    order = [
        "sample_size",
        "p_value",
        "effect_size",
        "confidence_interval",
        "mean_sd_median",
        "concentration",
        "correlation",
        "other",
    ]
    lines = ["=== Key Quantitative Data Points ==="]
    for cat in order:
        items = by_category.get(cat, [])
        if not items:
            continue
        # Dedupe by snippet text, keep first (year, ref)
        seen: Dict[str, Tuple[Optional[int], str]] = {}
        for y, r, snip in items:
            key = snip[:80].strip()
            if key not in seen:
                seen[key] = (y, r)
        parts = []
        for snip_short, (y, r) in seen.items():
            year_ref = f"{y}, [{r}]" if y is not None else f"[{r}]"
            parts.append(f"{snip_short} ({year_ref})")
        label = _CAT_LABELS.get(cat, cat)
        lines.append(f"{label}: " + "; ".join(parts[:5]))  # cap 5 per category
    return "\n".join(lines)


def classify_evidence_type(section_title: Optional[str]) -> str:
    """
    从 section_title 推断证据类型。

    Args:
        section_title: 章节路径/标题（如 "Results > Table 3"）

    Returns:
        证据类型: finding | method | interpretation | background | summary
    """
    if not section_title:
        return "background"

    lower = section_title.lower()
    for keywords, etype in _EVIDENCE_TYPE_RULES:
        for kw in keywords:
            if kw in lower:
                return etype
    return "background"


# ============================================================
# SynthesisMeta
# ============================================================

@dataclass
class SynthesisMeta:
    """证据综合元数据"""

    year_range: Tuple[Optional[int], Optional[int]] = (None, None)
    source_breakdown: Dict[str, int] = field(default_factory=dict)  # chunk 级：每个 chunk 算一次
    unique_source_breakdown: Dict[str, int] = field(default_factory=dict)  # 来源级：同 URL/doc_id 只算一次
    evidence_type_breakdown: Dict[str, int] = field(default_factory=dict)
    cross_validated_count: int = 0
    total_documents: int = 0
    quant_chunk_count: int = 0  # chunks containing quantitative data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year_range": list(self.year_range),
            "source_breakdown": self.source_breakdown,
            "unique_source_breakdown": self.unique_source_breakdown,
            "evidence_type_breakdown": self.evidence_type_breakdown,
            "cross_validated_count": self.cross_validated_count,
            "total_documents": self.total_documents,
            "quant_chunk_count": self.quant_chunk_count,
        }


# ============================================================
# EvidenceSynthesizer
# ============================================================

class EvidenceSynthesizer:
    """
    证据综合器。

    对 EvidencePack 做结构化综合：时间线排序、来源分组、证据强度分级、
    交叉验证标注。输出增强的 context string + 综合元数据。
    """

    def __init__(self, max_chunks: int = 10):
        self.max_chunks = max_chunks

    def synthesize(
        self,
        pack: EvidencePack,
        max_chunks: Optional[int] = None,
    ) -> Tuple[str, SynthesisMeta]:
        """
        对 EvidencePack 做结构化综合。

        Args:
            pack: 检索结果包
            max_chunks: 最大使用的 chunk 数量

        Returns:
            (context_string, synthesis_meta)
        """
        limit = max_chunks or self.max_chunks
        chunks = pack.chunks[:limit]

        if not chunks:
            return "（本轮暂无检索结果）", SynthesisMeta()

        # 1. 为每个 chunk 分类证据类型
        for chunk in chunks:
            if not chunk.evidence_type:
                chunk.evidence_type = classify_evidence_type(chunk.section_title)

        # 2. 计算元数据
        meta = self._compute_meta(chunks)

        # 3. 检测交叉验证的 doc_id
        cross_validated_docs = self._find_cross_validated(chunks)
        meta.cross_validated_count = len(cross_validated_docs)

        # 4. 排序：先按年份（升序），同年按证据强度（finding 优先）
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                c.year or 9999,
                _EVIDENCE_STRENGTH_ORDER.get(c.evidence_type or "background", 4),
            ),
        )

        # 5. 构建 context string（使用 [ref:xxxx] 作为引用标记）
        context = self._build_context(sorted_chunks, meta, cross_validated_docs)

        # 6. 回写到 pack
        pack.synthesis_meta = meta.to_dict()

        return context, meta

    def _compute_meta(self, chunks: List[EvidenceChunk]) -> SynthesisMeta:
        """计算综合元数据（chunk 级 + 来源级双层统计）"""
        years = [c.year for c in chunks if c.year is not None]
        year_range = (min(years), max(years)) if years else (None, None)

        # chunk 级：每个 chunk 算一次（同文档多个 chunk 分别计数）
        chunk_counts: Dict[str, int] = defaultdict(int)
        # 来源级：同 doc_group_key（URL/doc_id）只算一次
        source_docs: Dict[str, set] = defaultdict(set)

        for c in chunks:
            prov = getattr(c, "provider", None)
            if not prov:
                prov = "local" if c.source_type in _LOCAL_SOURCES else "web"
            chunk_counts[prov] += 1
            source_docs[prov].add(c.doc_group_key)

        unique_source_counts = {k: len(v) for k, v in source_docs.items()}

        # 证据类型统计
        type_counts: Dict[str, int] = defaultdict(int)
        for c in chunks:
            type_counts[c.evidence_type or "background"] += 1

        # 独立文献数
        doc_keys = {c.doc_group_key for c in chunks}

        # 含定量数据的 chunk 数
        quant_chunk_count = sum(1 for c in chunks if _has_quantitative_data(c.text or ""))

        return SynthesisMeta(
            year_range=year_range,
            source_breakdown=dict(chunk_counts),
            unique_source_breakdown=unique_source_counts,
            evidence_type_breakdown=dict(type_counts),
            cross_validated_count=0,
            total_documents=len(doc_keys),
            quant_chunk_count=quant_chunk_count,
        )

    def _find_cross_validated(self, chunks: List[EvidenceChunk]) -> set:
        """
        检测本地+网络双重覆盖的 doc_id。

        当同一 doc_id 同时有 local 和 web 来源的 chunk 时，标记为交叉验证。
        """
        doc_sources: Dict[str, set] = defaultdict(set)
        for c in chunks:
            if not c.doc_id:
                continue
            key = "local" if c.source_type in _LOCAL_SOURCES else "web"
            doc_sources[c.doc_id].add(key)

        return {
            doc_id
            for doc_id, sources in doc_sources.items()
            if len(sources) > 1
        }

    def _build_context(
        self,
        sorted_chunks: List[EvidenceChunk],
        meta: SynthesisMeta,
        cross_validated_docs: set,
    ) -> str:
        """构建结构化 context string"""
        parts = []

        # === Header: 综合概览 ===
        parts.append(self._build_header(meta))

        # === Quantitative summary (when present) ===
        quant_summary = _build_quant_summary(sorted_chunks, cross_validated_docs)
        if quant_summary:
            parts.append("\n" + quant_summary)

        # === Body: 按时间排序的证据 ===
        parts.append("\n=== 证据（按时间排序）===\n")

        for chunk in sorted_chunks:
            entry = self._format_chunk(chunk, cross_validated_docs)
            if entry:
                parts.append(entry)

        return "\n".join(parts)

    def _build_header(self, meta: SynthesisMeta) -> str:
        """构建综合概览 header"""
        lines = ["=== 证据综合 ==="]

        # 时间跨度
        y_min, y_max = meta.year_range
        if y_min and y_max:
            if y_min == y_max:
                lines.append(f"时间跨度: {y_min} ({meta.total_documents}篇文献)")
            else:
                lines.append(f"时间跨度: {y_min}–{y_max} ({meta.total_documents}篇文献)")
        elif meta.total_documents:
            lines.append(f"涉及文献: {meta.total_documents}篇")

        _PROVIDER_LABELS = {
            "local": "本地", "tavily": "Tavily", "google": "Google",
            "scholar": "Google Scholar", "semantic": "Semantic Scholar",
            "ncbi": "NCBI PubMed", "web": "网络",
        }
        source_parts = []
        for prov, cnt in sorted(meta.source_breakdown.items(), key=lambda x: -x[1]):
            label = _PROVIDER_LABELS.get(prov, prov)
            n_sources = meta.unique_source_breakdown.get(prov, 0)
            if n_sources and n_sources != cnt:
                source_parts.append(f"{label} {n_sources} 篇/{cnt} 条")
            else:
                source_parts.append(f"{label} {cnt} 条")
        if meta.cross_validated_count:
            source_parts.append(f"交叉验证 {meta.cross_validated_count} 条")
        if source_parts:
            lines.append(f"来源构成: {' | '.join(source_parts)}")

        # 证据类型
        type_parts = []
        for etype in ["finding", "method", "interpretation", "background", "summary"]:
            count = meta.evidence_type_breakdown.get(etype, 0)
            if count:
                label = _EVIDENCE_TYPE_LABELS.get(etype, etype)
                type_parts.append(f"{label} {count}")
        if type_parts:
            lines.append(f"证据类型: {' | '.join(type_parts)}")

        return "\n".join(lines)

    def _format_chunk(
        self,
        chunk: EvidenceChunk,
        cross_validated_docs: set,
    ) -> str:
        """格式化单个 chunk，使用 ref_hash 作为引用标记。"""
        text = (chunk.text or "").strip()
        if not text:
            return ""

        # 标签部分
        tags = []

        # 年份
        if chunk.year:
            tags.append(str(chunk.year))

        # 证据类型
        etype = chunk.evidence_type or "background"
        tags.append(etype)

        # 来源
        is_local = chunk.source_type in _LOCAL_SOURCES
        source_label = "local" if is_local else "web"

        # 交叉验证标记
        if chunk.doc_id and chunk.doc_id in cross_validated_docs:
            source_label = "local+web ✓"

        tags.append(source_label)

        # 定量数据标记
        if _has_quantitative_data(text):
            tags.append("Q")

        tag_str = " | ".join(tags)

        # 引用标记：使用带命名空间前缀的稳定哈希（格式 [ref:xxxxxxxx]）
        ref = chunk.ref_hash

        return f"[{tag_str}] [{ref}]\n{text}"


# ============================================================
# 增强 system prompt
# ============================================================

SYNTHESIS_SYSTEM_PROMPT = _pm.render("evidence_synthesis_system.txt")


def build_synthesis_system_prompt(context: str) -> str:
    """构建包含综合证据的 system prompt"""
    return SYNTHESIS_SYSTEM_PROMPT + (context or "（本轮暂无检索结果）")
