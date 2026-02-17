"""
检索证据包 - 统一检索层输出格式

EvidenceChunk / EvidencePack 确保上层可追溯，支持 to_context_string() 生成 LLM 上下文。
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional

# chunk 引用哈希的长度（SHA-256 前 N 位十六进制）
REF_HASH_LENGTH = 8


@dataclass
class EvidenceChunk:
    """单个证据块"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_type: Literal["dense", "sparse", "graph", "web"]

    doc_title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    evidence_type: Optional[str] = None  # finding | method | interpretation | background | summary

    @property
    def ref_hash(self) -> str:
        """
        8 位 SHA-256 哈希，用作 LLM 上下文中的稳定引用标记。
        后续 resolve_response_citations() 会将其替换为正式的 cite_key。
        """
        return hashlib.sha256(self.chunk_id.encode("utf-8")).hexdigest()[:REF_HASH_LENGTH]

    @property
    def doc_group_key(self) -> str:
        """用于按文档分组的 key：Web 来源用 url，本地来源用 doc_id。"""
        if self.url:
            return self.url.strip()
        return (self.doc_id or self.chunk_id or "").strip()


@dataclass
class EvidencePack:
    """检索结果的统一封装"""

    query: str
    chunks: List[EvidenceChunk]

    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    synthesis_meta: Optional[dict] = None  # 由 EvidenceSynthesizer 填充
    diagnostics: Optional[dict] = None  # 检索诊断信息（各阶段计数+耗时）

    def get_chunks_by_source(self, source_type: str) -> List[EvidenceChunk]:
        return [c for c in self.chunks if c.source_type == source_type]

    def to_context_string(self, max_chunks: int = 10) -> str:
        """生成 LLM 可用的上下文字符串（使用 ref_hash 作为引用标记）"""
        lines = []
        for chunk in self.chunks[:max_chunks]:
            ref = f"[{chunk.ref_hash}]"
            lines.append(f"{ref} {chunk.text}")
        return "\n\n".join(lines)
