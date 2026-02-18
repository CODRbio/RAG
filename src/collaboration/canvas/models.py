"""
综述画布 - 数据模型（含大纲与草稿）

Citation / KnowledgeFragment / OutlineSection / DraftBlock / SurveyCanvas。
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional


@dataclass
class OutlineSection:
    """大纲章节"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    level: int = 1  # 1=章, 2=节, 3=小节
    order: int = 0
    parent_id: Optional[str] = None
    status: Literal["todo", "drafting", "done"] = "todo"
    guidance: Optional[str] = None


@dataclass
class DraftBlock:
    """草稿块（对应 OutlineSection）"""

    section_id: str = ""
    content_md: str = ""
    version: int = 1
    used_fragment_ids: List[str] = field(default_factory=list)
    used_citation_ids: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Citation:
    """引文记录"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doc_id: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    bibtex: Optional[str] = None
    cite_key: Optional[str] = None
    bbox: Optional[list[float]] = None
    page_num: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeFragment:
    """原子化知识片段（含来源追溯）"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    source_chunk_id: str = ""
    citation_id: str = ""
    linked_section_id: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Annotation:
    """行内批注（用户在 Refine 阶段对具体段落提出的修改意见）"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    section_id: str = ""           # 关联章节（可为空表示全局批注）
    target_text: str = ""          # 选中的原文片段
    directive: str = ""            # 用户的修改意见
    status: Literal["pending", "applied", "rejected"] = "pending"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchBrief:
    """研究简报 - Explore 阶段的结构化输出（对应 Survey Canvas 的 Goal/Hypothesis/Questions 等板块）"""

    scope: str = ""                          # 研究范围/边界
    success_criteria: List[str] = field(default_factory=list)  # 完成标准 / 假设
    key_questions: List[str] = field(default_factory=list)     # 核心问题
    exclusions: List[str] = field(default_factory=list)        # 明确排除的内容
    time_range: str = ""                     # 文献时间范围
    source_priority: List[str] = field(default_factory=list)   # 优先来源类型
    action_plan: str = ""                    # 行动计划：拿到数据后怎么用


@dataclass
class SurveyCanvas:
    """综述画布 - 含 topic / outline / drafts / citations / fragments"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    topic: str = ""
    working_title: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    stage: Literal["explore", "outline", "drafting", "refine"] = "explore"
    refined_markdown: str = ""  # 全文精炼稿（Refine 阶段可反复迭代）
    outline: List[OutlineSection] = field(default_factory=list)
    drafts: Dict[str, DraftBlock] = field(default_factory=dict)  # key: section_id
    citation_pool: Dict[str, Citation] = field(default_factory=dict)
    knowledge_pool: Dict[str, KnowledgeFragment] = field(default_factory=dict)
    identified_gaps: List[str] = field(default_factory=list)
    user_directives: List[str] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    research_brief: Optional[ResearchBrief] = None
    # 研究洞察：来自 Research Insights Ledger 的人类可读摘要
    research_insights: List[str] = field(default_factory=list)
    # 阶段跳过控制（Deep Research 流程中用户选择跳过的阶段）
    skip_draft_review: bool = False
    skip_refine_review: bool = False
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_section_by_id(self, section_id: str) -> Optional[OutlineSection]:
        for s in self.outline:
            if s.id == section_id:
                return s
        return None

    def get_fragments_for_section(self, section_id: str) -> List[KnowledgeFragment]:
        return [f for f in self.knowledge_pool.values() if f.linked_section_id == section_id]

    def add_citation(self, citation: Citation) -> str:
        self.citation_pool[citation.id] = citation
        return citation.id

    def add_fragment(self, fragment: KnowledgeFragment) -> str:
        self.knowledge_pool[fragment.id] = fragment
        return fragment.id
