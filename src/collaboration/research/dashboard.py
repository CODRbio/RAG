"""
ReCAP (Recursive Context-Aware Planning) Research Dashboard。

始终保持"已知/未知/已排除"的全局视图，钉在 system prompt 顶部防止 context drift。
每轮 Agent 迭代时，Dashboard 注入 system prompt 开头，确保 LLM 始终看到全局目标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchBrief:
    """研究简报：Phase 1 Scoping 的输出"""
    topic: str = ""
    scope: str = ""                          # 明确研究边界
    success_criteria: List[str] = field(default_factory=list)  # 完成标准
    key_questions: List[str] = field(default_factory=list)     # 核心问题
    exclusions: List[str] = field(default_factory=list)        # 明确排除的内容
    time_range: str = ""                     # 文献时间范围
    source_priority: List[str] = field(default_factory=list)   # 优先来源类型


@dataclass
class SectionStatus:
    """章节状态"""
    title: str
    status: str = "pending"  # pending | researching | writing | reviewing | done
    coverage_score: float = 0.0  # 0-1，信息充分度
    source_count: int = 0
    gaps: List[str] = field(default_factory=list)  # 该章节的信息缺口
    research_rounds: int = 0  # 该章节已执行的研究轮次（用于 per-section 限制）
    evidence_scarce: bool = False  # 检索证据不足（用于写作降级与最终限制说明）


@dataclass
class ResearchDashboard:
    """
    研究仪表盘 — ReCAP 的核心状态。

    每轮迭代注入 system prompt 顶部，确保 LLM 始终看到：
    - 研究范围和目标
    - 各章节进度
    - 信息缺口
    - 整体置信度
    """
    brief: ResearchBrief = field(default_factory=ResearchBrief)
    sections: List[SectionStatus] = field(default_factory=list)
    overall_confidence: str = "low"  # low | medium | high
    total_sources: int = 0
    total_iterations: int = 0
    coverage_gaps: List[str] = field(default_factory=list)  # 全局信息缺口
    conflict_notes: List[str] = field(default_factory=list)  # 来源冲突记录

    def add_section(self, title: str) -> SectionStatus:
        s = SectionStatus(title=title)
        self.sections.append(s)
        return s

    def get_section(self, title: str) -> Optional[SectionStatus]:
        for s in self.sections:
            if s.title == title:
                return s
        return None

    def update_section(self, title: str, **kwargs) -> None:
        s = self.get_section(title)
        if s:
            for k, v in kwargs.items():
                if hasattr(s, k):
                    setattr(s, k, v)

    def compute_overall_progress(self) -> float:
        """计算整体进度 0-1"""
        if not self.sections:
            return 0.0
        done = sum(1 for s in self.sections if s.status == "done")
        return done / len(self.sections)

    def compute_coverage(self) -> float:
        """计算整体信息覆盖度 0-1"""
        if not self.sections:
            return 0.0
        return sum(s.coverage_score for s in self.sections) / len(self.sections)

    def update_confidence(self) -> None:
        """根据覆盖度自动更新置信度"""
        coverage = self.compute_coverage()
        if coverage >= 0.8 and not self.coverage_gaps:
            self.overall_confidence = "high"
        elif coverage >= 0.5:
            self.overall_confidence = "medium"
        else:
            self.overall_confidence = "low"

    def get_next_section(self) -> Optional[SectionStatus]:
        """获取下一个需要处理的章节"""
        for s in self.sections:
            if s.status in ("pending", "researching"):
                return s
        # 检查是否有 reviewing 的章节需要补充
        for s in self.sections:
            if s.status == "reviewing" and s.coverage_score < 0.6:
                return s
        return None

    def all_done(self) -> bool:
        return all(s.status == "done" for s in self.sections) if self.sections else False

    def to_system_prompt(self) -> str:
        """生成注入 system prompt 顶部的仪表盘文本"""
        lines = ["═══ RESEARCH DASHBOARD ═══"]

        # 研究简报
        b = self.brief
        lines.append(f"Topic: {b.topic}")
        if b.scope:
            lines.append(f"Scope: {b.scope}")
        if b.success_criteria:
            lines.append("Success Criteria: " + "; ".join(b.success_criteria))
        if b.exclusions:
            lines.append("Exclusions: " + "; ".join(b.exclusions))

        # 进度
        progress = self.compute_overall_progress()
        lines.append(f"\nProgress: {progress:.0%} | Sources: {self.total_sources} | Confidence: {self.overall_confidence}")

        # 各章节状态
        lines.append("\nSection Status:")
        for s in self.sections:
            icon = {"pending": "⬜", "researching": "🔍", "writing": "✍️",
                    "reviewing": "🔄", "done": "✅"}.get(s.status, "⬜")
            cov = f"{s.coverage_score:.0%}" if s.coverage_score > 0 else "—"
            gaps_str = f" [Gaps: {', '.join(s.gaps[:2])}]" if s.gaps else ""
            lines.append(f"  {icon} {s.title} (Coverage:{cov}, Sources:{s.source_count}){gaps_str}")

        # 全局缺口
        if self.coverage_gaps:
            lines.append("\nGlobal Information Gaps:")
            for g in self.coverage_gaps[-5:]:
                lines.append(f"  ❗ {g}")

        # 冲突
        if self.conflict_notes:
            lines.append(f"\nSource Conflicts ({len(self.conflict_notes)}):")
            for c in self.conflict_notes[-3:]:
                lines.append(f"  ⚠️ {c}")

        lines.append("═══════════════════════")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（用于 SSE/API 传输）"""
        return {
            "topic": self.brief.topic,
            "scope": self.brief.scope,
            "progress": self.compute_overall_progress(),
            "coverage": self.compute_coverage(),
            "confidence": self.overall_confidence,
            "total_sources": self.total_sources,
            "total_iterations": self.total_iterations,
            "sections": [
                {
                    "title": s.title,
                    "status": s.status,
                    "coverage_score": s.coverage_score,
                    "source_count": s.source_count,
                    "gaps": s.gaps,
                    "research_rounds": s.research_rounds,
                    "evidence_scarce": s.evidence_scarce,
                }
                for s in self.sections
            ],
            "coverage_gaps": self.coverage_gaps,
            "conflict_notes": self.conflict_notes,
        }
