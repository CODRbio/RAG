"""
RE-TRAC (Recursive Trajectory Compression)

解决长研究链的 context drift 问题：
- 当 context 接近上限时，将详细轨迹压缩为高层摘要
- 保留战略发现、未探索分支、关键引用
- 丢弃重复内容、中间推理步骤
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.log import get_logger
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

logger = get_logger(__name__)


@dataclass
class SearchAction:
    """一次搜索动作的记录"""
    query: str
    tool: str  # search_local / search_web / search_scholar / explore_graph
    result_summary: str  # 压缩后的结果摘要
    source_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchBranch:
    """一条研究分支"""
    id: str
    title: str
    status: str = "pending"  # pending | in_progress | done | discarded
    key_findings: List[str] = field(default_factory=list)
    search_actions: List[SearchAction] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)


@dataclass
class ResearchTrajectory:
    """
    研究轨迹：记录整个研究过程的状态。

    RE-TRAC 的核心数据结构，维护"已知/未知/已排除"的全局视图。
    """
    topic: str
    branches: List[ResearchBranch] = field(default_factory=list)
    compressed_summaries: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    discarded: List[str] = field(default_factory=list)
    total_sources: int = 0
    compression_count: int = 0

    def add_branch(self, branch_id: str, title: str) -> ResearchBranch:
        branch = ResearchBranch(id=branch_id, title=title)
        self.branches.append(branch)
        return branch

    def get_branch(self, branch_id: str) -> Optional[ResearchBranch]:
        for b in self.branches:
            if b.id == branch_id:
                return b
        return None

    def add_search_action(self, branch_id: str, action: SearchAction) -> None:
        branch = self.get_branch(branch_id)
        if branch:
            branch.search_actions.append(action)
            self.total_sources += action.source_count

    def add_finding(self, branch_id: str, finding: str) -> None:
        branch = self.get_branch(branch_id)
        if branch:
            branch.key_findings.append(finding)
            self.known_facts.append(finding)

    def estimate_token_count(self) -> int:
        """粗估当前轨迹占用的 token 数（1 char ≈ 0.5 token for CJK, 0.25 for EN）"""
        total_chars = len(self.topic)
        for b in self.branches:
            total_chars += len(b.title) + sum(len(f) for f in b.key_findings)
            for a in b.search_actions:
                total_chars += len(a.query) + len(a.result_summary)
        for s in self.compressed_summaries:
            total_chars += len(s)
        total_chars += sum(len(f) for f in self.known_facts)
        total_chars += sum(len(q) for q in self.open_questions)
        return int(total_chars * 0.4)  # 粗估

    def needs_compression(self, max_tokens: int = 30000) -> bool:
        return self.estimate_token_count() > max_tokens

    def to_context_string(self) -> str:
        """将轨迹序列化为可注入 system prompt 的字符串"""
        parts = [f"## 研究轨迹: {self.topic}"]
        parts.append(f"已处理来源: {self.total_sources} | 压缩次数: {self.compression_count}")

        if self.compressed_summaries:
            parts.append("\n### 已压缩的研究摘要")
            for i, s in enumerate(self.compressed_summaries):
                parts.append(f"[摘要 {i+1}] {s}")

        active_branches = [b for b in self.branches if b.status != "discarded"]
        if active_branches:
            parts.append("\n### 研究分支")
            for b in active_branches:
                status_icon = {"pending": "⏳", "in_progress": "🔍", "done": "✅"}.get(b.status, "")
                parts.append(f"\n**{b.title}** ({status_icon}{b.status})")
                if b.key_findings:
                    for f in b.key_findings[-5:]:  # 只保留最近 5 个
                        parts.append(f"  - {f}")

        if self.open_questions:
            parts.append("\n### 待回答的问题")
            for q in self.open_questions[-10:]:
                parts.append(f"  - {q}")

        if self.discarded:
            parts.append(f"\n### 已排除方向 ({len(self.discarded)})")
            for d in self.discarded[-5:]:
                parts.append(f"  - {d}")

        return "\n".join(parts)


# ────────────────────────────────────────────────
# RE-TRAC 压缩
# ────────────────────────────────────────────────



def compress_trajectory(
    trajectory: ResearchTrajectory,
    llm_client: Any,
    model: Optional[str] = None,
) -> str:
    """
    当 context 接近上限时，执行 RE-TRAC 压缩。

    将所有分支的详细搜索记录压缩为高层摘要，释放 context 空间。
    """
    # 构建要压缩的详细内容
    detail_parts = []
    for branch in trajectory.branches:
        if branch.status == "discarded":
            continue
        detail_parts.append(f"\n分支: {branch.title} ({branch.status})")
        for action in branch.search_actions:
            detail_parts.append(f"  搜索 [{action.tool}]: {action.query}")
            detail_parts.append(f"  结果: {action.result_summary[:300]}")
        for finding in branch.key_findings:
            detail_parts.append(f"  发现: {finding}")

    trajectory_text = "\n".join(detail_parts)

    prompt = _pm.render("trajectory_compress.txt", trajectory=trajectory_text)

    try:
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": _pm.render("trajectory_compress_system.txt")},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=1500,
        )
        summary = (resp.get("final_text") or "").strip()
    except Exception as e:
        logger.warning(f"RE-TRAC compression failed: {e}")
        # 降级：手动拼接关键发现
        summary = "压缩失败，保留关键发现:\n" + "\n".join(trajectory.known_facts[-20:])

    # 执行压缩：清除详细记录，保留摘要
    trajectory.compressed_summaries.append(summary)
    trajectory.compression_count += 1

    # 清除已压缩分支的详细搜索记录
    for branch in trajectory.branches:
        if branch.status in ("done", "in_progress"):
            branch.search_actions = []  # 清除详细搜索记录
            # 保留 key_findings 的最近几条
            branch.key_findings = branch.key_findings[-3:]

    logger.info(f"RE-TRAC compression #{trajectory.compression_count}: "
                f"{len(summary)} chars, {trajectory.estimate_token_count()} est tokens remaining")

    return summary
