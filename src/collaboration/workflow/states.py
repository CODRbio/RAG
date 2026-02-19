"""
工作流阶段定义与配置。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


class WorkflowStage(Enum):
    EXPLORE = "explore"
    OUTLINE = "outline"
    DRAFTING = "drafting"
    REFINE = "refine"


@dataclass
class StageConfig:
    """阶段配置"""

    name: WorkflowStage
    description: str
    system_prompt_template: str
    allowed_intents: List[str]
    on_enter_actions: Optional[List[str]] = None
    transition_triggers: Optional[Dict[WorkflowStage, str]] = None

    def __post_init__(self) -> None:
        if self.on_enter_actions is None:
            self.on_enter_actions = []
        if self.transition_triggers is None:
            self.transition_triggers = {}


STAGE_CONFIGS: Dict[WorkflowStage, StageConfig] = {
    WorkflowStage.EXPLORE: StageConfig(
        name=WorkflowStage.EXPLORE,
        description="探索阶段：帮助用户发散思维，确定综述范围",
        system_prompt_template=_pm.load("workflow_explore_system.txt"),
        allowed_intents=["chat", "deep_research"],
        transition_triggers={WorkflowStage.OUTLINE: "用户明确表示开始构建大纲"},
    ),
    WorkflowStage.OUTLINE: StageConfig(
        name=WorkflowStage.OUTLINE,
        description="大纲阶段：与用户共同敲定综述骨架",
        system_prompt_template=_pm.load("workflow_outline_system.txt"),
        allowed_intents=["chat", "deep_research"],
        on_enter_actions=["auto_generate_outline_if_empty"],
        transition_triggers={
            WorkflowStage.DRAFTING: "大纲确认完成且无重大缺口",
            WorkflowStage.EXPLORE: "用户要求重新探索",
        },
    ),
    WorkflowStage.DRAFTING: StageConfig(
        name=WorkflowStage.DRAFTING,
        description="写作阶段：逐段生成高质量草稿",
        system_prompt_template=_pm.load("workflow_draft_system.txt"),
        allowed_intents=["chat", "deep_research"],
        transition_triggers={
            WorkflowStage.REFINE: "所有章节初稿完成",
            WorkflowStage.OUTLINE: "需要修改大纲结构",
        },
    ),
    WorkflowStage.REFINE: StageConfig(
        name=WorkflowStage.REFINE,
        description="精修阶段：优化语言、检查逻辑、统一格式",
        system_prompt_template=_pm.load("workflow_refine_system.txt"),
        allowed_intents=["chat", "deep_research"],
        transition_triggers={WorkflowStage.DRAFTING: "需要重写某些章节"},
    ),
}


def get_stage_config(stage: str) -> Optional[StageConfig]:
    """按阶段名（str）获取配置，无效则返回 None。"""
    try:
        s = WorkflowStage(stage)
        return STAGE_CONFIGS.get(s)
    except ValueError:
        return None
