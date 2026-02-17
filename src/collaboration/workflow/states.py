"""
工作流阶段定义与配置。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


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
        system_prompt_template="""你是一个学术研究助手，正在帮助用户探索综述主题。
当前主题: {topic}

你的任务是：
1. 回答用户的开放式问题
2. 提供领域概览和关键研究方向
3. 在适当时候建议用户进入大纲构建阶段

请基于以下检索到的信息回答:
{context}
""",
        allowed_intents=["chat", "deep_research"],
        transition_triggers={WorkflowStage.OUTLINE: "用户明确表示开始构建大纲"},
    ),
    WorkflowStage.OUTLINE: StageConfig(
        name=WorkflowStage.OUTLINE,
        description="大纲阶段：与用户共同敲定综述骨架",
        system_prompt_template="""你是一个学术写作助手，正在帮助用户构建综述大纲。
主题: {topic}
当前大纲: {outline}

你的任务是：
1. 帮助生成或修改大纲结构
2. 对每个章节进行信息充足度分析
3. 识别需要补充检索的信息缺口

请基于以下信息协助用户:
{context}
""",
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
        system_prompt_template="""你是一个学术写作助手，正在帮助用户撰写综述。
主题: {topic}
当前章节: {current_section}
章节要点: {section_guidance}

写作要求：
1. 使用学术语言，逻辑清晰
2. 每个观点必须有文献支撑
3. 使用 [cite_key] 格式标注引用

可用的知识片段:
{fragments}

请撰写该章节内容。
""",
        allowed_intents=["chat", "deep_research"],
        transition_triggers={
            WorkflowStage.REFINE: "所有章节初稿完成",
            WorkflowStage.OUTLINE: "需要修改大纲结构",
        },
    ),
    WorkflowStage.REFINE: StageConfig(
        name=WorkflowStage.REFINE,
        description="精修阶段：优化语言、检查逻辑、统一格式",
        system_prompt_template="""你是一个学术编辑助手，正在帮助用户精修综述。
主题: {topic}

你的任务是：
1. 优化语言表达，使其更加学术化
2. 检查段落间的逻辑连贯性
3. 统一引用格式
4. 检查是否有遗漏的重要内容

当前内容:
{draft_content}

请基于以下信息协助用户:
{context}
""",
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
