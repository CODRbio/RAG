"""
根据当前阶段与意图计算下一阶段（简化版：Chat vs Deep Research）。
"""

from .states import WorkflowStage


# 意图 -> 目标阶段（仅 Deep Research 会强制切换阶段）
INTENT_TO_STAGE: dict[str, str] = {
    "deep_research": WorkflowStage.DRAFTING.value,
    # 兼容旧值
    "auto_complete": WorkflowStage.DRAFTING.value,
    "outline_generate": WorkflowStage.OUTLINE.value,
    "draft_section": WorkflowStage.DRAFTING.value,
    "edit_text": WorkflowStage.REFINE.value,
    "export_document": WorkflowStage.REFINE.value,
}


def compute_next_stage(current_stage: str, intent_type: str) -> str:
    """
    根据当前阶段与意图返回下一阶段。
    Chat 模式保持当前阶段不变；Deep Research 切换到 DRAFTING。
    """
    intent_val = getattr(intent_type, "value", intent_type) if intent_type else ""
    next_stage = INTENT_TO_STAGE.get(intent_val)
    if next_stage is not None:
        return next_stage
    return current_stage if current_stage else WorkflowStage.EXPLORE.value
