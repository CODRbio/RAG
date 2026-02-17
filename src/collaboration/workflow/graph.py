"""
LangGraph 最小工作流图：根据当前阶段与意图路由到下一阶段并产出阶段系统提示。
"""

from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from .states import WorkflowStage, get_stage_config
from .transitions import compute_next_stage


class WorkflowState(TypedDict, total=False):
    current_stage: str
    intent_type: str
    topic: str
    outline: str
    context: str
    current_section: str
    section_guidance: str
    fragments: str
    draft_content: str
    next_stage: str
    system_prompt: str


def _route_stage(state: WorkflowState) -> Dict[str, Any]:
    current = state.get("current_stage") or WorkflowStage.EXPLORE.value
    intent = state.get("intent_type") or ""
    next_stage = compute_next_stage(current, intent)
    config = get_stage_config(next_stage)
    system_prompt = ""
    if config:
        system_prompt = config.system_prompt_template.format(
            topic=state.get("topic") or "",
            outline=state.get("outline") or "",
            context=state.get("context") or "",
            current_section=state.get("current_section") or "",
            section_guidance=state.get("section_guidance") or "",
            fragments=state.get("fragments") or "",
            draft_content=state.get("draft_content") or "",
        )
    return {"next_stage": next_stage, "system_prompt": system_prompt}


def build_workflow_graph() -> Any:
    """构建并编译最小工作流图。"""
    builder = StateGraph(WorkflowState)
    builder.add_node("route_stage", _route_stage)
    builder.add_edge(START, "route_stage")
    builder.add_edge("route_stage", END)
    return builder.compile()


def run_workflow(
    current_stage: str,
    intent_type: str,
    *,
    topic: str = "",
    context: str = "",
    **kwargs: str,
) -> Dict[str, Any]:
    """
    运行工作流：返回 next_stage 与 system_prompt（已填充的阶段提示）。
    """
    graph = build_workflow_graph()
    state: WorkflowState = {
        "current_stage": current_stage,
        "intent_type": intent_type,
        "topic": topic,
        "context": context,
        **kwargs,
    }
    result = graph.invoke(state)
    return {
        "next_stage": result.get("next_stage") or current_stage,
        "system_prompt": result.get("system_prompt") or "",
    }
