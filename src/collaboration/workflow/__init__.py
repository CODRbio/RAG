"""
工作流状态机：四阶段 Explore / Outline / Drafting / Refine，LangGraph 最小图。
"""

from .graph import build_workflow_graph, run_workflow
from .states import WorkflowStage, StageConfig, STAGE_CONFIGS, get_stage_config
from .transitions import compute_next_stage

__all__ = [
    "WorkflowStage",
    "StageConfig",
    "STAGE_CONFIGS",
    "get_stage_config",
    "compute_next_stage",
    "build_workflow_graph",
    "run_workflow",
]
