"""
Tests for dynamic tool routing in src.llm.tools.
"""

import sys
from pathlib import Path

# Support direct execution: `python tests/test_tools_routing.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.tools import get_routed_skills, get_tools_by_names


def _tool_names(tools):
    return [t.name for t in tools]


def test_routing_local_always_on_when_search_enabled():
    names = _tool_names(
        get_routed_skills(
            message="随便聊聊",
            current_stage="explore",
            search_mode="local",
            allowed_web_providers=None,
        )
    )
    assert "search_local" in names


def test_routing_none_mode_can_be_tolless():
    names = _tool_names(
        get_routed_skills(
            message="随便聊聊",
            current_stage="explore",
            search_mode="none",
            allowed_web_providers=None,
        )
    )
    assert names == []


def test_routing_web_providers_subset_and_empty_list_semantics():
    subset = _tool_names(
        get_routed_skills(
            message="请检索外部信息",
            current_stage="explore",
            search_mode="hybrid",
            allowed_web_providers=["tavily", "scholar"],
        )
    )
    assert "search_local" in subset
    assert "search_web" in subset
    assert "search_scholar" in subset
    assert "search_ncbi" not in subset

    empty = _tool_names(
        get_routed_skills(
            message="请检索外部信息",
            current_stage="explore",
            search_mode="hybrid",
            allowed_web_providers=[],
        )
    )
    assert empty == ["search_local"]


def test_routing_analysis_and_graph_keywords():
    analysis = _tool_names(
        get_routed_skills(
            message="请对比不同方法并做统计计算",
            current_stage="explore",
            search_mode="local",
            allowed_web_providers=None,
        )
    )
    assert "compare_papers" in analysis
    assert "run_code" in analysis

    graph = _tool_names(
        get_routed_skills(
            message="帮我看下知识图谱关系网络",
            current_stage="explore",
            search_mode="local",
            allowed_web_providers=None,
        )
    )
    assert "explore_graph" in graph


def test_routing_collab_stage_and_keyword_trigger():
    by_stage = _tool_names(
        get_routed_skills(
            message="继续写",
            current_stage="refine",
            search_mode="local",
            allowed_web_providers=None,
        )
    )
    assert "canvas" in by_stage
    assert "get_citations" in by_stage

    by_keyword = _tool_names(
        get_routed_skills(
            message="更新画布草稿并整理引用",
            current_stage="explore",
            search_mode="local",
            allowed_web_providers=None,
        )
    )
    assert "canvas" in by_keyword
    assert "get_citations" in by_keyword


def test_get_tools_by_names_preserves_core_order_and_ignores_unknown():
    names = _tool_names(get_tools_by_names(["run_code", "search_local", "unknown_tool"]))
    # CORE_TOOLS order places search_local before run_code
    assert names == ["search_local", "run_code"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
