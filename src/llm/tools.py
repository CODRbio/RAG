"""
统一 Tool 抽象层。

一次定义 Tool Schema，自动适配 OpenAI / Anthropic function calling 格式，
并提供 prompt-based 降级方案。

Usage:
    from src.llm.tools import CORE_TOOLS, ToolDef, to_openai_tools, to_anthropic_tools, execute_tool_call
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.log import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────
# Tool 定义
# ────────────────────────────────────────────────

@dataclass
class ToolDef:
    """Provider 无关的 Tool 定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (type: object)
    handler: Optional[Callable[..., Any]] = None  # 实际执行函数

    # ── 转换为 OpenAI function calling 格式 ──
    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ── 转换为 Anthropic tool 格式 ──
    def to_anthropic(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """统一的 tool call 解析结果（provider 无关）"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool 执行结果"""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


# ────────────────────────────────────────────────
# 格式转换工具
# ────────────────────────────────────────────────

def to_openai_tools(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    return [t.to_openai() for t in tools]


def to_anthropic_tools(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    return [t.to_anthropic() for t in tools]


def tools_to_prompt(tools: List[ToolDef]) -> str:
    """降级：将 tools 描述注入 system prompt（用于不支持 FC 的模型）"""
    lines = ["你可以调用以下工具来完成任务。调用格式：",
             '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>',
             "", "可用工具："]
    for t in tools:
        params_desc = []
        props = t.parameters.get("properties", {})
        for pname, pinfo in props.items():
            req = "必填" if pname in t.parameters.get("required", []) else "可选"
            params_desc.append(f"    - {pname} ({pinfo.get('type', 'any')}, {req}): {pinfo.get('description', '')}")
        lines.append(f"\n**{t.name}**: {t.description}")
        if params_desc:
            lines.append("  参数:")
            lines.extend(params_desc)
    return "\n".join(lines)


# ────────────────────────────────────────────────
# 响应解析：从 raw LLM response 中提取 tool calls
# ────────────────────────────────────────────────

def parse_tool_calls_openai(raw: Dict[str, Any]) -> List[ToolCall]:
    """从 OpenAI-compatible 响应中解析 tool_calls"""
    calls = []
    choices = raw.get("choices") or []
    if not choices:
        return calls
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        calls.append(ToolCall(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            arguments=args,
        ))
    return calls


def parse_tool_calls_anthropic(raw: Dict[str, Any]) -> List[ToolCall]:
    """从 Anthropic 响应中解析 tool_use content blocks"""
    calls = []
    content = raw.get("content") or []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input") or {},
            ))
    return calls


def parse_tool_calls_prompt(text: str) -> List[ToolCall]:
    """从纯文本响应中解析 <tool_call> 标签（降级模式）"""
    calls = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for i, m in enumerate(re.finditer(pattern, text, re.DOTALL)):
        try:
            data = json.loads(m.group(1))
            calls.append(ToolCall(
                id=f"prompt_tc_{i}",
                name=data.get("name", ""),
                arguments=data.get("arguments") or data.get("args") or {},
            ))
        except json.JSONDecodeError:
            continue
    return calls


def parse_tool_calls(raw: Dict[str, Any], is_anthropic: bool = False) -> List[ToolCall]:
    """统一解析入口"""
    if is_anthropic:
        calls = parse_tool_calls_anthropic(raw)
    else:
        calls = parse_tool_calls_openai(raw)
    # 如果原生 FC 没有返回 tool_calls，尝试从文本中降级解析
    if not calls:
        text = ""
        if is_anthropic:
            for block in (raw.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        else:
            choices = raw.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                c = msg.get("content")
                if isinstance(c, str):
                    text = c
        if "<tool_call>" in text:
            calls = parse_tool_calls_prompt(text)
    return calls


def has_tool_calls(raw: Dict[str, Any], is_anthropic: bool = False) -> bool:
    """快速判断响应中是否包含 tool calls"""
    if is_anthropic:
        stop = raw.get("stop_reason", "")
        if stop == "tool_use":
            return True
        for block in (raw.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    else:
        choices = raw.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if msg.get("tool_calls"):
                return True
            finish = choices[0].get("finish_reason", "")
            if finish in ("tool_calls", "function_call"):
                return True
    return False


# ────────────────────────────────────────────────
# Tool 执行
# ────────────────────────────────────────────────

def execute_tool_call(
    tool_call: ToolCall,
    tools: List[ToolDef],
) -> ToolResult:
    """查找并执行 tool call，返回结果"""
    tool_map = {t.name: t for t in tools}
    tool = tool_map.get(tool_call.name)
    if tool is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error: unknown tool '{tool_call.name}'",
            is_error=True,
        )
    if tool.handler is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error: tool '{tool_call.name}' has no handler",
            is_error=True,
        )
    try:
        result = tool.handler(**tool_call.arguments)
        # 将结果序列化为字符串
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict) or isinstance(result, list):
            content = json.dumps(result, ensure_ascii=False, default=str)
        else:
            content = str(result)
        # 截断过长结果
        if len(content) > 8000:
            content = content[:7500] + "\n... (truncated)"
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
        )
    except Exception as e:
        logger.warning(f"Tool execution error [{tool_call.name}]: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error executing {tool_call.name}: {e}",
            is_error=True,
        )


# ────────────────────────────────────────────────
# 构建 tool result 消息（喂回 LLM）
# ────────────────────────────────────────────────

def tool_result_to_openai_message(result: ToolResult) -> Dict[str, Any]:
    """将 ToolResult 转为 OpenAI tool message"""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": result.content,
    }


def tool_result_to_anthropic_content(result: ToolResult) -> Dict[str, Any]:
    """将 ToolResult 转为 Anthropic tool_result content block"""
    return {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content,
        "is_error": result.is_error,
    }


# ────────────────────────────────────────────────
# 8 个核心 Tool 定义（handler 在 register 时绑定）
# ────────────────────────────────────────────────

_SEARCH_LOCAL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索查询文本"},
        "top_k": {"type": "integer", "description": "返回结果数量", "default": 10},
    },
    "required": ["query"],
}

_SEARCH_WEB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "网络搜索查询"},
        "top_k": {"type": "integer", "description": "返回结果数量", "default": 10},
    },
    "required": ["query"],
}

_SEARCH_SCHOLAR_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "学术论文搜索查询"},
        "year_from": {"type": "integer", "description": "起始年份（可选）"},
        "limit": {"type": "integer", "description": "结果数量", "default": 5},
    },
    "required": ["query"],
}

_EXPLORE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_name": {"type": "string", "description": "实体名称"},
        "depth": {"type": "integer", "description": "扩展深度 (1-3)", "default": 1},
    },
    "required": ["entity_name"],
}

_CANVAS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "update", "get"], "description": "操作类型"},
        "canvas_id": {"type": "string", "description": "画布 ID（update/get 时必填）"},
        "topic": {"type": "string", "description": "主题（create 时必填）"},
        "content": {"type": "string", "description": "要更新的 Markdown 内容"},
    },
    "required": ["action"],
}

_CITATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "canvas_id": {"type": "string", "description": "画布 ID"},
        "format": {"type": "string", "enum": ["text", "bibtex"], "description": "引文格式", "default": "text"},
    },
    "required": ["canvas_id"],
}

_COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要比较的论文 ID 列表 (2-5)",
            "minItems": 2,
            "maxItems": 5,
        },
        "aspects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "比较维度",
        },
    },
    "required": ["paper_ids"],
}

_RUN_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "要执行的 Python 代码（简单计算/数据处理）"},
    },
    "required": ["code"],
}


# ── Handler 实现 ──

def _handle_search_local(query: str, top_k: int = 10, **_) -> str:
    from src.retrieval.service import get_retrieval_service
    svc = get_retrieval_service()
    pack = svc.search(query=query, mode="local", top_k=top_k)
    return pack.to_context_string(max_chunks=min(top_k, 15))


def _handle_search_web(query: str, top_k: int = 10, **_) -> str:
    from src.retrieval.service import get_retrieval_service
    svc = get_retrieval_service()
    pack = svc.search(query=query, mode="web", top_k=top_k)
    return pack.to_context_string(max_chunks=min(top_k, 15))


def _handle_search_scholar(query: str, year_from: Optional[int] = None, limit: int = 5, **_) -> str:
    try:
        from src.retrieval.semantic_scholar import SemanticScholarSearch
        ss = SemanticScholarSearch()
        results = ss.search(query, year_from=year_from, limit=limit)
        if not results:
            return "未找到相关学术论文。"
        lines = []
        for r in results[:limit]:
            title = r.get("title", "")
            year = r.get("year", "")
            abstract = (r.get("abstract") or "")[:300]
            doi = r.get("externalIds", {}).get("DOI", "")
            lines.append(f"- **{title}** ({year}) DOI:{doi}\n  {abstract}")
        return "\n".join(lines)
    except Exception as e:
        return f"学术搜索失败: {e}"


def _handle_explore_graph(entity_name: str, depth: int = 1, **_) -> str:
    try:
        from src.api.routes_graph import graph_neighbors
        result = graph_neighbors(entity_name, depth=depth)
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        lines = [f"实体 '{entity_name}' 的知识图谱（深度={depth}）:"]
        lines.append(f"节点数: {len(nodes)}, 边数: {len(edges)}")
        entity_nodes = [n for n in nodes if n.get("type") != "CHUNK"]
        for n in entity_nodes[:20]:
            marker = " [中心]" if n.get("is_center") else ""
            lines.append(f"  - {n['id']} ({n['type']}){marker}")
        for e in edges[:20]:
            lines.append(f"  {e['source']} --[{e['relation']}]--> {e['target']}")
        return "\n".join(lines)
    except Exception as e:
        return f"图谱查询失败: {e}"


def _handle_canvas(action: str, canvas_id: str = "", topic: str = "", content: str = "", **_) -> str:
    from src.collaboration.canvas.canvas_manager import create_canvas, get_canvas, update_canvas
    if action == "create":
        canvas = create_canvas(topic=topic or "Untitled")
        return json.dumps({"canvas_id": canvas.id, "topic": canvas.topic}, ensure_ascii=False)
    elif action == "get":
        canvas = get_canvas(canvas_id)
        if canvas is None:
            return f"画布 '{canvas_id}' 不存在"
        return json.dumps({"canvas_id": canvas.id, "topic": canvas.topic, "markdown": canvas.markdown[:3000]}, ensure_ascii=False)
    elif action == "update":
        update_canvas(canvas_id, markdown=content)
        return f"画布 '{canvas_id}' 已更新"
    return f"未知操作: {action}"


def _handle_citations(canvas_id: str, format: str = "text", **_) -> str:
    from src.collaboration.citation.formatter import format_reference_list
    from src.collaboration.canvas.canvas_manager import get_canvas_citations
    citations = get_canvas_citations(canvas_id)
    if not citations:
        return "该画布暂无引文。"
    # 映射前端参数到有效 style
    style_map = {"text": "custom", "bibtex": "apa", "apa": "apa", "numeric": "numeric", "custom": "custom"}
    style = style_map.get(format, "custom")
    return format_reference_list(citations, style=style)


def _handle_compare(paper_ids: List[str], aspects: Optional[List[str]] = None, **_) -> str:
    try:
        from src.api.routes_compare import compare_papers, CompareRequest
        req = CompareRequest(paper_ids=paper_ids, aspects=aspects or ["objective", "methodology", "key_findings", "limitations"])
        resp = compare_papers(req)
        parts = []
        if resp.narrative:
            parts.append(f"综合分析: {resp.narrative}")
        for aspect, cells in resp.comparison_matrix.items():
            parts.append(f"\n[{aspect}]")
            for pid, desc in cells.items():
                parts.append(f"  {pid}: {desc}")
        return "\n".join(parts) if parts else "对比结果为空"
    except Exception as e:
        return f"论文对比失败: {e}"


def _handle_run_code(code: str, **_) -> str:
    """安全沙盒执行简单 Python 代码"""
    import io
    import contextlib
    # 限制危险操作
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__",
                 "open(", "shutil", "pathlib"]
    for f in forbidden:
        if f in code:
            return f"安全限制: 不允许使用 '{f}'"
    stdout = io.StringIO()
    local_ns: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                          "sum": sum, "min": min, "max": max, "abs": abs,
                                          "round": round, "sorted": sorted, "enumerate": enumerate,
                                          "zip": zip, "map": map, "filter": filter,
                                          "int": int, "float": float, "str": str, "list": list,
                                          "dict": dict, "set": set, "tuple": tuple, "bool": bool,
                                          "True": True, "False": False, "None": None}},
                 local_ns)
        output = stdout.getvalue()
        return output.strip() if output.strip() else "(代码执行完毕，无输出)"
    except Exception as e:
        return f"执行错误: {e}"


# ── 注册核心 Tools ──

CORE_TOOLS: List[ToolDef] = [
    ToolDef(
        name="search_local",
        description="检索本地知识库（向量数据库 + 图谱融合检索），适用于查找已入库的论文和文档内容。",
        parameters=_SEARCH_LOCAL_SCHEMA,
        handler=_handle_search_local,
    ),
    ToolDef(
        name="search_web",
        description="网络搜索（Tavily/Google），获取最新的在线信息和网页内容。",
        parameters=_SEARCH_WEB_SCHEMA,
        handler=_handle_search_web,
    ),
    ToolDef(
        name="search_scholar",
        description="学术论文搜索（Semantic Scholar），查找特定领域的学术文献，获取标题、摘要、DOI。",
        parameters=_SEARCH_SCHOLAR_SCHEMA,
        handler=_handle_search_scholar,
    ),
    ToolDef(
        name="explore_graph",
        description="知识图谱探索，查看指定实体的关联实体和关系，发现跨文档的知识连接。",
        parameters=_EXPLORE_GRAPH_SCHEMA,
        handler=_handle_explore_graph,
    ),
    ToolDef(
        name="canvas",
        description="操作研究画布：创建(create)、获取(get)、更新(update)画布内容。",
        parameters=_CANVAS_SCHEMA,
        handler=_handle_canvas,
    ),
    ToolDef(
        name="get_citations",
        description="获取画布的引文列表，支持 text 和 bibtex 格式。",
        parameters=_CITATIONS_SCHEMA,
        handler=_handle_citations,
    ),
    ToolDef(
        name="compare_papers",
        description="多文档对比：选择 2-5 篇论文，自动生成结构化对比矩阵和分析。",
        parameters=_COMPARE_SCHEMA,
        handler=_handle_compare,
    ),
    ToolDef(
        name="run_code",
        description="执行简单的 Python 代码进行数据计算、统计验证或格式转换。",
        parameters=_RUN_CODE_SCHEMA,
        handler=_handle_run_code,
    ),
]
