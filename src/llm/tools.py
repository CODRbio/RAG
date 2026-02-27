"""
统一 Tool 抽象层。

一次定义 Tool Schema，自动适配 OpenAI / Anthropic function calling 格式，
并提供 prompt-based 降级方案。

Usage:
    from src.llm.tools import CORE_TOOLS, ToolDef, to_openai_tools, to_anthropic_tools, execute_tool_call
    from src.llm.tools import get_routed_skills, get_tools_by_names
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import threading

from src.log import get_logger

logger = get_logger(__name__)

# Thread-local collector for EvidenceChunks produced by Agent tool calls.
# routes_chat.py activates/reads this around the react_loop invocation.
_agent_chunks_local = threading.local()


def set_tool_collection(collection: Optional[str]) -> None:
    """Set the Milvus collection for search_local/search_web in this thread."""
    _agent_chunks_local.collection = collection or None


def start_agent_chunk_collector() -> None:
    """Activate the per-thread chunk collector (call before react_loop)."""
    _agent_chunks_local.chunks = []


def drain_agent_chunks() -> list:
    """Return and clear all collected chunks (call after react_loop)."""
    chunks = getattr(_agent_chunks_local, "chunks", None) or []
    _agent_chunks_local.chunks = []
    return chunks


def _collect_chunks(chunks: list) -> None:
    """Append EvidenceChunks to the thread-local collector if active."""
    store = getattr(_agent_chunks_local, "chunks", None)
    if store is not None:
        store.extend(chunks)


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

_SEARCH_NCBI_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "PubMed 生物医学搜索查询（英文关键词效果最佳）",
        },
        "limit": {
            "type": "integer",
            "description": "返回结果数量",
            "default": 5,
        },
    },
    "required": ["query"],
}


# ── Handler 实现 ──

def _handle_search_local(query: str, top_k: int = 10, **_) -> str:
    from src.retrieval.service import get_retrieval_service
    col = getattr(_agent_chunks_local, "collection", None)
    svc = get_retrieval_service(collection=col)
    pack = svc.search(query=query, mode="local", top_k=top_k)
    _collect_chunks(pack.chunks[:min(top_k, 15)])
    return pack.to_context_string(max_chunks=min(top_k, 15))


def _handle_search_web(query: str, top_k: int = 10, **_) -> str:
    from src.retrieval.service import get_retrieval_service
    col = getattr(_agent_chunks_local, "collection", None)
    svc = get_retrieval_service(collection=col)
    pack = svc.search(query=query, mode="web", top_k=top_k)
    _collect_chunks(pack.chunks[:min(top_k, 15)])
    return pack.to_context_string(max_chunks=min(top_k, 15))


def _handle_search_scholar(query: str, year_from: Optional[int] = None, limit: int = 5, **_) -> str:
    try:
        from src.retrieval.semantic_scholar import SemanticScholarSearch
        from src.retrieval.evidence import EvidenceChunk
        ss = SemanticScholarSearch()
        results = ss.search(query, year_from=year_from, limit=limit)
        if not results:
            return "未找到相关学术论文。"
        lines = []
        chunks = []
        for r in results[:limit]:
            title = r.get("title", "")
            year = r.get("year", "")
            abstract = (r.get("abstract") or "")[:300]
            doi = r.get("externalIds", {}).get("DOI", "")
            paper_id = r.get("paperId", "")
            authors_raw = r.get("authors") or []
            author_names = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]
            url = f"https://api.semanticscholar.org/CorpusID:{paper_id}" if paper_id else ""
            chunk = EvidenceChunk(
                chunk_id=f"scholar_{paper_id or title[:20]}",
                doc_id=doi or paper_id or title[:30],
                text=f"{title}. {abstract}",
                score=0.0,
                source_type="web",
                doc_title=title,
                authors=author_names[:5],
                year=int(year) if year else None,
                url=url,
                doi=doi or None,
                provider="semantic",
            )
            chunks.append(chunk)
            lines.append(f"[{chunk.ref_hash}] **{title}** ({year}) DOI:{doi}\n  {abstract}")
        _collect_chunks(chunks)
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
    style_map = {
        "text": "custom",
        "bibtex": "apa",
        "apa": "apa",
        "ieee": "ieee",
        "numeric": "numeric",
        "custom": "custom",
    }
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


def _handle_search_ncbi(query: str, limit: int = 5, **_) -> str:
    """调用 NCBI PubMed E-Utilities，返回生物医学文献摘要信息。"""
    try:
        import asyncio
        from src.retrieval.ncbi_search import get_ncbi_searcher
        from src.retrieval.evidence import EvidenceChunk

        searcher = get_ncbi_searcher()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, searcher.search(query, limit=limit))
                    results = future.result(timeout=30)
            else:
                results = loop.run_until_complete(searcher.search(query, limit=limit))
        except RuntimeError:
            results = asyncio.run(searcher.search(query, limit=limit))

        if not results:
            return "PubMed 未找到相关文献。"

        lines = []
        chunks = []
        for r in results[:limit]:
            meta = r.get("metadata", {})
            title = meta.get("title", r.get("content", ""))
            year = meta.get("year", "")
            doi = meta.get("doi", "")
            authors = meta.get("authors", [])
            authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
            url = meta.get("url", "")
            pmid = meta.get("pmid", "")
            abstract = (r.get("content") or meta.get("abstract") or "")[:400]
            chunk = EvidenceChunk(
                chunk_id=f"ncbi_{pmid or doi or title[:20]}",
                doc_id=doi or pmid or title[:30],
                text=f"{title}. {abstract}",
                score=0.0,
                source_type="web",
                doc_title=title,
                authors=authors[:5],
                year=int(year) if year else None,
                url=url or None,
                doi=doi or None,
                provider="ncbi",
            )
            chunks.append(chunk)
            lines.append(
                f"[{chunk.ref_hash}] **{title}** ({year})\n"
                f"  Authors: {authors_str}\n"
                f"  DOI: {doi or '—'}  URL: {url}"
            )
        _collect_chunks(chunks)
        return "\n".join(lines)
    except Exception as e:
        return f"NCBI 搜索失败: {e}"


def _handle_run_code(code: str, **_) -> str:
    """以子进程方式执行 Python 代码。"""
    # TODO(Security): 生产环境强烈建议将 subprocess 替换为安全的隔离沙盒 API (如 E2B)。
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        with tmp_file:
            tmp_file.write(code)

        result = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout
        return f"{result.stdout}{result.stderr}"
    except subprocess.TimeoutExpired:
        return "代码执行超时（15秒），已中止。"
    except Exception as e:
        return f"执行错误: {e}"
    finally:
        if tmp_file is not None:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass


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
    ToolDef(
        name="search_ncbi",
        description=(
            "搜索 NCBI PubMed 生物医学文献库，专攻生物学、医学、基因组学、海洋生态等领域。"
            "返回标题、作者、年份、DOI，适合精确的生物医学文献检索。"
        ),
        parameters=_SEARCH_NCBI_SCHEMA,
        handler=_handle_search_ncbi,
    ),
]


# ────────────────────────────────────────────────
# Tool Registry & Dynamic Skill Routing
# ────────────────────────────────────────────────

_TOOL_REGISTRY: Dict[str, ToolDef] = {t.name: t for t in CORE_TOOLS}

_GROUP_SEARCH_LOCAL = frozenset({"search_local"})
_GROUP_WEB = frozenset({"search_web", "search_scholar", "search_ncbi"})
_GROUP_ANALYSIS = frozenset({"compare_papers", "run_code"})
_GROUP_GRAPH = frozenset({"explore_graph"})
_GROUP_COLLAB = frozenset({"canvas", "get_citations"})

_WEB_PROVIDER_TO_TOOL: Dict[str, str] = {
    "tavily": "search_web",
    "google": "search_web",
    "scholar": "search_scholar",
    "semantic": "search_scholar",
    "ncbi": "search_ncbi",
    "pubmed": "search_ncbi",
}

_RE_ANALYSIS = re.compile(
    r"对比|比较|差异|统计|计算|代码|数据分析|分析数据"
    r"|compare|contrast|diff|statistic|calculat|code|data\s*analy",
    re.IGNORECASE,
)
_RE_GRAPH = re.compile(
    r"关系|网络|图谱|关联|知识图"
    r"|relation|network|graph|connection|linked|topology",
    re.IGNORECASE,
)
_RE_COLLAB = re.compile(
    r"画布|草稿|大纲|引用|参考文献|引文"
    r"|canvas|draft|outline|citation|reference|bibliography",
    re.IGNORECASE,
)

_TOOL_ORDER: Dict[str, int] = {t.name: i for i, t in enumerate(CORE_TOOLS)}


def get_tools_by_names(names: List[str]) -> List[ToolDef]:
    """Return ToolDef instances matching the given tool names, preserving CORE_TOOLS order."""
    tools = [_TOOL_REGISTRY[n] for n in names if n in _TOOL_REGISTRY]
    tools.sort(key=lambda t: _TOOL_ORDER.get(t.name, 999))
    return tools


def get_routed_skills(
    message: str,
    current_stage: str,
    search_mode: str,
    allowed_web_providers: Optional[List[str]] = None,
) -> List[ToolDef]:
    """
    Dynamic skill routing — select only the tools relevant to the current
    request instead of mounting all CORE_TOOLS.

    This reduces prompt token cost and lowers the probability of the LLM
    hallucinating tool calls to irrelevant tools.

    Routing rules
    ─────────────
    1. search_local: always on when search_mode != "none"
    2. Web group (search_web / search_scholar / search_ncbi):
       active when search_mode allows web; narrowed by allowed_web_providers
    3. Analysis group (compare_papers / run_code):
       keyword-triggered by comparison / statistics / code mentions
    4. Graph group (explore_graph):
       keyword-triggered by relationship / graph / network mentions
    5. Collab group (canvas / get_citations):
       stage-triggered (drafting / refine) or keyword-triggered
    """
    selected: set[str] = set()

    # 1. Local search — the backbone of RAG
    if search_mode != "none":
        selected |= _GROUP_SEARCH_LOCAL

    # 2. Web tools — gated by search_mode and optionally by explicit provider list
    if search_mode in ("web", "hybrid"):
        if allowed_web_providers is not None:
            for provider in allowed_web_providers:
                tool_name = _WEB_PROVIDER_TO_TOOL.get(provider.lower().strip())
                if tool_name and tool_name in _GROUP_WEB:
                    selected.add(tool_name)
        else:
            selected |= _GROUP_WEB

    # 3. Analysis group — keyword activated
    if _RE_ANALYSIS.search(message):
        selected |= _GROUP_ANALYSIS

    # 4. Graph group — keyword activated
    if _RE_GRAPH.search(message):
        selected |= _GROUP_GRAPH

    # 5. Collaboration group — stage or keyword activated
    stage_lower = (current_stage or "").lower()
    if stage_lower in ("drafting", "draft", "refine", "writing"):
        selected |= _GROUP_COLLAB
    elif _RE_COLLAB.search(message):
        selected |= _GROUP_COLLAB

    # Fallback: guarantee at least search_local so the agent is never toolless
    if not selected and search_mode != "none":
        selected.add("search_local")

    tools = [_TOOL_REGISTRY[name] for name in selected if name in _TOOL_REGISTRY]
    tools.sort(key=lambda t: _TOOL_ORDER.get(t.name, 999))

    logger.info(
        "skill_router | stage=%s mode=%s providers=%s → tools=[%s] (%d/%d)",
        current_stage, search_mode, allowed_web_providers,
        ", ".join(t.name for t in tools), len(tools), len(CORE_TOOLS),
    )
    return tools
