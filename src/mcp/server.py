"""
MCP Server — 将 RAG 系统能力暴露给 MCP 客户端。

使用 FastMCP 高层 API，将 P0 定义的 8 个核心 Tool 暴露为 MCP Tools，
Canvas/论文列表作为 MCP Resources。

运行方式:
    python -m src.mcp.server
    # 或
    uvicorn src.mcp.server:app --host 0.0.0.0 --port 8100

连接测试:
    npx -y @modelcontextprotocol/inspector
    → http://localhost:8100/mcp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ────────────────────────────────────────────────
# MCP Server 实例
# ────────────────────────────────────────────────

mcp = FastMCP(
    "DeepSea RAG Research System",
    json_response=True,
)


# ────────────────────────────────────────────────
# Tools — 对应 src/llm/tools.py 中的 8 个核心 Tool
# ────────────────────────────────────────────────

@mcp.tool()
def search_local(query: str, top_k: int = 10) -> str:
    """检索本地知识库（向量数据库 + 图谱融合检索），适用于查找已入库的论文和文档内容。"""
    from src.retrieval.service import get_retrieval_service
    svc = get_retrieval_service()
    pack = svc.search(query=query, mode="local", top_k=top_k)
    return pack.to_context_string(max_chunks=min(top_k, 15))


@mcp.tool()
def search_web(query: str, top_k: int = 10) -> str:
    """网络搜索（Tavily/Google），获取最新的在线信息和网页内容。"""
    from src.retrieval.service import get_retrieval_service
    svc = get_retrieval_service()
    pack = svc.search(query=query, mode="web", top_k=top_k)
    return pack.to_context_string(max_chunks=min(top_k, 15))


@mcp.tool()
def search_scholar(query: str, year_from: Optional[int] = None, limit: int = 5) -> str:
    """学术论文搜索（Semantic Scholar），查找特定领域的学术文献，获取标题、摘要、DOI。"""
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


@mcp.tool()
def explore_graph(entity_name: str, depth: int = 1) -> str:
    """知识图谱探索，查看指定实体的关联实体和关系，发现跨文档的知识连接。"""
    try:
        from src.graph.hippo_rag import get_hipporag_instance
        hippo = get_hipporag_instance()
        if hippo is None or hippo.graph is None:
            return "知识图谱未加载。"
        G = hippo.graph
        if entity_name not in G:
            return f"实体 '{entity_name}' 不在图谱中。"
        # BFS 获取邻域
        visited = {entity_name}
        frontier = [entity_name]
        edges_out = []
        for _ in range(depth):
            next_frontier = []
            for n in frontier:
                for nb in G.neighbors(n):
                    edge_data = G.edges[n, nb]
                    edges_out.append(f"  {n} --[{edge_data.get('relation', '?')}]--> {nb}")
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        lines = [f"实体 '{entity_name}' 的知识图谱（深度={depth}）:"]
        lines.append(f"节点数: {len(visited)}, 边数: {len(edges_out)}")
        lines.extend(edges_out[:30])
        return "\n".join(lines)
    except Exception as e:
        return f"图谱查询失败: {e}"


@mcp.tool()
def canvas(action: str, canvas_id: str = "", topic: str = "", content: str = "") -> str:
    """操作研究画布：创建(create)、获取(get)、更新(update)画布内容。"""
    from src.collaboration.canvas.canvas_manager import create_canvas, get_canvas, update_canvas
    if action == "create":
        canvas = create_canvas(topic=topic or "Untitled")
        return json.dumps({"canvas_id": canvas.id, "topic": canvas.topic}, ensure_ascii=False)
    elif action == "get":
        canvas = get_canvas(canvas_id)
        if canvas is None:
            return f"画布 '{canvas_id}' 不存在"
        return json.dumps({
            "canvas_id": canvas.id,
            "topic": canvas.topic,
            "markdown": canvas.markdown[:3000],
        }, ensure_ascii=False)
    elif action == "update":
        update_canvas(canvas_id, markdown=content)
        return f"画布 '{canvas_id}' 已更新"
    return f"未知操作: {action}"


@mcp.tool()
def get_citations(canvas_id: str, format: str = "text") -> str:
    """获取画布的引文列表，支持 text 和 bibtex 格式。"""
    from src.collaboration.citation.formatter import format_reference_list
    from src.collaboration.canvas.canvas_manager import get_canvas_citations
    citations = get_canvas_citations(canvas_id)
    if not citations:
        return "该画布暂无引文。"
    style_map = {"text": "custom", "bibtex": "apa", "apa": "apa", "numeric": "numeric", "custom": "custom"}
    style = style_map.get(format, "custom")
    return format_reference_list(citations, style=style)


@mcp.tool()
def compare_papers(paper_ids: list, aspects: Optional[list] = None) -> str:
    """多文档对比：选择 2-5 篇论文，自动生成结构化对比矩阵和分析。"""
    try:
        import importlib
        mod = importlib.import_module("src.api.routes_compare")
        req = mod.CompareRequest(
            paper_ids=paper_ids,
            aspects=aspects or ["objective", "methodology", "key_findings", "limitations"],
        )
        resp = mod.compare_papers(req)
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


@mcp.tool()
def run_code(code: str) -> str:
    """执行简单的 Python 代码进行数据计算、统计验证或格式转换。"""
    import io
    import contextlib
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(",
                 "__import__", "open(", "shutil", "pathlib"]
    for f in forbidden:
        if f in code:
            return f"安全限制: 不允许使用 '{f}'"
    stdout = io.StringIO()
    local_ns: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {
                "print": print, "range": range, "len": len,
                "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "int": int, "float": float, "str": str, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "bool": bool,
                "True": True, "False": False, "None": None,
            }}, local_ns)
        output = stdout.getvalue()
        return output.strip() if output.strip() else "(代码执行完毕，无输出)"
    except Exception as e:
        return f"执行错误: {e}"


# ────────────────────────────────────────────────
# Resources — Canvas 列表 + 论文列表
# ────────────────────────────────────────────────

@mcp.resource("rag://canvases")
def list_canvases() -> str:
    """获取所有研究画布列表。"""
    try:
        from src.collaboration.canvas.canvas_manager import list_canvases as _list
        canvases = _list()
        items = []
        for c in canvases[:50]:
            items.append({
                "id": c.id,
                "topic": c.topic,
                "updated_at": str(getattr(c, "updated_at", "")),
            })
        return json.dumps(items, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("rag://papers")
def list_papers() -> str:
    """获取已入库的论文列表。"""
    try:
        from pathlib import Path as P
        data_dir = P(_project_root) / "data" / "parsed"
        if not data_dir.exists():
            return json.dumps([])
        papers = []
        for d in sorted(data_dir.iterdir()):
            enriched = d / "enriched.json"
            if enriched.exists():
                try:
                    meta = json.loads(enriched.read_text("utf-8"))
                    papers.append({
                        "id": d.name,
                        "title": meta.get("title", d.name),
                        "authors": meta.get("authors", [])[:3],
                    })
                except Exception:
                    papers.append({"id": d.name, "title": d.name})
        return json.dumps(papers[:100], ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("rag://canvas/{canvas_id}")
def get_canvas_content(canvas_id: str) -> str:
    """获取指定画布的完整内容。"""
    try:
        from src.collaboration.canvas.canvas_manager import get_canvas
        canvas = get_canvas(canvas_id)
        if canvas is None:
            return json.dumps({"error": f"Canvas '{canvas_id}' not found"})
        return json.dumps({
            "id": canvas.id,
            "topic": canvas.topic,
            "markdown": canvas.markdown,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────

app = mcp.streamable_http_app()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
