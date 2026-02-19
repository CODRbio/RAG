"""
多文档比较 API：选择多篇论文，生成结构化对比。
支持从会话引文聚合候选（GET /compare/candidates）与本地文库分页（GET /compare/papers）。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import settings
from src.collaboration.memory.session_memory import get_session_store
from src.log import get_logger
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()

logger = get_logger(__name__)

router = APIRouter(prefix="/compare", tags=["compare"])

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"


class CompareRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_length=2, max_length=5, description="要比较的 paper_id 列表 (2-5)")
    aspects: List[str] = Field(
        default_factory=lambda: ["objective", "methodology", "key_findings", "limitations"],
        description="比较维度",
    )
    llm_provider: Optional[str] = Field(None, description="LLM 提供商")
    model_override: Optional[str] = Field(None, description="覆盖默认模型")


class PaperSummary(BaseModel):
    paper_id: str = ""
    title: str = ""
    year: Optional[int] = None
    abstract: str = ""


class CompareResponse(BaseModel):
    papers: List[PaperSummary] = Field(default_factory=list)
    comparison_matrix: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="对比矩阵: {aspect: {paper_id: description}}",
    )
    narrative: str = Field("", description="LLM 生成的对比叙述")


class _CompareLLMResponse(BaseModel):
    matrix: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    narrative: str = ""


def _load_paper_data(paper_id: str) -> Optional[Dict[str, Any]]:
    """加载 enriched.json"""
    parsed_dir = settings.path.data / "parsed"
    # 尝试精确匹配和模糊匹配
    candidates = list(parsed_dir.glob(f"{paper_id}/enriched.json"))
    if not candidates:
        candidates = list(parsed_dir.glob(f"*{paper_id}*/enriched.json"))
    if not candidates:
        return None
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_paper_summary(paper_id: str, data: Dict[str, Any]) -> PaperSummary:
    """从 enriched.json 提取摘要信息"""
    if not isinstance(data, dict):
        data = {}

    # 尝试从 content_flow 找 abstract
    abstract = ""
    content_flow = data.get("content_flow")
    if not isinstance(content_flow, list):
        content_flow = []
    for block in content_flow:
        if not isinstance(block, dict):
            continue
        hp = block.get("heading_path", [])
        hp_lower = " ".join(str(h) for h in hp).lower()
        if "abstract" in hp_lower and block.get("text"):
            abstract = str(block.get("text") or "")[:500]
            break

    # 尝试从 global_summary 获取
    if not abstract:
        abstract = str(data.get("global_summary") or "")[:500]

    # 尝试从文件名解析年份
    year = None
    import re
    for part in paper_id.replace("-", "_").split("_"):
        if re.match(r"^(19|20)\d{2}$", part):
            year = int(part)
            break

    return PaperSummary(
        paper_id=paper_id,
        title=str(data.get("doc_id") or paper_id),
        year=year,
        abstract=abstract,
    )


def _extract_sections_text(data: Dict[str, Any], max_chars: int = 3000) -> str:
    """提取关键章节文本用于 LLM 比较"""
    if not isinstance(data, dict):
        return ""

    parts = []
    total = 0
    target_sections = ["abstract", "method", "result", "conclusion", "discussion", "finding"]

    content_flow = data.get("content_flow")
    if not isinstance(content_flow, list):
        content_flow = []
    for block in content_flow:
        if not isinstance(block, dict):
            continue
        hp = block.get("heading_path", [])
        hp_lower = " ".join(str(h) for h in hp).lower()
        text = str(block.get("text") or "")
        if not text.strip():
            continue
        if any(s in hp_lower for s in target_sections):
            if total + len(text) > max_chars:
                text = text[: max_chars - total]
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break

    return "\n\n".join(parts) if parts else ""




@router.post("", response_model=CompareResponse)
def compare_papers(body: CompareRequest) -> CompareResponse:
    """多文档对比：加载论文数据，LLM 生成结构化对比"""
    papers_data: List[tuple] = []
    summaries: List[PaperSummary] = []

    for pid in body.paper_ids:
        data = _load_paper_data(pid)
        if data is None:
            raise HTTPException(status_code=404, detail=f"论文 '{pid}' 未找到（请确认已解析）")
        papers_data.append((pid, data))
        summaries.append(_extract_paper_summary(pid, data))

    # 构建 LLM prompt
    paper_blocks = []
    for pid, data in papers_data:
        text = _extract_sections_text(data, max_chars=2000)
        if not text:
            text = "(No key sections extracted)"
        paper_blocks.append(f"--- Paper: {pid} ---\n{text}")

    aspects_str = ", ".join(body.aspects)
    prompt = _pm.render(
        "compare_papers.txt",
        n=len(body.paper_ids),
        aspects=aspects_str,
        paper_blocks="\n\n".join(paper_blocks),
    )

    # 调用 LLM
    from src.llm.llm_manager import get_manager
    manager = get_manager(str(_CONFIG_PATH))
    client = manager.get_client(body.llm_provider or None)

    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": _pm.render("compare_system.txt")},
                {"role": "user", "content": prompt},
            ],
            model=body.model_override or None,
            max_tokens=3000,
            response_model=_CompareLLMResponse,
        )
        parsed: Optional[_CompareLLMResponse] = resp.get("parsed_object")
        if parsed is None:
            raw = (resp.get("final_text") or "").strip()
            if raw:
                parsed = _CompareLLMResponse.model_validate_json(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {e}")

    matrix = parsed.matrix if parsed is not None else {}
    narrative = parsed.narrative if parsed is not None else (resp.get("final_text") or "")

    return CompareResponse(
        papers=summaries,
        comparison_matrix=matrix,
        narrative=narrative,
    )


def _citation_key(c: Dict[str, Any]) -> str:
    """Stable key for deduping: prefer doc_id/paper_id, then doi, url, title."""
    pid = (c.get("doc_id") or c.get("paper_id") or "").strip()
    if pid:
        return pid
    doi = (c.get("doi") or "").strip()
    if doi:
        return f"doi:{doi}"
    url = (c.get("url") or "").strip()
    if url:
        return f"url:{url}"
    title = (c.get("title") or "").strip() or "unknown"
    return f"title:{title}"


@router.get("/candidates")
def list_compare_candidates(
    session_id: str = Query(..., description="会话 ID"),
    scope: str = Query("session", description="session | current_turn"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """从会话引文中聚合对比候选：按 doc_id/doi/url/title 去重，带引用次数与最近引用轮次。"""
    store = get_session_store()
    meta = store.get_session_meta(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="session not found")

    turns = store.get_turns(session_id)
    # key -> { paper_id, title, year, abstract, citation_count, last_cited_turn_index, is_local_ready }
    agg: Dict[str, Dict[str, Any]] = {}

    for turn_index, t in enumerate(turns):
        for c in (t.citations or []):
            key = _citation_key(c)
            if not key or key.startswith("title:unknown"):
                continue
            paper_id = (c.get("doc_id") or c.get("paper_id") or key).strip()
            if key.startswith("doi:"):
                paper_id = key
            elif key.startswith("url:"):
                paper_id = key
            elif key.startswith("title:"):
                paper_id = key

            if key not in agg:
                year = c.get("year")
                if isinstance(year, str) and year.isdigit():
                    year = int(year)
                agg[key] = {
                    "paper_id": paper_id,
                    "title": (c.get("title") or paper_id),
                    "year": year,
                    "abstract": "",
                    "citation_count": 0,
                    "last_cited_turn_index": turn_index,
                    "is_local_ready": False,
                }
            agg[key]["citation_count"] += 1
            agg[key]["last_cited_turn_index"] = max(
                agg[key]["last_cited_turn_index"], turn_index
            )

    # Resolve paper_id for local check: use doc_id if it matches a parsed dir
    parsed_dir = settings.path.data / "parsed"
    for key, row in agg.items():
        pid = row["paper_id"]
        if pid.startswith(("doi:", "url:", "title:")):
            row["is_local_ready"] = False
            continue
        data = _load_paper_data(pid)
        if data is not None:
            row["is_local_ready"] = True
            if not row.get("abstract"):
                row["abstract"] = (
                    _extract_paper_summary(pid, data).abstract or ""
                )[:500]

    # Sort by citation_count desc, then last_cited_turn_index desc; then paginate
    items = sorted(
        agg.values(),
        key=lambda x: (-x["citation_count"], -x["last_cited_turn_index"]),
    )
    total = len(items)
    page = items[offset : offset + limit]

    return {"candidates": page, "total": total}


@router.get("/papers")
def list_available_papers(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, description="搜索标题/paper_id"),
) -> Dict[str, Any]:
    """列出所有可用于比较的论文，支持分页与搜索。"""
    parsed_dir = settings.path.data / "parsed"
    if not parsed_dir.exists():
        return {"papers": [], "total": 0}

    papers = []
    for enriched_path in sorted(parsed_dir.glob("*/enriched.json")):
        pid = enriched_path.parent.name
        try:
            with open(enriched_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = _extract_paper_summary(pid, data)
            papers.append(summary.model_dump())
        except Exception:
            papers.append({"paper_id": pid, "title": pid, "year": None, "abstract": ""})

    if q and q.strip():
        ql = q.strip().lower()
        papers = [
            p
            for p in papers
            if ql in (p.get("paper_id") or "").lower()
            or ql in (p.get("title") or "").lower()
        ]

    total = len(papers)
    page = papers[offset : offset + limit]
    return {"papers": page, "total": total}
