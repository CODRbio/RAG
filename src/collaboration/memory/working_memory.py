"""
Working Memory：与 Canvas 绑定的状态摘要，由 LLM 生成并持久化。
底层存储已迁移至 data/rag.db (working_memory 表)，通过 SQLModel 访问。
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlmodel import Session, select

from src.collaboration.canvas.models import SurveyCanvas
from src.db.engine import get_engine
from src.db.models import WorkingMemory as WorkingMemoryRow
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()


def get_working_memory(canvas_id: str) -> Optional[Dict[str, Any]]:
    """返回缓存的 working memory，无则返回 None。"""
    if not canvas_id:
        return None
    with Session(get_engine()) as session:
        row = session.get(WorkingMemoryRow, canvas_id)
    if row is None:
        return None
    return {
        "canvas_id": row.canvas_id,
        "summary": row.summary or "",
        "meta": row.get_meta(),
        "updated_at": row.updated_at,
    }


def update_working_memory(canvas_id: str, summary: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """写入或更新 working memory 缓存。"""
    if not canvas_id:
        return
    now = datetime.now().isoformat()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with Session(get_engine()) as session:
        row = session.get(WorkingMemoryRow, canvas_id)
        if row is None:
            row = WorkingMemoryRow(
                canvas_id=canvas_id,
                summary=summary,
                meta_json=meta_json,
                updated_at=now,
            )
            session.add(row)
        else:
            row.summary = summary
            row.meta_json = meta_json
            row.updated_at = now
            session.add(row)
        session.commit()


def _canvas_to_context_string(canvas: SurveyCanvas) -> str:
    """把 Canvas 关键信息拼成供 LLM 用的文本。"""
    parts = [
        f"主题: {canvas.topic or '(未设)'}",
        f"阶段: {canvas.stage}",
        f"工作标题: {canvas.working_title or '(未设)'}",
    ]
    if canvas.outline:
        parts.append("大纲章节:")
        for s in canvas.outline:
            parts.append(f"  - [{s.id}] {s.title} (level={s.level}, status={s.status})")
    if canvas.drafts:
        parts.append("草稿概况:")
        for sid, b in canvas.drafts.items():
            parts.append(f"  - 章节 {sid}: {len(b.content_md)} 字, v{b.version}")
    if canvas.identified_gaps:
        parts.append("已识别缺口: " + "; ".join(canvas.identified_gaps[:5]))
    if canvas.research_insights:
        parts.append("研究洞察:")
        for ins in canvas.research_insights[:10]:
            parts.append(f"  - {ins}")
    return "\n".join(parts)


def generate_working_memory_summary(canvas_id: str, config_path=None) -> str:
    """
    根据 Canvas 用 LLM 生成摘要并写入缓存，返回 summary 文本。
    若 canvas 不存在则返回空字符串。
    """
    from src.collaboration.canvas.canvas_manager import get_canvas
    from src.llm.llm_manager import get_manager

    canvas = get_canvas(canvas_id)
    if canvas is None:
        return ""
    context = _canvas_to_context_string(canvas)
    prompt = _pm.render("working_memory_progress.txt", context=context)
    try:
        manager = get_manager(str(config_path) if config_path else None)
        client = manager.get_client()
        resp = client.chat(
            [
                {"role": "system", "content": _pm.render("working_memory_progress_system.txt")},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
        )
        summary = (resp.get("final_text") or "").strip()
    except Exception:
        summary = f"主题: {canvas.topic}，阶段: {canvas.stage}，共 {len(canvas.outline)} 个大纲章节，{len(canvas.drafts)} 个草稿。"
    meta: Dict[str, Any] = {
        "stage": canvas.stage,
        "outline_count": len(canvas.outline),
        "draft_count": len(canvas.drafts),
    }
    if canvas.research_insights:
        meta["research_insights"] = canvas.research_insights[:20]
    update_working_memory(canvas_id, summary, meta)
    return summary


def get_or_generate_working_memory(canvas_id: str, config_path=None) -> Optional[Dict[str, Any]]:
    """
    若已有缓存则返回，否则生成并缓存后返回。
    """
    if not canvas_id:
        return None
    cached = get_working_memory(canvas_id)
    if cached and cached.get("summary"):
        return cached
    summary = generate_working_memory_summary(canvas_id, config_path)
    if not summary:
        return None
    return get_working_memory(canvas_id)
