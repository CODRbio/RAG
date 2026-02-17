"""
Working Memory：与 Canvas 绑定的状态摘要，由 LLM 生成并 SQLite 缓存。
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.collaboration.canvas.models import SurveyCanvas
from src.collaboration.canvas.canvas_manager import get_canvas
from src.llm.llm_manager import get_manager


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "working_memory.db"


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS working_memory (
            canvas_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            meta_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_working_memory(canvas_id: str) -> Optional[Dict[str, Any]]:
    """返回缓存的 working memory，无则返回 None。"""
    if not canvas_id:
        return None
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT canvas_id, summary, meta_json, updated_at FROM working_memory WHERE canvas_id = ?",
            (canvas_id,),
        ).fetchone()
    if row is None:
        return None
    meta = {}
    if row["meta_json"]:
        try:
            meta = json.loads(row["meta_json"])
        except Exception:
            pass
    return {
        "canvas_id": row["canvas_id"],
        "summary": row["summary"] or "",
        "meta": meta,
        "updated_at": row["updated_at"],
    }


def update_working_memory(canvas_id: str, summary: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """写入或更新 working memory 缓存。"""
    if not canvas_id:
        return
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.execute(
            """INSERT INTO working_memory (canvas_id, summary, meta_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(canvas_id) DO UPDATE SET summary = ?, meta_json = ?, updated_at = ?""",
            (canvas_id, summary, meta_json, now, summary, meta_json, now),
        )
        conn.commit()


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


def generate_working_memory_summary(canvas_id: str, config_path: Optional[Path] = None) -> str:
    """
    根据 Canvas 用 LLM 生成摘要并写入缓存，返回 summary 文本。
    若 canvas 不存在则返回空字符串。
    """
    canvas = get_canvas(canvas_id)
    if canvas is None:
        return ""
    context = _canvas_to_context_string(canvas)
    prompt = f"""你是一个学术写作助手。请根据以下综述画布状态，用 2-4 句话概括当前进度与下一步建议。只输出概括内容，不要其他解释。

画布状态：
{context}
"""
    try:
        manager = get_manager(str(config_path) if config_path else None)
        client = manager.get_client()
        resp = client.chat(
            [
                {"role": "system", "content": "你只输出简短概括，不要 markdown 或标题。"},
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
    # Include open research insights in meta for cross-session reuse
    if canvas.research_insights:
        meta["research_insights"] = canvas.research_insights[:20]
    update_working_memory(canvas_id, summary, meta)
    return summary


def get_or_generate_working_memory(canvas_id: str, config_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
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
