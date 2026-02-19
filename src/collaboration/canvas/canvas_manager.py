"""
Canvas 管理服务：CRUD、版本快照、导出；与 session 绑定。
"""

import json
from typing import Any, Dict, List, Optional

from src.collaboration.canvas.canvas_store import CanvasStore
from src.collaboration.canvas.models import Annotation, Citation, DraftBlock, OutlineSection, ResearchBrief, SurveyCanvas


def get_canvas_store() -> CanvasStore:
    return CanvasStore()


def create_canvas(session_id: str = "", topic: str = "", user_id: str = "") -> SurveyCanvas:
    store = get_canvas_store()
    return store.create(session_id=session_id, topic=topic, user_id=user_id)


def get_canvas(canvas_id: str) -> Optional[SurveyCanvas]:
    return get_canvas_store().get(canvas_id)


def update_canvas(canvas_id: str, **fields: Any) -> bool:
    return get_canvas_store().update(canvas_id, **fields)


def delete_canvas(canvas_id: str) -> bool:
    return get_canvas_store().delete(canvas_id)


def list_canvases(user_id: str = "", limit: int = 50) -> List[SurveyCanvas]:
    """列出所有画布（可选按用户过滤）。"""
    from sqlmodel import Session, select
    from src.db.engine import get_engine
    from src.db.models import Canvas as CanvasRow

    store = get_canvas_store()
    with Session(get_engine()) as session:
        stmt = select(CanvasRow.id).order_by(CanvasRow.updated_at.desc()).limit(limit)
        if user_id:
            stmt = select(CanvasRow.id).where(CanvasRow.user_id == user_id).order_by(CanvasRow.updated_at.desc()).limit(limit)
        ids = session.exec(stmt).all()

    canvases = []
    for cid in ids:
        canvas = store.get(cid)
        if canvas:
            canvases.append(canvas)
    return canvases


def upsert_outline(canvas_id: str, sections: List[OutlineSection]) -> None:
    get_canvas_store().upsert_outline(canvas_id, sections)


def upsert_draft(canvas_id: str, block: DraftBlock) -> None:
    get_canvas_store().upsert_draft(canvas_id, block)


def create_snapshot(canvas_id: str) -> int:
    return get_canvas_store().snapshot(canvas_id)


def restore_snapshot(canvas_id: str, version_number: int) -> bool:
    return get_canvas_store().restore(canvas_id, version_number)


def list_snapshots(canvas_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    return get_canvas_store().list_versions(canvas_id, limit=limit)


def get_canvas_citations(canvas_id: str) -> List[Citation]:
    """返回画布引用列表（Citation）。"""
    return get_canvas_store().get_citations(canvas_id)


def delete_canvas_citation(canvas_id: str, cite_key: str) -> bool:
    """删除画布中指定 cite_key 的引用。"""
    return get_canvas_store().delete_citation(canvas_id, cite_key)


def filter_canvas_citations(canvas_id: str, keep_keys: List[str]) -> int:
    """筛选引用池，仅保留 keep_keys 中的引用，其余删除。返回删除的数量。"""
    return get_canvas_store().filter_citations(canvas_id, keep_keys)


def export_canvas(canvas_id: str) -> Dict[str, Any]:
    """导出画布为可序列化 JSON 结构。"""
    c = get_canvas(canvas_id)
    if c is None:
        raise ValueError(f"canvas not found: {canvas_id}")
    citations = [
        {
            "id": cit.id,
            "cite_key": cit.cite_key or cit.id,
            "title": cit.title,
            "authors": cit.authors or [],
            "year": cit.year,
            "doi": cit.doi,
            "url": cit.url,
            "bibtex": cit.bibtex,
            "created_at": cit.created_at.isoformat(),
        }
        for cit in c.citation_pool.values()
    ]
    # 序列化 annotations
    annotations_out = []
    for ann in (c.annotations or []):
        annotations_out.append({
            "id": ann.id,
            "section_id": ann.section_id,
            "target_text": ann.target_text,
            "directive": ann.directive,
            "status": ann.status,
            "created_at": ann.created_at.isoformat(),
        })

    # 序列化 research_brief
    brief_out = None
    if c.research_brief:
        rb = c.research_brief
        brief_out = {
            "scope": rb.scope,
            "success_criteria": rb.success_criteria,
            "key_questions": rb.key_questions,
            "exclusions": rb.exclusions,
            "time_range": rb.time_range,
            "source_priority": rb.source_priority,
            "action_plan": rb.action_plan,
        }

    return {
        "id": c.id,
        "session_id": c.session_id,
        "topic": c.topic,
        "working_title": c.working_title,
        "abstract": c.abstract,
        "keywords": c.keywords,
        "stage": c.stage,
        "refined_markdown": c.refined_markdown,
        "outline": [
            {
                "id": s.id,
                "title": s.title,
                "level": s.level,
                "order": s.order,
                "parent_id": s.parent_id,
                "status": s.status,
                "guidance": s.guidance,
            }
            for s in c.outline
        ],
        "drafts": {
            sid: {
                "section_id": b.section_id,
                "content_md": b.content_md,
                "version": b.version,
                "used_fragment_ids": b.used_fragment_ids,
                "used_citation_ids": b.used_citation_ids,
                "updated_at": b.updated_at.isoformat(),
            }
            for sid, b in c.drafts.items()
        },
        "citation_pool": citations,
        "identified_gaps": c.identified_gaps,
        "user_directives": c.user_directives,
        "annotations": annotations_out,
        "research_brief": brief_out,
        "research_insights": c.research_insights or [],
        "skip_draft_review": c.skip_draft_review,
        "skip_refine_review": c.skip_refine_review,
        "version": c.version,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }
