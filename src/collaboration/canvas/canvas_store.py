"""
Canvas SQLite 持久化：canvases / outline_sections / draft_blocks / canvas_versions / canvas_citations。
底层存储已迁移至 data/rag.db，通过 SQLModel 访问。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from src.collaboration.canvas.models import (
    Annotation,
    Citation,
    DraftBlock,
    OutlineSection,
    ResearchBrief,
    SurveyCanvas,
)
from src.db.engine import get_engine
from src.db.models import (
    Canvas,
    CanvasCitation,
    CanvasDraftBlock,
    CanvasOutlineSection,
    CanvasVersion,
)


class CanvasStore:
    def __init__(self, db_path: Optional[Path] = None):
        # db_path ignored — shared engine is used
        pass

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_outline(self, canvas_id: str) -> List[OutlineSection]:
        with Session(get_engine()) as session:
            stmt = (
                select(CanvasOutlineSection)
                .where(CanvasOutlineSection.canvas_id == canvas_id)
                .order_by(CanvasOutlineSection.order, CanvasOutlineSection.level)
            )
            rows = session.exec(stmt).all()
        return [
            OutlineSection(
                id=r.id,
                title=r.title or "",
                level=int(r.level or 1),
                order=int(r.order or 0),
                parent_id=r.parent_id,
                status=r.status or "todo",
                guidance=r.guidance,
            )
            for r in rows
        ]

    def _get_drafts(self, canvas_id: str) -> Dict[str, DraftBlock]:
        with Session(get_engine()) as session:
            stmt = select(CanvasDraftBlock).where(CanvasDraftBlock.canvas_id == canvas_id)
            rows = session.exec(stmt).all()
        out = {}
        for r in rows:
            out[r.section_id] = DraftBlock(
                section_id=r.section_id,
                content_md=r.content_md or "",
                version=int(r.version or 1),
                used_fragment_ids=r.get_used_fragment_ids(),
                used_citation_ids=r.get_used_citation_ids(),
                updated_at=datetime.fromisoformat(r.updated_at),
            )
        return out

    def _row_to_canvas(self, row: Canvas) -> SurveyCanvas:
        brief_raw = row.get_research_brief()
        brief_obj = None
        if isinstance(brief_raw, dict) and brief_raw:
            brief_obj = ResearchBrief(
                scope=brief_raw.get("scope", ""),
                success_criteria=brief_raw.get("success_criteria", []),
                key_questions=brief_raw.get("key_questions", []),
                exclusions=brief_raw.get("exclusions", []),
                time_range=brief_raw.get("time_range", ""),
                source_priority=brief_raw.get("source_priority", []),
                action_plan=brief_raw.get("action_plan", ""),
            )
        outline = self._get_outline(row.id)
        drafts = self._get_drafts(row.id)
        citation_pool = {c.cite_key or c.id: c for c in self.get_citations(row.id)}
        return SurveyCanvas(
            id=row.id,
            session_id=row.session_id or "",
            topic=row.topic or "",
            working_title=row.working_title or "",
            abstract=row.abstract or "",
            keywords=row.get_keywords(),
            stage=row.stage or "explore",
            refined_markdown=row.refined_markdown or "",
            outline=outline,
            drafts=drafts,
            citation_pool=citation_pool,
            knowledge_pool={},
            identified_gaps=row.get_identified_gaps(),
            user_directives=row.get_user_directives(),
            annotations=[],
            research_brief=brief_obj,
            research_insights=row.get_research_insights(),
            skip_draft_review=bool(row.skip_draft_review),
            skip_refine_review=bool(row.skip_refine_review),
            version=int(row.version or 1),
            created_at=datetime.fromisoformat(row.created_at),
            updated_at=datetime.fromisoformat(row.updated_at),
        )

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, session_id: str = "", topic: str = "", user_id: str = "") -> SurveyCanvas:
        cid = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            row = Canvas(
                id=cid,
                session_id=session_id,
                topic=topic,
                archived=0,
                user_id=user_id or "",
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            session.commit()
        return self.get(cid) or SurveyCanvas(id=cid, session_id=session_id, topic=topic)

    def get(self, canvas_id: str) -> Optional[SurveyCanvas]:
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
        if row is None:
            return None
        return self._row_to_canvas(row)

    def update(self, canvas_id: str, **fields: Any) -> bool:
        allowed = {
            "session_id", "topic", "working_title", "abstract", "keywords",
            "stage", "identified_gaps", "user_directives", "research_brief",
            "research_insights", "refined_markdown", "skip_draft_review",
            "skip_refine_review", "version",
        }
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
            if not row:
                return False
            for k, v in fields.items():
                if k not in allowed:
                    continue
                if k in ("keywords", "identified_gaps", "user_directives", "research_insights") and isinstance(v, list):
                    v = json.dumps(v, ensure_ascii=False)
                if k == "research_brief" and isinstance(v, dict):
                    v = json.dumps(v, ensure_ascii=False)
                if k in ("skip_draft_review", "skip_refine_review"):
                    v = 1 if bool(v) else 0
                setattr(row, k, v)
            row.updated_at = datetime.now().isoformat()
            session.add(row)
            session.commit()
        return True

    def get_citations(self, canvas_id: str) -> List[Citation]:
        with Session(get_engine()) as session:
            stmt = select(CanvasCitation).where(CanvasCitation.canvas_id == canvas_id)
            rows = session.exec(stmt).all()
        return [
            Citation(
                id=r.citation_id,
                cite_key=r.cite_key,
                title=r.title or "",
                authors=r.get_authors(),
                year=int(r.year) if r.year is not None else None,
                doi=r.doi,
                url=r.url,
                bibtex=r.bibtex,
                created_at=datetime.fromisoformat(r.created_at),
            )
            for r in rows
        ]

    def upsert_citations(self, canvas_id: str, citations: List[Citation]) -> None:
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            # Delete existing
            stmt = select(CanvasCitation).where(CanvasCitation.canvas_id == canvas_id)
            for old in session.exec(stmt).all():
                session.delete(old)

            for c in citations:
                key = c.cite_key or c.id
                row = CanvasCitation(
                    canvas_id=canvas_id,
                    cite_key=key,
                    citation_id=c.id,
                    title=c.title or "",
                    authors_json=json.dumps(c.authors or [], ensure_ascii=False),
                    year=c.year,
                    doi=c.doi,
                    url=c.url,
                    bibtex=c.bibtex,
                    created_at=(c.created_at if c.created_at else datetime.now()).isoformat(),
                )
                session.add(row)

            # Touch updated_at on canvas
            canvas_row = session.get(Canvas, canvas_id)
            if canvas_row:
                canvas_row.updated_at = now
                session.add(canvas_row)

            session.commit()

    def delete_citation(self, canvas_id: str, cite_key: str) -> bool:
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            stmt = select(CanvasCitation).where(
                CanvasCitation.canvas_id == canvas_id,
                CanvasCitation.cite_key == cite_key,
            )
            row = session.exec(stmt).first()
            if not row:
                return False
            session.delete(row)
            canvas_row = session.get(Canvas, canvas_id)
            if canvas_row:
                canvas_row.updated_at = now
                session.add(canvas_row)
            session.commit()
        return True

    def filter_citations(self, canvas_id: str, keep_keys: List[str]) -> int:
        existing = self.get_citations(canvas_id)
        keep_set = set(keep_keys)
        to_remove = [c.cite_key or c.id for c in existing if (c.cite_key or c.id) not in keep_set]
        removed = 0
        for key in to_remove:
            if self.delete_citation(canvas_id, key):
                removed += 1
        return removed

    def delete(self, canvas_id: str) -> bool:
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
            if not row:
                return False
            session.delete(row)  # cascade deletes all child tables
            session.commit()
        return True

    def archive(self, canvas_id: str) -> bool:
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
            if not row:
                return False
            row.archived = 1
            row.updated_at = datetime.now().isoformat()
            session.add(row)
            session.commit()
        return True

    def unarchive(self, canvas_id: str) -> bool:
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
            if not row:
                return False
            row.archived = 0
            row.updated_at = datetime.now().isoformat()
            session.add(row)
            session.commit()
        return True

    def get_canvas_owner(self, canvas_id: str) -> Optional[str]:
        with Session(get_engine()) as session:
            row = session.get(Canvas, canvas_id)
        if not row:
            return None
        return (row.user_id or "").strip() or None

    def list_by_user(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        with Session(get_engine()) as session:
            stmt = (
                select(Canvas)
                .where(Canvas.user_id == user_id)
                .order_by(Canvas.updated_at.desc())
            )
            if not include_archived:
                stmt = stmt.where((Canvas.archived == 0) | (Canvas.archived == None))
            rows = session.exec(stmt).all()
        return [
            {
                "id": r.id,
                "title": r.working_title or r.topic or "",
                "topic": r.topic or "",
                "working_title": r.working_title or "",
                "stage": r.stage or "explore",
                "archived": bool(r.archived) if r.archived is not None else False,
                "session_id": r.session_id or "",
                "created_at": r.created_at or "",
                "updated_at": r.updated_at or "",
            }
            for r in rows
        ]

    def upsert_outline(self, canvas_id: str, sections: List[OutlineSection]) -> None:
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            # Delete existing sections
            stmt = select(CanvasOutlineSection).where(CanvasOutlineSection.canvas_id == canvas_id)
            for old in session.exec(stmt).all():
                session.delete(old)

            for s in sections:
                row = CanvasOutlineSection(
                    canvas_id=canvas_id,
                    id=s.id,
                    title=s.title,
                    level=s.level,
                    order=s.order,
                    parent_id=s.parent_id,
                    status=s.status,
                    guidance=s.guidance,
                )
                session.add(row)

            # Outline changed — clear refined_markdown
            canvas_row = session.get(Canvas, canvas_id)
            if canvas_row:
                canvas_row.refined_markdown = ""
                canvas_row.updated_at = now
                session.add(canvas_row)

            session.commit()

    def upsert_draft(self, canvas_id: str, block: DraftBlock) -> None:
        now = datetime.now().isoformat()
        with Session(get_engine()) as session:
            row = session.get(CanvasDraftBlock, (canvas_id, block.section_id))
            if row is None:
                row = CanvasDraftBlock(
                    canvas_id=canvas_id,
                    section_id=block.section_id,
                    content_md=block.content_md,
                    version=block.version,
                    used_fragment_ids=json.dumps(block.used_fragment_ids, ensure_ascii=False),
                    used_citation_ids=json.dumps(block.used_citation_ids, ensure_ascii=False),
                    updated_at=block.updated_at.isoformat(),
                )
                session.add(row)
            else:
                row.content_md = block.content_md
                row.version = block.version
                row.used_fragment_ids = json.dumps(block.used_fragment_ids, ensure_ascii=False)
                row.used_citation_ids = json.dumps(block.used_citation_ids, ensure_ascii=False)
                row.updated_at = block.updated_at.isoformat()
                session.add(row)

            # Draft changed — clear refined_markdown
            canvas_row = session.get(Canvas, canvas_id)
            if canvas_row:
                canvas_row.refined_markdown = ""
                canvas_row.updated_at = now
                session.add(canvas_row)

            session.commit()

    def snapshot(self, canvas_id: str) -> int:
        c = self.get(canvas_id)
        if c is None:
            raise ValueError(f"canvas not found: {canvas_id}")
        snap = _canvas_to_snapshot_dict(c)
        with Session(get_engine()) as session:
            stmt = (
                select(CanvasVersion)
                .where(CanvasVersion.canvas_id == canvas_id)
                .order_by(CanvasVersion.version_number.desc())
                .limit(1)
            )
            last = session.exec(stmt).first()
            ver = (last.version_number + 1) if last else 1
            row = CanvasVersion(
                canvas_id=canvas_id,
                version_number=ver,
                snapshot_json=json.dumps(snap, ensure_ascii=False),
                created_at=datetime.now().isoformat(),
            )
            session.add(row)
            session.commit()
        return ver

    def restore(self, canvas_id: str, version_number: int) -> bool:
        with Session(get_engine()) as session:
            row = session.get(CanvasVersion, (canvas_id, version_number))
        if row is None:
            return False
        snap = json.loads(row.snapshot_json)
        _apply_snapshot(self, canvas_id, snap)
        return True

    def list_versions(self, canvas_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with Session(get_engine()) as session:
            stmt = (
                select(CanvasVersion)
                .where(CanvasVersion.canvas_id == canvas_id)
                .order_by(CanvasVersion.version_number.desc())
                .limit(max(1, int(limit)))
            )
            rows = session.exec(stmt).all()
        return [
            {"version_number": int(r.version_number), "created_at": str(r.created_at or "")}
            for r in rows
        ]


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _citation_to_dict(c: Citation) -> Dict[str, Any]:
    return {
        "id": c.id,
        "cite_key": c.cite_key or c.id,
        "title": c.title,
        "authors": c.authors or [],
        "year": c.year,
        "doi": c.doi,
        "url": c.url,
        "bibtex": c.bibtex,
        "created_at": c.created_at.isoformat(),
    }


def _canvas_to_snapshot_dict(c: SurveyCanvas) -> Dict[str, Any]:
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
                "id": s.id, "title": s.title, "level": s.level,
                "order": s.order, "parent_id": s.parent_id,
                "status": s.status, "guidance": s.guidance,
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
        "citation_pool": [_citation_to_dict(cit) for cit in c.citation_pool.values()],
        "identified_gaps": c.identified_gaps,
        "user_directives": c.user_directives,
        "research_brief": {
            "scope": c.research_brief.scope,
            "success_criteria": c.research_brief.success_criteria,
            "key_questions": c.research_brief.key_questions,
            "exclusions": c.research_brief.exclusions,
            "time_range": c.research_brief.time_range,
            "source_priority": c.research_brief.source_priority,
            "action_plan": c.research_brief.action_plan,
        } if c.research_brief else {},
        "skip_draft_review": c.skip_draft_review,
        "skip_refine_review": c.skip_refine_review,
        "version": c.version,
    }


def _apply_snapshot(store: CanvasStore, canvas_id: str, snap: Dict[str, Any]) -> None:
    store.update(
        canvas_id,
        topic=snap.get("topic", ""),
        working_title=snap.get("working_title", ""),
        abstract=snap.get("abstract", ""),
        keywords=snap.get("keywords", []),
        stage=snap.get("stage", "explore"),
        refined_markdown=snap.get("refined_markdown", ""),
        identified_gaps=snap.get("identified_gaps", []),
        user_directives=snap.get("user_directives", []),
        research_brief=snap.get("research_brief", {}),
        skip_draft_review=bool(snap.get("skip_draft_review", False)),
        skip_refine_review=bool(snap.get("skip_refine_review", False)),
        version=snap.get("version", 1),
    )
    outline = [
        OutlineSection(
            id=s["id"],
            title=s.get("title", ""),
            level=int(s.get("level", 1)),
            order=int(s.get("order", 0)),
            parent_id=s.get("parent_id"),
            status=s.get("status", "todo"),
            guidance=s.get("guidance"),
        )
        for s in snap.get("outline", [])
    ]
    store.upsert_outline(canvas_id, outline)
    for sid, d in snap.get("drafts", {}).items():
        store.upsert_draft(
            canvas_id,
            DraftBlock(
                section_id=sid,
                content_md=d.get("content_md", ""),
                version=int(d.get("version", 1)),
                used_fragment_ids=d.get("used_fragment_ids", []),
                used_citation_ids=d.get("used_citation_ids", []),
                updated_at=datetime.fromisoformat(d.get("updated_at", datetime.now().isoformat())),
            ),
        )
    if snap.get("citation_pool"):
        all_cits = [
            Citation(
                id=d.get("id", ""),
                cite_key=d.get("cite_key") or d.get("id"),
                title=d.get("title", ""),
                authors=d.get("authors", []),
                year=d.get("year"),
                doi=d.get("doi"),
                url=d.get("url"),
                bibtex=d.get("bibtex"),
                created_at=datetime.fromisoformat(d.get("created_at", datetime.now().isoformat())),
            )
            for d in snap["citation_pool"]
        ]
        store.upsert_citations(canvas_id, all_cits)
