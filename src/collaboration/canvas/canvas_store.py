"""
Canvas SQLite 持久化：canvases / outline_sections / draft_blocks / canvas_versions。
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.collaboration.canvas.models import (
    Annotation,
    Citation,
    DraftBlock,
    OutlineSection,
    ResearchBrief,
    SurveyCanvas,
)


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "canvas.db"


def _ensure_canvas_archive_columns(conn: sqlite3.Connection) -> None:
    """Add archived and user_id to canvases if missing (migration)."""
    cur = conn.execute("PRAGMA table_info(canvases)")
    columns = {row[1] for row in cur.fetchall()}
    if "archived" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    if "user_id" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
    if "research_brief" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN research_brief TEXT NOT NULL DEFAULT '{}'")
    if "skip_draft_review" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN skip_draft_review INTEGER NOT NULL DEFAULT 0")
    if "skip_refine_review" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN skip_refine_review INTEGER NOT NULL DEFAULT 0")
    if "research_insights" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN research_insights TEXT NOT NULL DEFAULT '[]'")
    if "refined_markdown" not in columns:
        conn.execute("ALTER TABLE canvases ADD COLUMN refined_markdown TEXT NOT NULL DEFAULT ''")


class CanvasStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canvases (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL DEFAULT '',
                    working_title TEXT NOT NULL DEFAULT '',
                    abstract TEXT NOT NULL DEFAULT '',
                    keywords TEXT NOT NULL DEFAULT '[]',
                    stage TEXT NOT NULL DEFAULT 'explore',
                    refined_markdown TEXT NOT NULL DEFAULT '',
                    identified_gaps TEXT NOT NULL DEFAULT '[]',
                    user_directives TEXT NOT NULL DEFAULT '[]',
                    research_brief TEXT NOT NULL DEFAULT '{}',
                    skip_draft_review INTEGER NOT NULL DEFAULT 0,
                    skip_refine_review INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            _ensure_canvas_archive_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outline_sections (
                    canvas_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    level INTEGER NOT NULL DEFAULT 1,
                    "order" INTEGER NOT NULL DEFAULT 0,
                    parent_id TEXT,
                    status TEXT NOT NULL DEFAULT 'todo',
                    guidance TEXT,
                    PRIMARY KEY (canvas_id, id),
                    FOREIGN KEY (canvas_id) REFERENCES canvases(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS draft_blocks (
                    canvas_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    content_md TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 1,
                    used_fragment_ids TEXT NOT NULL DEFAULT '[]',
                    used_citation_ids TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (canvas_id, section_id),
                    FOREIGN KEY (canvas_id) REFERENCES canvases(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canvas_versions (
                    canvas_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (canvas_id, version_number),
                    FOREIGN KEY (canvas_id) REFERENCES canvases(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canvas_citations (
                    canvas_id TEXT NOT NULL,
                    citation_id TEXT NOT NULL,
                    cite_key TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    year INTEGER,
                    doi TEXT,
                    url TEXT,
                    bibtex TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (canvas_id, cite_key),
                    FOREIGN KEY (canvas_id) REFERENCES canvases(id)
                )
                """
            )
            conn.commit()

    def create(self, session_id: str = "", topic: str = "", user_id: str = "") -> SurveyCanvas:
        cid = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            _ensure_canvas_archive_columns(conn)
            conn.execute(
                """INSERT INTO canvases (id, session_id, topic, created_at, updated_at, archived, user_id)
                   VALUES (?, ?, ?, ?, ?, 0, ?)""",
                (cid, session_id, topic, now, now, user_id or ""),
            )
            conn.commit()
        return self.get(cid) or SurveyCanvas(id=cid, session_id=session_id, topic=topic)

    def get(self, canvas_id: str) -> Optional[SurveyCanvas]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            _ensure_canvas_archive_columns(conn)
            row = conn.execute(
                "SELECT id, session_id, topic, working_title, abstract, keywords, stage, refined_markdown, "
                "identified_gaps, user_directives, research_brief, research_insights, "
                "skip_draft_review, skip_refine_review, "
                "version, created_at, updated_at FROM canvases WHERE id = ?",
                (canvas_id,),
            ).fetchone()
        if row is None:
            return None
        kw = json.loads(row["keywords"] or "[]")
        gaps = json.loads(row["identified_gaps"] or "[]")
        directives = json.loads(row["user_directives"] or "[]")
        brief_raw = json.loads(row["research_brief"] or "{}")
        insights = json.loads(row["research_insights"] or "[]") if "research_insights" in row.keys() else []
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
        outline = self._get_outline(canvas_id)
        drafts = self._get_drafts(canvas_id)
        citation_pool = {c.cite_key or c.id: c for c in self.get_citations(canvas_id)}
        return SurveyCanvas(
            id=row["id"],
            session_id=row["session_id"] or "",
            topic=row["topic"] or "",
            working_title=row["working_title"] or "",
            abstract=row["abstract"] or "",
            keywords=kw,
            stage=row["stage"] or "explore",
            refined_markdown=row["refined_markdown"] or "",
            outline=outline,
            drafts=drafts,
            citation_pool=citation_pool,
            knowledge_pool={},
            identified_gaps=gaps,
            user_directives=directives,
            annotations=[],
            research_brief=brief_obj,
            research_insights=insights,
            skip_draft_review=bool(row["skip_draft_review"]),
            skip_refine_review=bool(row["skip_refine_review"]),
            version=int(row["version"] or 1),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _get_outline(self, canvas_id: str) -> List[OutlineSection]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT id, title, level, "order", parent_id, status, guidance FROM outline_sections WHERE canvas_id = ? ORDER BY "order", level',
                (canvas_id,),
            ).fetchall()
        return [
            OutlineSection(
                id=r["id"],
                title=r["title"] or "",
                level=int(r["level"] or 1),
                order=int(r["order"] or 0),
                parent_id=r["parent_id"],
                status=r["status"] or "todo",
                guidance=r["guidance"],
            )
            for r in rows
        ]

    def _get_drafts(self, canvas_id: str) -> Dict[str, DraftBlock]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT section_id, content_md, version, used_fragment_ids, used_citation_ids, updated_at "
                "FROM draft_blocks WHERE canvas_id = ?",
                (canvas_id,),
            ).fetchall()
        out = {}
        for r in rows:
            sid = r["section_id"]
            out[sid] = DraftBlock(
                section_id=sid,
                content_md=r["content_md"] or "",
                version=int(r["version"] or 1),
                used_fragment_ids=json.loads(r["used_fragment_ids"] or "[]"),
                used_citation_ids=json.loads(r["used_citation_ids"] or "[]"),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
        return out

    def update(self, canvas_id: str, **fields: Any) -> bool:
        allowed = {
            "session_id",
            "topic",
            "working_title",
            "abstract",
            "keywords",
            "stage",
            "identified_gaps",
            "user_directives",
            "research_brief",
            "research_insights",
            "refined_markdown",
            "skip_draft_review",
            "skip_refine_review",
            "version",
        }
        updates = []
        params = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in ("keywords", "identified_gaps", "user_directives", "research_insights") and isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            if k == "research_brief" and isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            if k in ("skip_draft_review", "skip_refine_review"):
                v = 1 if bool(v) else 0
            updates.append(f"{k} = ?")
            params.append(v)
        if not updates:
            return True
        params.append(datetime.now().isoformat())
        params.append(canvas_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE canvases SET {', '.join(updates)}, updated_at = ? WHERE id = ?",
                params,
            )
            conn.commit()
        return True

    def get_citations(self, canvas_id: str) -> List[Citation]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT citation_id, cite_key, title, authors_json, year, doi, url, bibtex, created_at "
                "FROM canvas_citations WHERE canvas_id = ?",
                (canvas_id,),
            ).fetchall()
        return [
            Citation(
                id=r["citation_id"],
                cite_key=r["cite_key"],
                title=r["title"] or "",
                authors=json.loads(r["authors_json"] or "[]"),
                year=int(r["year"]) if r["year"] is not None else None,
                doi=r["doi"],
                url=r["url"],
                bibtex=r["bibtex"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    def upsert_citations(self, canvas_id: str, citations: List[Citation]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM canvas_citations WHERE canvas_id = ?", (canvas_id,))
            for c in citations:
                key = c.cite_key or c.id
                conn.execute(
                    """INSERT INTO canvas_citations (canvas_id, citation_id, cite_key, title, authors_json, year, doi, url, bibtex, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        canvas_id,
                        c.id,
                        key,
                        c.title or "",
                        json.dumps(c.authors or [], ensure_ascii=False),
                        c.year,
                        c.doi,
                        c.url,
                        c.bibtex,
                        (c.created_at if c.created_at else datetime.now()).isoformat(),
                    ),
                )
            conn.execute("UPDATE canvases SET updated_at = ? WHERE id = ?", (datetime.now().isoformat(), canvas_id))
            conn.commit()

    def delete_citation(self, canvas_id: str, cite_key: str) -> bool:
        """删除指定 cite_key 的引用。"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM canvas_citations WHERE canvas_id = ? AND cite_key = ?",
                (canvas_id, cite_key),
            )
            if cur.rowcount > 0:
                conn.execute(
                    "UPDATE canvases SET updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), canvas_id),
                )
            conn.commit()
        return cur.rowcount > 0

    def filter_citations(self, canvas_id: str, keep_keys: List[str]) -> int:
        """
        筛选引用池，仅保留 keep_keys 中的引用，其余删除。
        返回删除的数量。
        """
        existing = self.get_citations(canvas_id)
        keep_set = set(keep_keys)
        to_remove = [c.cite_key or c.id for c in existing if (c.cite_key or c.id) not in keep_set]
        removed = 0
        for key in to_remove:
            if self.delete_citation(canvas_id, key):
                removed += 1
        return removed

    def delete(self, canvas_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM canvas_versions WHERE canvas_id = ?", (canvas_id,))
            conn.execute("DELETE FROM canvas_citations WHERE canvas_id = ?", (canvas_id,))
            conn.execute("DELETE FROM draft_blocks WHERE canvas_id = ?", (canvas_id,))
            conn.execute("DELETE FROM outline_sections WHERE canvas_id = ?", (canvas_id,))
            cur = conn.execute("DELETE FROM canvases WHERE id = ?", (canvas_id,))
            conn.commit()
        return cur.rowcount > 0

    def archive(self, canvas_id: str) -> bool:
        """Mark canvas as archived (excluded from lifecycle cleanup)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("UPDATE canvases SET archived = 1, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), canvas_id))
            conn.commit()
        return cur.rowcount > 0

    def unarchive(self, canvas_id: str) -> bool:
        """Clear archived flag."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("UPDATE canvases SET archived = 0, updated_at = ? WHERE id = ?", (datetime.now().isoformat(), canvas_id))
            conn.commit()
        return cur.rowcount > 0

    def get_canvas_owner(self, canvas_id: str) -> Optional[str]:
        """Return user_id that owns the canvas, or None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            _ensure_canvas_archive_columns(conn)
            row = conn.execute("SELECT user_id FROM canvases WHERE id = ?", (canvas_id,)).fetchone()
        if not row:
            return None
        return (row[0] or "").strip() or None

    def list_by_user(self, user_id: str, include_archived: bool = False) -> List[Dict[str, Any]]:
        """List canvases for user. Returns list of {id, title, topic, working_title, stage, archived, created_at, updated_at, session_id}."""
        with sqlite3.connect(self.db_path) as conn:
            _ensure_canvas_archive_columns(conn)
            conn.row_factory = sqlite3.Row
            if include_archived:
                rows = conn.execute(
                    "SELECT id, topic, working_title, stage, archived, session_id, created_at, updated_at FROM canvases WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, topic, working_title, stage, archived, session_id, created_at, updated_at FROM canvases WHERE user_id = ? AND (archived = 0 OR archived IS NULL) ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["working_title"] or r["topic"] or "",  # 前端期望 title 字段
                "topic": r["topic"] or "",
                "working_title": r["working_title"] or "",
                "stage": r["stage"] or "explore",
                "archived": bool(r["archived"]) if r["archived"] is not None else False,
                "session_id": r["session_id"] or "",
                "created_at": r["created_at"] or "",
                "updated_at": r["updated_at"] or "",
            }
            for r in rows
        ]

    def upsert_outline(self, canvas_id: str, sections: List[OutlineSection]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM outline_sections WHERE canvas_id = ?", (canvas_id,))
            for s in sections:
                conn.execute(
                    """INSERT INTO outline_sections (canvas_id, id, title, level, "order", parent_id, status, guidance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (canvas_id, s.id, s.title, s.level, s.order, s.parent_id, s.status, s.guidance),
                )
            # Outline changed; existing full-refine markdown may be stale.
            conn.execute(
                "UPDATE canvases SET refined_markdown = '', updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), canvas_id),
            )
            conn.commit()

    def upsert_draft(self, canvas_id: str, block: DraftBlock) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO draft_blocks (canvas_id, section_id, content_md, version, used_fragment_ids, used_citation_ids, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    canvas_id,
                    block.section_id,
                    block.content_md,
                    block.version,
                    json.dumps(block.used_fragment_ids, ensure_ascii=False),
                    json.dumps(block.used_citation_ids, ensure_ascii=False),
                    block.updated_at.isoformat(),
                ),
            )
            # Draft changed; existing full-refine markdown may be stale.
            conn.execute(
                "UPDATE canvases SET refined_markdown = '', updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), canvas_id),
            )
            conn.commit()

    def snapshot(self, canvas_id: str) -> int:
        c = self.get(canvas_id)
        if c is None:
            raise ValueError(f"canvas not found: {canvas_id}")
        snap = _canvas_to_snapshot_dict(c)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM canvas_versions WHERE canvas_id = ?",
                (canvas_id,),
            )
            ver = cur.fetchone()[0]
            conn.execute(
                "INSERT INTO canvas_versions (canvas_id, version_number, snapshot_json, created_at) VALUES (?, ?, ?, ?)",
                (canvas_id, ver, json.dumps(snap, ensure_ascii=False), datetime.now().isoformat()),
            )
            conn.commit()
        return ver

    def restore(self, canvas_id: str, version_number: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT snapshot_json FROM canvas_versions WHERE canvas_id = ? AND version_number = ?",
                (canvas_id, version_number),
            ).fetchone()
        if row is None:
            return False
        snap = json.loads(row["snapshot_json"])
        _apply_snapshot(self, canvas_id, snap)
        return True

    def list_versions(self, canvas_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT version_number, created_at FROM canvas_versions WHERE canvas_id = ? ORDER BY version_number DESC LIMIT ?",
                (canvas_id, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "version_number": int(r["version_number"]),
                "created_at": str(r["created_at"] or ""),
            }
            for r in rows
        ]


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
            {"id": s.id, "title": s.title, "level": s.level, "order": s.order, "parent_id": s.parent_id, "status": s.status, "guidance": s.guidance}
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
