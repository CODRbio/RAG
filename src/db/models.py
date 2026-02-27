"""
SQLModel table definitions — single source of truth for all 21 tables
that were previously spread across 8 separate SQLite databases.

Design rules for SQLModel compatibility:
  - primary_key=True and foreign_key="..." must be set in Field() only,
    never combined with sa_column (SQLModel raises RuntimeError otherwise).
  - JSON list/dict columns stay as TEXT with Python-side serialization
    so SQLite and PostgreSQL (JSONB swap) are both supported transparently.
  - ForeignKey relationships use cascade="all, delete-orphan" to enable
    cross-entity cascade deletes impossible with separate DBs.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Float, Index, Integer, Text, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def _now_iso() -> str:
    return datetime.now().isoformat()


def _now_ts() -> float:
    import time
    return time.time()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Canvas  (formerly canvas.db)
# ──────────────────────────────────────────────────────────────────────────────

class Canvas(SQLModel, table=True):
    __tablename__ = "canvases"

    id: str = Field(primary_key=True)
    session_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    topic: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    working_title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    abstract: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    keywords: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    stage: str = Field(default="explore", sa_column=Column(Text, nullable=False, server_default="explore"))
    refined_markdown: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    identified_gaps: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    user_directives: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    research_brief: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    research_insights: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    skip_draft_review: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    skip_refine_review: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    archived: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    user_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False, server_default="1"))
    created_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    outline_sections: List["CanvasOutlineSection"] = Relationship(
        back_populates="canvas",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    draft_blocks: List["CanvasDraftBlock"] = Relationship(
        back_populates="canvas",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    versions: List["CanvasVersion"] = Relationship(
        back_populates="canvas",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    citations: List["CanvasCitation"] = Relationship(
        back_populates="canvas",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    def get_keywords(self) -> List[str]:
        try:
            return json.loads(self.keywords or "[]")
        except Exception:
            return []

    def get_identified_gaps(self) -> List[str]:
        try:
            return json.loads(self.identified_gaps or "[]")
        except Exception:
            return []

    def get_user_directives(self) -> List[str]:
        try:
            return json.loads(self.user_directives or "[]")
        except Exception:
            return []

    def get_research_brief(self) -> Dict[str, Any]:
        try:
            return json.loads(self.research_brief or "{}")
        except Exception:
            return {}

    def get_research_insights(self) -> List[str]:
        try:
            return json.loads(self.research_insights or "[]")
        except Exception:
            return []


class CanvasOutlineSection(SQLModel, table=True):
    __tablename__ = "outline_sections"

    # composite PK — must use Field(primary_key=True), no sa_column on PK/FK cols
    canvas_id: str = Field(foreign_key="canvases.id", primary_key=True)
    id: str = Field(primary_key=True)
    title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    level: int = Field(default=1, sa_column=Column(Integer, nullable=False, server_default="1"))
    order: int = Field(default=0, sa_column=Column("order", Integer, nullable=False, server_default="0"))
    parent_id: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    status: str = Field(default="todo", sa_column=Column(Text, nullable=False, server_default="todo"))
    guidance: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    canvas: Optional[Canvas] = Relationship(back_populates="outline_sections")


class CanvasDraftBlock(SQLModel, table=True):
    __tablename__ = "draft_blocks"

    canvas_id: str = Field(foreign_key="canvases.id", primary_key=True)
    section_id: str = Field(primary_key=True)
    content_md: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    version: int = Field(default=1, sa_column=Column(Integer, nullable=False, server_default="1"))
    used_fragment_ids: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    used_citation_ids: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    canvas: Optional[Canvas] = Relationship(back_populates="draft_blocks")

    def get_used_fragment_ids(self) -> List[str]:
        try:
            return json.loads(self.used_fragment_ids or "[]")
        except Exception:
            return []

    def get_used_citation_ids(self) -> List[str]:
        try:
            return json.loads(self.used_citation_ids or "[]")
        except Exception:
            return []


class CanvasVersion(SQLModel, table=True):
    __tablename__ = "canvas_versions"

    canvas_id: str = Field(foreign_key="canvases.id", primary_key=True)
    version_number: int = Field(primary_key=True)
    snapshot_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    created_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    canvas: Optional[Canvas] = Relationship(back_populates="versions")


class CanvasCitation(SQLModel, table=True):
    __tablename__ = "canvas_citations"

    canvas_id: str = Field(foreign_key="canvases.id", primary_key=True)
    cite_key: str = Field(primary_key=True)
    citation_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    authors_json: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    year: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    doi: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    url: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    bibtex: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    canvas: Optional[Canvas] = Relationship(back_populates="citations")

    def get_authors(self) -> List[str]:
        try:
            return json.loads(self.authors_json or "[]")
        except Exception:
            return []


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sessions  (formerly sessions.db)
# ──────────────────────────────────────────────────────────────────────────────

class ChatSession(SQLModel, table=True):
    __tablename__ = "sessions"

    session_id: str = Field(primary_key=True)
    canvas_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    stage: str = Field(default="explore", sa_column=Column(Text, nullable=False, server_default="explore"))
    rolling_summary: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    summary_at_turn: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    created_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    turns: List["Turn"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Turn(SQLModel, table=True):
    __tablename__ = "turns"

    session_id: str = Field(foreign_key="sessions.session_id", primary_key=True)
    turn_index: int = Field(primary_key=True)
    role: str = Field(sa_column=Column(Text, nullable=False))
    content: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    intent: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    evidence_pack_id: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    canvas_patch: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    citations_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    timestamp: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    session: Optional[ChatSession] = Relationship(back_populates="turns")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Users  (formerly persistent.db)
# ──────────────────────────────────────────────────────────────────────────────

class UserProfile(SQLModel, table=True):
    __tablename__ = "user_profiles"

    user_id: str = Field(primary_key=True)
    preferences_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    password_hash: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    role: str = Field(default="user", sa_column=Column(Text, nullable=False, server_default="user"))
    is_active: int = Field(default=1, sa_column=Column(Integer, nullable=False, server_default="1"))
    created_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    projects: List["UserProject"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    def get_preferences(self) -> Dict[str, Any]:
        try:
            return json.loads(self.preferences_json or "{}")
        except Exception:
            return {}


class UserProject(SQLModel, table=True):
    __tablename__ = "user_projects"

    user_id: str = Field(foreign_key="user_profiles.user_id", primary_key=True)
    canvas_id: str = Field(primary_key=True)
    title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    user: Optional[UserProfile] = Relationship(back_populates="projects")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Working Memory  (formerly working_memory.db)
# ──────────────────────────────────────────────────────────────────────────────

class WorkingMemory(SQLModel, table=True):
    __tablename__ = "working_memory"

    canvas_id: str = Field(primary_key=True)
    summary: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    meta_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    updated_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))

    def get_meta(self) -> Dict[str, Any]:
        try:
            return json.loads(self.meta_json or "{}")
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# 5. Papers  (formerly papers.db)
# ──────────────────────────────────────────────────────────────────────────────

class Paper(SQLModel, table=True):
    __tablename__ = "papers"
    __table_args__ = (
        UniqueConstraint("collection", "paper_id", name="uq_papers_collection_paper"),
        Index("idx_papers_collection", "collection"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    collection: str = Field(sa_column=Column(Text, nullable=False))
    paper_id: str = Field(sa_column=Column("paper_id", Text, nullable=False))
    filename: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    file_path: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    file_size: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    chunk_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    row_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    enrich_tables_enabled: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    enrich_figures_enabled: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    table_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    figure_count: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    table_success: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    figure_success: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    status: str = Field(default="done", sa_column=Column(Text, nullable=False, server_default="done"))
    error_message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    content_hash: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Ingest Jobs  (formerly ingest_jobs.db)
# ──────────────────────────────────────────────────────────────────────────────

class IngestJob(SQLModel, table=True):
    __tablename__ = "ingest_jobs"
    __table_args__ = (
        Index("idx_ingest_jobs_created_at", "created_at"),
    )

    job_id: str = Field(primary_key=True)
    collection: str = Field(sa_column=Column(Text, nullable=False))
    status: str = Field(default="pending", sa_column=Column(Text, nullable=False, server_default="pending"))
    total_files: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    processed_files: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    failed_files: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    total_chunks: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    total_upserted: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    current_file: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    current_stage: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    error_message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    payload_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    updated_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    finished_at: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    events: List["IngestJobEvent"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    def get_payload(self) -> Dict[str, Any]:
        try:
            return json.loads(self.payload_json or "{}")
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["payload"] = self.get_payload()
        d.pop("payload_json", None)
        return d


class IngestJobEvent(SQLModel, table=True):
    __tablename__ = "ingest_job_events"
    __table_args__ = (
        Index("idx_ingest_job_events_job_id_id", "job_id", "id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="ingest_jobs.job_id")
    event: str = Field(sa_column=Column(Text, nullable=False))
    data_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[IngestJob] = Relationship(back_populates="events")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Deep Research Jobs  (formerly deep_research_jobs.db)
# ──────────────────────────────────────────────────────────────────────────────

class DeepResearchJob(SQLModel, table=True):
    __tablename__ = "deep_research_jobs"
    __table_args__ = (
        Index("idx_dr_jobs_created_at", "created_at"),
    )

    job_id: str = Field(primary_key=True)
    topic: str = Field(sa_column=Column(Text, nullable=False))
    session_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    canvas_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    status: str = Field(default="pending", sa_column=Column(Text, nullable=False, server_default="pending"))
    current_stage: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    error_message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    request_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    result_markdown: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    result_citations: str = Field(default="[]", sa_column=Column(Text, nullable=False, server_default="[]"))
    result_dashboard: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    total_time_ms: float = Field(default=0.0, sa_column=Column(Float, nullable=False, server_default="0"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    updated_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    finished_at: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    events: List["DRJobEvent"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    section_reviews: List["DRSectionReview"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    resume_queue: List["DRResumeQueue"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    gap_supplements: List["DRGapSupplement"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    insights: List["DRInsight"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    checkpoints: List["DRCheckpoint"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        try:
            d["request"] = json.loads(d.pop("request_json", "{}") or "{}")
        except Exception:
            d["request"] = {}
        try:
            d["result_citations"] = json.loads(d.get("result_citations") or "[]")
        except Exception:
            d["result_citations"] = []
        try:
            d["result_dashboard"] = json.loads(d.get("result_dashboard") or "{}")
        except Exception:
            d["result_dashboard"] = {}
        d.pop("request_json", None)
        return d


class DRJobEvent(SQLModel, table=True):
    __tablename__ = "deep_research_job_events"
    __table_args__ = (
        Index("idx_dr_job_events_job_id_id", "job_id", "id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="deep_research_jobs.job_id")
    event: str = Field(sa_column=Column(Text, nullable=False))
    data_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[DeepResearchJob] = Relationship(back_populates="events")


class DRSectionReview(SQLModel, table=True):
    __tablename__ = "deep_research_section_reviews"
    __table_args__ = (
        UniqueConstraint("job_id", "section_id", name="uq_dr_section_reviews"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="deep_research_jobs.job_id")
    section_id: str = Field(sa_column=Column(Text, nullable=False))
    action: str = Field(default="approve", sa_column=Column(Text, nullable=False, server_default="approve"))
    feedback: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[DeepResearchJob] = Relationship(back_populates="section_reviews")


class DRResumeQueue(SQLModel, table=True):
    __tablename__ = "deep_research_resume_queue"
    __table_args__ = (
        Index("idx_dr_resume_queue_status_created", "status", "created_at"),
        Index("idx_dr_resume_queue_owner_status", "owner_instance", "status", "created_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="deep_research_jobs.job_id")
    owner_instance: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    source: str = Field(default="review", sa_column=Column(Text, nullable=False, server_default="review"))
    status: str = Field(default="pending", sa_column=Column(Text, nullable=False, server_default="pending"))
    message: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    updated_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[DeepResearchJob] = Relationship(back_populates="resume_queue")


class DRGapSupplement(SQLModel, table=True):
    __tablename__ = "deep_research_gap_supplements"
    __table_args__ = (
        Index("idx_dr_gap_supplements_job_section", "job_id", "section_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="deep_research_jobs.job_id")
    section_id: str = Field(sa_column=Column(Text, nullable=False))
    gap_text: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    supplement_type: str = Field(default="material", sa_column=Column(Text, nullable=False, server_default="material"))
    content_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    status: str = Field(default="pending", sa_column=Column(Text, nullable=False, server_default="pending"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))
    consumed_at: Optional[float] = Field(default=None, sa_column=Column(Float, nullable=True))

    job: Optional[DeepResearchJob] = Relationship(back_populates="gap_supplements")

    def get_content(self) -> Dict[str, Any]:
        try:
            return json.loads(self.content_json or "{}")
        except Exception:
            return {}


class DRInsight(SQLModel, table=True):
    __tablename__ = "deep_research_insights"
    __table_args__ = (
        Index("idx_dr_insights_job_id", "job_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="deep_research_jobs.job_id")
    section_id: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    insight_type: str = Field(default="gap", sa_column=Column(Text, nullable=False, server_default="gap"))
    text: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    source_context: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    status: str = Field(default="open", sa_column=Column(Text, nullable=False, server_default="open"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[DeepResearchJob] = Relationship(back_populates="insights")


class DRCheckpoint(SQLModel, table=True):
    __tablename__ = "deep_research_checkpoints"
    __table_args__ = (
        Index("idx_dr_checkpoints_job_created", "job_id", "created_at"),
        Index("idx_dr_checkpoints_job_phase", "job_id", "phase"),
    )

    job_id: str = Field(foreign_key="deep_research_jobs.job_id", primary_key=True)
    phase: str = Field(primary_key=True)
    section_title: str = Field(default="", primary_key=True)
    state_json: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))

    job: Optional[DeepResearchJob] = Relationship(back_populates="checkpoints")

    def get_state(self) -> Dict[str, Any]:
        try:
            return json.loads(self.state_json or "{}")
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# 8. Paper Metadata  (formerly paper_metadata.db)
# ──────────────────────────────────────────────────────────────────────────────

class PaperMetadata(SQLModel, table=True):
    __tablename__ = "paper_metadata"
    __table_args__ = (
        Index("idx_pm_ndoi", "normalized_doi"),
        Index("idx_pm_ntitle", "normalized_title"),
    )

    paper_id: str = Field(primary_key=True)
    doi: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    normalized_doi: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    normalized_title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    authors: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    year: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    source: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    extra: str = Field(default="{}", sa_column=Column(Text, nullable=False, server_default="{}"))


class CrossrefCache(SQLModel, table=True):
    __tablename__ = "crossref_cache"

    normalized_title: str = Field(primary_key=True)
    doi: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    title: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    authors: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    year: Optional[int] = Field(default=None, sa_column=Column(Integer, nullable=True))
    venue: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    created_at: float = Field(default_factory=_now_ts, sa_column=Column(Float, nullable=False))


# ──────────────────────────────────────────────────────────────────────────────
# 9. Auth  (JWT token revocation list)
# ──────────────────────────────────────────────────────────────────────────────

class RevokedToken(SQLModel, table=True):
    """Stores SHA-256 hashes of explicitly revoked JWT tokens.

    Only invalidated tokens are stored here; normal validation never touches
    this table, keeping the hot path pure CPU (JWT decode + signature check).
    Rows whose `expires_at` is in the past can be safely purged.
    """

    __tablename__ = "revoked_tokens"
    __table_args__ = (
        Index("idx_revoked_tokens_expires_at", "expires_at"),
    )

    # SHA-256 hex digest of the raw JWT string — avoids storing the full token.
    token_hash: str = Field(sa_column=Column(Text, primary_key=True, nullable=False))
    # ISO-8601 timestamp copied from the JWT `exp` claim; used for cleanup.
    expires_at: str = Field(sa_column=Column(Text, nullable=False))
    revoked_at: str = Field(default_factory=_now_iso, sa_column=Column(Text, nullable=False))
