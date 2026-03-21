"""
Centralized SQLAlchemy/SQLModel engine and session factory.

All store modules import `engine` and `get_session` from here.
The database URL is resolved from config/rag_config.local.json or the
RAG_DATABASE_URL environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

_engine: Engine | None = None


def _resolve_db_url() -> str:
    """
    Resolve database URL with precedence:
    1. RAG_DATABASE_URL environment variable
    2. config/rag_config.local.json  database.url
    3. config/rag_config.json        database.url
    """
    env_url = os.environ.get("RAG_DATABASE_URL")
    if env_url:
        return env_url

    root = Path(__file__).resolve().parents[2]
    for cfg_name in ("rag_config.local.json", "rag_config.json"):
        cfg_path = root / "config" / cfg_name
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                url = cfg.get("database", {}).get("url")
                if url:
                    return url
            except Exception:
                pass

    raise RuntimeError(
        "No database URL configured. "
        "Set RAG_DATABASE_URL environment variable or add database.url to config/rag_config.local.json. "
        "Example: postgresql+psycopg://user:password@localhost:5432/rag"
    )


def get_resolved_db_url() -> str:
    """Return the effective database URL (for display in UI). Does not create the engine."""
    return _resolve_db_url()


def _config_root() -> Path:
    """Project config directory."""
    return Path(__file__).resolve().parents[2] / "config"


def get_local_config_path() -> Path:
    """Path to rag_config.local.json (for reading/writing database.url)."""
    return _config_root() / "rag_config.local.json"


def _normalize_db_url(url: str) -> str:
    """
    Normalize PostgreSQL URL variants to the psycopg driver form.

    - postgres://...     → postgresql+psycopg://...
    - postgresql://...   → postgresql+psycopg://...
    - postgresql+psycopg://... (already explicit) → unchanged
    """
    normalized = (url or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("postgres://"):
        normalized = "postgresql://" + normalized[len("postgres://"):]
    if normalized.startswith("postgresql://"):
        return "postgresql+psycopg://" + normalized[len("postgresql://"):]
    return normalized


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    db_url = _normalize_db_url(_resolve_db_url())

    _engine = create_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
    )

    return _engine


def get_session() -> Generator[Session, None, None]:
    """FastAPI-style dependency that yields a SQLModel session."""
    with Session(get_engine()) as session:
        yield session


def init_db() -> None:
    """
    Create all tables that are not yet present.
    Called once at application startup after models are imported.
    In production the Alembic migration already handles table creation;
    this is a safety net for tests and fresh installs.
    """
    from src.db import models as _models  # noqa: F401 — ensure all models are registered
    SQLModel.metadata.create_all(get_engine())
    _ensure_schema_updates()


def _ensure_schema_updates() -> None:
    """Add columns introduced after initial table creation (idempotent)."""
    import sqlalchemy as sa
    engine = get_engine()
    _add_column_if_missing = [
        ("deep_research_jobs", "started_at", "REAL"),
        ("sessions", "user_id", "TEXT NOT NULL DEFAULT ''"),
        ("canvases", "preliminary_knowledge", "TEXT NOT NULL DEFAULT ''"),
        ("scholar_libraries", "folder_path", "TEXT"),
        ("scholar_library_papers", "venue", "TEXT NOT NULL DEFAULT ''"),
        ("scholar_library_papers", "normalized_journal_name", "TEXT NOT NULL DEFAULT ''"),
        ("scholar_library_papers", "paper_uid", "TEXT NOT NULL DEFAULT ''"),
        ("paper_metadata", "paper_uid", "TEXT NOT NULL DEFAULT ''"),
        ("papers", "paper_uid", "TEXT NOT NULL DEFAULT ''"),
    ]
    _create_index_if_missing = [
        "CREATE INDEX IF NOT EXISTS idx_pm_paper_uid ON paper_metadata (paper_uid)",
        "CREATE INDEX IF NOT EXISTS idx_scholar_lib_papers_paper_uid ON scholar_library_papers (paper_uid)",
        "CREATE INDEX IF NOT EXISTS idx_papers_paper_uid ON papers (paper_uid)",
    ]
    with engine.connect() as conn:
        for table, col, col_type in _add_column_if_missing:
            try:
                conn.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()
        for stmt in _create_index_if_missing:
            try:
                conn.execute(sa.text(stmt))
                conn.commit()
            except Exception:
                conn.rollback()
