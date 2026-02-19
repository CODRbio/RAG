"""
Centralized SQLAlchemy/SQLModel engine and session factory.

All store modules import `engine` and `get_session` from here.
The database URL is resolved from config/rag_config.json or the
RAG_DATABASE_URL environment variable, making the switch to
PostgreSQL a single configuration change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

_engine: Engine | None = None


def _resolve_db_url() -> str:
    """
    Resolve database URL with precedence:
    1. RAG_DATABASE_URL environment variable
    2. config/rag_config.local.json  database.url
    3. config/rag_config.json        database.url
    4. Fallback: sqlite:///data/rag.db
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

    return "sqlite:///data/rag.db"


def _make_absolute_sqlite_url(url: str) -> str:
    """
    Resolve relative sqlite:/// paths to absolute so the DB is always
    created in <project_root>/data/rag.db regardless of cwd.
    """
    if not url.startswith("sqlite:///"):
        return url
    rel_path = url[len("sqlite:///"):]
    if os.path.isabs(rel_path):
        return url
    root = Path(__file__).resolve().parents[2]
    abs_path = (root / rel_path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{abs_path}"


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    raw_url = _resolve_db_url()
    db_url = _make_absolute_sqlite_url(raw_url)

    is_sqlite = db_url.startswith("sqlite")
    connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}

    _engine = create_engine(
        db_url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    if is_sqlite:
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

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
