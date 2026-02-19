from __future__ import annotations

import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import pool
from sqlmodel import SQLModel

from alembic import context

# Ensure project root is on sys.path so src.db imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Alembic Config object — provides access to values in alembic.ini.
config = context.config

# Set up loggers from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all SQLModel models so their metadata is registered.
import src.db.models as _models  # noqa: F401, E402

# Import the project engine so we get consistent DB-URL resolution.
from src.db.engine import get_engine, _make_absolute_sqlite_url, _resolve_db_url  # noqa: E402

target_metadata = SQLModel.metadata


def _get_url() -> str:
    """Resolve URL: honour alembic.ini override, otherwise use project logic."""
    ini_url = config.get_main_option("sqlalchemy.url", default="")
    # If ini still has the default placeholder, derive from project config.
    if not ini_url or ini_url.startswith("driver://"):
        ini_url = _resolve_db_url()
    return _make_absolute_sqlite_url(ini_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL scripts)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (applies directly to the DB)."""
    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # required for SQLite ALTER TABLE support
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
