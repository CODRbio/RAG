#!/usr/bin/env python
"""
Copy data from the local SQLite runtime DB into PostgreSQL.

Usage:
  conda run -n deepsea-rag python scripts/migrate_sqlite_to_postgres.py \
    --target-url "postgresql+psycopg://user:pass@host:5432/dbname"

Recommended flow:
  1. Stop the app and create a cold backup.
  2. Point Alembic at the target PostgreSQL URL and run `alembic upgrade head`.
  3. Run this script.
  4. Run `scripts/verify_postgres_migration.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from sqlalchemy import MetaData, create_engine, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import SQLModel


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.engine import _make_absolute_sqlite_url, _normalize_db_url, _resolve_db_url  # noqa: E402
import src.db.models as _models  # noqa: F401,E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate data from SQLite rag.db to PostgreSQL.")
    parser.add_argument(
        "--source-url",
        default=_make_absolute_sqlite_url(_normalize_db_url(_resolve_db_url())),
        help="Source SQLite URL. Defaults to the current configured DB URL.",
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="Target PostgreSQL URL. Use postgresql+psycopg://...",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per insert batch (default: 1000).",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional subset of table names to migrate.",
    )
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="Create tables from SQLModel metadata on the target before copying data.",
    )
    return parser.parse_args()


def _validate_urls(source_url: str, target_url: str) -> tuple[str, str]:
    source = _make_absolute_sqlite_url(_normalize_db_url(source_url))
    target = _make_absolute_sqlite_url(_normalize_db_url(target_url))
    if not source.startswith("sqlite"):
        raise ValueError(f"Source must be SQLite, got: {source}")
    if not target.startswith("postgresql+psycopg://"):
        raise ValueError(
            "Target must use psycopg. Example: postgresql+psycopg://user:pass@host:5432/dbname"
        )
    if source == target:
        raise ValueError("Source and target URLs are identical.")
    return source, target


def _connect(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def _iter_tables(selected: List[str] | None) -> Iterable:
    wanted = set(selected or [])
    for table in SQLModel.metadata.sorted_tables:
        if wanted and table.name not in wanted:
            continue
        yield table


def _count_rows(engine: Engine, table) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _ensure_target_schema(dst_engine: Engine, create_schema: bool) -> None:
    if create_schema:
        SQLModel.metadata.create_all(dst_engine)
        return
    inspector = inspect(dst_engine)
    missing = [t.name for t in SQLModel.metadata.sorted_tables if not inspector.has_table(t.name)]
    if missing:
        raise RuntimeError(
            "Target schema is incomplete. Run `alembic upgrade head` first "
            f"or rerun with --create-schema. Missing tables: {', '.join(missing)}"
        )


def _copy_table(src_engine: Engine, dst_engine: Engine, table, batch_size: int) -> Dict[str, int]:
    src_count = _count_rows(src_engine, table)
    dst_before = _count_rows(dst_engine, table)
    if src_count == 0:
        return {"source": 0, "before": dst_before, "after": dst_before, "inserted": 0}

    metadata = MetaData()
    metadata.reflect(bind=dst_engine, only=[table.name])
    target_table = metadata.tables[table.name]

    inserted = 0
    offset = 0
    while True:
        with src_engine.connect() as src_conn:
            rows = src_conn.execute(select(table).offset(offset).limit(batch_size)).mappings().all()
        if not rows:
            break
        payload = [dict(row) for row in rows]
        stmt = pg_insert(target_table).values(payload).on_conflict_do_nothing()
        with dst_engine.begin() as dst_conn:
            result = dst_conn.execute(stmt)
            inserted += max(int(result.rowcount or 0), 0)
        offset += len(payload)

    dst_after = _count_rows(dst_engine, table)
    return {"source": src_count, "before": dst_before, "after": dst_after, "inserted": inserted}


def _reset_sequences(dst_engine: Engine) -> None:
    with dst_engine.begin() as conn:
        table_names = [t.name for t in SQLModel.metadata.sorted_tables]
        for table_name in table_names:
            cols = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = :table_name
                      AND column_default LIKE 'nextval(%'
                    """
                ),
                {"table_name": table_name},
            ).scalars().all()
            for column_name in cols:
                seq_name = conn.execute(
                    text("SELECT pg_get_serial_sequence(:table_name, :column_name)"),
                    {"table_name": table_name, "column_name": column_name},
                ).scalar_one_or_none()
                if not seq_name:
                    continue
                max_value = conn.execute(
                    text(f'SELECT COALESCE(MAX("{column_name}"), 0) FROM "{table_name}"')
                ).scalar_one()
                conn.execute(
                    text("SELECT setval(:seq_name, :value, :is_called)"),
                    {
                        "seq_name": seq_name,
                        "value": int(max_value) if max_value else 1,
                        "is_called": bool(max_value),
                    },
                )


def main() -> None:
    args = _parse_args()
    source_url, target_url = _validate_urls(args.source_url, args.target_url)
    src_engine = _connect(source_url)
    dst_engine = _connect(target_url)
    _ensure_target_schema(dst_engine, args.create_schema)

    print(f"Source: {source_url}")
    print(f"Target: {target_url}")
    print(f"Batch size: {args.batch_size}")

    summary: Dict[str, Dict[str, int]] = {}
    try:
        for table in _iter_tables(args.tables):
            stats = _copy_table(src_engine, dst_engine, table, args.batch_size)
            summary[table.name] = stats
            print(
                f"- {table.name}: source={stats['source']} "
                f"before={stats['before']} inserted={stats['inserted']} after={stats['after']}"
            )
        _reset_sequences(dst_engine)
    except SQLAlchemyError as exc:
        raise SystemExit(f"Migration failed: {exc}") from exc
    finally:
        src_engine.dispose()
        dst_engine.dispose()

    print("\nMigration summary:")
    for table_name, stats in summary.items():
        print(
            f"  {table_name}: source={stats['source']} "
            f"before={stats['before']} inserted={stats['inserted']} after={stats['after']}"
        )


if __name__ == "__main__":
    main()
