#!/usr/bin/env python
"""
Compare row counts between the local SQLite DB and a PostgreSQL target.

Usage:
  conda run -n deepsea-rag python scripts/verify_postgres_migration.py \
    --target-url "postgresql+psycopg://user:pass@host:5432/dbname"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from sqlalchemy import create_engine, func, select
from sqlmodel import SQLModel


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.engine import _make_absolute_sqlite_url, _normalize_db_url, _resolve_db_url  # noqa: E402
import src.db.models as _models  # noqa: F401,E402


def _count(engine, table) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SQLite to PostgreSQL migration by row counts.")
    parser.add_argument(
        "--source-url",
        default=_make_absolute_sqlite_url(_normalize_db_url(_resolve_db_url())),
        help="Source SQLite URL. Defaults to current configured DB URL.",
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="Target PostgreSQL URL. Use postgresql+psycopg://...",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional subset of table names to verify.",
    )
    args = parser.parse_args()

    source_url = _make_absolute_sqlite_url(_normalize_db_url(args.source_url))
    target_url = _make_absolute_sqlite_url(_normalize_db_url(args.target_url))

    src_engine = create_engine(source_url, pool_pre_ping=True)
    dst_engine = create_engine(target_url, pool_pre_ping=True)

    wanted = set(args.tables or [])
    mismatches: List[str] = []
    try:
        for table in SQLModel.metadata.sorted_tables:
            if wanted and table.name not in wanted:
                continue
            src_count = _count(src_engine, table)
            dst_count = _count(dst_engine, table)
            status = "OK" if src_count == dst_count else "MISMATCH"
            print(f"{status:8s} {table.name:35s} sqlite={src_count:8d} postgres={dst_count:8d}")
            if src_count != dst_count:
                mismatches.append(table.name)
    finally:
        src_engine.dispose()
        dst_engine.dispose()

    if mismatches:
        raise SystemExit("Row count mismatch detected: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
