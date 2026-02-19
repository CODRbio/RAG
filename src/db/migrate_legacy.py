"""
Legacy SQLite → rag.db one-time data migration.

Reads rows from the 8 old .db files and bulk-inserts them into the
consolidated data/rag.db.  Safe to run multiple times (INSERT OR IGNORE).
After a fully successful migration the old files are renamed to *.db.bak.

Run manually:
    conda run -n deepsea-rag python -m src.db.migrate_legacy

Or call migrate_if_needed() from application startup to auto-migrate.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Path resolution ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NEW_DB = _PROJECT_ROOT / "data" / "rag.db"


def _find_old_db(name: str) -> Optional[Path]:
    """Search both data/ and src/data/ for an old .db file."""
    candidates = [
        _PROJECT_ROOT / "data" / name,
        _PROJECT_ROOT / "src" / "data" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-table migration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _copy_table(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    src_table: str,
    dst_table: str,
    column_map: Optional[Dict[str, str]] = None,
    extra_defaults: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Copy all rows from src_table → dst_table.

    column_map  : {src_col: dst_col} rename map (identity if absent)
    extra_defaults : {dst_col: value} columns not present in src that need a default
    Returns number of rows inserted.
    """
    src_conn.row_factory = sqlite3.Row
    rows = src_conn.execute(f"SELECT * FROM {src_table}").fetchall()
    if not rows:
        return 0

    # Determine destination columns from the first row + extras
    sample = rows[0]
    src_cols = list(sample.keys())
    col_map = column_map or {}

    dst_cols = [col_map.get(c, c) for c in src_cols]
    if extra_defaults:
        for col in extra_defaults:
            if col not in dst_cols:
                dst_cols.append(col)

    placeholders = ", ".join(["?" for _ in dst_cols])
    cols_sql = ", ".join([f'"{c}"' for c in dst_cols])
    insert_sql = f"INSERT OR IGNORE INTO {dst_table} ({cols_sql}) VALUES ({placeholders})"

    inserted = 0
    for row in rows:
        vals = [row[c] for c in src_cols]
        if extra_defaults:
            for col, default_val in extra_defaults.items():
                if col not in dst_cols[:len(src_cols)]:
                    vals.append(default_val)
        try:
            dst_conn.execute(insert_sql, vals)
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate — already present
    dst_conn.commit()
    return inserted


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ──────────────────────────────────────────────────────────────────────────────
# Per-database migration functions
# ──────────────────────────────────────────────────────────────────────────────

def _migrate_canvas(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("canvas.db")
    if not src_path:
        return "canvas.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        for tbl in ("canvases", "outline_sections", "draft_blocks", "canvas_versions", "canvas_citations"):
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  canvas.db → %s: %d rows", tbl, n)
    return "canvas.db", total


def _migrate_sessions(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("sessions.db")
    if not src_path:
        return "sessions.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        for tbl in ("sessions", "turns"):
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  sessions.db → %s: %d rows", tbl, n)
    return "sessions.db", total


def _migrate_persistent(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("persistent.db")
    if not src_path:
        return "persistent.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        for tbl in ("user_profiles", "user_projects"):
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  persistent.db → %s: %d rows", tbl, n)
    return "persistent.db", total


def _migrate_working_memory(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("working_memory.db")
    if not src_path:
        return "working_memory.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        if _table_exists(src, "working_memory"):
            n = _copy_table(src, dst, "working_memory", "working_memory")
            total += n
            logger.info("  working_memory.db → working_memory: %d rows", n)
    return "working_memory.db", total


def _migrate_papers(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("papers.db")
    if not src_path:
        return "papers.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        if _table_exists(src, "papers"):
            n = _copy_table(src, dst, "papers", "papers")
            total += n
            logger.info("  papers.db → papers: %d rows", n)
    return "papers.db", total


def _migrate_ingest_jobs(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("ingest_jobs.db")
    if not src_path:
        return "ingest_jobs.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        for tbl in ("ingest_jobs", "ingest_job_events"):
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  ingest_jobs.db → %s: %d rows", tbl, n)
    return "ingest_jobs.db", total


def _migrate_deep_research(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("deep_research_jobs.db")
    if not src_path:
        return "deep_research_jobs.db", 0
    total = 0
    tables = [
        "deep_research_jobs",
        "deep_research_job_events",
        "deep_research_section_reviews",
        "deep_research_resume_queue",
        "deep_research_gap_supplements",
        "deep_research_insights",
    ]
    with sqlite3.connect(src_path) as src:
        for tbl in tables:
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  deep_research_jobs.db → %s: %d rows", tbl, n)
    return "deep_research_jobs.db", total


def _migrate_paper_metadata(dst: sqlite3.Connection) -> Tuple[str, int]:
    src_path = _find_old_db("paper_metadata.db")
    if not src_path:
        return "paper_metadata.db", 0
    total = 0
    with sqlite3.connect(src_path) as src:
        for tbl in ("paper_metadata", "crossref_cache"):
            if _table_exists(src, tbl):
                n = _copy_table(src, dst, tbl, tbl)
                total += n
                logger.info("  paper_metadata.db → %s: %d rows", tbl, n)
    return "paper_metadata.db", total


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_migration(rename_old: bool = True) -> Dict[str, int]:
    """
    Migrate all old .db files into data/rag.db.

    Parameters
    ----------
    rename_old : if True, rename successfully-migrated source files to *.db.bak
    
    Returns dict of {db_name: rows_migrated}.
    """
    if not _NEW_DB.exists():
        logger.error("Target %s does not exist — run 'alembic upgrade head' first.", _NEW_DB)
        raise FileNotFoundError(f"rag.db not found at {_NEW_DB}")

    results: Dict[str, int] = {}
    migrated_paths: List[Path] = []

    dst_conn = sqlite3.connect(str(_NEW_DB), timeout=30)
    dst_conn.execute("PRAGMA journal_mode=WAL")
    dst_conn.execute("PRAGMA foreign_keys=OFF")  # disable during bulk import

    try:
        for migrate_fn in [
            _migrate_canvas,
            _migrate_sessions,
            _migrate_persistent,
            _migrate_working_memory,
            _migrate_papers,
            _migrate_ingest_jobs,
            _migrate_deep_research,
            _migrate_paper_metadata,
        ]:
            db_name, count = migrate_fn(dst_conn)
            results[db_name] = count
            if count > 0:
                src = _find_old_db(db_name)
                if src:
                    migrated_paths.append(src)

        dst_conn.execute("PRAGMA foreign_keys=ON")
    finally:
        dst_conn.close()

    total = sum(results.values())
    logger.info("Migration complete: %d total rows migrated from %d source databases.", total, len(migrated_paths))

    if rename_old and migrated_paths:
        for src in migrated_paths:
            bak = src.with_suffix(".db.bak")
            src.rename(bak)
            logger.info("Renamed %s → %s", src.name, bak.name)

    return results


def migrate_if_needed() -> bool:
    """
    Auto-migration: runs only when old .db files exist and rag.db is present.
    Returns True if migration ran, False if skipped.
    Suitable to call at application startup.
    """
    old_dbs = [
        "canvas.db", "sessions.db", "persistent.db", "working_memory.db",
        "papers.db", "ingest_jobs.db", "deep_research_jobs.db", "paper_metadata.db",
    ]
    found_any = any(_find_old_db(n) for n in old_dbs)
    if not found_any:
        return False
    if not _NEW_DB.exists():
        return False

    logger.info("[migrate_legacy] Old .db files found. Running auto-migration into %s …", _NEW_DB)
    try:
        results = run_migration(rename_old=True)
        total = sum(results.values())
        logger.info("[migrate_legacy] Auto-migration done: %d rows total.", total)
        return True
    except Exception as e:
        logger.error("[migrate_legacy] Auto-migration failed: %s", e, exc_info=True)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dry = "--dry-run" in sys.argv
    keep = "--keep" in sys.argv  # don't rename old files

    if dry:
        print("DRY RUN — no files will be renamed, no data will be written.")
    else:
        results = run_migration(rename_old=not keep)
        print("\nMigration summary:")
        for db_name, count in sorted(results.items()):
            status = "skipped (not found)" if count == 0 else f"{count} rows"
            print(f"  {db_name:35s}: {status}")
        print(f"\nTotal rows migrated: {sum(results.values())}")
