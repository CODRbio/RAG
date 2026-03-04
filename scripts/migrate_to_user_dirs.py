#!/usr/bin/env python
"""
Migrate existing data/raw_papers and data/parsed into data/users/default/
and update papers.file_path in rag.db to point to the new paths.

Run from project root: python scripts/migrate_to_user_dirs.py [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.path_manager import PathManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate data to user-scoped dirs (data/users/default/...)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()
    dry_run = args.dry_run

    data_root = PathManager.get_data_root()
    old_raw = data_root / "raw_papers"
    old_parsed = data_root / "parsed"
    new_raw = PathManager.get_user_raw_papers_path("default")
    new_parsed = PathManager.get_user_parsed_path("default")

    if dry_run:
        print("[dry-run] Would perform:")
    else:
        print("Migrating to user dirs...")

    # 1. Move raw_papers
    if old_raw.exists() and old_raw.is_dir():
        if new_raw.exists() and list(new_raw.iterdir()):
            print(f"  Target {new_raw} already has content; skipping move of raw_papers (merge manually if needed).")
        else:
            if dry_run:
                print(f"  Move {old_raw} -> {new_raw}")
            else:
                new_raw.parent.mkdir(parents=True, exist_ok=True)
                for item in old_raw.iterdir():
                    dest = new_raw / item.name
                    if dest.exists():
                        continue
                    shutil.move(str(item), str(dest))
                try:
                    old_raw.rmdir()
                except OSError:
                    pass
    elif not old_raw.exists():
        print("  No existing data/raw_papers/ to migrate.")
    else:
        print(f"  Skipping {old_raw} (not a directory).")

    # 2. Move parsed
    if old_parsed.exists() and old_parsed.is_dir():
        if new_parsed.exists() and list(new_parsed.iterdir()):
            print(f"  Target {new_parsed} already has content; skipping move of parsed (merge manually if needed).")
        else:
            if dry_run:
                print(f"  Move {old_parsed} -> {new_parsed}")
            else:
                new_parsed.parent.mkdir(parents=True, exist_ok=True)
                for item in old_parsed.iterdir():
                    dest = new_parsed / item.name
                    if dest.exists():
                        continue
                    shutil.move(str(item), str(dest))
                try:
                    old_parsed.rmdir()
                except OSError:
                    pass
    elif not old_parsed.exists():
        print("  No existing data/parsed/ to migrate.")
    else:
        print(f"  Skipping {old_parsed} (not a directory).")

    # 3. Update papers.file_path in rag.db
    old_raw_str = str(old_raw.resolve())
    old_raw_str_trailing = old_raw_str.rstrip("/") + "/"
    new_raw_str = str(new_raw.resolve())
    new_raw_str_trailing = new_raw_str.rstrip("/") + "/"

    from src.db.engine import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, file_path FROM papers WHERE file_path != ''"))
        rows = result.fetchall()
    updated = 0
    for row in rows:
        pid, fpath = row[0], (row[1] or "").strip()
        if not fpath:
            continue
        new_path = None
        if fpath.startswith(old_raw_str_trailing) or fpath.startswith(old_raw_str):
            new_path = fpath.replace(old_raw_str_trailing, new_raw_str_trailing).replace(old_raw_str, new_raw_str)
        elif fpath.startswith(str(old_raw) + "/") or fpath == str(old_raw):
            new_path = fpath.replace(str(old_raw), str(new_raw), 1)
        if new_path and new_path != fpath:
            if dry_run:
                print(f"  Would update paper id={pid}: file_path -> {new_path[:80]}...")
            else:
                with engine.begin() as c:
                    c.execute(text("UPDATE papers SET file_path = :fp WHERE id = :id"), {"fp": new_path, "id": pid})
            updated += 1
    if updated:
        print(f"  Updated {updated} papers.file_path to new prefix.")
    else:
        print("  No papers.file_path rows needed updating.")

    print("Done.")


if __name__ == "__main__":
    main()
