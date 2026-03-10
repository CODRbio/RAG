#!/usr/bin/env python
"""
Create a cold backup bundle for the local runtime state before DB migration.

Default outputs:
  backups/pre_pg_migration_<timestamp>/
    manifest.json
    rag.db
    rag.db-wal           (if present)
    rag.db-shm           (if present)
    raw_papers.tar.gz    (if source dir exists)
    parsed.tar.gz        (if source dir exists)
    artifacts.tar.gz     (if source dir exists)

Run from project root:
  conda run -n deepsea-rag python scripts/backup_runtime_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = PROJECT_ROOT / "backups"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(source)) as src, sqlite3.connect(str(destination)) as dst:
        src.backup(dst)


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def _archive_dir(source_dir: Path, archive_path: Path) -> bool:
    if not source_dir.exists() or not source_dir.is_dir():
        return False
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    return True


def _record_file(manifest_files: List[Dict[str, object]], path: Path) -> None:
    manifest_files.append(
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a cold backup for rag.db and runtime data.")
    parser.add_argument(
        "--label",
        default="pre_pg_migration",
        help="Backup directory label prefix (default: pre_pg_migration)",
    )
    parser.add_argument(
        "--backup-root",
        default=str(DEFAULT_BACKUP_ROOT),
        help="Directory that stores generated backups",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = Path(args.backup_root).expanduser().resolve()
    backup_dir = backup_root / f"{args.label}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    db_path = PROJECT_ROOT / "data" / "rag.db"
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    manifest: Dict[str, object] = {
        "created_at_utc": timestamp,
        "hostname": socket.gethostname(),
        "project_root": str(PROJECT_ROOT),
        "backup_dir": str(backup_dir),
        "files": [],
        "notes": [
            "Create this backup only after application processes are stopped.",
            "Restore rag.db together with rag.db-wal/rag.db-shm if those files were captured.",
        ],
    }
    manifest_files = manifest["files"]
    assert isinstance(manifest_files, list)

    db_backup = backup_dir / "rag.db"
    _backup_sqlite(db_path, db_backup)
    _record_file(manifest_files, db_backup)

    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        sidecar_backup = backup_dir / sidecar.name
        if _copy_if_exists(sidecar, sidecar_backup):
            _record_file(manifest_files, sidecar_backup)

    archives = {
        "raw_papers": PROJECT_ROOT / "data" / "raw_papers",
        "parsed": PROJECT_ROOT / "data" / "parsed",
        "artifacts": PROJECT_ROOT / "artifacts",
    }
    for archive_name, source_dir in archives.items():
        archive_path = backup_dir / f"{archive_name}.tar.gz"
        if _archive_dir(source_dir, archive_path):
            _record_file(manifest_files, archive_path)

    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Backup created: {backup_dir}")
    print(f"Manifest: {manifest_path}")
    for item in manifest_files:
        print(f"- {item['path']} ({item['size_bytes']} bytes)")


if __name__ == "__main__":
    main()
