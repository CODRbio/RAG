"""
Config API: read/update database URL for advanced settings UI.
Prefix: /config
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.db.engine import get_resolved_db_url, get_local_config_path
from src.api.routes_auth import get_current_user_id

router = APIRouter(prefix="/config", tags=["config"])

DEFAULT_DATABASE_URL = "sqlite:///data/rag.db"


@router.get("/database")
def get_config_database() -> dict:
    """Return the current effective database URL (for display in advanced settings)."""
    return {"url": get_resolved_db_url()}


@router.patch("/database")
def patch_config_database(
    body: dict,
    _user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    Update database.url in config/rag_config.local.json.
    Takes effect after server restart.
    """
    url = body.get("url")
    if url is None or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="url is required and must be a string")
    url = url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url cannot be empty")

    local_path = get_local_config_path()
    try:
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data["database"] = {**(data.get("database") or {}), "url": url}
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"url": url, "message": "Saved. Restart the server for the change to take effect."}
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")


@router.get("/list-dir")
def list_dir(
    path: Optional[str] = Query(default=None, description="Directory path to list; defaults to home directory"),
    _user_id: str = Depends(get_current_user_id),
) -> dict:
    """
    List subdirectories of a server-side path.
    Returns entries (name, path, is_dir) plus the resolved current path and parent path.
    Only directories are returned so the UI can navigate to select a folder.
    """
    if path:
        target = Path(path).expanduser().resolve()
    else:
        target = Path.home()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    try:
        entries: List[dict] = []
        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
                entries.append({
                    "name": entry.name,
                    "path": str(entry),
                    "is_dir": is_dir,
                })
            except PermissionError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    parent = str(target.parent) if target != target.parent else None

    return {
        "current": str(target),
        "parent": parent,
        "home": str(Path.home()),
        "entries": entries,
    }
