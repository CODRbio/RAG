"""
Persistent Store：用户级持久化（偏好、历史项目、认证）。
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "persistent.db"


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            preferences_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _ensure_user_profile_auth_columns(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_projects (
            user_id TEXT NOT NULL,
            canvas_id TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, canvas_id)
        )
        """
    )
    conn.commit()


def _ensure_user_profile_auth_columns(conn: sqlite3.Connection) -> None:
    """Add auth columns to user_profiles if missing (migration)."""
    cur = conn.execute("PRAGMA table_info(user_profiles)")
    columns = {row[1] for row in cur.fetchall()}
    if "password_hash" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''")
    if "role" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    if "is_active" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """返回用户偏好（含 role/is_active），无则返回 None。"""
    if not user_id:
        return None
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT user_id, preferences_json, created_at, updated_at, password_hash, role, is_active FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    prefs = {}
    if row["preferences_json"]:
        try:
            prefs = json.loads(row["preferences_json"])
        except Exception:
            pass
    return {
        "user_id": row["user_id"],
        "preferences": prefs,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "password_hash": row["password_hash"] or "",
        "role": row["role"] or "user",
        "is_active": bool(row["is_active"]) if row["is_active"] is not None else True,
    }


def upsert_user_profile(user_id: str, preferences: Dict[str, Any]) -> None:
    """写入或更新用户偏好。"""
    if not user_id:
        return
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    prefs_json = json.dumps(preferences, ensure_ascii=False)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.execute(
            """INSERT INTO user_profiles (user_id, preferences_json, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET preferences_json = ?, updated_at = ?""",
            (user_id, prefs_json, now, now, prefs_json, now),
        )
        conn.commit()


def add_user_project(user_id: str, canvas_id: str, title: str = "") -> None:
    """记录用户关联的画布项目。"""
    if not user_id or not canvas_id:
        return
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.execute(
            """INSERT INTO user_projects (user_id, canvas_id, title, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, canvas_id) DO UPDATE SET title = ?, updated_at = ?""",
            (user_id, canvas_id, title or canvas_id, now, title or canvas_id, now),
        )
        conn.commit()


def get_user_projects(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """返回用户最近项目列表。"""
    if not user_id:
        return []
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT user_id, canvas_id, title, updated_at FROM user_projects WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [
        {"user_id": r["user_id"], "canvas_id": r["canvas_id"], "title": r["title"], "updated_at": r["updated_at"]}
        for r in rows
    ]


def create_user(
    user_id: str,
    password: str,
    role: str = "user",
) -> None:
    """创建用户（管理员调用）。password 将被哈希存储。"""
    from src.auth.password import hash_password

    if not user_id or not password:
        raise ValueError("user_id and password are required")
    if role not in ("admin", "user"):
        raise ValueError("role must be admin or user")
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    password_hash = hash_password(password)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.execute(
            """INSERT INTO user_profiles (user_id, preferences_json, created_at, updated_at, password_hash, role, is_active)
               VALUES (?, '{}', ?, ?, ?, ?, 1)""",
            (user_id, now, now, password_hash, role),
        )
        conn.commit()


def verify_password(user_id: str, plain_password: str) -> bool:
    """校验用户密码。"""
    from src.auth.password import verify_password as _verify

    profile = get_user_profile(user_id)
    if not profile or not profile.get("is_active", True):
        return False
    return _verify(plain_password, profile.get("password_hash") or "")


def list_users() -> List[Dict[str, Any]]:
    """列出所有用户（管理员用）。不含 password_hash。"""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _init_schema(conn)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT user_id, role, is_active, created_at, updated_at FROM user_profiles ORDER BY created_at"
        ).fetchall()
    return [
        {
            "user_id": r["user_id"],
            "role": r["role"] or "user",
            "is_active": bool(r["is_active"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
