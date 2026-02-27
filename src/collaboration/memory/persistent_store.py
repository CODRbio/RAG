"""
Persistent Store：用户级持久化（偏好、历史项目、认证）。
底层存储已迁移至 data/rag.db，通过 SQLModel 访问。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import UserProfile, UserProject


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """返回用户偏好（含 role/is_active），无则返回 None。"""
    if not user_id:
        return None
    with Session(get_engine()) as session:
        row = session.get(UserProfile, user_id)
    if row is None:
        return None
    return {
        "user_id": row.user_id,
        "preferences": row.get_preferences(),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "password_hash": row.password_hash or "",
        "role": row.role or "user",
        "is_active": bool(row.is_active) if row.is_active is not None else True,
    }


def upsert_user_profile(user_id: str, preferences: Dict[str, Any]) -> None:
    """写入或更新用户偏好。"""
    if not user_id:
        return
    now = datetime.now().isoformat()
    prefs_json = json.dumps(preferences, ensure_ascii=False)
    with Session(get_engine()) as session:
        row = session.get(UserProfile, user_id)
        if row is None:
            row = UserProfile(
                user_id=user_id,
                preferences_json=prefs_json,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
        else:
            row.preferences_json = prefs_json
            row.updated_at = now
            session.add(row)
        session.commit()


def add_user_project(user_id: str, canvas_id: str, title: str = "") -> None:
    """记录用户关联的画布项目。"""
    if not user_id or not canvas_id:
        return
    now = datetime.now().isoformat()
    with Session(get_engine()) as session:
        row = session.get(UserProject, (user_id, canvas_id))
        if row is None:
            row = UserProject(
                user_id=user_id,
                canvas_id=canvas_id,
                title=title or canvas_id,
                updated_at=now,
            )
            session.add(row)
        else:
            row.title = title or canvas_id
            row.updated_at = now
            session.add(row)
        session.commit()


def delete_user_project(user_id: str, canvas_id: str) -> bool:
    """删除用户下的某画布关联记录。"""
    if not user_id or not canvas_id:
        return False
    with Session(get_engine()) as session:
        row = session.get(UserProject, (user_id, canvas_id))
        if row is None:
            return False
        session.delete(row)
        session.commit()
    return True


def get_user_projects(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """返回用户最近项目列表。"""
    if not user_id:
        return []
    with Session(get_engine()) as session:
        stmt = (
            select(UserProject)
            .where(UserProject.user_id == user_id)
            .order_by(UserProject.updated_at.desc())
            .limit(limit)
        )
        rows = session.exec(stmt).all()
    return [
        {"user_id": r.user_id, "canvas_id": r.canvas_id, "title": r.title, "updated_at": r.updated_at}
        for r in rows
    ]


def create_user(user_id: str, password: str, role: str = "user") -> None:
    """创建用户（管理员调用）。password 将被哈希存储。"""
    from src.auth.password import hash_password

    if not user_id or not password:
        raise ValueError("user_id and password are required")
    if role not in ("admin", "user"):
        raise ValueError("role must be admin or user")
    now = datetime.now().isoformat()
    password_hash = hash_password(password)
    with Session(get_engine()) as session:
        row = UserProfile(
            user_id=user_id,
            preferences_json="{}",
            created_at=now,
            updated_at=now,
            password_hash=password_hash,
            role=role,
            is_active=1,
        )
        session.add(row)
        session.commit()


def verify_password(user_id: str, plain_password: str) -> bool:
    """校验用户密码。"""
    from src.auth.password import verify_password as _verify

    profile = get_user_profile(user_id)
    if not profile or not profile.get("is_active", True):
        return False
    return _verify(plain_password, profile.get("password_hash") or "")


def list_users() -> List[Dict[str, Any]]:
    """列出所有用户（管理员用）。不含 password_hash。"""
    with Session(get_engine()) as session:
        rows = session.exec(select(UserProfile).order_by(UserProfile.created_at)).all()
    return [
        {
            "user_id": r.user_id,
            "role": r.role or "user",
            "is_active": bool(r.is_active),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]
