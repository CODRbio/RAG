"""
认证 API：登录、管理员创建/列出用户。
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Header, Query

from config.settings import settings
from src.api.schemas import LoginRequest, LoginResponse, CreateUserRequest, UserItem
from src.auth.session import create_token, verify_token
from src.collaboration.memory.persistent_store import (
    create_user as store_create_user,
    verify_password as store_verify_password,
    list_users as store_list_users,
    get_user_profile,
)

router = APIRouter(prefix="/auth", tags=["auth"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def _get_token_from_header(authorization: str | None = Header(None)) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip() or None


def get_current_user_id(authorization: str | None = Header(None)) -> str:
    """Dependency: require valid token, return user_id."""
    token = _get_token_from_header(authorization)
    user_id = verify_token(token) if token else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return user_id


def get_current_admin_id(authorization: str | None = Header(None)) -> str:
    """Dependency: require valid token and role=admin."""
    user_id = get_current_user_id(authorization)
    profile = get_user_profile(user_id)
    if not profile or profile.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required")
    return user_id


def get_optional_user_id(authorization: str | None = Header(None)) -> str | None:
    """Dependency: return user_id if valid token present, else None (no 401)."""
    token = _get_token_from_header(authorization)
    return verify_token(token) if token else None


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """用户名+密码登录，返回 token。"""
    if not store_verify_password(body.user_id, body.password):
        raise HTTPException(status_code=401, detail="Invalid user_id or password")
    profile = get_user_profile(body.user_id)
    if not profile or not profile.get("is_active", True):
        raise HTTPException(status_code=401, detail="Account disabled")
    token = create_token(body.user_id, expire_hours=settings.auth.token_expire_hours)
    return LoginResponse(
        token=token,
        user_id=body.user_id,
        role=profile.get("role") or "user",
    )


@admin_router.post("/users")
def admin_create_user(
    body: CreateUserRequest,
    _admin_id: str = Depends(get_current_admin_id),
) -> dict:
    """管理员创建用户。"""
    try:
        store_create_user(user_id=body.user_id, password=body.password, role=body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user_id": body.user_id, "role": body.role}


@admin_router.get("/users", response_model=list[UserItem])
def admin_list_users(
    _admin_id: str = Depends(get_current_admin_id),
) -> list[UserItem]:
    """管理员列出所有用户。"""
    users = store_list_users()
    return [
        UserItem(
            user_id=u["user_id"],
            role=u["role"],
            is_active=u["is_active"],
            created_at=u["created_at"],
            updated_at=u["updated_at"],
        )
        for u in users
    ]


@admin_router.delete("/cache/crossref")
def admin_clear_crossref_cache(
    older_than_days: int = Query(default=0, ge=0, description="仅删除超过指定天数的条目；0 表示清空全部"),
    _admin_id: str = Depends(get_current_admin_id),
) -> dict:
    """清理 Crossref 元数据缓存（crossref_cache 和 crossref_cache_by_doi）。
    这些是从 Crossref API 拉取的文献元数据缓存，删除后会在下次需要时重新拉取。"""
    from sqlmodel import Session, select, delete as sql_delete
    from src.db.engine import get_engine
    from src.db.models import CrossrefCache, CrossrefCacheByDoi

    if older_than_days > 0:
        cutoff_ts = time.time() - older_than_days * 86400
        with Session(get_engine()) as session:
            cc = session.exec(select(CrossrefCache).where(CrossrefCache.created_at < cutoff_ts)).all()
            cc_doi = session.exec(select(CrossrefCacheByDoi).where(CrossrefCacheByDoi.created_at < cutoff_ts)).all()
            n_cc, n_doi = len(cc), len(cc_doi)
            for r in cc:
                session.delete(r)
            for r in cc_doi:
                session.delete(r)
            session.commit()
    else:
        with Session(get_engine()) as session:
            n_cc = len(session.exec(select(CrossrefCache)).all())
            n_doi = len(session.exec(select(CrossrefCacheByDoi)).all())
            session.exec(sql_delete(CrossrefCache))
            session.exec(sql_delete(CrossrefCacheByDoi))
            session.commit()

    return {
        "ok": True,
        "crossref_cache_deleted": n_cc,
        "crossref_cache_by_doi_deleted": n_doi,
        "older_than_days": older_than_days or "all",
    }


@admin_router.delete("/cache/paper-metadata")
def admin_clear_paper_metadata_cache(
    _admin_id: str = Depends(get_current_admin_id),
) -> dict:
    """清空 paper_metadata 表（论文 DOI/标题/作者等富化缓存）。
    删除后不影响已入库的向量数据，但文献详情页将无法显示富化元数据，直到重新入库或手动补全。"""
    from sqlmodel import Session, delete as sql_delete
    from src.db.engine import get_engine
    from src.db.models import PaperMetadata
    import sqlalchemy as sa

    with Session(get_engine()) as session:
        result = session.exec(sa.select(sa.func.count()).select_from(PaperMetadata))
        n = result.one()
        session.exec(sql_delete(PaperMetadata))
        session.commit()

    return {"ok": True, "paper_metadata_deleted": n}
