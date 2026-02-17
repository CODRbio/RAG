"""
认证 API：登录、管理员创建/列出用户。
"""

from fastapi import APIRouter, Depends, HTTPException, Header

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
