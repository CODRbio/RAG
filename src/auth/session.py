"""JWT-based token generation and verification.

Tokens are stateless HS256-signed JWTs — no server-side storage is required
for normal validation, making the implementation safe for multi-worker and
multi-replica deployments.

Token revocation (e.g. logout) is handled via a lightweight DB table
(`revoked_tokens`) that stores SHA-256 hashes of invalidated tokens.  The
hot path (verify_token) only queries the DB when the JWT signature is valid,
keeping unauthenticated or tampered requests pure CPU.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _secret() -> str:
    """Return the signing secret from application settings."""
    from config.settings import settings
    return settings.auth.secret_key


def _token_hash(token: str) -> str:
    """Return the SHA-256 hex digest of a raw JWT string."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_revoked(token: str) -> bool:
    """Return True if the token's hash is in the revocation table."""
    try:
        from src.db.engine import get_engine
        from src.db.models import RevokedToken
        from sqlmodel import Session

        h = _token_hash(token)
        with Session(get_engine()) as session:
            return session.get(RevokedToken, h) is not None
    except Exception:
        # If the DB is unavailable we fail open (treat as not revoked) rather
        # than locking out all users.  A warning is logged so ops can act.
        logger.warning("revocation DB unavailable during verify_token; treating token as not revoked")
        return False


# ---------------------------------------------------------------------------
# Public API  (same surface as the old in-memory implementation)
# ---------------------------------------------------------------------------

def create_token(user_id: str, expire_hours: Optional[float] = None) -> str:
    """Sign and return a new JWT for *user_id*.

    Args:
        user_id: The authenticated user's identifier stored in the ``sub`` claim.
        expire_hours: Lifetime in hours; defaults to ``settings.auth.token_expire_hours``.

    Returns:
        A compact HS256 JWT string.
    """
    from config.settings import settings

    hours = expire_hours if expire_hours is not None else settings.auth.token_expire_hours
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(hours=hours),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def verify_token(token: str) -> Optional[str]:
    """Validate *token* and return the user_id (``sub`` claim) if valid.

    Validation steps:
    1. JWT decode — verifies signature and ``exp`` claim.
    2. Revocation check — queries the DB revocation table.

    Returns ``None`` on any failure (expired, tampered, revoked, missing).
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        return None

    if _is_revoked(token):
        return None

    return user_id


def revoke_token(token: str) -> bool:
    """Add *token* to the revocation list so it can no longer be used.

    The raw token is never stored; only its SHA-256 hash and the original
    expiry timestamp (for later cleanup) are persisted.

    Returns ``True`` if the token was successfully revoked (or was already
    revoked), ``False`` on error.
    """
    if not token:
        return False
    try:
        # Decode without verifying expiry so already-expired tokens can still
        # be explicitly revoked (prevents replay attacks near expiry boundary).
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=["HS256"],
            options={"verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return False

    exp = payload.get("exp")
    if exp is not None:
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    else:
        expires_at = datetime.now(tz=timezone.utc).isoformat()

    try:
        from src.db.engine import get_engine
        from src.db.models import RevokedToken
        from sqlmodel import Session

        h = _token_hash(token)
        with Session(get_engine()) as session:
            if session.get(RevokedToken, h) is None:
                session.add(RevokedToken(
                    token_hash=h,
                    expires_at=expires_at,
                    revoked_at=datetime.now(tz=timezone.utc).isoformat(),
                ))
                session.commit()
        return True
    except Exception:
        logger.exception("Failed to persist token revocation")
        return False


def purge_expired_revocations() -> int:
    """Delete revocation records whose JWT has already expired.

    Call periodically (e.g. from a scheduled task or at app startup) to keep
    the `revoked_tokens` table small.  Returns the number of rows deleted.
    """
    try:
        from src.db.engine import get_engine
        from src.db.models import RevokedToken
        from sqlmodel import Session, select

        cutoff = datetime.now(tz=timezone.utc).isoformat()
        with Session(get_engine()) as session:
            rows = session.exec(
                select(RevokedToken).where(RevokedToken.expires_at < cutoff)
            ).all()
            count = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()
        return count
    except Exception:
        logger.exception("Failed to purge expired token revocations")
        return 0


def get_current_user_id(token: str) -> Optional[str]:
    """Alias for :func:`verify_token` — used as a FastAPI dependency."""
    return verify_token(token)
