"""Simple token generation and verification. Tokens stored in memory (or SQLite via persistent_store)."""

import secrets
import time
from typing import Optional

# In-memory token store: token -> (user_id, expiry_timestamp)
_tokens: dict[str, tuple[str, float]] = {}


def create_token(user_id: str, expire_hours: float = 24.0) -> str:
    """Generate a new token for user_id. Returns token string."""
    token = secrets.token_urlsafe(32)
    expiry = time.time() + expire_hours * 3600
    _tokens[token] = (user_id, expiry)
    return token


def verify_token(token: str) -> Optional[str]:
    """Verify token and return user_id if valid, else None."""
    if not token:
        return None
    entry = _tokens.get(token)
    if not entry:
        return None
    user_id, expiry = entry
    if time.time() > expiry:
        del _tokens[token]
        return None
    return user_id


def revoke_token(token: str) -> bool:
    """Remove token so it can no longer be used."""
    if token in _tokens:
        del _tokens[token]
        return True
    return False


def get_current_user_id(token: str) -> Optional[str]:
    """Alias for verify_token for use in dependencies."""
    return verify_token(token)
