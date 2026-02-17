"""Password hashing and verification using bcrypt."""

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plain password. Returns bcrypt hash string."""
    if not plain:
        raise ValueError("password cannot be empty")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify plain password against stored hash."""
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
