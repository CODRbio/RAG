from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import settings


_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_store: Optional["BaseMediaStore"] = None


@dataclass(frozen=True)
class MediaAsset:
    url: str
    key: str
    backend: str
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class ImageMediaDetails:
    content_type: str
    file_ext: str


def detect_image_media_details(content: bytes) -> ImageMediaDetails:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ImageMediaDetails(content_type="image/png", file_ext=".png")
    if content.startswith(b"\xff\xd8\xff"):
        return ImageMediaDetails(content_type="image/jpeg", file_ext=".jpg")
    if content.startswith((b"GIF87a", b"GIF89a")):
        return ImageMediaDetails(content_type="image/gif", file_ext=".gif")
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ImageMediaDetails(content_type="image/webp", file_ext=".webp")
    return ImageMediaDetails(content_type="image/png", file_ext=".png")


def get_media_local_root() -> Path:
    configured = Path(settings.storage.media.local.root)
    if configured.is_absolute():
        root = configured
    else:
        root = settings.path.base / configured
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_ext(file_ext: str) -> str:
    ext = (file_ext or "").strip().lower()
    if not ext:
        return ".bin"
    return ext if ext.startswith(".") else f".{ext}"


def _sanitize_segment(segment: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("-", (segment or "").strip()).strip(".-")
    return cleaned or "asset"


def _normalize_public_base_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _join_url(base: str, suffix: str) -> str:
    base_clean = _normalize_public_base_url(base)
    suffix_clean = suffix.lstrip("/")
    if not base_clean:
        return f"/{suffix_clean}"
    return f"{base_clean}/{suffix_clean}"


def _build_asset_key(*, category: str, session_id: Optional[str], logical_name: Optional[str], file_ext: str) -> str:
    filename_base = _sanitize_segment(logical_name) if logical_name else uuid.uuid4().hex
    session_segment = _sanitize_segment(session_id) if session_id else "anon"
    return "/".join(
        [
            _sanitize_segment(category),
            session_segment,
            f"{filename_base}{_normalize_ext(file_ext)}",
        ]
    )


class BaseMediaStore:
    backend_name = "base"

    def store_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
        file_ext: str,
        session_id: str | None = None,
        logical_name: str | None = None,
    ) -> MediaAsset:
        raise NotImplementedError


class LocalMediaStore(BaseMediaStore):
    backend_name = "local"

    def __init__(self) -> None:
        self.root = get_media_local_root()
        self.public_base_url = _normalize_public_base_url(settings.storage.media.public_base_url)

    def store_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
        file_ext: str,
        session_id: str | None = None,
        logical_name: str | None = None,
    ) -> MediaAsset:
        key = _build_asset_key(
            category=category,
            session_id=session_id,
            logical_name=logical_name,
            file_ext=file_ext,
        )
        target = self.root / Path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

        url = _join_url(self.public_base_url or "/media", key)
        return MediaAsset(
            url=url,
            key=key,
            backend=self.backend_name,
            content_type=content_type,
            size_bytes=len(content),
        )


def _load_boto3_session_cls():
    try:
        from boto3.session import Session
    except ImportError as exc:
        raise RuntimeError("boto3 is required for storage.media.backend = 's3'") from exc
    return Session


class S3MediaStore(BaseMediaStore):
    backend_name = "s3"

    def __init__(self) -> None:
        cfg = settings.storage.media.s3
        self.bucket = cfg.bucket.strip()
        self.endpoint = cfg.endpoint.strip() or None
        self.region = cfg.region.strip() or os.getenv("AWS_DEFAULT_REGION") or None
        self.key_prefix = cfg.key_prefix.strip().strip("/")
        self.public_base_url = _normalize_public_base_url(cfg.public_base_url or settings.storage.media.public_base_url)

        if not self.bucket:
            raise ValueError("storage.media.s3.bucket is required when backend='s3'")
        if not self.public_base_url:
            raise ValueError("storage.media.s3.public_base_url or storage.media.public_base_url is required")

        session_cls = _load_boto3_session_cls()
        session = session_cls(
            aws_access_key_id=os.getenv("MEDIA_S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("MEDIA_S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("MEDIA_S3_SESSION_TOKEN") or os.getenv("AWS_SESSION_TOKEN"),
            region_name=self.region,
        )
        self.client = session.client("s3", endpoint_url=self.endpoint, region_name=self.region)

    def store_bytes(
        self,
        *,
        category: str,
        content: bytes,
        content_type: str,
        file_ext: str,
        session_id: str | None = None,
        logical_name: str | None = None,
    ) -> MediaAsset:
        key = _build_asset_key(
            category=category,
            session_id=session_id,
            logical_name=logical_name,
            file_ext=file_ext,
        )
        if self.key_prefix:
            key = f"{self.key_prefix}/{key}"

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )
        url = _join_url(self.public_base_url, key)
        return MediaAsset(
            url=url,
            key=key,
            backend=self.backend_name,
            content_type=content_type,
            size_bytes=len(content),
        )


def get_media_store() -> BaseMediaStore:
    global _store
    if _store is not None:
        return _store

    backend = (settings.storage.media.backend or "local").strip().lower()
    if backend == "s3":
        _store = S3MediaStore()
    else:
        _store = LocalMediaStore()
    return _store
