"""
本地模型管理：检查是否存在 + 同步/升级模型缓存。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class ModelStatusItem:
    name: str
    model_id: str
    cache_dir: str
    exists: bool
    local_files_only: bool
    error: Optional[str] = None


@dataclass
class ModelSyncItem:
    name: str
    model_id: str
    cache_dir: str
    local_files_only: bool
    updated: bool
    status: str
    message: Optional[str] = None
    error: Optional[str] = None
    resolved_path: Optional[str] = None


def _get_model_targets() -> List[tuple[str, str, str]]:
    """返回需要管理的模型列表：(name, model_id, cache_dir)."""
    targets = [
        ("bge_m3", settings.model.embedding_model, settings.model.embedding_cache_dir or ""),
        ("bge_reranker", settings.model.reranker_model, settings.model.reranker_cache_dir or ""),
    ]
    if settings.search.use_colbert_reranker:
        targets.append(
            ("colbert", settings.search.colbert_model, settings.model.colbert_cache_dir or "")
        )
    return targets


def check_models(local_files_only: Optional[bool] = None) -> List[ModelStatusItem]:
    from huggingface_hub import snapshot_download

    local_only = (
        settings.model.local_files_only
        if local_files_only is None
        else bool(local_files_only)
    )
    results: List[ModelStatusItem] = []
    for name, model_id, cache_dir in _get_model_targets():
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=True,
                local_dir_use_symlinks=False,
            )
            results.append(
                ModelStatusItem(
                    name=name,
                    model_id=model_id,
                    cache_dir=cache_dir,
                    exists=True,
                    local_files_only=local_only,
                )
            )
        except Exception as e:
            results.append(
                ModelStatusItem(
                    name=name,
                    model_id=model_id,
                    cache_dir=cache_dir,
                    exists=False,
                    local_files_only=local_only,
                    error=str(e),
                )
            )
    return results


def _get_cache_root(cache_dir: str) -> Path:
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path(os.getenv("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)).expanduser()


def _get_repo_cache_dir(model_id: str, cache_dir: str) -> Path:
    return _get_cache_root(cache_dir) / f"models--{model_id.replace('/', '--')}"


def _read_local_main_revision(model_id: str, cache_dir: str) -> Optional[str]:
    refs_main = _get_repo_cache_dir(model_id, cache_dir) / "refs" / "main"
    try:
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            return revision or None
    except Exception:
        return None
    return None


def _normalize_endpoint(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    ep = endpoint.strip()
    if not ep:
        return None
    if "://" not in ep:
        ep = f"https://{ep}"
    return ep.rstrip("/")


def _get_hf_endpoint_candidates() -> List[Optional[str]]:
    """
    获取 HuggingFace endpoint 候选列表（按优先级）：
    1) RAG_HF_ENDPOINTS=ep1,ep2（显式多候选）
    2) HF_ENDPOINT / HF_MIRROR（兼容单 endpoint）
    3) RAG_HF_USE_CN_MIRROR=true 时附加 hf-mirror.com
    4) 最后回退官方 endpoint（None）
    """
    raw_multi = os.getenv("RAG_HF_ENDPOINTS", "").strip()
    candidates: List[Optional[str]] = []

    if raw_multi:
        for item in raw_multi.split(","):
            ep = _normalize_endpoint(item)
            if ep:
                candidates.append(ep)
    else:
        ep = _normalize_endpoint(os.getenv("HF_ENDPOINT", "")) or _normalize_endpoint(
            os.getenv("HF_MIRROR", "")
        )
        if ep:
            candidates.append(ep)

    use_cn_mirror = os.getenv("RAG_HF_USE_CN_MIRROR", "false").lower() == "true"
    if use_cn_mirror:
        candidates.append("https://hf-mirror.com")

    # 官方 endpoint 作为最终回退
    candidates.append(None)

    deduped: List[Optional[str]] = []
    seen = set()
    for ep in candidates:
        key = ep or "__default__"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ep)
    return deduped


def _fetch_remote_main_revision_with_fallback(model_id: str) -> Optional[str]:
    from huggingface_hub import HfApi

    for endpoint in _get_hf_endpoint_candidates():
        try:
            client = HfApi(endpoint=endpoint) if endpoint else HfApi()
            info = client.model_info(repo_id=model_id)
            sha = getattr(info, "sha", None)
            if sha:
                return sha
        except Exception as e:
            label = endpoint or "https://huggingface.co"
            logger.warning(
                f"[model_sync] fetch remote revision failed for {model_id} via {label}: {e}"
            )
    return None


def _snapshot_download_with_fallback(
    *,
    model_id: str,
    cache_dir: str,
    local_files_only: bool,
    force_download: bool,
) -> tuple[str, Optional[str]]:
    from huggingface_hub import snapshot_download

    if local_files_only:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            force_download=force_download,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return path, None

    last_error: Optional[Exception] = None
    for endpoint in _get_hf_endpoint_candidates():
        try:
            kwargs = {}
            if endpoint:
                kwargs["endpoint"] = endpoint
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
                force_download=force_download,
                local_dir_use_symlinks=False,
                resume_download=True,
                **kwargs,
            )
            return path, endpoint
        except Exception as e:
            last_error = e
            label = endpoint or "https://huggingface.co"
            logger.warning(
                f"[model_sync] snapshot download failed for {model_id} via {label}: {e}"
            )
    raise RuntimeError(f"all endpoints failed for {model_id}: {last_error}")


def sync_models(
    force_update: bool = False,
    local_files_only: Optional[bool] = None,
) -> List[ModelSyncItem]:
    """
    同步模型缓存：
    - force_update: True 时强制下载最新版本
    - local_files_only: True 时仅检查本地，False 时允许联网下载
    """
    local_only = (
        settings.model.local_files_only
        if local_files_only is None
        else bool(local_files_only)
    )
    results: List[ModelSyncItem] = []

    for name, model_id, cache_dir in _get_model_targets():
        try:
            local_revision_before = _read_local_main_revision(model_id, cache_dir)

            # 在线且非强制更新时：已是最新版本则跳过。
            if not local_only and not force_update:
                remote_revision = _fetch_remote_main_revision_with_fallback(model_id)
                if (
                    remote_revision
                    and local_revision_before
                    and remote_revision == local_revision_before
                ):
                    results.append(
                        ModelSyncItem(
                            name=name,
                            model_id=model_id,
                            cache_dir=cache_dir,
                            local_files_only=local_only,
                            updated=False,
                            status="already_latest",
                            message="已是最新版本，已跳过",
                        )
                    )
                    continue

            path, endpoint_used = _snapshot_download_with_fallback(
                model_id=model_id,
                cache_dir=cache_dir,
                local_files_only=local_only,
                force_download=bool(force_update),
            )
            local_revision_after = _read_local_main_revision(model_id, cache_dir)
            changed_revision = (
                bool(local_revision_before)
                and bool(local_revision_after)
                and local_revision_before != local_revision_after
            )
            first_download = (not local_revision_before) and bool(local_revision_after)
            updated = bool(force_update) or changed_revision or first_download
            if local_only:
                message = "离线模式：已校验本地缓存"
            elif updated:
                message = "模型已升级到最新版本" if changed_revision else "模型已下载到本地"
            else:
                message = "模型已是最新版本，未执行升级"
            if endpoint_used:
                message = f"{message}（endpoint: {endpoint_used}）"
            results.append(
                ModelSyncItem(
                    name=name,
                    model_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_only,
                    updated=updated,
                    status="ok",
                    message=message,
                    resolved_path=path,
                )
            )
        except Exception as e:
            logger.warning(f"[model_sync] {name} failed: {e}")
            results.append(
                ModelSyncItem(
                    name=name,
                    model_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_only,
                    updated=False,
                    status="failed",
                    message="模型同步失败",
                    error=str(e),
                )
            )
    return results
