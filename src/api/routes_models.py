"""
本地模型管理 API：
- GET /models/status
- POST /models/sync
- GET /llm/providers
"""

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter

from src.api.schemas import ModelStatusResponse, ModelSyncRequest, ModelSyncResponse
from src.utils.model_sync import check_models, sync_models

router = APIRouter(tags=["models"])

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"


@router.get("/models/status", response_model=ModelStatusResponse)
def get_model_status() -> dict:
    items = check_models()
    return {"items": [asdict(i) for i in items]}


@router.post("/models/sync", response_model=ModelSyncResponse)
def sync_local_models(body: ModelSyncRequest) -> dict:
    items = sync_models(force_update=body.force_update, local_files_only=body.local_files_only)
    return {"items": [asdict(i) for i in items]}


@router.get("/llm/providers")
def list_llm_providers() -> dict:
    """列出可用的 LLM 提供商及其默认模型"""
    from src.llm.llm_manager import get_manager
    manager = get_manager(str(_CONFIG_PATH))
    providers = []
    for name in manager.get_provider_names():
        cfg = manager.config.providers.get(name)
        if cfg:
            providers.append({
                "id": name,
                "default_model": cfg.default_model,
                "models": list(cfg.models.keys()),
            })
    parser_defaults = {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parser_cfg = raw.get("parser", {}) if isinstance(raw, dict) else {}
        parser_defaults = {
            "llm_text_provider": parser_cfg.get("llm_text_provider", "deepseek"),
            "llm_text_model": parser_cfg.get("llm_text_model"),
            "llm_text_concurrency": int(parser_cfg.get("llm_text_concurrency", 1) or 1),
            "llm_vision_provider": parser_cfg.get("llm_vision_provider", "gemini-vision"),
            "llm_vision_model": parser_cfg.get("llm_vision_model"),
            "llm_vision_concurrency": int(parser_cfg.get("llm_vision_concurrency", 1) or 1),
        }
    except Exception:
        parser_defaults = {
            "llm_text_provider": "deepseek",
            "llm_text_model": None,
            "llm_text_concurrency": 1,
            "llm_vision_provider": "gemini-vision",
            "llm_vision_model": None,
            "llm_vision_concurrency": 1,
        }

    return {
        "default": manager.config.default,
        "providers": providers,
        "parser_defaults": parser_defaults,
    }
