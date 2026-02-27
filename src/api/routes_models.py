"""
模型管理 API：
- GET  /models/status               — 本地 embedding/reranker 模型状态
- POST /models/sync                 — 同步本地模型
- GET  /llm/platforms               — 列出已配置的平台凭据
- GET  /llm/providers               — 列出可用 LLM provider 及配置（静态）
- GET  /llm/providers/registry      — 列出所有支持的 provider 模板
- GET  /llm/models                  — 并行实时拉取所有平台可用模型列表
- GET  /llm/ultra_lite_providers     — 可用于长文本压缩的 provider（与拉取模型比对，仅返回可识别的）
- GET  /llm/providers/{name}/models — 动态拉取指定 provider 的可用模型列表
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ModelStatusResponse, ModelSyncRequest, ModelSyncResponse
from src.utils.model_sync import check_models, sync_models

_log = logging.getLogger(__name__)

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


@router.get("/llm/platforms")
def list_llm_platforms() -> dict:
    """列出已配置的平台凭据（脱敏显示 api_key）"""
    from src.llm.llm_manager import get_manager, mask_secret

    manager = get_manager(str(_CONFIG_PATH))
    platforms = []
    for name, pcfg in manager.config.platforms.items():
        platforms.append({
            "name": name,
            "base_url": pcfg.base_url,
            "has_key": bool(pcfg.api_key and pcfg.api_key not in ("sk-xxx", "sk-ant-xxx", "AIzxxx", "")),
            "key_hint": mask_secret(pcfg.api_key) if pcfg.api_key else "",
        })
    return {"platforms": platforms}


@router.get("/llm/providers")
def list_llm_providers() -> dict:
    """列出可用的 LLM 提供商及其默认模型、platform、image 能力标记"""
    from src.llm.llm_manager import get_manager
    from src.llm.model_registry import get_registry

    manager = get_manager(str(_CONFIG_PATH))
    registry = get_registry()
    supported = registry.supported_providers()

    providers = []
    for name in manager.get_provider_names():
        cfg = manager.config.providers.get(name)
        if not cfg:
            continue

        registry_key = registry.resolve_provider_for_config(name)
        meta = supported.get(registry_key) if registry_key else None

        providers.append({
            "id": name,
            "platform": cfg.platform,
            "default_model": cfg.default_model,
            "models": list(cfg.models.keys()) if cfg.models else [cfg.default_model] if cfg.default_model else [],
            "supports_image": meta.supports_image if meta else False,
            "registry_key": registry_key,
            "label": meta.label if meta else name,
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


@router.get("/llm/providers/registry")
def list_provider_registry() -> dict:
    """列出所有支持的 provider 模板信息（用于前端"添加新 provider"下拉）"""
    from src.llm.model_registry import get_registry

    registry = get_registry()
    supported = registry.supported_providers()

    templates = []
    for name, meta in supported.items():
        templates.append({
            "name": meta.name,
            "label": meta.label,
            "default_base_url": meta.default_base_url,
            "supports_image": meta.supports_image,
            "env_key_hint": meta.env_key_hint,
        })

    return {"providers": templates}


@router.get("/llm/models")
def list_all_live_models(
    no_cache: bool = Query(False, description="跳过缓存，强制重新拉取"),
) -> dict:
    """
    并行实时拉取所有已配置平台的可用模型列表。

    - 每个平台独立并发请求，单个超时不影响其他平台
    - 结果按 platform 分组返回；拉取失败的平台返回 error 字段
    - 前端调用此接口后，用实时结果替换静态配置中的 models 列表
    """
    from src.llm.llm_manager import get_manager
    from src.llm.model_registry import get_registry, ModelInfo

    manager = get_manager(str(_CONFIG_PATH))
    registry = get_registry()

    # 收集每个 platform 的 api_key + registry_key
    # 注意：model listing 用 model_registry 的默认 URL，不用 config 的 base_url。
    # config 的 base_url 是 chat completions 端点（可能是 OpenAI-compat 代理），
    # 而 model listing 需要原生 API 端点（如 Gemini 用 /v1beta 而非 /v1beta/openai）。
    platforms_to_fetch: dict[str, dict] = {}
    for pname, pcfg in manager.config.platforms.items():
        if not pcfg.api_key or pcfg.api_key in ("sk-xxx", "sk-ant-xxx", "AIzxxx", ""):
            continue
        rk = registry.resolve_provider_for_config(pname)
        if not rk:
            continue
        platforms_to_fetch[pname] = {
            "api_key": pcfg.api_key,
            "registry_key": rk,
        }

    PER_PLATFORM_TIMEOUT = 12  # seconds

    def _fetch(pname: str, info: dict) -> tuple[str, list[ModelInfo] | None, str | None]:
        try:
            models = registry.fetch_models(
                info["registry_key"],
                info["api_key"],
                use_cache=not no_cache,
            )
            return pname, models, None
        except Exception as exc:
            return pname, None, str(exc)

    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=len(platforms_to_fetch) or 1) as pool:
        futures = {
            pool.submit(_fetch, pname, info): pname
            for pname, info in platforms_to_fetch.items()
        }
        try:
            for future in as_completed(futures, timeout=PER_PLATFORM_TIMEOUT + 2):
                pname, models, error = future.result()
                if models is not None:
                    results[pname] = {
                        "models": [m.id for m in models],
                        "count": len(models),
                    }
                else:
                    results[pname] = {"models": [], "count": 0, "error": error}
        except FuturesTimeout:
            _log.warning("as_completed timed out; some platforms did not respond in time")

    for pname in platforms_to_fetch:
        if pname not in results:
            results[pname] = {"models": [], "count": 0, "error": "timeout"}

    return {"platforms": results}


# 用于长文本压缩等超轻量任务的 provider 配置键（与拉取模型列表比对后只返回可识别的）
ULTRA_LITE_PROVIDER_KEYS = ["openai-mini", "gemini-flash", "deepseek", "claude-haiku"]

# 展示用标签（provider_id -> 前端显示名）
ULTRA_LITE_LABELS = {
    "openai-mini": "OpenAI (gpt-5-mini)",
    "gemini-flash": "Gemini Flash (gemini-flash-latest)",
    "deepseek": "DeepSeek (deepseek-chat)",
    "claude-haiku": "Claude Haiku (claude-haiku-4-5)",
}


@router.get("/llm/ultra_lite_providers")
def list_ultra_lite_providers(
    no_cache: bool = Query(False, description="跳过缓存，强制重新拉取平台模型列表"),
) -> dict:
    """
    返回可用于 Ultra Lite（长文本压缩等）的 provider 列表，
    仅包含其 default_model 在「拉取下来的」平台模型列表中出现过的项，供前端下拉选择。
    """
    from src.llm.llm_manager import get_manager
    from src.llm.model_registry import get_registry

    manager = get_manager(str(_CONFIG_PATH))
    registry = get_registry()

    # 1) 拉取各平台 live 模型列表（复用 list_all_live_models 逻辑）
    platforms_to_fetch = {}
    for pname, pcfg in manager.config.platforms.items():
        if not pcfg.api_key or pcfg.api_key in ("sk-xxx", "sk-ant-xxx", "AIzxxx", ""):
            continue
        rk = registry.resolve_provider_for_config(pname)
        if not rk:
            continue
        platforms_to_fetch[pname] = {"api_key": pcfg.api_key, "registry_key": rk}

    live_models_by_platform: dict[str, list[str]] = {}
    for pname, info in platforms_to_fetch.items():
        try:
            models = registry.fetch_models(
                info["registry_key"],
                info["api_key"],
                use_cache=not no_cache,
            )
            live_models_by_platform[pname] = [m.id for m in models]
        except Exception as exc:
            _log.debug("ultra_lite_providers: fetch %s failed: %s", pname, exc)
            live_models_by_platform[pname] = []

    def _model_recognized(want: str, live_ids: list[str]) -> bool:
        if not want or not live_ids:
            return False
        want_lower = want.lower()
        for mid in live_ids:
            if want_lower == mid.lower() or want_lower in mid.lower() or mid.lower().startswith(want_lower):
                return True
        return False

    # 2) 只保留在 config 中且 default_model 在对应平台 live 列表中被识别的
    providers = []
    for provider_id in ULTRA_LITE_PROVIDER_KEYS:
        cfg = manager.config.providers.get(provider_id)
        if not cfg:
            continue
        platform = getattr(cfg, "platform", None) or ""
        default_model = getattr(cfg, "default_model", None) or ""
        if not platform or not default_model:
            continue
        live_ids = live_models_by_platform.get(platform, [])
        if not _model_recognized(default_model, live_ids):
            continue
        providers.append({
            "id": provider_id,
            "label": ULTRA_LITE_LABELS.get(provider_id, provider_id),
            "default_model": default_model,
            "platform": platform,
        })

    return {"providers": providers, "default": "openai-mini" if any(p["id"] == "openai-mini" for p in providers) else (providers[0]["id"] if providers else None)}


@router.get("/llm/providers/{provider_name}/models")
def fetch_provider_models(
    provider_name: str,
    api_key: str = Query(None, description="API Key（不传则使用配置文件/平台中的 key）"),
    base_url: str = Query(None, description="自定义 base_url（不传则使用默认值）"),
    no_cache: bool = Query(False, description="跳过缓存，强制重新拉取"),
) -> dict:
    """
    动态拉取指定 provider 的可用模型列表。

    api_key 解析优先级：参数传入 > provider 配置 > platform 配置
    支持 rag_config.json 中的 provider 名（如 'openai-thinking'）或
    registry 标准名（如 'openai'）。
    """
    from src.llm.llm_manager import get_manager
    from src.llm.model_registry import get_registry

    registry = get_registry()

    registry_key = registry.resolve_provider_for_config(provider_name)
    if not registry_key:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider_name}. "
                   f"Supported: {list(registry.supported_providers().keys())}",
        )

    resolved_api_key = api_key
    resolved_base_url = base_url

    if not resolved_api_key or not resolved_base_url:
        try:
            manager = get_manager(str(_CONFIG_PATH))

            # 1) 精确匹配 provider 名
            cfg = manager.config.providers.get(provider_name)
            if cfg:
                if not resolved_api_key:
                    resolved_api_key = cfg.api_key
                if not resolved_base_url:
                    resolved_base_url = cfg.base_url
            else:
                # 2) 按 registry_key 找同一平台的任意 provider
                for p_name, p_cfg in manager.config.providers.items():
                    rk = registry.resolve_provider_for_config(p_name)
                    if rk == registry_key:
                        if not resolved_api_key and p_cfg.api_key:
                            resolved_api_key = p_cfg.api_key
                        if not resolved_base_url and p_cfg.base_url:
                            resolved_base_url = p_cfg.base_url
                        if resolved_api_key:
                            break

            # 3) 直接查 platform
            if not resolved_api_key:
                plat = manager.config.platforms.get(registry_key)
                if plat and plat.api_key:
                    resolved_api_key = plat.api_key
                    if not resolved_base_url:
                        resolved_base_url = plat.base_url
        except Exception:
            pass

    if not resolved_api_key:
        raise HTTPException(
            status_code=400,
            detail=f"No API key found for provider '{provider_name}'. "
                   "Pass api_key as query parameter or configure it in rag_config.json platforms.",
        )

    try:
        models = registry.fetch_models(
            registry_key,
            resolved_api_key,
            base_url=resolved_base_url,
            use_cache=not no_cache,
        )
    except Exception as e:
        _log.warning("Failed to fetch models for %s: %s", provider_name, e)
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {e}")

    return {
        "provider": provider_name,
        "registry_key": registry_key,
        "models": [
            {
                "id": m.id,
                "owned_by": m.owned_by,
                "supports_image": m.supports_image,
                **({"extra": m.extra} if m.extra else {}),
            }
            for m in models
        ],
        "count": len(models),
    }
