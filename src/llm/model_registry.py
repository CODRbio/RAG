"""
动态模型注册表 —— 按 API Key 实时拉取各厂商可用模型

支持的 Provider:
  OpenAI / DeepSeek / Qwen / Kimi / Perplexity  — OpenAI-compatible /models
  Claude (Anthropic)                              — x-api-key + anthropic-version
  Gemini (Google)                                 — query-param key

每个 provider 可标注 supports_image 以便前端做图片解析路由。
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import requests

_log = logging.getLogger(__name__)

_FETCH_TIMEOUT = 15  # seconds


# ============================================================
# Data structures
# ============================================================

@dataclass
class ProviderMeta:
    """Provider 元信息（注册表条目）"""
    name: str
    label: str
    default_base_url: str
    supports_image: bool = False
    env_key_hint: str = ""


@dataclass
class ModelInfo:
    """单个模型的描述信息"""
    id: str
    owned_by: str = ""
    supports_image: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Base class
# ============================================================

class ModelProvider(ABC):
    """所有 Provider 的基类"""

    meta: ProviderMeta

    def __init__(self, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = (base_url or self.meta.default_base_url).rstrip("/")

    @abstractmethod
    def fetch_models(self) -> List[ModelInfo]:
        """从远程 API 拉取可用模型列表"""
        ...


# ============================================================
# OpenAI-compatible provider (shared by many vendors)
# ============================================================

class OpenAICompatibleProvider(ModelProvider):
    """处理 OpenAI 及其兼容 API（DeepSeek, Qwen, Kimi, Perplexity）"""

    # 子类可覆盖：正则白名单，None 表示不过滤
    _model_include_re: re.Pattern | None = None
    # 子类可覆盖：正则黑名单
    _model_exclude_re: re.Pattern | None = None

    def fetch_models(self) -> List[ModelInfo]:
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw_models = data.get("data") or []
        results: List[ModelInfo] = []
        for m in raw_models:
            mid = m.get("id", "")
            if not mid:
                continue
            if self._model_include_re and not self._model_include_re.search(mid):
                continue
            if self._model_exclude_re and self._model_exclude_re.search(mid):
                continue
            results.append(ModelInfo(
                id=mid,
                owned_by=m.get("owned_by", ""),
            ))
        results.sort(key=lambda x: x.id)
        return results


# ============================================================
# Concrete providers
# ============================================================

_VISION_PATTERN = re.compile(
    r"(vision|vl|image|visual|multimodal|4o|gpt-5|gemini|claude-.*sonnet|claude-.*opus"
    r"|claude-.*haiku|kimi-.*vl|qwen-.*vl)",
    re.IGNORECASE,
)


def _guess_image_support(model_id: str) -> bool:
    return bool(_VISION_PATTERN.search(model_id))


class OpenAIProvider(OpenAICompatibleProvider):
    meta = ProviderMeta(
        name="openai",
        label="OpenAI",
        default_base_url="https://api.openai.com/v1",
        supports_image=True,
        env_key_hint="OPENAI_API_KEY",
    )
    _model_exclude_re = re.compile(r"(whisper|tts|dall-e|embedding|davinci|babbage|moderation)", re.I)


class DeepSeekProvider(OpenAICompatibleProvider):
    meta = ProviderMeta(
        name="deepseek",
        label="DeepSeek",
        default_base_url="https://api.deepseek.com/v1",
        supports_image=False,
        env_key_hint="DEEPSEEK_API_KEY",
    )


class QwenProvider(OpenAICompatibleProvider):
    meta = ProviderMeta(
        name="qwen",
        label="Qwen (通义千问)",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        supports_image=True,
        env_key_hint="QWEN_API_KEY",
    )


class KimiProvider(OpenAICompatibleProvider):
    meta = ProviderMeta(
        name="kimi",
        label="Kimi (Moonshot)",
        default_base_url="https://api.moonshot.ai/v1",
        supports_image=True,
        env_key_hint="KIMI_API_KEY",
    )


class PerplexityProvider(OpenAICompatibleProvider):
    meta = ProviderMeta(
        name="perplexity",
        label="Perplexity (Sonar)",
        default_base_url="https://api.perplexity.ai",
        supports_image=False,
        env_key_hint="PERPLEXITY_API_KEY",
    )

    _FALLBACK_MODELS = [
        "sonar-reasoning-pro", "sonar-reasoning",
        "sonar-pro", "sonar", "sonar-deep-research",
    ]

    def fetch_models(self) -> List[ModelInfo]:
        try:
            return super().fetch_models()
        except (requests.exceptions.RequestException, KeyError):
            _log.info("Perplexity /models endpoint unavailable, using fallback list")
            return [ModelInfo(id=m) for m in self._FALLBACK_MODELS]


class ClaudeProvider(ModelProvider):
    meta = ProviderMeta(
        name="claude",
        label="Claude (Anthropic)",
        default_base_url="https://api.anthropic.com",
        supports_image=True,
        env_key_hint="ANTHROPIC_API_KEY",
    )

    def fetch_models(self) -> List[ModelInfo]:
        url = f"{self.base_url}/v1/models"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw_models = data.get("data") or []
        results = [
            ModelInfo(id=m["id"], owned_by="anthropic")
            for m in raw_models if m.get("id")
        ]
        results.sort(key=lambda x: x.id)
        return results


class GeminiProvider(ModelProvider):
    meta = ProviderMeta(
        name="gemini",
        label="Gemini (Google)",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        supports_image=True,
        env_key_hint="GEMINI_API_KEY",
    )

    _EXCLUDE_RE = re.compile(r"(embedding|aqa|imagen|veo|chirp|text-bison|chat-bison)", re.I)

    def fetch_models(self) -> List[ModelInfo]:
        url = f"{self.base_url}/models"
        params = {"key": self.api_key}
        resp = requests.get(url, params=params, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw_models = data.get("models") or []
        results: List[ModelInfo] = []
        for m in raw_models:
            name = m.get("name", "")
            model_id = name.replace("models/", "") if name.startswith("models/") else name
            if not model_id:
                continue
            if self._EXCLUDE_RE.search(model_id):
                continue
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            results.append(ModelInfo(
                id=model_id,
                owned_by="google",
                extra={
                    "display_name": m.get("displayName", ""),
                    "input_token_limit": m.get("inputTokenLimit"),
                    "output_token_limit": m.get("outputTokenLimit"),
                },
            ))
        results.sort(key=lambda x: x.id)
        return results


# ============================================================
# Registry
# ============================================================

_PROVIDER_CLASSES: Dict[str, Type[ModelProvider]] = {
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "kimi": KimiProvider,
    "perplexity": PerplexityProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
}


@dataclass
class _CacheEntry:
    models: List[ModelInfo]
    fetched_at: float


class ModelRegistry:
    """
    动态模型注册表。

    用法:
        registry = ModelRegistry()
        models = registry.fetch_models("openai", api_key="sk-xxx")
        models = registry.fetch_models("claude", api_key="sk-ant-xxx")
    """

    _CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        self._cache: Dict[str, _CacheEntry] = {}

    @staticmethod
    def supported_providers() -> Dict[str, ProviderMeta]:
        """返回所有支持的 provider 元信息"""
        return {name: cls.meta for name, cls in _PROVIDER_CLASSES.items()}

    def fetch_models(
        self,
        provider_name: str,
        api_key: str,
        base_url: str | None = None,
        *,
        use_cache: bool = True,
    ) -> List[ModelInfo]:
        """
        拉取指定 provider 的可用模型列表。

        Args:
            provider_name: provider 标识（openai / deepseek / qwen / kimi / perplexity / claude / gemini）
            api_key: API Key
            base_url: 可选自定义 base_url（覆盖默认值）
            use_cache: 是否使用缓存（默认 True，TTL 5 分钟）

        Returns:
            ModelInfo 列表
        """
        key = provider_name.lower()
        provider_cls = _PROVIDER_CLASSES.get(key)
        if not provider_cls:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported: {list(_PROVIDER_CLASSES.keys())}"
            )

        cache_key = f"{key}:{base_url or 'default'}"
        if use_cache:
            entry = self._cache.get(cache_key)
            if entry and (time.time() - entry.fetched_at) < self._CACHE_TTL:
                return entry.models

        try:
            provider = provider_cls(api_key=api_key, base_url=base_url)
            models = provider.fetch_models()
            self._cache[cache_key] = _CacheEntry(models=models, fetched_at=time.time())
            return models
        except requests.exceptions.RequestException as e:
            _log.warning("[%s] Failed to fetch models: %s", provider_name, e)
            entry = self._cache.get(cache_key)
            if entry:
                return entry.models
            return []

    def invalidate_cache(self, provider_name: str | None = None) -> None:
        """清除缓存。provider_name=None 时清除全部。"""
        if provider_name is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{provider_name}:")]
            for k in keys_to_remove:
                del self._cache[k]

    def resolve_provider_for_config(
        self,
        config_provider_name: str,
    ) -> str | None:
        """
        将 rag_config.json 中的 provider 名（如 'openai-thinking', 'gemini-vision'）
        映射到注册表的 provider key。

        规则: 取 '-' 前的基础名，如果匹配注册表则返回。
        """
        base = config_provider_name.split("-")[0].lower()
        if base in _PROVIDER_CLASSES:
            return base
        if config_provider_name.lower() in _PROVIDER_CLASSES:
            return config_provider_name.lower()
        if config_provider_name.lower().startswith("sonar"):
            return "perplexity"
        return None


# Global singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """获取全局 ModelRegistry 单例"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
