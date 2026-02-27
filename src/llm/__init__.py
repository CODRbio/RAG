"""
LLM 统一管理模块

提供:
- LLMManager: 配置加载与客户端获取
- BaseChatClient: 统一调用接口
- RawLogStore: 原始响应日志
- ModelRegistry: 动态模型列表拉取
"""

from .llm_manager import (
    LLMManager,
    BaseChatClient,
    HTTPChatClient,
    DryRunChatClient,
    RawLogStore,
    PlatformConfig,
    ProviderConfig,
    LLMConfig,
)
from .model_registry import (
    ModelRegistry,
    ModelInfo,
    ProviderMeta,
    get_registry,
)

__all__ = [
    "LLMManager",
    "BaseChatClient",
    "HTTPChatClient",
    "DryRunChatClient",
    "RawLogStore",
    "PlatformConfig",
    "ProviderConfig",
    "LLMConfig",
    "ModelRegistry",
    "ModelInfo",
    "ProviderMeta",
    "get_registry",
]
