"""
意图识别：/ 命令 + LLM 二分类（Chat vs Deep Research）。
"""

from .commands import build_search_query_from_context, get_search_query_from_intent
from .parser import (
    COMMAND_PATTERNS,
    IntentParser,
    IntentType,
    ParsedIntent,
    is_deep_research,
    is_retrieval_intent,  # 兼容旧调用
)

__all__ = [
    "COMMAND_PATTERNS",
    "IntentParser",
    "IntentType",
    "ParsedIntent",
    "is_deep_research",
    "is_retrieval_intent",
    "get_search_query_from_intent",
    "build_search_query_from_context",
]
