"""
意图识别：/ 命令 + LLM 二分类（Chat vs Deep Research）+ 统一上下文分析。
"""

from .commands import (
    ContextAnalysis,
    COLLECTION_SCOPE,
    analyze_chat_context,
    build_search_query_from_context,
    check_query_collection_scope,
    get_search_query_from_intent,
)
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
    "ContextAnalysis",
    "IntentParser",
    "IntentType",
    "ParsedIntent",
    "analyze_chat_context",
    "is_deep_research",
    "is_retrieval_intent",
    "get_search_query_from_intent",
    "build_search_query_from_context",
    "COLLECTION_SCOPE",
    "check_query_collection_scope",
]
