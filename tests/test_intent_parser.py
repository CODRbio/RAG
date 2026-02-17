"""
IntentParser 单元测试：命令解析 / 自然语言分类 / 边界情况
"""

import json
import pytest
from unittest.mock import MagicMock

from src.collaboration.intent.parser import (
    IntentType,
    IntentParser,
    ParsedIntent,
    is_deep_research,
    is_retrieval_intent,
)


# ── IntentType 枚举 ──

class TestIntentType:
    def test_basic_values(self):
        assert IntentType.CHAT.value == "chat"
        assert IntentType.DEEP_RESEARCH.value == "deep_research"
        assert IntentType.UNCLEAR.value == "unclear"

    def test_legacy_mapping(self):
        """旧值应正确映射到新类型"""
        assert IntentType("search_exploratory") == IntentType.CHAT
        assert IntentType("auto_complete") == IntentType.DEEP_RESEARCH
        assert IntentType("outline_generate") == IntentType.CHAT
        assert IntentType("chitchat") == IntentType.CHAT

    def test_unknown_legacy_returns_none(self):
        with pytest.raises(ValueError):
            IntentType("nonexistent_intent")


# ── 命令解析 ──

class TestCommandParsing:
    @pytest.fixture
    def parser(self, mock_llm_client):
        return IntentParser(mock_llm_client)

    def test_auto_command(self, parser):
        result = parser.parse("/auto")
        assert result.intent_type == IntentType.DEEP_RESEARCH
        assert result.confidence == 1.0
        assert result.from_command is True

    def test_auto_with_args(self, parser):
        result = parser.parse("/auto 写一篇关于深海的综述")
        assert result.intent_type == IntentType.DEEP_RESEARCH
        assert "深海" in result.params.get("args", "")

    def test_search_command(self, parser):
        result = parser.parse("/search hydrothermal vents")
        assert result.intent_type == IntentType.CHAT
        assert result.from_command is True

    def test_outline_add_compound(self, parser):
        result = parser.parse("/outline add Introduction")
        assert result.intent_type == IntentType.CHAT
        assert result.from_command is True

    def test_unknown_command(self, parser):
        result = parser.parse("/unknown_cmd blah")
        assert result.intent_type == IntentType.CHAT
        assert result.confidence == 0.5

    def test_all_known_commands(self, parser):
        cmds = ["/search", "/explore", "/outline", "/draft", "/edit", "/status", "/export", "/set"]
        for cmd in cmds:
            result = parser.parse(cmd)
            assert result.intent_type == IntentType.CHAT
            assert result.from_command is True


# ── 自然语言解析 ──

class TestNaturalLanguageParsing:
    def test_chat_intent(self, mock_llm_client):
        mock_llm_client.chat.return_value = {
            "final_text": json.dumps({"intent": "chat", "confidence": 0.95, "params": {}})
        }
        parser = IntentParser(mock_llm_client)
        result = parser.parse("什么是深海热泉？")
        assert result.intent_type == IntentType.CHAT
        assert result.confidence >= 0.9

    def test_deep_research_intent(self, mock_llm_client):
        # 需要清除 conftest 中设的 side_effect，否则 return_value 不生效
        mock_llm_client.chat.side_effect = None
        mock_llm_client.chat.return_value = {
            "final_text": json.dumps({"intent": "deep_research", "confidence": 0.92, "params": {}})
        }
        parser = IntentParser(mock_llm_client)
        result = parser.parse("帮我写一篇关于深海化能合成的完整综述")
        assert result.intent_type == IntentType.DEEP_RESEARCH

    def test_llm_failure_fallback(self, mock_llm_client):
        """LLM 调用失败时应回退到 CHAT"""
        mock_llm_client.chat.side_effect = Exception("API timeout")
        parser = IntentParser(mock_llm_client)
        result = parser.parse("test query")
        assert result.intent_type == IntentType.CHAT
        assert result.confidence <= 0.5

    def test_llm_invalid_json(self, mock_llm_client):
        """LLM 返回非法 JSON 应回退"""
        mock_llm_client.chat.return_value = {"final_text": "not valid json at all"}
        parser = IntentParser(mock_llm_client)
        result = parser.parse("some question")
        assert result.intent_type == IntentType.CHAT

    def test_empty_input(self, mock_llm_client):
        parser = IntentParser(mock_llm_client)
        result = parser.parse("")
        # 空输入走 NL 路径（不以 / 开头）
        assert isinstance(result, ParsedIntent)

    def test_with_history(self, mock_llm_client):
        mock_llm_client.chat.return_value = {
            "final_text": json.dumps({"intent": "chat", "confidence": 0.85, "params": {}})
        }
        parser = IntentParser(mock_llm_client)
        history = [{"role": "user", "content": "之前的问题"}, {"role": "assistant", "content": "之前的回答"}]
        result = parser.parse("继续", history=history)
        assert isinstance(result, ParsedIntent)


# ── 辅助函数 ──

class TestHelpers:
    def test_is_deep_research_true(self):
        p = ParsedIntent(intent_type=IntentType.DEEP_RESEARCH, confidence=1.0)
        assert is_deep_research(p) is True

    def test_is_deep_research_false(self):
        p = ParsedIntent(intent_type=IntentType.CHAT, confidence=1.0)
        assert is_deep_research(p) is False

    def test_is_retrieval_intent_compat(self):
        """兼容函数应与 is_deep_research 行为一致"""
        p = ParsedIntent(intent_type=IntentType.DEEP_RESEARCH, confidence=1.0)
        assert is_retrieval_intent(p) == is_deep_research(p)
