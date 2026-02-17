"""
简化意图解析：Chat vs Deep Research 二分类。

- 检索由前端 search_mode 决定，不再由意图控制
- 意图只决定执行路径：普通对话 or 多步综述流水线
- 保留 /auto 命令触发 Deep Research，其余 / 命令均为 Chat 内 prompt hints
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

# ---------------------------------------------------------------------------
# 意图类型（简化为 3 种）
# ---------------------------------------------------------------------------

class IntentType(Enum):
    CHAT = "chat"                    # 普通对话（含搜索、写作、编辑、闲聊等）
    DEEP_RESEARCH = "deep_research"  # 多步综述流水线（原 auto_complete）
    UNCLEAR = "unclear"              # 无法判断，视为 CHAT

    # ---- 旧值兼容别名（前端/日志可能还存有旧值，反序列化时不报错）----
    @classmethod
    def _missing_(cls, value: object):
        """将旧 intent 值映射到新类型"""
        _LEGACY_MAP = {
            "search_exploratory": cls.CHAT,
            "search_targeted": cls.CHAT,
            "outline_generate": cls.CHAT,
            "outline_modify": cls.CHAT,
            "draft_section": cls.CHAT,
            "edit_text": cls.CHAT,
            "query_state": cls.CHAT,
            "set_directive": cls.CHAT,
            "export_document": cls.CHAT,
            "auto_complete": cls.DEEP_RESEARCH,
            "chitchat": cls.CHAT,
        }
        if isinstance(value, str):
            mapped = _LEGACY_MAP.get(value.strip().lower())
            if mapped is not None:
                return mapped
        return None


# 显式指令映射 —— /auto 触发 Deep Research，其余均为 Chat
COMMAND_PATTERNS = {
    "/auto": IntentType.DEEP_RESEARCH,
    "/search": IntentType.CHAT,
    "/explore": IntentType.CHAT,
    "/outline": IntentType.CHAT,
    "/outline add": IntentType.CHAT,
    "/outline delete": IntentType.CHAT,
    "/outline move": IntentType.CHAT,
    "/draft": IntentType.CHAT,
    "/edit": IntentType.CHAT,
    "/status": IntentType.CHAT,
    "/export": IntentType.CHAT,
    "/set": IntentType.CHAT,
}


@dataclass
class ParsedIntent:
    intent_type: IntentType
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    raw_input: str = ""
    from_command: bool = False


def is_deep_research(parsed: ParsedIntent) -> bool:
    """判断是否走 Deep Research 路径"""
    return parsed.intent_type == IntentType.DEEP_RESEARCH


# 保留旧函数名作为兼容（但逻辑已改为：retrieval 由 search_mode 决定）
def is_retrieval_intent(parsed: ParsedIntent) -> bool:
    """兼容旧调用：Deep Research 一定需要检索"""
    return is_deep_research(parsed)


class IntentParser:
    """简化版意图解析：/ 命令优先，否则 LLM 二分类。"""

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    def parse(
        self,
        user_input: str,
        current_stage: str = "explore",
        history: Optional[Iterable[Any]] = None,
    ) -> ParsedIntent:
        user_input = (user_input or "").strip()
        if user_input.startswith("/"):
            return self._parse_command(user_input)
        return self._parse_natural_language(user_input, current_stage, history=history)

    def _parse_command(self, user_input: str) -> ParsedIntent:
        parts = user_input.split(maxsplit=2)
        cmd = parts[0].lower()
        if len(parts) >= 2:
            compound = f"{parts[0]} {parts[1]}".lower()
            if compound in COMMAND_PATTERNS:
                return ParsedIntent(
                    intent_type=COMMAND_PATTERNS[compound],
                    confidence=1.0,
                    params={"args": (parts[2] if len(parts) > 2 else "").strip()},
                    raw_input=user_input,
                    from_command=True,
                )
        if cmd in COMMAND_PATTERNS:
            return ParsedIntent(
                intent_type=COMMAND_PATTERNS[cmd],
                confidence=1.0,
                params={"args": (parts[1] if len(parts) > 1 else "").strip()},
                raw_input=user_input,
                from_command=True,
            )
        # 未知 / 命令
        return ParsedIntent(
            intent_type=IntentType.CHAT,
            confidence=0.5,
            raw_input=user_input,
            from_command=True,
        )

    def _parse_natural_language(
        self,
        user_input: str,
        current_stage: str,
        history: Optional[Iterable[Any]] = None,
    ) -> ParsedIntent:
        history_block = _format_history(history)
        prompt = f"""判断用户意图属于以下哪一类。当前工作阶段: {current_stage}

对话上下文:
{history_block or "（无）"}

用户输入: "{user_input}"

只有两种意图：
- chat: 普通对话、提问、搜索、写作、编辑、闲聊 —— 绝大多数情况都是 chat
- deep_research: 用户明确要求生成一篇完整的多章节综述/报告（如"帮我写一篇完整综述"、"自动完成一篇关于XXX的报告"）

判断原则：
1. 如果用户只是提问、搜索文献、写某个段落、编辑文本、闲聊 → chat
2. 只有用户明确表达要"完整综述"、"全文生成"、"一键综述"时 → deep_research
3. 不确定时默认 chat

请只返回一行 JSON，不要其他文字:
{{"intent": "chat 或 deep_research", "confidence": 0.0-1.0, "params": {{}}}}"""
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": "你是一个意图分析助手，只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
            )
            text = (resp.get("final_text") or "").strip()
        except Exception:
            return ParsedIntent(
                intent_type=IntentType.CHAT,
                confidence=0.5,
                raw_input=user_input,
            )
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        try:
            data = json.loads(text)
            intent_str = (data.get("intent") or "chat").strip().lower()
            intent_type = IntentType(intent_str)
        except (ValueError, KeyError, json.JSONDecodeError):
            return ParsedIntent(
                intent_type=IntentType.CHAT,
                confidence=0.5,
                raw_input=user_input,
            )
        return ParsedIntent(
            intent_type=intent_type,
            confidence=float(data.get("confidence", 0.8)),
            params=dict(data.get("params") or {}),
            raw_input=user_input,
            from_command=False,
        )


def _format_history(history: Optional[Iterable[Any]], max_turns: int = 6, max_len: int = 200) -> str:
    if not history:
        return ""
    turns = list(history)
    if max_turns > 0:
        turns = turns[-max_turns:]
    lines = []
    for t in turns:
        role = getattr(t, "role", "") or ""
        content = getattr(t, "content", "") or ""
        if not isinstance(content, str):
            continue
        text = content.strip().replace("\n", " ")
        if max_len and len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        role_label = "用户" if role == "user" else "助手"
        lines.append(f"{role_label}: {text}")
    return "\n".join(lines)
