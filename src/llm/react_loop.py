"""
ReAct 执行循环：LLM → tool_call → execute → feed back → repeat。

支持 OpenAI / Anthropic 两种 tool calling 协议，自动降级为 prompt-based。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.tools import (
    ToolDef,
    ToolCall,
    ToolResult,
    execute_tool_call,
    parse_tool_calls,
    has_tool_calls,
    tool_result_to_openai_message,
    tool_result_to_anthropic_content,
    tools_to_prompt,
)
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class ReactResult:
    """ReAct 循环的最终结果"""
    final_text: str = ""
    reasoning_text: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)  # 工具调用记录
    iterations: int = 0
    total_time_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


def react_loop(
    messages: List[Dict[str, Any]],
    tools: List[ToolDef],
    llm_client: Any,
    max_iterations: int = 10,
    model: Optional[str] = None,
    **llm_kwargs,
) -> ReactResult:
    """
    ReAct 循环核心。

    Args:
        messages: 初始消息列表（含 system + user）
        tools: 可用工具列表
        llm_client: BaseChatClient 实例
        max_iterations: 最大迭代次数（防止无限循环）
        model: 可选模型覆盖
        **llm_kwargs: 传递给 llm_client.chat() 的额外参数

    Returns:
        ReactResult
    """
    t0 = time.perf_counter()
    result = ReactResult()

    # 检测 provider 类型
    is_anthropic = getattr(getattr(llm_client, "config", None), "is_anthropic", lambda: False)()

    # 决定是否使用原生 FC
    supports_fc = True  # 当前所有 provider 都支持
    working_messages = list(messages)

    # 如果不支持 FC，注入 prompt-based tool 描述到 system message
    if not supports_fc:
        prompt_desc = tools_to_prompt(tools)
        if working_messages and working_messages[0].get("role") == "system":
            working_messages[0] = {
                **working_messages[0],
                "content": working_messages[0]["content"] + "\n\n" + prompt_desc,
            }
        else:
            working_messages.insert(0, {"role": "system", "content": prompt_desc})

    for iteration in range(max_iterations):
        result.iterations = iteration + 1

        # 调用 LLM
        try:
            resp = llm_client.chat(
                messages=working_messages,
                model=model,
                tools=tools if supports_fc else None,
                **llm_kwargs,
            )
        except Exception as e:
            logger.error(f"ReAct LLM call failed at iteration {iteration}: {e}")
            result.final_text = f"[LLM 调用失败: {e}]"
            break

        raw = resp.get("raw", {})
        result.raw_response = resp

        # 检查是否有 tool calls
        tool_calls = resp.get("tool_calls") or []
        if not tool_calls and supports_fc:
            # 原生 FC 没返回 tool_calls → 模型给出了最终回复
            result.final_text = resp.get("final_text", "")
            result.reasoning_text = resp.get("reasoning_text", "")
            break

        if not tool_calls and not supports_fc:
            # prompt-based 降级模式：从文本中解析
            text = resp.get("final_text", "")
            tool_calls = parse_tool_calls(raw, is_anthropic)
            if not tool_calls:
                # 无 tool call → 最终回复
                result.final_text = text
                result.reasoning_text = resp.get("reasoning_text", "")
                break

        # 执行 tool calls
        tool_results: List[ToolResult] = []
        for tc in tool_calls:
            logger.info(f"ReAct [{iteration}] calling tool: {tc.name}({tc.arguments})")
            tr = execute_tool_call(tc, tools)
            tool_results.append(tr)

            # 记录到 trace
            result.tool_trace.append({
                "iteration": iteration,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result": tr.content[:500],
                "is_error": tr.is_error,
            })

        # 将 assistant 的 tool call + tool results 追加到消息历史
        if is_anthropic:
            # Anthropic: assistant message 的 content 包含 text + tool_use blocks
            assistant_content = []
            # 保留 text 部分
            final_text = resp.get("final_text", "")
            if final_text:
                assistant_content.append({"type": "text", "text": final_text})
            # 添加 tool_use blocks
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            working_messages.append({"role": "assistant", "content": assistant_content})

            # tool results 作为 user message
            tool_result_content = [tool_result_to_anthropic_content(tr) for tr in tool_results]
            working_messages.append({"role": "user", "content": tool_result_content})
        else:
            # OpenAI: assistant message 包含 tool_calls 字段
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("final_text") or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
            working_messages.append(assistant_msg)

            # 每个 tool result 作为独立的 tool message
            for tr in tool_results:
                working_messages.append(tool_result_to_openai_message(tr))
    else:
        # 达到最大迭代次数
        logger.warning(f"ReAct loop reached max iterations ({max_iterations})")
        result.final_text = resp.get("final_text", "") if 'resp' in dir() else ""

    result.total_time_ms = (time.perf_counter() - t0) * 1000
    return result
