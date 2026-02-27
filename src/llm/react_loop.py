"""
ReAct 执行循环：LLM → tool_call → execute → feed back → repeat。

支持 OpenAI / Anthropic 两种 tool calling 协议，自动降级为 prompt-based。
"""

from __future__ import annotations

import json as _json
import time
from collections import Counter
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
from src.observability.tracing import traceable

logger = get_logger(__name__)


@dataclass
class ReactResult:
    """ReAct 循环的最终结果"""
    final_text: str = ""
    reasoning_text: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    total_time_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    agent_stats: Dict[str, Any] = field(default_factory=dict)


def _build_agent_stats(result: ReactResult) -> Dict[str, Any]:
    """从完成的 ReactResult 中计算 agent_stats 摘要。"""
    tool_counts: Counter = Counter()
    total_tool_ms = 0.0
    total_llm_ms = 0.0
    error_count = 0
    for entry in result.tool_trace:
        tool_counts[entry["tool"]] += 1
        total_tool_ms += entry.get("tool_latency_ms", 0)
        total_llm_ms += entry.get("llm_latency_ms", 0)
        if entry.get("is_error"):
            error_count += 1
    return {
        "total_iterations": result.iterations,
        "total_tool_calls": len(result.tool_trace),
        "tools_used_summary": dict(tool_counts),
        "total_tool_time_ms": round(total_tool_ms),
        "total_llm_time_ms": round(total_llm_ms),
        "total_agent_time_ms": round(result.total_time_ms),
        "error_count": error_count,
    }


@traceable(run_type="agent", name="react_loop")
def react_loop(
    messages: List[Dict[str, Any]],
    tools: List[ToolDef],
    llm_client: Any,
    max_iterations: int = 10,
    model: Optional[str] = None,
    session_id: str = "",
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
        session_id: 会话 ID（用于 debug 日志追溯）
        **llm_kwargs: 传递给 llm_client.chat() 的额外参数

    Returns:
        ReactResult
    """
    from src.debug import get_debug_logger
    dbg = get_debug_logger()

    t0 = time.perf_counter()
    result = ReactResult()

    is_anthropic = getattr(getattr(llm_client, "config", None), "is_anthropic", lambda: False)()

    supports_fc = True
    working_messages = list(messages)

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

        # ── LLM 调用（计时）──
        t_llm = time.perf_counter()
        try:
            resp = llm_client.chat(
                messages=working_messages,
                model=model,
                tools=tools if supports_fc else None,
                **llm_kwargs,
            )
        except Exception as e:
            msg_preview = []
            for m in working_messages[-8:]:
                content = m.get("content", "")
                msg_preview.append(
                    {
                        "role": m.get("role"),
                        "has_tool_calls": bool(m.get("tool_calls")),
                        "tool_call_id": m.get("tool_call_id"),
                        "content_type": type(content).__name__,
                        "content_len": len(content) if isinstance(content, str) else len(str(content)),
                    }
                )
            logger.error(
                "ReAct LLM call failed at iteration %d: %s | message_preview=%s",
                iteration,
                e,
                msg_preview,
            )
            result.final_text = f"[LLM 调用失败: {e}]"
            break
        llm_elapsed_ms = round((time.perf_counter() - t_llm) * 1000)

        raw = resp.get("raw", {})
        result.raw_response = resp

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls and supports_fc:
            result.final_text = resp.get("final_text", "")
            result.reasoning_text = resp.get("reasoning_text", "")

            dbg.log_agent_iteration(
                session_id,
                iteration=iteration,
                event="final_response",
                llm_latency_ms=llm_elapsed_ms,
                final_text_len=len(result.final_text),
            )
            break

        if not tool_calls and not supports_fc:
            text = resp.get("final_text", "")
            tool_calls = parse_tool_calls(raw, is_anthropic)
            if not tool_calls:
                result.final_text = text
                result.reasoning_text = resp.get("reasoning_text", "")
                break

        # ── 执行 tool calls（逐个计时）──
        tool_results: List[ToolResult] = []
        for i, tc in enumerate(tool_calls):
            if not tc.id:
                tc.id = f"call_{iteration}_{i}"
                logger.warning(
                    "ReAct [%d] tool_call missing id, auto-filled with %s for %s",
                    iteration,
                    tc.id,
                    tc.name,
                )
            logger.info(f"ReAct [{iteration}] calling tool: {tc.name}({tc.arguments})")

            t_tool = time.perf_counter()
            tr = execute_tool_call(tc, tools)
            tool_elapsed_ms = round((time.perf_counter() - t_tool) * 1000)

            tool_results.append(tr)

            result.tool_trace.append({
                "iteration": iteration,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result": tr.content[:500],
                "is_error": tr.is_error,
                "tool_latency_ms": tool_elapsed_ms,
                "llm_latency_ms": llm_elapsed_ms,
            })

            dbg.log_agent_iteration(
                session_id,
                iteration=iteration,
                event="tool_call",
                tool_name=tc.name,
                tool_arguments=tc.arguments,
                tool_result=tr.content[:2000],
                tool_is_error=tr.is_error,
                tool_latency_ms=tool_elapsed_ms,
                llm_latency_ms=llm_elapsed_ms,
                llm_input_messages=[
                    {"role": m.get("role"), "content_len": len(str(m.get("content", "")))}
                    for m in working_messages
                ],
                llm_raw_final_text=resp.get("final_text", "")[:1000],
            )

        # ── 将 tool call + results 追加到消息历史 ──
        if is_anthropic:
            assistant_content = []
            final_text = resp.get("final_text", "")
            if final_text:
                assistant_content.append({"type": "text", "text": final_text})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            working_messages.append({"role": "assistant", "content": assistant_content})
            tool_result_content = [tool_result_to_anthropic_content(tr) for tr in tool_results]
            working_messages.append({"role": "user", "content": tool_result_content})
        else:
            # Use "" instead of None: many OpenAI-compatible providers (Moonshot/Kimi,
            # Qwen, etc.) reject content:null even when tool_calls are present.
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("final_text") or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
            working_messages.append(assistant_msg)
            for tr in tool_results:
                working_messages.append(tool_result_to_openai_message(tr))
    else:
        logger.warning(f"ReAct loop reached max iterations ({max_iterations})")
        result.final_text = resp.get("final_text", "") if 'resp' in dir() else ""

    result.total_time_ms = (time.perf_counter() - t0) * 1000
    result.agent_stats = _build_agent_stats(result)
    return result
