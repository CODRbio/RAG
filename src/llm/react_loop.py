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
from src.utils.token_counter import count_tokens, get_context_window, needs_sliding_window

logger = get_logger(__name__)

_REACT_SYSTEM_HARD_MAX_CHARS = 50_000


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return _json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += 6
        total += count_tokens(_message_content_to_text(m.get("content")))
    return total


def _apply_iteration_prompt_budget(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    min_output_tokens: int,
    safety_margin: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pruned = list(messages)
    context_window = get_context_window(model)
    before = _estimate_prompt_tokens(pruned)
    trimmed_messages = 0
    trimmed_tool_like = 0
    used_system_cap = False

    while len(pruned) > 2 and needs_sliding_window(
        _estimate_prompt_tokens(pruned),
        context_window,
        safety_margin=safety_margin,
        min_output_tokens=min_output_tokens,
    ):
        removed = pruned.pop(1)
        trimmed_messages += 1
        if removed.get("role") in ("tool", "assistant"):
            trimmed_tool_like += 1

    if pruned and pruned[0].get("role") == "system":
        sys_text = _message_content_to_text(pruned[0].get("content"))
        if len(sys_text) > _REACT_SYSTEM_HARD_MAX_CHARS:
            pruned[0] = {
                **pruned[0],
                "content": sys_text[:_REACT_SYSTEM_HARD_MAX_CHARS] + "\n\n[budget-trimmed]",
            }
            used_system_cap = True

    while len(pruned) > 2 and needs_sliding_window(
        _estimate_prompt_tokens(pruned),
        context_window,
        safety_margin=safety_margin,
        min_output_tokens=min_output_tokens,
    ):
        removed = pruned.pop(1)
        trimmed_messages += 1
        if removed.get("role") in ("tool", "assistant"):
            trimmed_tool_like += 1

    after = _estimate_prompt_tokens(pruned)
    return pruned, {
        "prompt_tokens_before": before,
        "prompt_tokens_after": after,
        "message_count_before": len(messages),
        "message_count_after": len(pruned),
        "trimmed_messages": trimmed_messages,
        "trimmed_tool_like": trimmed_tool_like,
        "used_system_hard_cap": used_system_cap,
        "context_window": context_window,
        "min_output_tokens": min_output_tokens,
        "safety_margin": safety_margin,
    }


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
    prompt_budget_min_output_tokens: int = 4096,
    prompt_budget_safety_margin: float = 0.10,
    **llm_kwargs,
) -> ReactResult:
    """
    ReAct 循环核心。

    Args:
        messages: 初始消息列表（含 system + user）
        tools: 可用工具列表
        llm_client: BaseChatClient 实例
        max_iterations: 最大轮数；最后一轮（iteration == max_iterations - 1）固定为「仅总结」轮，不传 tools，以保证返回非空回答。
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
    provider_supports_platform_tools = bool(getattr(llm_client, "supports_platform_tool_calls", True))
    if tools and not provider_supports_platform_tools:
        provider_name = getattr(getattr(llm_client, "config", None), "name", type(llm_client).__name__)
        raise ValueError(
            f"Provider '{provider_name}' does not support the platform ReAct tool loop."
        )

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

    last_non_empty_final_text = ""
    resp: Optional[Dict[str, Any]] = None
    for iteration in range(max_iterations):
        result.iterations = iteration + 1

        working_messages, _budget_diag = _apply_iteration_prompt_budget(
            working_messages,
            model=model,
            min_output_tokens=max(512, int(prompt_budget_min_output_tokens or 4096)),
            safety_margin=float(prompt_budget_safety_margin or 0.10),
        )
        if _budget_diag.get("trimmed_messages", 0) or _budget_diag.get("used_system_hard_cap", False):
            logger.info(
                "ReAct [%d] prompt budget applied | tokens=%s→%s | messages=%d→%d | trimmed=%d | trimmed_tool_like=%d | system_hard_cap=%s",
                iteration,
                _budget_diag.get("prompt_tokens_before"),
                _budget_diag.get("prompt_tokens_after"),
                _budget_diag.get("message_count_before"),
                _budget_diag.get("message_count_after"),
                _budget_diag.get("trimmed_messages"),
                _budget_diag.get("trimmed_tool_like"),
                _budget_diag.get("used_system_hard_cap"),
            )

        # ── LLM 调用（计时）──
        # 最后一轮不传 tools，强制模型只输出总结，避免到上限时 response_len=0
        is_last_round = iteration == max_iterations - 1
        tools_for_this_round = None if is_last_round else (tools if supports_fc else None)
        t_llm = time.perf_counter()
        try:
            resp = llm_client.chat(
                messages=working_messages,
                model=model,
                tools=tools_for_this_round,
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
            # Log API error body when available (e.g. OpenAI 400 with error.message)
            api_error_detail = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = e.response.text
                    if body:
                        api_error_detail = body[:1000] if len(body) > 1000 else body
                        try:
                            data = _json.loads(body)
                            err = data.get("error") or data
                            if isinstance(err, dict):
                                api_error_detail = err.get("message", api_error_detail)
                                if err.get("code"):
                                    api_error_detail = f"[{err['code']}] {api_error_detail}"
                        except Exception:
                            pass
                except Exception:
                    pass
            logger.error(
                "ReAct LLM call failed at iteration %d: %s | message_preview=%s%s",
                iteration,
                e,
                msg_preview,
                f" | api_error={api_error_detail}" if api_error_detail else "",
            )
            result.final_text = f"[LLM 调用失败: {e}]"
            break
        llm_elapsed_ms = round((time.perf_counter() - t_llm) * 1000)

        # Codex app-server: same logical thread must be passed on subsequent iterations
        # (each llm_client.chat() starts a short-lived subprocess).
        if isinstance(resp, dict):
            _codex_tid = (resp.get("meta") or {}).get("codex_thread_id")
            if _codex_tid:
                llm_kwargs["codex_thread_id"] = str(_codex_tid)

        raw = resp.get("raw", {})
        result.raw_response = resp
        _ft = (resp.get("final_text") or "").strip()
        if _ft:
            last_non_empty_final_text = _ft

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
                prompt_budget=_budget_diag,
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
        result.final_text = (resp.get("final_text") or "").strip() if resp else ""
        if not result.final_text and last_non_empty_final_text:
            result.final_text = last_non_empty_final_text
            logger.info("ReAct: using last non-empty final_text from earlier iteration (len=%d)", len(result.final_text))

    result.total_time_ms = (time.perf_counter() - t0) * 1000
    result.agent_stats = _build_agent_stats(result)
    return result
