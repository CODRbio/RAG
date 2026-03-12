"""
统一 Tool 抽象层。

一次定义 Tool Schema，自动适配 OpenAI / Anthropic function calling 格式，
并提供 prompt-based 降级方案。

Usage:
    from src.llm.tools import CORE_TOOLS, ToolDef, to_openai_tools, to_anthropic_tools, execute_tool_call
    from src.llm.tools import get_routed_skills, get_tools_by_names
"""

from __future__ import annotations

import ast
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import threading

try:
    import resource
except Exception:  # pragma: no cover - non-POSIX fallback
    resource = None  # type: ignore[assignment]

from src.log import get_logger

logger = get_logger(__name__)

# Thread-local collector for EvidenceChunks produced by Agent tool calls.
# routes_chat.py activates/reads this around the react_loop invocation.
_agent_chunks_local = threading.local()


def set_tool_collection(collection: Optional[str], collections: Optional[List[str]] = None) -> None:
    """Set the Milvus collection(s) for search_local/search_web in this thread.
    
    If ``collections`` is provided with more than one entry, a MultiCollectionRetrievalService
    will be used with equal quota split. ``collection`` is kept as the primary fallback.
    """
    _agent_chunks_local.collection = collection or None
    # Store multi-collection list (None or a list of >=2 names)
    _agent_chunks_local.collections = (
        [c.strip() for c in collections if (c or "").strip()]
        if collections and len(collections) > 1
        else None
    )


def set_tool_step_top_k(step_top_k: Optional[int]) -> None:
    """Set request-scoped step_top_k for agent retrieval tools."""
    try:
        v = int(step_top_k) if step_top_k is not None else None
    except Exception:
        v = None
    _agent_chunks_local.step_top_k = v if (v is not None and v > 0) else None


def get_tool_step_top_k() -> Optional[int]:
    """Return request-scoped step_top_k if available."""
    return getattr(_agent_chunks_local, "step_top_k", None)


def set_agent_sonar_model(model: Optional[str]) -> None:
    """Set the Sonar model for search_sonar in this thread (e.g. sonar | sonar-pro). Used when Sonar is enabled as a web retrieval tool."""
    _agent_chunks_local.agent_sonar_model = (model or "sonar-pro").strip() or "sonar-pro"


def set_tool_retrieval_params(params: Optional[Dict[str, Any]]) -> None:
    """Set retrieval filters (providers, configs, year window) for tool calls in this thread."""
    _agent_chunks_local.retrieval_params = params or {}


def get_tool_retrieval_params() -> Dict[str, Any]:
    """Get the thread-local retrieval params."""
    return getattr(_agent_chunks_local, "retrieval_params", {}) or {}


def get_agent_sonar_model() -> str:
    """Return the thread-local Sonar model for search_sonar, default sonar-pro."""
    return getattr(_agent_chunks_local, "agent_sonar_model", None) or "sonar-pro"


def start_agent_chunk_collector() -> None:
    """Activate the per-thread chunk collector (call before react_loop)."""
    _agent_chunks_local.chunks = []


def drain_agent_chunks() -> list:
    """Return and clear all collected chunks (call after react_loop)."""
    chunks = getattr(_agent_chunks_local, "chunks", None) or []
    _agent_chunks_local.chunks = []
    return chunks


def _collect_chunks(chunks: list) -> None:
    """Append EvidenceChunks to the thread-local collector if active."""
    store = getattr(_agent_chunks_local, "chunks", None)
    if store is not None:
        store.extend(chunks)


_RUN_CODE_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
_RUN_CODE_BLOCKED_NAMES = {
    "aiohttp",
    "asyncio",
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
}
_RUN_CODE_ALLOWED_BUILTINS = [
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "ArithmeticError",
    "AssertionError",
    "IndexError",
    "KeyError",
    "ZeroDivisionError",
]


def _get_tool_execution_settings():
    from config.settings import settings
    return settings.tool_execution


def _run_code_enabled() -> bool:
    try:
        return bool(_get_tool_execution_settings().run_code_enabled)
    except Exception:
        return False


def _tool_is_enabled(name: str) -> bool:
    if name == "run_code":
        return _run_code_enabled()
    return True


# run_code 并发槽位 Semaphore。
# 实际 max_concurrent 从配置读取，但 Semaphore 需要在模块加载时初始化。
# _handle_run_code 会在获取锁之前动态检查当前配置值，
# 此处初始值仅作为默认兜底，不影响运行时行为。
_RUN_CODE_SEMAPHORE: threading.Semaphore = threading.Semaphore(2)
_RUN_CODE_SEMAPHORE_SIZE: int = 2  # 跟踪当前 Semaphore 容量，用于懒更新


def _get_run_code_semaphore() -> threading.Semaphore:
    """返回与当前配置 max_concurrent 对齐的 Semaphore（懒更新）。"""
    global _RUN_CODE_SEMAPHORE, _RUN_CODE_SEMAPHORE_SIZE
    try:
        desired = max(1, _get_tool_execution_settings().max_concurrent)
    except Exception:
        desired = 2
    if desired != _RUN_CODE_SEMAPHORE_SIZE:
        _RUN_CODE_SEMAPHORE = threading.Semaphore(desired)
        _RUN_CODE_SEMAPHORE_SIZE = desired
    return _RUN_CODE_SEMAPHORE


class _RunCodeValidationError(ValueError):
    pass


class _RunCodeSafetyValidator(ast.NodeVisitor):
    def __init__(self, allowed_modules: set[str]):
        self.allowed_modules = allowed_modules

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = (alias.name or "").split(".", 1)[0]
            if root not in self.allowed_modules:
                raise _RunCodeValidationError(f"禁止导入模块: {root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise _RunCodeValidationError("禁止相对导入")
        root = ((node.module or "").split(".", 1)[0]).strip()
        if root not in self.allowed_modules:
            raise _RunCodeValidationError(f"禁止导入模块: {root or '(empty)'}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if str(node.attr or "").startswith("__"):
            raise _RunCodeValidationError("禁止访问双下划线属性")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _RUN_CODE_BLOCKED_NAMES:
            raise _RunCodeValidationError(f"禁止访问名称: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _RUN_CODE_BLOCKED_CALLS:
            raise _RunCodeValidationError(f"禁止调用函数: {fn.id}")
        if isinstance(fn, ast.Attribute) and fn.attr in _RUN_CODE_BLOCKED_CALLS:
            raise _RunCodeValidationError(f"禁止调用函数: {fn.attr}")
        self.generic_visit(node)


def _validate_run_code(code: str, *, max_code_chars: int, allowed_modules: List[str]) -> None:
    source = (code or "").strip()
    if not source:
        raise _RunCodeValidationError("代码为空")
    if len(source) > max_code_chars:
        raise _RunCodeValidationError(f"代码长度超过上限（{max_code_chars} 字符）")
    try:
        tree = ast.parse(source, filename="<agent-run_code>", mode="exec")
    except SyntaxError as exc:
        raise _RunCodeValidationError(f"代码语法错误: {exc.msg} (line {exc.lineno})") from exc
    _RunCodeSafetyValidator({m.strip() for m in allowed_modules if m.strip()}).visit(tree)


def _build_run_code_wrapper(code: str, allowed_modules: List[str]) -> str:
    payload = json.dumps(code, ensure_ascii=False)
    allowed_json = json.dumps(sorted({m.strip() for m in allowed_modules if m.strip()}), ensure_ascii=False)
    safe_builtins_json = json.dumps(_RUN_CODE_ALLOWED_BUILTINS, ensure_ascii=False)
    return f"""
import builtins as _b

_ALLOWED_MODULES = set({allowed_json})
_SAFE_BUILTIN_NAMES = {safe_builtins_json}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = (name or "").split(".", 1)[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"Import '{{root}}' is not allowed in run_code.")
    return _b.__import__(name, globals, locals, fromlist, level)

_SAFE_BUILTINS = {{
    name: getattr(_b, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(_b, name)
}}
_SAFE_BUILTINS["__import__"] = _safe_import

_USER_GLOBALS = {{
    "__builtins__": _SAFE_BUILTINS,
    "__name__": "__main__",
}}
_CODE = {payload}
exec(compile(_CODE, "<agent-run_code>", "exec"), _USER_GLOBALS, _USER_GLOBALS)
"""


def _run_code_preexec(max_memory_mb: int, cpu_seconds: int) -> Callable[[], None] | None:
    if os.name != "posix" or resource is None:
        return None

    import platform as _platform
    if _platform.system() == "Darwin":
        # macOS 上 RLIMIT_AS / RLIMIT_DATA 不被内核强制执行，
        # 内存限制实际无效。生产环境建议部署在 Linux 上。
        logger.warning(
            "run_code: macOS 下 RLIMIT_AS/RLIMIT_DATA 无法强制内存上限，"
            "max_memory_mb=%d 配置在此平台不生效。建议生产环境使用 Linux。",
            max_memory_mb,
        )

    memory_bytes = max_memory_mb * 1024 * 1024

    def _apply_limits() -> None:
        # start_new_session=True 已在 fork 后调用 os.setsid()，此处无需重复
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except Exception:
            pass
        for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (memory_bytes, memory_bytes))
            except Exception:
                pass
        for limit_name, soft_value in (("RLIMIT_NOFILE", 32), ("RLIMIT_CORE", 0)):
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (soft_value, soft_value))
            except Exception:
                pass
        # RLIMIT_NPROC 是 per-UID 的全局配额，只降软限制，保留系统硬限制，
        # 避免影响同 UID 下其他服务的 fork 能力。
        nproc_limit = getattr(resource, "RLIMIT_NPROC", None)
        if nproc_limit is not None:
            try:
                _, hard = resource.getrlimit(nproc_limit)
                resource.setrlimit(nproc_limit, (min(8, hard), hard))
            except Exception:
                pass

    return _apply_limits


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _collect_process_output(
    proc: subprocess.Popen[bytes],
    *,
    timeout_seconds: int,
    max_output_chars: int,
) -> tuple[str, str, Optional[str]]:
    if proc.stdout is None or proc.stderr is None:
        out, err = proc.communicate(timeout=timeout_seconds)
        return (out or b"").decode("utf-8", errors="replace"), (err or b"").decode("utf-8", errors="replace"), None

    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    streams = {
        proc.stdout.fileno(): ("stdout", proc.stdout),
        proc.stderr.fileno(): ("stderr", proc.stderr),
    }
    out_chunks: List[bytes] = []
    err_chunks: List[bytes] = []
    collected = 0
    limit_bytes = max_output_chars * 4
    failure_reason: Optional[str] = None

    for fd, (_name, stream) in streams.items():
        try:
            os.set_blocking(fd, False)
        except Exception:
            pass
        selector.register(stream, selectors.EVENT_READ, data=fd)

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = "timeout"
                _terminate_process(proc)
                break
            events = selector.select(timeout=min(remaining, 0.1))
            if not events and proc.poll() is not None:
                break
            for key, _mask in events:
                fd = int(key.data)
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                collected += len(chunk)
                if collected > limit_bytes:
                    failure_reason = "output_limit"
                    _terminate_process(proc)
                    break
                if fd == proc.stdout.fileno():
                    out_chunks.append(chunk)
                else:
                    err_chunks.append(chunk)
            if failure_reason:
                break
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # 极少数情况下 SIGKILL 仍未响应，放弃等待
    finally:
        try:
            selector.close()
        except Exception:
            pass

    stdout = b"".join(out_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(err_chunks).decode("utf-8", errors="replace")
    return stdout, stderr, failure_reason


# ────────────────────────────────────────────────
# Tool 定义
# ────────────────────────────────────────────────

@dataclass
class ToolDef:
    """Provider 无关的 Tool 定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (type: object)
    handler: Optional[Callable[..., Any]] = None  # 实际执行函数

    # ── 转换为 OpenAI function calling 格式 ──
    def to_openai(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ── 转换为 Anthropic tool 格式 ──
    def to_anthropic(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """统一的 tool call 解析结果（provider 无关）"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool 执行结果"""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


# ────────────────────────────────────────────────
# 格式转换工具
# ────────────────────────────────────────────────

def to_openai_tools(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    return [t.to_openai() for t in tools]


def to_anthropic_tools(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    return [t.to_anthropic() for t in tools]


def tools_to_prompt(tools: List[ToolDef]) -> str:
    """降级：将 tools 描述注入 system prompt（用于不支持 FC 的模型）"""
    lines = ["你可以调用以下工具来完成任务。调用格式：",
             '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>',
             "", "可用工具："]
    for t in tools:
        params_desc = []
        props = t.parameters.get("properties", {})
        for pname, pinfo in props.items():
            req = "必填" if pname in t.parameters.get("required", []) else "可选"
            params_desc.append(f"    - {pname} ({pinfo.get('type', 'any')}, {req}): {pinfo.get('description', '')}")
        lines.append(f"\n**{t.name}**: {t.description}")
        if params_desc:
            lines.append("  参数:")
            lines.extend(params_desc)
    return "\n".join(lines)


# ────────────────────────────────────────────────
# 响应解析：从 raw LLM response 中提取 tool calls
# ────────────────────────────────────────────────

def parse_tool_calls_openai(raw: Dict[str, Any]) -> List[ToolCall]:
    """从 OpenAI-compatible 响应中解析 tool_calls"""
    calls = []
    if (raw.get("_api_mode") or "") == "responses" or raw.get("output"):
        for item in raw.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            args_str = item.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {"_raw": args_str}
            calls.append(ToolCall(
                id=item.get("call_id", "") or item.get("id", ""),
                name=item.get("name", ""),
                arguments=args,
            ))
        return calls
    choices = raw.get("choices") or []
    if not choices:
        return calls
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        calls.append(ToolCall(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            arguments=args,
        ))
    return calls


def parse_tool_calls_anthropic(raw: Dict[str, Any]) -> List[ToolCall]:
    """从 Anthropic 响应中解析 tool_use content blocks"""
    calls = []
    content = raw.get("content") or []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=block.get("input") or {},
            ))
    return calls


def parse_tool_calls_prompt(text: str) -> List[ToolCall]:
    """从纯文本响应中解析 <tool_call> 标签（降级模式）"""
    calls = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for i, m in enumerate(re.finditer(pattern, text, re.DOTALL)):
        try:
            data = json.loads(m.group(1))
            calls.append(ToolCall(
                id=f"prompt_tc_{i}",
                name=data.get("name", ""),
                arguments=data.get("arguments") or data.get("args") or {},
            ))
        except json.JSONDecodeError:
            continue
    return calls


def parse_tool_calls(raw: Dict[str, Any], is_anthropic: bool = False) -> List[ToolCall]:
    """统一解析入口"""
    if is_anthropic:
        calls = parse_tool_calls_anthropic(raw)
    else:
        calls = parse_tool_calls_openai(raw)
    # 如果原生 FC 没有返回 tool_calls，尝试从文本中降级解析
    if not calls:
        text = ""
        if is_anthropic:
            for block in (raw.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        else:
            choices = raw.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                c = msg.get("content")
                if isinstance(c, str):
                    text = c
        if "<tool_call>" in text:
            calls = parse_tool_calls_prompt(text)
    return calls


def has_tool_calls(raw: Dict[str, Any], is_anthropic: bool = False) -> bool:
    """快速判断响应中是否包含 tool calls"""
    if is_anthropic:
        stop = raw.get("stop_reason", "")
        if stop == "tool_use":
            return True
        for block in (raw.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True
    else:
        if (raw.get("_api_mode") or "") == "responses" or raw.get("output"):
            for item in raw.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    return True
        else:
            choices = raw.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                if msg.get("tool_calls"):
                    return True
                finish = choices[0].get("finish_reason", "")
                if finish in ("tool_calls", "function_call"):
                    return True
    return False


# ────────────────────────────────────────────────
# Tool 执行
# ────────────────────────────────────────────────

def execute_tool_call(
    tool_call: ToolCall,
    tools: List[ToolDef],
) -> ToolResult:
    """查找并执行 tool call，返回结果"""
    tool_map = {t.name: t for t in tools}
    tool = tool_map.get(tool_call.name)
    if tool is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error: unknown tool '{tool_call.name}'",
            is_error=True,
        )
    if tool.handler is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error: tool '{tool_call.name}' has no handler",
            is_error=True,
        )
    try:
        result = tool.handler(**tool_call.arguments)
        # 将结果序列化为字符串
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict) or isinstance(result, list):
            content = json.dumps(result, ensure_ascii=False, default=str)
        else:
            content = str(result)
        # 截断过长结果
        if len(content) > 8000:
            content = content[:7500] + "\n... (truncated)"
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
        )
    except Exception as e:
        logger.warning(f"Tool execution error [{tool_call.name}]: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"Error executing {tool_call.name}: {e}",
            is_error=True,
        )


# ────────────────────────────────────────────────
# 构建 tool result 消息（喂回 LLM）
# ────────────────────────────────────────────────

def tool_result_to_openai_message(result: ToolResult) -> Dict[str, Any]:
    """将 ToolResult 转为 OpenAI tool message"""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": result.content,
    }


def tool_result_to_anthropic_content(result: ToolResult) -> Dict[str, Any]:
    """将 ToolResult 转为 Anthropic tool_result content block"""
    return {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content,
        "is_error": result.is_error,
    }


# ────────────────────────────────────────────────
# 8 个核心 Tool 定义（handler 在 register 时绑定）
# ────────────────────────────────────────────────

_SEARCH_LOCAL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索查询文本"},
        "top_k": {"type": "integer", "description": "返回结果数量", "default": 10},
    },
    "required": ["query"],
}

_SEARCH_WEB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "网络搜索查询"},
        "top_k": {"type": "integer", "description": "返回结果数量", "default": 10},
    },
    "required": ["query"],
}

_SEARCH_SONAR_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "需要深度联网推理的查询问题"},
    },
    "required": ["query"],
}

_SEARCH_SCHOLAR_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "学术论文搜索查询"},
        "year_from": {"type": "integer", "description": "起始年份（可选）"},
        "limit": {"type": "integer", "description": "结果数量", "default": 5},
    },
    "required": ["query"],
}

_EXPLORE_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_name": {"type": "string", "description": "实体名称"},
        "depth": {"type": "integer", "description": "扩展深度 (1-3)", "default": 1},
    },
    "required": ["entity_name"],
}

_CANVAS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["create", "update", "get"], "description": "操作类型"},
        "canvas_id": {"type": "string", "description": "画布 ID（update/get 时必填）"},
        "topic": {"type": "string", "description": "主题（create 时必填）"},
        "content": {"type": "string", "description": "要更新的 Markdown 内容"},
    },
    "required": ["action"],
}

_CITATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "canvas_id": {"type": "string", "description": "画布 ID"},
        "format": {"type": "string", "enum": ["text", "bibtex"], "description": "引文格式", "default": "text"},
    },
    "required": ["canvas_id"],
}

_COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要比较的论文 ID 列表 (2-5)",
            "minItems": 2,
            "maxItems": 5,
        },
        "aspects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "比较维度",
        },
    },
    "required": ["paper_ids"],
}

_RUN_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "要执行的 Python 代码（简单计算/数据处理）"},
    },
    "required": ["code"],
}

_SEARCH_NCBI_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "PubMed 生物医学搜索查询（英文关键词效果最佳）",
        },
        "limit": {
            "type": "integer",
            "description": "返回结果数量",
            "default": 5,
        },
    },
    "required": ["query"],
}


# ── Handler 实现 ──

def _get_tool_retrieval_svc():
    """Return the appropriate retrieval service for the current thread (single or multi-collection)."""
    from src.retrieval.service import get_retrieval_service, get_multi_collection_retrieval_service

    multi_cols = getattr(_agent_chunks_local, "collections", None)
    if multi_cols and len(multi_cols) > 1:
        equal = 1.0 / len(multi_cols)
        return get_multi_collection_retrieval_service(
            collection_quotas={c: equal for c in multi_cols}
        )
    col = getattr(_agent_chunks_local, "collection", None)
    return get_retrieval_service(collection=col)


def _handle_search_local(query: str, top_k: int = 10, **_) -> str:
    requested_top_k = max(1, int(top_k or 10))
    step_top_k = get_tool_step_top_k()
    effective_top_k = min(requested_top_k, step_top_k) if step_top_k else requested_top_k
    svc = _get_tool_retrieval_svc()
    # Agent tool retrieval should return a source-ordered candidate pool; the
    # authority rerank happens later in chat final fusion.
    tool_filters: Dict[str, Any] = {"reranker_mode": "bge_only", "pool_only": True}
    if step_top_k:
        tool_filters["step_top_k"] = step_top_k
    pack = svc.search(query=query, mode="local", top_k=effective_top_k, filters=tool_filters)
    _collect_chunks(pack.chunks)
    return pack.to_context_string(max_chunks=min(effective_top_k, 15))


def _handle_search_web(query: str, top_k: int = 10, **_) -> str:
    requested_top_k = max(1, int(top_k or 10))
    step_top_k = get_tool_step_top_k()
    effective_top_k = min(requested_top_k, step_top_k) if step_top_k else requested_top_k
    svc = _get_tool_retrieval_svc()
    tool_filters: Dict[str, Any] = {"reranker_mode": "bge_only", "pool_only": True}
    if step_top_k:
        tool_filters["step_top_k"] = step_top_k
    
    params = get_tool_retrieval_params()
    if params.get("web_providers"):
        tool_filters["web_providers"] = params["web_providers"]
    if params.get("web_source_configs"):
        tool_filters["web_source_configs"] = params["web_source_configs"]
    for k in ["year_start", "year_end"]:
        if params.get(k) is not None:
            tool_filters[k] = params[k]
    
    logger.info(f"Agent search_web query='{query}' providers={tool_filters.get('web_providers')} year={tool_filters.get('year_start')}-{tool_filters.get('year_end')}")
    pack = svc.search(query=query, mode="web", top_k=effective_top_k, filters=tool_filters)
    _collect_chunks(pack.chunks)
    return pack.to_context_string(max_chunks=min(effective_top_k, 15))


def invoke_sonar_search(query: str, model: str = "sonar-pro") -> tuple:
    """Call Sonar API and return (response_text, parsed_chunks).

    Reusable by both Agent tool handler and Deep Research pipeline.
    Returns: (str, list) — text is the model reply; list elements are EvidenceChunk-like dicts/chunks.
    """
    from src.llm.llm_manager import get_manager
    from src.retrieval.sonar_citations import parse_sonar_citations

    manager = get_manager()
    provider = "sonar" if manager.is_available("sonar") else ("perplexity" if manager.is_available("perplexity") else None)
    if not provider:
        return (
            "Sonar/Perplexity 未配置或不可用，请在 config/rag_config.local.json 的 llm.platforms.perplexity 中设置 api_key（如 pplx-xxx）后重试。",
            [],
        )
    client = manager.get_client(provider)
    prompt = (
        "Answer the following question concisely with key points and sources. "
        "Use the same language as the question. Keep under 400 words.\n\n"
    ) + (query or "").strip()
    resp = client.chat(
        [{"role": "user", "content": prompt}],
        model=model,
        timeout_seconds=50,
    )
    raw = resp.get("raw") or {}
    citations = raw.get("citations")
    search_results = raw.get("search_results")
    if not citations and resp.get("citations") is not None:
        citations = resp.get("citations")
    if not search_results and resp.get("search_results") is not None:
        search_results = resp.get("search_results")
    text = (resp.get("final_text") or "").strip()
    if "</think>" in text:
        idx = text.find("</think>")
        text = text[idx + 7 :].lstrip()
    chunks = parse_sonar_citations(
        citations=citations,
        search_results=search_results,
        response_text=text,
        query=query or "",
    )
    return (text if text else "Sonar 未返回有效内容。", chunks)


def _handle_search_sonar(query: str, **_) -> str:
    """使用 Perplexity Sonar 联网深度搜索，返回带引用的回答；引用纳入引文池。"""
    try:
        model = get_agent_sonar_model()
        text, chunks = invoke_sonar_search(query, model=model)
        _collect_chunks(chunks)
        return text
    except Exception as e:
        logger.debug("search_sonar failed: %s", e)
        return f"Sonar 搜索失败: {e}"


def _handle_search_scholar(query: str, year_from: Optional[int] = None, limit: int = 5, **_) -> str:
    try:
        import asyncio
        from src.retrieval.semantic_scholar import SemanticScholarSearcher
        from src.retrieval.evidence import EvidenceChunk
        
        params = get_tool_retrieval_params()
        effective_start = year_from if year_from is not None else params.get("year_start")
        effective_end = params.get("year_end")
        
        requested_limit = max(1, int(limit or 5))
        ss = SemanticScholarSearcher()

        async def _search_and_close() -> list:
            try:
                return await ss.search(query, limit=requested_limit, year_start=effective_start, year_end=effective_end)
            finally:
                await ss.close()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _search_and_close())
                    results = future.result(timeout=45)
            else:
                results = loop.run_until_complete(_search_and_close())
        except RuntimeError:
            results = asyncio.run(_search_and_close())
        if not results:
            return "未找到相关学术论文。"
        lines = []
        chunks = []
        for r in results[:requested_limit]:
            meta = r.get("metadata", {})
            title = meta.get("title", r.get("content", ""))
            year = meta.get("year", "")
            abstract = (r.get("content") or "")[:300]
            doi = meta.get("doi", "")
            paper_id = meta.get("paper_id", "")
            authors_raw = meta.get("authors") or []
            author_names = (
                [a.get("name", "") for a in authors_raw if isinstance(a, dict)]
                if authors_raw and isinstance(authors_raw[0], dict)
                else [str(a) for a in authors_raw]
            )
            url = meta.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else "")
            chunk = EvidenceChunk(
                chunk_id=f"scholar_{paper_id or title[:20]}",
                doc_id=doi or paper_id or title[:30],
                text=f"{title}. {abstract}",
                score=0.0,
                source_type="web",
                doc_title=title,
                authors=author_names[:5],
                year=int(year) if year else None,
                url=url,
                doi=doi or None,
                provider="semantic",
            )
            chunks.append(chunk)
            lines.append(f"[{chunk.ref_hash}] **{title}** ({year}) DOI:{doi}\n  {abstract}")
        _collect_chunks(chunks)
        return "\n".join(lines)
    except Exception as e:
        return f"学术搜索失败: {e}"


def _handle_explore_graph(entity_name: str, depth: int = 1, **_) -> str:
    try:
        from src.api.routes_graph import graph_neighbors
        result = graph_neighbors(entity_name, depth=depth)
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        lines = [f"实体 '{entity_name}' 的知识图谱（深度={depth}）:"]
        lines.append(f"节点数: {len(nodes)}, 边数: {len(edges)}")
        entity_nodes = [n for n in nodes if n.get("type") != "CHUNK"]
        for n in entity_nodes[:20]:
            marker = " [中心]" if n.get("is_center") else ""
            lines.append(f"  - {n['id']} ({n['type']}){marker}")
        for e in edges[:20]:
            lines.append(f"  {e['source']} --[{e['relation']}]--> {e['target']}")
        return "\n".join(lines)
    except Exception as e:
        return f"图谱查询失败: {e}"


def _handle_canvas(action: str, canvas_id: str = "", topic: str = "", content: str = "", **_) -> str:
    from src.collaboration.canvas.canvas_manager import create_canvas, get_canvas, update_canvas
    if action == "create":
        canvas = create_canvas(topic=topic or "Untitled")
        return json.dumps({"canvas_id": canvas.id, "topic": canvas.topic}, ensure_ascii=False)
    elif action == "get":
        canvas = get_canvas(canvas_id)
        if canvas is None:
            return f"画布 '{canvas_id}' 不存在"
        return json.dumps({"canvas_id": canvas.id, "topic": canvas.topic, "markdown": canvas.markdown[:3000]}, ensure_ascii=False)
    elif action == "update":
        update_canvas(canvas_id, markdown=content)
        return f"画布 '{canvas_id}' 已更新"
    return f"未知操作: {action}"


def _handle_citations(canvas_id: str, format: str = "text", **_) -> str:
    from src.collaboration.citation.formatter import format_reference_list
    from src.collaboration.canvas.canvas_manager import get_canvas_citations
    citations = get_canvas_citations(canvas_id)
    if not citations:
        return "该画布暂无引文。"
    # 映射前端参数到有效 style
    style_map = {
        "text": "custom",
        "bibtex": "apa",
        "apa": "apa",
        "ieee": "ieee",
        "numeric": "numeric",
        "custom": "custom",
    }
    style = style_map.get(format, "custom")
    return format_reference_list(citations, style=style)


def _handle_compare(paper_ids: List[str], aspects: Optional[List[str]] = None, **_) -> str:
    try:
        from src.api.routes_compare import compare_papers, CompareRequest
        req = CompareRequest(paper_ids=paper_ids, aspects=aspects or ["objective", "methodology", "key_findings", "limitations"])
        resp = compare_papers(req)
        parts = []
        if resp.narrative:
            parts.append(f"综合分析: {resp.narrative}")
        for aspect, cells in resp.comparison_matrix.items():
            parts.append(f"\n[{aspect}]")
            for pid, desc in cells.items():
                parts.append(f"  {pid}: {desc}")
        return "\n".join(parts) if parts else "对比结果为空"
    except Exception as e:
        return f"论文对比失败: {e}"


def _handle_search_ncbi(query: str, limit: int = 5, **_) -> str:
    """调用 NCBI PubMed E-Utilities，返回生物医学文献摘要信息。"""
    try:
        import asyncio
        from src.retrieval.ncbi_search import get_ncbi_searcher
        from src.retrieval.evidence import EvidenceChunk

        params = get_tool_retrieval_params()
        y_start = params.get("year_start")
        y_end = params.get("year_end")

        requested_limit = max(1, int(limit or 5))
        searcher = get_ncbi_searcher()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, searcher.search(query, limit=requested_limit, year_start=y_start, year_end=y_end))
                    results = future.result(timeout=30)
            else:
                results = loop.run_until_complete(searcher.search(query, limit=requested_limit, year_start=y_start, year_end=y_end))
        except RuntimeError:
            results = asyncio.run(searcher.search(query, limit=requested_limit, year_start=y_start, year_end=y_end))

        if not results:
            return "PubMed 未找到相关文献。"

        lines = []
        chunks = []
        for r in results[:requested_limit]:
            meta = r.get("metadata", {})
            title = meta.get("title", r.get("content", ""))
            year = meta.get("year", "")
            doi = meta.get("doi", "")
            authors = meta.get("authors", [])
            authors_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
            url = meta.get("url", "")
            pmid = meta.get("pmid", "")
            abstract = (r.get("content") or meta.get("abstract") or "")[:400]
            chunk = EvidenceChunk(
                chunk_id=f"ncbi_{pmid or doi or title[:20]}",
                doc_id=doi or pmid or title[:30],
                text=f"{title}. {abstract}",
                score=0.0,
                source_type="web",
                doc_title=title,
                authors=authors[:5],
                year=int(year) if year else None,
                url=url or None,
                doi=doi or None,
                provider="ncbi",
            )
            chunks.append(chunk)
            lines.append(
                f"[{chunk.ref_hash}] **{title}** ({year})\n"
                f"  Authors: {authors_str}\n"
                f"  DOI: {doi or '—'}  URL: {url}"
            )
        _collect_chunks(chunks)
        return "\n".join(lines)
    except Exception as e:
        return f"NCBI 搜索失败: {e}"


# ── summarize_quantitative handler ──

_SUMMARIZE_QUANT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_text": {
            "type": "string",
            "description": (
                "包含定量数据的证据文本（直接复制 context 中标记了 [Q] 的证据段落，"
                "保留 [ref:xxxx] 标记以便引用溯源）"
            ),
        },
        "variable": {
            "type": "string",
            "description": (
                "待比较的变量名称（如 'pH', 'sample size', 'Shannon diversity', "
                "'concentration', 'species richness' 等）"
            ),
        },
    },
    "required": ["evidence_text", "variable"],
}


def _handle_summarize_quantitative(evidence_text: str, variable: str, **_) -> str:
    """Parse numeric values from evidence text for the given variable, compute statistics."""
    import statistics as _stat

    # Find ref tags so we can attribute values
    ref_pattern = re.compile(r"\[ref:[a-f0-9]{6,12}\]")
    # Generic number extraction: integers, decimals, ranges with ± or -
    num_pattern = re.compile(
        r"(?P<val>\d+\.?\d*)"
        r"(?:\s*[±\+\-/]\s*(?P<err>\d+\.?\d*))?"
    )

    text = evidence_text or ""
    variable_lower = variable.strip().lower()
    if not text or not variable_lower:
        return "Error: evidence_text and variable are required."

    # Split into per-ref segments by looking for [ref:...] boundaries
    segments: list[tuple[str, str]] = []  # (ref_tag, text_after)
    parts = ref_pattern.split(text)
    refs_found = ref_pattern.findall(text)
    if refs_found:
        for i, ref in enumerate(refs_found):
            segment_text = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append((ref, segment_text))
    else:
        segments.append(("", text))

    # For each segment, look for lines mentioning the variable, extract numbers
    data_points: list[dict] = []
    for ref_tag, seg_text in segments:
        seg_lower = seg_text.lower()
        if variable_lower not in seg_lower:
            continue
        # Find sentences containing the variable
        sentences = re.split(r"[.。;；\n]", seg_text)
        for sent in sentences:
            if variable_lower not in sent.lower():
                continue
            for m in num_pattern.finditer(sent):
                try:
                    val = float(m.group("val"))
                    err = float(m.group("err")) if m.group("err") else None
                    context_snip = sent.strip()[:120]
                    data_points.append({
                        "ref": ref_tag,
                        "value": val,
                        "error": err,
                        "snippet": context_snip,
                    })
                except (ValueError, TypeError):
                    continue

    if not data_points:
        return f"No numeric values found for variable '{variable}' in the provided evidence."

    values = [dp["value"] for dp in data_points]
    n = len(values)
    v_min = min(values)
    v_max = max(values)
    v_mean = _stat.mean(values)
    v_median = _stat.median(values)
    v_stdev = _stat.stdev(values) if n >= 2 else 0.0

    lines = [
        f"=== Cross-Study Summary: {variable} ===",
        f"Data points: {n}",
        f"Range: {v_min} – {v_max}",
        f"Mean: {v_mean:.4g}",
        f"Median: {v_median:.4g}",
    ]
    if n >= 2:
        lines.append(f"Std Dev: {v_stdev:.4g}")
    lines.append("")
    lines.append("Per-source breakdown:")

    for dp in data_points:
        val_str = f"{dp['value']}"
        if dp["error"] is not None:
            val_str += f" ± {dp['error']}"
        ref_str = f" {dp['ref']}" if dp["ref"] else ""
        lines.append(f"  - {val_str}{ref_str}  |  \"{dp['snippet']}\"")

    return "\n".join(lines)


def _handle_run_code(code: str, **_) -> str:
    """以受限子进程方式执行 Python 代码。默认关闭，仅适用于受信任环境。"""
    cfg = _get_tool_execution_settings()
    if not _run_code_enabled():
        return (
            "run_code 已禁用。请改用 summarize_quantitative，"
            "或在受信任环境中显式开启 tool_execution.run_code_enabled。"
        )

    try:
        _validate_run_code(
            code,
            max_code_chars=cfg.max_code_chars,
            allowed_modules=list(cfg.allowed_modules),
        )
    except _RunCodeValidationError as exc:
        return f"代码执行被拒绝: {exc}"

    sem = _get_run_code_semaphore()
    if not sem.acquire(blocking=False):
        return (
            f"run_code 并发上限已满（max_concurrent={cfg.max_concurrent}），请稍后重试。"
        )
    try:
        with tempfile.TemporaryDirectory(prefix="agent_run_code_") as tmp_dir:
            script_path = os.path.join(tmp_dir, "runner.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(_build_run_code_wrapper(code, list(cfg.allowed_modules)))

            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-B", script_path],
                cwd=tmp_dir,
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=_run_code_preexec(cfg.max_memory_mb, cfg.cpu_seconds),
                start_new_session=True,
            )
            stdout, stderr, failure_reason = _collect_process_output(
                proc,
                timeout_seconds=cfg.timeout_seconds,
                max_output_chars=cfg.max_output_chars,
            )
            stdout = stdout[: cfg.max_output_chars]
            stderr = stderr[: cfg.max_output_chars]

            if failure_reason == "timeout":
                return f"代码执行超时（{cfg.timeout_seconds}秒），已终止。"
            if failure_reason == "output_limit":
                return f"代码输出超过上限（{cfg.max_output_chars} 字符），已终止。"

            if proc.returncode == 0:
                return stdout.strip() or "(代码执行成功，无输出)"

            if proc.returncode is not None and proc.returncode < 0:
                signum = -proc.returncode
                if signum in {getattr(signal, "SIGKILL", 9), getattr(signal, "SIGXCPU", 24)}:
                    return "代码因资源限制被终止。"
                return f"代码被信号终止: SIG{signum}"

            combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part).strip()
            if not combined:
                combined = f"exit={proc.returncode}"
            return f"代码执行失败: {combined}"
    except Exception as exc:
        logger.exception("run_code execution failed")
        return f"执行错误: {exc}"
    finally:
        sem.release()


# ── 注册核心 Tools ──

CORE_TOOLS: List[ToolDef] = [
    ToolDef(
        name="search_local",
        description="检索本地知识库（向量数据库 + 图谱融合检索），适用于查找已入库的论文和文档内容。",
        parameters=_SEARCH_LOCAL_SCHEMA,
        handler=_handle_search_local,
    ),
    ToolDef(
        name="search_web",
        description="网络搜索（Tavily/Google），获取最新的在线信息和网页内容。",
        parameters=_SEARCH_WEB_SCHEMA,
        handler=_handle_search_web,
    ),
    ToolDef(
        name="search_scholar",
        description="学术论文搜索（Semantic Scholar），查找特定领域的学术文献，获取标题、摘要、DOI。",
        parameters=_SEARCH_SCHOLAR_SCHEMA,
        handler=_handle_search_scholar,
    ),
    ToolDef(
        name="explore_graph",
        description="知识图谱探索，查看指定实体的关联实体和关系，发现跨文档的知识连接。",
        parameters=_EXPLORE_GRAPH_SCHEMA,
        handler=_handle_explore_graph,
    ),
    ToolDef(
        name="canvas",
        description="操作研究画布：创建(create)、获取(get)、更新(update)画布内容。",
        parameters=_CANVAS_SCHEMA,
        handler=_handle_canvas,
    ),
    ToolDef(
        name="get_citations",
        description="获取画布的引文列表，支持 text 和 bibtex 格式。",
        parameters=_CITATIONS_SCHEMA,
        handler=_handle_citations,
    ),
    ToolDef(
        name="compare_papers",
        description="多文档对比：选择 2-5 篇论文，自动生成结构化对比矩阵和分析。",
        parameters=_COMPARE_SCHEMA,
        handler=_handle_compare,
    ),
    ToolDef(
        name="run_code",
        description="执行简单的 Python 代码进行数据计算、统计验证或格式转换。",
        parameters=_RUN_CODE_SCHEMA,
        handler=_handle_run_code,
    ),
    ToolDef(
        name="summarize_quantitative",
        description=(
            "从证据文本中提取指定变量的定量数据，自动计算跨研究范围、均值、中位数、标准差，"
            "并输出结构化的跨研究对比表。适用于需要汇总多篇论文中同一指标数值的场景。"
        ),
        parameters=_SUMMARIZE_QUANT_SCHEMA,
        handler=_handle_summarize_quantitative,
    ),
    ToolDef(
        name="search_ncbi",
        description=(
            "搜索 NCBI PubMed 生物医学文献库，专攻生物学、医学、基因组学、海洋生态等领域。"
            "返回标题、作者、年份、DOI，适合精确的生物医学文献检索。"
        ),
        parameters=_SEARCH_NCBI_SCHEMA,
        handler=_handle_search_ncbi,
    ),
    ToolDef(
        name="search_sonar",
        description=(
            "使用 Perplexity Sonar Reasoning Pro 进行联网深度搜索和推理分析，"
            "适用于需要最新信息、跨领域知识或深度推理的问题。返回带引用的结构化回答，引用会纳入引文池。"
        ),
        parameters=_SEARCH_SONAR_SCHEMA,
        handler=_handle_search_sonar,
    ),
]


# ────────────────────────────────────────────────
# Tool Registry & Dynamic Skill Routing
# ────────────────────────────────────────────────

_TOOL_REGISTRY: Dict[str, ToolDef] = {t.name: t for t in CORE_TOOLS}

_GROUP_SEARCH_LOCAL = frozenset({"search_local"})
_GROUP_WEB = frozenset({"search_web", "search_scholar", "search_ncbi"})
_GROUP_SONAR = frozenset({"search_sonar"})
_GROUP_ANALYSIS = frozenset({"compare_papers", "summarize_quantitative"})
_GROUP_ANALYSIS_OPTIONAL = frozenset({"run_code"})
_GROUP_GRAPH = frozenset({"explore_graph"})
_GROUP_COLLAB = frozenset({"canvas", "get_citations"})

_WEB_PROVIDER_TO_TOOL: Dict[str, str] = {
    "tavily": "search_web",
    "google": "search_web",
    "scholar": "search_scholar",
    "semantic": "search_scholar",
    "ncbi": "search_ncbi",
    "pubmed": "search_ncbi",
    "sonar": "search_sonar",  # 独立检索工具，与预研究分离；模型由 agent_sonar_model 指定
}

_RE_ANALYSIS = re.compile(
    r"对比|比较|差异|统计|计算|代码|数据分析|分析数据|定量|数值|浓度|样本量"
    r"|compare|contrast|diff|statistic|calculat|code|data\s*analy|quantitat|numeric|sample.size|concentration",
    re.IGNORECASE,
)
_RE_GRAPH = re.compile(
    r"关系|网络|图谱|关联|知识图"
    r"|relation|network|graph|connection|linked|topology",
    re.IGNORECASE,
)
_RE_COLLAB = re.compile(
    r"画布|草稿|大纲|引用|参考文献|引文"
    r"|canvas|draft|outline|citation|reference|bibliography",
    re.IGNORECASE,
)

_TOOL_ORDER: Dict[str, int] = {t.name: i for i, t in enumerate(CORE_TOOLS)}


def get_tools_by_names(names: List[str]) -> List[ToolDef]:
    """Return ToolDef instances matching the given tool names, preserving CORE_TOOLS order."""
    tools = [_TOOL_REGISTRY[n] for n in names if n in _TOOL_REGISTRY and _tool_is_enabled(n)]
    tools.sort(key=lambda t: _TOOL_ORDER.get(t.name, 999))
    return tools


def get_routed_skills(
    message: str,
    current_stage: str,
    search_mode: str,
    allowed_web_providers: Optional[List[str]] = None,
) -> List[ToolDef]:
    """
    Dynamic skill routing — select only the tools relevant to the current
    request instead of mounting all CORE_TOOLS.

    This reduces prompt token cost and lowers the probability of the LLM
    hallucinating tool calls to irrelevant tools.

    Routing rules
    ─────────────
    1. search_local: always on when search_mode != "none"
    2. Web group (search_web / search_scholar / search_ncbi):
       active when search_mode allows web; narrowed by allowed_web_providers
    3. Sonar (search_sonar): when web/hybrid and "sonar" in allowed_web_providers (independent
       from pre-research; model set via agent_sonar_model).
    4. Analysis group (compare_papers / run_code):
       keyword-triggered by comparison / statistics / code mentions
    5. Graph group (explore_graph):
       keyword-triggered by relationship / graph / network mentions
    6. Collab group (canvas / get_citations):
       stage-triggered (drafting / refine) or keyword-triggered
    """
    selected: set[str] = set()

    # 1. Local search — the backbone of RAG
    if search_mode != "none":
        selected |= _GROUP_SEARCH_LOCAL

    # 2. Web tools — gated by search_mode and optionally by explicit provider list
    if search_mode in ("web", "hybrid"):
        if allowed_web_providers is not None:
            for provider in allowed_web_providers:
                tool_name = _WEB_PROVIDER_TO_TOOL.get(provider.lower().strip())
                if tool_name and tool_name in _GROUP_WEB:
                    selected.add(tool_name)
        else:
            selected |= _GROUP_WEB
        # Sonar 作为独立 Web 检索工具：仅当 allowed_web_providers 含 "sonar" 时加入（与预研究分离，模型由 agent_sonar_model 指定）
        if allowed_web_providers is not None and "sonar" in [p.lower().strip() for p in allowed_web_providers]:
            selected |= _GROUP_SONAR
        elif allowed_web_providers is None:
            # 未传列表时保持原行为：web/hybrid 即提供 Sonar
            selected |= _GROUP_SONAR

    # 3. Analysis group — keyword activated
    if _RE_ANALYSIS.search(message):
        selected |= _GROUP_ANALYSIS
        if _run_code_enabled():
            selected |= _GROUP_ANALYSIS_OPTIONAL

    # 4. Graph group — keyword activated
    if _RE_GRAPH.search(message):
        selected |= _GROUP_GRAPH

    # 5. Collaboration group — stage or keyword activated
    stage_lower = (current_stage or "").lower()
    if stage_lower in ("drafting", "draft", "refine", "writing"):
        selected |= _GROUP_COLLAB
    elif _RE_COLLAB.search(message):
        selected |= _GROUP_COLLAB

    # Fallback: guarantee at least search_local so the agent is never toolless
    if not selected and search_mode != "none":
        selected.add("search_local")

    tools = [_TOOL_REGISTRY[name] for name in selected if name in _TOOL_REGISTRY and _tool_is_enabled(name)]
    tools.sort(key=lambda t: _TOOL_ORDER.get(t.name, 999))

    logger.info(
        "skill_router | stage=%s mode=%s providers=%s → tools=[%s] (%d/%d)",
        current_stage, search_mode, allowed_web_providers,
        ", ".join(t.name for t in tools), len(tools), len(CORE_TOOLS),
    )
    return tools
