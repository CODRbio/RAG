"""
Codex App Server Provider
=========================
通过 `codex app-server --listen stdio://` 启动本地子进程，使用
双向 JSON-RPC 2.0（省略 jsonrpc 头）over stdio 与之通信。

官方定位：development / debug 深度集成接口，可能随版本变动。
本 provider 作为独立 experimental provider 接入平台，不影响主链路。

协议要点：
  - 每行一条 JSON（JSONL）
  - 请求：{"method": "...", "params": {...}, "id": N}
  - 响应：{"id": N, "result": {...}} 或 {"id": N, "error": {...}}
  - 通知：{"method": "...", "params": {...}}（无 id）
  - 握手：initialize → initialized（必须在任何业务请求前完成）

会话模型：
  - 进程可短（per-request），线程要长（跨请求复用 thread_id）
  - codex_thread_id 存入调用方传入的 overrides 或返回 meta 中，
    由 routes 层决定是否持久化到 session.preferences

部署前提：
  npm i -g @openai/codex   # codex --version 验证
"""

from __future__ import annotations

from collections import deque
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from src.log import get_logger
from src.llm.llm_manager import (
    _chunk_text_for_stream,
    BaseChatClient,
    ProviderConfig,
    RawLogStore,
    messages_digest,
    now_iso,
)

_log = get_logger(__name__)

# ── 静态 fallback 模型（app-server model/list 不可用时兜底）
CODEX_FALLBACK_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
]
DEFAULT_CODEX_MODEL = "gpt-5.4"

# app-server 启动后等待第一行输出的超时（秒）
_BOOT_TIMEOUT = 15.0
# 单次 RPC 请求默认超时
_DEFAULT_RPC_TIMEOUT = 180.0


# ============================================================
# JSON-RPC Process
# ============================================================


class CodexRpcError(Exception):
    """app-server 返回 error 对象时抛出"""

    def __init__(self, method: str, error: Dict[str, Any]):
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message", "unknown error")
        super().__init__(f"[{method}] RPC error {self.code}: {self.message}")


class CodexRpcProcess:
    """
    管理单个 `codex app-server` 子进程的生命周期及消息收发。

    设计决策：
      - 后台 daemon 线程持续读 stdout，按 id 派发响应或缓存通知
      - _pending 用于 request/response 配对（id → Queue）
      - _notifications 缓存所有无 id 的通知，供 iter_notifications 消费
      - 线程安全：_lock 保护 _msg_id 和 stdin 写操作
    """

    def __init__(
        self,
        cli_path: str = "codex",
        timeout: float = _DEFAULT_RPC_TIMEOUT,
        experimental_api: bool = False,
        env_override: Optional[Dict[str, str]] = None,
    ):
        self._cli_path = cli_path
        self._timeout = timeout
        self._experimental_api = experimental_api
        self._env_override = env_override or {}

        self._proc: Optional[subprocess.Popen] = None
        self._msg_id = 0
        self._lock = threading.Lock()
        self._pending: Dict[int, queue.Queue] = {}
        self._notifications: queue.Queue = queue.Queue()
        self._stderr_lock = threading.Lock()
        self._recent_stderr: deque[str] = deque(maxlen=20)
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._closed = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the app-server process and start the background reader."""
        env = {**os.environ, **self._env_override}
        cmd = [self._cli_path, "app-server", "--listen", "stdio://"]
        _log.debug("Spawning codex app-server: %s", cmd)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="codex-rpc-reader"
        )
        self._reader_thread.start()
        if self._proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, daemon=True, name="codex-rpc-stderr"
            )
            self._stderr_thread.start()
        _log.debug("codex app-server started, pid=%s", self._proc.pid)

    def close(self) -> None:
        """Terminate the app-server process gracefully."""
        if self._closed:
            return
        self._closed = True
        if self._proc and self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._reader_thread:
            self._reader_thread.join(timeout=0.2)
        if self._stderr_thread:
            self._stderr_thread.join(timeout=0.2)
        _log.debug("codex app-server process closed")

    def __enter__(self) -> "CodexRpcProcess":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Internal reader loop ───────────────────────────────────

    def _read_loop(self) -> None:
        """Background thread: read JSONL from stdout and dispatch."""
        assert self._proc and self._proc.stdout
        read_error: Optional[str] = None
        try:
            for raw_line in self._proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    _log.warning("codex: undecodable line: %s", line[:200])
                    continue
                if not isinstance(msg, dict):
                    _log.debug(
                        "codex: ignoring non-object stdout message type=%s payload=%r",
                        type(msg).__name__,
                        msg,
                    )
                    continue

                msg_id = msg.get("id")
                if msg_id is not None:
                    # Response to a pending request
                    pending_q = self._pending.pop(msg_id, None)
                    if pending_q is not None:
                        pending_q.put(msg)
                    else:
                        _log.warning("codex: orphan response id=%s", msg_id)
                else:
                    # Notification (no id)
                    self._notifications.put(msg)
        except Exception as exc:
            read_error = str(exc)
            _log.debug("codex reader loop exception: %s", exc)
        finally:
            # Always unblock any still-pending requests (normal exit or exception).
            # Use list() snapshot to avoid RuntimeError from concurrent pop() in request().
            error_msg = read_error or "codex process stdout closed"
            for q in list(self._pending.values()):
                q.put({"error": {"code": -1, "message": error_msg}})

    def _stderr_loop(self) -> None:
        """Drain stderr continuously so the child process cannot block on a full pipe."""
        assert self._proc and self._proc.stderr
        try:
            for raw_line in self._proc.stderr:
                line = raw_line.strip()
                if line:
                    with self._stderr_lock:
                        self._recent_stderr.append(line)
                    _log.debug("codex stderr: %s", line[:500])
        except Exception as exc:
            _log.debug("codex stderr loop exception: %s", exc)

    def transport_error_summary(self) -> Optional[str]:
        with self._stderr_lock:
            lines = list(self._recent_stderr)
        for line in reversed(lines):
            summary = _summarize_codex_transport_stderr(line)
            if summary:
                return summary
        return None

    # ── Messaging ──────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._msg_id += 1
            return self._msg_id

    def _write(self, payload: Dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Send a JSON-RPC request and block until response arrives.
        Raises CodexRpcError on error responses, queue.Empty on timeout.
        """
        msg_id = self._next_id()
        resp_q: queue.Queue = queue.Queue()
        self._pending[msg_id] = resp_q

        # Newer Codex CLI builds require the `params` field to exist even for
        # methods that take no arguments (for example `model/list`).
        payload: Dict[str, Any] = {"method": method, "id": msg_id, "params": params or {}}

        self._write(payload)

        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            resp = resp_q.get(timeout=effective_timeout)
        except queue.Empty:
            self._pending.pop(msg_id, None)
            transport_summary = self.transport_error_summary()
            if transport_summary:
                raise queue.Empty(f"{transport_summary} (RPC method: {method})")
            raise queue.Empty(f"codex RPC timeout after {effective_timeout}s: {method}")

        if "error" in resp:
            error = resp["error"]
            if isinstance(error, dict):
                transport_summary = self.transport_error_summary()
                error_message = str(error.get("message", ""))
                if transport_summary and (
                    error.get("code") == -1
                    or "stdout closed" in error_message.lower()
                    or "rpc timeout" in error_message.lower()
                ):
                    error = dict(error)
                    error["message"] = transport_summary
            raise CodexRpcError(method, error)
        return resp.get("result")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        payload: Dict[str, Any] = {"method": method, "params": params or {}}
        self._write(payload)

    def iter_notifications(
        self, timeout: float | None = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield notifications from the queue until timeout.
        Caller is responsible for breaking on `turn/completed`.
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self._timeout)
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log.warning("codex: iter_notifications hit deadline")
                break
            try:
                msg = self._notifications.get(timeout=min(1.0, remaining))
                yield msg
            except queue.Empty:
                # Check if process died
                if self._proc and self._proc.poll() is not None:
                    _log.warning("codex: process exited during iter_notifications")
                    break


# ============================================================
# Helpers
# ============================================================


def _summarize_codex_transport_stderr(line: str) -> Optional[str]:
    match = re.search(
        r"failed to connect to websocket:\s*(.*?)(?:,\s*url:\s*(\S+))?$",
        line,
        re.IGNORECASE,
    )
    if not match:
        return None
    detail = match.group(1).strip()
    url = (match.group(2) or "").strip()
    suffix = f" ({url})" if url else ""
    lowered = detail.lower()

    if "401 unauthorized" in lowered:
        return (
            f"Codex authentication failed while opening the websocket{suffix}. "
            f"Re-run `codex login`, or explicitly configure a valid "
            f"`RAG_LLM__CODEX_APP_SERVER__API_KEY`. Last error: {detail}."
        )
    if "connection reset by peer" in lowered or "tls handshake eof" in lowered or "handshake eof" in lowered:
        return (
            f"Codex transport failed while opening the websocket{suffix}. "
            f"This is usually a network, proxy, VPN, firewall, or TLS interception issue "
            f"outside the app. Last error: {detail}."
        )
    return f"Codex websocket connection failed{suffix}: {detail}."


def _summarize_codex_turn_error(message: str) -> Optional[str]:
    text = str(message or "").strip()
    if not text:
        return None
    lowered = text.lower()
    url_match = re.search(r"(https?://\S+|wss?://\S+)", text)
    url = (url_match.group(1) if url_match else "").rstrip(").,")
    suffix = f" ({url})" if url else ""

    if "401 unauthorized" in lowered:
        return (
            f"Codex authentication failed while contacting the response endpoint{suffix}. "
            f"Re-run `codex login`, or explicitly configure a valid "
            f"`RAG_LLM__CODEX_APP_SERVER__API_KEY`. Last error: {text}."
        )
    if (
        "stream disconnected before completion" in lowered
        or "error sending request for url" in lowered
        or "failed to lookup address information" in lowered
        or "connection reset by peer" in lowered
        or "tls handshake eof" in lowered
        or "handshake eof" in lowered
    ):
        return (
            f"Codex transport failed while contacting the response endpoint{suffix}. "
            f"This is usually a network, proxy, VPN, firewall, or TLS interception issue "
            f"outside the app. Last error: {text}."
        )
    return None


def _messages_to_input(
    messages: List[Dict[str, Any]],
    *,
    has_thread: bool,
) -> str:
    """
    Convert OpenAI-format messages list to a single input text for codex.

    - has_thread=True:  只返回最新一条 user 消息（历史已在 thread 里）
    - has_thread=False: 把所有消息拼成首轮上下文（system + history + user）
    """
    if has_thread:
        # Only send the latest user turn
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    return "\n".join(parts)
                return str(content)
        return ""
    else:
        # First call: combine everything into context
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            prefix = {"system": "[System]", "assistant": "[Assistant]"}.get(
                role, "[User]"
            )
            parts.append(f"{prefix}: {content}")
        return "\n\n".join(parts)


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_text(v) for v in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "value", "content", "delta"):
            extracted = _extract_text(value.get(key))
            if extracted:
                return extracted
    return ""


def _find_codex_bin() -> str:
    found = shutil.which("codex")
    if found:
        return found
    return "codex"


# ============================================================
# Chat Client
# ============================================================


class CodexAppServerChatClient(BaseChatClient):
    """
    BaseChatClient 实现，委托给 `codex app-server` 子进程。

    每次 chat() / stream_chat() 调用都会 spawn 一个独立的 app-server 进程，
    完成后关闭（per-request 模式）。codex thread 可跨调用复用：
      - 调用方通过 overrides["codex_thread_id"] 传入已有 thread_id
      - 返回 meta["codex_thread_id"] 供调用方持久化（存 session.preferences）

    認證優先順序：
      1. config.api_key 非空 → account/login/start(type=apiKey)
      2. api_key 为空 → account/read（走本机已缓存登录态）
         若未登录且无 key → 抛明确错误提示

    審批／沙箱策略（首版保守）：
      - 默认 readOnly 沙箱
      - overrides["sandbox_policy"] 可覆盖为 workspaceWrite
      - 任何审批请求（commandExecution / fileChange）在 readOnly 下默认 decline
    """

    supports_platform_tool_calls = False

    def __init__(
        self,
        config: ProviderConfig,
        log_store: Optional[RawLogStore] = None,
    ):
        self.config = config
        self.log_store = log_store
        self._cli_path = _find_codex_bin()

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or self.config.default_model or DEFAULT_CODEX_MODEL

    def _build_structured_output_schema(self, response_model: Any) -> Optional[Dict[str, Any]]:
        try:
            schema = response_model.model_json_schema()
        except Exception:
            return None
        return schema if isinstance(schema, dict) else None

    def _parse_structured_output(self, response_model: Any, final_text: str) -> Any:
        text = (final_text or "").strip()
        candidates: List[str] = []
        if text:
            candidates.append(text)
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            fenced = fence_match.group(1).strip()
            if fenced:
                candidates.append(fenced)
        start = min((idx for idx in (text.find("{"), text.find("[")) if idx >= 0), default=-1)
        if start >= 0:
            end_obj = text.rfind("}")
            end_arr = text.rfind("]")
            end = max(end_obj, end_arr)
            if end >= start:
                snippet = text[start:end + 1].strip()
                if snippet:
                    candidates.append(snippet)
        seen: set[str] = set()
        last_error: Optional[Exception] = None
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return response_model.model_validate_json(candidate)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("empty structured output")

    def _normalize_overrides(
        self,
        overrides: Dict[str, Any],
        response_model: Optional[Any],
    ) -> Dict[str, Any]:
        normalized = dict(overrides)
        if response_model is not None and "output_schema" not in normalized:
            schema = self._build_structured_output_schema(response_model)
            if schema:
                normalized["output_schema"] = schema
        return normalized

    def _build_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if self.config.api_key:
            env["OPENAI_API_KEY"] = self.config.api_key
        return env

    def _handshake(self, rpc: CodexRpcProcess, app_version: str = "1.0.0") -> None:
        """Execute initialize → initialized handshake."""
        rpc.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "rag_platform",
                    "title": "RAG Platform",
                    "version": app_version,
                },
                "capabilities": {
                    "experimentalApi": False,
                },
            },
            timeout=_BOOT_TIMEOUT,
        )
        rpc.notify("initialized", {})
        _log.debug("codex: handshake complete")

    def _authenticate(self, rpc: CodexRpcProcess) -> None:
        """Authenticate via API key or cached local session."""
        if self.config.api_key:
            rpc.request(
                "account/login/start",
                {"type": "apiKey", "apiKey": self.config.api_key},
                timeout=30.0,
            )
            _log.debug("codex: authenticated via api_key")
        else:
            result = rpc.request("account/read", timeout=10.0)
            account = (result or {}).get("account")
            if account is None:
                requires_auth = (result or {}).get("requiresOpenaiAuth", True)
                if requires_auth:
                    raise RuntimeError(
                        "codex app-server: no local login session found and api_key is empty. "
                        "Run `codex login` on the server, or configure api_key in rag_config.json."
                    )
            _log.debug("codex: using cached local session, account=%s", account)

    def _ensure_thread(
        self,
        rpc: CodexRpcProcess,
        thread_id: Optional[str],
    ) -> str:
        """Start a new thread or resume an existing one. Returns thread_id."""
        if thread_id:
            rpc.request("thread/resume", {"threadId": thread_id}, timeout=15.0)
            _log.debug("codex: resumed thread=%s", thread_id)
            return thread_id
        else:
            result = rpc.request("thread/start", {}, timeout=15.0)
            new_id = (result or {}).get("thread", {}).get("id") or (result or {}).get("id")
            if not new_id:
                raise RuntimeError("codex: thread/start returned no thread id")
            _log.debug("codex: started new thread=%s", new_id)
            return str(new_id)

    def _default_sandbox_policy(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        custom = overrides.get("sandbox_policy")
        if isinstance(custom, dict):
            return custom
        return {"type": "readOnly"}

    def _handle_approval_request(
        self,
        rpc: CodexRpcProcess,
        method: str,
        params: Dict[str, Any],
        sandbox_policy: Dict[str, Any],
        approval_policy: str,
    ) -> None:
        policy_type = str(sandbox_policy.get("type") or "readOnly")
        allow_changes = policy_type != "readOnly" and approval_policy not in {"never", "deny"}
        decision = "approve" if allow_changes else "decline"
        if method == "item/commandExecution/requestApproval":
            exec_id = params.get("id") or params.get("commandExecutionId")
            if exec_id:
                rpc.request(
                    "item/commandExecution/setApprovalDecision",
                    {"commandExecutionId": exec_id, "decision": decision},
                    timeout=10.0,
                )
        elif method == "item/fileChange/requestApproval":
            change_id = params.get("id") or params.get("fileChangeId")
            if change_id:
                rpc.request(
                    "item/fileChange/setApprovalDecision",
                    {"fileChangeId": change_id, "decision": decision},
                    timeout=10.0,
                )

    def _run_turn(
        self,
        rpc: CodexRpcProcess,
        *,
        thread_id: str,
        input_text: str,
        model: str,
        overrides: Dict[str, Any],
        stream_cb=None,
    ) -> tuple[str, Optional[str]]:
        """
        Execute turn/start and collect events.

        Returns (final_text, turn_id).
        If stream_cb is provided, calls stream_cb(event_dict) for each text_delta.
        """
        sandbox_policy = self._default_sandbox_policy(overrides)
        approval_policy = str(overrides.get("approval_policy", "never"))
        turn_params: Dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": input_text}],
            "model": model,
            "effort": overrides.get("effort", "medium"),
            "summary": overrides.get("summary", "concise"),
            "approvalPolicy": approval_policy,
            "sandboxPolicy": sandbox_policy,
        }
        if overrides.get("cwd"):
            turn_params["cwd"] = overrides["cwd"]
        if overrides.get("personality"):
            turn_params["personality"] = overrides["personality"]
        if overrides.get("output_schema"):
            turn_params["outputSchema"] = overrides["output_schema"]

        turn_result = rpc.request("turn/start", turn_params, timeout=30.0)
        turn_id = (turn_result or {}).get("turn", {}).get("id") or (turn_result or {}).get("id")
        _log.debug("codex: turn started id=%s", turn_id)

        accumulated_text: List[str] = []
        # Track per-item final text (item/completed overrides deltas)
        item_final: Dict[str, str] = {}
        turn_completed = False
        turn_status = ""
        turn_error_message = ""
        recent_error_message = ""

        effective_timeout = float(overrides.get("timeout", self.config.params.get("timeout", _DEFAULT_RPC_TIMEOUT)))

        for notif in rpc.iter_notifications(timeout=effective_timeout):
            if not isinstance(notif, dict):
                _log.debug(
                    "codex: ignoring non-object notification type=%s payload=%r",
                    type(notif).__name__,
                    notif,
                )
                continue
            method = str(notif.get("method", ""))
            raw_params = notif.get("params")
            params = _as_dict(raw_params)

            if method == "item/agentMessage/delta":
                delta = _extract_text(params.get("delta", raw_params))
                if delta:
                    accumulated_text.append(delta)
                    if stream_cb:
                        # Match HTTPChatClient / routes_chat contract: `delta` + optional `text`
                        stream_cb({"type": "text_delta", "delta": delta, "text": delta})

            elif method == "item/completed":
                raw_item = params.get("item")
                item = _as_dict(raw_item)
                final = ""
                item_id = ""
                item_type = str(item.get("type") or params.get("type") or "")
                if item_type == "agentMessage":
                    # Use authoritative final text from item/completed
                    final = _extract_text(item.get("text") or item.get("content") or raw_item)
                    item_id = str(item.get("id", ""))
                elif raw_item is not None and not item_type:
                    final = _extract_text(raw_item)
                    item_id = str(params.get("id") or params.get("itemId") or f"item_{len(item_final) + 1}")
                if final:
                    item_final[item_id or f"item_{len(item_final) + 1}"] = final
                if stream_cb:
                    stream_cb({"type": "item_completed", "item": raw_item if raw_item is not None else item})

            elif method == "item/commandExecution/requestApproval":
                try:
                    self._handle_approval_request(
                        rpc, method, params, sandbox_policy, approval_policy,
                    )
                except Exception as e:
                    _log.warning("codex: failed to handle command approval: %s", e)

            elif method == "item/fileChange/requestApproval":
                try:
                    self._handle_approval_request(
                        rpc, method, params, sandbox_policy, approval_policy,
                    )
                except Exception as e:
                    _log.warning("codex: failed to handle file approval: %s", e)

            elif method == "turn/plan/updated":
                if stream_cb:
                    stream_cb({"type": "plan_update", "plan": params.get("items", [])})

            elif method == "turn/diff/updated":
                if stream_cb:
                    stream_cb({"type": "diff_update", "diff": _extract_text(params.get("diff", raw_params))})

            elif method == "item/commandExecution/outputDelta":
                output_delta = _extract_text(params.get("delta", raw_params))
                if stream_cb:
                    stream_cb({
                        "type": "tool_output",
                        "output": output_delta,
                    })

            elif method == "error":
                error_bits = [
                    _extract_text(params.get("message") if isinstance(params, dict) else None),
                    _extract_text(params.get("additionalDetails") if isinstance(params, dict) else None),
                    _extract_text(raw_params),
                ]
                recent_error_message = next((bit for bit in error_bits if bit), "")
                if recent_error_message:
                    _log.debug("codex: notification error=%s", recent_error_message)

            elif method == "turn/completed":
                turn_completed = True
                turn = _as_dict(params.get("turn"))
                turn_status = str(
                    turn.get("status") or params.get("status") or _extract_text(raw_params)
                )
                turn_error_message = _extract_text(turn.get("error") or params.get("error"))
                _log.debug("codex: turn completed status=%s", turn_status)
                break

            elif raw_params is not None and not isinstance(raw_params, dict):
                _log.debug(
                    "codex: unexpected notification params shape method=%s type=%s payload=%r",
                    method,
                    type(raw_params).__name__,
                    raw_params,
                )

        # Prefer item/completed authoritative text, fallback to accumulated deltas
        transport_summary_getter = getattr(rpc, "transport_error_summary", None)
        transport_summary = transport_summary_getter() if callable(transport_summary_getter) else None
        if not turn_completed:
            if transport_summary:
                raise RuntimeError(transport_summary)
            turn_failure_summary = _summarize_codex_turn_error(recent_error_message)
            if turn_failure_summary:
                raise RuntimeError(turn_failure_summary)
            if recent_error_message:
                raise RuntimeError(f"codex turn failed before completion: {recent_error_message}")
            raise RuntimeError("codex turn ended before completion")
        if turn_status and turn_status.lower() not in {"completed", "complete", "succeeded", "success"}:
            raise RuntimeError(
                transport_summary
                or _summarize_codex_turn_error(turn_error_message)
                or _summarize_codex_turn_error(recent_error_message)
                or (f"codex turn failed: {turn_error_message}" if turn_error_message else "")
                or (f"codex turn failed: {recent_error_message}" if recent_error_message else "")
                or f"codex turn failed with status={turn_status}"
            )
        if item_final:
            final_text = "\n".join(item_final.values())
        else:
            final_text = "".join(accumulated_text)

        return final_text, str(turn_id) if turn_id else None

    def _spawn_rpc(self) -> CodexRpcProcess:
        rpc = CodexRpcProcess(
            cli_path=self._cli_path,
            timeout=_DEFAULT_RPC_TIMEOUT,
            experimental_api=False,
            env_override=self._build_env(),
        )
        rpc.start()
        return rpc

    # ── BaseChatClient interface ───────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        tools = overrides.get("tools") or []
        if tools:
            raise ValueError(
                "Codex provider does not support the platform ReAct tool loop. "
                "Use direct chat with provider='codex', or switch to a standard HTTP provider for platform tools."
            )
        resolved_model = self._resolve_model(model)
        effective_overrides = self._normalize_overrides(dict(overrides), response_model)
        thread_id: Optional[str] = effective_overrides.get("codex_thread_id")
        input_text = _messages_to_input(messages, has_thread=bool(thread_id))

        t0 = time.time()
        rpc = self._spawn_rpc()
        parsed_object = None
        try:
            self._handshake(rpc)
            self._authenticate(rpc)
            thread_id = self._ensure_thread(rpc, thread_id)
            final_text, turn_id = self._run_turn(
                rpc,
                thread_id=thread_id,
                input_text=input_text,
                model=resolved_model,
                overrides=effective_overrides,
            )
            error = None
        except Exception as exc:
            final_text = ""
            error = str(exc)
            turn_id = None
            _log.error("codex chat error: %s", error)
        finally:
            rpc.close()

        latency_ms = int((time.time() - t0) * 1000)

        if error:
            raise RuntimeError(error)

        if response_model is not None:
            try:
                parsed_object = self._parse_structured_output(response_model, final_text)
            except Exception as exc:
                _log.warning("codex structured output parse failed: %s", exc)

        result: Dict[str, Any] = {
            "provider": self.config.name,
            "model": resolved_model,
            "final_text": final_text,
            "reasoning_text": None,
            "parsed_object": parsed_object,
            "raw": {
                "provider": "codex_app_server",
                "model": resolved_model,
                "thread_id": thread_id,
                "turn_id": turn_id,
            },
            "params": effective_overrides,
            "meta": {
                "usage": None,
                "latency_ms": latency_ms,
                "refusal": None,
                "error": None,
                "codex_thread_id": thread_id,
                "codex_turn_id": turn_id,
            },
        }

        if self.log_store:
            self.log_store.write({
                "timestamp": now_iso(),
                "provider": self.config.name,
                "model": resolved_model,
                "params": effective_overrides,
                "messages_digest": messages_digest(messages),
                "final_text": final_text,
                "reasoning_text": None,
                "raw_response": result["raw"],
                "meta": result["meta"],
                "error": None,
            })

        return result

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides: Any,
    ) -> Iterator[Dict[str, Any]]:
        tools = overrides.get("tools") or []
        if tools:
            raise ValueError(
                "Codex provider does not support the platform ReAct tool loop. "
                "Use direct chat with provider='codex', or switch to a standard HTTP provider for platform tools."
            )
        if response_model is not None:
            final_result = self.chat(
                messages,
                model=model,
                return_reasoning=return_reasoning,
                response_model=response_model,
                **overrides,
            )
            final_text = final_result.get("final_text") or ""
            for chunk in _chunk_text_for_stream(final_text):
                yield {"type": "text_delta", "delta": chunk}
            yield {"type": "completed", "response": final_result}
            return
        resolved_model = self._resolve_model(model)
        effective_overrides = self._normalize_overrides(dict(overrides), response_model)
        thread_id: Optional[str] = effective_overrides.get("codex_thread_id")
        input_text = _messages_to_input(messages, has_thread=bool(thread_id))

        t0 = time.time()
        rpc = self._spawn_rpc()

        # Collect streamed events in a thread-safe queue, yield from main thread
        event_q: queue.Queue = queue.Queue()

        def _stream_cb(event: Dict[str, Any]) -> None:
            event_q.put(event)

        def _run() -> None:
            try:
                self._handshake(rpc)
                self._authenticate(rpc)
                nonlocal thread_id
                thread_id = self._ensure_thread(rpc, thread_id)
                final_text, turn_id = self._run_turn(
                    rpc,
                    thread_id=thread_id,
                    input_text=input_text,
                    model=resolved_model,
                    overrides=effective_overrides,
                    stream_cb=_stream_cb,
                )
                latency_ms = int((time.time() - t0) * 1000)
                response_dict: Dict[str, Any] = {
                    "provider": self.config.name,
                    "model": resolved_model,
                    "final_text": final_text,
                    "reasoning_text": None,
                    "parsed_object": None,
                    "raw": {
                        "provider": "codex_app_server",
                        "model": resolved_model,
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                    },
                    "params": effective_overrides,
                    "meta": {
                        "usage": None,
                        "latency_ms": latency_ms,
                        "refusal": None,
                        "error": None,
                        "codex_thread_id": thread_id,
                        "codex_turn_id": turn_id,
                    },
                }
                event_q.put({
                    "type": "completed",
                    "response": response_dict,
                    "provider": self.config.name,
                    "model": resolved_model,
                    "final_text": final_text,
                    "reasoning_text": None,
                    "meta": response_dict["meta"],
                })
            except Exception as exc:
                latency_ms = int((time.time() - t0) * 1000)
                err_response: Dict[str, Any] = {
                    "provider": self.config.name,
                    "model": resolved_model,
                    "final_text": "",
                    "reasoning_text": None,
                    "parsed_object": None,
                    "raw": {},
                    "params": effective_overrides,
                    "meta": {
                        "usage": None,
                        "latency_ms": latency_ms,
                        "error": str(exc),
                        "codex_thread_id": thread_id,
                        "codex_turn_id": None,
                    },
                }
                event_q.put({
                    "type": "completed",
                    "response": err_response,
                    "provider": self.config.name,
                    "model": resolved_model,
                    "final_text": "",
                    "reasoning_text": None,
                    "meta": err_response["meta"],
                })
                _log.error("codex stream_chat error: %s", exc)
            finally:
                rpc.close()
                event_q.put(None)  # sentinel

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        try:
            while True:
                event = event_q.get(timeout=_DEFAULT_RPC_TIMEOUT + 10)
                if event is None:
                    break
                yield event
                if event.get("type") == "completed":
                    break
        finally:
            worker.join(timeout=5)

    def interrupt(self, thread_id: str, turn_id: str) -> None:
        """
        中断进行中的 turn（需要进程仍在运行，不适用于 per-request 模式）。
        此方法保留用于未来长驻进程池场景。
        """
        _log.warning(
            "codex.interrupt() called but per-request mode has no live process; "
            "thread_id=%s turn_id=%s",
            thread_id,
            turn_id,
        )


# ============================================================
# Model list fetch (for model_registry)
# ============================================================


def fetch_codex_models(
    api_key: str = "",
    cli_path: str = "codex",
    timeout: float = 20.0,
) -> List[str]:
    """
    Start a short-lived app-server process to call model/list.
    Returns model id list, or raises on error.
    Used by CodexProvider.fetch_models() in model_registry.py.
    """
    env: Dict[str, str] = {}
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    with CodexRpcProcess(
        cli_path=cli_path, timeout=timeout, env_override=env
    ) as rpc:
        rpc.request(
            "initialize",
            {
                "clientInfo": {"name": "rag_platform", "title": "RAG Platform", "version": "1.0.0"},
                "capabilities": {"experimentalApi": False},
            },
            timeout=_BOOT_TIMEOUT,
        )
        rpc.notify("initialized", {})

        if api_key:
            rpc.request(
                "account/login/start",
                {"type": "apiKey", "apiKey": api_key},
                timeout=15.0,
            )

        result = rpc.request("model/list", timeout=timeout)
        models = (result or {}).get("models") or []
        return [m["id"] for m in models if m.get("id")]
