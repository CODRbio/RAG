"""
统一 LLM 管理模块

功能：
- Provider 基类统一封装（OpenAI-compatible / Anthropic）
- 输出规范化：final_text / reasoning_text 隔离
- Raw JSON 落库（JSONL）+ 清理策略（10天/100MB）
- API key 环境变量覆盖 + 脱敏
- dry_run 模式支持

配置来源：config/rag_config.json（可选 config/rag_config.local.json 覆盖）
"""

from __future__ import annotations

import logging
import os
import json
import threading
import time
import hashlib

_log = logging.getLogger(__name__)

# ── Observability ──
try:
    from src.observability import metrics as _obs_metrics, tracer as _obs_tracer
except Exception:
    _obs_metrics = None  # type: ignore
    _obs_tracer = None  # type: ignore

from src.observability.tracing import traceable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

import requests

try:
    from config.settings import settings
except Exception:
    settings = None  # type: ignore

# ============================================================
# Constants
# ============================================================

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 120  # seconds
LOG_DIR_NAME = "llm_raw"
LOG_MAX_AGE_DAYS = 10
LOG_MAX_TOTAL_MB = 100
MESSAGE_DIGEST_LENGTH = 200


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class PlatformConfig:
    """平台级配置：api_key + base_url，多个 provider 变体共享"""
    name: str
    api_key: str
    base_url: str


@dataclass
class ProviderConfig:
    """单个 provider 的配置（api_key/base_url 从所属 platform 继承）"""
    name: str
    api_key: str
    base_url: str
    default_model: str
    platform: str = ""
    models: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def is_anthropic(self) -> bool:
        """判断是否为 Anthropic 协议"""
        return "anthropic.com" in self.base_url or self.name.startswith("claude")


@dataclass
class LLMConfig:
    """完整 LLM 配置"""
    default: str
    dry_run: bool
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)


# ============================================================
# Helper Functions
# ============================================================

def load_json(path: str | Path) -> Dict[str, Any]:
    """加载 JSON 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_with_local(path: str | Path) -> Dict[str, Any]:
    """
    加载 JSON 配置文件并合并本地覆盖文件：
    - base: config/rag_config.json
    - local: config/rag_config.local.json（可选）
    """
    base_path = Path(path)
    base = load_json(base_path)
    local_path = base_path.with_name(f"{base_path.stem}.local{base_path.suffix}")
    if local_path.exists():
        base = deep_merge(base, load_json(local_path))
    return base


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归深合并两个字典。
    override 中的值会覆盖 base 中的同名键；嵌套 dict 会递归合并。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    脱敏显示密钥，保留前后若干字符。
    例如: "sk-abc...xyz"
    """
    if not secret:
        return "(empty)"
    if len(secret) <= show_chars * 2 + 3:
        return "*" * len(secret)
    return f"{secret[:show_chars]}...{secret[-show_chars:]}"


def provider_env_var(provider_name: str) -> str:
    """
    生成 provider 对应的环境变量名。
    规则: RAG_LLM__{PROVIDER}__API_KEY
    provider 名称转大写，- 替换为 _
    例如: claude-thinking => RAG_LLM__CLAUDE_THINKING__API_KEY
    """
    normalized = provider_name.upper().replace("-", "_")
    return f"RAG_LLM__{normalized}__API_KEY"


def now_iso() -> str:
    """返回 ISO 格式的当前时间戳"""
    return datetime.now().isoformat()


def messages_digest(messages: List[Dict[str, Any]], max_len: int = MESSAGE_DIGEST_LENGTH) -> str:
    """
    生成 messages 的摘要（用于日志，不含完整内容）。
    格式: role: content[:max_len]
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            # 多模态消息，提取文本部分
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if isinstance(content, str):
            truncated = content[:max_len] + ("..." if len(content) > max_len else "")
            parts.append(f"[{role}] {truncated}")
    return "\n".join(parts)


# ============================================================
# RawLogStore
# ============================================================

class RawLogStore:
    """
    原始 JSON 响应日志存储。
    - 按天写入 JSONL 文件: YYYY-MM-DD.jsonl
    - 支持清理策略: 超过 N 天删除，总大小超过 M MB 删除最旧文件
    """

    def __init__(self, log_dir: Path | str | None = None):
        if log_dir is None:
            # 默认: 项目根目录/logs/llm_raw/
            log_dir = Path(__file__).parent.parent.parent / "logs" / LOG_DIR_NAME
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        """当天的日志文件路径"""
        return self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def write(self, record: Dict[str, Any]) -> None:
        """
        写入一条日志记录。
        record 应包含: timestamp, provider, model, params, messages_digest,
                      final_text, reasoning_text, raw_response, meta, error
        """
        record.setdefault("timestamp", now_iso())
        log_file = self._today_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def cleanup(
        self,
        max_age_days: int = LOG_MAX_AGE_DAYS,
        max_total_mb: int = LOG_MAX_TOTAL_MB
    ) -> Dict[str, Any]:
        """
        清理日志文件。
        1. 删除超过 max_age_days 天的文件
        2. 若总大小 > max_total_mb MB，从最旧文件开始删除

        Returns:
            清理报告: {"deleted_by_age": [...], "deleted_by_size": [...], "remaining_mb": float}
        """
        report = {"deleted_by_age": [], "deleted_by_size": [], "remaining_mb": 0.0}
        
        if not self.log_dir.exists():
            return report

        # 获取所有 jsonl 文件，按修改时间排序（旧 -> 新）
        log_files = sorted(
            self.log_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime
        )

        cutoff = datetime.now() - timedelta(days=max_age_days)

        # 1. 删除超龄文件
        remaining_files = []
        for f in log_files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                report["deleted_by_age"].append(f.name)
                f.unlink()
            else:
                remaining_files.append(f)

        # 2. 检查总大小，删除最旧文件
        max_bytes = max_total_mb * 1024 * 1024
        while remaining_files:
            total_size = sum(f.stat().st_size for f in remaining_files)
            if total_size <= max_bytes:
                break
            oldest = remaining_files.pop(0)
            report["deleted_by_size"].append(oldest.name)
            oldest.unlink()

        # 计算剩余大小
        if remaining_files:
            report["remaining_mb"] = sum(f.stat().st_size for f in remaining_files) / (1024 * 1024)

        return report


# ============================================================
# Provider Classes
# ============================================================

class Provider(ABC):
    """Provider 基类：负责 HTTP 请求"""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def request(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        发送请求到 LLM API。
        
        Args:
            payload: 请求体（已合并参数）
            
        Returns:
            原始 JSON 响应
        """
        raise NotImplementedError


def _llm_perf_timeout() -> int:
    if settings and hasattr(settings, "perf_llm"):
        return getattr(settings.perf_llm, "timeout_seconds", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT
    return DEFAULT_TIMEOUT


def _llm_perf_retry() -> tuple:
    if settings and hasattr(settings, "perf_llm"):
        max_r = getattr(settings.perf_llm, "max_retries", 2) or 0
        backoff = getattr(settings.perf_llm, "retry_backoff", 1.5) or 1.5
        return max_r, backoff
    return 0, 1.5


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    **kwargs: Any,
) -> requests.Response:
    max_retries, backoff = _llm_perf_retry()
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in (429, 500, 503) and attempt < max_retries:
                time.sleep(backoff ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            last_err = e
            if e.response.status_code not in (429, 500, 503) or attempt >= max_retries:
                raise
            time.sleep(backoff ** attempt)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(backoff ** attempt)
    if last_err:
        raise last_err
    raise RuntimeError("request_with_retry failed")


class OpenAICompatProvider(Provider):
    """
    OpenAI 兼容协议 Provider。
    适用于: OpenAI, DeepSeek, Gemini, Kimi 等
    使用 Session 复用连接，支持可配置超时与重试。
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._session = requests.Session()

    def request(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        timeout = int(timeout or _llm_perf_timeout())
        resp = _request_with_retry(
            self._session, "POST", url, timeout,
            headers=headers, json=payload,
        )
        return resp.json()


class AnthropicProvider(Provider):
    """
    Anthropic 协议 Provider。
    适用于: Claude 系列
    使用 Session 复用连接，支持可配置超时与重试。
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._session = requests.Session()

    def request(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        timeout = int(timeout or _llm_perf_timeout())
        resp = _request_with_retry(
            self._session, "POST", url, timeout,
            headers=headers, json=payload,
        )
        return resp.json()


# ============================================================
# Response Normalization
# ============================================================

def normalize_response(provider_name: str, raw: Dict[str, Any], is_anthropic: bool = False) -> Dict[str, Any]:
    """
    规范化 LLM 响应，提取 final_text 和 reasoning_text。
    
    Args:
        provider_name: provider 名称（用于日志/调试）
        raw: 原始 JSON 响应
        is_anthropic: 是否为 Anthropic 协议
        
    Returns:
        {
            "final_text": str | None,
            "reasoning_text": str | None,
            "usage": dict | None,
            "refusal": bool | None
        }
    """
    result = {
        "final_text": None,
        "reasoning_text": None,
        "usage": None,
        "refusal": None,
    }

    try:
        if is_anthropic:
            result.update(_normalize_anthropic(raw))
        else:
            result.update(_normalize_openai_compat(raw))
    except Exception:
        # 容错：抽取失败不抛异常，raw 已保存
        pass

    return result


def _normalize_openai_compat(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI-compatible 响应抽取规则。
    
    final_text: 
        - choices[0].message.content (str)
        - 若为 list，拼接 type=="text" 的段落
        
    reasoning_text (best-effort):
        - choices[0].message.reasoning / thoughts
        - content 为 list 时，拼接 type in ("reasoning", "thinking") 的段落
    """
    result = {"final_text": None, "reasoning_text": None, "usage": None, "refusal": None}

    # usage
    result["usage"] = raw.get("usage")

    choices = raw.get("choices") or []
    if not choices:
        return result

    message = choices[0].get("message") or {}
    content = message.get("content")
    
    # refusal 检测
    if message.get("refusal"):
        result["refusal"] = True

    # 处理 content
    if isinstance(content, str):
        result["final_text"] = content
    elif isinstance(content, list):
        # 多段内容
        text_parts = []
        reasoning_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            text = block.get("text", "")
            if block_type == "text":
                text_parts.append(text)
            elif block_type in ("reasoning", "thinking"):
                reasoning_parts.append(text)
        result["final_text"] = "".join(text_parts) if text_parts else None
        if reasoning_parts:
            result["reasoning_text"] = "".join(reasoning_parts)

    # 尝试从其他字段抽取 reasoning（某些模型的特殊字段）
    if not result["reasoning_text"]:
        for field_name in ("reasoning", "thoughts", "reasoning_content"):
            reasoning = message.get(field_name)
            if reasoning and isinstance(reasoning, str):
                result["reasoning_text"] = reasoning
                break

    return result


def _normalize_anthropic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anthropic 响应抽取规则。
    
    final_text: 拼接 content[] 中 type=="text" 的 .text
    reasoning_text: 拼接 content[] 中 type=="thinking" 的 .thinking
    """
    result = {"final_text": None, "reasoning_text": None, "usage": None, "refusal": None}

    # usage
    result["usage"] = raw.get("usage")

    # stop_reason 检测
    stop_reason = raw.get("stop_reason")
    if stop_reason == "refusal":
        result["refusal"] = True

    content = raw.get("content") or []
    text_parts = []
    thinking_parts = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            thinking_parts.append(block.get("thinking", ""))

    result["final_text"] = "".join(text_parts) if text_parts else None
    result["reasoning_text"] = "".join(thinking_parts) if thinking_parts else None

    return result


# ============================================================
# Chat Clients
# ============================================================

class BaseChatClient(ABC):
    """Chat 客户端基类"""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Dict[str, Any]:
        """
        发送消息并获取响应。
        
        Args:
            messages: OpenAI 格式消息列表 [{"role": "user", "content": "..."}]
            model: 可选的模型名或别名，覆盖默认
            return_reasoning: 是否在返回中包含 reasoning_text（默认 False）
            response_model: 可选 Pydantic BaseModel 类；设置后自动启用 JSON 模式，
                            并将解析结果存入返回字典的 ``parsed_object`` 字段。
                            解析失败时自动追加错误信息并重试一次。
            **overrides: 覆盖 provider 默认参数
            
        Returns:
            {
                "provider": str,
                "model": str,
                "final_text": str,
                "reasoning_text": str | None,  # 仅当 return_reasoning=True 或始终返回
                "parsed_object": BaseModel | None,  # 仅当 response_model 不为 None
                "raw": dict,
                "params": dict,
                "meta": {"usage": dict, "latency_ms": int, "refusal": bool}
            }
        """
        raise NotImplementedError


class DryRunChatClient(BaseChatClient):
    """
    Dry-run 模式客户端。
    不实际调用 API，返回模拟结构。
    """

    def __init__(self, config: ProviderConfig, log_store: Optional[RawLogStore] = None):
        self.config = config
        self.log_store = log_store

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Dict[str, Any]:
        resolved_model = self._resolve_model(model)
        merged_params = deep_merge(self.config.params, overrides)

        result = {
            "provider": self.config.name,
            "model": resolved_model,
            "final_text": f"[DRY_RUN] provider={self.config.name}, model={resolved_model}",
            "reasoning_text": None,
            "parsed_object": None,
            "raw": {
                "dry_run": True,
                "note": "This is a dry-run response. No actual API call was made.",
                "messages_count": len(messages),
            },
            "params": merged_params,
            "meta": {
                "usage": None,
                "latency_ms": 0,
                "refusal": None,
            },
        }

        # 记录日志
        if self.log_store:
            self.log_store.write({
                "timestamp": now_iso(),
                "provider": self.config.name,
                "model": resolved_model,
                "params": merged_params,
                "messages_digest": messages_digest(messages),
                "final_text": result["final_text"],
                "reasoning_text": None,
                "raw_response": result["raw"],
                "meta": result["meta"],
                "error": None,
                "dry_run": True,
            })

        return result

    def _resolve_model(self, model: Optional[str]) -> str:
        if model:
            return self.config.models.get(model, model)
        default = self.config.default_model
        return self.config.models.get(default, default)


class HTTPChatClient(BaseChatClient):
    """
    HTTP 实际调用客户端。
    根据 provider 类型选择 OpenAI-compatible 或 Anthropic 协议。
    """

    def __init__(
        self,
        config: ProviderConfig,
        provider: Provider,
        log_store: Optional[RawLogStore] = None,
        semaphore: Optional[threading.Semaphore] = None,
    ):
        self.config = config
        self.provider = provider
        self.log_store = log_store
        self._semaphore = semaphore

    @traceable(run_type="llm", name="llm.chat")
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        tools: Optional[List] = None,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Dict[str, Any]:
        resolved_model = self._resolve_model(model)
        timeout_override = overrides.pop("timeout_seconds", None) or overrides.pop("timeout", None)
        if timeout_override is not None:
            try:
                timeout_override = int(timeout_override)
                if timeout_override <= 0:
                    timeout_override = None
            except Exception:
                timeout_override = None
        merged_params = deep_merge(self.config.params, overrides)
        is_anthropic = self.config.is_anthropic()

        # 结构化输出：为 OpenAI-compat 协议注入 JSON 模式
        if response_model is not None and not is_anthropic:
            is_perplexity = "api.perplexity.ai" in (self.config.base_url or "")
            if is_perplexity:
                merged_params.setdefault("response_format", {
                    "type": "json_schema",
                    "json_schema": {"schema": response_model.model_json_schema()},
                })
            else:
                merged_params.setdefault("response_format", {"type": "json_object"})

        # 构建请求 payload
        if is_anthropic:
            payload = self._build_anthropic_payload(messages, resolved_model, merged_params, tools=tools)
        else:
            payload = self._build_openai_payload(messages, resolved_model, merged_params, tools=tools)

        # 发送请求（可选并发限流）
        start_time = time.time()
        error = None
        raw = {}

        def _do_request():
            return self.provider.request(payload, timeout=timeout_override)

        try:
            if self._semaphore:
                with self._semaphore:
                    raw = _do_request()
            else:
                raw = _do_request()
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            raise
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            # ── Observability: LLM 指标 ──
            if _obs_metrics:
                _prov = self.config.name
                _mod = resolved_model
                _obs_metrics.llm_requests_total.labels(provider=_prov, model=_mod).inc()
                _obs_metrics.llm_duration_seconds.labels(provider=_prov, model=_mod).observe(latency_ms / 1000.0)
                if error:
                    _obs_metrics.llm_errors_total.labels(provider=_prov, model=_mod).inc()
                # token 计量
                _norm = normalize_response(self.config.name, raw, is_anthropic) if raw else {}
                _usage = _norm.get("usage") or {}
                if _usage.get("prompt_tokens"):
                    _obs_metrics.llm_tokens_used.labels(provider=_prov, model=_mod, direction="input").inc(_usage["prompt_tokens"])
                if _usage.get("completion_tokens"):
                    _obs_metrics.llm_tokens_used.labels(provider=_prov, model=_mod, direction="output").inc(_usage["completion_tokens"])

            # 记录日志
            if self.log_store:
                normalized = normalize_response(self.config.name, raw, is_anthropic) if raw else {}
                self.log_store.write({
                    "timestamp": now_iso(),
                    "provider": self.config.name,
                    "model": resolved_model,
                    "params": merged_params,
                    "messages_digest": messages_digest(messages),
                    "final_text": normalized.get("final_text"),
                    "reasoning_text": normalized.get("reasoning_text"),
                    "raw_response": raw,
                    "meta": {
                        "usage": normalized.get("usage"),
                        "latency_ms": latency_ms,
                        "refusal": normalized.get("refusal"),
                    },
                    "error": error,
                })

        # 规范化响应
        normalized = normalize_response(self.config.name, raw, is_anthropic)
        latency_ms = int((time.time() - start_time) * 1000)

        result = {
            "provider": self.config.name,
            "model": resolved_model,
            "final_text": normalized["final_text"] or "",
            "reasoning_text": normalized["reasoning_text"],
            "raw": raw,
            "params": merged_params,
            "meta": {
                "usage": normalized["usage"],
                "latency_ms": latency_ms,
                "refusal": normalized["refusal"],
            },
        }

        # ── Function Calling: 解析 tool_calls ──
        if tools:
            from src.llm.tools import parse_tool_calls, has_tool_calls
            if has_tool_calls(raw, is_anthropic):
                result["tool_calls"] = parse_tool_calls(raw, is_anthropic)
            else:
                result["tool_calls"] = []

        # ── Structured Output: Pydantic 解析 + 自动重试一次 ──
        if response_model is not None:
            final_text = result.get("final_text") or ""
            try:
                result["parsed_object"] = response_model.model_validate_json(final_text)
            except Exception as val_err:
                retry_msgs = list(messages) + [
                    {"role": "assistant", "content": final_text},
                    {"role": "user", "content": (
                        f"Your previous response could not be parsed as JSON. "
                        f"Validation error: {val_err}. "
                        "Please return ONLY valid JSON matching the required schema, with no markdown."
                    )},
                ]
                retry_payload = (
                    self._build_anthropic_payload(retry_msgs, resolved_model, merged_params, tools=tools)
                    if is_anthropic
                    else self._build_openai_payload(retry_msgs, resolved_model, merged_params, tools=tools)
                )
                try:
                    if self._semaphore:
                        with self._semaphore:
                            retry_raw = self.provider.request(retry_payload, timeout=timeout_override)
                    else:
                        retry_raw = self.provider.request(retry_payload, timeout=timeout_override)
                    retry_norm = normalize_response(self.config.name, retry_raw, is_anthropic)
                    retry_text = retry_norm.get("final_text") or ""
                    result["final_text"] = retry_text
                    result["reasoning_text"] = retry_norm.get("reasoning_text")
                    result["raw"] = retry_raw
                    result["meta"]["usage"] = retry_norm.get("usage")
                    result["meta"]["refusal"] = retry_norm.get("refusal")
                    result["meta"]["latency_ms"] = int((time.time() - start_time) * 1000)
                    result["parsed_object"] = response_model.model_validate_json(retry_text)
                    _log.debug("Structured output validation succeeded on retry")
                except Exception as retry_err:
                    _log.warning("Structured output retry failed: %s", retry_err)
                    result["parsed_object"] = None

        return result

    def _resolve_model(self, model: Optional[str]) -> str:
        if model:
            return self.config.models.get(model, model)
        default = self.config.default_model
        return self.config.models.get(default, default)

    def _build_openai_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        """构建 OpenAI-compatible 请求 payload"""
        # Sanitize messages for stricter OpenAI-compatible providers (e.g. Moonshot/Kimi):
        # - Never send content=null
        # - Ensure assistant.tool_calls[*].id/function.arguments are valid
        # - Ensure tool messages carry a valid tool_call_id
        def _to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                return str(v)

        sanitized: List[Dict[str, Any]] = []
        pending_tool_ids: List[str] = []
        auto_idx = 0
        for msg in messages:
            role = str(msg.get("role") or "")
            normalized = dict(msg)
            normalized["role"] = role
            normalized["content"] = _to_text(normalized.get("content"))

            if role == "assistant" and isinstance(normalized.get("tool_calls"), list):
                tc_list = []
                for tc in normalized.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    tc_id = str(tc.get("id") or "")
                    if not tc_id:
                        tc_id = f"call_auto_{auto_idx}"
                        auto_idx += 1
                    fn_name = str(fn.get("name") or tc.get("name") or "")
                    fn_args = fn.get("arguments", {})
                    if not isinstance(fn_args, str):
                        fn_args = _to_text(fn_args)
                    tc_list.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args,
                            },
                        }
                    )
                    pending_tool_ids.append(tc_id)
                normalized["tool_calls"] = tc_list

            if role == "tool":
                tc_id = str(normalized.get("tool_call_id") or "")
                if not tc_id:
                    tc_id = pending_tool_ids.pop(0) if pending_tool_ids else f"call_auto_{auto_idx}"
                    auto_idx += 1
                normalized["tool_call_id"] = tc_id
                # Some providers are stricter and expect name on tool messages.
                if normalized.get("name") is None:
                    normalized["name"] = "tool"

            sanitized.append(normalized)
        payload = {
            "model": model,
            "messages": sanitized,
        }

        # ── Function Calling: 注入 tools ──
        if tools:
            from src.llm.tools import ToolDef
            openai_tools = [t.to_openai() if isinstance(t, ToolDef) else t for t in tools]
            payload["tools"] = openai_tools

        # 标准参数直接放入 payload
        standard_params = {
            "max_tokens", "max_completion_tokens", "temperature", "top_p",
            "frequency_penalty", "presence_penalty", "stop",
            "stream", "n", "response_format",
        }

        # Gemini OpenAI-compatible endpoint does not accept some params
        is_gemini = "generativelanguage.googleapis.com" in (self.config.base_url or "")
        for key, value in params.items():
            if is_gemini and key == "media_resolution":
                continue
            if key in standard_params:
                payload[key] = value
            else:
                # 非标准参数也放入（某些平台支持）
                payload[key] = value

        # Remove None-valued token fields — None means "let the API decide"
        if payload.get("max_tokens") is None:
            payload.pop("max_tokens", None)
        if payload.get("max_completion_tokens") is None:
            payload.pop("max_completion_tokens", None)

        # Reasoning/thinking models: avoid restricting chain-of-thought with a small limit.
        # Covers: OpenAI reasoning_effort, DeepSeek thinking.type, Kimi thinking.type,
        #         Qwen enable_thinking
        _is_thinking_openai_compat = (
            payload.get("reasoning_effort")
            or (payload.get("thinking") or {}).get("type") in ("enabled",)
            or payload.get("enable_thinking") is True
        )
        if _is_thinking_openai_compat:
            for key in ("max_tokens", "max_completion_tokens"):
                val = payload.get(key)
                if val is not None and val < 8000:
                    payload.pop(key, None)

        # OpenAI 新 API: 使用 max_completion_tokens
        is_openai = "api.openai.com" in (self.config.base_url or "")
        if is_openai and "max_tokens" in payload and "max_completion_tokens" not in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        return payload

    def _build_anthropic_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        """构建 Anthropic 请求 payload"""
        # 分离 system 消息
        system_content = None
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append(msg)

        payload = {
            "model": model,
            "messages": user_messages,
        }

        if system_content:
            payload["system"] = system_content

        # 合并参数
        for key, value in params.items():
            payload[key] = value

        # Anthropic API 必须提供大于 0 的整型 max_tokens。
        # Sonnet 4.6 / Opus 4.6 支持最高 128K output，给一个宽裕的默认值。
        thinking_cfg = payload.get("thinking") or {}
        thinking_type = thinking_cfg.get("type")
        is_thinking = thinking_type in ("enabled", "adaptive")

        if payload.get("max_tokens") is None:
            payload["max_tokens"] = 64000 if is_thinking else 16384

        # Extended thinking: max_tokens 必须 > budget_tokens，否则 API 报错
        if is_thinking and thinking_type == "enabled":
            budget = int(thinking_cfg.get("budget_tokens") or 10000)
            if payload["max_tokens"] <= budget:
                payload["max_tokens"] = budget + 8000

        # ── Function Calling: 注入 tools ──
        if tools:
            from src.llm.tools import ToolDef
            anthropic_tools = [t.to_anthropic() if isinstance(t, ToolDef) else t for t in tools]
            payload["tools"] = anthropic_tools

        return payload


# ============================================================
# LLMManager
# ============================================================

class LLMManager:
    """
    LLM 统一管理器。
    
    职责：
    - 加载配置
    - 获取/创建 ChatClient
    - 解析模型别名
    """

    def __init__(self, config: LLMConfig, log_store: Optional[RawLogStore] = None):
        self.config = config
        self.log_store = log_store or RawLogStore()
        self._clients: Dict[str, BaseChatClient] = {}
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._sem_lock = threading.Lock()

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMManager":
        """
        从 JSON 配置文件加载 LLMManager。
        
        Args:
            path: 配置文件路径（如 config/rag_config.json）
            
        Returns:
            LLMManager 实例
        """
        raw = load_json_with_local(path)
        llm_section = raw.get("llm", {})

        default = llm_section.get("default", "claude")
        dry_run = llm_section.get("dry_run", False)

        # ── 1. 加载 platforms（平台级 api_key + base_url）──
        platforms_raw = llm_section.get("platforms", {})
        platforms: Dict[str, PlatformConfig] = {}
        for pname, pcfg in platforms_raw.items():
            env_var = provider_env_var(pname)
            api_key = os.getenv(env_var) or pcfg.get("api_key", "")
            platforms[pname] = PlatformConfig(
                name=pname,
                api_key=api_key,
                base_url=pcfg.get("base_url", ""),
            )

        # ── 2. 加载 providers（变体配置，api_key/base_url 从 platform 继承）──
        providers_raw = llm_section.get("providers", {})
        providers: Dict[str, ProviderConfig] = {}
        for name, pcfg in providers_raw.items():
            platform_name = pcfg.get("platform", "")
            platform = platforms.get(platform_name)

            # api_key 解析优先级:
            #   provider 级环境变量 > provider 级 JSON > platform 环境变量 > platform JSON
            env_var = provider_env_var(name)
            api_key = os.getenv(env_var) or pcfg.get("api_key", "")
            if not api_key and platform:
                api_key = platform.api_key

            # base_url: provider 级 > platform 级
            base_url = pcfg.get("base_url", "")
            if not base_url and platform:
                base_url = platform.base_url

            providers[name] = ProviderConfig(
                name=name,
                api_key=api_key,
                base_url=base_url,
                default_model=pcfg.get("default_model", ""),
                platform=platform_name,
                models=pcfg.get("models", {}),
                params=pcfg.get("params", {}),
            )

        config = LLMConfig(
            default=default, dry_run=dry_run,
            platforms=platforms, providers=providers,
        )
        return cls(config)

    def get_provider_names(self) -> List[str]:
        """获取所有可用的 provider 名称列表"""
        return list(self.config.providers.keys())

    def get_client(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> BaseChatClient:
        """
        获取指定 provider 的 ChatClient。
        
        Args:
            provider: provider 名称（默认使用 config.default）
            api_key: 可选，覆盖配置中的 api_key
            
        Returns:
            BaseChatClient 实例
        """
        provider_name = provider or self.config.default

        if provider_name not in self.config.providers:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {self.get_provider_names()}")

        pcfg = self.config.providers[provider_name]

        # api_key 优先级: 参数 > 环境变量 > JSON（已在 from_json 处理环境变量）
        if api_key:
            pcfg = ProviderConfig(
                name=pcfg.name,
                api_key=api_key,
                base_url=pcfg.base_url,
                default_model=pcfg.default_model,
                models=pcfg.models,
                params=pcfg.params,
            )

        # dry_run 模式
        if self.config.dry_run:
            return DryRunChatClient(pcfg, self.log_store)

        # 检查 api_key
        if not pcfg.api_key or pcfg.api_key in ("sk-xxx", "sk-ant-xxx", "AIzxxx"):
            raise ValueError(
                f"Invalid or missing API key for provider '{provider_name}'. "
                f"Set via environment variable {provider_env_var(provider_name)} or in config file. "
                f"Current key: {mask_secret(pcfg.api_key)}"
            )

        # 创建实际 Provider
        if pcfg.is_anthropic():
            http_provider = AnthropicProvider(pcfg)
        else:
            http_provider = OpenAICompatProvider(pcfg)

        semaphore = None
        if settings and hasattr(settings, "perf_llm"):
            max_conc = getattr(settings.perf_llm, "max_concurrent_per_provider", 5) or 5
            with self._sem_lock:
                if provider_name not in self._semaphores:
                    self._semaphores[provider_name] = threading.Semaphore(max_conc)
                semaphore = self._semaphores[provider_name]

        return HTTPChatClient(pcfg, http_provider, self.log_store, semaphore=semaphore)

    # ── Thinking → Base provider mapping ──
    # "openai-thinking" → "openai", "claude-thinking" → "claude", etc.
    _THINKING_DOWNGRADE: Dict[str, str] = {}  # populated lazily

    def _build_downgrade_map(self) -> None:
        """Build a {thinking_provider -> base_provider} mapping from config."""
        self._THINKING_DOWNGRADE.clear()
        for name in self.config.providers:
            if "-thinking" in name:
                base = name.replace("-thinking", "")
                if base in self.config.providers:
                    self._THINKING_DOWNGRADE[name] = base

    def get_lite_client(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseChatClient:
        """
        获取轻量级 client —— 自动将 thinking 变体降级到同平台基础 provider。

        用于路由分类、查询生成、JSON 解析等不需要 reasoning 能力的场景。
        例如: "claude-thinking" → "claude", "openai-thinking" → "openai"
        非 thinking 变体直接透传。
        """
        provider_name = provider or self.config.default
        if not self._THINKING_DOWNGRADE:
            self._build_downgrade_map()
        lite_provider = self._THINKING_DOWNGRADE.get(provider_name, provider_name)
        return self.get_client(lite_provider, api_key=api_key)

    def get_ultra_lite_client(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseChatClient:
        """
        获取极轻量级 client (Ultra Lite)
        专门用于长文本压缩、简单抽取等超高上下文、低逻辑要求的任务。
        如果未指定 provider，默认回退到最高性价比模型 (openai-mini)。
        """
        provider_name = provider or "openai-mini"
        if provider_name not in self.config.providers:
            provider_name = self.config.default
            
        if not self._THINKING_DOWNGRADE:
            self._build_downgrade_map()
        ultra_lite_provider = self._THINKING_DOWNGRADE.get(provider_name, provider_name)
        return self.get_client(ultra_lite_provider, api_key=api_key)

    def resolve_model(self, provider: str, model: Optional[str] = None) -> str:
        """
        解析模型名。
        
        优先级:
        1. model 参数（若存在于 models 映射则使用映射值）
        2. provider.default_model（若存在于 models 映射则使用映射值）
        
        Args:
            provider: provider 名称
            model: 可选的模型名或别名
            
        Returns:
            解析后的实际模型名
        """
        if provider not in self.config.providers:
            raise ValueError(f"Unknown provider: {provider}")

        pcfg = self.config.providers[provider]
        models = pcfg.models

        if model:
            return models.get(model, model)

        default = pcfg.default_model
        return models.get(default, default)

    def is_available(self, provider: str) -> bool:
        """检查 provider 是否有有效的 API key"""
        if provider not in self.config.providers:
            return False
        pcfg = self.config.providers[provider]
        key = pcfg.api_key
        return bool(key and key not in ("sk-xxx", "sk-ant-xxx", "AIzxxx", ""))

    def cleanup_logs(
        self,
        max_age_days: int = LOG_MAX_AGE_DAYS,
        max_total_mb: int = LOG_MAX_TOTAL_MB
    ) -> Dict[str, Any]:
        """清理日志文件"""
        return self.log_store.cleanup(max_age_days, max_total_mb)


# ============================================================
# Convenience Functions
# ============================================================

# 全局单例（延迟初始化）
_manager: Optional[LLMManager] = None


def get_manager(config_path: Optional[str | Path] = None) -> LLMManager:
    """
    获取全局 LLMManager 单例。
    
    Args:
        config_path: 配置文件路径（仅首次调用时生效）
        
    Returns:
        LLMManager 实例
    """
    global _manager
    if _manager is None:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "rag_config.json"
        _manager = LLMManager.from_json(config_path)
    return _manager


def chat(
    messages: List[Dict[str, Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：快速调用 LLM。
    
    Args:
        messages: 消息列表
        provider: provider 名称（默认使用配置中的 default）
        model: 模型名或别名
        **kwargs: 传递给 chat() 的其他参数
        
    Returns:
        响应字典
    """
    manager = get_manager()
    client = manager.get_client(provider)
    return client.chat(messages, model=model, **kwargs)
