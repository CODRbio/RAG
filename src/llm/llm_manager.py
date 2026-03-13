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

import os
import json
import threading
import time
import hashlib
import base64

from src.log import get_logger as _get_logger
_log = _get_logger(__name__)

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
from typing import Dict, Any, Iterator, List, Optional
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
DEFAULT_TIMEOUT = 180  # seconds; 与 config performance.llm.timeout_seconds 一致，无 config 时使用
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


def _is_openai_api(base_url: str) -> bool:
    return "api.openai.com" in (base_url or "")


def _is_gemini_api(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in (base_url or "")


def _gemini_native_base_url(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/openai"):
        root = root[: -len("/openai")]
    return root


def _is_qwen_api(base_url: str) -> bool:
    return "dashscope.aliyuncs.com" in (base_url or "") or "dashscope-intl.aliyuncs.com" in (base_url or "")


def _is_perplexity_api(base_url: str) -> bool:
    return "api.perplexity.ai" in (base_url or "")


def _qwen_responses_url(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/compatible-mode/v1"):
        prefix = root[: -len("/compatible-mode/v1")]
        return f"{prefix}/api/v2/apps/protocols/compatible-mode/v1/responses"
    return f"{root}/responses"


def _is_openai_reasoning_model(model: Optional[str]) -> bool:
    m = (model or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _apply_openai_compat_best_practices(
    payload: Dict[str, Any],
    *,
    base_url: str,
    provider_name: str,
    model: str,
    has_tools: bool,
) -> Dict[str, Any]:
    """
    Normalize OpenAI-compatible payloads according to provider/model capabilities.

    Current policy:
    - OpenAI GPT reasoning models: when using /chat/completions with tools, drop
      reasoning_effort because OpenAI requires Responses API for that combo.
    - OpenAI non-reasoning models: drop reasoning_effort entirely.
    - Gemini OpenAI-compatible API: normalize `minimal` -> `low` for
      reasoning_effort to avoid known compatibility quirks.
    """
    out = dict(payload)

    if _is_perplexity_api(base_url):
        allowed_keys = {
            "model",
            "messages",
            "max_tokens",
            "stream",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "web_search_options",
            "search_mode",
            "return_images",
            "return_related_questions",
            "enable_search_classifier",
            "disable_search",
            "search_domain_filter",
            "search_language_filter",
            "search_recency_filter",
            "search_after_date_filter",
            "search_before_date_filter",
            "last_updated_before_filter",
            "last_updated_after_filter",
            "image_format_filter",
            "image_domain_filter",
            "stream_mode",
            "reasoning_effort",
        }
        dropped_keys = sorted(k for k in out.keys() if k not in allowed_keys)
        if dropped_keys:
            _log.info(
                "Perplexity compat: dropped unsupported fields provider=%s model=%s keys=%s",
                provider_name,
                model,
                dropped_keys,
            )
        out = {k: v for k, v in out.items() if k in allowed_keys}

        messages = []
        for msg in out.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            messages.append({"role": role, "content": msg.get("content")})
        out["messages"] = messages

        response_format = out.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object"},
                },
            }
            _log.info(
                "Perplexity compat: upgraded response_format json_object -> json_schema for provider=%s model=%s",
                provider_name,
                model,
            )

        max_tokens = out.get("max_tokens")
        try:
            if max_tokens is not None:
                out["max_tokens"] = min(int(max_tokens), 128_000)
        except Exception:
            out.pop("max_tokens", None)

        return out

    reasoning_effort = out.get("reasoning_effort")
    if not reasoning_effort:
        return out

    if _is_openai_api(base_url):
        if not _is_openai_reasoning_model(model):
            out.pop("reasoning_effort", None)
            _log.info(
                "OpenAI compat: dropped reasoning_effort for non-reasoning model=%s provider=%s",
                model,
                provider_name,
            )
            return out
        if has_tools:
            out.pop("reasoning_effort", None)
            _log.info(
                "OpenAI compat: dropped reasoning_effort for model=%s provider=%s because /chat/completions + tools is incompatible; Responses API is preferred",
                model,
                provider_name,
            )
            return out

    if _is_gemini_api(base_url):
        eff = str(reasoning_effort).strip().lower()
        if eff == "minimal":
            out["reasoning_effort"] = "low"
            _log.info(
                "Gemini compat: normalized reasoning_effort minimal -> low for model=%s provider=%s",
                model,
                provider_name,
            )

    return out


def _should_use_openai_responses_api(
    *,
    base_url: str,
    model: str,
    has_tools: bool,
    params: Dict[str, Any],
    response_model: Optional[Any],
) -> bool:
    if not _is_openai_api(base_url):
        return False
    if response_model is not None:
        return False
    if not _is_openai_reasoning_model(model):
        return False
    return bool(has_tools or params.get("reasoning_effort"))


def _should_use_qwen_responses_api(
    *,
    base_url: str,
    has_tools: bool,
    params: Dict[str, Any],
    response_model: Optional[Any],
) -> bool:
    if not _is_qwen_api(base_url):
        return False
    if response_model is not None:
        return False
    return bool(has_tools or params.get("enable_thinking") is True)


def _can_use_gemini_native_api(
    *,
    base_url: str,
    messages: List[Dict[str, Any]],
    response_model: Optional[Any],
) -> bool:
    if not _is_gemini_api(base_url):
        return False
    if response_model is not None:
        return False
    for msg in messages:
        content = msg.get("content")
        if content is None or isinstance(content, str):
            continue
        return False
    return True


def _is_claude_opus_46(model: Optional[str]) -> bool:
    return "claude-opus-4-6" in (model or "").lower()


def _infer_anthropic_adaptive_effort(budget_tokens: Optional[int]) -> str:
    try:
        budget = int(budget_tokens or 0)
    except Exception:
        budget = 0
    if budget <= 8000:
        return "low"
    if budget <= 16000:
        return "medium"
    return "high"


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

    def request_stream(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Best-effort streaming API. Providers may raise NotImplementedError."""
        raise NotImplementedError


def _llm_perf_timeout() -> int:
    if settings and hasattr(settings, "perf_llm"):
        return getattr(settings.perf_llm, "timeout_seconds", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT
    return DEFAULT_TIMEOUT


def _normalize_timeout(timeout: Any) -> Any:
    if timeout is None:
        return _llm_perf_timeout()
    if isinstance(timeout, tuple):
        return timeout
    return int(timeout)


def _timeout_for_model(model: Optional[str] = None, base: Optional[int] = None) -> int:
    """按模型名返回合适的请求超时秒数。

    规则：
    - sonar-deep-research          → 900s  (15 min，后台联网研究流程)
    - sonar-reasoning-pro / *-pro  → 300s  (推理模型，给充足余量)
    - sonar* / perplexity*         → 180s  (普通 Sonar 模型)
    - 其他                         → base 或 perf_llm.timeout_seconds 或 DEFAULT_TIMEOUT
    """
    m = (model or "").lower()
    if "deep-research" in m:
        return 900
    if "reasoning-pro" in m or ("reasoning" in m and "sonar" in m):
        return 300
    if "sonar" in m or "perplexity" in m:
        return 180
    return int(base if base is not None else _llm_perf_timeout())


def _soft_timeout_timer(hard_timeout: float, label: str, soft_ratio: float = 0.6) -> threading.Timer:
    """返回一个已启动的 daemon Timer，在 hard_timeout * soft_ratio 秒后打印 WARNING 日志。

    这是"软超时"的日志层：到点只警告不 kill，真正的 kill 仍由 requests timeout 完成。
    调用方负责在请求完成后调用 timer.cancel() 防止无谓日志。
    """
    soft_secs = hard_timeout * soft_ratio

    def _warn() -> None:
        _log.warning(
            "[soft_timeout] %s 已等待 %.0f 秒（软阈值 %.0f s / 硬上限 %.0f s），仍在运行中",
            label, soft_secs, soft_secs, hard_timeout,
        )

    t = threading.Timer(soft_secs, _warn)
    t.daemon = True
    t.start()
    return t


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
    timeout: Union[int, float, tuple],
    **kwargs: Any,
) -> requests.Response:
    max_retries, backoff = _llm_perf_retry()
    last_err = None
    for attempt in range(max_retries + 1):
        label = f"LLM request attempt={attempt+1} url={url}"
        hard_timeout = float(timeout[0] + timeout[1]) if isinstance(timeout, tuple) else float(timeout)
        soft_timer = _soft_timeout_timer(hard_timeout, label)
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            soft_timer.cancel()
            if resp.status_code in (429, 500, 503) and attempt < max_retries:
                time.sleep(backoff ** attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            soft_timer.cancel()
            last_err = e
            if e.response.status_code not in (429, 500, 503) or attempt >= max_retries:
                raise
            time.sleep(backoff ** attempt)
        except Exception as e:
            soft_timer.cancel()
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(backoff ** attempt)
    if last_err:
        raise last_err
    raise RuntimeError("request_with_retry failed")


def _qwen_image_generation_url(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/compatible-mode/v1"):
        prefix = root[: -len("/compatible-mode/v1")]
        return f"{prefix}/api/v1/services/aigc/multimodal-generation/generation"
    return f"{root}/api/v1/services/aigc/multimodal-generation/generation"


def _normalize_image_size(size: str, *, provider: str) -> str:
    normalized = (size or "1024x1024").strip().lower().replace("×", "x")
    if provider == "qwen":
        return normalized.replace("x", "*")
    return normalized


def _extract_openai_image_bytes(data: Dict[str, Any]) -> bytes:
    items = data.get("data") or []
    if not items:
        raise ValueError(f"Image response contained no data items. Keys: {list(data.keys())}")
    first = items[0] or {}
    b64_payload = first.get("b64_json")
    if isinstance(b64_payload, str) and b64_payload:
        return base64.b64decode(b64_payload)
    raise ValueError(f"OpenAI image response contained no b64_json payload. Keys: {list(first.keys())}")


def _extract_qwen_image_url(data: Dict[str, Any]) -> str:
    output = data.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        for item in message.get("content") or []:
            if isinstance(item, dict) and isinstance(item.get("image"), str) and item.get("image"):
                return item["image"]
    results = output.get("results") or []
    if results:
        first = results[0] or {}
        for key in ("url", "image", "image_url"):
            if isinstance(first.get(key), str) and first.get(key):
                return first[key]
    raise ValueError(f"Qwen image response contained no image URL. Keys: {list(output.keys())}")


def _request_stream_no_retry(
    session: requests.Session,
    method: str,
    url: str,
    timeout: Union[int, float, tuple],
    **kwargs: Any,
) -> requests.Response:
    resp = session.request(method, url, timeout=timeout, stream=True, **kwargs)
    resp.raise_for_status()
    return resp


def _iter_sse_events(resp: requests.Response) -> Iterator[Dict[str, Any]]:
    event_name = "message"
    data_lines: List[str] = []
    event_id = ""

    # Force UTF-8: SSE APIs (e.g. Gemini native streamGenerateContent) often omit
    # charset in "Content-Type: text/event-stream".  requests then defaults to
    # ISO-8859-1 (HTTP/1.1 spec default for text/*), which garbles multi-byte
    # UTF-8 characters (Chinese, etc.) when decode_unicode=True is used.
    resp.encoding = "utf-8"

    for raw_line in resp.iter_lines(decode_unicode=True):
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                payload: Any = data_str
                if data_str != "[DONE]":
                    try:
                        payload = json.loads(data_str)
                    except Exception:
                        payload = data_str
                yield {"event": event_name or "message", "data": payload, "id": event_id}
            event_name = "message"
            data_lines = []
            event_id = ""
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue
        if line.startswith("id:"):
            event_id = line[3:].strip()

    if data_lines:
        data_str = "\n".join(data_lines)
        payload = data_str
        if data_str != "[DONE]":
            try:
                payload = json.loads(data_str)
            except Exception:
                payload = data_str
        yield {"event": event_name or "message", "data": payload, "id": event_id}


def _chunk_text_for_stream(text: str, chunk_size: int = 80) -> Iterator[str]:
    if not text:
        return
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


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
        api_mode = str(payload.pop("_api_mode", "chat_completions"))
        if api_mode == "responses":
            if _is_qwen_api(self.config.base_url):
                url = _qwen_responses_url(self.config.base_url)
            else:
                url = f"{self.config.base_url.rstrip('/')}/responses"
        else:
            url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        timeout = _normalize_timeout(timeout)
        resp = _request_with_retry(
            self._session, "POST", url, timeout,
            headers=headers, json=payload,
        )
        data = resp.json()
        if isinstance(data, dict):
            data["_api_mode"] = api_mode
        return data

    def request_stream(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        stream_payload = dict(payload)
        api_mode = str(stream_payload.pop("_api_mode", "chat_completions"))
        if api_mode == "responses":
            if _is_qwen_api(self.config.base_url):
                url = _qwen_responses_url(self.config.base_url)
            else:
                url = f"{self.config.base_url.rstrip('/')}/responses"
        else:
            url = f"{self.config.base_url.rstrip('/')}/chat/completions"
            stream_options = stream_payload.get("stream_options")
            if not isinstance(stream_options, dict):
                stream_options = {}
            stream_options.setdefault("include_usage", True)
            stream_payload["stream_options"] = stream_options
        stream_payload["stream"] = True
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        timeout = _normalize_timeout(timeout)
        with _request_stream_no_retry(
            self._session,
            "POST",
            url,
            timeout,
            headers=headers,
            json=stream_payload,
        ) as resp:
            for item in _iter_sse_events(resp):
                yield {**item, "api_mode": api_mode}


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
        if self.config.params.get("enable_prompt_cache"):
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        timeout = _normalize_timeout(timeout)
        resp = _request_with_retry(
            self._session, "POST", url, timeout,
            headers=headers, json=payload,
        )
        return resp.json()

    def request_stream(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        url = f"{self.config.base_url.rstrip('/')}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.config.params.get("enable_prompt_cache"):
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        timeout = _normalize_timeout(timeout)
        with _request_stream_no_retry(
            self._session,
            "POST",
            url,
            timeout,
            headers=headers,
            json=stream_payload,
        ) as resp:
            for item in _iter_sse_events(resp):
                yield item


class GeminiNativeProvider(Provider):
    """
    Gemini native API provider with automatic fallback to OpenAI-compatible mode.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._session = requests.Session()
        self._fallback = OpenAICompatProvider(config)

    @staticmethod
    def _adapt_native_response(raw: Dict[str, Any]) -> Dict[str, Any]:
        candidates = raw.get("candidates") or []
        usage_meta = raw.get("usageMetadata") or raw.get("usage_metadata") or {}
        content_text_parts: List[str] = []
        reasoning_text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        finish_reason = ""
        if candidates:
            cand = candidates[0] or {}
            finish_reason = str(cand.get("finishReason") or cand.get("finish_reason") or "").lower()
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    content_text_parts.append(part.get("text", ""))
                if isinstance(part.get("thought"), str):
                    reasoning_text_parts.append(part.get("thought", ""))
                elif part.get("thought") is True and isinstance(part.get("text"), str):
                    # In some versions, thought is a boolean flag on the text part
                    reasoning_text_parts.append(part.get("text", ""))
                    if content_text_parts and content_text_parts[-1] == part.get("text"):
                        content_text_parts.pop()

                if isinstance(part.get("functionCall"), dict):
                    fc = part.get("functionCall") or {}
                    tool_calls.append(
                        {
                            "id": fc.get("id") or fc.get("callId") or f"gemini_call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False),
                            },
                        }
                    )
        return {
            "_api_mode": "gemini_native",
            "usage": {
                "prompt_tokens": usage_meta.get("promptTokenCount") or usage_meta.get("prompt_token_count"),
                "completion_tokens": usage_meta.get("candidatesTokenCount") or usage_meta.get("candidates_token_count"),
                "total_tokens": usage_meta.get("totalTokenCount") or usage_meta.get("total_token_count"),
                "reasoning_tokens": usage_meta.get("thoughtsTokenCount") or usage_meta.get("thoughts_token_count"),
            },
            "choices": [
                {
                    "finish_reason": "tool_calls" if tool_calls else finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_text_parts),
                        "reasoning": "".join(reasoning_text_parts) if reasoning_text_parts else None,
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "candidates": candidates,
            "raw_native": raw,
        }

    def request(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        if str(payload.get("_api_mode") or "") != "gemini_native":
            return self._fallback.request(payload, timeout=timeout)

        fallback_payload = payload.pop("_fallback_payload", None)
        api_mode = payload.pop("_api_mode", None)
        model = str(payload.pop("_model_id", payload.get("model") or ""))
        payload.pop("model", None)
        url = f"{_gemini_native_base_url(self.config.base_url)}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }
        timeout = _normalize_timeout(timeout)
        try:
            resp = _request_with_retry(
                self._session,
                "POST",
                url,
                timeout,
                headers=headers,
                json=payload,
            )
            data = resp.json()
            adapted = self._adapt_native_response(data)
            adapted["_api_mode"] = api_mode or "gemini_native"
            return adapted
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if fallback_payload:
                _log.warning(
                    "Gemini native request failed (HTTP %s), falling back to compat mode: %s",
                    status, e,
                )
                return self._fallback.request(fallback_payload, timeout=timeout)
            if status is not None and 400 <= status < 500 and status != 429:
                try:
                    _err_body = e.response.text[:500] if e.response is not None else ""
                except Exception:
                    _err_body = ""
                msg = (
                    f"Gemini native API error (no fallback): {status} for url={url!r} "
                    f"provider={self.config.name!r} model={model!r}. "
                    f"response_body={_err_body!r}"
                )
                _log.warning("%s", msg)
                raise RuntimeError(msg) from e
            raise
        except Exception as e:
            if fallback_payload:
                _log.warning("Gemini native request failed, falling back to compat mode: %s", e)
                return self._fallback.request(fallback_payload, timeout=timeout)
            raise

    def request_stream(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        if str(payload.get("_api_mode") or "") != "gemini_native":
            yield from self._fallback.request_stream(payload, timeout=timeout)
            return

        fallback_payload = payload.pop("_fallback_payload", None)
        api_mode = payload.pop("_api_mode", None)
        model = str(payload.pop("_model_id", payload.get("model") or ""))
        payload.pop("model", None)
        # Use alt=sse for server-sent events
        url = f"{_gemini_native_base_url(self.config.base_url)}/models/{model}:streamGenerateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }
        timeout = _normalize_timeout(timeout)
        try:
            with _request_stream_no_retry(
                self._session,
                "POST",
                url,
                timeout,
                headers=headers,
                json=payload,
            ) as resp:
                for item in _iter_sse_events(resp):
                    yield {**item, "_api_mode": api_mode or "gemini_native"}
        except Exception as e:
            if fallback_payload:
                _log.warning("Gemini native stream failed, falling back to compat mode: %s", e)
                yield from self._fallback.request_stream(fallback_payload, timeout=timeout)
                return
            raise



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
        "citations": None,
        "search_results": None,
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
    result = {
        "final_text": None,
        "reasoning_text": None,
        "usage": None,
        "refusal": None,
        "citations": None,
        "search_results": None,
    }

    # usage
    result["usage"] = raw.get("usage")

    # Perplexity API: top-level citations (URLs) and search_results (title, url, date, snippet)
    citations = raw.get("citations")
    if citations is not None and isinstance(citations, list):
        result["citations"] = [str(u) for u in citations if u]
    search_results = raw.get("search_results")
    if search_results is not None and isinstance(search_results, list):
        result["search_results"] = search_results

    if (raw.get("_api_mode") or "") == "responses" or raw.get("output"):
        text_parts = []
        reasoning_parts = []
        for item in raw.get("output") or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                for block in item.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in ("output_text", "text"):
                        text_parts.append(block.get("text", ""))
            elif item_type == "reasoning":
                for block in item.get("summary") or []:
                    if isinstance(block, dict) and block.get("type") == "summary_text":
                        reasoning_parts.append(block.get("text", ""))
        result["final_text"] = "".join(text_parts) if text_parts else (raw.get("output_text") or None)
        if reasoning_parts:
            result["reasoning_text"] = "".join(reasoning_parts)
        return result

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

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Iterator[Dict[str, Any]]:
        """Stream normalized chat events: `text_delta` and terminal `completed`."""
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

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Iterator[Dict[str, Any]]:
        result = self.chat(
            messages,
            model=model,
            return_reasoning=return_reasoning,
            response_model=response_model,
            **overrides,
        )
        final_text = result.get("final_text") or ""
        for chunk in _chunk_text_for_stream(final_text):
            yield {"type": "text_delta", "delta": chunk}
        yield {"type": "completed", "response": result}

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
            payload = self._build_openai_payload(
                messages,
                resolved_model,
                merged_params,
                tools=tools,
                response_model=response_model,
            )

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
                    else self._build_openai_payload(
                        retry_msgs,
                        resolved_model,
                        merged_params,
                        tools=tools,
                        response_model=response_model,
                    )
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

    @traceable(run_type="llm", name="llm.stream_chat")
    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        return_reasoning: bool = False,
        tools: Optional[List] = None,
        response_model: Optional[Any] = None,
        **overrides
    ) -> Iterator[Dict[str, Any]]:
        resolved_model = self._resolve_model(model)
        timeout_override = overrides.pop("timeout_seconds", None) or overrides.pop("timeout", None)
        if timeout_override is not None:
            try:
                timeout_override = int(timeout_override)
                if timeout_override <= 0:
                    timeout_override = None
            except Exception:
                timeout_override = None
        # 流量感知超时：idle_timeout_seconds 指每两个 chunk 之间的最大等待秒数。
        # 只要 token 持续生成，总调用时长不受限制；空闲超过此值才视为超时。
        # 转换为 requests 支持的 (connect_timeout, read_timeout) 元组格式。
        idle_timeout = overrides.pop("idle_timeout_seconds", None)
        if idle_timeout is not None and timeout_override is None:
            try:
                idle_secs = int(idle_timeout)
                if idle_secs > 0:
                    timeout_override = (30, idle_secs)
            except Exception:
                pass
        merged_params = deep_merge(self.config.params, overrides)
        is_anthropic = self.config.is_anthropic()

        if response_model is not None:
            final_result = self.chat(
                messages,
                model=model,
                return_reasoning=return_reasoning,
                tools=tools,
                response_model=response_model,
                **overrides,
            )
            for chunk in _chunk_text_for_stream(final_result.get("final_text") or ""):
                yield {"type": "text_delta", "delta": chunk}
            yield {"type": "completed", "response": final_result}
            return

        if is_anthropic:
            payload = self._build_anthropic_payload(messages, resolved_model, merged_params, tools=tools)
        else:
            payload = self._build_openai_payload(
                messages,
                resolved_model,
                merged_params,
                tools=tools,
                response_model=response_model,
            )

        start_time = time.time()
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        final_raw: Dict[str, Any] = {}
        usage: Dict[str, Any] | None = None
        refusal: Optional[bool] = None
        error: Optional[str] = None

        try:
            stream_iter = self.provider.request_stream(payload, timeout=timeout_override)
            if self._semaphore:
                with self._semaphore:
                    for event in self._normalize_stream_events(stream_iter, payload, is_anthropic):
                        if event["type"] == "text_delta":
                            delta = str(event.get("delta") or "")
                            if delta:
                                text_parts.append(delta)
                        elif event["type"] == "reasoning_delta" and return_reasoning:
                            reasoning_parts.append(str(event.get("delta") or ""))
                        elif event["type"] == "completed":
                            final_raw = dict(event.get("raw") or {})
                            usage = event.get("usage") or usage
                            refusal = event.get("refusal")
                            continue
                        yield event
            else:
                for event in self._normalize_stream_events(stream_iter, payload, is_anthropic):
                    if event["type"] == "text_delta":
                        delta = str(event.get("delta") or "")
                        if delta:
                            text_parts.append(delta)
                    elif event["type"] == "reasoning_delta" and return_reasoning:
                        reasoning_parts.append(str(event.get("delta") or ""))
                    elif event["type"] == "completed":
                        final_raw = dict(event.get("raw") or {})
                        usage = event.get("usage") or usage
                        refusal = event.get("refusal")
                        continue
                    yield event
        except Exception as exc:
            error = str(exc)
            _log.info("LLM native stream unavailable for provider=%s model=%s, falling back: %s", self.config.name, resolved_model, exc)
            fallback_result = self.chat(
                messages,
                model=model,
                return_reasoning=return_reasoning,
                tools=tools,
                response_model=response_model,
                **overrides,
            )
            for chunk in _chunk_text_for_stream(fallback_result.get("final_text") or ""):
                yield {"type": "text_delta", "delta": chunk}
            error = None
            yield {"type": "completed", "response": fallback_result}
            return
        finally:
            latency_ms = int((time.time() - start_time) * 1000)
            if _obs_metrics:
                _prov = self.config.name
                _mod = resolved_model
                _obs_metrics.llm_requests_total.labels(provider=_prov, model=_mod).inc()
                _obs_metrics.llm_duration_seconds.labels(provider=_prov, model=_mod).observe(latency_ms / 1000.0)
                if error:
                    _obs_metrics.llm_errors_total.labels(provider=_prov, model=_mod).inc()
                if usage and usage.get("prompt_tokens"):
                    _obs_metrics.llm_tokens_used.labels(provider=_prov, model=_mod, direction="input").inc(usage["prompt_tokens"])
                if usage and usage.get("completion_tokens"):
                    _obs_metrics.llm_tokens_used.labels(provider=_prov, model=_mod, direction="output").inc(usage["completion_tokens"])
            if self.log_store:
                self.log_store.write({
                    "timestamp": now_iso(),
                    "provider": self.config.name,
                    "model": resolved_model,
                    "params": merged_params,
                    "messages_digest": messages_digest(messages),
                    "final_text": "".join(text_parts),
                    "reasoning_text": "".join(reasoning_parts) if reasoning_parts else None,
                    "raw_response": final_raw,
                    "meta": {
                        "usage": usage,
                        "latency_ms": latency_ms,
                        "refusal": refusal,
                    },
                    "error": error,
                })
        yield {
            "type": "completed",
            "response": {
                "provider": self.config.name,
                "model": resolved_model,
                "final_text": "".join(text_parts),
                "reasoning_text": "".join(reasoning_parts) if reasoning_parts and return_reasoning else None,
                "raw": final_raw,
                "params": merged_params,
                "meta": {
                    "usage": usage,
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "refusal": refusal,
                },
            },
        }

    def _normalize_stream_events(
        self,
        stream_iter: Iterator[Dict[str, Any]],
        payload: Dict[str, Any],
        is_anthropic: bool,
    ) -> Iterator[Dict[str, Any]]:
        if is_anthropic:
            yield from self._normalize_anthropic_stream(stream_iter)
            return
        if isinstance(self.provider, GeminiNativeProvider) and str(payload.get("_api_mode")) == "gemini_native":
            yield from self._normalize_gemini_stream(stream_iter, payload)
            return
        yield from self._normalize_openai_stream(stream_iter, payload)

    def _normalize_gemini_stream(
        self,
        stream_iter: Iterator[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        usage: Dict[str, Any] | None = None
        final_raw: Dict[str, Any] = {}
        saw_completed = False

        for event in stream_iter:
            data = event.get("data")
            if not isinstance(data, dict):
                continue

            final_raw = data
            candidates = data.get("candidates") or []
            usage_meta = data.get("usageMetadata") or data.get("usage_metadata")
            if usage_meta:
                usage = {
                    "prompt_tokens": usage_meta.get("promptTokenCount") or usage_meta.get("prompt_token_count"),
                    "completion_tokens": usage_meta.get("candidatesTokenCount") or usage_meta.get("candidates_token_count"),
                    "total_tokens": usage_meta.get("totalTokenCount") or usage_meta.get("total_token_count"),
                    "reasoning_tokens": usage_meta.get("thoughtsTokenCount") or usage_meta.get("thoughts_token_count"),
                }

            for cand in candidates:
                content = cand.get("content") or {}
                parts = content.get("parts") or []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        # Some versions use a boolean thought flag on the text part
                        if part.get("thought") is True:
                            yield {"type": "reasoning_delta", "delta": part["text"]}
                        else:
                            yield {"type": "text_delta", "delta": part["text"]}
                    if "thought" in part and not isinstance(part["thought"], bool):
                        # Some versions use a separate thought field for the string
                        yield {"type": "reasoning_delta", "delta": part["thought"]}

                finish_reason = str(cand.get("finishReason") or cand.get("finish_reason") or "").upper()
                if finish_reason and finish_reason != "FINISH_REASON_UNSPECIFIED" and not saw_completed:
                    saw_completed = True
                    yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": False}

        if not saw_completed:
            yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": False}

    def _normalize_openai_stream(
        self,
        stream_iter: Iterator[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        api_mode = str(payload.get("_api_mode") or "chat_completions")
        usage: Dict[str, Any] | None = None
        final_raw: Dict[str, Any] = {}
        refusal: Optional[bool] = None
        saw_completed = False

        for event in stream_iter:
            name = str(event.get("event") or "message")
            data = event.get("data")
            if data == "[DONE]":
                saw_completed = True
                yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}
                continue
            if not isinstance(data, dict):
                continue

            if api_mode == "responses":
                if name == "response.output_text.delta":
                    delta = str(data.get("delta") or "")
                    if delta:
                        yield {"type": "text_delta", "delta": delta}
                elif name == "response.completed":
                    final_raw = dict(data.get("response") or data)
                    normalized = normalize_response(self.config.name, final_raw, False)
                    usage = normalized.get("usage") or usage
                    refusal = normalized.get("refusal")
                    saw_completed = True
                    yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}
                elif name == "error":
                    raise RuntimeError(str((data.get("error") or {}).get("message") or data.get("message") or "stream error"))
                continue

            # 持续更新 final_raw；对于 Perplexity，最后一个含 finish_reason 的
            # chunk 里带有 citations / search_results 顶层字段，必须保留。
            final_raw = dict(data)
            choices = data.get("choices") or []
            if data.get("usage"):
                usage = data.get("usage")
            for choice in choices:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thought")
                if isinstance(content, str) and content:
                    yield {"type": "text_delta", "delta": content}
                if isinstance(reasoning, str) and reasoning:
                    yield {"type": "reasoning_delta", "delta": reasoning}
                
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        text = str(block.get("text") or "")
                        if block_type == "text" and text:
                            yield {"type": "text_delta", "delta": text}
                        elif block_type in ("reasoning", "thinking") and text:
                            yield {"type": "reasoning_delta", "delta": text}
                refusal = bool(delta.get("refusal")) or refusal
                if choice.get("finish_reason") and not saw_completed:
                    saw_completed = True
                    yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}

        if not saw_completed:
            yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}

    def _normalize_anthropic_stream(self, stream_iter: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        usage: Dict[str, Any] | None = None
        final_raw: Dict[str, Any] = {}
        refusal: Optional[bool] = None
        completed = False

        for event in stream_iter:
            name = str(event.get("event") or "message")
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            if name == "content_block_delta":
                delta = data.get("delta") or {}
                delta_type = str(delta.get("type") or "")
                if delta_type == "text_delta":
                    text = str(delta.get("text") or "")
                    if text:
                        yield {"type": "text_delta", "delta": text}
                elif delta_type == "thinking_delta":
                    text = str(delta.get("thinking") or "")
                    if text:
                        yield {"type": "reasoning_delta", "delta": text}
            elif name == "message_start":
                message = data.get("message") or {}
                if isinstance(message, dict):
                    usage = message.get("usage") or usage
                    final_raw = dict(message)
            elif name == "message_delta":
                if isinstance(data.get("usage"), dict):
                    usage = data.get("usage")
            elif name == "message_stop":
                completed = True
                normalized = normalize_response(self.config.name, final_raw, True) if final_raw else {}
                usage = normalized.get("usage") or usage
                refusal = normalized.get("refusal")
                yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}
            elif name == "error":
                err = data.get("error") or {}
                raise RuntimeError(str(err.get("message") or data.get("message") or "stream error"))

        if not completed:
            yield {"type": "completed", "raw": final_raw, "usage": usage, "refusal": refusal}

    def _resolve_model(self, model: Optional[str]) -> str:
        if model:
            return self.config.models.get(model, model)
        default = self.config.default_model
        return self.config.models.get(default, default)

    def _build_openai_responses_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        def _to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                return str(v)

        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []
        pending_tool_ids: List[str] = []
        auto_idx = 0
        for msg in messages:
            role = str(msg.get("role") or "")
            content = _to_text(msg.get("content"))
            if role == "system":
                if content:
                    instructions_parts.append(content)
                continue
            if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    tc_id = str(tc.get("id") or "") or f"call_auto_{auto_idx}"
                    if not tc.get("id"):
                        auto_idx += 1
                    fn_name = str(fn.get("name") or tc.get("name") or "")
                    fn_args = fn.get("arguments", {})
                    if not isinstance(fn_args, str):
                        fn_args = _to_text(fn_args)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc_id,
                            "name": fn_name,
                            "arguments": fn_args,
                        }
                    )
                    pending_tool_ids.append(tc_id)
                continue
            if role == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                if not tc_id:
                    tc_id = pending_tool_ids.pop(0) if pending_tool_ids else f"call_auto_{auto_idx}"
                    auto_idx += 1
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc_id,
                        "output": content,
                    }
                )
                continue
            input_items.append({"role": role, "content": content})

        payload: Dict[str, Any] = {
            "_api_mode": "responses",
            "model": model,
            "input": input_items,
        }
        if instructions_parts:
            payload["instructions"] = "\n\n".join(p for p in instructions_parts if p.strip())

        if tools:
            from src.llm.tools import ToolDef
            resp_tools = []
            for t in tools:
                if isinstance(t, ToolDef):
                    resp_tools.append(
                        {
                            "type": "function",
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    )
                elif isinstance(t, dict):
                    if t.get("type") == "function" and isinstance(t.get("function"), dict):
                        fn = t.get("function") or {}
                        resp_tools.append(
                            {
                                "type": "function",
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                            }
                        )
                    else:
                        resp_tools.append(t)
            payload["tools"] = resp_tools

        for key, value in params.items():
            if key == "reasoning_effort":
                payload["reasoning"] = {"effort": value}
            elif key == "max_completion_tokens":
                payload["max_output_tokens"] = value
            elif key == "max_tokens":
                payload["max_output_tokens"] = value
            elif key == "response_format":
                # Keep structured-output on chat.completions for now.
                continue
            else:
                payload[key] = value

        if "max_output_tokens" not in payload and "max_tokens" in params:
            payload["max_output_tokens"] = params["max_tokens"]

        if payload.get("enable_thinking") is True and "thinking_budget" not in payload:
            cap = int(payload.get("max_output_tokens") or 65_536)
            payload["thinking_budget"] = max(25_000, cap - 8_000)

        return payload

    def _build_gemini_native_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
        *,
        fallback_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _to_text(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                return str(v)

        system_parts: List[Dict[str, Any]] = []
        contents: List[Dict[str, Any]] = []
        tool_id_to_name: Dict[str, str] = {}
        auto_idx = 0
        for msg in messages:
            role = str(msg.get("role") or "")
            content = _to_text(msg.get("content"))
            if role == "system":
                if content:
                    system_parts.append({"text": content})
                continue
            if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                parts: List[Dict[str, Any]] = []
                if content:
                    parts.append({"text": content})
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    tc_id = str(tc.get("id") or "") or f"gemini_call_{auto_idx}"
                    if not tc.get("id"):
                        auto_idx += 1
                    fn_name = str(fn.get("name") or tc.get("name") or "")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {"_raw": fn_args}
                    tool_id_to_name[tc_id] = fn_name
                    # Gemini v1beta does NOT accept "id" in input FunctionCall parts —
                    # the model returns ids in its output but rejects them as input fields,
                    # causing HTTP 400 at iteration 1+. Use name-based matching only.
                    parts.append({"functionCall": {"name": fn_name, "args": fn_args or {}}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue
            if role == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                fn_name = str(msg.get("name") or tool_id_to_name.get(tc_id) or "tool")
                response_payload: Any = content
                try:
                    response_payload = json.loads(content) if content else {}
                except Exception:
                    response_payload = {"content": content}
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": fn_name,
                                    "response": response_payload if isinstance(response_payload, dict) else {"content": content},
                                }
                            }
                        ],
                    }
                )
                continue
            contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]})

        payload: Dict[str, Any] = {
            "_api_mode": "gemini_native",
            "_model_id": model,
            "model": model,
            "contents": contents,
        }
        if fallback_payload is not None:
            payload["_fallback_payload"] = fallback_payload
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation_config: Dict[str, Any] = {}
        if params.get("temperature") is not None:
            generation_config["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            generation_config["topP"] = params["top_p"]
        stop = params.get("stop")
        if stop:
            generation_config["stopSequences"] = stop if isinstance(stop, list) else [stop]
        max_out = params.get("max_completion_tokens") or params.get("max_tokens")
        if max_out is not None:
            generation_config["maxOutputTokens"] = max_out
        reasoning_effort = params.get("reasoning_effort")
        if reasoning_effort:
            effort = str(reasoning_effort).strip().lower()
            # Gemini REST API uses thinkingConfig.thinkingBudget (integer token count),
            # not thinkingLevel (which is not a valid REST API field and causes 400 errors).
            _effort_to_budget = {
                "minimal": 512,
                "low": 1024,
                "medium": 8192,
                "high": 24576,
            }
            budget = _effort_to_budget.get(effort, 8192)
            generation_config["thinkingConfig"] = {"thinkingBudget": budget}
        if generation_config:
            payload["generationConfig"] = generation_config

        if tools:
            from src.llm.tools import ToolDef
            function_declarations = []
            for t in tools:
                if isinstance(t, ToolDef):
                    function_declarations.append(
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    )
                elif isinstance(t, dict):
                    if t.get("type") == "function" and isinstance(t.get("function"), dict):
                        fn = t.get("function") or {}
                        function_declarations.append(
                            {
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                            }
                        )
            if function_declarations:
                payload["tools"] = [{"functionDeclarations": function_declarations}]

        return payload

    def _build_openai_compat_chat_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible chat/completions payload."""
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
                if normalized.get("name") is None:
                    normalized["name"] = "tool"

            sanitized.append(normalized)
        payload = {
            "model": model,
            "messages": sanitized,
        }

        if tools:
            from src.llm.tools import ToolDef
            openai_tools = [t.to_openai() if isinstance(t, ToolDef) else t for t in tools]
            payload["tools"] = openai_tools

        standard_params = {
            "max_tokens", "max_completion_tokens", "temperature", "top_p",
            "frequency_penalty", "presence_penalty", "stop",
            "stream", "n", "response_format",
        }

        is_gemini = "generativelanguage.googleapis.com" in (self.config.base_url or "")
        for key, value in params.items():
            if is_gemini and key == "media_resolution":
                continue
            if key in standard_params:
                payload[key] = value
            else:
                payload[key] = value

        payload = _apply_openai_compat_best_practices(
            payload,
            base_url=self.config.base_url,
            provider_name=self.config.name,
            model=model,
            has_tools=bool(tools),
        )

        if payload.get("max_tokens") is None:
            payload.pop("max_tokens", None)
        if payload.get("max_completion_tokens") is None:
            payload.pop("max_completion_tokens", None)

        provider_name = (self.config.name or "").lower()
        if "max_tokens" not in payload and "max_completion_tokens" not in payload:
            if "kimi" in provider_name:
                payload["max_tokens"] = 32_768
            elif "deepseek" in provider_name:
                payload["max_tokens"] = 8_192
            elif "gemini" in provider_name:
                payload["max_tokens"] = 65_536
            elif "openai" in provider_name:
                payload["max_tokens"] = 128_000
            elif "qwen" in provider_name:
                model_lower = (model or "").lower()
                if "plus" in model_lower or "max" in model_lower:
                    payload["max_tokens"] = 65_536
                else:
                    payload["max_tokens"] = 32_768
            elif "perplexity" in provider_name or "sonar" in provider_name:
                payload["max_tokens"] = 16_384

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

        _MIN_RESPONSE_TOKENS = 8_000
        if payload.get("enable_thinking") is True and "thinking_budget" not in payload:
            cap = payload.get("max_tokens") or 65_536
            payload["thinking_budget"] = max(25_000, cap - _MIN_RESPONSE_TOKENS)

        is_openai = "api.openai.com" in (self.config.base_url or "")
        if is_openai and "max_tokens" in payload and "max_completion_tokens" not in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        return payload

    def _build_openai_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        params: Dict[str, Any],
        tools: Optional[List] = None,
        response_model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """构建 OpenAI-compatible 请求 payload"""
        if _can_use_gemini_native_api(
            base_url=self.config.base_url,
            messages=messages,
            response_model=response_model,
        ):
            fallback_payload = self._build_openai_compat_chat_payload(messages, model, params, tools=tools)
            return self._build_gemini_native_payload(
                messages,
                model,
                params,
                tools=tools,
                fallback_payload=fallback_payload,
            )
        if _should_use_openai_responses_api(
            base_url=self.config.base_url,
            model=model,
            has_tools=bool(tools),
            params=params,
            response_model=response_model,
        ):
            return self._build_openai_responses_payload(messages, model, params, tools=tools)
        if _should_use_qwen_responses_api(
            base_url=self.config.base_url,
            has_tools=bool(tools),
            params=params,
            response_model=response_model,
        ):
            return self._build_openai_responses_payload(messages, model, params, tools=tools)
        return self._build_openai_compat_chat_payload(messages, model, params, tools=tools)

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
            if params.get("enable_prompt_cache"):
                # Anthropic prompt caching: system 作为带 cache_control 的 content 数组
                payload["system"] = [{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]
            else:
                payload["system"] = system_content

        # 合并参数（注意排除非 API 参数）
        _NON_API_PARAMS = {"enable_prompt_cache"}
        for key, value in params.items():
            if key not in _NON_API_PARAMS:
                payload[key] = value

        thinking_cfg = payload.get("thinking") or {}
        thinking_type = str(thinking_cfg.get("type") or "").strip().lower()
        is_opus_46 = _is_claude_opus_46(model)
        if is_opus_46 and thinking_type == "enabled":
            # Anthropic recommends adaptive thinking for Opus 4.6.
            effort = _infer_anthropic_adaptive_effort(thinking_cfg.get("budget_tokens"))
            payload["thinking"] = {"type": "adaptive", "effort": effort}
            thinking_cfg = payload["thinking"]
            thinking_type = "adaptive"
            _log.info(
                "Anthropic compat: upgraded thinking mode to adaptive for model=%s with effort=%s",
                model,
                effort,
            )
        elif thinking_type == "adaptive" and not thinking_cfg.get("effort"):
            thinking_cfg = dict(thinking_cfg)
            thinking_cfg["effort"] = "medium"
            payload["thinking"] = thinking_cfg
            thinking_type = "adaptive"

        # Anthropic API 必须提供大于 0 的整型 max_tokens。
        # Claude Sonnet/Haiku 4.5/4.6 最大输出 64K；Opus 4.6 最大输出 128K。
        is_thinking = thinking_type in ("enabled", "adaptive")
        _CLAUDE_45_MAX_OUTPUT = 64_000
        _CLAUDE_OPUS_46_MAX_OUTPUT = 128_000
        _MIN_VISIBLE_TOKENS = 8_000
        max_model_output = _CLAUDE_OPUS_46_MAX_OUTPUT if is_opus_46 else _CLAUDE_45_MAX_OUTPUT

        if payload.get("max_tokens") is None:
            payload["max_tokens"] = max_model_output if is_thinking else 16384

        # Extended thinking: max_tokens 必须 > budget_tokens，否则 API 报错。
        # Cap budget so that (budget + visible response) stays within max_tokens.
        if is_thinking and thinking_type == "enabled":
            budget = int(thinking_cfg.get("budget_tokens") or 10000)
            max_out = payload["max_tokens"]
            if max_out <= budget:
                payload["max_tokens"] = budget + _MIN_VISIBLE_TOKENS
            # Ensure we do not exceed platform cap and that budget leaves room for answer
            payload["max_tokens"] = min(payload["max_tokens"], max_model_output)
            max_budget = payload["max_tokens"] - _MIN_VISIBLE_TOKENS
            if budget > max_budget:
                thinking_cfg = dict(thinking_cfg)
                thinking_cfg["budget_tokens"] = max_budget
                payload["thinking"] = thinking_cfg

        if is_thinking and thinking_type == "adaptive":
            payload["max_tokens"] = min(int(payload["max_tokens"]), max_model_output)

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
        elif pcfg.platform == "gemini" or _is_gemini_api(pcfg.base_url):
            http_provider = GeminiNativeProvider(pcfg)
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

    def _get_image_platform(self, provider: str) -> PlatformConfig:
        platform = self.config.platforms.get(provider)
        if not platform:
            raise ValueError(f"Platform '{provider}' not found in config")
        if not platform.api_key or platform.api_key in ("sk-xxx", "sk-ant-xxx", "AIzxxx", ""):
            raise ValueError(f"API key for platform '{provider}' is not configured")
        return platform

    def generate_image(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        timeout: Optional[int] = None,
    ) -> bytes:
        platform = self._get_image_platform(provider)
        session = requests.Session()
        timeout = _normalize_timeout(timeout)

        if provider == "gemini":
            root = _gemini_native_base_url(platform.base_url)
            url = f"{root}/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            }
            resp = _request_with_retry(
                session,
                "POST",
                url,
                timeout,
                headers={"Content-Type": "application/json", "x-goog-api-key": platform.api_key},
                json=payload,
            )
            data = resp.json()
            for part in (data.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
                if "inlineData" in part:
                    return base64.b64decode(part["inlineData"]["data"])
            raise ValueError(f"Gemini returned no image data. Response keys: {list(data.keys())}")

        if provider == "openai":
            url = f"{platform.base_url.rstrip('/')}/images/generations"
            payload = {
                "model": model,
                "prompt": prompt,
                "size": _normalize_image_size(size, provider="openai"),
            }
            resp = _request_with_retry(
                session,
                "POST",
                url,
                timeout,
                headers={"Authorization": f"Bearer {platform.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            return _extract_openai_image_bytes(resp.json())

        if provider == "qwen":
            url = _qwen_image_generation_url(platform.base_url)
            payload = {
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}],
                        }
                    ]
                },
                "parameters": {
                    "size": _normalize_image_size(size, provider="qwen"),
                    "watermark": False,
                    "prompt_extend": True,
                },
            }
            resp = _request_with_retry(
                session,
                "POST",
                url,
                timeout,
                headers={"Authorization": f"Bearer {platform.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            image_url = _extract_qwen_image_url(resp.json())
            image_resp = _request_with_retry(session, "GET", image_url, timeout)
            return image_resp.content

        if provider == "kimi":
            raise NotImplementedError(
                f"Image generation is not supported for provider '{provider}' in the current Moonshot/Kimi integration"
            )

        raise NotImplementedError(f"Image generation is not implemented for provider '{provider}'")

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

def _stream_and_collect(
    client: "LLMClient",
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    idle_timeout_seconds: int = 90,
    **overrides,
) -> Dict[str, Any]:
    """流式调用并累积为与 chat() 相同格式的响应字典（流量感知超时）。

    idle_timeout_seconds: 相邻两个 chunk 之间最大等待秒数（活跃度窗口）。
    只要模型持续生成 token，总调用时长不受限制；空闲超过此值才触发 Timeout。
    citations / search_results 等元数据从流式最终 chunk 中提取（需配合已修复的
    _normalize_openai_stream 使用）。
    """
    final_response: Optional[Dict[str, Any]] = None
    text_parts: List[str] = []

    for event in client.stream_chat(
        messages,
        model=model,
        idle_timeout_seconds=idle_timeout_seconds,
        **overrides,
    ):
        ev_type = event.get("type")
        if ev_type == "text_delta":
            delta = str(event.get("delta") or "")
            if delta:
                text_parts.append(delta)
        elif ev_type == "completed":
            resp = event.get("response")
            if isinstance(resp, dict):
                final_response = resp

    if final_response is not None:
        # stream_chat 内部已累积 final_text；若为空则用我们自己的 text_parts
        if not final_response.get("final_text") and text_parts:
            final_response = dict(final_response)
            final_response["final_text"] = "".join(text_parts)
        return final_response

    return {
        "final_text": "".join(text_parts),
        "raw": {},
        "meta": {},
    }


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


def stream_chat(
    messages: List[Dict[str, Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Iterator[Dict[str, Any]]:
    manager = get_manager()
    client = manager.get_client(provider)
    yield from client.stream_chat(messages, model=model, **kwargs)
