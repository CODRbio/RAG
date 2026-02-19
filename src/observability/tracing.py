"""
OpenTelemetry tracing 配置 + LangSmith / Langfuse Agentic Tracing。

提供全局 tracer 供业务代码使用：
    from src.observability import tracer
    with tracer.start_as_current_span("retrieval.vector"):
        ...

LangSmith 追踪通过环境变量控制，未配置时静默跳过：
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your-key>
    LANGCHAIN_PROJECT=deepsea-rag  # 可选
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

_resource = Resource.create({"service.name": "deepsea-rag", "service.version": "0.1.0"})

_provider = TracerProvider(resource=_resource)

# 默认导出到控制台（生产环境可替换为 OTLP exporter）
# 仅在 RAG_TRACE_CONSOLE=1 时启用 console export，避免日志噪音
if os.getenv("RAG_TRACE_CONSOLE", "0") == "1":
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("deepsea-rag", "0.1.0")

# ── LangSmith / Langfuse Agentic Tracing ──
# 当 LANGCHAIN_TRACING_V2=true 时启用，否则静默跳过，不报错。
langsmith_enabled: bool = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true")

try:
    if not langsmith_enabled:
        raise ImportError("LANGCHAIN_TRACING_V2 not set")
    from langsmith import traceable as _langsmith_traceable

    def traceable(*args, **kwargs):  # type: ignore[misc]
        """
        LangSmith traceable 兼容层。
        兼容 run_type="agent" 的调用习惯，内部映射到受支持的 "chain"。
        """
        if kwargs.get("run_type") == "agent":
            kwargs = dict(kwargs)
            kwargs["run_type"] = "chain"
            kwargs.setdefault("name", "agent")
        return _langsmith_traceable(*args, **kwargs)
except ImportError:
    langsmith_enabled = False

    def traceable(*args, **kwargs):  # type: ignore[misc]
        """Dummy @traceable — langsmith 未安装或追踪未启用时使用。"""
        if args and callable(args[0]):
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator
