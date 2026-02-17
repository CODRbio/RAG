"""
OpenTelemetry tracing 配置。

提供全局 tracer 供业务代码使用：
    from src.observability import tracer
    with tracer.start_as_current_span("retrieval.vector"):
        ...
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

_resource = Resource.create({"service.name": "deepsea-rag", "service.version": "0.1.0"})

_provider = TracerProvider(resource=_resource)

# 默认导出到控制台（生产环境可替换为 OTLP exporter）
# 仅在 RAG_TRACE_CONSOLE=1 时启用 console export，避免日志噪音
import os
if os.getenv("RAG_TRACE_CONSOLE", "0") == "1":
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("deepsea-rag", "0.1.0")
