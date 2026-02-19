"""
Observability 模块：OpenTelemetry tracing + Prometheus metrics。

用法：
    from src.observability import setup_observability, metrics, tracer

    # 在 FastAPI lifespan 中初始化
    setup_observability(app)

    # 业务代码中手动埋点
    with tracer.start_as_current_span("my_operation"):
        ...

    metrics.retrieval_latency.observe(elapsed_seconds)
"""

from src.observability.setup import setup_observability
from src.observability.metrics import metrics
from src.observability.tracing import tracer, traceable, langsmith_enabled

__all__ = ["setup_observability", "metrics", "tracer", "traceable", "langsmith_enabled"]
