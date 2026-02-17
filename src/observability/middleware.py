"""
FastAPI 中间件：自动采集 HTTP 请求延迟 / 计数 / 状态码，并注入 trace context。
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.metrics import metrics
from src.observability.tracing import tracer


def _normalize_path(path: str) -> str:
    """
    将 path 中的动态 ID 替换为占位符，防止高基数指标。
    e.g. /canvas/abc123/snapshots → /canvas/{id}/snapshots
    """
    parts = path.strip("/").split("/")
    normalized = []
    skip_next = False
    for i, part in enumerate(parts):
        if skip_next:
            normalized.append("{id}")
            skip_next = False
            continue
        # 常见的 RESTful 资源路径：下一个 segment 是 ID
        if part in ("canvas", "projects", "sessions", "graph"):
            normalized.append(part)
            skip_next = True
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """采集每个 HTTP 请求的延迟和计数指标，并创建 trace span。"""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = _normalize_path(request.url.path)

        # 跳过 /metrics 和 /health 本身，避免自引用噪音
        if request.url.path in ("/metrics", "/health"):
            return await call_next(request)

        with tracer.start_as_current_span(
            f"{method} {path}",
            attributes={"http.method": method, "http.url": str(request.url)},
        ) as span:
            start = time.perf_counter()
            response: Response = await call_next(request)
            elapsed = time.perf_counter() - start

            status_code = str(response.status_code)
            span.set_attribute("http.status_code", response.status_code)

            metrics.http_requests_total.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=method, endpoint=path
            ).observe(elapsed)

            return response
