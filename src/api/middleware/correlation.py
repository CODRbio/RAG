"""
Correlation ID 中间件：为每个 HTTP 请求生成或透传 X-Correlation-ID 头。

- 如果请求头已携带 X-Correlation-ID，则复用（方便前端追踪）。
- 否则自动生成 req-xxxxxxxx 格式的 ID。
- 通过 ContextVar 将 ID 注入到整个请求调用链，所有 logger 自动携带。
- 响应头也会回写 X-Correlation-ID，便于前端和 API 客户端对账。

注意：使用纯 ASGI 中间件而非 BaseHTTPMiddleware，避免后者缓冲
StreamingResponse 导致 Content-Type charset 丢失（SSE 流中文乱码）。
"""
from __future__ import annotations

import uuid
from typing import Callable

from src.log.context import set_correlation_id, reset_correlation_id

HEADER_NAME = "X-Correlation-ID"
HEADER_NAME_BYTES = HEADER_NAME.lower().encode("latin-1")


class CorrelationMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Resolve correlation ID from request headers (case-insensitive lookup).
        raw_headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
        cid: str | None = None
        for k, v in raw_headers:
            if k.lower() == HEADER_NAME_BYTES:
                cid = v.decode("latin-1", errors="replace")
                break
        if not cid:
            cid = f"req-{uuid.uuid4().hex[:8]}"

        token = set_correlation_id(cid)
        cid_bytes = cid.encode("latin-1", errors="replace")

        async def send_with_cid(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((HEADER_NAME_BYTES, cid_bytes))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cid)
        finally:
            reset_correlation_id(token)
