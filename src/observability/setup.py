"""
一键初始化 Observability：注册中间件 + /metrics 端点 + 应用元信息。
"""

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.observability.middleware import ObservabilityMiddleware
from src.observability.metrics import metrics
from src.log import get_logger

logger = get_logger(__name__)


def setup_observability(app: FastAPI) -> None:
    """
    在 FastAPI app 上挂载 Observability 组件。

    应在 router 注册之后、启动之前调用。
    """
    # 1. 注册中间件
    app.add_middleware(ObservabilityMiddleware)

    # 2. 注册 /metrics 端点（Prometheus 拉取）
    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # 3. 增强 /health 端点（带组件状态）
    @app.get("/health/detailed", tags=["observability"])
    def health_detailed():
        """详细健康检查：各组件可达性"""
        checks = {}

        # Milvus
        try:
            from src.indexing.milvus_ops import milvus
            milvus.list_collections()
            checks["milvus"] = "ok"
        except Exception as e:
            checks["milvus"] = f"error: {e}"

        # LLM
        try:
            from src.llm.llm_manager import get_manager
            m = get_manager()
            checks["llm"] = "ok" if m else "not_configured"
        except Exception as e:
            checks["llm"] = f"error: {e}"

        # HippoRAG graph
        try:
            from config.settings import settings
            graph_path = settings.path.data / "hippo_graph.json"
            checks["hippo_graph"] = "available" if graph_path.exists() else "not_built"
        except Exception:
            checks["hippo_graph"] = "unknown"

        overall = "ok" if all(v == "ok" or v == "available" or v == "not_built" for v in checks.values()) else "degraded"
        return {"status": overall, "components": checks}

    # 4. 设置应用元信息
    metrics.app_info.info({"version": "0.1.0", "service": "deepsea-rag"})

    logger.info("[observability] middleware + /metrics + /health/detailed registered")
