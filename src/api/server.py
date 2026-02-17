"""
FastAPI 应用入口 - 多轮对话 API
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes_auth import router as auth_router, admin_router
from src.api.routes_canvas import router as canvas_router
from src.api.routes_chat import router as chat_router
from src.api.routes_export import router as export_router
from src.api.routes_auto import router as auto_router
from src.api.routes_project import router as project_router
from src.api.routes_models import router as models_router
from src.api.routes_ingest import router as ingest_router
from src.api.routes_graph import router as graph_router
from src.api.routes_compare import router as compare_router
from src.log import get_logger
from src.utils.storage_cleaner import run_cleanup, get_storage_stats
from src.observability import setup_observability

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时执行存储清理"""
    if settings.storage.cleanup_on_startup:
        try:
            result = run_cleanup(
                max_age_days=settings.storage.max_age_days,
                max_size_gb=settings.storage.max_size_gb,
                batch_size=settings.storage.cleanup_batch_size,
            )
            stats = get_storage_stats()
            logger.info(f"[startup] storage cleanup done: {result}, current={stats}")
        except Exception as e:
            logger.warning(f"[startup] storage cleanup failed: {e}")
    yield


app = FastAPI(
    title="DeepSea RAG Chat API",
    description="多轮对话与综述协作 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(project_router)
app.include_router(chat_router)
app.include_router(canvas_router)
app.include_router(export_router)
app.include_router(auto_router)
app.include_router(models_router)
app.include_router(ingest_router)
app.include_router(graph_router)
app.include_router(compare_router)

# Observability: 中间件 + /metrics + /health/detailed
setup_observability(app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/storage/stats")
def storage_stats() -> dict:
    """获取存储统计信息"""
    return get_storage_stats()
