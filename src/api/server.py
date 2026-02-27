"""
FastAPI 应用入口 - 多轮对话 API
"""

import asyncio
from contextlib import asynccontextmanager

# 在 Python < 3.11 下启用 aiohttp HTTPS 代理 (TLS-in-TLS)，避免 RuntimeWarning 与请求失败
from src.utils.aiohttp_tls_patch import apply_aiohttp_tls_in_tls_patch
apply_aiohttp_tls_in_tls_patch()

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
from src.api.routes_debug import router as debug_router
from src.api.routes_tasks import router as tasks_router
from src.log import get_logger
from src.utils.storage_cleaner import run_cleanup, get_storage_stats
from src.utils.task_runner import cleanup_stale_jobs, run_background_worker
from src.observability import setup_observability

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：DB初始化 → 历史迁移 → 清理残留任务 → 存储清理 → 启动后台 Worker"""

    # 0. 确保 rag.db 表结构存在，自动迁移旧 .db 文件（仅首次启动时执行）
    from src.db.engine import init_db
    from src.db.migrate_legacy import migrate_if_needed
    try:
        init_db()
    except Exception as e:
        logger.warning("[startup] init_db failed (may be OK if alembic already ran): %s", e)
    try:
        migrate_if_needed()
    except Exception as e:
        logger.warning("[startup] migrate_if_needed failed: %s", e)

    # 0a. JWT secret key safety check
    _DEFAULT_SECRET = "change-me-in-local"
    if settings.auth.secret_key == _DEFAULT_SECRET:
        logger.warning(
            "[startup] SECURITY WARNING: auth.secret_key is still set to the default value '%s'. "
            "All JWT tokens can be trivially forged. "
            "Set a strong random value in config/rag_config.local.json → auth.secret_key "
            "before deploying to production.",
            _DEFAULT_SECRET,
        )

    # 0b. Purge expired JWT revocation records to keep the table compact
    try:
        from src.auth.session import purge_expired_revocations
        purged = purge_expired_revocations()
        if purged:
            logger.info("[startup] purged %d expired token revocation record(s)", purged)
    except Exception as e:
        logger.warning("[startup] purge_expired_revocations failed: %s", e)

    # 1. 将上次进程中断时残留的 running/cancelling 任务重置为 error
    cleanup_stale_jobs()

    # 2. 常规存储清理
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

    # 3. 初始化全局 DebugLogger
    from src.debug import init_debug_logger
    dl = init_debug_logger(enabled=settings.debug)
    logger.info("[startup] debug mode: %s (log_dir=%s)", "ON" if dl.enabled else "OFF", dl.log_dir)

    # 4. 启动后台任务轮询 Worker
    worker_task = asyncio.create_task(run_background_worker())

    yield

    # Shutdown: 取消 Worker
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


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
app.include_router(tasks_router)
app.include_router(canvas_router)
app.include_router(export_router)
app.include_router(auto_router)
app.include_router(models_router)
app.include_router(ingest_router)
app.include_router(graph_router)
app.include_router(compare_router)
app.include_router(debug_router)

# Observability: 中间件 + /metrics + /health/detailed
setup_observability(app)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/detailed")
def health_detailed() -> dict:
    """Health + queue and Redis status for monitoring."""
    out: dict = {"status": "ok", "queue": None}
    try:
        from src.tasks import get_task_queue
        q = get_task_queue()
        out["queue"] = {
            "active_count": q.active_count(),
            "pending_count": q.pending_count(),
            "max_slots": settings.tasks.max_active_slots,
        }
    except Exception as e:
        out["queue"] = {"error": str(e)}
    return out


@app.get("/storage/stats")
def storage_stats() -> dict:
    """获取存储统计信息"""
    return get_storage_stats()
