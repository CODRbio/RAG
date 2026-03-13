"""统一日志管理：分层路由、correlation_id 注入、error 聚合、自动清理。"""
from .log_manager import (
    LogManager,
    get_logger,
    init_logging,
    cleanup_logs,
)
from .context import (
    get_correlation_id,
    set_correlation_id,
    new_correlation_id,
)

__all__ = [
    "LogManager",
    "get_logger",
    "init_logging",
    "cleanup_logs",
    "get_correlation_id",
    "set_correlation_id",
    "new_correlation_id",
]
