"""统一日志管理：分级、按运行实例命名、自动清理。"""
from .log_manager import (
    LogManager,
    get_logger,
    init_logging,
    cleanup_logs,
)

__all__ = ["LogManager", "get_logger", "init_logging", "cleanup_logs"]
