"""
src/retrieval/downloader/log_utils.py

Thin bridge: redirects core download code (paper_downloader_refactored.py,
browser_manager.py) log_utils imports to RAG's logging system.

Core code usage:
  from log_utils import setup_logger, cleanup_old_logs
  logger = setup_logger('PaperDownloader', level="INFO")
  from log_utils import logger

This shim keeps the same API surface.
"""

import logging
from src.log import get_logger as _rag_get_logger

# Global logger used by browser_manager.py and display_manager.py
logger = _rag_get_logger("scholar_downloader")


def setup_logger(
    name: str = "scholar_downloader",
    level: str | int = logging.INFO,
    log_dir: str = "logs",
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Compatible with original log_utils.setup_logger().
    Returns RAG logger; log_dir etc. are ignored (RAG has its own logging config).
    """
    rag_logger = _rag_get_logger(f"scholar_downloader.{name}")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    rag_logger.setLevel(level)
    return rag_logger


def cleanup_old_logs(log_dir: str = "logs", days: int = 1) -> None:
    """
    Compatible with original log_utils.cleanup_old_logs().
    RAG has its own log rotation; this is a no-op.
    """
    pass


def log_search_results(*args, **kwargs) -> None:
    """Compatibility placeholder; originally logged search result table."""
    pass


def format_scholar_result(*args, **kwargs) -> str:
    """Compatibility placeholder."""
    return ""
