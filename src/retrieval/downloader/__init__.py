"""
Scholar paper downloader: PDF download via Playwright, Sci-Hub, Anna's Archive.
Search is delegated to RAG's google_search / semantic_scholar / ncbi_search.
"""

from src.retrieval.downloader.adapter import (
    ScholarDownloaderAdapter,
    get_adapter,
    shutdown_adapter,
)

__all__ = ["ScholarDownloaderAdapter", "get_adapter", "shutdown_adapter"]
