"""
Standalone utilities for the scholar downloader (no PaperDownloader dependency).
"""

import os


def is_valid_pdf(filepath: str) -> bool:
    """Check PDF magic bytes and minimum size without instantiating PaperDownloader."""
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            return False
        with open(filepath, "rb") as f:
            header = f.read(1024)
        return header.startswith(b"%PDF-") and any(
            token in header for token in (b"obj", b"stream", b"/Type", b"/Pages")
        )
    except OSError:
        return False
