"""
Enable aiohttp HTTPS-over-HTTPS proxy (TLS-in-TLS) on Python < 3.11.

stdlib asyncio disables TLS-in-TLS by default until Python 3.11, which causes
RuntimeWarning and can break requests when using an HTTPS proxy (e.g. BrightData).
This module applies the recommended monkeypatch on import so that aiohttp works
with HTTPS proxies. Safe to import multiple times (idempotent).

Refs:
- https://bugs.python.org/issue37179
- https://github.com/aio-libs/aiohttp/discussions/6044#discussioncomment-1432443
- https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support
"""

import logging
import sys

logger = logging.getLogger(__name__)
_PATCH_APPLIED = False


def apply_aiohttp_tls_in_tls_patch() -> bool:
    """
    Enable TLS-in-TLS for asyncio so aiohttp can use HTTPS proxies on Python < 3.11.
    Returns True if the patch was applied, False if skipped (e.g. already 3.11+).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True
    if sys.version_info >= (3, 11):
        return False
    try:
        import asyncio.sslproto
        setattr(asyncio.sslproto._SSLProtocolTransport, "_start_tls_compatible", True)
        _PATCH_APPLIED = True
        logger.debug("aiohttp TLS-in-TLS patch applied (Python < 3.11)")
        return True
    except AttributeError:
        # e.g. uvloop or other transport; patch not applicable
        logger.debug("aiohttp TLS-in-TLS patch skipped (transport has no _start_tls_compatible)")
        return False
