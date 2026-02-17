"""
并发限制：信号量包装，用于限制同时执行的任务数。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional


class ConcurrencyLimiter:
    """基于 threading.Semaphore 的并发限制器。"""

    def __init__(self, max_parallel: int = 1):
        self._sem = threading.Semaphore(max(1, max_parallel))

    @contextmanager
    def acquire(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

    def run(self, fn, *args, **kwargs):
        """在限制下执行 fn(*args, **kwargs)。"""
        with self.acquire():
            return fn(*args, **kwargs)


_global_limiter: Optional[ConcurrencyLimiter] = None
_limiter_lock = threading.Lock()


def get_global_limiter(max_parallel: int = 5) -> ConcurrencyLimiter:
    """获取或创建全局并发限制器（按首次调用的 max_parallel）。"""
    global _global_limiter
    with _limiter_lock:
        if _global_limiter is None:
            _global_limiter = ConcurrencyLimiter(max_parallel)
        return _global_limiter


def reset_global_limiter() -> None:
    """测试用：重置全局限制器。"""
    global _global_limiter
    with _limiter_lock:
        _global_limiter = None
