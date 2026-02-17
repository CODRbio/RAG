"""
内存 TTL 缓存：按 key 缓存，过期自动失效，不落盘。
"""

from __future__ import annotations

import hashlib
import json
import time
from threading import RLock
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def _make_key(prefix: str, *parts: Any) -> str:
    """生成缓存键。parts 会做稳定序列化。"""
    raw = [prefix]
    for p in parts:
        if p is None:
            raw.append("")
        elif isinstance(p, (str, int, float, bool)):
            raw.append(str(p))
        else:
            try:
                raw.append(json.dumps(p, sort_keys=True, default=str))
            except Exception:
                raw.append(repr(p))
    blob = "|".join(raw).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class TTLCache:
    """
    线程安全的内存 TTL 缓存。
    - maxsize: 最大条目数，超过时淘汰最久未访问的。
    - ttl_seconds: 条目过期时间，0 表示不过期。
    """

    __slots__ = ("_store", "_expiry", "_order", "_maxsize", "_ttl", "_lock")

    def __init__(self, maxsize: int = 1024, ttl_seconds: int = 3600):
        self._store: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._order: list[str] = []
        self._maxsize = max(1, maxsize)
        self._ttl = max(0, ttl_seconds)
        self._lock = RLock()

    def _evict_if_needed(self) -> None:
        now = time.monotonic()
        while self._order and len(self._store) >= self._maxsize:
            k = self._order.pop(0)
            self._store.pop(k, None)
            self._expiry.pop(k, None)
        # drop expired
        expired = [k for k, t in self._expiry.items() if self._ttl > 0 and (now - t) > self._ttl]
        for k in expired:
            self._store.pop(k, None)
            self._expiry.pop(k, None)
            if k in self._order:
                self._order.remove(k)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            if self._ttl > 0 and (time.monotonic() - self._expiry.get(key, 0)) > self._ttl:
                self._store.pop(key, None)
                self._expiry.pop(key, None)
                if key in self._order:
                    self._order.remove(key)
                return None
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return self._store[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._evict_if_needed()
            self._store[key] = value
            self._expiry[key] = time.monotonic()
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            if key in self._order:
                self._order.remove(key)

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        val = self.get(key)
        if val is not None:
            return val
        val = factory()
        self.set(key, val)
        return val


def get_cache(
    enabled: bool,
    ttl_seconds: int = 3600,
    maxsize: int = 1024,
    prefix: str = "cache",
) -> Optional[TTLCache]:
    """若 enabled 为 False 返回 None，否则返回一个 TTLCache。"""
    if not enabled:
        return None
    return TTLCache(maxsize=maxsize, ttl_seconds=ttl_seconds)
