"""
缓存工具

TTLCache  — 内存 LRU-TTL 缓存，进程内快速命中。
DiskCache — SQLite 持久化 TTL 缓存，跨进程/重启复用。可单独使用，也可作为 TTLCache 的 L2 层。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
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


class DiskCache:
    """
    SQLite 持久化 TTL 缓存，带热度晋升机制。

    规则
    ----
    - 普通条目：超过 ttl_seconds 后过期删除（默认 30 天）。
    - 热门晋升：同一条目在缓存期内命中次数 >= promote_threshold（默认 3）时，
      自动标记为 permanent=1，此后永不过期，也不再刷新 created_at。
    - permanent 条目只能通过 delete() 或 clear_permanent() 手动删除。

    其他特性
    --------
    - 跨进程/重启复用。
    - 线程安全（check_same_thread=False + RLock）。
    - WAL 模式：并发写不阻塞读。
    - TTL 到期惰性删除（get 时检查）+ 每小时主动 GC。
    - 兼容旧库（自动迁移 ALTER TABLE 补列）。
    """

    GC_INTERVAL_S: int = 3600

    def __init__(
        self,
        db_path: str,
        ttl_seconds: int = 2592000,    # 30 days
        promote_threshold: int = 3,
    ):
        """
        Args:
            db_path:            SQLite 文件路径，父目录不存在时自动创建。
            ttl_seconds:        普通条目有效期（秒）。0 = 永不过期。默认 30 天。
            promote_threshold:  命中次数达到该值后晋升为永久缓存。默认 3。
        """
        self._ttl = max(0, ttl_seconds)
        self._promote_threshold = max(1, promote_threshold)
        self._lock = RLock()
        self._last_gc = time.monotonic()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key        TEXT PRIMARY KEY,"
            "  value      TEXT    NOT NULL,"
            "  created_at REAL    NOT NULL,"
            "  hit_count  INTEGER NOT NULL DEFAULT 0,"
            "  permanent  INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
        self._conn.commit()
        self._migrate()

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        """
        返回缓存值；未命中或已过期返回 None。

        命中时递增 hit_count；达到 promote_threshold 后将条目标记为 permanent。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created_at, hit_count, permanent FROM cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            value, created_at, hit_count, permanent = row

            # 非永久条目检查 TTL
            if not permanent and self._ttl > 0 and (time.time() - created_at) > self._ttl:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None

            # 递增命中计数
            new_hit_count = hit_count + 1
            if not permanent and new_hit_count >= self._promote_threshold:
                self._conn.execute(
                    "UPDATE cache SET hit_count = ?, permanent = 1 WHERE key = ?",
                    (new_hit_count, key),
                )
                self._conn.commit()
            else:
                self._conn.execute(
                    "UPDATE cache SET hit_count = ? WHERE key = ?",
                    (new_hit_count, key),
                )
                self._conn.commit()
            return value

    def set(self, key: str, value: str) -> None:
        """
        写入缓存条目。

        - 若条目已是 permanent，不做任何更新（内容已足够稳定，不应被新抓取覆盖）。
        - 若条目已存在但未晋升，更新 value/created_at，保留 hit_count 继续累计。
        - 新条目：hit_count=0, permanent=0。
        """
        with self._lock:
            existing = self._conn.execute(
                "SELECT hit_count, permanent FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if existing:
                _, perm = existing
                if perm:
                    return  # 永久条目不覆盖
                self._conn.execute(
                    "UPDATE cache SET value = ?, created_at = ? WHERE key = ?",
                    (value, time.time(), key),
                )
            else:
                self._conn.execute(
                    "INSERT INTO cache (key, value, created_at, hit_count, permanent)"
                    " VALUES (?, ?, ?, 0, 0)",
                    (key, value, time.time()),
                )
            self._conn.commit()
            self._maybe_gc()

    def delete(self, key: str) -> None:
        """强制删除条目（含 permanent）。"""
        with self._lock:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()

    def clear_expired(self) -> int:
        """主动清除所有过期的非永久条目，返回删除数量。"""
        if self._ttl == 0:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE permanent = 0 AND ? - created_at > ?",
                (time.time(), self._ttl),
            )
            self._conn.commit()
            self._last_gc = time.monotonic()
            return cur.rowcount

    def clear_permanent(self) -> int:
        """清除所有 permanent 条目，返回删除数量。"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM cache WHERE permanent = 1")
            self._conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            permanent = self._conn.execute(
                "SELECT COUNT(*) FROM cache WHERE permanent = 1"
            ).fetchone()[0]
            expired = 0
            if self._ttl > 0:
                expired = self._conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE permanent = 0 AND ? - created_at > ?",
                    (time.time(), self._ttl),
                ).fetchone()[0]
            return {
                "total": total,
                "permanent": permanent,
                "expired": expired,
                "ttl_seconds": self._ttl,
                "promote_threshold": self._promote_threshold,
            }

    # ── internal ─────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """兼容旧库：若缺少 hit_count / permanent 列则自动添加。"""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(cache)")}
        with self._lock:
            if "hit_count" not in cols:
                self._conn.execute(
                    "ALTER TABLE cache ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0"
                )
            if "permanent" not in cols:
                self._conn.execute(
                    "ALTER TABLE cache ADD COLUMN permanent INTEGER NOT NULL DEFAULT 0"
                )
            self._conn.commit()

    def _maybe_gc(self) -> None:
        if time.monotonic() - self._last_gc > self.GC_INTERVAL_S:
            self.clear_expired()


def get_disk_cache(
    enabled: bool,
    db_path: str = "data/cache/web_content.db",
    ttl_seconds: int = 2592000,
    promote_threshold: int = 3,
) -> Optional[DiskCache]:
    """若 enabled 为 False 返回 None，否则返回一个 DiskCache。"""
    if not enabled:
        return None
    return DiskCache(db_path=db_path, ttl_seconds=ttl_seconds, promote_threshold=promote_threshold)
