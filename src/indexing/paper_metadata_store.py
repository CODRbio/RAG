"""
论文元数据持久化存储 (SQLite)

提供 DOI / title / authors / year 的持久化查询与写入，
替代原来的 paper_metadata.json 平面文件。

特性:
- SQLite 标准库，零外部依赖
- DOI + normalized_title 双索引，O(1) 查询
- 线程安全（SQLite WAL 模式 + check_same_thread=False）
- 首次启动自动迁移旧 paper_metadata.json
- 单例模式，全局复用

使用:
    from src.indexing.paper_metadata_store import paper_meta_store

    paper_meta_store.upsert("paper_id", doi="10.1038/xxx", title="Full Title", ...)
    row = paper_meta_store.get("paper_id")
    doi_set = paper_meta_store.all_dois()
    title_set = paper_meta_store.all_normalized_titles()
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/paper_metadata.db")
_JSON_PATH = Path("data/paper_metadata.json")


def _normalize_doi(doi: Optional[str]) -> str:
    if not doi or not isinstance(doi, str):
        return ""
    d = doi.strip().lower()
    d = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", d)
    return d.rstrip("/.")


def _normalize_title(title: Optional[str]) -> str:
    if not title or not isinstance(title, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


class PaperMetadataStore:
    """SQLite-backed paper metadata store (singleton)."""

    _instance: Optional["PaperMetadataStore"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table()
        self._maybe_migrate_json()
        self._initialized = True
        count = self._conn.execute("SELECT COUNT(*) FROM paper_metadata").fetchone()[0]
        logger.info("PaperMetadataStore ready: %s (%d entries)", self._db_path, count)

    def _create_table(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_metadata (
                paper_id        TEXT PRIMARY KEY,
                doi             TEXT DEFAULT '',
                normalized_doi  TEXT DEFAULT '',
                title           TEXT DEFAULT '',
                normalized_title TEXT DEFAULT '',
                authors         TEXT DEFAULT '',
                year            INTEGER,
                source          TEXT DEFAULT '',
                extra           TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_pm_ndoi ON paper_metadata(normalized_doi) WHERE normalized_doi != '';
            CREATE INDEX IF NOT EXISTS idx_pm_ntitle ON paper_metadata(normalized_title) WHERE normalized_title != '';

            CREATE TABLE IF NOT EXISTS crossref_cache (
                normalized_title TEXT PRIMARY KEY,
                doi              TEXT DEFAULT '',
                title            TEXT DEFAULT '',
                authors          TEXT DEFAULT '',
                year             INTEGER,
                venue            TEXT DEFAULT '',
                created_at       REAL DEFAULT (julianday('now'))
            );
        """)

    def _maybe_migrate_json(self) -> None:
        """首次启动：若 SQLite 为空且旧 JSON 存在，自动导入。"""
        count = self._conn.execute("SELECT COUNT(*) FROM paper_metadata").fetchone()[0]
        if count > 0:
            return
        if not _JSON_PATH.exists():
            return
        try:
            with open(_JSON_PATH, "r", encoding="utf-8") as f:
                data: Dict[str, Dict[str, Any]] = json.load(f)
            if not data:
                return
            rows = []
            for paper_id, meta in data.items():
                doi = meta.get("doi") or ""
                title = meta.get("title") or ""
                authors = meta.get("authors")
                year = meta.get("year")
                source = meta.get("source") or ""
                authors_str = json.dumps(authors, ensure_ascii=False) if authors else ""
                rows.append((
                    paper_id,
                    doi,
                    _normalize_doi(doi),
                    title,
                    _normalize_title(title),
                    authors_str,
                    int(year) if year else None,
                    source,
                    "{}",
                ))
            self._conn.executemany(
                "INSERT OR IGNORE INTO paper_metadata "
                "(paper_id, doi, normalized_doi, title, normalized_title, authors, year, source, extra) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
            logger.info("Migrated %d entries from paper_metadata.json -> SQLite", len(rows))
        except Exception as e:
            logger.warning("Failed to migrate paper_metadata.json: %s", e)

    # ── 写入 ──────────────────────────────────────────

    def upsert(
        self,
        paper_id: str,
        doi: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        source: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """插入或更新一条论文元数据。"""
        existing = self.get(paper_id)
        doi = doi or (existing.get("doi") if existing else "") or ""
        title = title or (existing.get("title") if existing else "") or ""
        authors_str = json.dumps(authors, ensure_ascii=False) if authors else (existing.get("authors_raw") if existing else "") or ""
        yr = year or (existing.get("year") if existing else None)
        src = source or (existing.get("source") if existing else "") or ""
        ext = json.dumps(extra or {}, ensure_ascii=False) if extra else (existing.get("extra_raw") if existing else "{}") or "{}"

        self._conn.execute(
            "INSERT INTO paper_metadata "
            "(paper_id, doi, normalized_doi, title, normalized_title, authors, year, source, extra) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET "
            "doi=excluded.doi, normalized_doi=excluded.normalized_doi, "
            "title=excluded.title, normalized_title=excluded.normalized_title, "
            "authors=excluded.authors, year=excluded.year, "
            "source=excluded.source, extra=excluded.extra",
            (
                paper_id,
                doi,
                _normalize_doi(doi),
                title,
                _normalize_title(title),
                authors_str,
                yr,
                src,
                ext,
            ),
        )
        self._conn.commit()

    def upsert_batch(self, records: List[Tuple[str, Dict[str, Any]]]) -> int:
        """批量 upsert [(paper_id, {doi, title, authors, year, source}), ...]。"""
        rows = []
        for paper_id, meta in records:
            doi = meta.get("doi") or ""
            title = meta.get("title") or ""
            authors = meta.get("authors")
            authors_str = json.dumps(authors, ensure_ascii=False) if authors else ""
            year = meta.get("year")
            source = meta.get("source") or ""
            rows.append((
                paper_id,
                doi,
                _normalize_doi(doi),
                title,
                _normalize_title(title),
                authors_str,
                int(year) if year else None,
                source,
                "{}",
            ))
        self._conn.executemany(
            "INSERT INTO paper_metadata "
            "(paper_id, doi, normalized_doi, title, normalized_title, authors, year, source, extra) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET "
            "doi=excluded.doi, normalized_doi=excluded.normalized_doi, "
            "title=excluded.title, normalized_title=excluded.normalized_title, "
            "authors=excluded.authors, year=excluded.year, "
            "source=excluded.source, extra=excluded.extra",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # ── 查询 ──────────────────────────────────────────

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """按 paper_id 查询单条。"""
        row = self._conn.execute(
            "SELECT paper_id, doi, title, authors, year, source, extra "
            "FROM paper_metadata WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """按 DOI 查询。"""
        ndoi = _normalize_doi(doi)
        if not ndoi:
            return None
        row = self._conn.execute(
            "SELECT paper_id, doi, title, authors, year, source, extra "
            "FROM paper_metadata WHERE normalized_doi = ? LIMIT 1",
            (ndoi,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def has_doi(self, doi: str) -> bool:
        ndoi = _normalize_doi(doi)
        if not ndoi:
            return False
        r = self._conn.execute(
            "SELECT 1 FROM paper_metadata WHERE normalized_doi = ? LIMIT 1", (ndoi,)
        ).fetchone()
        return r is not None

    def has_title(self, title: str) -> bool:
        nt = _normalize_title(title)
        if not nt:
            return False
        r = self._conn.execute(
            "SELECT 1 FROM paper_metadata WHERE normalized_title = ? LIMIT 1", (nt,)
        ).fetchone()
        return r is not None

    def all_dois(self) -> Set[str]:
        """返回所有非空的 normalized_doi 集合。"""
        rows = self._conn.execute(
            "SELECT DISTINCT normalized_doi FROM paper_metadata WHERE normalized_doi != ''"
        ).fetchall()
        return {r[0] for r in rows}

    def all_normalized_titles(self) -> Set[str]:
        """返回所有非空的 normalized_title 集合。"""
        rows = self._conn.execute(
            "SELECT DISTINCT normalized_title FROM paper_metadata WHERE normalized_title != ''"
        ).fetchall()
        return {r[0] for r in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM paper_metadata").fetchone()[0]

    def all_paper_ids_with_doi(self) -> Set[str]:
        """返回已有 DOI 的 paper_id 集合（用于 backfill 跳过已处理的）。"""
        rows = self._conn.execute(
            "SELECT paper_id FROM paper_metadata WHERE normalized_doi != ''"
        ).fetchall()
        return {r[0] for r in rows}

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        paper_id, doi, title, authors_str, year, source, extra_str = row
        authors = None
        if authors_str:
            try:
                authors = json.loads(authors_str)
            except Exception:
                pass
        return {
            "paper_id": paper_id,
            "doi": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "source": source,
            "authors_raw": authors_str,
            "extra_raw": extra_str,
        }

    # ── CrossRef 缓存 ─────────────────────────────────────

    def crossref_get(self, title: str) -> Optional[Dict[str, Any]]:
        """按 normalized_title 查 CrossRef 缓存，命中返回 dict，未命中返回 None。"""
        nt = _normalize_title(title)
        if not nt:
            return None
        row = self._conn.execute(
            "SELECT doi, title, authors, year, venue FROM crossref_cache WHERE normalized_title = ?",
            (nt,),
        ).fetchone()
        if not row:
            return None
        doi, cr_title, authors_str, year, venue = row
        if not doi:
            return None
        authors = None
        if authors_str:
            try:
                authors = json.loads(authors_str)
            except Exception:
                pass
        return {"doi": doi, "title": cr_title, "authors": authors, "year": year, "venue": venue}

    def crossref_put(self, title: str, result: Optional[Dict[str, Any]]) -> None:
        """写入 CrossRef 查询结果到缓存。result=None 表示查询无结果（负缓存）。"""
        nt = _normalize_title(title)
        if not nt:
            return
        if result:
            doi = result.get("doi") or ""
            cr_title = result.get("title") or ""
            authors = result.get("authors")
            authors_str = json.dumps(authors, ensure_ascii=False) if authors else ""
            year = result.get("year")
            venue = result.get("venue") or ""
        else:
            doi = cr_title = authors_str = venue = ""
            year = None
        self._conn.execute(
            "INSERT INTO crossref_cache (normalized_title, doi, title, authors, year, venue) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(normalized_title) DO UPDATE SET "
            "doi=excluded.doi, title=excluded.title, authors=excluded.authors, "
            "year=excluded.year, venue=excluded.venue, created_at=julianday('now')",
            (nt, doi, cr_title, authors_str, year, venue),
        )
        self._conn.commit()

    def crossref_has(self, title: str) -> bool:
        """是否已缓存（包括负缓存）。"""
        nt = _normalize_title(title)
        if not nt:
            return False
        r = self._conn.execute(
            "SELECT 1 FROM crossref_cache WHERE normalized_title = ? LIMIT 1", (nt,)
        ).fetchone()
        return r is not None

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# 全局单例
paper_meta_store = PaperMetadataStore()
