"""
论文元数据持久化存储。

提供 DOI / title / authors / year 的持久化查询与写入。
底层存储已迁移至 data/rag.db (paper_metadata / crossref_cache 表)，通过 SQLModel 访问。

特性:
- SQLModel + shared engine，连接池线程安全
- DOI + normalized_title 双索引，O(1) 查询
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
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlmodel import Session, select, and_

from src.db.engine import get_engine
from src.db.models import CrossrefCache, PaperMetadata

logger = logging.getLogger(__name__)

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
    """SQLModel-backed paper metadata store (singleton)."""

    _instance: Optional["PaperMetadataStore"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._maybe_migrate_json()
        self._initialized = True
        count = self.count()
        logger.info("PaperMetadataStore ready (%d entries)", count)

    def _maybe_migrate_json(self) -> None:
        """首次启动：若 SQLite 为空且旧 JSON 存在，自动导入。"""
        if self.count() > 0:
            return
        if not _JSON_PATH.exists():
            return
        try:
            with open(_JSON_PATH, "r", encoding="utf-8") as f:
                data: Dict[str, Dict[str, Any]] = json.load(f)
            if not data:
                return
            records = [(pid, meta) for pid, meta in data.items()]
            imported = self.upsert_batch(records)
            logger.info("Migrated %d entries from paper_metadata.json → SQLite", imported)
        except Exception as e:
            logger.warning("Failed to migrate paper_metadata.json: %s", e)

    # ── 写入 ──────────────────────────────────────────────────────────────────

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
        authors_str = (
            json.dumps(authors, ensure_ascii=False)
            if authors
            else (existing.get("authors_raw") if existing else "") or ""
        )
        yr = year or (existing.get("year") if existing else None)
        src = source or (existing.get("source") if existing else "") or ""
        ext = (
            json.dumps(extra or {}, ensure_ascii=False)
            if extra
            else (existing.get("extra_raw") if existing else "{}") or "{}"
        )

        with Session(get_engine()) as session:
            row = session.get(PaperMetadata, paper_id)
            if row is None:
                row = PaperMetadata(
                    paper_id=paper_id,
                    doi=doi,
                    normalized_doi=_normalize_doi(doi),
                    title=title,
                    normalized_title=_normalize_title(title),
                    authors=authors_str,
                    year=yr,
                    source=src,
                    extra=ext,
                )
                session.add(row)
            else:
                row.doi = doi
                row.normalized_doi = _normalize_doi(doi)
                row.title = title
                row.normalized_title = _normalize_title(title)
                row.authors = authors_str
                row.year = yr
                row.source = src
                row.extra = ext
                session.add(row)
            session.commit()

    def upsert_batch(self, records: List[Tuple[str, Dict[str, Any]]]) -> int:
        """批量 upsert [(paper_id, {doi, title, authors, year, source}), ...]。"""
        with Session(get_engine()) as session:
            count = 0
            for paper_id, meta in records:
                doi = meta.get("doi") or ""
                title = meta.get("title") or ""
                authors = meta.get("authors")
                authors_str = json.dumps(authors, ensure_ascii=False) if authors else ""
                year = meta.get("year")
                source = meta.get("source") or ""
                row = session.get(PaperMetadata, paper_id)
                if row is None:
                    row = PaperMetadata(
                        paper_id=paper_id,
                        doi=doi,
                        normalized_doi=_normalize_doi(doi),
                        title=title,
                        normalized_title=_normalize_title(title),
                        authors=authors_str,
                        year=int(year) if year else None,
                        source=source,
                        extra="{}",
                    )
                    session.add(row)
                else:
                    row.doi = doi
                    row.normalized_doi = _normalize_doi(doi)
                    row.title = title
                    row.normalized_title = _normalize_title(title)
                    row.authors = authors_str
                    row.year = int(year) if year else None
                    row.source = source
                    session.add(row)
                count += 1
            session.commit()
        return count

    # ── 查询 ──────────────────────────────────────────────────────────────────

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """按 paper_id 查询单条。"""
        with Session(get_engine()) as session:
            row = session.get(PaperMetadata, paper_id)
        if not row:
            return None
        return self._row_to_dict(row)

    def get_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """按 DOI 查询。"""
        ndoi = _normalize_doi(doi)
        if not ndoi:
            return None
        with Session(get_engine()) as session:
            stmt = select(PaperMetadata).where(PaperMetadata.normalized_doi == ndoi).limit(1)
            row = session.exec(stmt).first()
        if not row:
            return None
        return self._row_to_dict(row)

    def has_doi(self, doi: str) -> bool:
        ndoi = _normalize_doi(doi)
        if not ndoi:
            return False
        with Session(get_engine()) as session:
            stmt = select(PaperMetadata).where(PaperMetadata.normalized_doi == ndoi).limit(1)
            return session.exec(stmt).first() is not None

    def has_title(self, title: str) -> bool:
        nt = _normalize_title(title)
        if not nt:
            return False
        with Session(get_engine()) as session:
            stmt = select(PaperMetadata).where(PaperMetadata.normalized_title == nt).limit(1)
            return session.exec(stmt).first() is not None

    def all_dois(self) -> Set[str]:
        """返回所有非空的 normalized_doi 集合。"""
        with Session(get_engine()) as session:
            rows = session.exec(
                select(PaperMetadata.normalized_doi).where(PaperMetadata.normalized_doi != "")
            ).all()
        return set(rows)

    def all_normalized_titles(self) -> Set[str]:
        """返回所有非空的 normalized_title 集合。"""
        with Session(get_engine()) as session:
            rows = session.exec(
                select(PaperMetadata.normalized_title).where(PaperMetadata.normalized_title != "")
            ).all()
        return set(rows)

    def count(self) -> int:
        with Session(get_engine()) as session:
            from sqlmodel import func
            result = session.exec(select(func.count()).select_from(PaperMetadata)).one()
            return result

    def all_paper_ids_with_doi(self) -> Set[str]:
        """返回已有 DOI 的 paper_id 集合（用于 backfill 跳过已处理的）。"""
        with Session(get_engine()) as session:
            rows = session.exec(
                select(PaperMetadata.paper_id).where(PaperMetadata.normalized_doi != "")
            ).all()
        return set(rows)

    def _row_to_dict(self, row: PaperMetadata) -> Dict[str, Any]:
        authors = None
        if row.authors:
            try:
                authors = json.loads(row.authors)
            except Exception:
                pass
        return {
            "paper_id": row.paper_id,
            "doi": row.doi,
            "title": row.title,
            "authors": authors,
            "year": row.year,
            "source": row.source,
            "authors_raw": row.authors,
            "extra_raw": row.extra,
        }

    # ── CrossRef 缓存 ─────────────────────────────────────────────────────────

    def crossref_get(self, title: str) -> Optional[Dict[str, Any]]:
        """按 normalized_title 查 CrossRef 缓存，命中返回 dict，未命中返回 None。"""
        nt = _normalize_title(title)
        if not nt:
            return None
        with Session(get_engine()) as session:
            row = session.get(CrossrefCache, nt)
        if not row or not row.doi:
            return None
        authors = None
        if row.authors:
            try:
                authors = json.loads(row.authors)
            except Exception:
                pass
        return {"doi": row.doi, "title": row.title, "authors": authors, "year": row.year, "venue": row.venue}

    def crossref_put(self, title: str, result: Optional[Dict[str, Any]]) -> None:
        """写入 CrossRef 查询结果到缓存。result=None 表示查询无结果（负缓存）。"""
        import time as _time
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
        with Session(get_engine()) as session:
            row = session.get(CrossrefCache, nt)
            if row is None:
                row = CrossrefCache(
                    normalized_title=nt,
                    doi=doi,
                    title=cr_title,
                    authors=authors_str,
                    year=year,
                    venue=venue,
                    created_at=_time.time(),
                )
                session.add(row)
            else:
                row.doi = doi
                row.title = cr_title
                row.authors = authors_str
                row.year = year
                row.venue = venue
                row.created_at = _time.time()
                session.add(row)
            session.commit()

    def crossref_has(self, title: str) -> bool:
        """是否已缓存（包括负缓存）。"""
        nt = _normalize_title(title)
        if not nt:
            return False
        with Session(get_engine()) as session:
            return session.get(CrossrefCache, nt) is not None

    def close(self) -> None:
        """No-op: connection lifecycle managed by the shared engine."""
        pass


# 全局单例
paper_meta_store = PaperMetadataStore()
