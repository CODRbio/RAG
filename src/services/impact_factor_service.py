"""
SQL-backed impact factor index built from docs/impact_factor.xlsx.
Rebuilds automatically when the workbook file changes (mtime/size/hash).
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlmodel import Session, select

from src.db.engine import get_engine
from src.db.models import ImpactFactorIndexMeta, ImpactFactorJournal
from src.log import get_logger

logger = get_logger(__name__)

# Default path relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX_PATH = PROJECT_ROOT / "docs" / "impact_factor.xlsx"

# Excel column names (JCR-style export)
COL_JOURNAL_NAME = "Journal name"
COL_JCR_ABBREVIATION = "JCR Abbreviation"
COL_PUBLISHER = "Publisher"
COL_ISSN = "ISSN"
COL_EISSN = "eISSN"
COL_CATEGORY = "Category"
COL_EDITION = "Edition"
COL_IMPACT_FACTOR = "2024 JIF"
COL_JIF_QUARTILE = "JIF Quartile"
COL_JIF_RANK = "JIF Rank"
COL_JIF_5YEAR = "5 Year JIF"


def _normalize_journal_name(name: str) -> str:
    """Lowercase, strip, collapse spaces and common punctuation variants."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,;:\-–—]\s*", " ", s)
    s = " ".join(s.split())
    return s


def _normalize_issn(value: Any) -> str:
    """Digits only for ISSN matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().replace("-", "").replace(" ", "")
    return "".join(c for c in s if c.isdigit())[:8]


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _file_fingerprint(path: Path) -> tuple[float, int, str]:
    """Return (mtime, size, sha256_hex)."""
    stat = path.stat()
    mtime = stat.st_mtime
    size = stat.st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return mtime, size, h.hexdigest()


def _load_xlsx(path: Path) -> pd.DataFrame:
    """Load impact factor workbook; normalize column names if needed."""
    df = pd.read_excel(path, engine="openpyxl")
    # Map possible column name variants to canonical names
    renames = {}
    for c in df.columns:
        c_str = str(c).strip()
        if "journal" in c_str.lower() and "name" in c_str.lower():
            renames[c] = COL_JOURNAL_NAME
        elif "jcr" in c_str.lower() and "abbreviation" in c_str.lower():
            renames[c] = COL_JCR_ABBREVIATION
        elif c_str == "Publisher" or c_str == "publisher":
            renames[c] = COL_PUBLISHER
        elif c_str == "ISSN" or c_str == "issn":
            renames[c] = COL_ISSN
        elif c_str == "eISSN" or c_str.lower() == "eissn":
            renames[c] = COL_EISSN
        elif c_str == "Category" or c_str == "category":
            renames[c] = COL_CATEGORY
        elif c_str == "Edition" or c_str == "edition":
            renames[c] = COL_EDITION
        elif "2024" in c_str and "JIF" in c_str:
            renames[c] = COL_IMPACT_FACTOR
        elif "quartile" in c_str.lower():
            renames[c] = COL_JIF_QUARTILE
        elif "rank" in c_str.lower() and "jif" in c_str.lower():
            renames[c] = COL_JIF_RANK
        elif "5 year" in c_str.lower() or "5 year jif" in c_str.lower():
            renames[c] = COL_JIF_5YEAR
    if renames:
        df = df.rename(columns=renames)
    return df


def _rebuild_index(xlsx_path: Path, source_file_key: str) -> int:
    """Parse Excel, replace all rows in impact_factor_journals for this source, update meta. Returns row_count."""
    df = _load_xlsx(xlsx_path)
    mtime, size, file_hash = _file_fingerprint(xlsx_path)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    source_version = f"{int(mtime)}-{size}-{file_hash[:16]}"

    # Ensure columns exist (allow missing)
    journal_col = COL_JOURNAL_NAME if COL_JOURNAL_NAME in df.columns else df.columns[0]
    abbr_col = COL_JCR_ABBREVIATION if COL_JCR_ABBREVIATION in df.columns else None
    issn_col = COL_ISSN if COL_ISSN in df.columns else None
    eissn_col = COL_EISSN if COL_EISSN in df.columns else None
    if_col = COL_IMPACT_FACTOR if COL_IMPACT_FACTOR in df.columns else None
    quartile_col = COL_JIF_QUARTILE if COL_JIF_QUARTILE in df.columns else None
    rank_col = COL_JIF_RANK if COL_JIF_RANK in df.columns else None
    jif5_col = COL_JIF_5YEAR if COL_JIF_5YEAR in df.columns else None
    category_col = COL_CATEGORY if COL_CATEGORY in df.columns else None
    edition_col = COL_EDITION if COL_EDITION in df.columns else None
    publisher_col = COL_PUBLISHER if COL_PUBLISHER in df.columns else None

    rows: List[ImpactFactorJournal] = []
    for _, r in df.iterrows():
        jname = _safe_str(r.get(journal_col))
        if not jname:
            continue
        norm_name = _normalize_journal_name(jname)
        abbr = _safe_str(r.get(abbr_col)) if abbr_col else ""
        norm_abbr = _normalize_journal_name(abbr) if abbr else ""
        issn = _normalize_issn(r.get(issn_col)) if issn_col else ""
        eissn = _normalize_issn(r.get(eissn_col)) if eissn_col else ""
        rows.append(
            ImpactFactorJournal(
                source_file=source_file_key,
                source_version=source_version,
                journal_name=jname,
                normalized_journal_name=norm_name,
                jcr_abbreviation=abbr,
                normalized_jcr_abbreviation=norm_abbr,
                issn=issn,
                eissn=eissn,
                category=_safe_str(r.get(category_col)) if category_col else "",
                edition=_safe_str(r.get(edition_col)) if edition_col else "",
                impact_factor=_safe_float(r.get(if_col)) if if_col else None,
                jif_quartile=_safe_str(r.get(quartile_col)) if quartile_col else "",
                jif_rank=_safe_str(r.get(rank_col)) if rank_col else "",
                jif_5year=_safe_float(r.get(jif5_col)) if jif5_col else None,
                publisher=_safe_str(r.get(publisher_col)) if publisher_col else "",
            )
        )

    engine = get_engine()
    with Session(engine) as session:
        to_delete = list(session.exec(select(ImpactFactorJournal).where(ImpactFactorJournal.source_file == source_file_key)))
        for row in to_delete:
            session.delete(row)
        for row in rows:
            session.add(row)
        meta = session.get(ImpactFactorIndexMeta, source_file_key)
        if meta is None:
            meta = ImpactFactorIndexMeta(
                source_file=source_file_key,
                last_mtime=mtime,
                last_size=size,
                last_hash=file_hash,
                indexed_at=now_iso,
                row_count=len(rows),
                version=1,
            )
            session.add(meta)
        else:
            meta.last_mtime = mtime
            meta.last_size = size
            meta.last_hash = file_hash
            meta.indexed_at = now_iso
            meta.row_count = len(rows)
            meta.version += 1
            session.add(meta)
        session.commit()
    logger.info("Impact factor index rebuilt: source_file=%s rows=%d", source_file_key, len(rows))
    return len(rows)


def ensure_impact_factor_index_current(
    xlsx_path: Optional[Path] = None,
    source_file_key: Optional[str] = None,
) -> None:
    """
    If the workbook has changed since last index, rebuild the SQL index.
    Call before first lookup (e.g. at startup or first request).
    """
    path = xlsx_path or DEFAULT_XLSX_PATH
    key = source_file_key or path.name
    if not path.exists():
        logger.warning("Impact factor workbook not found: %s", path)
        return
    mtime, size, file_hash = _file_fingerprint(path)
    engine = get_engine()
    with Session(engine) as session:
        meta = session.get(ImpactFactorIndexMeta, key)
        if meta is None:
            _rebuild_index(path, key)
            return
        if meta.last_mtime != mtime or meta.last_size != size or meta.last_hash != file_hash:
            _rebuild_index(path, key)


def lookup_by_venue(venue: str, ensure_current: bool = True) -> Optional[Dict[str, Any]]:
    """
    Look up impact factor data by venue string (journal name or abbreviation).
    Matching order: normalized name, normalized JCR abbreviation, then fallback.
    """
    if ensure_current:
        ensure_impact_factor_index_current()
    if not venue or not isinstance(venue, str):
        return None
    norm = _normalize_journal_name(venue)
    if not norm:
        return None
    engine = get_engine()
    with Session(engine) as session:
        # 1. exact normalized journal name
        row = session.exec(
            select(ImpactFactorJournal).where(ImpactFactorJournal.normalized_journal_name == norm).limit(1)
        ).first()
        if row is not None:
            return _row_to_dict(row)
        # 2. exact normalized JCR abbreviation
        row = session.exec(
            select(ImpactFactorJournal).where(ImpactFactorJournal.normalized_jcr_abbreviation == norm).limit(1)
        ).first()
        if row is not None:
            return _row_to_dict(row)
        # No fuzzy/substring fallback: unknown journals must not get another journal's IF.
        # Callers should display "n.a." when lookup returns None.
    return None


def _row_to_dict(row: ImpactFactorJournal) -> Dict[str, Any]:
    return {
        "journal_name": row.journal_name or "",
        "normalized_journal_name": row.normalized_journal_name or "",
        "jcr_abbreviation": row.jcr_abbreviation or "",
        "impact_factor": row.impact_factor,
        "jif_quartile": row.jif_quartile or "",
        "jif_5year": row.jif_5year,
        "category": row.category or "",
    }


def lookup_many(venues: List[str], ensure_current: bool = True) -> Dict[str, Dict[str, Any]]:
    """Batch lookup: returns dict mapping normalized input venue -> IF data (or missing key if no match)."""
    if ensure_current:
        ensure_impact_factor_index_current()
    out: Dict[str, Dict[str, Any]] = {}
    seen_norm: Dict[str, str] = {}  # norm -> first original venue
    for v in venues:
        if not v or not isinstance(v, str):
            continue
        norm = _normalize_journal_name(v)
        if not norm or norm in out:
            continue
        seen_norm.setdefault(norm, v)
    for norm, first_venue in seen_norm.items():
        data = lookup_by_venue(norm, ensure_current=False)
        if data is not None:
            out[norm] = data
            # Also key by original venue so caller can use either
            if first_venue != norm:
                out[first_venue] = data
    return out
