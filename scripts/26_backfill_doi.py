#!/usr/bin/env python
"""
步骤26: 为已入库论文回填 DOI 和完整标题

基于 Docling 已有 parse 产物（data/parsed/*/enriched.json），
而非重新打开 PDF，保持与已有框架一致。

策略：
1. 从 enriched.json 的 content_flow 前 N 块正则提取 DOI（零成本）
2. 若提取到 DOI → CrossRef /works/{doi} 获取完整元数据
3. 若未提取到 → CrossRef /works?query.bibliographic= 按文件名搜索

持久化: data/paper_metadata.db（SQLite, 增量式, 可重跑）

用法:
    python scripts/26_backfill_doi.py                  # 正常运行
    python scripts/26_backfill_doi.py --dry-run        # 仅扫描，不调用 CrossRef
    python scripts/26_backfill_doi.py --force           # 忽略已有结果，全量重跑
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

import requests

from config.settings import settings
from src.log import get_logger
from src.indexing.paper_metadata_store import PaperMetadataStore

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────

DOI_RE = re.compile(
    r"(?:doi[:\s]*|(?:https?://)?(?:dx\.)?doi\.org/)"
    r"?(10\.\d{4,9}/[^\s,;)\]\"'<>]+[^\s,;)\]\"'<>.:])",
    re.IGNORECASE,
)

CROSSREF_BASE = "https://api.crossref.org"
CROSSREF_MAILTO = ""
CROSSREF_TIMEOUT = 20
RATE_LIMIT_SLEEP = 0.3
SCAN_BLOCKS = 30  # enriched.json 前多少个 block 内搜索 DOI


# ── 从 enriched.json 提取 ─────────────────────────────

def extract_doi_from_enriched(content_flow: List[Dict], max_blocks: int = SCAN_BLOCKS) -> Optional[str]:
    """从 content_flow 前 N 个 block 的 text 中正则提取 DOI"""
    for block in content_flow[:max_blocks]:
        text = block.get("text")
        if not text or not isinstance(text, str):
            continue
        m = DOI_RE.search(text)
        if m:
            return m.group(1).rstrip(".:")
    return None


def extract_title_from_enriched(content_flow: List[Dict]) -> Optional[str]:
    """
    从 content_flow 头部提取论文标题。

    Pass 1: heading 块中找长度 10-300 且非分类标签的第一个。
    Pass 2: 若 heading 没命中，退回到 page_index==0 的 text 块中
            找第一个看起来像标题的（15-300 字符，排除 URL / 作者行等噪声）。
    """
    skip_labels = {
        "original article", "research article", "review", "letter",
        "communication", "open", "open access", "article", "brief report",
        "research paper", "full paper", "short communication",
        "abstract", "abstract:", "data note", "introduction",
        "keywords", "highlights", "graphical abstract", "contents",
    }

    # Pass 1: heading blocks
    for block in content_flow[:15]:
        bt = (block.get("block_type") or "").lower()
        text = (block.get("text") or "").strip()
        if bt != "heading" or not text:
            continue
        if text.lower() in skip_labels:
            continue
        if 10 <= len(text) <= 300:
            return text

    # Pass 2: text blocks on page 0 — find first title-like block
    for block in content_flow[:15]:
        bt = (block.get("block_type") or "").lower()
        if bt not in ("text", ""):
            continue
        text = (block.get("text") or "").strip()
        page = block.get("page_index", 0)
        if page != 0 or not text:
            continue
        if not (15 <= len(text) <= 300):
            continue
        tl = text.lower()
        if tl in skip_labels:
            continue
        # skip URLs, copyright lines, author lines with affiliations
        if tl.startswith(("http", "www.", "©", "copyright")):
            continue
        if re.search(r"\d{4}\s*(international|society|elsevier|springer|wiley|nature)", tl):
            continue
        # author lines typically have many superscript numbers/commas
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.08:
            continue
        return text

    return None


# ── CrossRef API ──────────────────────────────────────

def _crossref_headers() -> Dict[str, str]:
    ua = "RAG-DOI-Backfill/1.0"
    if CROSSREF_MAILTO:
        ua += f" (mailto:{CROSSREF_MAILTO})"
    return {"User-Agent": ua}


def _crossref_get(url: str, params: Optional[Dict] = None) -> Optional[Dict]:
    params = params or {}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    for attempt in range(3):
        try:
            resp = requests.get(
                url, params=params, headers=_crossref_headers(), timeout=CROSSREF_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("  CrossRef 429, retry in %ds", wait)
                time.sleep(wait)
                continue
            logger.warning("  CrossRef HTTP %d for %s", resp.status_code, url[:120])
            return None
        except requests.RequestException as e:
            logger.warning("  CrossRef error (attempt %d): %s", attempt + 1, e)
            time.sleep(1)
    return None


def _parse_crossref_item(item: Dict) -> Dict[str, Any]:
    titles = item.get("title", [])
    title = titles[0] if titles else None
    doi = item.get("DOI")

    authors: List[str] = []
    for a in item.get("author", []):
        name = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if name:
            authors.append(name)

    year = None
    for date_field in ("published-print", "published-online", "created"):
        dp = (item.get(date_field) or {}).get("date-parts", [[]])
        if dp and dp[0] and dp[0][0]:
            year = dp[0][0]
            break

    return {"doi": doi, "title": title, "authors": authors or None, "year": year}


def crossref_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    encoded = requests.utils.quote(doi, safe="")
    data = _crossref_get(f"{CROSSREF_BASE}/works/{encoded}")
    if data:
        return _parse_crossref_item(data.get("message", {}))
    return None


def crossref_by_title(query: str, rows: int = 5) -> Optional[Dict[str, Any]]:
    data = _crossref_get(
        f"{CROSSREF_BASE}/works",
        params={"query.bibliographic": query, "rows": rows},
    )
    if not data:
        return None
    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    query_words = _word_set(query)
    best, best_score = None, -1.0
    for item in items:
        parsed = _parse_crossref_item(item)
        score = _title_overlap(query_words, parsed.get("title") or "")
        if score > best_score:
            best_score = score
            best = parsed

    if best_score < 0.35:
        logger.info("  CrossRef best score %.2f < 0.35, rejected", best_score)
        return None
    return best


# ── 相似度 ────────────────────────────────────────────

def _word_set(text: str) -> set:
    return {w for w in re.sub(r"[^a-z0-9\s]", "", text.lower()).split() if len(w) >= 3}


def _title_overlap(query_words: set, crossref_title: str) -> float:
    if not query_words:
        return 0.0
    cr_words = _word_set(crossref_title)
    if not cr_words:
        return 0.0
    return len(query_words & cr_words) / len(query_words)


# ── 文件名 → 查询 ────────────────────────────────────

def query_from_filename(paper_id: str) -> str:
    q = paper_id.replace("_", " ").replace("-", " ")
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ── 核心 ──────────────────────────────────────────────

def process_paper(
    paper_id: str,
    content_flow: List[Dict],
    existing: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "doi": None, "title": None, "authors": None, "year": None, "source": None,
    }

    # ① 正则提取 DOI
    doi = extract_doi_from_enriched(content_flow)
    title = extract_title_from_enriched(content_flow)

    if doi:
        logger.info("  [enriched] DOI = %s", doi)
        result["doi"] = doi
        result["title"] = title
        result["source"] = "enriched_regex"

        # 仅当本地无标题且无历史记录时才调 CrossRef
        if not title and not dry_run:
            meta = crossref_by_doi(doi)
            if meta and meta.get("title"):
                result["title"] = meta["title"]
                result["authors"] = meta.get("authors")
                result["year"] = meta.get("year")
                result["source"] = "crossref_by_doi"
        return result

    # ② 本地无 DOI → 先看历史记录里有没有
    if existing and existing.get("doi"):
        logger.info("  [cached] reuse previous DOI = %s", existing["doi"])
        result["doi"] = existing["doi"]
        result["title"] = title or existing.get("title")
        result["authors"] = existing.get("authors")
        result["year"] = existing.get("year")
        result["source"] = "cached"
        return result

    # ③ 真正的新纸、本地也提不到 → CrossRef
    if dry_run:
        result["source"] = "need_crossref"
        result["title"] = title
        return result

    query = title or query_from_filename(paper_id)
    logger.info("  [CrossRef] query = '%s'", query[:80])
    meta = crossref_by_title(query)

    if meta and meta.get("doi"):
        result.update(meta)
        result["source"] = "crossref_by_title"
    else:
        if title:
            result["title"] = title
            result["source"] = "enriched_title_only"
        logger.warning("  [MISS] no DOI for %s", paper_id[:60])

    return result


# ── main ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill DOI & title from enriched.json + CrossRef")
    parser.add_argument("--dry-run", action="store_true", help="Only scan enriched.json, skip CrossRef")
    parser.add_argument("--force", action="store_true", help="Re-process all (ignore existing metadata)")
    args = parser.parse_args()

    settings.path.ensure_dirs()
    parsed_dir = settings.path.parsed
    store = PaperMetadataStore()

    logger.info("=" * 60)
    logger.info("DOI & Title Backfill  (dry_run=%s, force=%s)", args.dry_run, args.force)
    logger.info("=" * 60)
    logger.info("Parsed dir: %s | DB entries: %d", parsed_dir, store.count())

    enriched_files = sorted(parsed_dir.rglob("enriched.json"))
    logger.info("enriched.json count: %d", len(enriched_files))

    if not enriched_files:
        logger.warning("No enriched.json found. Run ingestion first.")
        return

    stats: Dict[str, int] = {}

    for i, json_path in enumerate(enriched_files, 1):
        paper_id = json_path.parent.name

        existing = store.get(paper_id)
        if not args.force and existing and existing.get("doi") and existing.get("title"):
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue

        logger.info("[%d/%d] %s", i, len(enriched_files), paper_id[:70])

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            logger.warning("  Failed to read %s: %s", json_path, e)
            stats["error"] = stats.get("error", 0) + 1
            continue

        content_flow = doc.get("content_flow", [])

        # DOI already known but title missing → try enriched.json locally first
        if existing and existing.get("doi") and not existing.get("title"):
            title = extract_title_from_enriched(content_flow)
            if title:
                store.upsert(paper_id, title=title)
                stats["title_from_enriched"] = stats.get("title_from_enriched", 0) + 1
                continue
            if not args.dry_run:
                doi = existing["doi"]
                logger.info("  [CrossRef] title missing locally, fetching by DOI %s", doi)
                meta = crossref_by_doi(doi)
                if meta and meta.get("title"):
                    store.upsert(
                        paper_id,
                        title=meta["title"],
                        authors=meta.get("authors"),
                        year=meta.get("year"),
                        source="crossref_by_doi",
                    )
                stats["title_from_crossref"] = stats.get("title_from_crossref", 0) + 1
                time.sleep(RATE_LIMIT_SLEEP)
            continue

        result = process_paper(paper_id, content_flow, existing=existing, dry_run=args.dry_run)
        store.upsert(
            paper_id,
            doi=result.get("doi"),
            title=result.get("title"),
            authors=result.get("authors"),
            year=result.get("year"),
            source=result.get("source"),
        )

        src = result.get("source") or "miss"
        stats[src] = stats.get(src, 0) + 1

        if not args.dry_run and src in ("crossref_by_doi", "crossref_by_title"):
            time.sleep(RATE_LIMIT_SLEEP)

    total = store.count()
    doi_n = len(store.all_dois())
    title_n = len(store.all_normalized_titles())

    logger.info("=" * 60)
    logger.info("DB     : %s (%d entries)", store._db_path, total)
    logger.info("Stats  : %s", {k: v for k, v in stats.items() if v})
    logger.info("DOI    : %d / %d  (%.1f%%)", doi_n, total, 100 * doi_n / max(total, 1))
    logger.info("Title  : %d / %d  (%.1f%%)", title_n, total, 100 * title_n / max(total, 1))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
