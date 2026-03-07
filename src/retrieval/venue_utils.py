"""
Shared helpers to extract and normalize journal/venue names from noisy provider strings.
Used by SerpAPI, Google Scholar, and impact factor lookup so search, storage, and filtering align.
"""

from __future__ import annotations

import re
from typing import Optional


def normalize_journal_name(name: str) -> str:
    """Lowercase, strip, collapse spaces and common punctuation for consistent matching."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,;:\-–—]\s*", " ", s)
    s = " ".join(s.split())
    return s


def extract_clean_venue(summary: str) -> str:
    """
    Extract a clean journal/source name from a noisy string such as:
      "J Smith, A Doe - Nature, 2022 - Springer"
      "Nature, 123(4), 2022"
    Returns the best-effort journal name for display and IF matching; empty if none found.
    """
    if not summary or not isinstance(summary, str):
        return ""
    raw = summary.strip()
    if not raw:
        return ""

    # Split by common separators (em-dash, en-dash, hyphen with spaces)
    parts = re.split(r"\s*[–—\-]\s*", raw, maxsplit=3)
    # Often: [authors], [journal, year], [publisher]
    candidate = raw
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        # Skip obvious author-like segment (contains many commas or "et al")
        if " et al" in p.lower() or (p.count(",") >= 2 and len(p) < 80):
            continue
        # Prefer segment that contains a 4-digit year (journal, year)
        if re.search(r",?\s*(19|20)\d{2}\s*$", p):
            candidate = p
            break
        # Otherwise take first non-author-looking segment (often index 1)
        if i == 1 or (i == 0 and len(parts) > 1 and not re.search(r"^[A-Z]\.\s", p)):
            candidate = p
            break
        candidate = p

    # Strip trailing year and parenthetical (e.g. "Nature, 2022" -> "Nature", "123(4), 2022" -> before comma)
    candidate = re.sub(r",\s*(19|20)\d{2}\s*$", "", candidate)
    candidate = re.sub(r"\s*\(\d{4}\)\s*$", "", candidate)
    candidate = re.sub(r",\s*vol\.?\s*\d+.*$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r",\s*\d+\s*\(\d+\)\s*,?\s*.*$", "", candidate)

    # Strip trailing publisher-like suffix (e.g. " - Springer" already split; single segment "Journal Name, Springer")
    candidate = candidate.strip()
    if candidate.count(",") >= 1:
        # Last comma segment might be publisher
        last_part = candidate.split(",")[-1].strip()
        if last_part and not re.search(r"(19|20)\d{2}", last_part) and len(last_part) < 30:
            candidate = candidate.rsplit(",", 1)[0].strip()
    candidate = re.sub(r"\s+-\s+[A-Za-z\s]+$", "", candidate)

    return candidate.strip() if candidate else raw.strip()
