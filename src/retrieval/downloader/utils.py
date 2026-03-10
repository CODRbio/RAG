"""
Standalone utilities for the scholar downloader (no PaperDownloader dependency).
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from src.log import get_logger

logger = get_logger(__name__)


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


# ── Supplementary classification helpers ──────────────────────────────────────

_SUPP_PHRASES = (
    "supplementary material",
    "supplemental material",
    "supporting information",
    "supplementary appendix",
    "supplementary data",
    "supplementary figures",
    "supplementary tables",
    "supplementary methods",
    "supplementary notes",
    "supporting material",
    "supporting information for",
    "supplementary material for",
    "supplementary material of",
    "supplementary information",
    "supplemental information",
    "supplemental data",
    "supplemental figures",
    "supplemental tables",
)


def _get_page_no(elem: Any) -> int:
    """Get page number (1-based) from a Docling element's provenance."""
    prov = getattr(elem, "prov", None) or []
    if not prov:
        return 1
    p = prov[0] if isinstance(prov, (list, tuple)) else prov
    return int(getattr(p, "page_no", 1) or 1)


def extract_pdf_leading_words(filepath: str, max_words: int = 200) -> str:
    """
    Extract the first `max_words` words from a PDF using Docling (project standard).
    Reads page 1 first; if that yields fewer than 20 words, also includes page 2.
    Uses a light pipeline (no OCR, no table structure) for speed.
    Returns normalised whitespace plain text.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError as e:
        logger.warning("Docling not available for PDF leading words: %s", e)
        return ""

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(filepath))
        doc = result.document
    except Exception as exc:
        logger.debug("Docling convert failed for %s: %s", filepath, exc)
        return ""

    texts = getattr(doc, "texts", []) or []
    # Group by page (1-based), then take page 1 first, then page 2
    by_page: Dict[int, list[str]] = {}
    for t in texts:
        page_no = _get_page_no(t)
        txt = str(getattr(t, "text", "") or "").strip()
        if txt:
            by_page.setdefault(page_no, []).append(txt)

    collected_words: list[str] = []
    for page_no in sorted(by_page.keys())[:2]:
        page_text = " ".join(by_page[page_no])
        collected_words.extend(page_text.split())
        if len(collected_words) >= max_words:
            break
    return " ".join(collected_words[:max_words])


def detect_obvious_supplementary(text: str) -> Tuple[bool, str]:
    """
    Rule-based check — no LLM required.
    Returns (is_supplementary, reason).
    Matches on the first ~600 characters (header region) only, case-insensitive.
    """
    header = text[:600].lower()
    for phrase in _SUPP_PHRASES:
        if phrase in header:
            return True, f"obvious_phrase:{phrase}"
    return False, ""


def classify_pdf_supplementary(
    filepath: str,
    *,
    ultra_lite_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a freshly-downloaded PDF as supplementary material or not.

    Strategy:
    1. Extract leading words.
    2. Run rule-based obvious-phrase check.
    3. If no obvious match, call ultra-lite LLM.
    4. Fail-closed: any error returns is_supplementary=False.

    Returns a dict with keys:
      is_supplementary: bool
      reason: str
      evidence_text: str  (the extracted leading words)
    """
    leading = extract_pdf_leading_words(filepath)
    base: Dict[str, Any] = {
        "is_supplementary": False,
        "reason": "",
        "evidence_text": leading,
    }

    if not leading:
        base["reason"] = "no_text_extracted"
        return base

    is_supp, rule_reason = detect_obvious_supplementary(leading)
    if is_supp:
        base["is_supplementary"] = True
        base["reason"] = rule_reason
        return base

    # ── Ultra-lite LLM fallback ──────────────────────────────────────────────
    try:
        from src.llm.llm_manager import get_manager
        from src.utils.prompt_manager import PromptManager

        manager = get_manager()
        client = manager.get_ultra_lite_client(ultra_lite_provider)
        pm = PromptManager()
        prompt = pm.render("downloader_supplementary_classify.txt", leading_text=leading)
        resp = client.chat(messages=[{"role": "user", "content": prompt}])
        raw = (resp.get("final_text") or "").strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())

        parsed = json.loads(raw)
        base["is_supplementary"] = bool(parsed.get("is_supplementary", False))
        base["reason"] = str(parsed.get("reason", "llm_classified"))
    except Exception as exc:
        logger.debug("supplementary LLM classification failed (non-fatal): %s", exc)
        base["reason"] = "llm_failed_or_skipped"

    return base
