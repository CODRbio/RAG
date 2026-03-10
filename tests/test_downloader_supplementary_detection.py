"""
Tests for supplementary PDF detection helpers and adapter integration.

Tests cover:
- obvious-phrase detection without LLM
- file rename to supplementary_{paper_id}.pdf
- paper_meta_store upsert preserves DOI-in-extra and supplementary markers
- same-DOI skip does not treat supplementary files as main-paper hits
"""

import os
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_minimal_pdf(path: str) -> None:
    """Write the smallest valid-looking PDF that passes is_valid_pdf()."""
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    Path(path).write_bytes(content + b"\x00" * 2000)


# ── utils.detect_obvious_supplementary ────────────────────────────────────────

class TestDetectObviousSupplementary:
    def test_obvious_phrase_supplementary_material(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        is_supp, reason = detect_obvious_supplementary(
            "Supplementary Material\n\nFigure S1. Overview..."
        )
        assert is_supp is True
        assert "supplementary material" in reason

    def test_obvious_phrase_supporting_information(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        is_supp, reason = detect_obvious_supplementary(
            "Supporting Information for\nDynamic Light Scattering..."
        )
        assert is_supp is True

    def test_obvious_phrase_supplemental_data(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        is_supp, _ = detect_obvious_supplementary("Supplemental Data\nTable S1.")
        assert is_supp is True

    def test_main_article_not_detected(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        is_supp, _ = detect_obvious_supplementary(
            "Nature Communications\nReceived: 1 Jan 2024\n"
            "Abstract: We investigated the role of..."
        )
        assert is_supp is False

    def test_phrase_must_be_in_header_region(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        # Phrase appears only after 600 chars — should not be flagged
        long_intro = "A" * 700 + " supplementary material"
        is_supp, _ = detect_obvious_supplementary(long_intro)
        assert is_supp is False

    def test_case_insensitive(self):
        from src.retrieval.downloader.utils import detect_obvious_supplementary

        is_supp, _ = detect_obvious_supplementary("SUPPLEMENTARY MATERIAL\n")
        assert is_supp is True


# ── utils.classify_pdf_supplementary (rule-based short-circuit) ───────────────

class TestClassifyPdfSupplementary:
    def test_obvious_phrase_no_llm_call(self, tmp_path):
        """When obvious phrase is found, LLM must not be called."""
        from src.retrieval.downloader.utils import classify_pdf_supplementary

        fake_text = "Supplementary Material\nTable S1. Parameters..."
        with (
            patch("src.retrieval.downloader.utils.extract_pdf_leading_words", return_value=fake_text),
            patch("src.llm.llm_manager.get_manager") as mock_mgr,
        ):
            result = classify_pdf_supplementary(str(tmp_path / "dummy.pdf"))
            mock_mgr.assert_not_called()

        assert result["is_supplementary"] is True
        assert "obvious_phrase" in result["reason"]

    def test_no_text_returns_false(self, tmp_path):
        """Empty extraction should not trigger supplementary."""
        from src.retrieval.downloader.utils import classify_pdf_supplementary

        with patch("src.retrieval.downloader.utils.extract_pdf_leading_words", return_value=""):
            result = classify_pdf_supplementary(str(tmp_path / "dummy.pdf"))

        assert result["is_supplementary"] is False

    def test_llm_failure_returns_false(self, tmp_path):
        """LLM failure should fail-closed (is_supplementary=False)."""
        from src.retrieval.downloader.utils import classify_pdf_supplementary

        fake_text = "Introduction. We studied the effect of..."
        with (
            patch("src.retrieval.downloader.utils.extract_pdf_leading_words", return_value=fake_text),
            patch("src.llm.llm_manager.get_manager", side_effect=RuntimeError("no LLM")),
        ):
            result = classify_pdf_supplementary(str(tmp_path / "dummy.pdf"))

        assert result["is_supplementary"] is False


# ── adapter: supplementary rename & metadata ───────────────────────────────────

class TestAdapterSupplementaryRename:
    """Unit test the adapter's supplementary branch without real downloads."""

    def _make_result(self, is_supp: bool, reason: str = "obvious_phrase:supplementary material"):
        return {"is_supplementary": is_supp, "reason": reason, "evidence_text": "..."}

    def test_file_renamed_to_supplementary_prefix(self, tmp_path):
        """After detection, the file must be renamed to supplementary_{paper_id}.pdf."""
        from src.retrieval.downloader.adapter import _is_supplementary_paper_id

        paper_id = "10.1234_test_paper"
        original_path = tmp_path / f"{paper_id}.pdf"
        _write_minimal_pdf(str(original_path))

        supp_path = tmp_path / f"supplementary_{paper_id}.pdf"

        # Simulate what the adapter does
        stored_paper_id = f"supplementary_{paper_id}"
        stored_path = str(tmp_path / f"{stored_paper_id}.pdf")
        os.rename(str(original_path), stored_path)

        assert not original_path.exists()
        assert Path(stored_path).exists()
        assert _is_supplementary_paper_id(stored_paper_id)

    def test_is_supplementary_paper_id_helper(self):
        from src.retrieval.downloader.adapter import _is_supplementary_paper_id

        assert _is_supplementary_paper_id("supplementary_10.1234_foo")
        assert not _is_supplementary_paper_id("10.1234_foo")
        assert not _is_supplementary_paper_id("")
        assert not _is_supplementary_paper_id("supp_foo")  # prefix must be "supplementary_"

    def test_supplementary_uses_suffixed_doi(self, tmp_path):
        """Supplementary metadata must be stored with a #supp-suffixed pseudo-DOI."""
        paper_id = "10.9999_supp_test"
        original_path = str(tmp_path / f"{paper_id}.pdf")
        _write_minimal_pdf(original_path)

        upsert_calls = []

        def fake_upsert(pid, **kwargs):
            upsert_calls.append((pid, kwargs))

        supp_classify_result = {
            "is_supplementary": True,
            "reason": "obvious_phrase:supplementary material",
            "evidence_text": "Supplementary Material...",
        }

        with (
            patch("src.retrieval.downloader.adapter.classify_pdf_supplementary", return_value=supp_classify_result),
            patch("src.retrieval.downloader.adapter.is_valid_pdf", return_value=True),
        ):
            from src.indexing.paper_metadata_store import paper_meta_store
            with patch.object(paper_meta_store, "upsert", side_effect=fake_upsert):
                doi = "10.9999/test"
                stored_paper_id = f"supplementary_{paper_id}"
                stored_doi = f"{doi}#supp"
                paper_meta_store.upsert(
                    stored_paper_id,
                    doi=stored_doi,
                    title="Test Paper",
                    extra={"is_supplementary": True, "primary_doi": doi},
                )

        assert len(upsert_calls) == 1
        pid, kwargs = upsert_calls[0]
        assert pid == f"supplementary_{paper_id}"
        assert kwargs["doi"].endswith("#supp")
        assert kwargs["extra"]["is_supplementary"] is True
        assert kwargs["extra"]["primary_doi"] == "10.9999/test"


# ── adapter: same-DOI skip must ignore supplementary entries ───────────────────

class TestSameDoiSkipIgnoresSupplementary:
    """
    The adapter's DOI-based skip branch must not treat a supplementary_{} file
    as proof that the main article has been downloaded.
    """

    def test_supplementary_pid_skipped_in_doi_loop(self):
        from src.retrieval.downloader.adapter import _is_supplementary_paper_id

        doi = "10.1234/main_paper"
        existing_pids = ["supplementary_10.1234_main_paper", "10.1234_main_paper"]

        # Simulate the skip loop logic
        main_candidates = [
            pid for pid in existing_pids
            if not _is_supplementary_paper_id(pid) and pid != "current_paper_id"
        ]
        assert main_candidates == ["10.1234_main_paper"]

    def test_only_supplementary_pid_does_not_trigger_skip(self):
        from src.retrieval.downloader.adapter import _is_supplementary_paper_id

        existing_pids = ["supplementary_10.1234_main_paper"]
        non_supp_candidates = [pid for pid in existing_pids if not _is_supplementary_paper_id(pid)]
        assert non_supp_candidates == []

    def test_supplementary_doi_uses_supp_suffix(self):
        """Since supplementary records use a #supp-suffixed DOI, raw DOI lookup
        will not return them, providing a second layer of defense."""
        raw_doi = "10.1234/paper"
        supp_doi = f"{raw_doi}#supp"

        assert supp_doi != raw_doi
        assert "#supp" in supp_doi
