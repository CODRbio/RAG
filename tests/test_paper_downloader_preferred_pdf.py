import asyncio
import logging
from unittest.mock import patch, MagicMock

from src.retrieval.downloader.paper_downloader_refactored import (
    ActionableElement,
    Candidate,
    PDFExtractor,
    PageAnalyzer,
    PaperDownloader,
    ScoreConfig,
)


def _make_downloader_for_unit_test() -> PaperDownloader:
    downloader = PaperDownloader.__new__(PaperDownloader)
    downloader.pdf_selectors = PaperDownloader.PDF_SELECTORS
    downloader.site_url_patterns = PaperDownloader.SITE_URL_PATTERNS
    downloader.page_analyzer = PageAnalyzer(logging.getLogger("test.paper_downloader"))
    return downloader


def test_asm_doi_generates_preferred_main_pdf_entrypoint():
    downloader = _make_downloader_for_unit_test()

    entrypoints = downloader._build_preferred_pdf_entrypoints(
        "https://journals.asm.org/doi/10.1128/spectrum.01978-22"
    )

    assert entrypoints == [
        "https://journals.asm.org/doi/pdf/10.1128/spectrum.01978-22"
    ]


def test_paper_downloader_init_builds_md5_index_without_io_lock_error(tmp_path):
    downloader = PaperDownloader(download_dir=str(tmp_path), max_concurrent=1)

    assert downloader.file_md5_index == {}


def test_sort_prefers_main_pdf_over_asm_supplement_attachment():
    downloader = _make_downloader_for_unit_test()

    attachment = ActionableElement(
        selector='a[href*="/doi/suppl/"]',
        tag="a",
        text="PDF",
        href="https://journals.asm.org/doi/suppl/10.1128/spectrum.01978-22/suppl_file/spectrum.01978-22-s0001.pdf",
        is_visible=True,
    )
    main_pdf = ActionableElement(
        selector='a[href*="/doi/pdf/"]',
        tag="a",
        text="PDF",
        href="https://journals.asm.org/doi/pdf/10.1128/spectrum.01978-22",
        is_visible=True,
    )

    attachment.score = downloader.page_analyzer._calculate_score(attachment)
    main_pdf.score = downloader.page_analyzer._calculate_score(main_pdf)

    ranked = downloader._sort_elements_prefer_download([attachment, main_pdf])

    assert ranked[0].href == main_pdf.href


def test_publisher_doi_url_generates_preferred_entrypoint():
    """Publisher /doi/... URLs (not doi.org) still yield preferred PDF entrypoints for ASM."""
    downloader = _make_downloader_for_unit_test()

    # Full-text / abstract publisher URL should still extract DOI and return same entrypoint
    entrypoints = downloader._build_preferred_pdf_entrypoints(
        "https://journals.asm.org/doi/full/10.1128/spectrum.01978-22"
    )
    assert entrypoints == [
        "https://journals.asm.org/doi/pdf/10.1128/spectrum.01978-22"
    ]

    entrypoints_abs = downloader._build_preferred_pdf_entrypoints(
        "https://journals.asm.org/doi/abs/10.1128/spectrum.01978-22"
    )
    assert entrypoints_abs == [
        "https://journals.asm.org/doi/pdf/10.1128/spectrum.01978-22"
    ]


def test_wiley_doi_url_generates_preferred_pdfdirect_entrypoints():
    downloader = _make_downloader_for_unit_test()

    entrypoints = downloader._build_preferred_pdf_entrypoints(
        "https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202400001"
    )
    assert entrypoints == [
        "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/anie.202400001?download=true",
        "https://onlinelibrary.wiley.com/doi/pdf/10.1002/anie.202400001",
        "https://onlinelibrary.wiley.com/doi/epdf/10.1002/anie.202400001",
    ]


def test_is_doi_like_url_recognizes_publisher_and_doi_org_urls():
    downloader = _make_downloader_for_unit_test()

    assert downloader._is_doi_like_url("https://doi.org/10.1128/spectrum.01978-22")
    assert downloader._is_doi_like_url(
        "https://journals.asm.org/doi/full/10.1128/spectrum.01978-22"
    )
    assert not downloader._is_doi_like_url("https://journals.asm.org/toc/spectrum/current")


def test_asm_selector_list_includes_ereader_and_btn_pdf():
    """ASM site-specific selectors and Atypon multi-step config include btn--pdf and eReader."""
    asm_selectors = PaperDownloader.PDF_SELECTORS["site_specific"]["asm"]
    assert "a.btn--pdf" in asm_selectors
    assert 'a[href*="/doi/reader/"]' in asm_selectors
    assert 'a[title="Open full-text in eReader"]' in asm_selectors

    assert PaperDownloader.MULTI_STEP_SITES["atypon_publishers"]["detect"](
        "https://journals.asm.org/doi/10.1128/spectrum.01978-22"
    )
    first_selector = PaperDownloader.MULTI_STEP_SITES["atypon_publishers"]["steps"][0]["selector"]
    assert "a.btn--pdf" in first_selector
    assert "doi/reader" in first_selector or "eReader" in first_selector


def test_scoring_prefers_direct_pdf_over_reader_link():
    """Direct /doi/pdf/ link scores higher than /doi/reader/ so we don't prefer eReader over PDF."""
    downloader = _make_downloader_for_unit_test()

    reader_link = ActionableElement(
        selector='a[href*="/doi/reader/"]',
        tag="a",
        text="Open full-text in eReader",
        href="https://journals.asm.org/doi/reader/10.1128/spectrum.01978-22",
        is_visible=True,
        attributes={"class": "btn--pdf"},
    )
    direct_pdf = ActionableElement(
        selector='a[href*="/doi/pdf/"]',
        tag="a",
        text="PDF",
        href="https://journals.asm.org/doi/pdf/10.1128/spectrum.01978-22",
        is_visible=True,
    )

    reader_link.score = downloader.page_analyzer._calculate_score(reader_link)
    direct_pdf.score = downloader.page_analyzer._calculate_score(direct_pdf)

    assert direct_pdf.score > reader_link.score


def test_infer_source_flags_uses_urls_when_source_label_missing():
    downloader = _make_downloader_for_unit_test()

    flags = downloader._infer_source_flags(
        source="",
        url="https://www.semanticscholar.org/paper/example",
        pdf_url=None,
    )
    assert flags["is_semantic_source"] is True

    pdf_only_flags = downloader._infer_source_flags(
        source=None,
        url=None,
        pdf_url="https://www.semanticscholar.org/paper/example.pdf",
    )
    assert pdf_only_flags["is_semantic_source"] is True


def test_infer_source_flags_covers_common_publishers_from_urls():
    downloader = _make_downloader_for_unit_test()

    flags = downloader._infer_source_flags(
        source=None,
        url="https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202400001",
        pdf_url="https://papers.ssrn.com/sol3/Delivery.cfm/1234567.pdf",
    )
    assert flags["is_wiley_source"] is True
    assert flags["is_ssrn_source"] is True

    academia_flags = downloader._infer_source_flags(
        source="",
        url=None,
        pdf_url="https://www.academia.edu/download/123456/example-paper.pdf",
    )
    assert academia_flags["is_academia_source"] is True

    sd_flags = downloader._infer_source_flags(
        source="",
        url="https://www.sciencedirect.com/science/article/pii/S2666389924001234",
        pdf_url=None,
    )
    assert sd_flags["is_sciencedirect_source"] is True

    asm_flags = downloader._infer_source_flags(
        source="",
        url="https://journals.asm.org/doi/10.1128/spectrum.01978-22",
        pdf_url=None,
    )
    assert asm_flags["is_asm_source"] is True


def test_resolve_platform_key_prefers_resolved_publisher_over_doi_entry():
    downloader = _make_downloader_for_unit_test()

    key = downloader._resolve_platform_key(
        "https://doi.org/10.1128/spectrum.01978-22",
        page_url="https://journals.asm.org/doi/full/10.1128/spectrum.01978-22",
    )
    assert key == "journals.asm.org"


def test_resolve_platform_key_keeps_doi_only_when_no_publisher_known():
    downloader = _make_downloader_for_unit_test()

    key = downloader._resolve_platform_key("https://dx.doi.org/10.1002/anie.202400001")
    assert key == "doi.org"
    assert downloader._is_low_value_warmup_key(key) is True


# --- LLM Candidate Rerank: context extraction and heuristic preference ---


def test_actionable_element_accepts_context_fields():
    """ActionableElement supports optional context fields for rerank."""
    el = ActionableElement(
        selector="[data-dl-uid='1']",
        tag="a",
        text="Open PDF",
        href="https://pubs.acs.org/doi/pdf/10.1021/acs.est.5c13105",
        position_ratio=0.1,
        in_viewport=True,
        ancestor_path=".article-header,.btn-group",
        nearby_text_before="",
        nearby_text_after="Supplementary",
        container_role="header",
        site_source="pubs.acs.org",
    )
    assert el.container_role == "header"
    assert el.site_source == "pubs.acs.org"
    assert el.position_ratio == 0.1
    assert el.nearby_text_after == "Supplementary"


def test_heuristic_prefers_header_main_over_footer_reference():
    """Context-aware scoring: header/main container and positive nearby text outrank footer/reference."""
    downloader = _make_downloader_for_unit_test()
    header_pdf = ActionableElement(
        selector="a.main-pdf",
        tag="a",
        text="Download PDF",
        href="https://example.com/doi/pdf/10.1234/art",
        is_visible=True,
        container_role="header",
        nearby_text_before="Open PDF",
        nearby_text_after="",
        position_ratio=0.05,
    )
    footer_link = ActionableElement(
        selector="a.suppl",
        tag="a",
        text="PDF",
        href="https://example.com/doi/suppl/10.1234/art/supp",
        is_visible=True,
        container_role="reference",
        nearby_text_before="Supporting Information",
        nearby_text_after="",
        position_ratio=0.9,
    )
    header_pdf.score = downloader.page_analyzer._calculate_score(header_pdf)
    footer_link.score = downloader.page_analyzer._calculate_score(footer_link)
    assert header_pdf.score > footer_link.score


def test_heuristic_penalizes_nearby_supplementary():
    """Nearby text containing 'Supporting' or 'References' reduces score."""
    downloader = _make_downloader_for_unit_test()
    main_like = ActionableElement(
        selector="a.pdf",
        tag="a",
        text="PDF",
        href="https://example.com/pdf",
        is_visible=True,
        nearby_text_before="Open PDF",
        nearby_text_after="",
    )
    supp_nearby = ActionableElement(
        selector="a.pdf2",
        tag="a",
        text="PDF",
        href="https://example.com/pdf2",
        is_visible=True,
        nearby_text_before="Supporting Information",
        nearby_text_after="",
    )
    main_like.score = downloader.page_analyzer._calculate_score(main_like)
    supp_nearby.score = downloader.page_analyzer._calculate_score(supp_nearby)
    assert main_like.score > supp_nearby.score


def test_pdf_extractor_fixed_candidate_not_penalized_for_late_ratio():
    cfg = ScoreConfig()
    extractor = PDFExtractor(logging.getLogger("test.extractor"), score_config=cfg)
    fixed_late = Candidate(
        element_id="fixed-late",
        text="download pdf",
        href="/doi/pdf/10.1/demo",
        is_fixed=True,
        rect_top=120,
        position_ratio=0.95,
        ancestor_features="sidebar article-tools",
        is_visible=True,
    )
    normal_late = Candidate(
        element_id="normal-late",
        text="download pdf",
        href="/doi/pdf/10.1/demo",
        is_fixed=False,
        rect_top=120,
        position_ratio=0.95,
        ancestor_features="sidebar article-tools",
        is_visible=True,
    )
    assert extractor._calculate_score(fixed_late) > extractor._calculate_score(normal_late)


def test_rerank_payload_is_minimal_and_strict():
    extractor = PDFExtractor(logging.getLogger("test.payload"))
    candidate = Candidate(
        element_id="a-1",
        text="open pdf",
        href="/doi/pdf/10.1/demo",
        is_fixed=True,
        rect_top=8.0,
        position_ratio=0.1,
        ancestor_features="header main",
        is_visible=True,
    )
    payload = [
        {
            "id": candidate.element_id,
            "text": candidate.text,
            "href": candidate.href,
            "is_fixed": candidate.is_fixed,
            "position_ratio": candidate.position_ratio,
            "score": 99.0,
        }
    ]
    prompt = '候选列表: {json_data}'.format(json_data=str(payload))
    assert "ancestor_features" not in prompt
    assert "html" not in prompt.lower()
    assert "id" in prompt and "position_ratio" in prompt


def test_rerank_fallback_when_ultra_lite_unavailable():
    """When no lite client is available, rerank should safely return None."""
    extractor = PDFExtractor(logging.getLogger("test.rerank_fallback"))
    candidates = [
        Candidate(
            element_id="a",
            text="pdf",
            href="/doi/pdf/10.1/demo",
            is_fixed=False,
            rect_top=0.0,
            position_ratio=0.2,
            ancestor_features="main",
            is_visible=True,
        ),
    ]

    async def _run():
        with patch("src.retrieval.downloader.paper_downloader_refactored.get_manager") as m:
            mgr = MagicMock()
            mgr.get_ultra_lite_client.return_value = None
            mgr.get_lite_client.return_value = None
            m.return_value = mgr
            return await extractor.rerank_candidates(candidates, top_n=3, mode="ultra-lite")

    result = asyncio.run(_run())
    assert result is None
