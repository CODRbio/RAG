import logging

from src.retrieval.downloader.paper_downloader_refactored import (
    ActionableElement,
    PageAnalyzer,
    PaperDownloader,
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
