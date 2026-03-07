"""
src/retrieval/downloader/adapter.py

Adapter between RAG and PaperDownloader.
Search: delegate to RAG modules (google_search / semantic_scholar / ncbi_search).
Download: run PaperDownloader on the caller's async event loop.
"""

import hashlib
import os
import re
import asyncio
import random
import unicodedata
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from config.settings import settings
from src.log import get_logger
from src.utils.path_manager import PathManager

from .utils import is_valid_pdf

logger = get_logger(__name__)

# Academia.edu PDFs often require paid access; prefer DOI/Sci-Hub/Anna's first, use Academia as fallback.
ACADEMIA_DOMAIN = "academia.edu"
_UI_STRATEGY_IDS = (
    "direct_download",
    "playwright_download",
    "browser_lookup",
    "sci_hub",
    "brightdata",
    "anna",
)
_FALLBACK_DEFAULT_STRATEGY_ORDER = [
    "direct_download",
    "playwright_download",
    "browser_lookup",
    "sci_hub",
    "brightdata",
    "anna",
]
DEFAULT_STRATEGY_ORDER = [
    item for item in getattr(settings.scholar_downloader, "default_strategy_order", _FALLBACK_DEFAULT_STRATEGY_ORDER)
    if item in _UI_STRATEGY_IDS
]
for _item in _FALLBACK_DEFAULT_STRATEGY_ORDER:
    if _item not in DEFAULT_STRATEGY_ORDER:
        DEFAULT_STRATEGY_ORDER.append(_item)


def _is_academia_pdf_url(url: Optional[str]) -> bool:
    """True if the URL is an Academia.edu PDF/page (deprioritize: try other sources first)."""
    return bool(url and ACADEMIA_DOMAIN in (url or "").lower())


def _normalize_strategy_order(strategy_order: Optional[Sequence[str]]) -> List[str]:
    allowed = set(_UI_STRATEGY_IDS)
    ordered: List[str] = []
    for item in strategy_order or []:
        if item in allowed and item not in ordered:
            ordered.append(item)
    for item in DEFAULT_STRATEGY_ORDER:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _is_academia_url(url: Optional[str]) -> bool:
    """True if URL points to academia.edu."""
    return bool(url and ACADEMIA_DOMAIN in (url or "").lower())


def _normalize_to_paper_id(
    title: str,
    authors: Optional[List[str]] = None,
    year: Optional[int] = None,
) -> str:
    """Generate normalized paper_id from metadata (aligned with ingest paper_id = pdf stem)."""
    slug = title.strip()
    slug = unicodedata.normalize("NFKD", slug)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", slug.lower())
    slug = re.sub(r"[\s\-]+", "_", slug).strip("_")

    suffix_parts = []
    if authors and authors[0]:
        last_name = authors[0].split(",")[0].split()[-1]
        last_name = re.sub(r"[^\w]", "", last_name.lower())
        if last_name:
            suffix_parts.append(last_name)
    if year:
        suffix_parts.append(str(year))
    if suffix_parts:
        slug = f"{slug}_{'_'.join(suffix_parts)}"

    if not slug or len(slug) < 4:
        slug = hashlib.md5(title.encode("utf-8")).hexdigest()[:16]
    return slug[:80] if slug else f"paper_{os.urandom(4).hex()}"


def _doi_to_paper_id(doi: str) -> str:
    """Convert a DOI to a safe filesystem stem (used as paper_id).

    The DOI itself is stored in paper_meta_store keyed by this paper_id so
    reverse-conversion is not needed. Dedup works through the DB, not filenames.

    Examples:
        10.1038/s41586-023-06345-3  -> 10.1038_s41586-023-06345-3
        10.1093/bioinformatics/bty395 -> 10.1093_bioinformatics_bty395
    """
    slug = re.sub(r"[^\w\-.]", "_", doi.strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug[:100] if slug else hashlib.md5(doi.encode()).hexdigest()[:16]


class ScholarDownloaderAdapter:
    """
    RAG ↔ PaperDownloader adapter.
    Search: RAG modules only. Download: background event loop → shared PaperDownloader.
    """

    def __init__(self) -> None:
        cfg = settings.scholar_downloader
        self.download_dir = str(PathManager.get_user_raw_papers_path(PathManager.DEFAULT_USER_ID))
        self.show_browser = cfg.show_browser
        self.persist_browser = cfg.persist_browser
        self.proxy = cfg.proxy
        self.extension_path = getattr(settings, "capsolver_extension_path", "extra_tools/CapSolverExtension") or "extra_tools/CapSolverExtension"
        self.annas_api_key = cfg.annas_archive_api_key or ""
        self.max_concurrent = cfg.max_concurrent_downloads

        # 与当前项目配置统一：从 RAG settings 构建 downloader 使用的 config 结构
        api_keys: Dict[str, str] = {
            "annas_archive": (cfg.annas_archive_api_key or "").strip(),
            "twocaptcha": (cfg.twocaptcha_api_key or "").strip(),
        }
        if not api_keys["twocaptcha"]:
            cf = getattr(settings, "content_fetcher", None)
            api_keys["twocaptcha"] = (getattr(cf, "two_captcha_api_key", "") or "").strip()
        cf = getattr(settings, "content_fetcher", None)
        api_keys["brightdata"] = (getattr(cf, "brightdata_api_key", "") or "").strip()
        default_llm = getattr(cfg, "llm_provider", None) or "qwen-thinking"
        ext_path = (getattr(settings, "capsolver_extension_path", "") or "").strip() or None
        dl_opts: Dict[str, Any] = {
            "proxy": cfg.proxy,
            "capsolver_extension_path": ext_path,
        }
        if getattr(cfg, "experience_store_path", None) is not None:
            dl_opts["experience_store_path"] = cfg.experience_store_path
        if getattr(cfg, "timeouts", None):
            dl_opts["timeouts"] = cfg.timeouts
        initial_config = {
            "api_keys": api_keys,
            "llm": {"provider": default_llm},
            "downloader": dl_opts,
            "annas_keyword_max_pages": getattr(cfg, "annas_keyword_max_pages", 5),
        }

        from .paper_downloader_refactored import PaperDownloader

        self._dl = PaperDownloader(
            download_dir=self.download_dir,
            show_browser=self.show_browser,
            persist_browser=True,
            max_concurrent=self.max_concurrent,
            download_timeout=getattr(cfg, "download_timeout", 200),
            max_retries=getattr(cfg, "max_retries", 3),
            browser_type=getattr(cfg, "browser_type", "chrome") or "chrome",
            stealth_mode=getattr(cfg, "stealth_mode", True),
            initial_config=initial_config,
        )

    async def search_google_scholar(
        self,
        query: str,
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        job_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Google Scholar via RAG google_search."""
        from src.retrieval.google_search import GoogleSearcher

        searcher = GoogleSearcher()
        return await searcher.search_scholar(
            query=query,
            limit=limit,
            year_start=year_start,
            year_end=year_end,
            job_id=job_id,
        )

    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar relevance search (/paper/search)."""
        from src.retrieval.semantic_scholar import semantic_scholar_searcher

        return await semantic_scholar_searcher.search(
            query=query, limit=limit, year_start=year_start, year_end=year_end
        )

    async def search_semantic_scholar_relevance(
        self,
        query: str,
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar relevance search (same as search_semantic_scholar)."""
        return await self.search_semantic_scholar(
            query=query, limit=limit, year_start=year_start, year_end=year_end
        )

    async def search_semantic_scholar_bulk(
        self,
        query: str,
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar bulk/boolean search (/paper/search/bulk)."""
        from src.retrieval.semantic_scholar import semantic_scholar_searcher

        return await semantic_scholar_searcher.search_bulk(
            query=query, limit=limit, year_start=year_start, year_end=year_end
        )

    async def search_ncbi(
        self,
        query: str,
        limit: int = 10,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """PubMed/NCBI via RAG ncbi_search."""
        from src.retrieval.ncbi_search import get_ncbi_searcher

        searcher = get_ncbi_searcher()
        return await searcher.search(
            query=query, limit=limit, year_start=year_start, year_end=year_end
        )

    async def search_annas_archive(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Anna's Archive keyword search via shared downloader instance."""
        return await self._impl_search_annas(query=query, limit=limit)

    async def _impl_search_annas(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        raw = await self._dl.annas_keyword_search(query=query, limit=limit)
        return [
            {
                "content": r.get("snippet", ""),
                "score": 0.7,
                "metadata": {
                    "source": "annas_archive",
                    "title": r.get("title", ""),
                    "authors": r.get("authors", []),
                    "year": r.get("year"),
                    "doi": None,
                    "annas_md5": r.get("md5"),
                    "url": r.get("link"),
                    "downloadable": bool(r.get("md5")),
                },
            }
            for r in raw
        ]

    async def download_paper(
        self,
        *,
        title: str,
        doi: Optional[str] = None,
        pdf_url: Optional[str] = None,
        url: Optional[str] = None,
        annas_md5: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        download_dir: Optional[str] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
        assist_llm_enabled: Optional[bool] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_browser: Optional[bool] = None,
        include_academia: bool = False,
        strategy_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Download one paper PDF to download_dir (or override). Returns success, paper_id, filepath, message.
        If progress_callback is set, it is called with e.g. {"stage": "strategy_start", "strategy": "annas_md5"}.
        show_browser: optional override for headed(True)/headless(False); affects browser launch.
        include_academia: when False (default), Academia.edu pdf_url is skipped entirely to avoid hangs."""
        return await self._impl_download_inner(
            title, doi, pdf_url, url, annas_md5, authors, year,
            download_dir=download_dir,
            llm_provider=llm_provider,
            model_override=model_override,
            assist_llm_enabled=assist_llm_enabled,
            progress_callback=progress_callback,
            show_browser=show_browser,
            include_academia=include_academia,
            strategy_order=strategy_order,
        )

    async def _impl_download_inner(
        self,
        title: str,
        doi: Optional[str],
        pdf_url: Optional[str],
        url: Optional[str],
        annas_md5: Optional[str],
        authors: Optional[List[str]],
        year: Optional[int],
        download_dir: Optional[str] = None,
        llm_provider: Optional[str] = None,
        model_override: Optional[str] = None,
        assist_llm_enabled: Optional[bool] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_browser: Optional[bool] = None,
        include_academia: bool = False,
        strategy_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        target_dir = (download_dir or self.download_dir).rstrip("/")
        cb = progress_callback

        async def _do_download() -> Dict[str, Any]:
            os.makedirs(target_dir, exist_ok=True)
            paper_id = _doi_to_paper_id(doi) if doi else _normalize_to_paper_id(title, authors, year)
            filepath = os.path.join(target_dir, f"{paper_id}.pdf")

            if os.path.exists(filepath) and is_valid_pdf(filepath):
                logger.info("PDF already exists, skip: %s", filepath)
                try:
                    from src.indexing.paper_metadata_store import paper_meta_store
                    paper_meta_store.upsert(
                        paper_id, doi=doi or "", title=title,
                        authors=authors, year=year, source="download",
                    )
                except Exception:
                    pass
                return {
                    "success": True,
                    "paper_id": paper_id,
                    "filepath": filepath,
                    "message": "已存在，跳过下载",
                }

            # Same paper may exist under a different paper_id (e.g. different title normalization). Look up by DOI.
            if doi:
                try:
                    from src.indexing.paper_metadata_store import paper_meta_store
                    for pid in paper_meta_store.get_paper_ids_by_doi(doi):
                        if pid == paper_id:
                            continue
                        candidate_path = os.path.join(target_dir, f"{pid}.pdf")
                        if os.path.exists(candidate_path) and is_valid_pdf(candidate_path):
                            logger.info("PDF already exists (same DOI), skip: %s", candidate_path)
                            paper_meta_store.upsert(
                                pid, doi=doi or "", title=title,
                                authors=authors, year=year, source="download",
                            )
                            return {
                                "success": True,
                                "paper_id": pid,
                                "filepath": candidate_path,
                                "message": "已存在（同 DOI），跳过下载",
                            }
                except Exception as e:
                    logger.debug("DOI-based skip check failed (non-fatal): %s", e)

            # Persist the caller's browser mode onto the shared downloader instance.
            # The session pool/context lifecycle must follow this instance-level override
            # instead of the per-request ContextVar used inside individual page flows.
            logger.info(
                "[headed-diag] adapter: show_browser=%r (from API), "
                "current _show_browser_override=%r, self._dl.headless=%s",
                show_browser,
                getattr(self._dl, "_show_browser_override", None),
                self._dl.headless,
            )
            self._dl.set_show_browser_override(show_browser)

            # Make sure shared browser context is healthy; no-op when already ready.
            await self._dl.initialize()

            success = False
            message = ""
            effective_strategy_order = _normalize_strategy_order(strategy_order)
            pdf_url_is_academia = _is_academia_pdf_url(pdf_url)
            article_url_is_academia = _is_academia_url(url)
            llm_disabled = assist_llm_enabled is False
            effective_llm_provider = None if llm_disabled else llm_provider
            effective_model_override = None if llm_disabled else model_override
            doi_str = (doi or "").strip()
            doi_landing_url = (
                doi_str
                if (doi_str and doi_str.lower().startswith("http"))
                else (f"https://doi.org/{doi_str}" if doi_str else "")
            )
            browser_lookup_targets: List[tuple[str, str]] = []
            if not pdf_url:
                if doi_landing_url:
                    browser_lookup_targets.append(("doi", doi_landing_url))
                if url:
                    browser_lookup_targets.append(("url", url))
            if not include_academia:
                browser_lookup_targets = [
                    (kind, target)
                    for kind, target in browser_lookup_targets
                    if not _is_academia_url(target)
                ]

            route_diag = {
                "has_pdf_url": bool(pdf_url),
                "has_doi": bool(doi_str),
                "has_url": bool(url),
                "has_annas_md5": bool(annas_md5),
                "include_academia": bool(include_academia),
                "requested_strategy_order": list(strategy_order or []),
                "effective_strategy_order": list(effective_strategy_order),
                "browser_lookup_targets": [kind for kind, _ in browser_lookup_targets],
                "skipped": {},
                "attempted": [],
            }

            logger.info(
                "[download-route] paper=%s has_pdf_url=%s has_doi=%s has_url=%s has_annas_md5=%s strategy_order=%s browser_lookup_targets=%s",
                title[:80],
                bool(pdf_url),
                bool(doi_str),
                bool(url),
                bool(annas_md5),
                effective_strategy_order,
                [kind for kind, _ in browser_lookup_targets],
            )

            async def _run_direct_download() -> bool:
                nonlocal message
                if not pdf_url or (pdf_url_is_academia and not include_academia):
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "direct_download"})
                try:
                    ok = await self._dl.download_direct(
                        pdf_url,
                        filepath,
                        title=title,
                        authors=authors,
                        year=year,
                    )
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "direct_download", "success": ok})
                    if ok:
                        message = "Direct PDF URL"
                    return ok
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "direct_download", "success": False})
                    logger.warning("Direct PDF download failed: %s", e)
                    return False

            async def _run_playwright_download() -> bool:
                nonlocal message
                if not pdf_url or (pdf_url_is_academia and not include_academia):
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "playwright_download"})
                try:
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    ok = await self._dl.find_and_download_pdf_with_browser(
                        pdf_url,
                        filepath,
                        title=title,
                        authors=authors,
                        year=year,
                        llm_provider=effective_llm_provider,
                        model_override=effective_model_override,
                        progress_callback=cb,
                        show_browser=show_browser,
                        disable_assist_llm=llm_disabled,
                    )
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "playwright_download", "success": ok})
                    if ok:
                        message = "Playwright browser flow"
                    return ok
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "playwright_download", "success": False})
                    logger.warning("Playwright PDF download failed: %s", e)
                    return False

            async def _run_browser_lookup() -> bool:
                nonlocal message
                if not browser_lookup_targets:
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "browser_lookup"})
                try:
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    for target_kind, target_url in browser_lookup_targets:
                        ok = await self._dl.find_and_download_pdf_with_browser(
                            target_url,
                            filepath,
                            title=title,
                            authors=authors,
                            year=year,
                            llm_provider=effective_llm_provider,
                            model_override=effective_model_override,
                            progress_callback=cb,
                            show_browser=show_browser,
                            disable_assist_llm=llm_disabled,
                        )
                        if ok:
                            if cb:
                                cb({"stage": "strategy_done", "strategy": "browser_lookup", "success": True})
                            message = f"Browser lookup via {target_kind}"
                            return True
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "browser_lookup", "success": False})
                    return False
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "browser_lookup", "success": False})
                    logger.warning("Browser lookup failed: %s", e)
                    return False

            async def _run_sci_hub() -> bool:
                nonlocal message
                if not doi:
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "sci_hub"})
                try:
                    await asyncio.sleep(random.uniform(1.5, 3.5))
                    ok = await self._dl.download_with_sci_hub(
                        f"https://doi.org/{doi}",
                        filepath,
                        title=title,
                        authors=authors,
                        year=year,
                    )
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "sci_hub", "success": ok})
                    if ok:
                        message = "Sci-Hub"
                    return ok
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "sci_hub", "success": False})
                    logger.warning("Sci-Hub download failed: %s", e)
                    return False

            async def _run_brightdata() -> bool:
                nonlocal message
                if not pdf_url or not getattr(self._dl, "brightdata_api_key", None):
                    return False
                if pdf_url_is_academia and not include_academia:
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "brightdata"})
                try:
                    ok = await self._dl.download_with_solver(
                        pdf_url,
                        filepath,
                        title=title,
                        authors=authors,
                        year=year,
                    )
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "brightdata", "success": ok})
                    if ok:
                        message = "BrightData"
                    return ok
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "brightdata", "success": False})
                    logger.warning("BrightData download failed: %s", e)
                    return False

            async def _run_anna() -> bool:
                nonlocal message
                if not self.annas_api_key:
                    return False
                if not annas_md5 and not doi:
                    return False
                if cb:
                    cb({"stage": "strategy_start", "strategy": "anna"})
                try:
                    if annas_md5:
                        ok, _reason, _ = await self._dl.download_with_annas_archive(
                            annas_md5,
                            filepath,
                            title=title,
                            authors=authors,
                            year=year,
                        )
                        if ok:
                            if cb:
                                cb({"stage": "strategy_done", "strategy": "anna", "success": True})
                            message = "Anna's Archive (MD5)"
                            return True
                    if doi:
                        md5 = await self._dl._annas_search_md5(doi)
                        if md5:
                            ok, _, _ = await self._dl.download_with_annas_archive(
                                md5,
                                filepath,
                                title=title,
                                authors=authors,
                                year=year,
                            )
                            if ok:
                                if cb:
                                    cb({"stage": "strategy_done", "strategy": "anna", "success": True})
                                message = "Anna's Archive (DOI)"
                                return True
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "anna", "success": False})
                    return False
                except Exception as e:
                    if cb:
                        cb({"stage": "strategy_done", "strategy": "anna", "success": False})
                    logger.warning("Anna download failed: %s", e)
                    return False
            if pdf_url_is_academia and not include_academia:
                logger.info("Academia.edu pdf_url skipped (include_academia=False): %s", (pdf_url or "")[:80])
            if article_url_is_academia and not include_academia:
                logger.info("Academia.edu article url skipped (include_academia=False): %s", (url or "")[:80])

            runners = {
                "direct_download": _run_direct_download,
                "playwright_download": _run_playwright_download,
                "browser_lookup": _run_browser_lookup,
                "sci_hub": _run_sci_hub,
                "brightdata": _run_brightdata,
                "anna": _run_anna,
            }
            strategy_can_run = {
                "direct_download": bool(pdf_url) and (include_academia or not pdf_url_is_academia),
                "playwright_download": bool(pdf_url) and (include_academia or not pdf_url_is_academia),
                "browser_lookup": bool(browser_lookup_targets),
                "sci_hub": bool(doi_str),
                "brightdata": bool(pdf_url) and bool(getattr(self._dl, "brightdata_api_key", None))
                and (include_academia or not pdf_url_is_academia),
                "anna": bool(self.annas_api_key) and bool(annas_md5 or doi_str),
            }
            for strategy_id in effective_strategy_order:
                if success:
                    break
                runner = runners.get(strategy_id)
                if not runner:
                    continue
                if not strategy_can_run.get(strategy_id, True):
                    route_diag["skipped"][strategy_id] = "precondition_not_met"
                    continue
                route_diag["attempted"].append(strategy_id)
                success = await runner()

            if not success:
                return {
                    "success": False,
                    "paper_id": None,
                    "filepath": None,
                    "message": "所有下载策略均失败",
                    "route_diag": route_diag,
                }

            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "paper_id": None,
                    "filepath": None,
                    "message": "下载报告成功但文件不存在",
                }

            if cb:
                cb({"stage": "validating"})
            if not is_valid_pdf(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                return {
                    "success": False,
                    "paper_id": None,
                    "filepath": None,
                    "message": "下载的文件不是有效 PDF",
                }

            try:
                from src.indexing.paper_metadata_store import paper_meta_store
                paper_meta_store.upsert(
                    paper_id, doi=doi or "", title=title,
                    authors=authors, year=year, source="download",
                )
            except Exception:
                pass

            return {
                "success": True,
                "paper_id": paper_id,
                "filepath": filepath,
                "message": message,
                "route_diag": route_diag,
            }

        try:
            return await asyncio.wait_for(_do_download(), timeout=300)
        except asyncio.TimeoutError:
            return {
                "success": False,
                "paper_id": None,
                "filepath": None,
                "message": "下载超时（5分钟）",
            }
        except Exception as e:
            logger.error("download_paper fatal: %s", e)
            return {
                "success": False,
                "paper_id": None,
                "filepath": None,
                "message": f"下载异常: {e}",
            }

    async def batch_download(
        self,
        papers: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Batch download; returns success/failed counts and details."""
        sem = asyncio.Semaphore(max_concurrent or self.max_concurrent)
        results: Dict[str, Any] = {"success": 0, "failed": 0, "details": []}

        async def _one(p: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                return await self.download_paper(
                    title=p.get("title", ""),
                    doi=p.get("doi"),
                    pdf_url=p.get("pdf_url"),
                    url=p.get("url"),
                    annas_md5=p.get("annas_md5"),
                    authors=p.get("authors"),
                    year=p.get("year"),
                )

        for coro in asyncio.as_completed([_one(p) for p in papers]):
            r = await coro
            if r["success"]:
                results["success"] += 1
            else:
                results["failed"] += 1
            results["details"].append(r)

        return results

    def shutdown(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(self._dl.cleanup_resources(force_close=True))
            return

        try:
            asyncio.run(self._dl.cleanup_resources(force_close=True))
        except Exception as e:
            logger.warning("Downloader cleanup during shutdown failed: %s", e)


_adapter_instance: Optional[ScholarDownloaderAdapter] = None
_adapter_lock = threading.Lock()


def get_adapter() -> ScholarDownloaderAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        with _adapter_lock:
            if _adapter_instance is None:
                _adapter_instance = ScholarDownloaderAdapter()
    return _adapter_instance


def is_adapter_ready() -> bool:
    return _adapter_instance is not None


def shutdown_adapter() -> None:
    global _adapter_instance
    if _adapter_instance:
        _adapter_instance.shutdown()
        _adapter_instance = None
