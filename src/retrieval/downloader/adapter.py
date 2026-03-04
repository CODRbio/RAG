"""
src/retrieval/downloader/adapter.py

Adapter between RAG and PaperDownloader.
Search: delegate to RAG modules (google_search / semantic_scholar / ncbi_search).
Download: run PaperDownloader in ThreadPoolExecutor (separate event loop).
"""

import hashlib
import os
import re
import asyncio
import random
import unicodedata
import threading
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.log import get_logger

from .utils import is_valid_pdf

logger = get_logger(__name__)


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
        self.download_dir = cfg.download_dir
        self.show_browser = cfg.show_browser
        self.persist_browser = cfg.persist_browser
        self.proxy = cfg.proxy
        self.extension_path = getattr(settings, "capsolver_extension_path", "extra_tools/CapSolverExtension") or "extra_tools/CapSolverExtension"
        self.annas_api_key = cfg.annas_archive_api_key or ""
        self.max_concurrent = cfg.max_concurrent_downloads
        os.makedirs(self.download_dir, exist_ok=True)

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
        default_llm = getattr(settings.llm, "default", "deepseek") or "deepseek"
        try:
            prov = settings.llm.get_provider(default_llm)
            llm_key = (prov.get("api_key") or "").strip()
            if default_llm == "deepseek" and llm_key:
                api_keys["deepseek"] = llm_key
            elif default_llm in ("claude", "anthropic") and llm_key:
                api_keys["anthropic"] = llm_key
        except Exception:
            pass
        llm_provider = "auto"
        if default_llm == "deepseek" and api_keys.get("deepseek"):
            llm_provider = "deepseek"
        elif default_llm in ("claude", "anthropic") and api_keys.get("anthropic"):
            llm_provider = "anthropic"
        llm_model: Optional[str] = None
        if settings.llm.is_available(default_llm):
            try:
                llm_model = settings.llm.resolve_model(default_llm, None)
            except Exception:
                pass
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
            "llm": {"provider": llm_provider, "model": llm_model},
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

        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_bg_loop,
            daemon=True,
            name="scholar_dl_bg",
        )
        self._bg_thread.start()

        self._warmup_future = asyncio.run_coroutine_threadsafe(
            self._dl.initialize(),
            self._bg_loop,
        )

    def _run_bg_loop(self) -> None:
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

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
        """Anna's Archive keyword search running on shared downloader loop."""
        fut = asyncio.run_coroutine_threadsafe(
            self._impl_search_annas(query=query, limit=limit),
            self._bg_loop,
        )
        return await asyncio.wrap_future(fut)

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
        annas_md5: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        download_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download one paper PDF to download_dir (or override). Returns success, paper_id, filepath, message."""
        fut = asyncio.run_coroutine_threadsafe(
            self._impl_download_inner(title, doi, pdf_url, annas_md5, authors, year, download_dir=download_dir),
            self._bg_loop,
        )
        return await asyncio.wrap_future(fut)

    async def _impl_download_inner(
        self,
        title: str,
        doi: Optional[str],
        pdf_url: Optional[str],
        annas_md5: Optional[str],
        authors: Optional[List[str]],
        year: Optional[int],
        download_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_dir = (download_dir or self.download_dir).rstrip("/")

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

            # Make sure shared browser context is healthy; no-op when already ready.
            await self._dl.initialize()

            success = False
            message = ""

            if annas_md5 and self.annas_api_key:
                try:
                    ok, _reason, _ = await self._dl.download_with_annas_archive(
                        annas_md5, filepath, title=title, authors=authors, year=year
                    )
                    if ok:
                        success = True
                        message = "Anna's Archive (MD5)"
                except Exception as e:
                    logger.warning("Anna's MD5 download failed: %s", e)

            if not success and pdf_url:
                try:
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    success = await self._dl.find_and_download_pdf_with_browser(
                        pdf_url, filepath, title=title, authors=authors, year=year
                    )
                    if success:
                        message = "PDF URL 直接下载"
                except Exception as e:
                    logger.warning("PDF URL download failed: %s", e)

            if not success and doi:
                try:
                    await asyncio.sleep(random.uniform(1.5, 3.5))
                    success = await self._dl.download_with_sci_hub(
                        f"https://doi.org/{doi}",
                        filepath,
                        title=title,
                        authors=authors,
                        year=year,
                    )
                    if success:
                        message = "Sci-Hub"
                except Exception as e:
                    logger.warning("Sci-Hub download failed: %s", e)

            if not success and doi and self.annas_api_key:
                try:
                    md5 = await self._dl._annas_search_md5(doi)
                    if md5:
                        ok, _, _ = await self._dl.download_with_annas_archive(
                            md5, filepath, title=title, authors=authors, year=year
                        )
                        if ok:
                            success = True
                            message = "Anna's Archive (DOI)"
                except Exception as e:
                    logger.warning("Anna's DOI search failed: %s", e)

            if not success:
                return {
                    "success": False,
                    "paper_id": None,
                    "filepath": None,
                    "message": "所有下载策略均失败",
                }

            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "paper_id": None,
                    "filepath": None,
                    "message": "下载报告成功但文件不存在",
                }

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
        if self._bg_loop and self._bg_loop.is_running():
            cleanup_future = asyncio.run_coroutine_threadsafe(
                self._dl.cleanup_resources(force_close=True),
                self._bg_loop,
            )
            try:
                cleanup_future.result(timeout=15)
            except Exception as e:
                logger.warning("Downloader cleanup during shutdown failed: %s", e)
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=10)


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
