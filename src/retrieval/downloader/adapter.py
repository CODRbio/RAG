"""
src/retrieval/downloader/adapter.py

Adapter between RAG and PaperDownloader.
Search: delegate to RAG modules (google_search / semantic_scholar / ncbi_search).
Download: run PaperDownloader in ThreadPoolExecutor (separate event loop).
"""

import os
import re
import asyncio
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.log import get_logger

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

    return slug[:80] if slug else f"paper_{os.urandom(4).hex()}"


class ScholarDownloaderAdapter:
    """
    RAG ↔ PaperDownloader adapter.
    Search: RAG modules only. Download: ThreadPoolExecutor → PaperDownloader.
    """

    def __init__(self) -> None:
        cfg = settings.scholar_downloader
        self.download_dir = cfg.download_dir
        self.show_browser = cfg.show_browser
        self.persist_browser = cfg.persist_browser
        self.proxy = cfg.proxy
        self.extension_path = cfg.capsolver_extension_path
        self.annas_api_key = cfg.annas_archive_api_key or ""
        self.max_concurrent = cfg.max_concurrent_downloads
        os.makedirs(self.download_dir, exist_ok=True)

        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="scholar_dl",
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
    ) -> List[Dict[str, Any]]:
        """Semantic Scholar via RAG semantic_scholar."""
        from src.retrieval.semantic_scholar import semantic_scholar_searcher

        return await semantic_scholar_searcher.search(query=query, limit=limit)

    async def search_ncbi(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """PubMed/NCBI via RAG ncbi_search."""
        from src.retrieval.ncbi_search import get_ncbi_searcher

        searcher = get_ncbi_searcher()
        return await searcher.search(query=query, limit=limit)

    async def search_annas_archive(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Anna's Archive keyword search; no RAG module, run in executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_search_annas, query, limit
        )

    def _sync_search_annas(self, query: str, limit: int) -> List[Dict[str, Any]]:
        return asyncio.run(self._impl_search_annas(query, limit))

    async def _impl_search_annas(
        self, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        from .paper_downloader_refactored import PaperDownloader

        dl = PaperDownloader(
            download_dir=self.download_dir,
            show_browser=False,
            persist_browser=False,
        )
        try:
            dl.config.setdefault("api_keys", {})["annas_archive"] = self.annas_api_key
            await dl.initialize()
            raw = await dl.annas_keyword_search(query=query, limit=limit)
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
        finally:
            await dl.cleanup_resources(force_close=True)

    async def download_paper(
        self,
        *,
        title: str,
        doi: Optional[str] = None,
        pdf_url: Optional[str] = None,
        annas_md5: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Download one paper PDF to download_dir. Returns success, paper_id, filepath, message."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_download,
            title,
            doi,
            pdf_url,
            annas_md5,
            authors,
            year,
        )

    def _sync_download(
        self,
        title: str,
        doi: Optional[str],
        pdf_url: Optional[str],
        annas_md5: Optional[str],
        authors: Optional[List[str]],
        year: Optional[int],
    ) -> Dict[str, Any]:
        return asyncio.run(
            self._impl_download(
                title, doi, pdf_url, annas_md5, authors, year
            )
        )

    async def _impl_download(
        self,
        title: str,
        doi: Optional[str],
        pdf_url: Optional[str],
        annas_md5: Optional[str],
        authors: Optional[List[str]],
        year: Optional[int],
    ) -> Dict[str, Any]:
        from .paper_downloader_refactored import PaperDownloader

        paper_id = _normalize_to_paper_id(title, authors, year)
        filepath = os.path.join(self.download_dir, f"{paper_id}.pdf")

        if os.path.exists(filepath):
            try:
                dl_check = PaperDownloader(
                    download_dir=self.download_dir,
                    show_browser=False,
                    persist_browser=False,
                )
                if dl_check.is_valid_pdf_file(filepath):
                    logger.info("PDF already exists, skip: %s", filepath)
                    return {
                        "success": True,
                        "paper_id": paper_id,
                        "filepath": filepath,
                        "message": "已存在，跳过下载",
                    }
            except Exception:
                pass

        dl = PaperDownloader(
            download_dir=self.download_dir,
            show_browser=self.show_browser,
            persist_browser=False,
        )
        try:
            dl.config.setdefault("api_keys", {})["annas_archive"] = self.annas_api_key
            await dl.initialize()

            success = False
            message = ""

            if annas_md5 and self.annas_api_key:
                try:
                    ok, reason, _ = await dl.download_with_annas_archive(
                        annas_md5, filepath, title=title, authors=authors, year=year
                    )
                    if ok:
                        success = True
                        message = "Anna's Archive (MD5)"
                except Exception as e:
                    logger.warning("Anna's MD5 download failed: %s", e)

            if not success and pdf_url:
                try:
                    success = await dl.find_and_download_pdf_with_browser(
                        pdf_url, filepath, title=title, authors=authors, year=year
                    )
                    if success:
                        message = "PDF URL 直接下载"
                except Exception as e:
                    logger.warning("PDF URL download failed: %s", e)

            if not success and doi:
                try:
                    success = await dl.download_with_sci_hub(
                        f"https://doi.org/{doi}", filepath,
                        title=title, authors=authors, year=year,
                    )
                    if success:
                        message = "Sci-Hub"
                except Exception as e:
                    logger.warning("Sci-Hub download failed: %s", e)

            if not success and doi and self.annas_api_key:
                try:
                    md5 = await dl._annas_search_md5(doi)
                    if md5:
                        ok, _, _ = await dl.download_with_annas_archive(
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

            if not dl.is_valid_pdf_file(filepath):
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

            return {
                "success": True,
                "paper_id": paper_id,
                "filepath": filepath,
                "message": message,
            }

        except Exception as e:
            logger.error("download_paper fatal: %s", e)
            return {
                "success": False,
                "paper_id": None,
                "filepath": None,
                "message": f"下载异常: {e}",
            }
        finally:
            try:
                await dl.cleanup_resources(force_close=True)
            except Exception as e:
                logger.warning("Browser cleanup error (non-fatal): %s", e)

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
        self._executor.shutdown(wait=True)


_adapter_instance: Optional[ScholarDownloaderAdapter] = None


def get_adapter() -> ScholarDownloaderAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ScholarDownloaderAdapter()
    return _adapter_instance


def shutdown_adapter() -> None:
    global _adapter_instance
    if _adapter_instance:
        _adapter_instance.shutdown()
        _adapter_instance = None
