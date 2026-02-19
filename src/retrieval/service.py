"""
统一检索服务 - RetrievalService

封装 HybridRetriever + UnifiedWebSearch，对外提供 search(query, mode, filters) -> EvidencePack。
支持 local / web / hybrid 三种模式。hybrid 时 local 与 web 并行执行。
"""

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.retrieval.evidence import EvidenceChunk, EvidencePack
from src.retrieval.dedup import cross_source_dedup
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig, _rerank_candidates
from src.retrieval.unified_web_search import unified_web_searcher

logger = logging.getLogger(__name__)

# ── Observability ──
try:
    from src.observability import metrics as obs_metrics, tracer as obs_tracer
except Exception:
    obs_metrics = None  # type: ignore
    obs_tracer = None  # type: ignore


def _parse_paper_id(paper_id: str) -> Tuple[Optional[int], Optional[List[str]], Optional[str]]:
    """
    从 paper_id（文件名格式）解析 year、authors、title。
    
    支持格式：
    - "2026_Botté_et_al_Artificial_Light_at_Night" → (2026, ["Botté et al."], "Artificial Light at Night")
    - "Smith_Jones_2023_Deep_Sea_Cold_Seeps" → (2023, ["Smith", "Jones"], "Deep Sea Cold Seeps")
    - "2024_Wang_Microbiome_Analysis" → (2024, ["Wang"], "Microbiome Analysis")
    
    Returns:
        (year, authors, title) - 任一字段可能为 None
    """
    if not paper_id:
        return None, None, None
    
    year = None
    authors = None
    title = None
    
    # 按下划线分割
    parts = paper_id.replace("-", "_").split("_")
    if not parts:
        return None, None, None
    
    # 查找年份位置（4位数字，1900-2100）
    year_idx = -1
    for i, part in enumerate(parts):
        if re.match(r"^(19|20)\d{2}$", part):
            year = int(part)
            year_idx = i
            break
    
    # 常见的非作者名单词（可能是标题开头）
    TITLE_INDICATORS = {
        "a", "an", "the", "on", "in", "of", "for", "with", "to", "and", "or",
        "deep", "sea", "cold", "seep", "marine", "ocean", "coral", "reef",
        "analysis", "study", "review", "research", "investigation", "evidence",
        "effect", "effects", "impact", "role", "new", "novel", "first",
        "artificial", "natural", "environmental", "ecological", "biological",
        "microbiome", "microbiota", "bacteria", "microbial", "community",
    }
    
    def _is_likely_author(word: str) -> bool:
        """判断一个词是否可能是作者名"""
        if not word:
            return False
        # 全大写（如 DNA, RNA）不是作者名
        if word.isupper() and len(word) > 1:
            return False
        # 常见标题词不是作者名
        if word.lower() in TITLE_INDICATORS:
            return False
        # 太长的词（>15字符）不太可能是作者名
        if len(word) > 15:
            return False
        # 首字母大写，且比较短，可能是作者名
        return word[0].isupper()
    
    # 根据年份位置确定作者和标题
    if year_idx == 0:
        # 格式: 2026_Botté_et_al_Title...
        author_parts = []
        title_start_idx = 1
        for i in range(1, len(parts)):
            p = parts[i]
            # "et" 是 "et al." 的一部分
            if p.lower() == "et" and i + 1 < len(parts) and parts[i + 1].lower() == "al":
                author_parts.append("et al.")
                title_start_idx = i + 2
                break
            # 判断是否为作者名
            elif _is_likely_author(p):
                author_parts.append(p)
                title_start_idx = i + 1
            else:
                # 标题开始
                break
        
        if author_parts:
            authors = author_parts
        if title_start_idx < len(parts):
            title = " ".join(parts[title_start_idx:]).replace("_", " ")
    
    elif year_idx > 0:
        # 格式: Smith_Jones_2023_Title...
        authors = parts[:year_idx]
        if year_idx + 1 < len(parts):
            title = " ".join(parts[year_idx + 1:]).replace("_", " ")
    
    else:
        # 没找到年份，尝试从第一个部分判断
        # 可能整个就是标题
        title = " ".join(parts).replace("_", " ")
    
    return year, authors, title


def _hit_to_chunk(hit: Dict[str, Any], source_type: str, query: str) -> EvidenceChunk:
    """将 hybrid 或 web 的 hit 转为 EvidenceChunk"""
    meta = hit.get("metadata") or {}
    content = (hit.get("content") or "").strip()
    chunk_id = meta.get("chunk_id") or hit.get("chunk_id") or meta.get("doc_id") or meta.get("url") or str(id(hit))
    doc_id = meta.get("paper_id") or meta.get("doc_id") or chunk_id
    score = float(hit.get("score", 0.0))
    
    # 提取 authors（支持多种格式）
    authors_raw = meta.get("authors")
    if isinstance(authors_raw, list):
        authors = authors_raw
    elif isinstance(authors_raw, str) and authors_raw:
        # 可能是逗号分隔的字符串
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
    else:
        authors = None
    
    # 提取 year（支持 int 或 str）
    year_raw = meta.get("year")
    if isinstance(year_raw, int):
        year = year_raw
    elif isinstance(year_raw, str) and year_raw.isdigit():
        year = int(year_raw)
    else:
        year = None
    
    # 提取 title
    title = meta.get("title")
    
    # 如果缺少 authors/year/title，尝试从 paper_id（文件名）解析
    if (authors is None or year is None or title is None) and doc_id:
        parsed_year, parsed_authors, parsed_title = _parse_paper_id(doc_id)
        if authors is None and parsed_authors:
            authors = parsed_authors
        if year is None and parsed_year:
            year = parsed_year
        if title is None and parsed_title:
            title = parsed_title
    
    # 提取 bbox（Docling 物理坐标列表）
    raw_bbox = meta.get("bbox")
    bbox = None
    if isinstance(raw_bbox, list) and raw_bbox:
        # 兼容两种格式：
        # - [x0, y0, x1, y1]
        # - [[x0, y0, x1, y1], ...]（取第一块用于前端单框高亮）
        first = raw_bbox[0]
        if isinstance(first, (int, float)):
            bbox = raw_bbox
        elif isinstance(first, list) and len(first) >= 4:
            bbox = first

    return EvidenceChunk(
        chunk_id=str(chunk_id),
        doc_id=str(doc_id),
        text=content,
        score=score,
        source_type=source_type,
        doc_title=title,
        authors=authors,
        year=year,
        url=meta.get("url"),
        doi=meta.get("doi"),
        page_num=meta.get("page") if isinstance(meta.get("page"), int) else None,
        section_title=meta.get("section_path"),
        bbox=bbox,
    )


class RetrievalService:
    """
    统一检索服务。
    search(query, mode, filters) -> EvidencePack
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        collection: Optional[str] = None,
        top_k: int = 10,
    ):
        self.retriever = retriever or HybridRetriever()
        self.collection = collection or settings.collection.global_
        self.top_k = top_k

    def search(
        self,
        query: str,
        mode: str = "local",
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> EvidencePack:
        """
        统一检索入口。

        Args:
            query: 查询文本
            mode: "local" 仅本地向量/图, "web" 仅网络搜索, "hybrid" 本地+网络合并
            filters: 预留过滤条件
            top_k: 返回条数，默认使用实例 top_k

        Returns:
            EvidencePack
        """
        k = top_k if top_k is not None else self.top_k
        sources_used: List[str] = []
        all_chunks: List[EvidenceChunk] = []
        total_candidates = 0
        t0 = time.perf_counter()
        timeout_s = getattr(getattr(settings, "perf_retrieval", None), "timeout_seconds", 60) or 60

        # Amplify retrieval pool so that the reranker always operates on a
        # sufficiently large candidate set, regardless of the caller's top_k.
        actual_recall = max(80, k * 4)

        # 诊断信息收集
        diag: Dict[str, Any] = {}
        if actual_recall != k:
            diag["recall_amplification"] = {"requested_k": k, "actual_recall": actual_recall}

        def _do_local() -> List[Dict[str, Any]]:
            config = RetrievalConfig(mode="hybrid", top_k=actual_recall, rerank=True)
            return self.retriever.retrieve(query, self.collection, config, diagnostics=diag)

        def _do_web() -> List[Dict[str, Any]]:
            web_providers = (filters or {}).get("web_providers")
            if web_providers is not None and len(web_providers) == 0:
                return []
            web_source_configs = (filters or {}).get("web_source_configs") or {}
            use_query_expansion = (filters or {}).get("use_query_expansion")
            use_query_optimizer = (filters or {}).get("use_query_optimizer")
            query_optimizer_max_queries = (filters or {}).get("query_optimizer_max_queries")
            _llm_provider = (filters or {}).get("llm_provider")
            _model_override = (filters or {}).get("model_override")
            # 参数级联：根据 final_top_k 自动放大 per-provider 采集量
            _final = (filters or {}).get("final_top_k") or k
            _n_providers = max(len(web_providers or []), 1)
            _n_queries = query_optimizer_max_queries or 3
            _auto_per_provider = max(5, math.ceil(_final * 3.5 / (_n_providers * _n_queries)))
            logger.info(
                "web search params: providers=%s auto_per_provider=%s final_top_k=%s llm=%s/%s",
                web_providers,
                _auto_per_provider,
                _final,
                _llm_provider,
                _model_override,
            )
            use_content_fetcher = (filters or {}).get("use_content_fetcher")

            # 为 Lazy Fetching 预判构建 LLM 客户端（仅智能模式 use_content_fetcher=None）
            _llm_client = None
            if use_content_fetcher is None:
                try:
                    from src.llm.llm_manager import get_manager
                    _llm_client = get_manager().get_client(_llm_provider or "deepseek")
                except Exception as _e:
                    logger.debug("Lazy Fetching LLM 客户端初始化失败，降级全量抓取: %s", _e)

            t_web = time.perf_counter()
            results = unified_web_searcher.search_sync(
                query,
                providers=web_providers,
                source_configs=web_source_configs,
                max_results_per_provider=_auto_per_provider,
                use_query_expansion=use_query_expansion,
                use_query_optimizer=use_query_optimizer,
                query_optimizer_max_queries=query_optimizer_max_queries,
                llm_provider=_llm_provider,
                model_override=_model_override,
                use_content_fetcher=use_content_fetcher,
                llm_client=_llm_client,
            )
            web_ms = (time.perf_counter() - t_web) * 1000
            # 收集 web provider 诊断
            wp_diag: Dict[str, Any] = {}
            for h in results:
                prov = (h.get("metadata") or {}).get("provider") or "unknown"
                if prov not in wp_diag:
                    wp_diag[prov] = {"count": 0, "time_ms": round(web_ms, 1)}
                wp_diag[prov]["count"] += 1
            diag["web_providers"] = wp_diag
            # content_fetcher 统计（由 enrich_results 添加到 metadata）
            enriched = sum(1 for h in results if (h.get("metadata") or {}).get("content_type") == "full_text")
            if enriched > 0:
                diag["content_fetcher"] = {"enriched": enriched, "total": len(results)}
            return results

        # 获取阈值过滤参数和最终保留数量
        local_threshold = (filters or {}).get("local_threshold")
        final_top_k = (filters or {}).get("final_top_k")

        if mode == "hybrid":
            local_hits: List[Dict[str, Any]] = []
            web_hits: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=2) as ex:
                fl = ex.submit(_do_local)
                fw = ex.submit(_do_web)
                try:
                    local_hits = fl.result(timeout=timeout_s)
                except (FuturesTimeoutError, Exception):
                    pass
                try:
                    web_hits = fw.result(timeout=timeout_s)
                except (FuturesTimeoutError, Exception):
                    pass
            for h in local_hits:
                # 应用阈值过滤
                score = float(h.get("score", 0.0))
                if local_threshold is not None and score < local_threshold:
                    continue
                st = h.get("source") or (h.get("metadata") or {}).get("source")
                source_type = "graph" if st == "graph" else "dense"
                if "dense" not in sources_used and source_type == "dense":
                    sources_used.append("dense")
                if source_type == "graph" and "graph" not in sources_used:
                    sources_used.append("graph")
                all_chunks.append(_hit_to_chunk(h, source_type, query))
            total_candidates += len(local_hits) * 2 + len(web_hits)
            if not sources_used and all_chunks:
                sources_used.append("dense")
            # ── 跨源去重：拦截 web 中与本地重叠的文献 ──
            if web_hits and all_chunks:
                web_before = len(web_hits)
                web_hits = cross_source_dedup(all_chunks, web_hits)
                dedup_removed = web_before - len(web_hits)
                if dedup_removed > 0:
                    diag["cross_source_dedup"] = {"removed": dedup_removed, "remaining": len(web_hits)}
            # Web 结果 rerank（多语言 ColBERT 支持中英文）
            if web_hits:
                try:
                    web_hits = _rerank_candidates(
                        query, web_hits, top_k=min(k, len(web_hits))
                    )
                except Exception as e:
                    logger.warning("web rerank failed: %s", e)
            for h in web_hits:
                if "web" not in sources_used:
                    sources_used.append("web")
                all_chunks.append(_hit_to_chunk(h, "web", query))
        else:
            if mode == "local":
                config = RetrievalConfig(mode="hybrid", top_k=actual_recall, rerank=True)
                hits = self.retriever.retrieve(query, self.collection, config, diagnostics=diag)
                total_candidates += len(hits) * 2
                for h in hits:
                    # 应用阈值过滤
                    score = float(h.get("score", 0.0))
                    if local_threshold is not None and score < local_threshold:
                        continue
                    st = h.get("source") or (h.get("metadata") or {}).get("source")
                    source_type = "graph" if st == "graph" else "dense"
                    if "dense" not in sources_used and source_type == "dense":
                        sources_used.append("dense")
                    if source_type == "graph" and "graph" not in sources_used:
                        sources_used.append("graph")
                    all_chunks.append(_hit_to_chunk(h, source_type, query))
                if not sources_used and all_chunks:
                    sources_used.append("dense")
            elif mode == "web":
                try:
                    web_providers = (filters or {}).get("web_providers")
                    web_source_configs = (filters or {}).get("web_source_configs") or {}
                    use_query_expansion = (filters or {}).get("use_query_expansion")
                    use_query_optimizer = (filters or {}).get("use_query_optimizer")
                    query_optimizer_max_queries = (filters or {}).get("query_optimizer_max_queries")
                    _llm_provider = (filters or {}).get("llm_provider")
                    _model_override = (filters or {}).get("model_override")
                    # 参数级联
                    _final = (filters or {}).get("final_top_k") or k
                    _n_providers = max(len(web_providers or []), 1)
                    _n_queries = query_optimizer_max_queries or 3
                    _auto_per_provider = max(5, math.ceil(_final * 3.5 / (_n_providers * _n_queries)))
                    _use_content_fetcher = (filters or {}).get("use_content_fetcher")

                    # 为 Lazy Fetching 预判构建 LLM 客户端（仅智能模式 use_content_fetcher=None）
                    _llm_client = None
                    if _use_content_fetcher is None:
                        try:
                            from src.llm.llm_manager import get_manager
                            _llm_client = get_manager().get_client(_llm_provider or "deepseek")
                        except Exception as _e:
                            logger.debug("Lazy Fetching LLM 客户端初始化失败，降级全量抓取: %s", _e)

                    web_hits = unified_web_searcher.search_sync(
                        query,
                        providers=web_providers,
                        source_configs=web_source_configs,
                        max_results_per_provider=_auto_per_provider,
                        use_query_expansion=use_query_expansion,
                        use_query_optimizer=use_query_optimizer,
                        query_optimizer_max_queries=query_optimizer_max_queries,
                        llm_provider=_llm_provider,
                        model_override=_model_override,
                        use_content_fetcher=_use_content_fetcher,
                        llm_client=_llm_client,
                    )
                    total_candidates += len(web_hits)
                    sources_used.append("web")
                    # Web 结果 rerank
                    if web_hits:
                        try:
                            web_hits = _rerank_candidates(
                                query, web_hits, top_k=min(k, len(web_hits))
                            )
                        except Exception as e:
                            logger.warning("web rerank failed: %s", e)
                    for h in web_hits:
                        all_chunks.append(_hit_to_chunk(h, "web", query))
                except Exception:
                    pass

        retrieval_time_ms = (time.perf_counter() - t0) * 1000

        # ── Observability: 记录检索指标 ──
        if obs_metrics:
            elapsed_s = retrieval_time_ms / 1000.0
            obs_metrics.retrieval_total.labels(mode=mode).inc()
            obs_metrics.retrieval_duration_seconds.labels(mode=mode).observe(elapsed_s)
            obs_metrics.retrieval_chunks_returned.labels(mode=mode).observe(len(all_chunks))

        # 使用 final_top_k 作为最终保留数量，否则使用 k
        result_limit = final_top_k if final_top_k is not None else k
        
        return EvidencePack(
            query=query,
            chunks=all_chunks[:result_limit],
            total_candidates=total_candidates,
            retrieval_time_ms=retrieval_time_ms,
            sources_used=list(dict.fromkeys(sources_used)),
            diagnostics=diag if diag else None,
        )


# 全局实例（按 collection 复用）
_retrieval_services: Dict[str, RetrievalService] = {}


def get_retrieval_service(
    collection: Optional[str] = None,
    top_k: int = 10,
) -> RetrievalService:
    """获取 RetrievalService（按 collection 维度复用实例）"""
    key = (collection or settings.collection.global_ or "").strip() or settings.collection.global_
    svc = _retrieval_services.get(key)
    if svc is None:
        svc = RetrievalService(collection=key, top_k=top_k)
        _retrieval_services[key] = svc
    return svc
