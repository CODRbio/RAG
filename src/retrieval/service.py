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
from src.retrieval.fulltext_compressor import compress_fulltext_hits_sync
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig, _rerank_candidates
from src.retrieval.unified_web_search import unified_web_searcher

logger = logging.getLogger(__name__)


def _embedding_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Fast embedding-based reranking via BGE-M3 cosine similarity.

    Uses the already-loaded BGE-M3 bi-encoder to batch-encode documents and
    rank by dense cosine similarity with the query.  ~3-7 s for 200 docs on
    MPS vs ~45 s for cross-encoder — an order of magnitude faster while
    still providing genuine semantic ranking (unlike raw API score sort).
    """
    import numpy as np
    from src.indexing.embedder import embedder

    if not candidates:
        return []

    docs = [c.get("content") or "" for c in candidates]
    valid = [(i, d) for i, d in enumerate(docs) if d.strip()]
    if not valid:
        return candidates[:top_k]

    valid_indices, valid_docs = zip(*valid)

    query_dense = embedder.encode([query])["dense"][0]
    q_norm = np.linalg.norm(query_dense)
    if q_norm < 1e-10:
        return candidates[:top_k]
    query_unit = query_dense / q_norm

    doc_dense = embedder.encode(list(valid_docs))["dense"]

    scored: List[tuple] = []
    for j, orig_idx in enumerate(valid_indices):
        d_norm = np.linalg.norm(doc_dense[j])
        sim = float(np.dot(query_unit, doc_dense[j] / d_norm)) if d_norm > 1e-10 else 0.0
        scored.append((orig_idx, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [{**candidates[idx], "score": sc} for idx, sc in scored[:top_k]]

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

    provider = meta.get("provider") or meta.get("source")
    if not provider:
        provider = "local" if source_type in ("dense", "graph") else "web"

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
        provider=str(provider) if provider else None,
    )


def _compress_web_fulltext(
    web_hits: List[Dict[str, Any]],
    query: str,
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compress long full-text web hits (rerank 后、转 chunk 前). On failure return unchanged."""
    if not web_hits:
        return web_hits
    cfg = getattr(settings, "content_fetcher", None)
    if not cfg or not getattr(cfg, "compress_long_fulltext", True):
        return web_hits
    try:
        from src.llm.llm_manager import get_manager
        ultra_lite = (filters or {}).get("ultra_lite_provider")
        llm_client = get_manager().get_ultra_lite_client(ultra_lite)
    except Exception as e:
        logger.debug("fulltext compressor: no LLM client, skip: %s", e)
        return web_hits
    try:
        compress_fulltext_hits_sync(
            web_hits,
            query,
            llm_client,
            word_threshold=getattr(cfg, "compress_word_threshold", 300),
            max_output_words=getattr(cfg, "compress_max_output_words", 400),
            max_concurrent=getattr(cfg, "max_concurrent", 5),
        )
    except Exception as e:
        logger.warning("fulltext compression failed, using original content: %s", e)
    return web_hits


def _coerce_year(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year < 1900 or year > 2100:
        return None
    return year


def _normalize_year_window(filters: Optional[Dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
    year_start = _coerce_year((filters or {}).get("year_start"))
    year_end = _coerce_year((filters or {}).get("year_end"))
    if year_start is not None and year_end is not None and year_start > year_end:
        year_start, year_end = year_end, year_start
    return year_start, year_end


# ── Pool fusion constants ──────────────────────────────────────────────────────
# Fraction of top_k slots reserved for gap candidates when they are present.
_GAP_MIN_KEEP_RATIO: float = 0.25
# Additive gap score boost as a fraction of the reranked score range.
_GAP_SCORE_BOOST: float = 0.10


def fuse_pools_with_gap_protection(
    query: str,
    main_candidates: List[Dict[str, Any]],
    gap_candidates: List[Dict[str, Any]],
    top_k: int,
    *,
    gap_boost: float = _GAP_SCORE_BOOST,
    gap_min_keep: Optional[int] = None,
    reranker_mode: Optional[str] = None,
    skip_rerank: bool = False,
    diag: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Merge main and gap candidate pools with gap-quota protection.

    Algorithm:
      1. Tag all candidates with pool membership (_pool_tag: "main" | "gap").
      2. Run ONE global rerank pass on the combined pool so cross-source ordering
         is based on a single relevance scale.
      3. Apply additive gap_boost to gap candidates: score += gap_boost * score_range.
      4. Re-sort by boosted score.
      5. Enforce gap_min_keep: if fewer than gap_min_keep gap items land in the top_k
         slice, force-include the highest-scoring remaining gap candidates by swapping
         out the lowest-scoring main candidates (preserving output length == top_k).
      6. Strip internal tags and return.

    Args:
        main_candidates: primary pool hits (local dense + main web hits).
        gap_candidates:  gap/supplement pool hits (eval_supplement, gap-query hits).
        top_k:           final output count after fusion.
        gap_boost:       fractional boost for gap candidate scores
                         (relative to reranked score range). Default 0.10.
        gap_min_keep:    guaranteed gap slots in top_k output.
                         Defaults to ceil(top_k * _GAP_MIN_KEEP_RATIO) when
                         gap_candidates is non-empty; 0 otherwise.
        reranker_mode:   reranker mode forwarded to _rerank_candidates.
        skip_rerank:     use fast embedding rerank instead of cross-encoder.
        diag:            optional diagnostics dict; updated in-place with pool stats.

    Returns:
        Fused and ordered list of candidate dicts, length <= top_k.
        Internal keys (_pool_tag) are stripped; _source_type is preserved for
        callers that need it (e.g. service.py hybrid flow).
    """
    n_main = len(main_candidates)
    n_gap = len(gap_candidates)
    if n_main == 0 and n_gap == 0:
        return []

    # Tag all candidates with pool membership (copies dicts to avoid mutation)
    all_cands: List[Dict[str, Any]] = []
    for c in main_candidates:
        all_cands.append({**c, "_pool_tag": "main"})
    for c in gap_candidates:
        all_cands.append({**c, "_pool_tag": "gap"})

    n_total = len(all_cands)
    rerank_k = min(max(top_k * 2, top_k + n_gap), n_total)

    # Global rerank — one pass covering all candidates from both pools
    try:
        if skip_rerank:
            reranked = _embedding_rerank(query, all_cands, top_k=rerank_k)
        else:
            reranked = _rerank_candidates(
                query, all_cands, top_k=rerank_k, reranker_mode=reranker_mode,
            )
    except Exception as e:
        logger.warning("fuse_pools global rerank failed (%s); falling back to score sort", e)
        reranked = sorted(
            all_cands, key=lambda x: float(x.get("score", 0.0)), reverse=True
        )[:rerank_k]

    if not reranked:
        # Strip tags and return head slice
        return [{k: v for k, v in c.items() if k != "_pool_tag"} for c in all_cands[:top_k]]

    # Compute score range for proportional gap boost
    scores = [float(c.get("score", 0.0)) for c in reranked]
    score_max = max(scores)
    score_min = min(scores)
    score_range = score_max - score_min
    # When all scores are identical, use a small absolute boost so gap items
    # are still distinguishable.
    boost_abs = gap_boost * score_range if score_range > 1e-9 else gap_boost * 0.05

    # Apply boost to gap candidates and re-sort
    boosted: List[Dict[str, Any]] = []
    for c in reranked:
        if c.get("_pool_tag") == "gap":
            boosted.append({**c, "score": float(c.get("score", 0.0)) + boost_abs})
        else:
            boosted.append(c)
    boosted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # Resolve effective gap quota
    effective_min_keep: int = 0
    if n_gap > 0:
        if gap_min_keep is not None:
            effective_min_keep = gap_min_keep
        else:
            effective_min_keep = max(1, math.ceil(top_k * _GAP_MIN_KEEP_RATIO))
        effective_min_keep = min(effective_min_keep, n_gap, top_k)

    # Gap quota enforcement: force-include top gap items if quota not met in top_k
    top_slice = boosted[:top_k]
    if effective_min_keep > 0:
        gap_in_top = [c for c in top_slice if c.get("_pool_tag") == "gap"]
        gap_deficit = effective_min_keep - len(gap_in_top)
        if gap_deficit > 0:
            top_ids = {
                str(c.get("chunk_id") or (c.get("metadata") or {}).get("chunk_id") or i)
                for i, c in enumerate(top_slice)
            }
            gap_reserve = [
                c for c in boosted[top_k:]
                if c.get("_pool_tag") == "gap"
            ]
            forced = gap_reserve[:gap_deficit]
            if forced:
                # Replace lowest-scoring main candidates to keep length == top_k
                main_in_top = [i for i, c in enumerate(top_slice) if c.get("_pool_tag") == "main"]
                drop_indices = set(main_in_top[-len(forced):]) if main_in_top else set()
                top_slice = [c for i, c in enumerate(top_slice) if i not in drop_indices]
                top_slice.extend(forced)
                top_slice.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    out = top_slice

    # Strip _pool_tag (keep _source_type and other internal keys for callers)
    clean = [{k: v for k, v in c.items() if k != "_pool_tag"} for c in out]

    if diag is not None:
        gap_in_output = sum(1 for c in out if c.get("_pool_tag") == "gap")
        diag["pool_fusion"] = {
            "main_in": n_main,
            "gap_in": n_gap,
            "total_reranked": len(reranked),
            "gap_boost_abs": round(boost_abs, 5),
            "gap_min_keep": effective_min_keep,
            "gap_in_output": gap_in_output,
            "output_count": len(clean),
        }

    return clean


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
        step_top_k = (filters or {}).get("step_top_k")
        ui_reranker_mode = (filters or {}).get("reranker_mode") or None
        result_limit = step_top_k if step_top_k is not None else k
        _cascade_bge_k = (filters or {}).get("colbert_top_k") or (filters or {}).get("step_top_k")
        logger.debug(
            "[service.search] query=%r mode=%s top_k=%s → k=%d | step_top_k=%s → result_limit=%d"
            " | reranker_mode=%s | web_providers=%s | year=%s~%s | optimizer=%s | serpapi_ratio=%s",
            query[:60], mode, top_k, k,
            step_top_k, result_limit,
            ui_reranker_mode,
            ",".join((filters or {}).get("web_providers") or []) or "none",
            (filters or {}).get("year_start"), (filters or {}).get("year_end"),
            (filters or {}).get("use_query_optimizer"),
            (filters or {}).get("serpapi_ratio"),
        )

        sources_used: List[str] = []
        all_chunks: List[EvidenceChunk] = []
        total_candidates = 0
        t0 = time.perf_counter()
        timeout_s = getattr(getattr(settings, "perf_retrieval", None), "timeout_seconds", 60) or 60

        # Amplify retrieval pool so that the reranker always operates on a
        # sufficiently large candidate set, regardless of the caller's top_k.
        actual_recall = max(80, result_limit * 4)
        year_start, year_end = _normalize_year_window(filters)

        # 诊断信息收集
        diag: Dict[str, Any] = {}
        if actual_recall != k:
            diag["recall_amplification"] = {"requested_k": k, "actual_recall": actual_recall}
        if year_start is not None or year_end is not None:
            diag["year_window"] = {"year_start": year_start, "year_end": year_end}

        def _do_local() -> List[Dict[str, Any]]:
            config = RetrievalConfig(
                mode="hybrid",
                top_k=actual_recall,
                rerank=True,
                year_start=year_start,
                year_end=year_end,
                reranker_mode=ui_reranker_mode,
                step_top_k=(filters or {}).get("step_top_k"),
            )
            return self.retriever.retrieve(query, self.collection, config, diagnostics=diag)

        def _do_web() -> List[Dict[str, Any]]:
            web_providers = (filters or {}).get("web_providers")
            if web_providers is not None and len(web_providers) == 0:
                return []
            # Chat 时前端传 web_source_configs：每个 provider 在各组查询下使用该 provider 的 topK
            web_source_configs = (filters or {}).get("web_source_configs") or {}
            use_query_expansion = (filters or {}).get("use_query_expansion")
            use_query_optimizer = (filters or {}).get("use_query_optimizer")
            query_optimizer_max_queries = (filters or {}).get("query_optimizer_max_queries")
            _llm_provider = (filters or {}).get("llm_provider")
            _model_override = (filters or {}).get("model_override")
            # 无 web_source_configs 时用 step_top_k 推导统一上限；有则各 provider 用自身 topK
            _final = (filters or {}).get("step_top_k") or k
            _n_providers = max(len(web_providers or []), 1)
            _n_queries = query_optimizer_max_queries or 3
            _auto_per_provider = max(5, math.ceil(_final * 3.5 / (_n_providers * _n_queries)))
            logger.info(
                "web search params: providers=%s auto_per_provider=%s step_top_k=%s llm=%s/%s",
                web_providers,
                _auto_per_provider,
                _final,
                _llm_provider,
                _model_override,
            )
            use_content_fetcher = (filters or {}).get("use_content_fetcher")
            _queries_per_provider = (filters or {}).get("web_queries_per_provider")
            _semantic_query_map = (filters or {}).get("semantic_query_map")
            _serpapi_ratio = (filters or {}).get("serpapi_ratio")

            # 为 Lazy Fetching 预判构建 LLM 客户端（仅智能模式 use_content_fetcher=None 或 'auto'，使用 lite 降级）
            _llm_client = None
            if use_content_fetcher in (None, "auto"):
                try:
                    from src.llm.llm_manager import get_manager
                    _llm_client = get_manager().get_lite_client(_llm_provider or "deepseek")
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
                year_start=year_start,
                year_end=year_end,
                queries_per_provider=_queries_per_provider,
                semantic_query_map=_semantic_query_map,
                serpapi_ratio=_serpapi_ratio,
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
        skip_rerank = bool((filters or {}).get("skip_rerank"))

        if mode == "hybrid":
            local_hits: List[Dict[str, Any]] = []
            web_hits: List[Dict[str, Any]] = []

            # Return more local candidates so the global cross-source rerank has
            # a rich pool to work with; final truncation to result_limit is done
            # by fuse_pools_with_gap_protection after merging with web results.
            local_recall_k = min(actual_recall, max(result_limit * 2, 20))
            local_config = RetrievalConfig(
                mode="hybrid",
                top_k=actual_recall,
                rerank=not skip_rerank,
                year_start=year_start,
                year_end=year_end,
                reranker_mode=ui_reranker_mode,
                step_top_k=local_recall_k,
            )

            # ── Soft-wait: local gets a hard timeout; web gets an extended budget
            # so slow providers (Scholar browser) can complete rather than being
            # silently dropped.  Threads are not killed on timeout — they finish
            # in the background as I/O-bound work naturally completes.
            soft_wait_s = min(timeout_s * 5, 300)

            ex = ThreadPoolExecutor(max_workers=2)
            fl = ex.submit(
                self.retriever.retrieve,
                query, self.collection, local_config, diagnostics=diag,
            )
            fw = ex.submit(_do_web)

            t_parallel_start = time.perf_counter()
            try:
                local_hits = fl.result(timeout=timeout_s)
            except FuturesTimeoutError:
                diag["local_timeout"] = {"timeout_s": timeout_s}
                logger.warning("[hybrid] local retrieval timed out after %.0fs", timeout_s)
            except Exception as e:
                logger.warning("[hybrid] local retrieval failed: %s", e)

            elapsed_s = time.perf_counter() - t_parallel_start
            web_wait_s = max(1.0, soft_wait_s - elapsed_s)
            t_web_start = time.perf_counter()
            try:
                web_hits = fw.result(timeout=web_wait_s)
                web_elapsed_ms = (time.perf_counter() - t_web_start) * 1000
                if web_elapsed_ms > timeout_s * 1000:
                    diag["soft_wait_ms"] = round(web_elapsed_ms)
                    logger.info("[hybrid] web soft-wait completed in %.0fms", web_elapsed_ms)
            except FuturesTimeoutError:
                diag["web_timeout"] = {"soft_wait_s": round(web_wait_s, 1)}
                logger.warning("[hybrid] web retrieval soft-wait timed out after %.0fs", web_wait_s)
            except Exception as e:
                logger.warning("[hybrid] web retrieval failed: %s", e)
            finally:
                ex.shutdown(wait=False)

            # ── Threshold-filter local hits and tag with source type ──
            local_main: List[Dict[str, Any]] = []
            for h in local_hits:
                score = float(h.get("score", 0.0))
                if local_threshold is not None and score < local_threshold:
                    continue
                st = h.get("source") or (h.get("metadata") or {}).get("source")
                h["_source_type"] = "graph" if st == "graph" else "dense"
                local_main.append(h)
                if "dense" not in sources_used and h["_source_type"] == "dense":
                    sources_used.append("dense")
                if h["_source_type"] == "graph" and "graph" not in sources_used:
                    sources_used.append("graph")

            if not sources_used and local_main:
                sources_used.append("dense")
            total_candidates += len(local_hits) * 2 + len(web_hits)

            # ── Cross-source dedup: build EvidenceChunks from local for the dedup
            # reference set, then filter web_hits against local documents ──
            local_chunks_for_dedup = [
                _hit_to_chunk(h, h.get("_source_type", "dense"), query)
                for h in local_main
            ]
            if web_hits and local_chunks_for_dedup:
                web_before = len(web_hits)
                web_hits = cross_source_dedup(local_chunks_for_dedup, web_hits)
                dedup_removed = web_before - len(web_hits)
                if dedup_removed > 0:
                    diag["cross_source_dedup"] = {"removed": dedup_removed, "remaining": len(web_hits)}

            # ── Tag web hits with source type, compress fulltext ──
            web_main: List[Dict[str, Any]] = []
            for h in web_hits:
                h["_source_type"] = "web"
                web_main.append(h)
                if "web" not in sources_used:
                    sources_used.append("web")
            web_main = _compress_web_fulltext(web_main, query, filters)

            # ── Global fusion: single ranked pool → top result_limit ──
            # Local + web are merged and globally reranked in one pass so
            # cross-source ordering reflects a unified relevance scale.
            # For the chat path there is no dedicated gap pool; gap protection
            # is applied per-section in Deep Research via _rerank_section_pool_chunks.
            fused_hits = fuse_pools_with_gap_protection(
                query=query,
                main_candidates=local_main + web_main,
                gap_candidates=[],
                top_k=result_limit,
                reranker_mode=ui_reranker_mode,
                skip_rerank=skip_rerank,
                diag=diag,
            )

            for h in fused_hits:
                source_type = h.get("_source_type", "dense")
                all_chunks.append(_hit_to_chunk(h, source_type, query))
        else:
            if mode == "local":
                config = RetrievalConfig(
                    mode="hybrid",
                    top_k=actual_recall,
                    rerank=not skip_rerank,
                    year_start=year_start,
                    year_end=year_end,
                    reranker_mode=ui_reranker_mode,
                    step_top_k=(filters or {}).get("step_top_k"),
                )
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
                    # 各 provider 使用 web_source_configs 中该 provider 的 topK
                    web_source_configs = (filters or {}).get("web_source_configs") or {}
                    use_query_expansion = (filters or {}).get("use_query_expansion")
                    use_query_optimizer = (filters or {}).get("use_query_optimizer")
                    query_optimizer_max_queries = (filters or {}).get("query_optimizer_max_queries")
                    _llm_provider = (filters or {}).get("llm_provider")
                    _model_override = (filters or {}).get("model_override")
                    # 参数级联
                    _final = (filters or {}).get("step_top_k") or k
                    _n_providers = max(len(web_providers or []), 1)
                    _n_queries = query_optimizer_max_queries or 3
                    _auto_per_provider = max(5, math.ceil(_final * 3.5 / (_n_providers * _n_queries)))
                    _use_content_fetcher = (filters or {}).get("use_content_fetcher")
                    _semantic_query_map = (filters or {}).get("semantic_query_map")
                    _serpapi_ratio = (filters or {}).get("serpapi_ratio")

                    # 为 Lazy Fetching 预判构建 LLM 客户端（仅智能模式 _use_content_fetcher=None 或 'auto'，使用 lite 降级）
                    _llm_client = None
                    if _use_content_fetcher in (None, "auto"):
                        try:
                            from src.llm.llm_manager import get_manager
                            _llm_client = get_manager().get_lite_client(_llm_provider or "deepseek")
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
                        year_start=year_start,
                        year_end=year_end,
                        semantic_query_map=_semantic_query_map,
                        serpapi_ratio=_serpapi_ratio,
                    )
                    total_candidates += len(web_hits)
                    sources_used.append("web")
                    # Web 结果排序
                    if web_hits:
                        _web_top_k = min(result_limit, len(web_hits))
                        if skip_rerank:
                            try:
                                web_hits = _embedding_rerank(query, web_hits, top_k=_web_top_k)
                            except Exception as e:
                                logger.warning("embedding rerank failed: %s", e)
                                web_hits = web_hits[:_web_top_k]
                        else:
                            try:
                                web_hits = _rerank_candidates(
                                    query, web_hits, top_k=_web_top_k, reranker_mode=ui_reranker_mode,
                                )
                            except Exception as e:
                                logger.warning("web cross-encoder rerank failed, falling back to embedding: %s", e)
                                try:
                                    web_hits = _embedding_rerank(query, web_hits, top_k=_web_top_k)
                                except Exception:
                                    web_hits = web_hits[:_web_top_k]
                    web_hits = _compress_web_fulltext(web_hits, query, filters)
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

        # 使用 step_top_k 作为最终保留数量，否则使用 k
        result_limit = step_top_k if step_top_k is not None else k
        
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
