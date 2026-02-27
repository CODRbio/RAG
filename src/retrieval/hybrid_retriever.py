"""
混合检索模块（整合 HippoRAG）

支持三种检索模式：
1. vector: 纯向量检索（Dense + Sparse + Weighted RRF）
2. graph: 纯图检索（PPR）
3. hybrid: 向量 + 图融合（默认）

三阶段架构：
Stage 1: Recall (dense/sparse 各 80)
Stage 2: Weighted RRF Fusion (dense=0.6, sparse=0.4)
Stage 3: Rerank (输入 80-120 -> 输出 10)
"""

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError  # noqa: A001
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from pymilvus import AnnSearchRequest, RRFRanker

from config.settings import settings
from src.indexing.milvus_ops import milvus, MilvusOps
from src.indexing.embedder import embedder
from src.graph.hippo_rag import HippoRAG, get_hippo_rag
from src.graph.entity_extractor import ExtractorConfig
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

logger = get_logger(__name__)

try:
    from src.retrieval.dedup import dedup_and_diversify
except ImportError:
    def dedup_and_diversify(candidates: List[Dict], per_doc_cap: int = 3) -> List[Dict]:
        return candidates

try:
    from src.retrieval.colbert_reranker import colbert_reranker
except ImportError:
    colbert_reranker = None


def _count_entities(query: str) -> int:
    """简单实体计数：2+ 中文字符短语或首字母大写的英文词"""
    import re
    zh = len(re.findall(r"[\u4e00-\u9fff]{2,}", query or ""))
    en = len(re.findall(r"\b[A-Z][a-z]+\b", query or ""))
    return zh + en


def should_use_hipporag(query: str) -> bool:
    """是否触发 HippoRAG：多实体 + 关系类关键词"""
    relation_keywords = ["关系", "关联", "因果", "影响", "链", "比较", "对比", "相互作用", "联系"]
    multi_entity = _count_entities(query) >= 2
    has_relation_word = any(kw in (query or "") for kw in relation_keywords)
    return bool(multi_entity and has_relation_word)


def _get_search_params():
    return getattr(settings.search, "dense_recall_k", 80), getattr(settings.search, "sparse_recall_k", 80)


def weighted_rrf(
    dense_list: List[Dict],
    sparse_list: List[Dict],
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    k: int = 60,
) -> List[tuple[str, float]]:
    """应用层加权 RRF 融合"""
    scores: Dict[str, float] = defaultdict(float)
    for rank, doc in enumerate(dense_list):
        cid = doc.get("chunk_id") or doc.get("metadata", {}).get("chunk_id") or str(id(doc))
        scores[cid] += w_dense / (k + rank + 1)
    for rank, doc in enumerate(sparse_list):
        cid = doc.get("chunk_id") or doc.get("metadata", {}).get("chunk_id") or str(id(doc))
        scores[cid] += w_sparse / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _embedding_pre_filter(
    query: str,
    candidates: List[Dict],
    keep_k: int,
) -> List[Dict]:
    """Fast bi-encoder pre-filter: encode query + docs with BGE-M3, rank by
    cosine similarity, return the top *keep_k* candidates.

    Used as a funnel to shrink large candidate sets (200+) before feeding
    them to the expensive cross-encoder.  ~3-7 s for 200 docs on MPS.
    """
    import numpy as np

    docs = [c.get("content") or "" for c in candidates]
    valid = [(i, d) for i, d in enumerate(docs) if d.strip()]
    if not valid or len(valid) <= keep_k:
        logger.debug("embedding_pre_filter: %d candidates ≤ keep_k=%d, no filter needed", len(candidates), keep_k)
        return candidates[:keep_k]

    valid_indices, valid_docs = zip(*valid)
    t0 = time.perf_counter()
    try:
        query_dense = embedder.encode([query])["dense"][0]
        q_norm = np.linalg.norm(query_dense)
        if q_norm < 1e-10:
            logger.warning("embedding_pre_filter: zero-norm query vector, returning head slice")
            return candidates[:keep_k]
        query_unit = query_dense / q_norm

        doc_dense = embedder.encode(list(valid_docs))["dense"]
    except Exception as e:
        logger.error("embedding_pre_filter: encode failed (%s), returning head slice", e, exc_info=True)
        return candidates[:keep_k]

    scored: List[Tuple[int, float]] = []
    for j, orig_idx in enumerate(valid_indices):
        d_norm = np.linalg.norm(doc_dense[j])
        sim = float(np.dot(query_unit, doc_dense[j] / d_norm)) if d_norm > 1e-10 else 0.0
        scored.append((orig_idx, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "embedding_pre_filter: %d → %d candidates in %.0f ms (top_sim=%.4f)",
        len(candidates), keep_k, elapsed_ms,
        scored[0][1] if scored else 0.0,
    )
    return [candidates[idx] for idx, _ in scored[:keep_k]]


def _rerank_candidates(
    query: str,
    candidates: List[Dict],
    top_k: int,
    reranker_mode: Optional[str] = None,
    cascade_bge_k: Optional[int] = None,
) -> List[Dict]:
    """Cross-encoder reranking with an embedding funnel for large input sets.

    The funnel threshold is derived dynamically from the caller's *top_k* and
    ``settings.search.rerank_input_k`` (config / UI controllable).  When the
    candidate count exceeds that threshold the bi-encoder pre-filter narrows
    the set so the cross-encoder runs on a bounded number of documents.

    Supports bge_only | colbert_only | cascade.
    Priority: reranker_mode param > settings.search.reranker_mode.
    When cascade_bge_k is set (e.g. from UI step_top_k), cascade uses it as
    the BGE-stage output size instead of config colbert_top_k.
    """
    if not candidates:
        logger.debug("_rerank_candidates: empty candidates, skipping")
        return []

    t_total = time.perf_counter()
    cfg_input_k = getattr(settings.search, "rerank_input_k", 100)
    funnel_k = max(cfg_input_k, top_k * 3)
    use_colbert = getattr(settings.search, "use_colbert_reranker", False)
    # Per-request mode overrides global settings; ColBERT modes are only effective
    # when use_colbert_reranker is enabled or the request explicitly asks for colbert.
    if reranker_mode:
        if reranker_mode in ("colbert_only", "cascade") and not use_colbert:
            # ColBERT not available server-side → downgrade to bge_only
            mode = "bge_only"
        else:
            mode = reranker_mode
    else:
        mode = getattr(settings.search, "reranker_mode", "bge_only") if use_colbert else "bge_only"
    colbert_top_k = cascade_bge_k if cascade_bge_k is not None else getattr(settings.search, "colbert_top_k", 30)

    logger.info(
        "rerank_start: candidates=%d top_k=%d mode=%s (requested=%s) funnel_k=%d",
        len(candidates), top_k, mode, reranker_mode or "default", funnel_k,
    )

    # ── Funnel: shrink large sets before cross-encoder ──
    n_before_funnel = len(candidates)
    if len(candidates) > funnel_k:
        candidates = _embedding_pre_filter(query, candidates, funnel_k)
        logger.info(
            "rerank_funnel: %d → %d (rerank_input_k=%d, top_k=%d)",
            n_before_funnel, len(candidates), cfg_input_k, top_k,
        )

    docs = [c.get("content") or "" for c in candidates]

    # ── colbert_only ──
    if mode == "colbert_only" and colbert_reranker is not None and colbert_reranker.available:
        logger.info("rerank_colbert_only: %d docs → top_k=%d", len(docs), top_k)
        t = time.perf_counter()
        reranked = colbert_reranker.rerank(query, docs, top_k=min(top_k * 2, len(docs)))
        if reranked:
            result = [{**candidates[r.index], "score": r.score} for r in reranked][:top_k]
            logger.info("rerank_colbert_only: done in %.0f ms, returned %d", (time.perf_counter() - t) * 1000, len(result))
            return result
        logger.warning("rerank_colbert_only: empty result (%.0f ms) — falling back to BGE",
                       (time.perf_counter() - t) * 1000)

    # ── cascade: BGE → ColBERT ──
    if mode == "cascade" and colbert_reranker is not None and colbert_reranker.available:
        bge_k = min(colbert_top_k, len(docs))
        logger.info("rerank_cascade: BGE %d docs → top %d, then ColBERT → top_k=%d", len(docs), bge_k, top_k)
        t_bge = time.perf_counter()
        try:
            bge_out = embedder.rerank(query, docs, top_k=bge_k)
        except Exception as e:
            logger.error("rerank_cascade BGE failed (%.0f ms): %s — falling back to bge_only",
                         (time.perf_counter() - t_bge) * 1000, e, exc_info=True)
            # fall through to bge_only below
        else:
            logger.info("rerank_cascade BGE done in %.0f ms, got %d results",
                        (time.perf_counter() - t_bge) * 1000, len(bge_out))
            bge_candidates = [candidates[r.index] for r in bge_out]
            bge_docs = [c.get("content") or "" for c in bge_candidates]
            t_col = time.perf_counter()
            colbert_out = colbert_reranker.rerank(query, bge_docs, top_k=min(top_k * 2, len(bge_docs)))
            if colbert_out:
                result = [{**bge_candidates[r.index], "score": r.score} for r in colbert_out][:top_k]
                logger.info(
                    "rerank_cascade ColBERT done in %.0f ms, returned %d (total %.0f ms)",
                    (time.perf_counter() - t_col) * 1000, len(result),
                    (time.perf_counter() - t_total) * 1000,
                )
                return result
            # ColBERT returned empty (unavailable or runtime error) — use BGE results
            logger.warning(
                "rerank_cascade ColBERT returned empty (%.0f ms) — falling back to BGE result",
                (time.perf_counter() - t_col) * 1000,
            )
            result = [{**candidates[r.index], "score": r.score} for r in bge_out][:top_k]
            logger.info("rerank_cascade fallback: returning %d BGE results", len(result))
            return result

    # ── bge_only (default / fallback) ──
    logger.info("rerank_bge_only: %d docs → top_k=%d", len(docs), top_k)
    t_bge = time.perf_counter()
    try:
        reranked = embedder.rerank(query, docs, top_k=min(top_k * 2, len(docs)))
        result = [{**candidates[r.index], "score": r.score} for r in reranked][:top_k]
        logger.info(
            "rerank_bge_only done in %.0f ms, returned %d (total %.0f ms)",
            (time.perf_counter() - t_bge) * 1000, len(result),
            (time.perf_counter() - t_total) * 1000,
        )
        return result
    except Exception as e:
        logger.error("rerank_bge_only failed (%.0f ms): %s — returning head slice",
                     (time.perf_counter() - t_bge) * 1000, e, exc_info=True)
        return candidates[:top_k]


@dataclass
class RetrievalConfig:
    """检索配置"""
    mode: str = "hybrid"  # vector, graph, hybrid
    top_k: int = 10
    rerank: bool = True
    graph_weight: float = 0.3  # hybrid 模式下图检索权重
    year_start: Optional[int] = None  # 年份窗口起始（硬过滤）
    year_end: Optional[int] = None  # 年份窗口结束（硬过滤）
    reranker_mode: Optional[str] = None  # 覆盖全局 reranker_mode：bge_only | colbert_only | cascade
    colbert_top_k: Optional[int] = None  # cascade 时 BGE 阶段输出数，由 UI step_top_k 穿透时使用
    step_top_k: Optional[int] = None  # 每步检索最终保留数；未设时继承 settings.rerank_output_k


class HybridRetriever:
    """
    混合检索器

    整合向量检索和 HippoRAG 图检索
    """

    _year_field_cache: Dict[str, bool] = {}

    def __init__(self, graph_path: Optional[Path] = None):
        self.hippo: Optional[HippoRAG] = None
        self._graph_path = graph_path or (settings.path.data / "hippo_graph.json")
        self.logger = get_logger(__name__)
        perf = getattr(settings, "perf_retrieval", None)
        self._vector_cache: Optional[TTLCache] = (
            get_cache(
                getattr(perf, "cache_enabled", False),
                getattr(perf, "cache_ttl_seconds", 3600),
                prefix="retrieval_vector",
            )
            if perf else None
        )

    def _guard_year_expr(self, collection: str, year_expr: str) -> Optional[str]:
        """Return year_expr only if the collection supports 'year' filtering."""
        if collection not in self._year_field_cache:
            has_year = False
            try:
                schema = milvus.client.describe_collection(collection)
                has_year = any(f.get("name") == "year" for f in (schema.get("fields") or []))
            except Exception:
                pass
            if not has_year:
                try:
                    probe = milvus.client.query(
                        collection_name=collection, filter="year >= 0",
                        output_fields=["chunk_id"], limit=1,
                    )
                    has_year = len(probe) > 0
                except Exception:
                    pass
            self._year_field_cache[collection] = has_year
        if not self._year_field_cache[collection]:
            self.logger.info("collection '%s' has no 'year' field; skipping year filter", collection)
            return None
        return year_expr

    def _build_extractor_config(self) -> ExtractorConfig:
        cfg = settings.graph_entity_extraction
        return ExtractorConfig(
            strategy=cfg.strategy,
            fallback=cfg.fallback,
            ontology_path=cfg.ontology_path,
            gliner_model=cfg.gliner_model,
            gliner_threshold=cfg.gliner_threshold,
            gliner_device=cfg.gliner_device,
            llm_provider=cfg.llm_provider,
            llm_max_tokens=cfg.llm_max_tokens,
        )

    def _ensure_graph(self):
        """确保图谱已加载"""
        if self.hippo is None:
            if self._graph_path.exists():
                self.hippo = get_hippo_rag(
                    self._graph_path,
                    extractor_config=self._build_extractor_config(),
                )
            else:
                self.logger.warning("图谱文件不存在: %s", self._graph_path)
                self.logger.warning("请先运行: python scripts/03b_build_graph.py")

    def retrieve_vector(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        rerank: bool = True,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        diagnostics: Optional[Dict] = None,
        reranker_mode: Optional[str] = None,
        cascade_bge_k: Optional[int] = None,
        step_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        纯向量检索（Dense + Sparse + Weighted RRF + Rerank）
        Stage 1: 分离 dense/sparse 召回
        Stage 2: 应用层加权 RRF 融合
        Stage 3: Rerank (输入 80-120 -> 输出 10)

        Args:
            diagnostics: 可选字典，传入时填充各阶段 count + time_ms
        """
        dense_recall_k = getattr(settings.search, "dense_recall_k", 80)
        sparse_recall_k = getattr(settings.search, "sparse_recall_k", 80)
        rerank_input_k = getattr(settings.search, "rerank_input_k", 100)
        # 请求未传 step_top_k 时继承此默认（与 UI stepTopK 同义）
        rerank_output_k = getattr(settings.search, "rerank_output_k", 10)
        w_dense = getattr(settings.search, "rrf_dense_weight", 0.6)
        w_sparse = getattr(settings.search, "rrf_sparse_weight", 0.4)
        per_doc_cap = getattr(settings.search, "per_doc_cap", 3)

        cache_key = _make_key("retrieval_vector", query, collection, top_k, rerank, year_start, year_end, step_top_k)
        if self._vector_cache:
            cached = self._vector_cache.get(cache_key)
            if cached is not None:
                if diagnostics is not None:
                    diagnostics["cache_hit"] = True
                return cached

        emb = embedder.encode([query])
        dense_vec = emb["dense"][0]
        sparse_vec = emb["sparse"]._getrow(0)
        sparse_coo = sparse_vec.tocoo()
        sparse_dict = {int(col): float(val) for col, val in zip(sparse_coo.col, sparse_coo.data)}

        output_fields = [
            "content", "raw_content", "paper_id", "chunk_id",
            "domain", "content_type", "chunk_type", "section_path", "page"
        ]

        # Stage 1: 分离召回（可选并行 + 超时）
        year_expr = MilvusOps.build_year_expr(year_start=year_start, year_end=year_end) or None
        if year_expr:
            year_expr = self._guard_year_expr(collection, year_expr)
        dense_req = AnnSearchRequest(
            data=[dense_vec.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=dense_recall_k,
            expr=year_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=sparse_recall_k,
            expr=year_expr,
        )
        timeout_s = getattr(settings, "perf_retrieval", None)
        timeout_s = getattr(timeout_s, "timeout_seconds", 60) if timeout_s else 60
        parallel = getattr(getattr(settings, "perf_retrieval", None), "parallel_dense_sparse", True)
        max_workers = getattr(getattr(settings, "perf_retrieval", None), "max_workers", 4) or 4

        def _do_dense():
            return milvus.hybrid_search(
                collection=collection,
                reqs=[dense_req],
                ranker=RRFRanker(k=settings.search.rrf_k),
                limit=dense_recall_k,
                output_fields=output_fields,
            )

        def _do_sparse():
            return milvus.hybrid_search(
                collection=collection,
                reqs=[sparse_req],
                ranker=RRFRanker(k=settings.search.rrf_k),
                limit=sparse_recall_k,
                output_fields=output_fields,
            )

        dense_res, sparse_res = None, None
        t_recall = time.perf_counter()
        if parallel and max_workers >= 2:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fd = ex.submit(_do_dense)
                fs = ex.submit(_do_sparse)
                try:
                    dense_res = fd.result(timeout=timeout_s)
                except (FuturesTimeoutError, Exception) as e:
                    self.logger.warning("dense search timeout or error: %s", e)
                try:
                    sparse_res = fs.result(timeout=timeout_s)
                except (FuturesTimeoutError, Exception) as e:
                    self.logger.warning("sparse search timeout or error: %s", e)
        else:
            dense_res = _do_dense()
            sparse_res = _do_sparse()
        recall_ms = (time.perf_counter() - t_recall) * 1000

        def _hit_to_doc(hit: Any) -> Dict:
            e = getattr(hit, "entity", hit) if hasattr(hit, "entity") else (hit.get("entity", hit) if isinstance(hit, dict) else hit)
            cid = e.get("chunk_id", "") if isinstance(e, dict) else getattr(e, "chunk_id", "")
            return {
                "chunk_id": cid,
                "content": e.get("content") if isinstance(e, dict) else getattr(e, "content", None),
                "raw_content": e.get("raw_content") if isinstance(e, dict) else getattr(e, "raw_content", None),
                "metadata": {
                    "paper_id": e.get("paper_id", "") if isinstance(e, dict) else getattr(e, "paper_id", ""),
                    "chunk_id": cid,
                    "domain": e.get("domain", "") if isinstance(e, dict) else getattr(e, "domain", ""),
                    "content_type": e.get("content_type", "") if isinstance(e, dict) else getattr(e, "content_type", ""),
                    "chunk_type": e.get("chunk_type", "") if isinstance(e, dict) else getattr(e, "chunk_type", ""),
                    "section_path": e.get("section_path", "") if isinstance(e, dict) else getattr(e, "section_path", ""),
                    "page": e.get("page", 0) if isinstance(e, dict) else getattr(e, "page", 0),
                }
            }

        dense_hits = [_hit_to_doc(h) for h in (dense_res[0] if dense_res else [])]
        sparse_hits = [_hit_to_doc(h) for h in (sparse_res[0] if sparse_res else [])]

        # Stage 2: Weighted RRF
        t_fusion = time.perf_counter()
        fused = weighted_rrf(
            dense_hits, sparse_hits,
            w_dense=w_dense, w_sparse=w_sparse, k=settings.search.rrf_k
        )
        cid_to_doc = {d["chunk_id"]: d for d in dense_hits + sparse_hits}
        candidates = []
        for cid, _ in fused[:rerank_input_k]:
            if cid in cid_to_doc:
                candidates.append(cid_to_doc[cid])
        fusion_ms = (time.perf_counter() - t_fusion) * 1000

        # Stage 3: Rerank（bge_only | colbert_only | cascade）
        # 输出条数：请求 step_top_k 优先，否则继承 config rerank_output_k
        rerank_output_limit = step_top_k if step_top_k is not None else rerank_output_k
        t_rerank = time.perf_counter()
        reranked = False
        if rerank and candidates:
            docs = [c["content"] for c in candidates if c.get("content")]
            if docs:
                hits = _rerank_candidates(
                    query, candidates, top_k=min(rerank_output_limit, len(candidates)),
                    reranker_mode=reranker_mode, cascade_bge_k=cascade_bge_k,
                )
                hits = dedup_and_diversify(hits, per_doc_cap=per_doc_cap)
                out = hits[:step_top_k] if step_top_k is not None else hits[:top_k]
                reranked = True
        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        if not reranked:
            out = candidates[:step_top_k] if step_top_k is not None else candidates[:top_k]

        # 填充诊断信息
        if diagnostics is not None:
            diagnostics["stages"] = {
                "dense_recall": {"count": len(dense_hits), "time_ms": round(recall_ms, 1)},
                "sparse_recall": {"count": len(sparse_hits), "time_ms": round(recall_ms, 1)},
                "fusion": {"count": len(candidates), "time_ms": round(fusion_ms, 1)},
                "rerank": {"count": len(out), "time_ms": round(rerank_ms, 1)},
            }

        if self._vector_cache:
            self._vector_cache.set(cache_key, out)
        return out

    def retrieve_graph(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        纯图检索（PPR）
        """
        self._ensure_graph()
        if self.hippo is None:
            return []

        # 从查询抽取种子实体
        seed_entities = self.hippo.get_seed_entities(query)
        if not seed_entities:
            return []

        # PPR 扩展
        ppr_results = self.hippo.personalized_pagerank(seed_entities, top_k=top_k)

        # 转换为标准格式
        hits = []
        for chunk_id, score in ppr_results:
            paper_id = self.hippo.G.nodes[chunk_id].get("paper_id", "")
            hits.append({
                "content": "",  # 需要从 Milvus 补充
                "score": score,
                "metadata": {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                },
                "source": "graph"
            })

        return hits

    def retrieve_hybrid(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        rerank: bool = True,
        graph_weight: float = 0.3,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        diagnostics: Optional[Dict] = None,
        reranker_mode: Optional[str] = None,
        cascade_bge_k: Optional[int] = None,
        step_top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        混合检索（向量 + 图融合）
        条件触发 HippoRAG：多实体 + 关系类关键词时合并图检索结果
        """
        # 1. 向量检索
        vector_hits = self.retrieve_vector(
            query,
            collection,
            top_k=top_k * 2,
            rerank=False,
            year_start=year_start,
            year_end=year_end,
            diagnostics=diagnostics,
            cascade_bge_k=cascade_bge_k,
            step_top_k=step_top_k,
        )

        # 2. 条件触发 HippoRAG（多实体+关系词时融合图检索）
        fused_hits = vector_hits
        year_window_enabled = year_start is not None or year_end is not None
        if should_use_hipporag(query) and not year_window_enabled:
            self._ensure_graph()
            if self.hippo is not None:
                fused_hits = self.hippo.retrieve_with_graph(
                    query,
                    vector_hits,
                    top_k=top_k * 2,
                    graph_weight=graph_weight
                )

        # 3. Rerank（bge_only | colbert_only | cascade）
        if rerank and fused_hits:
            hits_with_content = [h for h in fused_hits if h.get("content")]
            if hits_with_content:
                # 请求未传 step_top_k 时继承 config rerank_output_k
                default_output_k = getattr(settings.search, "rerank_output_k", 10)
                rerank_limit = step_top_k if step_top_k is not None else default_output_k
                fused_hits = _rerank_candidates(
                    query, hits_with_content, top_k=min(rerank_limit, len(hits_with_content)),
                    reranker_mode=reranker_mode, cascade_bge_k=cascade_bge_k,
                )
                per_doc_cap = getattr(settings.search, "per_doc_cap", 3)
                fused_hits = dedup_and_diversify(fused_hits, per_doc_cap=per_doc_cap)

        return fused_hits[:step_top_k] if step_top_k is not None else fused_hits[:top_k]

    def retrieve(
        self,
        query: str,
        collection: str = None,
        config: RetrievalConfig = None,
        diagnostics: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        统一检索入口

        Args:
            query: 查询文本
            collection: Milvus collection 名（默认 global）
            config: 检索配置
            diagnostics: 可选字典，传入时填充检索诊断信息
        """
        if config is None:
            config = RetrievalConfig()

        if collection is None:
            collection = settings.collection.global_

        if config.mode == "vector":
            return self.retrieve_vector(
                query,
                collection,
                config.top_k,
                config.rerank,
                config.year_start,
                config.year_end,
                diagnostics=diagnostics,
                reranker_mode=config.reranker_mode,
                cascade_bge_k=config.colbert_top_k,
                step_top_k=config.step_top_k,
            )
        elif config.mode == "graph":
            return self.retrieve_graph(query, config.top_k)
        else:  # hybrid
            return self.retrieve_hybrid(
                query,
                collection,
                config.top_k,
                config.rerank,
                config.graph_weight,
                config.year_start,
                config.year_end,
                diagnostics=diagnostics,
                reranker_mode=config.reranker_mode,
                cascade_bge_k=config.colbert_top_k,
                step_top_k=config.step_top_k,
            )


# 全局实例
retriever = HybridRetriever()
