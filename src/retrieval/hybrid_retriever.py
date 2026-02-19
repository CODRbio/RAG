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
from src.indexing.milvus_ops import milvus
from src.indexing.embedder import embedder
from src.graph.hippo_rag import HippoRAG, get_hippo_rag
from src.graph.entity_extractor import ExtractorConfig
from src.log import get_logger
from src.utils.cache import TTLCache, _make_key, get_cache

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


def _rerank_candidates(
    query: str,
    candidates: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    按 settings.search.reranker_mode 做精排，返回带 score 的 hit 列表。
    支持 bge_only | colbert_only | cascade。需 use_colbert_reranker=true 时 ColBERT 才生效。
    """
    if not candidates:
        return []
    docs = [c.get("content") or "" for c in candidates]
    use_colbert = getattr(settings.search, "use_colbert_reranker", False)
    mode = getattr(settings.search, "reranker_mode", "bge_only") if use_colbert else "bge_only"
    colbert_top_k = getattr(settings.search, "colbert_top_k", 30)

    if mode == "colbert_only" and colbert_reranker is not None:
        try:
            reranked = colbert_reranker.rerank(query, docs, top_k=min(top_k * 2, len(docs)))
            return [
                {**candidates[r.index], "score": r.score}
                for r in reranked
            ][:top_k]
        except Exception:
            # ColBERT 不可用/加载失败时，回退到 BGE
            pass

    if mode == "cascade" and colbert_reranker is not None:
        bge_k = min(colbert_top_k, len(docs))
        bge_out = embedder.rerank(query, docs, top_k=bge_k)
        bge_candidates = [candidates[r.index] for r in bge_out]
        bge_docs = [c.get("content") or "" for c in bge_candidates]
        try:
            colbert_out = colbert_reranker.rerank(query, bge_docs, top_k=min(top_k * 2, len(bge_docs)))
            return [
                {**bge_candidates[r.index], "score": r.score}
                for r in colbert_out
            ][:top_k]
        except Exception:
            # ColBERT 不可用/加载失败时，回退到 BGE
            pass

    # bge_only 或 ColBERT 不可用时
    reranked = embedder.rerank(query, docs, top_k=min(top_k * 2, len(docs)))
    return [
        {**candidates[r.index], "score": r.score}
        for r in reranked
    ][:top_k]


@dataclass
class RetrievalConfig:
    """检索配置"""
    mode: str = "hybrid"  # vector, graph, hybrid
    top_k: int = 10
    rerank: bool = True
    graph_weight: float = 0.3  # hybrid 模式下图检索权重
    year_start: Optional[int] = None  # 年份窗口起始（硬过滤）
    year_end: Optional[int] = None  # 年份窗口结束（硬过滤）


class HybridRetriever:
    """
    混合检索器

    整合向量检索和 HippoRAG 图检索
    """

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
        rerank_output_k = getattr(settings.search, "rerank_output_k", 10)
        w_dense = getattr(settings.search, "rrf_dense_weight", 0.6)
        w_sparse = getattr(settings.search, "rrf_sparse_weight", 0.4)
        per_doc_cap = getattr(settings.search, "per_doc_cap", 3)

        cache_key = _make_key("retrieval_vector", query, collection, top_k, rerank, year_start, year_end)
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
        dense_req = AnnSearchRequest(
            data=[dense_vec.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=dense_recall_k,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=sparse_recall_k,
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
                year_start=year_start,
                year_end=year_end,
            )

        def _do_sparse():
            return milvus.hybrid_search(
                collection=collection,
                reqs=[sparse_req],
                ranker=RRFRanker(k=settings.search.rrf_k),
                limit=sparse_recall_k,
                output_fields=output_fields,
                year_start=year_start,
                year_end=year_end,
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
        t_rerank = time.perf_counter()
        reranked = False
        if rerank and candidates:
            docs = [c["content"] for c in candidates if c.get("content")]
            if docs:
                hits = _rerank_candidates(query, candidates, top_k=min(rerank_output_k * 2, len(candidates)))
                hits = dedup_and_diversify(hits, per_doc_cap=per_doc_cap)
                out = hits[:top_k]
                reranked = True
        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        if not reranked:
            out = candidates[:top_k]

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
                rerank_output_k = getattr(settings.search, "rerank_output_k", 10)
                fused_hits = _rerank_candidates(query, hits_with_content, top_k=min(rerank_output_k * 2, len(hits_with_content)))
                per_doc_cap = getattr(settings.search, "per_doc_cap", 3)
                fused_hits = dedup_and_diversify(fused_hits, per_doc_cap=per_doc_cap)

        return fused_hits[:top_k]

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
            )


# 全局实例
retriever = HybridRetriever()
