"""
HybridRetriever 单元测试：RRF 融合 / should_use_hipporag / HybridRetriever 构造
"""

import pytest
from unittest.mock import MagicMock, patch


# ── weighted_rrf (模块级纯函数) ──

class TestWeightedRRF:
    def test_basic_fusion(self):
        from src.retrieval.hybrid_retriever import weighted_rrf

        dense_hits = [
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.7},
        ]
        sparse_hits = [
            {"chunk_id": "b", "score": 0.95},
            {"chunk_id": "c", "score": 0.6},
        ]
        fused = weighted_rrf(dense_hits, sparse_hits, w_dense=0.7, w_sparse=0.3, k=60)
        ids = [cid for cid, _ in fused]
        # b 出现在两个列表中，分数应最高
        assert ids[0] == "b"
        assert len(fused) == 3

    def test_empty_inputs(self):
        from src.retrieval.hybrid_retriever import weighted_rrf
        fused = weighted_rrf([], [], w_dense=0.7, w_sparse=0.3, k=60)
        assert fused == []

    def test_single_list(self):
        from src.retrieval.hybrid_retriever import weighted_rrf
        hits = [{"chunk_id": "x", "score": 1.0}]
        fused = weighted_rrf(hits, [], w_dense=1.0, w_sparse=0.0, k=60)
        assert len(fused) == 1
        assert fused[0][0] == "x"

    def test_returns_sorted_descending(self):
        from src.retrieval.hybrid_retriever import weighted_rrf
        dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
        sparse = [{"chunk_id": "c"}, {"chunk_id": "a"}]
        fused = weighted_rrf(dense, sparse)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)


# ── should_use_hipporag (模块级纯函数) ──

class TestShouldUseHippoRAG:
    def test_multi_entity_with_relation(self):
        from src.retrieval.hybrid_retriever import should_use_hipporag
        # 英文大写词算实体，"关系"是关系关键词 → True
        result = should_use_hipporag("Tube Worm 和 Mussel 的关系是什么")
        assert result is True

    def test_no_relation_keyword(self):
        from src.retrieval.hybrid_retriever import should_use_hipporag
        # 无关系关键词 → False
        result = should_use_hipporag("深海热泉环境描述")
        assert result is False

    def test_empty_query(self):
        from src.retrieval.hybrid_retriever import should_use_hipporag
        assert should_use_hipporag("") is False

    def test_single_entity_with_relation(self):
        from src.retrieval.hybrid_retriever import should_use_hipporag
        # 只有一个实体 → False（需要多实体才触发）
        result = should_use_hipporag("管虫的关系")
        assert result is False


# ── _count_entities (模块级纯函数) ──

class TestCountEntities:
    def test_chinese_entities(self):
        from src.retrieval.hybrid_retriever import _count_entities
        # 连续中文字符算一个实体，用空格/非中文字符分隔才算多个
        count = _count_entities("管虫 贻贝 热泉")
        assert count >= 3

    def test_english_entities(self):
        from src.retrieval.hybrid_retriever import _count_entities
        count = _count_entities("Tube worms and Mussels")
        assert count >= 2

    def test_empty(self):
        from src.retrieval.hybrid_retriever import _count_entities
        assert _count_entities("") == 0


# ── RetrievalConfig 数据类 ──

class TestRetrievalConfig:
    def test_defaults(self):
        from src.retrieval.hybrid_retriever import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.mode == "hybrid"
        assert cfg.top_k == 10
        assert cfg.rerank is True
        assert cfg.graph_weight == 0.3

    def test_custom(self):
        from src.retrieval.hybrid_retriever import RetrievalConfig
        cfg = RetrievalConfig(mode="vector", top_k=20, rerank=False)
        assert cfg.mode == "vector"
        assert cfg.top_k == 20
        assert cfg.rerank is False
