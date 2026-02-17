"""
ColBERT 应用层精排器（MaxSim）
不存储向量，检索时对候选文档实时编码并计算 MaxSim，与 BGE-Reranker 接口对齐。
"""

from typing import List, Any
from dataclasses import dataclass

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """与 BGE-Reranker 一致的单个结果：index 为在原文 docs 列表中的下标，score 为分数"""
    index: int
    score: float


class ColBERTReranker:
    """ColBERT 精排器（懒加载，单例）"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self):
        """懒加载 ColBERT 模型（RAGatouille）"""
        if self._model is None:
            model_name = getattr(settings.search, "colbert_model", "colbert-ir/colbertv2.0")
            n_gpu = -1
            if settings.model.device and settings.model.device != "cpu":
                n_gpu = 1
            try:
                from ragatouille import RAGPretrainedModel
                logger.info(f"加载 ColBERT... ({model_name}, device={settings.model.device})")
                kwargs = {}
                if settings.model.colbert_cache_dir:
                    kwargs["cache_dir"] = settings.model.colbert_cache_dir
                if settings.model.local_files_only:
                    kwargs["local_files_only"] = True
                self._model = RAGPretrainedModel.from_pretrained(
                    model_name,
                    n_gpu=n_gpu,
                    verbose=0,
                    **kwargs,
                )
                logger.info("[OK] ColBERT 就绪")
            except Exception as e:
                raise RuntimeError(f"ColBERT 加载失败: {e}") from e
        return self._model

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 10,
        bsize: int = 64,
    ) -> List[RerankResult]:
        """
        对 docs 做 ColBERT MaxSim 精排，返回与 embedder.rerank 相同语义的结果：
        列表元素为 RerankResult(index=在 docs 中的下标, score=分数)，按 score 降序。
        """
        if not docs:
            return []
        top_k = min(top_k, len(docs))
        raw = self.model.rerank(
            query=query,
            documents=docs,
            k=top_k,
            zero_index_ranks=False,
            bsize=bsize,
        )
        # 单 query 时 ragatouille 返回 list[dict]，每个 dict 含 content, score, rank
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        out: List[RerankResult] = []
        doc_to_idx = {d: i for i, d in enumerate(docs)}
        for item in raw:
            content = item.get("content", "")
            score = float(item.get("score", 0.0))
            idx = doc_to_idx.get(content, -1)
            if idx >= 0:
                out.append(RerankResult(index=idx, score=score))
        return out


# 全局实例
colbert_reranker = ColBERTReranker()
