"""
Embedding 服务封装
自动根据环境选择设备
"""

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)


class Embedder:
    """Embedding 服务（单例）"""

    _instance = None
    _ef = None
    _reranker = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def ef(self) -> BGEM3EmbeddingFunction:
        """BGE-M3 Embedding（懒加载）"""
        if self._ef is None:
            logger.info(f"加载 BGE-M3... (device={settings.model.device})")
            kwargs = {}
            if settings.model.embedding_cache_dir:
                kwargs["cache_dir"] = settings.model.embedding_cache_dir
            if settings.model.local_files_only:
                kwargs["local_files_only"] = True
            self._ef = BGEM3EmbeddingFunction(
                model_name=settings.model.embedding_model,
                device=settings.model.device,
                use_fp16=settings.model.use_fp16,
                **kwargs,
            )
            logger.info("[OK] BGE-M3 就绪")
        return self._ef

    @property
    def reranker(self) -> BGERerankFunction:
        """BGE-Reranker（懒加载）"""
        if self._reranker is None:
            logger.info(f"加载 BGE-Reranker... (device={settings.model.device})")
            kwargs = {}
            if settings.model.reranker_cache_dir:
                kwargs["cache_dir"] = settings.model.reranker_cache_dir
            if settings.model.local_files_only:
                kwargs["local_files_only"] = True
            self._reranker = BGERerankFunction(
                model_name=settings.model.reranker_model,
                device=settings.model.device,
                **kwargs,
            )
            logger.info("[OK] BGE-Reranker 就绪")
        return self._reranker

    def encode(self, texts: list) -> dict:
        """生成 Dense + Sparse 向量"""
        return self.ef(texts)

    def rerank(self, query: str, docs: list, top_k: int = None) -> list:
        """重排序"""
        top_k = top_k or settings.search.rerank_top_k
        return self.reranker(query, docs, top_k=top_k)


# 全局实例
embedder = Embedder()
