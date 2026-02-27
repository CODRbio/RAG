"""
ColBERT 应用层精排器（MaxSim）
不存储向量，检索时对候选文档实时编码并计算 MaxSim，与 BGE-Reranker 接口对齐。
"""

import sys
import types
import warnings
from typing import List, Optional
from dataclasses import dataclass

from config.settings import settings
from src.log import get_logger

logger = get_logger(__name__)

# ── langchain 兼容垫片 ──
# ragatouille < 0.0.10 引用 langchain.retrievers.document_compressors.base，
# 该路径在 langchain >= 1.0 已迁移至 langchain_core.documents.compressor。
# 在 import ragatouille 之前把旧路径映射到新位置。
if "langchain.retrievers" not in sys.modules:
    try:
        from langchain_core.documents.compressor import BaseDocumentCompressor

        _fake_retrievers = types.ModuleType("langchain.retrievers")
        _fake_compressors = types.ModuleType("langchain.retrievers.document_compressors")
        _fake_base = types.ModuleType("langchain.retrievers.document_compressors.base")
        _fake_base.BaseDocumentCompressor = BaseDocumentCompressor  # type: ignore[attr-defined]
        sys.modules["langchain.retrievers"] = _fake_retrievers
        sys.modules["langchain.retrievers.document_compressors"] = _fake_compressors
        sys.modules["langchain.retrievers.document_compressors.base"] = _fake_base
        logger.debug("[ColBERT] langchain compat shim installed (langchain_core.documents.compressor)")
    except ImportError:
        logger.debug("[ColBERT] langchain_core 不可用，跳过兼容垫片")

# ── 抑制 ColBERT/ragatouille 依赖触发的已知提示（tokenizer 用法、GradScaler、RAGatouille 迁移公告等）──
def _suppress_colbert_warnings():
    warnings.filterwarnings("ignore", message=".*regex pattern.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*GradScaler.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*faster.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Please note that with a fast tokenizer.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*RAGatouille.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Future Release Notice.*", category=UserWarning)


# ── 在模块加载时做一次 import 探测，失败时记录原因并永久禁用 ──
_RAGATOUILLE_AVAILABLE: bool = False
_RAGATOUILLE_UNAVAIL_REASON: str = ""

try:
    _suppress_colbert_warnings()
    import ragatouille  # noqa: F401
    _RAGATOUILLE_AVAILABLE = True
    logger.info("[ColBERT] ragatouille 可用")
except Exception as _e:
    _RAGATOUILLE_UNAVAIL_REASON = str(_e)
    logger.warning(
        "[ColBERT] ragatouille 不可用，ColBERT 精排已禁用（cascade → BGE-only）。"
        " 原因: %s",
        _RAGATOUILLE_UNAVAIL_REASON,
    )


@dataclass
class RerankResult:
    """与 BGE-Reranker 一致的单个结果：index 为在原文 docs 列表中的下标，score 为分数"""
    index: int
    score: float


class ColBERTReranker:
    """ColBERT 精排器（懒加载，单例）。

    ``available`` 属性反映运行时可用性；调用方在使用前应先检查该属性。
    """

    _instance: Optional["ColBERTReranker"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def available(self) -> bool:
        """ragatouille 可以正常 import 且模型未永久失败。"""
        return _RAGATOUILLE_AVAILABLE and self._model is not False

    @property
    def model(self):
        """懒加载 ColBERT 模型（RAGatouille）。

        首次加载失败时将 _model 设为 False（哨兵值）以避免重复尝试。
        """
        if self._model is None:
            if not _RAGATOUILLE_AVAILABLE:
                logger.debug("[ColBERT] model 访问被拒绝：ragatouille 在模块加载时已失败")
                self._model = False
                return False
            model_name = getattr(settings.search, "colbert_model", "colbert-ir/colbertv2.0")
            n_gpu = -1
            if settings.model.device and settings.model.device != "cpu":
                n_gpu = 1
            logger.info("加载 ColBERT... (%s, device=%s)", model_name, settings.model.device)
            try:
                _suppress_colbert_warnings()  # 全局生效，加载与后续 rerank 均不再刷上述提示
                with warnings.catch_warnings():
                    _suppress_colbert_warnings()
                    from ragatouille import RAGPretrainedModel
                    load_kwargs = {"n_gpu": n_gpu, "verbose": 0}
                    if "jina" in model_name.lower() or "colbert-v2" in model_name:
                        load_kwargs["tokenizer_kwargs"] = {"fix_mistral_regex": True}
                    try:
                        self._model = RAGPretrainedModel.from_pretrained(model_name, **load_kwargs)
                    except TypeError:
                        load_kwargs.pop("tokenizer_kwargs", None)
                        self._model = RAGPretrainedModel.from_pretrained(model_name, **load_kwargs)
                logger.info("[OK] ColBERT 就绪")
            except Exception as e:
                logger.error(
                    "[ColBERT] 模型加载失败，本次运行内永久禁用: %s",
                    e, exc_info=True,
                )
                self._model = False  # 哨兵：避免重复尝试
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 10,
        bsize: int = 64,
    ) -> List[RerankResult]:
        """ColBERT MaxSim 精排。

        Returns 空列表（而非抛异常）当模型不可用时，让调用方决定如何降级。
        """
        if not docs:
            return []
        model = self.model
        if model is None:
            logger.warning("[ColBERT] rerank 被调用但模型不可用，返回空结果")
            return []
        top_k = min(top_k, len(docs))
        try:
            raw = model.rerank(
                query=query,
                documents=docs,
                k=top_k,
                zero_index_ranks=False,
                bsize=bsize,
            )
        except Exception as e:
            logger.error("[ColBERT] rerank 运行时错误: %s", e, exc_info=True)
            return []

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
