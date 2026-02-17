#!/usr/bin/env python3
"""
离线模型加载测试：
1) 设置 HF_LOCAL_FILES_ONLY=true
2) 尝试加载 BGE-M3 / BGE-Reranker / ColBERT
前置：模型已下载到本地缓存
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    os.environ["HF_LOCAL_FILES_ONLY"] = os.getenv("HF_LOCAL_FILES_ONLY", "true")

    print("[offline] HF_LOCAL_FILES_ONLY =", os.environ["HF_LOCAL_FILES_ONLY"])

    from src.indexing.embedder import embedder
    from src.retrieval.colbert_reranker import colbert_reranker

    # BGE-M3
    try:
        _ = embedder.ef
        print("[OK] BGE-M3 loaded")
    except Exception as e:
        print("[FAIL] BGE-M3 load failed:", e)

    # BGE-Reranker
    try:
        _ = embedder.reranker
        print("[OK] BGE-Reranker loaded")
    except Exception as e:
        print("[FAIL] BGE-Reranker load failed:", e)

    # ColBERT
    try:
        _ = colbert_reranker.model
        print("[OK] ColBERT loaded")
    except Exception as e:
        print("[FAIL] ColBERT load failed:", e)


if __name__ == "__main__":
    main()
