#!/usr/bin/env python3
"""
同步/检查本地模型缓存。
前置：可通过环境变量控制缓存路径与离线模式
- MODEL_CACHE_ROOT=~/Hug
- EMBEDDING_CACHE_DIR=...
- RERANKER_CACHE_DIR=...
- COLBERT_CACHE_DIR=...
- HF_LOCAL_FILES_ONLY=true
- HF_ENDPOINT=https://hf-mirror.com
- 或 RAG_HF_ENDPOINTS=https://hf-mirror.com,https://huggingface.co
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制更新模型缓存")
    parser.add_argument("--offline", action="store_true", help="仅检查本地缓存（不联网）")
    args = parser.parse_args()

    from src.utils.model_sync import check_models, sync_models

    if args.offline:
        items = check_models(local_files_only=True)
        print("[model_status]")
        for i in items:
            print(f"- {i.name}: exists={i.exists} cache_dir={i.cache_dir}")
        return

    items = sync_models(force_update=args.force, local_files_only=False)
    print("[model_sync]")
    for i in items:
        print(
            f"- {i.name}: status={i.status} updated={i.updated} message={i.message} cache_dir={i.cache_dir}"
        )


if __name__ == "__main__":
    main()
