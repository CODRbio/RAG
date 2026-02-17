#!/usr/bin/env python
"""步骤7: Tavily 网络搜索测试（与 RAG 检索格式兼容）"""

import sys
import argparse

sys.path.insert(0, ".")

from src.log import get_logger
from src.retrieval.web_search import web_searcher

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Tavily web search")
    parser.add_argument("query", nargs="?", default="deep sea hydrothermal vent microbes", help="Search query")
    parser.add_argument("--expand", action="store_true", help="Use LLM query expansion")
    parser.add_argument("--async", dest="async_", action="store_true", help="Run async_search")
    args = parser.parse_args()

    if not web_searcher.enabled:
        logger.warning("Web search is disabled or api_key not set.")
        logger.warning("Set config/rag_config.json -> web_search.api_key (or env RAG_LLM__* for LLM).")
        return 0

    query = args.query
    logger.info("Query: %s, Expand: %s", query, args.expand)
    if args.async_:
        import asyncio
        results = asyncio.run(web_searcher.async_search(query, use_query_expansion=args.expand))
    else:
        results = web_searcher.search(query, use_query_expansion=args.expand)

    logger.info("Results: %s", len(results))
    for i, hit in enumerate(results[:5], 1):
        meta = hit.get("metadata", {})
        print(f"  [{i}] score={hit.get('score', 0):.3f} | {meta.get('title', '')[:50]} | {meta.get('url', '')[:60]}")
        print(f"      content: {(hit.get('content') or '')[:120]}...")

    if results:
        from src.generation.context_packer import pack_qa_context
        ctx = pack_qa_context(results, top_n=3)
        print("\n--- pack_qa_context (top 3) ---")
        print(ctx[:800] + "..." if len(ctx) > 800 else ctx)  # 结果数据保留 print

    return 0


if __name__ == "__main__":
    sys.exit(main())
