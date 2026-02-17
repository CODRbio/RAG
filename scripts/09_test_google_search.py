#!/usr/bin/env python3
"""
Google Scholar / Google 搜索测试脚本

测试 GoogleSearcher 和 UnifiedWebSearcher 功能。

使用方法:
    python scripts/09_test_google_search.py "deep learning"
    python scripts/09_test_google_search.py "machine learning" --provider scholar
    python scripts/09_test_google_search.py "neural networks" --provider unified --limit 10
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def test_scholar_search(query: str, limit: int = 5):
    """测试 Google Scholar 搜索"""
    from src.retrieval.google_search import google_searcher
    
    print(f"\n{'='*60}")
    print(f"测试 Google Scholar 搜索")
    print(f"查询: {query}")
    print(f"限制: {limit} 条")
    print(f"{'='*60}\n")
    
    if not google_searcher.enabled or not google_searcher.scholar_enabled:
        print("⚠️  Google Scholar 搜索未启用，请检查配置")
        return []
    
    results = await google_searcher.search_scholar(query, limit=limit)
    
    print(f"✅ 返回 {len(results)} 条结果\n")
    
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"[{i}] {meta.get('title', '无标题')}")
        print(f"    来源: {meta.get('source', 'unknown')}")
        print(f"    URL: {meta.get('url', '无')}")
        if meta.get('year'):
            print(f"    年份: {meta.get('year')}")
        if meta.get('cited_by'):
            print(f"    引用: {meta.get('cited_by')}")
        print(f"    摘要: {r.get('content', '')[:100]}...")
        print()
    
    return results


async def test_google_search(query: str, limit: int = 5):
    """测试 Google 搜索"""
    from src.retrieval.google_search import google_searcher
    
    print(f"\n{'='*60}")
    print(f"测试 Google 搜索")
    print(f"查询: {query}")
    print(f"限制: {limit} 条")
    print(f"{'='*60}\n")
    
    if not google_searcher.enabled or not google_searcher.google_enabled:
        print("⚠️  Google 搜索未启用，请检查配置")
        return []
    
    results = await google_searcher.search_google(query, limit=limit)
    
    print(f"✅ 返回 {len(results)} 条结果\n")
    
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"[{i}] {meta.get('title', '无标题')}")
        print(f"    来源: {meta.get('source', 'unknown')}")
        print(f"    URL: {meta.get('url', '无')}")
        print(f"    摘要: {r.get('content', '')[:100]}...")
        print()
    
    return results


async def test_unified_search(query: str, limit: int = 5, providers: list = None):
    """测试统一搜索"""
    from src.retrieval.unified_web_search import unified_web_searcher
    
    print(f"\n{'='*60}")
    print(f"测试统一网络搜索")
    print(f"查询: {query}")
    print(f"来源: {providers or '自动检测'}")
    print(f"每来源限制: {limit} 条")
    print(f"{'='*60}\n")
    
    results = await unified_web_searcher.search(
        query, 
        providers=providers,
        max_results_per_provider=limit
    )
    
    print(f"✅ 合并去重后返回 {len(results)} 条结果\n")
    
    # 按来源分组统计
    source_counts = {}
    for r in results:
        source = r.get("metadata", {}).get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("来源分布:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  - {source}: {count} 条")
    print()
    
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"[{i}] [{meta.get('source', '?')}] {meta.get('title', '无标题')}")
        print(f"    URL: {meta.get('url', '无')}")
        print(f"    摘要: {r.get('content', '')[:80]}...")
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="测试 Google Scholar / Google 搜索")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--provider", choices=["scholar", "google", "unified"], 
                       default="scholar", help="搜索提供者")
    parser.add_argument("--limit", type=int, default=5, help="最大结果数")
    parser.add_argument("--output", "-o", help="输出 JSON 文件")
    
    args = parser.parse_args()
    
    if args.provider == "scholar":
        results = asyncio.run(test_scholar_search(args.query, args.limit))
    elif args.provider == "google":
        results = asyncio.run(test_google_search(args.query, args.limit))
    else:  # unified
        results = asyncio.run(test_unified_search(args.query, args.limit))
    
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
