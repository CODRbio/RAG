#!/usr/bin/env python3
"""使用项目自带的 SerpAPISearcher 测试 SerpAPI（Google Scholar + Google Web）。

本脚本为命令行默认/辅助测试；正式使用 SerpAPI 及所有选项以前端 UI 为主。"""
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))


async def main() -> int:
    from config.settings import settings
    from src.retrieval.serpapi_search import SerpAPISearcher

    cfg = getattr(settings, "serpapi", None)
    enabled = bool(cfg and getattr(cfg, "enabled", False))
    has_key = bool(cfg and (getattr(cfg, "api_key", "") or "").strip())

    print("SerpAPI 配置:")
    print(f"  enabled: {enabled}")
    print(f"  api_key: {'已设置' if has_key else '未设置'}")
    if not has_key:
        print("\n请在 config/rag_config.json 的 serpapi 中设置 api_key 并设置 enabled: true 后再运行。")
        return 1
    if not enabled:
        print("\n请在 config/rag_config.json 的 serpapi 中设置 enabled: true。")
        return 1

    searcher = SerpAPISearcher()
    if not searcher.enabled:
        print("SerpAPISearcher.enabled 为 False，请检查配置。")
        return 1

    try:
        # 1) Google Scholar
        query_scholar = "machine learning survey"
        print(f"\n--- Google Scholar 测试: {query_scholar!r} ---")
        scholar_results = await searcher.search_scholar(query_scholar, limit=3)
        print(f"  结果数: {len(scholar_results)}")
        for i, r in enumerate(scholar_results[:3], 1):
            meta = r.get("metadata") or {}
            print(f"  [{i}] {meta.get('title', '')[:60]}...")
            print(f"      url: {meta.get('url', '')[:70]}...")
        if not scholar_results:
            print("  (无结果，可能 API 额度或网络问题)")

        # 2) Google Web
        query_google = "Python asyncio tutorial"
        print(f"\n--- Google Web 测试: {query_google!r} ---")
        google_results = await searcher.search_google(query_google, limit=3)
        print(f"  结果数: {len(google_results)}")
        for i, r in enumerate(google_results[:3], 1):
            meta = r.get("metadata") or {}
            print(f"  [{i}] {meta.get('title', '')[:60]}...")
            print(f"      url: {meta.get('url', '')[:70]}...")
        if not google_results:
            print("  (无结果，可能 API 额度或网络问题)")

        await searcher.close()
        print("\nSerpAPI 测试完成。")
        return 0
    except Exception as e:
        print(f"\nSerpAPI 测试失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        try:
            await searcher.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
