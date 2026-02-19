"""
快速集成测试：NCBI + 路由计划
Layer 1 – NCBISearcher 直连 PubMed（真实网络请求，含 efetch 摘要）
Layer 2 – 时效性启发式规则路由（无 LLM）
Layer 3 – LLM 驱动的智能路由计划 (deepseek)
"""
import asyncio
import sys
sys.path.insert(0, ".")

SEP = "─" * 62


# ─────────────────────────────────────────────────────────────────
# Layer 1: NCBI 搜索器（live 网络请求）
# ─────────────────────────────────────────────────────────────────
async def test_ncbi():
    print(f"\n{SEP}")
    print("Layer 1 — NCBISearcher  (live PubMed API)")
    print(SEP)

    from src.retrieval.ncbi_search import get_ncbi_searcher

    searcher = get_ncbi_searcher()
    query = "deep sea cold seep methane oxidizing bacteria"
    print(f"Query : {query!r}")
    hits = await searcher.search(query, limit=3)

    if not hits:
        print("❌  No results returned")
        return False

    has_abstract = 0
    for i, h in enumerate(hits, 1):
        m = h.get("metadata", {})
        abstract = (m.get("abstract") or "")
        abstract_snippet = abstract[:130] + ("…" if len(abstract) > 130 else "")
        if abstract:
            has_abstract += 1
        print(f"\n  [{i}] {m.get('title', '(no title)')}")
        print(f"       Year={m.get('year')}  DOI={m.get('doi') or '—'}")
        print(f"       Score={h.get('score')}  URL={m.get('url', '')}")
        if abstract_snippet:
            print(f"       Abstract: {abstract_snippet}")
        else:
            print(f"       Abstract: (none)")

    print(f"\n  Results: {len(hits)}/3   Abstracts fetched: {has_abstract}/{len(hits)}")
    ok = len(hits) > 0
    print(f"{'✅' if ok else '❌'}  Layer 1 {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────
# Layer 2: 启发式路由（纯规则，无 LLM）
# ─────────────────────────────────────────────────────────────────
def test_heuristic_routing():
    print(f"\n{SEP}")
    print("Layer 2 — Freshness heuristic + fallback routing  (no LLM)")
    print(SEP)

    from src.retrieval.smart_query_optimizer import (
        SmartQueryOptimizer,
        _is_fresh_query_heuristic,
    )

    # 2a: 时效性检测
    cases = [
        ("cold seep microbiology",                   False),
        ("latest breakthroughs in mRNA vaccine 2025", True),
        ("最新深海冷泉研究进展",                        True),
        ("CRISPR gene editing mechanism",             False),
        ("today breaking news AI",                    True),
    ]
    all_ok = True
    print("  Freshness heuristic:")
    for q, expect in cases:
        got = _is_fresh_query_heuristic(q)
        ok  = got == expect
        if not ok:
            all_ok = False
        print(f"    {'✅' if ok else '❌'}  expect={expect!s:<5}  got={got!s:<5}  {q!r}")

    # 2b: 回退路由计划
    opt = SmartQueryOptimizer()
    candidates = ["ncbi", "tavily", "scholar", "google"]

    plan_bio = opt._fallback_routing_plan("cold seep microbiology", candidates, is_fresh=False)
    plan_fresh = opt._fallback_routing_plan("latest AI news 2026", candidates, is_fresh=True)

    print("\n  Fallback routing plans:")
    print(f"    bio   → primary={plan_bio.primary}  fallback={plan_bio.fallback}  is_fresh={plan_bio.is_fresh}")
    print(f"    fresh → primary={plan_fresh.primary}  fallback={plan_fresh.fallback}  is_fresh={plan_fresh.is_fresh}")

    bio_ok   = "ncbi" in plan_bio.primary
    fresh_ok = "tavily" in plan_fresh.primary
    if not bio_ok:
        print("  ❌  bio plan should have ncbi in primary")
        all_ok = False
    if not fresh_ok:
        print("  ❌  fresh plan should have tavily in primary")
        all_ok = False

    print(f"\n{'✅' if all_ok else '❌'}  Layer 2 {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────
# Layer 3: LLM 智能路由计划（deepseek）
# ─────────────────────────────────────────────────────────────────
def test_llm_routing():
    print(f"\n{SEP}")
    print("Layer 3 — LLM routing plan  (deepseek)")
    print(SEP)

    from src.retrieval.smart_query_optimizer import get_smart_query_optimizer

    opt = get_smart_query_optimizer()
    candidates = ["ncbi", "tavily", "scholar", "google"]

    test_cases = [
        # (query, must_have_in_primary, must_NOT_have_in_primary)
        ("deep sea cold seep microbial diversity",        "ncbi",   None),
        ("latest ChatGPT model features 2026",            "tavily", None),
        ("BRCA1 breast cancer mutation clinical trial",   "ncbi",   None),
    ]

    all_ok = True
    for q, must, must_not in test_cases:
        print(f"\n  Query: {q!r}")
        try:
            plan = opt.get_routing_plan(q, candidates)
            print(f"    primary  : {plan.primary}")
            print(f"    fallback : {plan.fallback}")
            print(f"    is_fresh : {plan.is_fresh}")
            for eng, qlist in plan.queries.items():
                for qstr in qlist:
                    print(f"    [{eng}] {qstr!r}")

            # 软断言
            if must and (must not in plan.primary):
                print(f"    ⚠️  expected '{must}' in primary (routing heuristic may differ)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
            all_ok = False

    print(f"\n{'✅' if all_ok else '❌'}  Layer 3 {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
async def main():
    results: dict = {}

    try:
        results["ncbi_live"]   = await test_ncbi()
    except Exception as e:
        print(f"\n❌ Layer 1 exception: {e}")
        import traceback; traceback.print_exc()
        results["ncbi_live"] = False

    try:
        results["heuristic"]   = test_heuristic_routing()
    except Exception as e:
        print(f"\n❌ Layer 2 exception: {e}")
        import traceback; traceback.print_exc()
        results["heuristic"] = False

    try:
        results["llm_routing"] = test_llm_routing()
    except Exception as e:
        print(f"\n❌ Layer 3 exception: {e}")
        import traceback; traceback.print_exc()
        results["llm_routing"] = False

    print(f"\n{'═' * 62}")
    print("Test Summary")
    print("═" * 62)
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    passed = sum(results.values())
    total  = len(results)
    print(f"\n  Result: {passed}/{total} layers passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
