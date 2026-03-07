import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.serpapi_search import SerpAPISearcher

async def main():
    searcher = SerpAPISearcher()
    print(f"Enabled: {searcher.enabled}")
    
    q = "biological irrigation cold seep deep sea"
    print(f"Searching scholar for: {q}")
    
    results = await searcher.search_scholar(q, limit=40, year_start=2024, year_end=2024)
    print(f"Got {len(results)} results")
    
    for i, r in enumerate(results[:3]):
        print(f"{i+1}. {r.get('metadata', {}).get('title')} - {r.get('metadata', {}).get('year')}")

if __name__ == "__main__":
    asyncio.run(main())
