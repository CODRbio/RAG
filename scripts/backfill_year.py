"""
Backfill 'year' dynamic field for an existing collection.

Reads full rows (including vectors), adds 'year' extracted from paper_id, then upserts back.

Usage:
    conda run -n deepsea-rag python scripts/backfill_year.py [--collection DeepSea_symbiosis] [--batch 200] [--dry-run]
"""

import re
import argparse
from pymilvus import connections, Collection, MilvusClient

YEAR_RE = re.compile(r"(?:^|_)(\d{4})(?:_|$)")

ALL_FIELDS = [
    "chunk_id", "content", "raw_content", "dense_vector", "sparse_vector",
    "paper_id", "domain", "content_type", "chunk_type", "section_path", "page",
]


def extract_year(paper_id: str) -> int:
    m = YEAR_RE.search(paper_id or "")
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y
    return 0


def sparse_to_dict(sparse_obj) -> dict:
    """Convert pymilvus sparse vector object to {index: value} dict for upsert."""
    if isinstance(sparse_obj, dict):
        return sparse_obj
    if hasattr(sparse_obj, "items"):
        return dict(sparse_obj.items())
    if hasattr(sparse_obj, "to_dict"):
        return sparse_obj.to_dict()
    return dict(sparse_obj) if sparse_obj else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="DeepSea_symbiosis")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--uri", default="http://localhost:19530")
    args = parser.parse_args()

    connections.connect(uri=args.uri)
    col = Collection(args.collection)
    col.load()

    client = MilvusClient(uri=args.uri)
    stats = client.get_collection_stats(args.collection)
    total = stats.get("row_count", 0)
    print(f"Collection: {args.collection}, rows: {total}")

    it = col.query_iterator(
        batch_size=args.batch,
        output_fields=ALL_FIELDS,
    )

    updated = 0
    no_year = 0

    while True:
        batch = it.next()
        if not batch:
            break

        upsert_data = []
        for r in batch:
            row = {}
            for f in ALL_FIELDS:
                val = r.get(f)
                if f == "sparse_vector":
                    val = sparse_to_dict(val)
                if f == "dense_vector" and hasattr(val, "tolist"):
                    val = val.tolist()
                row[f] = val
            year = extract_year(r.get("paper_id", ""))
            if year == 0:
                no_year += 1
            row["year"] = year
            upsert_data.append(row)

        if not args.dry_run:
            client.upsert(collection_name=args.collection, data=upsert_data)

        updated += len(upsert_data)
        pct = updated * 100 // max(total, 1)
        print(f"  {updated}/{total} ({pct}%)  no_year={no_year}")

    it.close()
    print(f"\nDone. updated={updated}, no_year={no_year} ({no_year*100//max(updated,1)}%)")
    if args.dry_run:
        print("(dry-run mode, no data was written)")


if __name__ == "__main__":
    main()
