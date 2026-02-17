#!/usr/bin/env python3
"""
导出联调：创建画布 -> 写入大纲与草稿 -> 导出 Markdown。

支持命令行参数：
  --cite-format: 引用键格式 (numeric | hash | author_date)

前置：先启动 API（python scripts/08_run_api.py）
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="导出测试：Canvas -> Markdown")
    parser.add_argument(
        "--cite-format",
        choices=["numeric", "hash", "author_date"],
        default=None,
        help="引用键格式（默认从配置读取）",
    )
    args = parser.parse_args()

    import requests
    from config.settings import settings

    base = f"http://{settings.api.host}:{settings.api.port}"

    # 1) 创建 Canvas
    r1 = requests.post(
        f"{base}/canvas",
        json={"session_id": "", "topic": "深海冷泉综述"},
        timeout=10,
    )
    r1.raise_for_status()
    canvas_id = r1.json()["id"]
    print(f"[1] Canvas created: {canvas_id}")

    # 2) 写入大纲
    requests.post(
        f"{base}/canvas/{canvas_id}/outline",
        json={
            "sections": [
                {"id": "s1", "title": "引言", "level": 1, "order": 0},
                {"id": "s2", "title": "研究进展", "level": 1, "order": 1},
                {"id": "s2-1", "title": "生态系统", "level": 2, "order": 2, "parent_id": "s2"},
            ]
        },
        timeout=10,
    ).raise_for_status()
    print("[2] Outline upserted")

    # 3) 写入草稿
    requests.post(
        f"{base}/canvas/{canvas_id}/drafts",
        json={
            "block": {
                "section_id": "s1",
                "content_md": "这是引言部分的草稿内容。深海冷泉是一种重要的海底地质现象 [Smith2023]。",
                "version": 1,
                "used_fragment_ids": [],
                "used_citation_ids": [],
            }
        },
        timeout=10,
    ).raise_for_status()
    print("[3] Draft upserted")

    # 4) 添加测试引用
    requests.post(
        f"{base}/canvas/{canvas_id}/citations",
        json={
            "citations": [
                {
                    "cite_key": "Smith2023",
                    "title": "Deep-sea cold seep ecosystems",
                    "authors": ["Smith, John", "Wang, Lei"],
                    "year": 2023,
                    "doi": "10.1234/example1",
                },
                {
                    "cite_key": "Jones2022",
                    "title": "Methane seepage in the South China Sea",
                    "authors": ["Jones, Mary", "Chen, Wei"],
                    "year": 2022,
                    "url": "https://example.com/paper2",
                },
                {
                    "cite_key": "Smith2023b",
                    "title": "Microbial communities at cold seeps",
                    "authors": ["Smith, John"],
                    "year": 2023,
                    "doi": "10.1234/example3",
                },
            ]
        },
        timeout=10,
    ).raise_for_status()
    print("[4] Citations added")

    # 5) 导出 Markdown（可指定引用格式）
    export_params = {"canvas_id": canvas_id, "format": "markdown"}
    if args.cite_format:
        export_params["cite_key_format"] = args.cite_format
        print(f"[5] Exporting with cite_key_format: {args.cite_format}")
    else:
        print("[5] Exporting with default cite_key_format from config")

    r5 = requests.post(
        f"{base}/export",
        json=export_params,
        timeout=10,
    )
    r5.raise_for_status()
    md = r5.json().get("content", "")
    print("\n" + "=" * 60)
    print("EXPORTED MARKDOWN:")
    print("=" * 60)
    print(md[:800] + "..." if len(md) > 800 else md)
    print("=" * 60)

    print("\nOK: export markdown exercised.")
    print(f"\nTip: 使用 --cite-format 参数切换引用格式：")
    print(f"  python {Path(__file__).name} --cite-format numeric")
    print(f"  python {Path(__file__).name} --cite-format hash")
    print(f"  python {Path(__file__).name} --cite-format author_date")


if __name__ == "__main__":
    main()
