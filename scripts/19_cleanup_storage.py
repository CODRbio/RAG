#!/usr/bin/env python3
"""
存储清理脚本：按生命周期和大小限制清理过期数据。

用法：
    python scripts/19_cleanup_storage.py                 # 使用配置默认值
    python scripts/19_cleanup_storage.py --max-age 7     # 清理 7 天前的数据
    python scripts/19_cleanup_storage.py --max-size 2    # 限制总大小 2GB
    python scripts/19_cleanup_storage.py --vacuum        # 清理后执行 VACUUM
    python scripts/19_cleanup_storage.py --stats         # 仅显示统计，不清理
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.utils.storage_cleaner import (
    run_cleanup,
    vacuum_databases,
    get_storage_stats,
)


def main():
    parser = argparse.ArgumentParser(description="持久化存储清理工具")
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help=f"数据保留天数（默认 {settings.storage.max_age_days}）",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        help=f"总大小上限 GB（默认 {settings.storage.max_size_gb}）",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="清理后执行 VACUUM 回收空间",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="仅显示统计信息，不执行清理",
    )
    args = parser.parse_args()

    # 显示当前状态
    stats_before = get_storage_stats()
    print(f"当前存储状态: {stats_before['total_size_mb']:.2f} MB")
    for db_name, size_mb in stats_before["databases"].items():
        print(f"  - {db_name}: {size_mb:.2f} MB")

    if args.stats:
        return

    # 执行清理
    max_age = args.max_age if args.max_age is not None else settings.storage.max_age_days
    max_size = args.max_size if args.max_size is not None else settings.storage.max_size_gb

    print(f"\n执行清理: max_age={max_age}天, max_size={max_size}GB")
    result = run_cleanup(
        max_age_days=max_age,
        max_size_gb=max_size,
        batch_size=settings.storage.cleanup_batch_size,
    )

    print(f"\n清理结果:")
    print(f"  - 按时间清理: canvas={result['age_cleanup']['canvas']}, "
          f"session={result['age_cleanup']['session']}, "
          f"project={result['age_cleanup']['project']}")
    print(f"  - 按大小清理: {result['size_cleanup']} 条记录")
    print(f"  - 当前大小: {result['final_size_mb']:.2f} MB")

    # VACUUM
    if args.vacuum:
        print("\n执行 VACUUM...")
        vacuum_databases()
        stats_after = get_storage_stats()
        print(f"VACUUM 后大小: {stats_after['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
