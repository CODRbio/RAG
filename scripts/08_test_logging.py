#!/usr/bin/env python
"""测试统一日志管理：分级输出、按运行实例命名、清理策略。"""

import sys

sys.path.insert(0, ".")

from src.log import get_logger, init_logging, cleanup_logs


def main():
    logger = get_logger(__name__)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")

    report = cleanup_logs()
    print("Cleanup report:", report)


if __name__ == "__main__":
    main()
