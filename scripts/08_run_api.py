#!/usr/bin/env python3
"""
启动多轮对话 API 服务

用法（需在 conda 环境 deepsea-rag 下）:
  conda run -n deepsea-rag python scripts/08_run_api.py
  conda run -n deepsea-rag python scripts/08_run_api.py --port 8001 --host 0.0.0.0
  conda run -n deepsea-rag python scripts/08_run_api.py --workers 2   # 长请求不阻塞其它请求

或先激活环境再执行:
  conda activate deepsea-rag
  python scripts/08_run_api.py
"""

import argparse
import sys
from pathlib import Path

# 项目根目录加入 path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from config.settings import settings
    parser = argparse.ArgumentParser(description="Run DeepSea RAG Chat API")
    parser.add_argument("--host", default=settings.api.host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.api.port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable reload (dev)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers (default 1). Use 2+ so long requests (e.g. deep-research) don't block others. Ignored when --reload.",
    )
    args = parser.parse_args()

    import uvicorn
    from src.api.server import app

    kwargs = {"host": args.host, "port": args.port, "reload": args.reload}
    if not args.reload and args.workers > 1:
        kwargs["workers"] = args.workers
    uvicorn.run(app, **kwargs)


if __name__ == "__main__":
    main()
