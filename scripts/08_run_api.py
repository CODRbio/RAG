#!/usr/bin/env python3
"""
启动多轮对话 API 服务

用法（需在 conda 环境 deepsea-rag 下）:
  conda run -n deepsea-rag python scripts/08_run_api.py
  conda run -n deepsea-rag python scripts/08_run_api.py --port 8001 --host 0.0.0.0

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
    args = parser.parse_args()

    import uvicorn
    from src.api.server import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
