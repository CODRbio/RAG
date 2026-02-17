#!/usr/bin/env bash
# ============================================================
# 一键启动 DeepSea RAG 全栈（后端 API + 前端 Dev Server）
#
# 用法:
#   bash scripts/start.sh              # 默认端口（后端 9999，前端 5173）
#   bash scripts/start.sh --backend-only   # 仅启动后端
#   bash scripts/start.sh --frontend-only  # 仅启动前端
#   API_PORT=8000 bash scripts/start.sh    # 自定义后端端口
#
# 退出: Ctrl+C 会同时关闭后端和前端进程
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── 运行时命令解析（优先 conda 环境） ──
# 目标：迁移时尽量复用 conda 管理，不依赖系统级 python/npm 绝对路径
CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepsea-rag}"
PY_CMD=(python)
NPM_CMD=(npm)

if command -v conda >/dev/null 2>&1; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
    if conda run -n "$CONDA_ENV_NAME" python --version >/dev/null 2>&1; then
      PY_CMD=(conda run -n "$CONDA_ENV_NAME" python)
    fi
    if conda run -n "$CONDA_ENV_NAME" npm -v >/dev/null 2>&1; then
      NPM_CMD=(conda run -n "$CONDA_ENV_NAME" npm)
    fi
  fi
fi

# ── HuggingFace 下载源（中国网络默认镜像 + 官方回退） ──
# 可通过外部环境变量覆盖：
# - RAG_HF_ENDPOINTS="https://hf-mirror.com,https://huggingface.co"
# - HF_ENDPOINT="https://hf-mirror.com"
if [ -z "${RAG_HF_ENDPOINTS:-}" ] && [ -z "${HF_ENDPOINT:-}" ] && [ -z "${HF_MIRROR:-}" ]; then
  export RAG_HF_ENDPOINTS="https://hf-mirror.com,https://huggingface.co"
fi

# ── 颜色 ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── 参数解析 ──
RUN_BACKEND=true
RUN_FRONTEND=true
for arg in "$@"; do
  case "$arg" in
    --backend-only)  RUN_FRONTEND=false ;;
    --frontend-only) RUN_BACKEND=false ;;
  esac
done

# ── 清理函数 ──
PIDS=()
cleanup() {
  echo ""
  echo -e "${YELLOW}[start.sh] 正在关闭服务...${NC}"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
  echo -e "${GREEN}[start.sh] 已退出${NC}"
}
trap cleanup EXIT INT TERM

# ── 启动后端 ──
if [ "$RUN_BACKEND" = true ]; then
  BACKEND_PORT="${API_PORT:-9999}"
  BACKEND_HOST="${API_HOST:-127.0.0.1}"
  echo -e "${CYAN}[start.sh] 启动后端 API → http://${BACKEND_HOST}:${BACKEND_PORT}${NC}"
  echo -e "${CYAN}[start.sh] HF endpoint candidates: ${RAG_HF_ENDPOINTS:-${HF_ENDPOINT:-${HF_MIRROR:-https://huggingface.co}}}${NC}"
  "${PY_CMD[@]}" -m uvicorn src.api.server:app \
    --host "$BACKEND_HOST" \
    --port "$BACKEND_PORT" \
    --reload &
  PIDS+=($!)
  # 等待后端就绪
  echo -e "${CYAN}[start.sh] 等待后端就绪...${NC}"
  for i in $(seq 1 30); do
    if curl -s "http://${BACKEND_HOST}:${BACKEND_PORT}/health" > /dev/null 2>&1; then
      echo -e "${GREEN}[start.sh] 后端已就绪 ✓${NC}"
      break
    fi
    sleep 1
  done
fi

# ── 启动前端 ──
if [ "$RUN_FRONTEND" = true ]; then
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo -e "${CYAN}[start.sh] 首次运行，安装前端依赖...${NC}"
    (cd "$ROOT/frontend" && "${NPM_CMD[@]}" install)
  fi
  echo -e "${CYAN}[start.sh] 启动前端 Dev Server → http://localhost:5173${NC}"
  (cd "$ROOT/frontend" && "${NPM_CMD[@]}" run dev) &
  PIDS+=($!)
fi

# ── 打印摘要 ──
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
if [ "$RUN_BACKEND" = true ]; then
  echo -e "${GREEN}  后端 API:    http://${BACKEND_HOST:-127.0.0.1}:${BACKEND_PORT:-9999}/docs${NC}"
fi
if [ "$RUN_FRONTEND" = true ]; then
  echo -e "${GREEN}  前端页面:    http://localhost:5173${NC}"
fi
echo -e "${GREEN}  退出:        Ctrl+C${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""

# ── 等待子进程 ──
wait
