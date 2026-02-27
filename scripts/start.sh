#!/usr/bin/env bash
# ============================================================
# 一键启动 DeepSea RAG 全栈（后端 API + 前端 Dev Server）
#
# 用法:
#   bash scripts/start.sh              # 默认端口（后端 9999，前端 5173）
#   bash scripts/start.sh --backend-only   # 仅启动后端
#   bash scripts/start.sh --frontend-only  # 仅启动前端
#   bash scripts/start.sh --no-redis       # 不自动检查/拉起 Redis（不推荐）
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

# ── HuggingFace 缓存目录（统一到 ~/Hug） ──
# 注意：BGE 系列模型可通过 settings.model.*_cache_dir 单独指定，但 ColBERT/ragatouille
# 依赖 transformers/huggingface_hub 的默认缓存机制，不支持传 cache_dir，因此需要靠 HF_HOME 统一控制。
export MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-$HOME/Hug}"
export HF_HOME="${HF_HOME:-$MODEL_CACHE_ROOT}"

# ── macOS 编译环境（torch C++ extension / ColBERT） ──
# ColBERT(ColBERT/Stanford) 在 CPU 路径会 JIT 编译 C++ 扩展 segmented_maxsim_cpp。
# 在 conda 环境里若未显式设置 SDKROOT/CC/CXX，clang 可能找不到标准库头文件（如 <cassert>）。
# 这里默认绑定到系统 Xcode Command Line Tools。
if [ -z "${SDKROOT:-}" ] && command -v xcrun >/dev/null 2>&1; then
  export SDKROOT="$(xcrun --show-sdk-path 2>/dev/null || true)"
fi
if [ -z "${CC:-}" ] && command -v xcrun >/dev/null 2>&1; then
  export CC="$(xcrun --find clang 2>/dev/null || true)"
fi
if [ -z "${CXX:-}" ] && command -v xcrun >/dev/null 2>&1; then
  export CXX="$(xcrun --find clang++ 2>/dev/null || true)"
fi
# CMake/torch 有时读取该变量决定 sysroot
export CMAKE_OSX_SYSROOT="${CMAKE_OSX_SYSROOT:-${SDKROOT:-}}"

# torch.utils.cpp_extension 在部分 conda 环境下不会自动注入 -isysroot，
# 导致系统标准库头文件（如 <cassert>）找不到。这里显式追加。
if [ -n "${SDKROOT:-}" ]; then
  SYSROOT_FLAGS="-isysroot ${SDKROOT} -I${SDKROOT}/usr/include/c++/v1"
  export CPPFLAGS="${CPPFLAGS:-} ${SYSROOT_FLAGS}"
  export CFLAGS="${CFLAGS:-} ${SYSROOT_FLAGS}"
  export CXXFLAGS="${CXXFLAGS:-} ${SYSROOT_FLAGS}"
  export LDFLAGS="${LDFLAGS:-} -Wl,-syslibroot,${SDKROOT}"
  # clang++ 在某些 conda/torch JIT 编译场景下会丢失默认 libc++ include path，
  # 导致 <cassert> 等标准头文件找不到。显式注入 CPLUS_INCLUDE_PATH 可兜底。
  _LIBCXX_INC="${SDKROOT}/usr/include/c++/v1"
  case ":${CPLUS_INCLUDE_PATH:-}:" in
    *":${_LIBCXX_INC}:"*) : ;;  # already present
    *) export CPLUS_INCLUDE_PATH="${_LIBCXX_INC}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" ;;
  esac
fi

# ── 本地模型优先：跳过 HuggingFace 联网校验 ──
export HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-true}"

# ── 严格离线模式（可选） ──
# HF_LOCAL_FILES_ONLY=true 仍可能下载 remote code（transformers_modules）或做元数据请求。
# 若你希望“运行时绝不联网/绝不下载”，启用：
#   RAG_STRICT_OFFLINE=true bash scripts/start.sh
if [ "${RAG_STRICT_OFFLINE:-false}" = "true" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# ── 颜色 ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── 参数解析 ──
RUN_BACKEND=true
RUN_FRONTEND=true
RUN_REDIS=true
for arg in "$@"; do
  case "$arg" in
    --backend-only)  RUN_FRONTEND=false ;;
    --frontend-only) RUN_BACKEND=false ;;
    --no-redis)      RUN_REDIS=false ;;
  esac
done

# ── 端口占用检查与清理 ──
kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${YELLOW}[start.sh] 端口 ${port} (${label}) 已被占用，正在清理旧进程: ${pids}${NC}"
    for p in $pids; do
      # 同时清理子进程（如 reranker multiprocessing worker）
      local children
      children=$(pgrep -P "$p" 2>/dev/null || true)
      if [ -n "$children" ]; then
        echo -e "${YELLOW}[start.sh]   └─ 清理子进程: ${children}${NC}"
        kill $children 2>/dev/null || true
      fi
      kill "$p" 2>/dev/null || true
    done
    sleep 1
    # 如果 SIGTERM 没杀掉，强制 SIGKILL
    local remaining
    remaining=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$remaining" ]; then
      echo -e "${YELLOW}[start.sh] 端口 ${port} 仍被占用，强制终止: ${remaining}${NC}"
      kill -9 $remaining 2>/dev/null || true
      sleep 1
    fi
    echo -e "${GREEN}[start.sh] 端口 ${port} 已释放 ✓${NC}"
  fi
}

# ── 清理函数 ──
PIDS=()
REDIS_STARTED_BY_SCRIPT=0
cleanup() {
  echo ""
  echo -e "${YELLOW}[start.sh] 正在关闭服务...${NC}"
  for pid in "${PIDS[@]}"; do
    # 先清理子进程
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$children" ]; then
      kill $children 2>/dev/null || true
    fi
    kill "$pid" 2>/dev/null || true
  done
  if [ "$REDIS_STARTED_BY_SCRIPT" = "1" ]; then
    local redis_host redis_port
    redis_host="${REDIS_HOST:-127.0.0.1}"
    redis_port="${REDIS_PORT:-6379}"
    echo -e "${YELLOW}[start.sh] 关闭脚本拉起的 Redis (${redis_host}:${redis_port})...${NC}"
    redis-cli -h "$redis_host" -p "$redis_port" shutdown nosave >/dev/null 2>&1 || true
  fi
  wait 2>/dev/null
  echo -e "${GREEN}[start.sh] 已退出${NC}"
}
trap cleanup EXIT INT TERM

# ── Redis 就绪检查（队列模式必需） ──
ensure_redis() {
  local redis_host redis_port
  redis_host="${REDIS_HOST:-127.0.0.1}"
  redis_port="${REDIS_PORT:-6379}"

  if ! command -v redis-cli >/dev/null 2>&1; then
    echo -e "${YELLOW}[start.sh] 未找到 redis-cli，请先安装 Redis（例如: brew install redis）${NC}"
    return 1
  fi

  if redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
    echo -e "${GREEN}[start.sh] Redis 已就绪 ✓ (${redis_host}:${redis_port})${NC}"
    return 0
  fi

  if ! command -v redis-server >/dev/null 2>&1; then
    echo -e "${YELLOW}[start.sh] Redis 未运行且未找到 redis-server，可手动启动后重试。${NC}"
    return 1
  fi

  echo -e "${CYAN}[start.sh] Redis 未就绪，尝试自动启动 (${redis_host}:${redis_port})...${NC}"
  # 仅用于本地开发，禁用持久化以便快速拉起
  redis-server --port "$redis_port" --save "" --appendonly no --daemonize yes >/dev/null 2>&1 || true
  sleep 1
  if redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
    REDIS_STARTED_BY_SCRIPT=1
    echo -e "${GREEN}[start.sh] Redis 启动成功 ✓ (${redis_host}:${redis_port})${NC}"
    return 0
  fi

  echo -e "${YELLOW}[start.sh] Redis 启动失败，请手动启动后重试。${NC}"
  return 1
}

# ── 启动后端 ──
if [ "$RUN_BACKEND" = true ]; then
  BACKEND_PORT="${API_PORT:-9999}"
  BACKEND_HOST="${API_HOST:-127.0.0.1}"
  kill_port "$BACKEND_PORT" "后端 API"

  if [ "$RUN_REDIS" = true ]; then
    ensure_redis || exit 1
  else
    echo -e "${YELLOW}[start.sh] 已跳过 Redis 检查 (--no-redis)；队列接口可能返回 503。${NC}"
  fi

  if [ "${SKIP_DB_MIGRATION:-0}" != "1" ]; then
    echo -e "${CYAN}[start.sh] 执行数据库迁移 (alembic upgrade head)...${NC}"
    if "${PY_CMD[@]}" -m alembic upgrade head; then
      echo -e "${GREEN}[start.sh] 数据库迁移完成 ✓${NC}"
    else
      echo -e "${YELLOW}[start.sh] 数据库迁移失败，已停止启动。可设置 SKIP_DB_MIGRATION=1 跳过。${NC}"
      exit 1
    fi
  else
    echo -e "${YELLOW}[start.sh] 已跳过数据库迁移 (SKIP_DB_MIGRATION=1)${NC}"
  fi

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
  FRONTEND_PORT="${FRONTEND_PORT:-5173}"
  kill_port "$FRONTEND_PORT" "前端 Dev Server"

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
