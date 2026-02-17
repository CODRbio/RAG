#!/usr/bin/env bash
# ============================================================
# 依赖核对脚本（Python + Frontend）
# - Python: 强制在 conda 环境 deepsea-rag 中检查
# - Frontend: 检查 Node/NPM 与 package-lock
# ============================================================

set -u

ENV_NAME="deepsea-rag"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

ok() {
  echo "[OK] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo "[FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
  echo "[WARN] $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

echo "========== 依赖核对开始 =========="
echo "项目目录: $ROOT_DIR"
echo "Conda 环境: $ENV_NAME"
echo ""

if ! command -v conda >/dev/null 2>&1; then
  fail "未找到 conda，无法执行 Python 依赖核对"
  echo ""
  echo "========== 核对结束 =========="
  echo "PASS=$PASS_COUNT FAIL=$FAIL_COUNT WARN=$WARN_COUNT"
  exit 1
fi

# ---------- Python 依赖核对 ----------
echo "---- Python 依赖核对（conda run -n ${ENV_NAME}）----"
if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  PY_VERSION="$(conda run -n "$ENV_NAME" python --version 2>/dev/null | tr -d '\r')"
  ok "$PY_VERSION"
else
  fail "conda 环境 '$ENV_NAME' 不可用"
fi

TMP_PY_SCRIPT="$(mktemp)"
cat > "$TMP_PY_SCRIPT" <<'PY'
from importlib.metadata import version, PackageNotFoundError

try:
    from packaging.specifiers import SpecifierSet
except Exception:
    SpecifierSet = None

pkgs = [
    "pymilvus",
    "docling",
    "pymupdf",
    "pillow",
    "transformers",
    "huggingface-hub",
    "sentence-transformers",
    "FlagEmbedding",
    "datasets",
    "ragatouille",
    "anthropic",
    "openai",
    "fastapi",
    "uvicorn",
    "pandas",
    "tqdm",
    "tavily-python",
    "trafilatura",
    "aiohttp",
    "playwright",
    "playwright-stealth",
    "beautifulsoup4",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "networkx",
    "pydantic",
    "requests",
    "bcrypt",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-exporter-prometheus",
    "prometheus-client",
    "mcp",
    "pytest",
]

constraints = {
    "transformers": ">=4.42.0,<5.0.0",
    "huggingface-hub": "==0.36.0",
    "fastapi": ">=0.100.0,<1.0.0",
    "pydantic": ">=2.0.0,<3.0.0",
    "openai": ">=1.0.0,<3.0.0",
    "mcp": ">=1.26.0",
}

missing = []
incompatible = []
for p in pkgs:
    try:
        v = version(p)
        print(f"{p}=={v}")
        if p in constraints and SpecifierSet is not None:
            spec = SpecifierSet(constraints[p])
            if v not in spec:
                incompatible.append(f"{p}=={v} (need {constraints[p]})")
    except PackageNotFoundError:
        print(f"{p}==MISSING")
        missing.append(p)

if missing:
    print("MISSING_COUNT=" + str(len(missing)))
    print("MISSING_LIST=" + ",".join(missing))
else:
    print("MISSING_COUNT=0")

if SpecifierSet is None:
    print("CONSTRAINT_CHECK=SKIPPED (packaging not available)")
elif incompatible:
    print("INCOMPATIBLE_COUNT=" + str(len(incompatible)))
    print("INCOMPATIBLE_LIST=" + "|".join(incompatible))
else:
    print("INCOMPATIBLE_COUNT=0")
PY

PY_AUDIT_OUTPUT="$(conda run -n "$ENV_NAME" python "$TMP_PY_SCRIPT" 2>/dev/null || true)"
rm -f "$TMP_PY_SCRIPT"

if [ -z "$PY_AUDIT_OUTPUT" ]; then
  warn "Python 包核对输出为空，请手动执行 conda run -n ${ENV_NAME} python -m pip list"
fi

while IFS= read -r line; do
  case "$line" in
    *"==MISSING")
      fail "Python 包缺失: $line"
      ;;
    MISSING_COUNT=0)
      ok "Python 依赖核对通过（无缺失）"
      ;;
    MISSING_COUNT=*)
      # handled by specific missing lines
      :
      ;;
    MISSING_LIST=*)
      warn "缺失列表: ${line#MISSING_LIST=}"
      ;;
    CONSTRAINT_CHECK=*)
      warn "${line#CONSTRAINT_CHECK=}"
      ;;
    INCOMPATIBLE_COUNT=0)
      ok "关键版本约束检查通过"
      ;;
    INCOMPATIBLE_COUNT=*)
      # handled by incompatible list
      :
      ;;
    INCOMPATIBLE_LIST=*)
      IFS='|' read -r -a bad_versions <<< "${line#INCOMPATIBLE_LIST=}"
      for item in "${bad_versions[@]}"; do
        fail "Python 包版本不满足约束: $item"
      done
      ;;
    *=*)
      echo "  $line"
      ;;
  esac
done <<< "$PY_AUDIT_OUTPUT"

# ---------- Frontend 依赖核对 ----------
echo ""
echo "---- Frontend 依赖核对 ----"

if [ ! -f "$FRONTEND_DIR/package.json" ]; then
  fail "未找到 frontend/package.json"
else
  ok "存在 frontend/package.json"
fi

if [ ! -f "$FRONTEND_DIR/package-lock.json" ]; then
  fail "未找到 frontend/package-lock.json"
else
  ok "存在 frontend/package-lock.json"
fi

NODE_CMD=""
NPM_CMD=""
if command -v node >/dev/null 2>&1; then
  NODE_CMD="$(command -v node)"
elif command -v conda >/dev/null 2>&1 && conda run -n "$ENV_NAME" node -v >/dev/null 2>&1; then
  NODE_CMD="conda-run"
fi

if command -v npm >/dev/null 2>&1; then
  NPM_CMD="$(command -v npm)"
elif command -v conda >/dev/null 2>&1 && conda run -n "$ENV_NAME" npm -v >/dev/null 2>&1; then
  NPM_CMD="conda-run"
fi

if [ -n "$NODE_CMD" ]; then
  if [ "$NODE_CMD" = "conda-run" ]; then
    NODE_VER_RAW="$(conda run -n "$ENV_NAME" node -v 2>/dev/null | tr -d '\r')"
    NODE_FROM="conda env: $ENV_NAME"
  else
    NODE_VER_RAW="$("$NODE_CMD" -v | tr -d '\r')"
    NODE_FROM="$NODE_CMD"
  fi
  NODE_VER="${NODE_VER_RAW#v}"
  ok "Node 版本: $NODE_VER_RAW (from $NODE_FROM)"

  # 要求: ^20.19.0 || >=22.12.0
  NODE_MAJOR="$(echo "$NODE_VER" | cut -d'.' -f1)"
  NODE_MINOR="$(echo "$NODE_VER" | cut -d'.' -f2)"
  NODE_PATCH="$(echo "$NODE_VER" | cut -d'.' -f3)"

  # 默认兜底
  NODE_MAJOR="${NODE_MAJOR:-0}"
  NODE_MINOR="${NODE_MINOR:-0}"
  NODE_PATCH="${NODE_PATCH:-0}"

  NODE_OK=0
  if [ "$NODE_MAJOR" -gt 22 ]; then
    NODE_OK=1
  elif [ "$NODE_MAJOR" -eq 22 ]; then
    if [ "$NODE_MINOR" -gt 12 ] || { [ "$NODE_MINOR" -eq 12 ] && [ "$NODE_PATCH" -ge 0 ]; }; then
      NODE_OK=1
    fi
  elif [ "$NODE_MAJOR" -eq 20 ]; then
    if [ "$NODE_MINOR" -gt 19 ] || { [ "$NODE_MINOR" -eq 19 ] && [ "$NODE_PATCH" -ge 0 ]; }; then
      NODE_OK=1
    fi
  fi

  if [ "$NODE_OK" -eq 1 ]; then
    ok "Node 版本满足要求 (^20.19.0 || >=22.12.0)"
  else
    fail "Node 版本不满足要求: $NODE_VER_RAW（需要 ^20.19.0 || >=22.12.0）"
  fi
else
  fail "未找到 node（PATH 或 conda env: $ENV_NAME），请安装 Node.js ^20.19.0 或 >=22.12.0"
fi

if [ -n "$NPM_CMD" ]; then
  if [ "$NPM_CMD" = "conda-run" ]; then
    NPM_VER="$(conda run -n "$ENV_NAME" npm -v 2>/dev/null | tr -d '\r')"
    ok "npm 版本: $NPM_VER (from conda env: $ENV_NAME)"
  else
    ok "npm 版本: $("$NPM_CMD" -v | tr -d '\r') (from $NPM_CMD)"
  fi
else
  fail "未找到 npm（PATH 或 conda env: $ENV_NAME）"
fi

echo ""
echo "========== 核对结束 =========="
echo "PASS=$PASS_COUNT FAIL=$FAIL_COUNT WARN=$WARN_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi

exit 0
