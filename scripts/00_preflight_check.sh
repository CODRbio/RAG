#!/usr/bin/env bash
# ============================================================
#  环境预检脚本 - 启动前必须执行
# ============================================================
set -euo pipefail

echo "========== 环境预检 =========="

# Python
if command -v python >/dev/null 2>&1; then
  echo "[OK] Python: $(python --version 2>&1)"
else
  echo "[FAIL] Python not found"
  exit 1
fi

# Docker
if command -v docker >/dev/null 2>&1; then
  echo "[OK] Docker: $(docker --version)"
else
  echo "[FAIL] Docker not found"
  exit 1
fi

# Docker Compose
if docker compose version >/dev/null 2>&1; then
  echo "[OK] Docker Compose: $(docker compose version --short)"
else
  echo "[FAIL] Docker Compose not found"
  exit 1
fi

# 端口检查
check_port() {
  local port=$1
  if lsof -i:"$port" >/dev/null 2>&1; then
    echo "[WARN] Port $port is in use"
  else
    echo "[OK] Port $port is free"
  fi
}

check_port 19530
check_port 9091
check_port 9000
check_port 2379

# .env 检查
if [[ -f ".env" ]]; then
  echo "[OK] .env exists"
  # 关键变量
  for var in RAG_ENV MILVUS_HOST COMPUTE_DEVICE INDEX_TYPE; do
    val=$(grep "^${var}=" .env 2>/dev/null | cut -d'=' -f2 || true)
    if [[ -n "$val" ]]; then
      echo "     $var=$val"
    else
      echo "[WARN] $var not set in .env"
    fi
  done
else
  echo "[WARN] .env not found, will copy from .env.example on install"
fi

# GPU 检查（仅 prod）
RAG_ENV=$(grep "^RAG_ENV=" .env 2>/dev/null | cut -d'=' -f2 || echo "dev")
if [[ "$RAG_ENV" == "prod" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[OK] nvidia-smi available"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -2
  else
    echo "[WARN] nvidia-smi not found (GPU may not work)"
  fi
fi

echo "========== 预检完成 =========="
