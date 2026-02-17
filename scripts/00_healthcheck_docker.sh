#!/usr/bin/env bash
# ============================================================
#  Docker 健康检查脚本 - 启动后必须执行
# ============================================================
set -euo pipefail

echo "========== Docker 健康检查 =========="

# 容器状态
echo "[1/4] 检查容器状态..."
docker compose ps

# Milvus 健康
echo ""
echo "[2/4] 检查 Milvus..."
if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
  echo "[OK] Milvus healthz OK"
else
  echo "[FAIL] Milvus healthz failed"
  echo "  尝试: docker logs --tail 50 deepsea-milvus"
fi

# MinIO 健康
echo ""
echo "[3/4] 检查 MinIO..."
if curl -sf http://localhost:9000/minio/health/live >/dev/null 2>&1; then
  echo "[OK] MinIO live OK"
else
  echo "[FAIL] MinIO health failed"
  echo "  尝试: docker logs --tail 50 deepsea-minio"
fi

# etcd 健康
echo ""
echo "[4/4] 检查 etcd..."
if docker exec deepsea-etcd etcdctl endpoint health >/dev/null 2>&1; then
  echo "[OK] etcd endpoint health OK"
else
  echo "[FAIL] etcd health failed"
  echo "  尝试: docker logs --tail 50 deepsea-etcd"
fi

echo ""
echo "========== 健康检查完成 =========="
