#!/usr/bin/env bash
# ============================================================
#  Docker 健康检查脚本 - 启动后必须执行
# ============================================================
set -euo pipefail

echo "========== Docker 健康检查 =========="

# 容器状态
echo "[1/6] 检查容器状态..."
docker compose ps

# PostgreSQL 健康
echo ""
echo "[2/6] 检查 PostgreSQL..."
if docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-rag}" -d "${POSTGRES_DB:-rag}" >/dev/null 2>&1; then
  echo "[OK] PostgreSQL ready"
else
  echo "[FAIL] PostgreSQL health failed"
  echo "  尝试: docker logs --tail 50 deepsea-rag-postgres"
fi

# Redis 健康
echo ""
echo "[3/6] 检查 Redis..."
if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
  echo "[OK] Redis ping OK"
else
  echo "[FAIL] Redis health failed"
  echo "  尝试: docker logs --tail 50 deepsea-rag-redis"
fi

# Milvus 健康
echo ""
echo "[4/6] 检查 Milvus..."
if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then
  echo "[OK] Milvus healthz OK"
else
  echo "[FAIL] Milvus healthz failed"
  echo "  尝试: docker logs --tail 50 deepsea-milvus"
fi

# MinIO 健康
echo ""
echo "[5/6] 检查 MinIO..."
if [ "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' deepsea-minio 2>/dev/null || true)" = "healthy" ]; then
  echo "[OK] MinIO health OK"
else
  echo "[FAIL] MinIO health failed"
  echo "  尝试: docker logs --tail 50 deepsea-minio"
fi

# etcd 健康
echo ""
echo "[6/6] 检查 etcd..."
if docker exec deepsea-etcd etcdctl endpoint health >/dev/null 2>&1; then
  echo "[OK] etcd endpoint health OK"
else
  echo "[FAIL] etcd health failed"
  echo "  尝试: docker logs --tail 50 deepsea-etcd"
fi

echo ""
echo "========== 健康检查完成 =========="
