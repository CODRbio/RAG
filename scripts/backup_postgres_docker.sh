#!/usr/bin/env bash
# ============================================================
# Docker Compose PostgreSQL 逻辑备份
# 用法:
#   bash scripts/backup_postgres_docker.sh
#   BACKUP_DIR=backups/postgres bash scripts/backup_postgres_docker.sh
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-backups/postgres}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DB_NAME="${POSTGRES_DB:-rag}"
DB_USER="${POSTGRES_USER:-rag}"

mkdir -p "$BACKUP_DIR"

DUMP_PATH="${BACKUP_DIR}/${DB_NAME}_${TS}.dump"
META_PATH="${BACKUP_DIR}/${DB_NAME}_${TS}.json"

echo "[backup_postgres] writing ${DUMP_PATH}"
docker compose exec -T postgres pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc > "$DUMP_PATH"

cat > "$META_PATH" <<EOF
{
  "created_at_utc": "${TS}",
  "database": "${DB_NAME}",
  "user": "${DB_USER}",
  "dump_path": "${DUMP_PATH}",
  "restore_example": "docker compose exec -T postgres pg_restore -U ${DB_USER} -d ${DB_NAME} --clean --if-exists < ${DUMP_PATH}"
}
EOF

echo "[backup_postgres] done"
echo "  dump: ${DUMP_PATH}"
echo "  meta: ${META_PATH}"
