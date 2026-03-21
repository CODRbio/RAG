# 运维与故障排查

本文档覆盖 DeepSea RAG 的日常运维操作、服务监控、常见故障诊断与处理流程。

---

## 1. 服务管理

### 1.1 后端服务（systemd）

```bash
# 查看状态
sudo systemctl status deepsea-rag-backend

# 启动 / 停止 / 重启
sudo systemctl start deepsea-rag-backend
sudo systemctl stop deepsea-rag-backend
sudo systemctl restart deepsea-rag-backend

# 查看实时日志
sudo journalctl -u deepsea-rag-backend -f

# 查看最近 100 行日志
sudo journalctl -u deepsea-rag-backend -n 100 --no-pager
```

### 1.2 基础设施服务（Docker Compose）

```bash
# 查看所有服务状态
docker compose ps

# 启动 / 停止所有服务
docker compose --profile prod up -d
docker compose --profile prod down

# 重启单个服务
docker compose restart redis
docker compose restart milvus-prod

# 查看服务日志
docker compose logs -f milvus-prod
docker compose logs -f redis --tail=50
```

### 1.3 前端（Nginx）

```bash
# 重新加载 Nginx 配置（不中断连接）
sudo systemctl reload nginx

# 测试配置语法
sudo nginx -t

# 查看 Nginx 访问日志
sudo tail -f /var/log/nginx/access.log

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log
```

---

## 2. 健康检查与监控

### 2.1 健康检查接口

```bash
# 基础检查（用于负载均衡 / 存活探针）
curl http://localhost:9999/health
# {"status": "ok", "version": "8.0.0"}

# 详细检查（用于运维诊断）
curl -H "Authorization: Bearer $TOKEN" http://localhost:9999/health/detailed
# {
#   "status": "ok",
#   "components": {
#     "database": "ok",
#     "milvus": "ok",
#     "redis": "ok",
#     "llm_default": "ok"
#   }
# }
```

### 2.2 Prometheus Metrics

```bash
# 查看 Metrics（Prometheus 格式）
curl http://localhost:9999/metrics
```

推荐监控指标：
- `http_requests_total`：请求量
- `http_request_duration_seconds`：响应延迟
- `active_tasks`：当前活跃任务数
- `milvus_search_duration_seconds`：向量检索耗时
- `rag_llm_requests_total`：LLM 请求总数
- `rag_llm_duration_seconds`：LLM 延迟
- `rag_llm_tokens_total`：LLM 输入/输出 token
- `rag_llm_cached_tokens_total`：provider cache 相关 token
- `rag_llm_cache_events_total`：app/provider cache hit/miss/write 事件

### 2.3 日志文件位置

| 日志类型 | 位置 | 说明 |
|---------|------|------|
| 后端结构化日志 | `logs/backend/` | JSON 格式，含 phase/section/job_id 上下文 |
| LLM 原始调用 | `logs/llm_raw/` | JSONL 格式，按大小/时间自动轮转 |
| Debug 诊断 | `logs/debug/` | 开启 Debug 面板后生成 |
| systemd 日志 | `journalctl -u deepsea-rag-backend` | 系统级服务日志 |

### 2.4 实时日志监控

```bash
# 监控后端结构化日志（最新文件）
tail -f logs/backend/$(ls -t logs/backend/ | head -1) | python3 -m json.tool

# 监控特定 job 的日志
grep "job_id=xxx" logs/backend/*.log

# 监控 LLM 调用（含 token 消耗）
tail -f logs/llm_raw/$(ls -t logs/llm_raw/ | head -1) | python3 -m json.tool
```

---

## 3. 任务管理与监控

### 3.1 查询任务状态

```bash
# Chat 任务状态（Redis）
redis-cli GET "rag:task:status:<task_id>"

# Deep Research 任务状态（DB）
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "SELECT job_id, status, phase, updated_at FROM deep_research_jobs ORDER BY updated_at DESC LIMIT 10;"

# Ingest 任务状态（DB）
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "SELECT job_id, status, file_count, done_count, updated_at FROM ingest_jobs ORDER BY updated_at DESC LIMIT 10;"
```

### 3.2 清理卡死任务

若任务长时间停留在 `running` 状态（服务意外崩溃后遗留）：

```bash
# 重启服务时会自动清理卡死任务
# 服务重启逻辑：
# - Ingest running → error
# - DR waiting_review → 保留（可继续）
# - DR 其他非终态 → error

# 手动清理（谨慎！）
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "UPDATE ingest_jobs SET status='error' WHERE status='running';"
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "UPDATE deep_research_jobs SET status='error' WHERE status='running';"
```

### 3.3 Deep Research Resume 队列

Deep Research 支持断点续传，恢复队列存储在 Redis：

```bash
# 查看等待恢复的任务
redis-cli LRANGE "deep_research_resume_queue" 0 -1

# 清空恢复队列（谨慎！会导致等待中的任务无法自动恢复）
redis-cli DEL "deep_research_resume_queue"
```

### 3.4 任务槽位监控

```bash
# 查看当前活跃槽位数
redis-cli GET "rag:active_slots:count"

# 若槽位泄漏（活跃数异常高但无实际任务运行），重启服务可自动清理
```

---

## 3.5 LLM Cache Rollout Playbook

### 推荐上线顺序

1. 先开启 provider 原生缓存：
   - OpenAI：`llm.providers.<name>.params.cache.mode=provider_only`
   - Anthropic：`strategy=auto` 作为默认起点
   - Gemini：先观察 implicit cache，只有需要显式复用大上下文时再传 `cached_content`
2. 观察 24-48 小时，再按需开启 `performance.llm.cache_enabled=true`
3. 只有结构化、模板稳定、可重复的调用点才建议升级为 `provider_plus_app`

### 回滚开关

- 关闭 provider 原生缓存：把对应 provider 的 `params.cache.mode` 设为 `off`
- 关闭应用层 response cache：把 `performance.llm.cache_enabled` 设为 `false`
- 缩短应用层缓存保留时间：调低 `performance.llm.cache_ttl_seconds`

### 需要重点监控

- `rag_llm_cache_events_total{source="provider",result="hit"}`
- `rag_llm_cache_events_total{source="app",result="hit"}`
- `rag_llm_cached_tokens_total`
- `rag_llm_duration_seconds`
- `rag_llm_errors_total`

### 异常排查

- 命中率低：检查 prompt 前缀是否稳定，确认动态内容是否被放到了后半段。
- Anthropic 未命中：检查请求是否真的带了 `cache_control`，以及是否走到了自动缓存或显式断点路径。
- OpenAI 命中不稳定：检查 `prompt_cache_key` 粒度是否过细，或同一 key 是否被高频并发打散。
- Gemini 未复用：检查是否实际传入了 `cachedContent`，以及调用是否仍走 native 路径而非 compat fallback。
- 应用层命中异常：确认该调用点是否包含 tools / streaming / 强时效上下文，这些场景默认不适合 response cache。

---

## 4. 存储清理

### 4.1 自动清理

系统内置存储清理脚本（`scripts/19_cleanup_storage.py`），可定期运行：

```bash
# 清理过期日志（默认保留 30 天）
python3 scripts/19_cleanup_storage.py --logs

# 清理临时文件
python3 scripts/19_cleanup_storage.py --temp

# 全量清理（慎用）
python3 scripts/19_cleanup_storage.py --all
```

### 4.2 手动清理日志

```bash
# 清理超过 30 天的后端日志
find logs/backend/ -name "*.log" -mtime +30 -delete

# 清理超过 7 天的 LLM 原始日志
find logs/llm_raw/ -name "*.jsonl" -mtime +7 -delete

# 清理超过 3 天的 debug 日志
find logs/debug/ -name "*.log" -mtime +3 -delete
```

### 4.3 数据库 VACUUM（PostgreSQL）

PostgreSQL 通常依赖 autovacuum；若批量删除后需要手动回收统计信息，可执行：

```bash
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "VACUUM (ANALYZE);"
```

---

## 5. 检索质量诊断

### 5.1 诊断信息解读

Chat SSE 响应中的 `diag` 事件包含检索诊断：

```json
{
  "phase": "chat_pre_agent_fusion",
  "pool_fusion": {
    "main_in": 50,
    "gap_in": 30,
    "total_reranked": 80,
    "rank_pool_k": 150,
    "rank_pool_multiplier": 3.0,
    "gap_deficit_before_fill": 0,
    "gap_min_keep": 3,
    "gap_in_output": 3,
    "output_count": 15
  },
  "soft_wait_ms": 85000,
  "local_timeout": false,
  "web_timeout": false
}
```

| 字段 | 含义 | 正常范围 |
|------|------|---------|
| `main_in` | 主检索候选数 | > 0 |
| `gap_in` | gap 补搜候选数 | 0 或 > 0（仅证据不足时） |
| `gap_in_output` | 最终 gap 占比 | ≥ gap_min_keep（满足配额） |
| `soft_wait_ms` | Web 超时等待 | < 300000（5 分钟） |
| `local_timeout` | 本地检索是否超时 | false |

### 5.2 常见检索问题

**问题：本地库有内容但检索结果为空**

诊断步骤：
1. 检查 Milvus 是否正常：`curl http://localhost:9091/healthz`
2. 检查 collection 是否存在：
   ```bash
   python3 -c "from pymilvus import connections, utility; connections.connect(); print(utility.list_collections())"
   ```
3. 检查 `local_top_k` 是否太小（推荐 ≥ 20）
4. 查看后端日志中的 `[retrieval]` 相关行

**问题：Web 搜索结果为空**

```bash
# 检查 Playwright 浏览器状态
# 查看 shared_browser 日志
grep "shared_browser" logs/backend/$(ls -t logs/backend/ | head -1)

# 检查 Tavily API Key
grep "web_search" config/rag_config.local.json

# 手动测试 Tavily
python3 -c "
from config.settings import settings
print('Tavily key:', settings.web_search.api_key[:10] if settings.web_search.api_key else 'NOT SET')
"
```

**问题：gap 配额保护日志 WARNING**

日志中出现以下警告均为正常（非错误）：
```
[fuse_pools] gap pool too small: desired_quota=3 but only 1 gap candidates available — using 1 as effective quota
```
表示 gap 池候选数少于理论配额，系统自动降低有效配额至实际 gap 数量。

---

## 6. Deep Research 运维

### 6.1 监控 Deep Research 进度

```bash
# 实时查看 DR 任务状态
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "SELECT job_id, status, phase, current_section, sections_done, sections_total, updated_at FROM deep_research_jobs WHERE status NOT IN ('done', 'error', 'cancelled') ORDER BY updated_at DESC;"
```

### 6.2 Review Gate 超时处理

若 DR 任务停留在 `waiting_review` 状态且用户长时间未审核：

```bash
# 查看等待审核的任务
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "SELECT job_id, waiting_review_at FROM deep_research_jobs WHERE status='waiting_review';"

# 通过 API 跳过审核（代用户操作，谨慎！）
curl -X POST http://localhost:9999/deep-research/jobs/<job_id>/review \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "skip"}'
```

### 6.3 强制收敛（cost_force_summary_steps）

DR 超过 `cost_force_summary_steps` 后会自动设置 `force_synthesize` flag，跳过剩余章节直接综合：

- `lite`：420 步后触发
- `comprehensive`：420 步后触发

若任务耗时异常长但未触发强制收敛，检查 `recursion_limit` 配置：
```bash
grep "recursion_limit\|cost_force" config/rag_config.json
```

### 6.4 深度研究失败恢复

若 DR 任务因错误中断，可查看错误日志后决定是否重新提交：

```bash
# 查看任务错误信息
docker exec deepsea-rag-postgres \
  psql -U rag -d rag -c "SELECT error_message FROM deep_research_jobs WHERE job_id='<job_id>';"

# 若是 LLM API 超时（常见），可重新提交
# 若是数据库/Milvus 问题，先修复基础设施再重提
```

---

## 7. 性能调优

### 7.1 并发配置

```json
// config/rag_config.json
{
  "tasks": {
    "max_active_slots": 3,        // Chat + DR 最大并发（建议 CPU 核数 / 4）
    "ingest_max_concurrent": 2    // Ingest 并发（受 PDF 解析内存限制）
  },
  "shared_browser": {
    "headless_context_pool_size": 4,  // 并发用户多时调大
    "headless_search_reserved_slots": 1
  }
}
```

### 7.2 检索性能调优

| 场景 | 建议调整 |
|------|---------|
| 本地检索慢 | 增加 `search.rerank_top_k`；检查 Milvus GPU 利用率 |
| Web 检索慢 | Scholar 约 145s 属正常；确认 `web_search.max_results` 不过大 |
| Chat 响应慢 | 降低 `local_top_k`（如从 45 → 20）；使用更快的 LLM |
| DR 每章耗时长 | 降低 `max_section_research_rounds`；使用 `depth=lite` |

### 7.3 内存优化

```bash
# 查看进程内存使用
ps aux | grep uvicorn

# 查看 Docker 容器内存
docker stats deepsea-milvus deepsea-rag-redis

# Milvus 内存不足时，考虑：
# 1. 升级服务器内存
# 2. 减少 Milvus 并发查询数
# 3. 使用磁盘索引（性能下降）
```

---

## 8. 日志分析技巧

### 8.1 查找特定任务日志

```bash
# 按 job_id 过滤
grep "job_id=abc-123" logs/backend/*.log

# 按 section 过滤（DR 调试）
grep '"section": "微生物群落"' logs/backend/*.log | python3 -m json.tool

# 查看所有 BGE rerank 操作
grep "pool_rerank\|fuse_pools" logs/backend/*.log | tail -20
```

### 8.2 错误日志分析

```bash
# 查看所有 ERROR 级别日志
grep '"level": "error"' logs/backend/*.log | python3 -m json.tool

# 查看 LLM 调用失败
grep "llm.*error\|api_error\|rate_limit" logs/backend/*.log -i | tail -20

# 查看检索超时
grep "timeout\|timed out" logs/backend/*.log | tail -20
```

### 8.3 Token 消耗分析

```bash
# 分析 LLM token 消耗（需开启 LLM 原始日志）
python3 -c "
import json, glob
total = 0
for f in glob.glob('logs/llm_raw/*.jsonl'):
    with open(f) as fp:
        for line in fp:
            try:
                d = json.loads(line)
                total += d.get('usage', {}).get('total_tokens', 0)
            except: pass
print(f'Total tokens: {total:,}')
"
```

---

## 9. 常见故障处理

### 9.1 服务无法启动

```bash
# 查看错误日志
sudo journalctl -u deepsea-rag-backend -n 50 --no-pager

# 常见原因：
# 1. 端口被占用
lsof -i :9999
kill -9 <pid>

# 2. 依赖服务未就绪
docker compose ps
curl http://localhost:19530  # Milvus
redis-cli ping               # Redis

# 3. 配置文件错误（JSON 语法）
python3 -c "import json; json.load(open('config/rag_config.local.json'))"

# 4. conda 环境未激活
conda activate deepsea-rag && uvicorn src.api.server:app ...
```

### 9.2 Milvus 连接失败

```bash
# 检查 Milvus 是否运行
docker compose ps milvus-prod
curl http://localhost:9091/healthz

# 检查端口是否开放
nc -zv localhost 19530

# 检查配置
grep "milvus" config/rag_config.json

# 重启 Milvus（需等待 30-60s 就绪）
docker compose restart milvus-prod
docker compose logs -f milvus-prod
```

### 9.3 Redis 数据丢失

Redis 默认配置开启了持久化（AOF + RDB），但若容器被删除重建会丢数据：

```bash
# 确认 Redis volume 挂载
docker compose config | grep -A 5 redis

# 若丢失 Chat 任务状态（仅影响进行中的任务，不影响历史数据）
# 重启后已完成的任务状态无法恢复，但历史记录在 DB 中

# 若丢失 DR resume 队列，需重新提交任务
```

### 9.4 浏览器无法启动（Playwright）

```bash
# 验证 Playwright 安装
playwright --version

# 重新安装浏览器
playwright install chromium --with-deps  # Ubuntu
playwright install chromium              # Mac

# 检查 display（有头浏览器需要 X11 或 Xvfb）
# 服务器环境启动 Xvfb
sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### 9.5 PDF 解析失败

```bash
# 查看 Ingest 错误日志
grep "parse\|docling\|error" logs/backend/*.log | tail -20

# 手动测试单文件解析
python3 scripts/test_parse.py --file /path/to/paper.pdf

# 常见原因：
# 1. PDF 加密/有密码 → 手动解密后重新上传
# 2. 扫描版 PDF 无文字 → 开启 OCR：config 中设置 parser.docling_ocr=true
# 3. 超大文件（>100MB）→ 检查解析超时配置 performance.parse_timeout_seconds
```

### 9.6 LLM API 错误

```bash
# 查看 LLM 调用错误
grep "api_error\|APIError\|RateLimitError" logs/backend/*.log | tail -10

# 常见错误：
# 1. rate_limit → 降低并发；在 config 中增加 retry 间隔
# 2. context_length_exceeded → 降低 write_top_k / step_top_k
# 3. api_key_invalid → 检查 rag_config.local.json 中的 API Key
# 4. timeout → 检查网络；考虑换 provider
```

### 9.7 前端无法访问后端

```bash
# 检查 CORS 配置（后端是否允许前端域名）
grep -r "cors\|CORS" src/api/server.py

# 检查 Nginx 代理配置
sudo nginx -t
cat /etc/nginx/sites-enabled/deepsea-rag

# 浏览器 Network 面板查看具体请求错误
# F12 → Network → 查看失败请求的 Response Headers
```

---

## 10. 紧急处置清单

### 服务完全不可用

```bash
1. docker compose ps                         # 检查基础设施
2. curl http://localhost:9999/health         # 检查后端
3. sudo journalctl -u deepsea-rag-backend -n 50  # 查看错误
4. sudo systemctl restart deepsea-rag-backend    # 重启后端
5. docker compose restart                    # 重启基础设施（若需要）
```

### 大量任务卡死

```bash
1. 重启后端（自动清理非终态任务）
   sudo systemctl restart deepsea-rag-backend

2. 检查 Redis 槽位
   redis-cli GET "rag:active_slots:count"

3. 若槽位泄漏，手动清零（谨慎！）
   redis-cli DEL "rag:active_slots:count"
```

### 磁盘空间耗尽

```bash
1. 查找大文件
   du -sh /opt/deepsea-rag/logs/*
   du -sh /opt/deepsea-rag/data/*

2. 清理日志
   find logs/ -name "*.log" -mtime +7 -delete
   find logs/ -name "*.jsonl" -mtime +3 -delete

3. 清理 Docker 镜像
   docker image prune -f
   docker volume prune -f  # 危险！会删除未挂载的 volume
```
