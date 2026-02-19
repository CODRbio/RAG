# 运维与排障

本文档面向日常运行维护，覆盖启动、监控、常见故障和恢复建议。

发布与迁移全流程请参考：`release_migration_ubuntu.md`。

## 一、启动与运行

### 本地全栈

```bash
bash scripts/start.sh
```

可选：

- `bash scripts/start.sh --backend-only`
- `bash scripts/start.sh --frontend-only`
- `API_PORT=8000 bash scripts/start.sh`

### 仅 API

```bash
python scripts/08_run_api.py --reload
```

### Ubuntu 生产托管（systemd）

生产环境建议使用 `systemd` 托管后端与前端服务。
若使用 Nginx 托管前端静态资源，前端 preview service 可停用，仅保留后端 API service。
完整 Nginx + HTTPS 模板见 `release_migration_ubuntu.md`。

后端示例：`/etc/systemd/system/deepsea-rag-api.service`

```ini
[Unit]
Description=DeepSea RAG FastAPI Service
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/path/to/RAG
Environment=PYTHONUNBUFFERED=1
Environment=API_HOST=0.0.0.0
Environment=API_PORT=9999
EnvironmentFile=-/path/to/RAG/.env
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag python scripts/08_run_api.py --host 0.0.0.0 --port 9999'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

前端示例：`/etc/systemd/system/deepsea-rag-frontend.service`

```ini
[Unit]
Description=DeepSea RAG Frontend Preview
After=network.target

[Service]
Type=simple
User=YOUR_USER
Group=YOUR_USER
WorkingDirectory=/path/to/RAG/frontend
Environment=NODE_ENV=production
Environment=CONDA_EXE=conda
ExecStart=/bin/bash -lc '$CONDA_EXE run -n deepsea-rag npm run preview -- --host 0.0.0.0 --port 5173'
Restart=always
RestartSec=5
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
```

启用命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepsea-rag-api deepsea-rag-frontend
sudo systemctl start deepsea-rag-api deepsea-rag-frontend
```

查看日志：

```bash
sudo journalctl -u deepsea-rag-api -f
sudo journalctl -u deepsea-rag-frontend -f
```

补充说明：

- 如果 `conda` 在 systemd 环境不可见，设置 `Environment=CONDA_EXE=/your/path/to/conda`（例如 `/opt/anaconda3/bin/conda`）。
- 这种写法适合把 Node/npm 放在 conda 环境里管理，迁移时无需写死 `/usr/bin/npm` 或固定 node 路径。

## 二、运行时检查

- `GET /health`：服务可用性
- `GET /health/detailed`：组件状态（检索、LLM、图谱等）
- `GET /metrics`：Prometheus 指标
- `GET /storage/stats`：存储情况

### Deep Research 任务检查（新增）

- `GET /deep-research/jobs/{job_id}`：当前状态（`pending/running/cancelling/done/error/cancelled`）
- `GET /deep-research/jobs/{job_id}/events?after_id=...`：增量事件流（推荐排障主入口）
- `GET /deep-research/jobs/{job_id}/reviews`：章节审核状态
- `GET /deep-research/jobs/{job_id}/gap-supplements`：缺口补充是否被采纳（`pending/consumed`）

### Resume Queue 运维检查（新增）

- `GET /deep-research/resume-queue`：查看 resume 队列
- `POST /deep-research/resume-queue/cleanup`：清理历史终态记录
- `POST /deep-research/resume-queue/{resume_id}/retry`：重试指定 resume 请求

建议先看：

- `status=running`：是否存在长时间未更新的恢复请求
- `status=error`：是否出现异常积压
- `owner_instance`：请求是否绑定到预期实例

## 三、Deep Research Resume Queue SOP（值班可执行）

### 场景 A：审核通过后任务未继续执行

目标：确认是否“恢复请求未入队 / 未消费 / 消费失败”。

1. 查看任务状态与事件：
   - `GET /deep-research/jobs/{job_id}`
   - `GET /deep-research/jobs/{job_id}/events`
2. 查看 resume 队列（按 job）：
   - `GET /deep-research/resume-queue?job_id={job_id}&limit=20`
3. 按结果处理：
   - 无队列项：前端/调用方重新提交 review（会重新 enqueue）。
   - `pending` 长时间不变：检查 worker 是否存活、`owner_instance` 是否正确。
   - `running` 长时间不变：优先重启服务（会触发 stale 清理），再判断是否需 retry。
   - `error/cancelled`：走“场景 C 手动重试”。

### 场景 B：批量清理历史垃圾记录

目标：删除旧终态，保留活跃任务。

推荐策略：

- 默认只删终态：`done/error/cancelled`
- 保留最近 48~72 小时

示例请求：

```bash
curl -X POST "http://127.0.0.1:9999/deep-research/resume-queue/cleanup" \
  -H "Content-Type: application/json" \
  -d '{
    "statuses": ["done", "error", "cancelled"],
    "before_hours": 72
  }'
```

注意：

- 不建议清理 `pending/running`，除非确认是僵尸数据并已完成人工评估。

### 场景 C：手动重试某条 resume 请求

适用前提：

- 队列项当前为终态（`done/error/cancelled`）
- 同一 `job_id` 没有活跃项（`pending/running`）

操作步骤：

1. 查询候选项：
   - `GET /deep-research/resume-queue?job_id={job_id}&status=error`
2. 对目标 `resume_id` 重试：

```bash
curl -X POST "http://127.0.0.1:9999/deep-research/resume-queue/{resume_id}/retry" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "manual retry by oncall"
  }'
```

3. 观察结果：
   - 队列项应变为 `pending`
   - 随后 worker 领取后变 `running -> done/error`
   - 同步关注：`GET /deep-research/jobs/{job_id}/events`

常见返回码：

- `404`：`resume_id` 不存在
- `409`：请求已活跃，或同 job 已有活跃恢复请求（需先排空冲突）

### 场景 D：实例切换 / 重启后的恢复策略

当前策略（已实现）：

- 服务启动会将中断态任务重置为 `error`（含 `waiting_review`）
- resume_queue 中 `running` 请求会重置为 `error`

值班建议：

1. 重启后先执行一次队列巡检：
   - `GET /deep-research/resume-queue?status=error&limit=100`
2. 对确需继续的任务，按“场景 C”逐条 retry。
3. 对无需继续的历史项，按“场景 B”批量清理。

### 快速排障决策树（简版）

- 任务没继续：
  - 先看 job events 有无 `section_review` / `resume_start`
  - 再看 resume_queue：
    - 没记录 -> 重新提交 review
    - pending -> 看 worker/实例绑定
    - running 卡住 -> 重启并重试
    - error -> 直接 retry

## 四、日志与产物

- 运行日志：`logs/`（按模块拆分）
- LLM 原始响应：`logs/llm_raw/`
- 评测/任务产物：`artifacts/`

## 五、常见问题与处理

### 1) Milvus 连接失败

排查顺序：

1. `bash scripts/00_healthcheck_docker.sh`
2. 检查 `MILVUS_HOST`、`MILVUS_PORT` 或 `config/rag_config.json`
3. 重启容器：`docker compose --profile dev up -d`

### 2) 检索结果为空

排查顺序：

1. 确认已执行 `02_parse_papers.py` 与 `03_index_papers.py`
2. 检查目标 collection 是否存在（`GET /ingest/collections`）
3. 用 `scripts/04_test_search.py` 单独验证检索链路

### 3) LLM 调用失败/401

排查：

- 检查 `RAG_LLM__{PROVIDER}__API_KEY`
- 检查 `GET /llm/providers` 返回
- 确认 `rag_config.local.json` 是否覆盖了错误值

### 4) Google/Scholar 不稳定

建议：

- 降低 `performance.unified_web_search.browser_providers_max_parallel`
- 安装并检查 Playwright：`playwright install chromium`
- 必要时切换 headless 策略或代理

### 5) ingest 任务卡住

排查：

- `GET /ingest/jobs/{job_id}`
- `GET /ingest/jobs/{job_id}/events`
- 必要时 `POST /ingest/jobs/{job_id}/cancel`

### 6) Deep Research 全部通过后未进入最终整合

排查顺序：

1. 检查审核记录是否覆盖全部章节：
   - `GET /deep-research/jobs/{job_id}/reviews`
2. 检查事件是否出现：
   - `progress(type=all_reviews_approved)`
   - 若没有，通常是章节名不一致或仍有 `pending_sections`
3. 检查是否出现：
   - `progress(type=review_gate_timeout)` / `progress(type=review_gate_early_stop)`
   - 表示审核门超时/早停自动放行
4. 任务结束后确认 `job.status=done` 且 Canvas stage 已切换 `refine`

### 7) 最终整合后引用异常

系统已内置 citation guard：

- 正常：`progress(type=global_refine_done)`
- 保护回退：`progress(type=citation_guard_fallback)`（自动回退到安全版本）

若频繁触发 `citation_guard_fallback`：

- 降低 `synthesize` 步骤模型温度/改用更稳模型
- 缩短单次整合输入规模（减少超长上下文）
- 检查原始草稿中的引用 key 规范性（避免非常规格式）

## 六、存储维护

自动清理配置：

- `storage.max_age_days`
- `storage.max_size_gb`
- `storage.cleanup_on_startup`

手动清理：

```bash
python scripts/19_cleanup_storage.py --vacuum
```

## 七、数据安全建议

- 生产环境关闭默认管理员密码
- 定期备份：
  - `data/parsed/`
  - `src/data/sessions.db`
  - 关键配置文件（脱敏后）
- 变更前先导出核心画布内容（`POST /export`）
