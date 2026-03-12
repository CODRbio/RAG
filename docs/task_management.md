# 长任务管理契约（Task Management）

本文档约定**长耗时任务**在资源管理、心跳、断点恢复、SSE 展示、前后端状态一致性、错误处理与可观测性等方面的设计与实现规范，确保任务健壮性和前后端一致。

---

## 1. 适用范围

适用任务类型（长耗时、可断连、需进度可见）：

| 任务类型 | 提交入口 | 流式/进度接口 | 典型耗时 |
|---------|----------|---------------|----------|
| **Ingest**（入库） | `POST /ingest/process` 等 | `GET /ingest/jobs/{job_id}/events`（SSE） | 分钟～小时 |
| **Deep Research** | `POST /deep-research/submit` | `GET /deep-research/jobs/{job_id}/stream`（SSE） | 几十分钟～几小时 |
| **Chat 流式** | `POST /chat/stream` | `GET /chat/stream/{task_id}`（SSE） | 十秒～分钟 |
| **Scholar 单篇下载/下载并入库** | `POST /scholar/download`（单篇） | `GET /scholar/task/{task_id}/stream`（SSE） | 十秒～数分钟 |
| **Scholar 批量下载** | `POST /scholar/download/batch` | `GET /scholar/task/{task_id}/stream`（SSE） | 分钟～小时 |
| **Scholar 推荐** | `POST /scholar/libraries/{lib_id}/recommend/start` | `GET /scholar/task/{task_id}/stream`（SSE） | 几十秒～数分钟 |
| **Library 入库** | `POST /scholar/libraries/{lib_id}/ingest` | 复用 Ingest：`GET /ingest/jobs/{job_id}/events`（SSE） | 分钟～小时 |

适用代码范围（含但不限于）：

- `src/api/routes_ingest.py`、`src/api/routes_chat.py`、`src/api/routes_scholar.py`、`src/api/routes_tasks.py`
- `src/indexing/ingest_job_store.py`、`src/collaboration/research/job_store.py`
- `src/utils/task_runner.py`、`src/tasks/dispatcher.py`、`src/tasks/redis_queue.py`、`src/tasks/task_state.py`
- 前端：`frontend/src/api/ingest.ts`、`frontend/src/api/chat.ts`、`frontend/src/api/scholar.ts` 及对应页面/组件

## 2. 目标与原则

- **资源管理**：并发槽位、取消、超时与清理，避免任务堆积与资源泄漏。
- **心跳**：在无业务事件时定期推送心跳，便于前端判断连接存活与任务仍在进行。
- **断点与恢复**：任务进度持久化（checkpoint），支持断连后重连或重启后从断点继续（或至少可查询进度）。
- **SSE 展示**：进度与结果通过 Server-Sent Events 实时推送到前端，事件带 `id` 支持断点续传。
- **状态一致性**：后端为**唯一真相源**；前端通过 `after_id` / `Last-Event-ID` 与终态事件与后端对齐，重连时以服务端状态为准。
- **错误韧性**：对可恢复错误自动重试，对不可恢复错误快速失败并通知，避免静默丢失。
- **可观测性**：关键生命周期节点产出结构化日志与 metrics，便于排查与告警。

## 3. 资源管理

### 3.1 并发与槽位

- **Deep Research**：单实例并发数由配置控制；排队任务写入 `deep_research_resume_queue`，由 `task_runner` 或接口按序拉取执行。与 **Chat** 共享全局槽位（`settings.tasks.max_active_slots`）。
- **Ingest**：由 `task_runner` 轮询 DB 待执行任务，并发上限由 `settings.tasks.ingest_max_concurrent` 控制，与 Chat/DR 全局槽位**独立**。每个 Ingest job 运行在独立的**子进程**（`multiprocessing "spawn"` 模式）中，子进程 OOM 或 segfault 不会影响主 uvicorn 进程，异常退出时由 task_runner 将 job 标记为 `error`。
- **Chat**：流式任务通过 Redis 队列分发，由 `task_runner` 内 `run_unified_chat_worker_once` 拉取执行，占用全局槽位。
- **Scholar**（单篇下载、批量下载、推荐）：**不占用** Chat/DR 全局槽位；状态与事件存 Redis（`rag:task:*`、`rag:task_events:*`）。批量下载在 API 进程内以 `asyncio.create_task` 运行，`max_concurrent` 由请求体 `BatchDownloadRequest.max_concurrent` 控制。

### 3.2 任务状态枚举

系统中存在三套并行的状态枚举，前后端对接时需注意映射关系：

| 系统 | 枚举值 | 说明 |
|------|--------|------|
| **Redis TaskStatus**（Chat/Scholar） | `queued` / `running` / `completed` / `error` / `cancelled` / `timeout` | `completed` 对应 Ingest/DR 的 `done`；`timeout` 仅 Chat/Scholar 有 |
| **IngestJob DB 状态** | `pending` / `running` / `done` / `error` / `cancelled` | `pending` 对应 Redis 的 `queued`；`done` 对应 Redis 的 `completed` |
| **DeepResearchJob DB 状态** | `pending` / `planning` / `running` / `waiting_review` / `cancelling` / `done` / `error` / `cancelled` | 含 `planning`、`waiting_review`、`cancelling` 等额外中间状态 |

前端展示时建议将 `completed`/`done` 统一展示为成功，`timeout` 可作为 `error` 的特殊子类展示。

### 3.3 取消

- **请求取消**：前端调用取消接口（如 `POST /ingest/jobs/{job_id}/cancel`、`POST /deep-research/jobs/{job_id}/cancel`、`POST /tasks/{task_id}/cancel`）。**Scholar 任务**（单篇/批量/推荐）统一使用 `POST /tasks/{task_id}/cancel`（与 Chat 共用，按 `task_id` 写 Redis 状态为 `cancelled`）。
- **后端语义**：
  - 将任务标记为「取消请求」或终态 `cancelled`；
  - 工作线程/协程轮询取消标志（如 `_is_cancel_requested(job_id)`、`_dr_request_cancel`），在安全点退出；
  - 取消后释放槽位并清理可释放的临时资源。
- **Scholar 取消**：单篇/批量/推荐均在循环内轮询 `_scholar_is_cancelled(task_id)` 并提前退出，推送 `cancelled` 事件。
- **前端**：收到 `cancelled`（或等价）事件后停止 SSE、更新 UI，以服务端状态为准。

### 3.4 超时与清理

- **任务级超时**：长步骤（如单文件解析）应有超时（如 Ingest 解析 1800s），超时后标记失败并进入错误/终态。Chat 排队/运行有 `queue_timeout_seconds`、`run_timeout_seconds`。
- **Checkpoint TTL**：已完成或已取消任务的 checkpoint 可在一段时间后清理（如 7 天），避免无限增长；由 `task_runner` 启动时或定时任务执行 `purge_stale_*_checkpoints`。
- **Resume 队列**：Deep Research 的 `resume_queue` 在任务终态或重启后需可清理/重置，见运维文档。
- **Scholar 任务状态 TTL**：Redis 中任务状态与事件流由 `settings.tasks.task_state_ttl_seconds` 控制过期，过期后前端无法再拉取历史事件；批量下载/推荐无单独 checkpoint，仅依赖 Redis 状态与事件。

### 3.5 优雅关闭与重启

服务关闭或重启时，需确保正在运行的长任务不会静默丢失或永久卡在非终态：

- **收到 SIGTERM/SIGINT 时**（uvicorn lifespan shutdown）：
  - 等待 Scholar 后台任务最多 `graceful_shutdown_timeout_seconds`（默认 30s），超时后取消剩余任务；
  - 取消 `run_background_worker` 任务；
  - 关闭 downloader adapter（浏览器等）与 SharedContextPool。
- **Ingest**：任务运行在**独立子进程**（`multiprocessing.get_context("spawn")`）中。服务进程退出或 SIGTERM 时，子进程可能继续在后台运行直到操作系统回收。重启后，`cleanup_stale_jobs()` 将仍处于 `running` 状态的 Ingest job 标记为 `error` 并**清除其 checkpoint**（因重试使用新 job_id，旧 checkpoint 无复用价值）；子进程入口（`_ingest_subprocess_target`）在执行前会重新读取 DB 状态，若 job 已被标记为 `error` 则直接退出，避免旧子进程与新任务并发写入同一 collection；前端需重新提交任务。
- **Deep Research**：重启后，`cleanup_stale_jobs()` 将仍处于 `running/cancelling` 的 DR job 标记为 `error` 并清除其 checkpoint；同时清理 DR resume_queue 中 status=running 的条目。**`waiting_review` 状态**的 DR job 单独处理——仍标为 `error` 但 **不清除其 checkpoint**，error_message 说明可通过 resume_queue 从上次 review 节点继续，前端可据此展示 resume 入口而非普通错误 UI。
- **Chat**：流式任务较短，宽限期内通常可完成；超时则标记为 `error` 并通知前端。启动时 `repair_stale_chat_tasks(max_age_seconds=60)` 扫描 Redis 中超时 `running` 的 Chat 任务并标为 `error`，同时**释放其 `rag:active_tasks` / `rag:active_sessions` 槽位**，避免同 session 的新 Chat 任务被阻塞。
- **Scholar**：无 DB checkpoint，强制终止后 Redis 中状态可能停留在 `running`。启动时 `repair_stale_scholar_tasks(max_age_seconds=300)` 扫描 Redis 中超时 `running` 任务并标为 `error`（附"服务重启，任务中断"说明）。
- **Redis 槽位 reconcile**：启动时释放 stale DR/Chat 任务占用的 `rag:active_tasks` / `rag:active_sessions`，避免同 session 的新任务被阻塞。运行中每 60s 执行一次 reconcile，清理僵尸条目。

## 4. 心跳（Heartbeat）

### 4.1 目的

- 在无业务事件时保持 SSE 连接活跃，便于前端与代理判断连接有效。
- 携带当前任务状态摘要（如当前文件、阶段、已用时间），便于 UI 展示「进行中」而非卡死。

### 4.2 各任务类型心跳约定

各任务类型的心跳实现有所差异，汇总如下：

| 任务类型 | 触发条件 | 间隔 | 事件名 | Payload 字段 |
|----------|----------|------|--------|--------------|
| **Ingest** | 每 5 个 idle tick（tick=1s） | ~5s | `heartbeat`（SSE 事件） | `{job_id, file, stage, message}` |
| **Deep Research** | 每 5 个 idle tick（tick=2s） | ~10s | `heartbeat`（SSE 事件） | 完整 DR 状态（含 `heartbeat_ts`、`elapsed_since_start_ms`、`current_stage`，见 `_dr_job_status_payload`） |
| **Chat 流式** | 固定间隔 | 15s | SSE 注释行（`: \n\n`，非事件） | 无 payload，仅保持连接 |
| **Scholar 批量流** | 固定间隔 | 5s | SSE 注释行（`: heartbeat\n\n`，非事件） | 无 payload，仅保持连接 |
| **Scholar 推荐 / 单篇** | 固定间隔 | 5s | `heartbeat`（SSE 事件） | `{stage, elapsed_s}` |

**注意**：
- Deep Research 的心跳事件名为 `heartbeat`（payload 复用 `_dr_job_status_payload` 结构，含 `heartbeat_ts`、`current_stage` 等）；终态事件名为 `job_status`，两者不同。前端收到 `heartbeat` 仅更新进度展示，收到 `job_status` 才代表流结束。
- Chat / Scholar 批量流使用 SSE 注释行（`: ...`），浏览器和代理会丢弃注释行内容，仅用于防止空闲断连；前端无需特殊处理。
- **前端通用**：收到 `heartbeat` / `job_status` 仅更新进度展示，可用来重置「无数据超时」计时器，不触发业务逻辑。

### 4.3 终态前最后一次状态

- 任务进入终态（`done` / `error` / `cancelled`）时，应先拉取并推送完所有未推送事件，再发送一次**终态状态事件**，然后关闭流，保证前端与后端状态一致。
- **Ingest**：`done` 事件的 data 包含 `errors` 数组（聚合所有文件级错误 `{file, stage, error}`），可直接用于 UI 展示哪些文件失败。
- **Deep Research**：流结束前发出 `job_status` 事件（payload 含终态 `status`、`result_dashboard`、`canvas_id` 等），是前端获取完整结果的主要来源。

## 5. 断点与恢复（Checkpoint & Resume）

### 5.1 Ingest

- **粒度**：按**文件 + 阶段**持久化（`ingest_checkpoints` 表），阶段例如：`parsed`、`chunked`、`embedded`、`indexed`。
- **写入时机**：每个文件每完成一个阶段即 `save_ingest_checkpoint(job_id, file_name, stage, ...)`，并发出 `checkpoint` 事件（可选，便于前端展示）。
- **恢复语义**：同一 job 重跑时，对每个文件根据 `get_last_completed_stage` 跳过已完成阶段，只做未完成部分。
- **取消/完成**：任务取消或完成后可 `purge_ingest_checkpoints(job_id)`；重启导致的「孤儿」job 由启动时 `purge_stale_ingest_checkpoints` 等清理。

### 5.2 Deep Research

- **粒度**：按**阶段/章节**持久化（`deep_research_checkpoints` 表），如 `plan`、`research`、`generate_claims`、`write`、`verify`、`section_done`、`review_gate`、`synthesize` 等。
- **写入时机**：在关键节点调用 `_save_phase_checkpoint`，并可选发出 `checkpoint_saved` 事件。
- **恢复**：通过 `resume_queue` 与 `load_checkpoint` 支持「从上次断点继续」；前端可从「未完成 job」入口恢复监测或触发 resume。
- **崩溃**：执行异常时保存 `crash` checkpoint，便于后续从最后完成章节恢复。
- **中间状态说明**：DR DB（`DeepResearchJob`）状态枚举比 Redis TaskStatus 更丰富，包含以下值：

  | 状态 | 含义 |
  |------|------|
  | `pending` | 已提交，等待 worker 领取 |
  | `planning` | 规划阶段（生成大纲等）正在进行 |
  | `running` | 主研究流程执行中 |
  | `waiting_review` | 触发 review gate，等待用户确认后继续 |
  | `cancelling` | 已收到取消请求，正在安全点退出 |
  | `done` | 正常完成 |
  | `error` | 执行出错 |
  | `cancelled` | 已取消 |

  `waiting_review` 是重要的挂起状态：任务暂停并通过 `DRResumeQueue` 等待前端的 resume 指令；前端需轮询 job 状态或监听 SSE 中的 `step: review_gate` 事件来识别此状态并展示确认 UI。

### 5.3 Chat / Scholar

- **Chat**：流式输出通过 Redis 等存储事件，`GET /chat/stream/{task_id}` 支持 `Last-Event-ID` 断点续传，不要求 DB 级 checkpoint。
- **Scholar**（单篇下载、批量下载、推荐）：
  - 任务状态与事件均存 Redis（`TaskState`、`read_events`/`push_event`），无 DB checkpoint。
  - 单篇：`process_download_and_ingest` 写进度与终态；可选触发入库时由 `_watch_ingest_and_update_scholar_task` 轮询 Ingest job 直至完成。
  - 批量：`POST /scholar/download/batch` 产生一个父 `task_id`，`_batch_job` 内对每篇调用单篇下载或 `process_download_and_ingest`，进度通过同一 `task_id` 的 event 流推送（`progress`、`heartbeat`、`done`/`error`）。
  - 推荐：`POST /scholar/libraries/{lib_id}/recommend/start` 产生 `task_id`，`_run_recommend_job` 写事件到同一 Redis 流。
  - 断点续传：`GET /scholar/task/{task_id}/stream` 支持 query `after_id` 或 header `Last-Event-ID`，后端 `read_events(task_id, after_id=last_id)` 只返回该 id 之后的事件；每条 SSE 带 `id: <id>`。
- **Library 入库**：`POST /scholar/libraries/{lib_id}/ingest` 创建的是标准 Ingest 任务（`create_job`），返回 `job_id`，进度与断点续传走 **Ingest** 的 `GET /ingest/jobs/{job_id}/events`，见 §5.1。

## 6. 错误分类与重试策略

### 6.1 错误分类

所有任务执行中的错误应归为以下两类，以决定后续处理方式：

| 分类 | 定义 | 示例 | 处理方式 |
|------|------|------|----------|
| **可重试（Transient）** | 临时性故障，重试有合理概率成功 | 网络超时、外部 API 限流（429）、Redis 连接瞬断、LLM API 临时 5xx | 自动重试，指数退避 |
| **不可重试（Fatal）** | 重试不会改变结果 | 文件格式无法解析、权限不足、参数非法、LLM 拒绝响应（内容违规） | 立即标记为 `error` 终态，推送错误事件 |

### 6.2 重试约定

- **最大重试次数**：建议 3 次（可由 `settings.tasks.max_retries` 配置）。
- **退避策略**：指数退避 + 抖动，初始间隔 1s，公式为 `min(base * 2^attempt + random(0, 1), max_interval)`，`max_interval` 建议 60s。
- **重试范围**：仅在**当前失败步骤**重试，不回退已完成的 checkpoint 阶段。
- **重试耗尽**：达到最大次数后标记为 `error`，写入最后一次错误详情，推送 `error` 事件。
- **错误事件 Payload（规范目标）**：应包含 `error_code`（机器可读，如 `PARSE_FAILED`、`LLM_TIMEOUT`）、`error_message`（人可读描述）、`retryable: boolean`，便于前端区分展示。
- **当前实现现状**：各任务类型的 `error` 事件 payload 当前只有 `{"message": "..."}` 字段；`error_code` 和 `retryable` 尚未在 SSE 事件层面落地（仅在内部函数 `_is_transient_ingest_error` 等中区分）。Ingest 文件级错误额外携带 `{file, stage, error}`。这是规范建议与当前实现之间已知的 gap，后续可逐步补充。

### 6.3 各任务重试现状

- **Ingest**：已实现对 parse / chunk / embed / upsert 四阶段分别做 transient 错误重试（`max_retries`，指数退避）。瞬态判断由 `_is_transient_ingest_error` 函数负责（TimeoutError、ConnectionError、HTTP 429/500/502/503、"rate limit" 等视为 transient）；单文件失败不阻塞其他文件。
- **Deep Research**：LLM 调用失败时部分步骤有内置 retry（依赖 LLM client 配置），章节级失败保存 `crash` checkpoint，可通过 resume 手动恢复。
- **Chat**：流式任务较短，一般不做后端自动重试；前端可通过重新提交实现。
- **Scholar**：外部下载（网络请求、验证码等）通过多策略切换机制（`adapter.download_paper`）覆盖部分重试场景；策略级重试在 downloader adapter 内部处理。

## 7. SSE 契约

### 7.1 媒体类型与头

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache, no-store, must-revalidate`
- `X-Accel-Buffering: no`（禁止代理缓冲）
- `Connection: keep-alive`

### 7.2 事件格式

- 每条 SSE 至少包含：
  - `event: <type>`
  - `data: <JSON>`
- **断点续传**：每条事件应带 `id: <id>`，前端重连时通过 `Last-Event-ID` header（或 query `after_id`）传递，后端只返回该 id 之后的事件，避免重复与状态错位。

**各任务类型 `after_id` 类型与断点续传差异**：

| 任务类型 | SSE `id:` 行 | after_id 类型 | 重连参数 |
|----------|-------------|---------------|----------|
| **Ingest** | 有（整数 `event_id`） | 整数（DB row id） | query `after_id` 或 `Last-Event-ID` header |
| **Deep Research** | **有**（整数 `event_id`，仅 DB 持久化事件带 `id:` 行；heartbeat/job_status 不带） | 整数（DB row id） | query `after_id`；前端通过 `streamSSEResumable` 自动重连并携带 `after_id` |
| **Chat** | 有（整数或字符串） | 字符串（Redis Stream ID，格式 `<ms>-<seq>`） | `Last-Event-ID` header |
| **Scholar** | 有（Redis Stream ID） | 字符串（Redis Stream ID，格式 `<ms>-<seq>`） | query `after_id` 或 `Last-Event-ID` header |

**注意**：Deep Research SSE 流的 `id:` 行仅在 DB 持久化事件（来自 `dr_job_events` 表）上输出，heartbeat 与 job_status 事件不含 `id:`，因此 `lastEventId` 仅在真实事件时更新，保证重连精确落在最后持久化事件之后。前端通过 `streamSSEResumable` 统一管理断点续传，无需手动读取 `data._event_id`。

### 7.3 终态事件

流必须在结束时发送明确的**终态事件**，前端据此停止重连并做最终 UI 更新：

| 任务类型 | 终态事件 | 说明 |
|----------|----------|------|
| **Ingest** | `done`、`error`、`cancelled` | **无 `timeout`**；`done` 事件 data 含 `errors` 数组汇总文件级错误 |
| **Deep Research** | `job_status`（含终态 `status` 字段） | 流结束前发出；之前可能有 `done`/`error`/`cancelled` 等业务事件 |
| **Chat** | `done`、`error`、`cancelled`、`timeout` | `timeout` 含 `queue_timeout` 和 `run_timeout` 两种 |
| **Scholar** | `done`、`error`、`cancelled`、`timeout` | 前端常量：`SCHOLAR_TERMINAL_EVENTS = ['done','error','cancelled','timeout']` |

### 7.4 当前实现与差异

| 项目 | Ingest | Deep Research | Chat | Scholar |
|------|--------|---------------|------|---------|
| SSE 端点 | `GET /ingest/jobs/{id}/events` | `GET /deep-research/jobs/{id}/stream` | `GET /chat/stream/{task_id}` | `GET /scholar/task/{task_id}/stream` |
| 断点参数 | `after_id`（query，整数） | `after_id`（query，整数） | `Last-Event-ID`（header，Redis Stream ID） | `after_id`（query）或 `Last-Event-ID`（header），Redis Stream ID |
| 事件是否带 `id:` 行 | 有（整数 `event_id`） | 有（整数 `event_id`；仅 DB 持久化事件；heartbeat/job_status 不带） | 有（Redis Stream ID） | 有（Redis Stream ID） |
| 心跳 | ~5s，`event: heartbeat`，payload `{job_id,file,stage,message}` | ~10s，`event: heartbeat`，payload 完整 DR 状态（注意事件名是 `heartbeat`，终态才是 `job_status`） | ~15s，SSE 注释行（`: \n\n`，无 payload） | ~5s；batch 流用注释行；推荐/单篇用 `event: heartbeat`，payload `{stage,elapsed_s}` |
| 终态 | `done`/`error`/`cancelled`（无 timeout） | `job_status`（含终态 status）或 `error`（job 不存在） | `done`/`error`/`cancelled`/`timeout` | `done`/`error`/`cancelled`/`timeout` |
| 前端 adapter | `streamSSEResumable` | `streamSSEResumable`（`DR_TERMINAL_EVENTS`） | `streamChatByTaskId`（内置重连） | `streamSSEResumable`（`SCHOLAR_TERMINAL_EVENTS`） |

**当前状态**：所有任务类型均已支持 SSE `id:` 行与断点续传；前端统一通过 `frontend/src/api/sse.ts` 的 `streamSSEResumable` adapter 重连，包括 Deep Research。DR 的 heartbeat 事件不带 `id:` 行，`lastEventId` 仅在真实 DB 事件时更新，保证重连精确落在持久化事件之后。

## 8. 状态一致性（前后端对齐）

### 8.1 后端为真相源

- 任务状态、进度、错误信息以**后端存储与事件流**为准；前端不臆断终态，仅在收到终态事件或终态状态接口返回后更新为终态。

### 8.2 前端重连策略

所有任务类型在 SSE 意外断开且未收到终态时，应采用**指数退避重连**：

- **退避公式**：`min(base_ms * 2^attempt + random(0, base_ms), max_ms)`
- **推荐参数**：`base_ms = 1000`，`max_ms = 30000`，最大重试次数 5 次。
- **重连时携带断点标识**，只拉取新事件，避免重复处理。

各任务类型重连细节：

- **Chat**：带 `Last-Event-ID` header 重连（Redis Stream ID 字符串）。
- **Ingest**：以当前已收的最大 `event_id`（整数）作为 `after_id` query 参数重连；也支持 `Last-Event-ID` header。
- **Deep Research**：SSE 流对 DB 持久化事件带 `id:` 行（整数），前端 `streamSSEResumable` 自动追踪 `lastEventId` 并在重连时拼入 `?after_id=<lastEventId>` query 参数；heartbeat 与 job_status 不带 `id:` 行，不影响重连位置。页面刷新后用 `localStorage` 保存的 `job_id` 重新打开 SSE。
- **Scholar**（单篇/批量/推荐）：用本地保存的 `task_id` 重新打开 `GET /scholar/task/{task_id}/stream`，传 query `after_id` 或 header `Last-Event-ID` 为上次收到的最后一条事件 id；终态前可轮询 `GET /scholar/task/{task_id}` 作辅助。

### 8.3 终态同步

- 收到终态事件后，前端应：
  - 停止 SSE 与重连；
  - 用终态 payload 或随后一次 `GET /jobs/{id}` 的结果做最终展示；
  - 清理本地「进行中」标记（如 `ingest_active_job_id`）。

### 8.4 双向确认语义

- **取消**：前端发取消 → 后端设取消标志并尽快写入 `cancelled` 与 `done` → 前端收到 `cancelled`/`done` 后确认。
- **恢复/继续**：前端发 resume 或「继续监测」→ 后端从 checkpoint 或 resume_queue 继续 → 通过 SSE 继续推送事件。
- **状态轮询**：在未建立 SSE 或 SSE 断开时，前端可用 `GET /jobs/{id}` 或 `GET /tasks/queue` 轮询状态（建议配合 `getWithRetry` 等），但不应以轮询替代「终态事件」作为终态判断依据，以避免竞态。

## 9. 日志与可观测性

### 9.1 结构化日志

所有长任务关键生命周期节点应输出结构化日志，至少包含以下字段：

- `task_type`（ingest / deep_research / chat / scholar_download / scholar_batch / scholar_recommend）
- `task_id` / `job_id`
- `event`（如 `task_started`、`task_completed`、`task_failed`、`task_cancelled`、`checkpoint_saved`、`retry_attempt`）
- `duration_ms`（适用于完成和失败）
- `error_code` + `error_message`（适用于失败）

示例：
```json
{
  "task_type": "ingest",
  "job_id": "abc-123",
  "event": "task_failed",
  "duration_ms": 45200,
  "error_code": "PARSE_FAILED",
  "error_message": "Unsupported file format: .xyz",
  "retryable": false
}
```

### 9.2 关键 Metrics（建议）

| Metric | 类型 | 说明 |
|--------|------|------|
| `task_active_count{type}` | Gauge | 各类型当前活跃任务数 |
| `task_queue_depth{type}` | Gauge | 各类型排队中任务数 |
| `task_duration_seconds{type, status}` | Histogram | 任务耗时分布（按终态区分） |
| `task_error_total{type, error_code}` | Counter | 错误计数（按类型和错误码） |
| `task_retry_total{type}` | Counter | 重试次数 |
| `sse_connection_count` | Gauge | 当前 SSE 活跃连接数 |

### 9.3 告警建议

- 任务队列深度超过阈值（如 > 50 且持续 5 分钟）。
- 单任务运行时间超过预期上限的 2 倍。
- 某类型任务连续失败超过 N 次（如 5 次）。
- 活跃 SSE 连接数异常增长（可能为前端重连风暴）。

## 10. 各任务类型对照摘要

| 维度 | Ingest | Deep Research | Chat Stream | Scholar（单篇/批量/推荐） |
|------|--------|---------------|-------------|---------------------------|
| 任务标识 | `job_id` | `job_id` | `task_id` | `task_id`（批量/推荐各一父 task_id） |
| 事件存储 | DB `ingest_job_events` | DB job_store events | Redis `rag:task_events:*` | Redis `rag:task_events:*` |
| Checkpoint | 按文件+阶段（DB） | 按阶段/章节（DB） | 无 | 无 |
| 心跳 | ~5s，`event: heartbeat` | ~10s，`event: heartbeat`（终态为 `job_status`） | ~15s，SSE 注释行 | ~5s；batch 用注释行，推荐/单篇用 `event: heartbeat` |
| 取消 | `POST /ingest/jobs/{id}/cancel` | `POST /deep-research/jobs/{id}/cancel` | `POST /tasks/{task_id}/cancel` | `POST /tasks/{task_id}/cancel`（批量/推荐/单篇循环内已轮询取消） |
| 断点续传 | `after_id`（整数）+ `Last-Event-ID`，事件带 `id:` | `after_id`（整数），DB 持久化事件带 `id:` 行；heartbeat 不带；前端用 `streamSSEResumable` | `Last-Event-ID`（Redis Stream ID） | `after_id` / `Last-Event-ID`（Redis Stream ID），事件带 `id:` |
| 重启恢复 | 标为 error + 清除 checkpoint（子进程隔离；子进程入口校验状态，新任务需重新提交） | running/cancelling：标为 error + 清除 checkpoint；waiting_review：标为 error + **保留 checkpoint**，可通过 resume_queue 从 review 节点恢复 | 标为 error + 释放 active slot（`repair_stale_chat_tasks`） | 启动时 `repair_stale_scholar_tasks` 将长时间 `running` 标为 error |

## 11. 参考实现与配置

- **Ingest**：`src/api/routes_ingest.py`（`_emit_job_event`、`stream_ingest_job_events`、`_run_ingest_job`、cancel/checkpoint；`_is_transient_ingest_error`：判断 TimeoutError / ConnectionError / OSError / HTTP 429 / 500 / 502 / 503 / "rate limit" 为 transient，其余为 fatal）、`src/indexing/ingest_job_store.py`。
- **Deep Research**：`src/api/routes_chat.py`（`stream_deep_research_events`、`_dr_job_status_payload`、cancel）、`src/collaboration/research/agent.py`（`_save_phase_checkpoint`、`_emit_progress`）、`src/collaboration/research/job_store.py`。
- **Chat**：`src/api/routes_chat.py`（`GET /chat/stream/{task_id}`、id 与 Last-Event-ID）、`frontend/src/api/chat.ts`（`streamChatByTaskId` 重连与终态判断）；队列与执行：`src/tasks/dispatcher.py`（`run_chat_task_sync`、`run_unified_chat_worker_once`）、`src/tasks/redis_queue.py`、`src/tasks/task_state.py`。
- **Scholar**：
  - 单篇下载/下载并入库：`src/tasks/dispatcher.py`（`process_download_and_ingest`、`_watch_ingest_and_update_scholar_task`）；状态与事件：Redis 同一套 `TaskQueue`。
  - 批量下载：`src/api/routes_scholar.py`（`scholar_batch_download`、`_batch_job`、`task_id = batch_dl_*`），流：`_scholar_event_stream`、`GET /scholar/task/{task_id}/stream`（`after_id` / `Last-Event-ID`）。
  - 推荐：`src/api/routes_scholar.py`（`recommend_library_papers_start`、`_run_recommend_job`），同一 SSE 端点。
  - 任务队列与取消：`src/api/routes_tasks.py`（`GET /tasks/queue`、`POST /tasks/{task_id}/cancel`）；Scholar 任务状态存 Redis，取消接口通用；批量/推荐/单篇在循环内轮询 `_scholar_is_cancelled` 并提前退出。
  - 文献下载策略与验证码等流程见 `docs/information_download_gather.md`。
- **Library 入库**：`src/api/routes_scholar.py`（`ingest_scholar_library`）内部调用 `create_job`，返回 `job_id`，进度走 Ingest 的 `GET /ingest/jobs/{job_id}/events`。
- **启动清理**：`src/utils/task_runner.py`（`cleanup_stale_jobs()`：重置 stale ingest/DR/resume_queue 任务为 error、清除对应 checkpoint（`waiting_review` 的 DR job 保留 checkpoint）、释放 Redis 槽位；`repair_stale_scholar_tasks(max_age_seconds=300)`：修复 Scholar 孤儿任务；`repair_stale_chat_tasks(max_age_seconds=60)`：修复 Chat 孤儿任务并释放 active slot；`_reconcile_redis_active_slots()`：运行中每 60s 清理僵尸 Redis slot）。

配置（`config/settings.py`、`rag_config.json` 等）：`tasks.max_active_slots`、`tasks.ingest_max_concurrent`、`tasks.queue_timeout_seconds`、`tasks.run_timeout_seconds`、`tasks.task_state_ttl_seconds`、`tasks.max_retries`、`tasks.graceful_shutdown_timeout_seconds`、`tasks.redis_url` 等。运维细节见 `configuration.md`、`operations_and_troubleshooting.md`。

## 12. 禁止与建议

- **禁止**：在未收到明确终态事件或终态接口结果前，仅凭超时或断连将前端状态置为终态（应重连或轮询后再判定）。
- **禁止**：SSE 在终态后继续推送业务事件（应先 flush 未推送事件，再发终态，再结束流）。
- **禁止**：对不可重试错误（Fatal）执行自动重试，浪费资源且可能产生副作用。
- **建议**：所有长任务 SSE 都带 `id` 并支持 `after_id`/`Last-Event-ID`，便于统一重连与去重。
- **建议**：心跳 payload 与终态状态 payload 结构在文档或类型定义中固定，便于前端稳定解析与展示。
- **建议**：错误事件统一包含 `error_code`、`error_message`、`retryable` 三字段。
- **建议**：前端对每种任务类型的 SSE 事件处理应有统一的 adapter 层，减少重复逻辑。

## 13. 回归测试场景

| 场景 | 预期 |
|------|------|
| 取消运行中 Scholar 批量/推荐 | 循环内检测到取消后尽快退出并推送 `cancelled`，前端收到后停止 SSE。 |
| SSE 断连后重连（Scholar/Ingest/DR） | 前端带 Last-Event-ID/after_id 重连，只收到断点之后的事件，无重复。 |
| DR 任务运行中网络中断 | `streamSSEResumable` 以指数退避重连，通过 `?after_id=<last_event_id>` 从最后持久化事件继续，不丢失进度。 |
| 服务重启期间 Scholar 任务在跑 | 启动后 300s 内将仍为 `running` 的 Scholar 任务标为 error（”服务重启，任务中断”）。 |
| 服务重启期间 Chat 任务在跑 | 启动后将仍为 `running` 的 Chat 任务标为 error，释放 active slot，同 session 立即可提交新任务。 |
| 服务重启期间 DR 任务处于 waiting_review | 标为 error 但 checkpoint 保留；error_message 提示可 resume；不清除 checkpoint。 |
| Ingest 子进程在服务重启后启动 | `_ingest_subprocess_target` 检测到 job 状态为 error 后直接退出，不覆盖状态，不写入 collection。 |
| Ingest 单文件临时错误（如网络） | 当前阶段重试最多 `max_retries` 次后仍失败再记 file_error。 |
| 优雅关闭 | 收到 SIGTERM 后先等待 Scholar 后台任务最多 `graceful_shutdown_timeout_seconds`，再取消 worker 并关闭 adapter。 |
