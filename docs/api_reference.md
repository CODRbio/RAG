# API 参考（按当前代码）

本文档基于 `src/api/routes_*.py` 与 `src/api/server.py` 的实际路由整理。

## 基础信息

- 默认服务地址：`http://127.0.0.1:9999`
- Swagger：`/docs`
- OpenAPI：`/openapi.json`

## 全局与健康

- `GET /health`：基础健康检查
- `GET /storage/stats`：存储统计
- `GET /metrics`：Prometheus 指标
- `GET /health/detailed`：组件级健康状态

## 认证与用户

### Auth

- `POST /auth/login`：用户登录，返回 token

### Admin

- `POST /admin/users`：管理员创建用户（需 admin token）
- `GET /admin/users`：管理员查看用户列表（需 admin token）

## 项目管理

- `GET /projects`：项目列表（可选 `include_archived`）
- `POST /projects/{canvas_id}/archive`：项目存档
- `POST /projects/{canvas_id}/unarchive`：取消存档
- `DELETE /projects/{canvas_id}`：删除项目

## 聊天与会话

- `POST /chat`：同步对话
- `POST /chat/stream`：SSE 流式对话
- `POST /intent/detect`：意图检测
- `GET /sessions`：会话列表
- `GET /sessions/{session_id}`：会话详情
- `DELETE /sessions/{session_id}`：删除会话

### Deep Research（推荐后台任务模式）

- `POST /deep-research/clarify`
  - 生成澄清问题（返回 `1-6` 个关键问题，主题不明确时会问更多）。
- `POST /deep-research/context-files`
  - 上传临时上下文文件（`pdf/md/txt`），提取文本后返回：
  - `documents: [{name, content}]`
  - 仅用于本次 Deep Research，不写入持久知识库。
- `POST /deep-research/start`
  - Phase-1（Scope + Plan），返回可编辑 `brief + outline`。
- `POST /deep-research/submit`（新增，推荐）
  - Phase-2 后台提交，立即返回 `job_id`。
  - 后续通过 jobs 接口轮询状态/事件。
- `GET /deep-research/jobs`
  - 列出后台任务（支持 `limit`、`status`）。
- `GET /deep-research/jobs/{job_id}`
  - 查询单个任务状态与结果摘要。
- `GET /deep-research/jobs/{job_id}/events`
  - 拉取任务事件（支持 `after_id` 增量读取）。
- `POST /deep-research/jobs/{job_id}/cancel`
  - 显式停止任务；除非调用此接口，否则前端断开/关闭不会自动终止任务。
- `POST /deep-research/jobs/{job_id}/review`
  - 提交章节审核结论（`approve | revise`）。
  - 支持重复提交（“重新确认”会覆盖上一条同章节记录）。
- `GET /deep-research/jobs/{job_id}/reviews`
  - 查看当前任务所有章节审核记录。
- `GET /deep-research/resume-queue`
  - 查看 resume 队列（支持按状态/实例/job 过滤）。
- `POST /deep-research/resume-queue/cleanup`
  - 清理 resume 队列（默认清理终态记录）。
- `POST /deep-research/resume-queue/{resume_id}/retry`
  - 手动重试某条 resume 请求（仅终态可重试）。
- `POST /deep-research/jobs/{job_id}/gap-supplement`
  - 提交章节级缺口补充（观点或材料线索）。
- `GET /deep-research/jobs/{job_id}/gap-supplements`
  - 查看补充记录及状态（`pending | consumed`）。
- `GET /deep-research/jobs/{job_id}/insights`
  - 查看研究洞察（`gap/conflict/limitation/future_direction`）。
- `POST /deep-research/jobs/{job_id}/insights/{insight_id}/status`
  - 更新洞察状态（`open | addressed | deferred`）。

### Deep Research（兼容流式模式）

- `POST /deep-research/confirm`
  - 旧的 SSE 直连执行模式，仍可用；新前端默认走 `/submit`。

### `chat/stream` SSE 事件

- `meta`
- `dashboard`
- `tool_trace`
- `delta`
- `done`

### `deep-research/confirm` SSE 事件

- `meta`
- `progress`
- `warning`
- `dashboard`
- `delta`
- `done`

### `deep-research/jobs/{job_id}/events` 事件

- `start`：任务启动
- `progress`：阶段进展（含 section/type/message）
- `warning`：风险或覆盖不足提醒
- `section_review`：用户提交章节审核（approve/revise）
- `gap_supplement`：用户提交章节缺口补充
- `progress(type=evidence_insufficient)`：章节证据不足（已触发自适应补搜但仍稀缺）
- `progress(type=section_degraded)`：章节降级写作（避免在弱证据下生成过度确定结论）
- `progress(type=search_self_correction)`：补缺阶段触发自校正，自动降低 `search_top_k_gap`
- `progress(type=coverage_plateau_early_stop)`：覆盖收益曲线趋平，提前停止继续补搜
- `progress(type=section_evaluate_done)`：章节评估结果（含 `coverage`、`coverage_gain`、`research_round`、`graph_steps`）
- `progress(type=write_verification_context)`：写作阶段加载二次取证窗口（`verification_k`）
- `progress(type=all_reviews_approved)`：所有章节审核通过，进入最终整合
- `progress(type=global_refine_done)`：已完成全文连贯性整合（跨章节一致性优化）
- `progress(type=citation_guard_fallback)`：整合后引用/证据标签疑似丢失，自动回退到安全版本
- `progress(type=step_model_resolved)`：步骤模型解析成功
- `progress(type=step_model_fallback)`：步骤模型解析失败并回退默认模型
- `progress(type=cost_monitor_tick)`：成本监控心跳（每 N 步上报当前 graph steps）
- `progress(type=cost_monitor_warn)`：图步数达到成本预警阈值，建议人工介入或缩小范围
- `progress(type=cost_monitor_force_summary)`：图步数过高，进入强制摘要模式
- `cancel_requested`：收到停止请求
- `done`：任务完成
- `cancelled`：任务取消完成
- `error`：任务失败

### Deep Research Phase-2 关键请求字段

`POST /deep-research/submit` 与 `POST /deep-research/confirm` 共用 `DeepResearchConfirmRequest`：

- 基础字段：
  - `topic`、`session_id`、`canvas_id`、`search_mode`
  - `confirmed_outline`、`confirmed_brief`
  - `output_language`、`step_models`、`step_model_strict`
  - `skip_draft_review`、`skip_refine_review`
  - 检索参数（`web_providers`、`local_top_k`、`final_top_k` 等）
- 研究深度（`depth`）：
  - `lite`：快速探索模式（~3-10 min）
  - `comprehensive`：全面学术综述模式（~15-40 min，**默认**）
  - 详见下方 **Research Depth Presets** 表格
- 人工介入字段：
  - `user_context`：用户补充观点/约束文本
  - `user_context_mode`：`supporting | direct_injection`
    - `supporting`：作为补充上下文
    - `direct_injection`：作为强提示直接注入（高优先级）
  - `user_documents`：`[{name, content}]` 临时材料文本（来自 `/deep-research/context-files`）

#### `step_model_strict` 行为

- `false`（默认）：
  - 某步骤指定模型解析失败时，自动回退到默认模型继续执行（任务不中断）。
- `true`：
  - 某步骤指定模型解析失败时，立即抛错并终止任务（严格保证指定模型生效）。

### 人工审核与缺口补充字段

#### `POST /deep-research/jobs/{job_id}/review`

- 请求体：
  - `section_id: string`
  - `action: "approve" | "revise"`
  - `feedback?: string`（`revise` 时建议填写）

#### `POST /deep-research/jobs/{job_id}/gap-supplement`

- 请求体：
  - `section_id: string`
  - `gap_text: string`
  - `supplement_type: "material" | "direct_info"`
  - `content: object`（推荐 `{"text": "...用户补充..."}`）

### Resume Queue 运维（查看 / 清理 / 重试）

用于后台任务的“审核后恢复（resume）”运维排障。

#### `GET /deep-research/resume-queue`

- Query 参数：
  - `limit`（默认 `50`，最大 `500`）
  - `status`（可选：`pending|running|done|error|cancelled`）
  - `owner_instance`（可选：实例标识）
  - `job_id`（可选）
- 返回：
  - `items`：队列项列表
  - `count`：条数

示例：

```http
GET /deep-research/resume-queue?status=error&limit=20
```

#### `POST /deep-research/resume-queue/cleanup`

- 请求体：
  - `statuses?: string[]`（不传时默认 `["done","error","cancelled"]`）
  - `before_hours?: number | null`（默认 `72`；`null` 表示不按时间过滤）
  - `owner_instance?: string`
  - `job_id?: string`
- 返回：
  - `deleted`：删除条数

示例：

```json
{
  "statuses": ["done", "error", "cancelled"],
  "before_hours": 48
}
```

#### `POST /deep-research/resume-queue/{resume_id}/retry`

- 请求体（可选）：
  - `owner_instance?: string`（不传则使用当前 worker 实例）
  - `message?: string`
- 返回：
  - `item`：更新后的队列项（状态变为 `pending`）

示例：

```json
{
  "message": "manual retry after review fix"
}
```

常见状态码：

- `404`：`resume_id` 不存在
- `409`：该请求已在 `pending/running`，或同一 `job_id` 已有活跃 resume 请求

### Research Depth Presets

`depth` 字段控制 Deep Research 全部循环上限、搜索策略和验证行为。

#### Iteration & Coverage

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `max_iterations_per_section` | 3 | 6 | 每章迭代预算（全局上限 = 此值 × 章节数） |
| `max_section_research_rounds` | 3 | 5 | 每章最大研究轮次（research→evaluate 循环） |
| `coverage_threshold` | 0.60 | 0.80 | 覆盖度达标阈值（达标即进入写作） |

#### Query Strategy (recall + precision)

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `recall_queries_per_section` | 2 | 4 | 广撒网查询（短、宽、同义词/变体） |
| `precision_queries_per_section` | 2 | 4 | 定向深钻查询（长、带方法/时间/数据约束） |
| 总查询 = recall + precision + gap queries | ~4+ | ~8+ | |

#### Tiered search_top_k (按研究阶段变化)

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `search_top_k_first` | 18 | 30 | 首轮广撒网 |
| `search_top_k_gap` | 10 | 15 | 复核/补缺定点搜索 |
| `search_top_k_write` | 8 | 10 | 写作前精选"最可引用"证据 |

#### 3-tier Verification

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `verify_light_threshold` | 0.20 | 0.15 | < 此值 → 仅标记，不中断 |
| `verify_medium_threshold` | 0.40 | 0.30 | light..severe 区间 → 记录 gaps，不回滚 |
| `verify_severe_threshold` | 0.45 | 0.35 | > 此值 → 回到 research 全面补证据 |

#### Review Gate (指数退避 + 早停)

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `review_gate_max_rounds` | 80 | 200 | 最大轮询轮数 |
| `review_gate_base_sleep` | 2s | 2s | 初始等待（指数退避：2→4→8→…） |
| `review_gate_max_sleep` | 15s | 20s | 单轮等待上限 |
| `review_gate_early_stop_unchanged` | 8 | 12 | 连续无变化 N 轮后自动放行 |

#### LangGraph

| 参数 | lite | comprehensive | 说明 |
|------|------|--------------|------|
| `recursion_limit` | 200 | 500 | LangGraph 编译时递归上限 |

> 阈值可在 `config/rag_config.json` → `deep_research.depth_presets` 中自定义覆盖。

## Canvas

- `POST /canvas`：创建画布
- `GET /canvas/{canvas_id}`：获取画布
- `PATCH /canvas/{canvas_id}`：更新画布基础字段
- `DELETE /canvas/{canvas_id}`：删除画布
- `POST /canvas/{canvas_id}/outline`：更新大纲
- `POST /canvas/{canvas_id}/drafts`：更新草稿
- `POST /canvas/{canvas_id}/snapshot`：创建快照
- `POST /canvas/{canvas_id}/restore/{version_number}`：恢复快照
- `GET /canvas/{canvas_id}/export`：按画布导出
- `GET /canvas/{canvas_id}/citations`：引用列表
- `POST /canvas/{canvas_id}/citations/filter`：引用过滤
- `DELETE /canvas/{canvas_id}/citations/{cite_key}`：删除引用
- `POST /canvas/{canvas_id}/ai-edit`：AI 段落编辑

## 导出

- `POST /export`：导出入口（支持 `canvas_id` 或 `session_id`）

说明：当前实现仅支持 `markdown` 格式。

## Auto Complete

- `POST /auto-complete`：一键综述

## Compare（多文档对比）

- `POST /compare`：对比 2-5 篇论文
- `GET /compare/candidates`：会话引文候选
- `GET /compare/papers`：本地文库分页/搜索

## Graph

- `GET /graph/stats`：图谱统计
- `GET /graph/entities`：实体查询
- `GET /graph/neighbors/{entity_name}`：实体邻居
- `GET /graph/chunk/{chunk_id}`：chunk 详情

## Ingest（在线入库）

- `GET /ingest/collections`
- `POST /ingest/collections`
- `DELETE /ingest/collections/{name}`
- `GET /ingest/collections/{name}/papers`
- `DELETE /ingest/collections/{name}/papers/{paper_id}`
- `POST /ingest/upload`
- `POST /ingest/process`
- `GET /ingest/jobs`
- `GET /ingest/jobs/{job_id}`
- `POST /ingest/jobs/{job_id}/cancel`
- `GET /ingest/jobs/{job_id}/events`

## Models

- `GET /models/status`
- `POST /models/sync`
- `GET /llm/providers`

## 鉴权说明

- 需要鉴权的接口使用 Header：
  - `Authorization: Bearer <token>`
- 普通用户与管理员权限由后端依赖项在路由层判断。
