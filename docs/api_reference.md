# API 参考

本文档基于 `src/api/routes_*.py` 与 `src/api/server.py` 的实际路由整理。

更新时间：2026-02-19

## 基础信息

- 默认服务地址：`http://127.0.0.1:9999`
- Swagger：`/docs`
- OpenAPI：`/openapi.json`

## 全局与健康

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 基础健康检查 |
| `GET` | `/health/detailed` | 组件级健康状态（检索/LLM/图谱等） |
| `GET` | `/storage/stats` | 存储统计 |
| `GET` | `/metrics` | Prometheus 指标导出 |

## 认证与用户（`/auth`、`/admin`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/auth/login` | 用户登录，返回 token |
| `POST` | `/admin/users` | 管理员创建用户（需 admin token） |
| `GET` | `/admin/users` | 管理员查看用户列表（需 admin token） |

### 鉴权说明

- 需要鉴权的接口使用 Header：`Authorization: Bearer <token>`
- 普通用户与管理员权限由路由层依赖项判断

## 项目管理（`/projects`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/projects` | 项目列表（可选 `include_archived`） |
| `POST` | `/projects/{canvas_id}/archive` | 存档 |
| `POST` | `/projects/{canvas_id}/unarchive` | 取消存档 |
| `DELETE` | `/projects/{canvas_id}` | 删除项目 |

## 聊天与会话

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/chat` | 同步对话 |
| `POST` | `/chat/stream` | SSE 流式对话 |
| `POST` | `/intent/detect` | 意图检测（Chat vs Deep Research） |
| `GET` | `/sessions` | 会话列表 |
| `GET` | `/sessions/{session_id}` | 会话详情 |
| `DELETE` | `/sessions/{session_id}` | 删除会话 |

### `chat/stream` SSE 事件

- `meta`：元信息（session_id 等）
- `dashboard`：仪表盘数据
- `tool_trace`：工具调用轨迹
- `delta`：流式文本增量
- `done`：完成

## Deep Research

### 推荐后台任务模式

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/deep-research/clarify` | 生成澄清问题（1-6 个关键问题） |
| `POST` | `/deep-research/context-files` | 上传临时上下文文件（pdf/md/txt），仅本次使用 |
| `POST` | `/deep-research/start` | Phase-1（Scope + Plan），返回可编辑 brief + outline |
| `POST` | `/deep-research/submit` | Phase-2 后台提交，返回 `job_id`（推荐） |
| `GET` | `/deep-research/jobs` | 列出后台任务（支持 `limit`、`status` 筛选） |
| `GET` | `/deep-research/jobs/{job_id}` | 查询单个任务状态与结果 |
| `GET` | `/deep-research/jobs/{job_id}/events` | 增量事件流（支持 `after_id`） |
| `POST` | `/deep-research/jobs/{job_id}/cancel` | 停止任务（前端关闭不会自动终止） |
| `POST` | `/deep-research/jobs/{job_id}/review` | 提交章节审核（approve / revise） |
| `GET` | `/deep-research/jobs/{job_id}/reviews` | 查看章节审核记录 |
| `POST` | `/deep-research/jobs/{job_id}/gap-supplement` | 提交章节缺口补充 |
| `GET` | `/deep-research/jobs/{job_id}/gap-supplements` | 查看补充记录（pending / consumed） |
| `GET` | `/deep-research/jobs/{job_id}/insights` | 查看研究洞察 |
| `POST` | `/deep-research/jobs/{job_id}/insights/{insight_id}/status` | 更新洞察状态 |

### Resume Queue 运维

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/deep-research/resume-queue` | 查看 resume 队列（支持 status/owner/job 过滤） |
| `POST` | `/deep-research/resume-queue/cleanup` | 清理终态记录 |
| `POST` | `/deep-research/resume-queue/{resume_id}/retry` | 重试指定 resume 请求 |

### 兼容流式模式

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/deep-research/confirm` | SSE 直连执行（旧模式，仍可用） |

### Phase-2 关键请求字段

`POST /deep-research/submit` 与 `/deep-research/confirm` 共用 `DeepResearchConfirmRequest`：

**基础字段：**
- `topic`、`session_id`、`canvas_id`、`search_mode`
- `confirmed_outline`、`confirmed_brief`
- `output_language`、`step_models`、`step_model_strict`
- `skip_draft_review`、`skip_refine_review`
- 检索参数（`web_providers`、`local_top_k`、`final_top_k` 等）

**研究深度（`depth`）：**
- `lite`：快速探索（~5-15 min）
- `comprehensive`：全面学术综述（~20-60 min，默认）

**人工介入字段：**
- `user_context`：用户补充观点/约束文本
- `user_context_mode`：`supporting`（补充上下文）| `direct_injection`（强提示直接注入）
- `user_documents`：`[{name, content}]` 临时材料（来自 `/deep-research/context-files`）

**`step_model_strict` 行为：**
- `false`（默认）：步骤模型解析失败时自动回退默认模型
- `true`：步骤模型解析失败时立即终止任务

### 人工审核字段

`POST /deep-research/jobs/{job_id}/review`：

```json
{
  "section_id": "string",
  "action": "approve | revise",
  "feedback": "optional string"
}
```

### 缺口补充字段

`POST /deep-research/jobs/{job_id}/gap-supplement`：

```json
{
  "section_id": "string",
  "gap_text": "string",
  "supplement_type": "material | direct_info",
  "content": {"text": "用户补充内容"}
}
```

### Resume Queue 运维字段

`POST /deep-research/resume-queue/cleanup`：

```json
{
  "statuses": ["done", "error", "cancelled"],
  "before_hours": 72,
  "owner_instance": "optional",
  "job_id": "optional"
}
```

`POST /deep-research/resume-queue/{resume_id}/retry`：

```json
{
  "message": "optional retry note"
}
```

### Deep Research 事件类型

`GET /deep-research/jobs/{job_id}/events` 返回的事件：

| 事件 | 说明 |
|---|---|
| `start` | 任务启动 |
| `progress` | 阶段进展（含 section/type/message） |
| `warning` | 风险或覆盖不足提醒 |
| `section_review` | 用户提交章节审核 |
| `gap_supplement` | 用户提交缺口补充 |
| `cancel_requested` | 收到停止请求 |
| `done` | 任务完成 |
| `cancelled` | 任务取消完成 |
| `error` | 任务失败 |

**`progress` 子类型（`type` 字段）：**

| type | 说明 |
|---|---|
| `evidence_insufficient` | 章节证据不足 |
| `section_degraded` | 章节降级写作 |
| `search_self_correction` | 补缺阶段自校正 |
| `coverage_plateau_early_stop` | 覆盖收益趋平，提前停止 |
| `section_evaluate_done` | 章节评估结果（含 coverage/gain/round/steps） |
| `write_verification_context` | 写作阶段二次取证 |
| `all_reviews_approved` | 所有章节审核通过 |
| `global_refine_done` | 全文连贯性整合完成 |
| `citation_guard_fallback` | 引用保护回退 |
| `step_model_resolved` | 步骤模型解析成功 |
| `step_model_fallback` | 步骤模型回退默认 |
| `cost_monitor_tick` | 成本监控心跳 |
| `cost_monitor_warn` | 成本预警 |
| `cost_monitor_force_summary` | 强制摘要模式 |

### Research Depth Presets

#### Iteration & Coverage

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `max_iterations_per_section` | 3 | 6 | 每章迭代预算 |
| `max_section_research_rounds` | 3 | 5 | 每章最大研究轮次 |
| `coverage_threshold` | 0.60 | 0.80 | 覆盖度达标阈值 |

#### Query Strategy

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `recall_queries_per_section` | 2 | 4 | 广撒网查询 |
| `precision_queries_per_section` | 2 | 4 | 定向深钻查询 |

#### Tiered search_top_k

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `search_top_k_first` | 18 | 30 | 首轮广撒网 |
| `search_top_k_gap` | 10 | 15 | 补缺定点搜索 |
| `search_top_k_write` | 8 | 10 | 写作前精选 |

#### 3-tier Verification

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `verify_light_threshold` | 0.20 | 0.15 | 轻微：仅标记 |
| `verify_medium_threshold` | 0.40 | 0.30 | 中等：记录 gaps |
| `verify_severe_threshold` | 0.45 | 0.35 | 严重：回到 research |

#### Review Gate

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `review_gate_max_rounds` | 80 | 200 | 最大轮询轮数 |
| `review_gate_base_sleep` | 2s | 2s | 初始等待 |
| `review_gate_max_sleep` | 15s | 20s | 单轮等待上限 |
| `review_gate_early_stop_unchanged` | 8 | 12 | 无变化 N 轮后放行 |

#### LangGraph & Cost

| 参数 | lite | comprehensive | 说明 |
|---|---|---|---|
| `recursion_limit` | 200 | 500 | 递归上限 |
| `cost_warn_steps` | 120 | 300 | 成本预警步数 |
| `cost_force_summary_steps` | 180 | 420 | 强制摘要步数 |

> 阈值可在 `config/rag_config.json` → `deep_research.depth_presets` 自定义覆盖。

## Canvas（`/canvas`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/canvas` | 创建画布 |
| `GET` | `/canvas/{canvas_id}` | 获取画布 |
| `PATCH` | `/canvas/{canvas_id}` | 更新画布基础字段 |
| `DELETE` | `/canvas/{canvas_id}` | 删除画布 |
| `POST` | `/canvas/{canvas_id}/outline` | 更新大纲 |
| `POST` | `/canvas/{canvas_id}/drafts` | 更新草稿 |
| `POST` | `/canvas/{canvas_id}/snapshot` | 创建快照 |
| `POST` | `/canvas/{canvas_id}/restore/{version_number}` | 恢复快照 |
| `GET` | `/canvas/{canvas_id}/snapshots` | 获取快照列表 |
| `GET` | `/canvas/{canvas_id}/export` | 按画布导出 |
| `POST` | `/canvas/{canvas_id}/refine-full` | 全文精炼 |
| `GET` | `/canvas/{canvas_id}/citations` | 引用列表 |
| `POST` | `/canvas/{canvas_id}/citations/filter` | 引用过滤 |
| `DELETE` | `/canvas/{canvas_id}/citations/{cite_key}` | 删除引用 |
| `POST` | `/canvas/{canvas_id}/ai-edit` | AI 段落编辑 |

## 导出（`/export`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/export` | 导出入口（支持 `canvas_id` 或 `session_id`，当前仅 Markdown） |

## Auto Complete

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/auto-complete` | 一键综述 |

## Compare 多文档对比（`/compare`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/compare` | 对比 2-5 篇论文 |
| `GET` | `/compare/candidates` | 会话引文候选 |
| `GET` | `/compare/papers` | 本地文库分页/搜索 |

## Graph 图谱（`/graph`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/graph/stats` | 图谱统计 |
| `GET` | `/graph/entities` | 实体查询 |
| `GET` | `/graph/neighbors/{entity_name}` | 实体邻居 |
| `GET` | `/graph/chunk/{chunk_id}` | chunk 详情 |
| `GET` | `/graph/pdf/{paper_id}` | PDF 原文访问 |

## Ingest 在线入库（`/ingest`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/ingest/collections` | 集合列表 |
| `POST` | `/ingest/collections` | 创建集合 |
| `DELETE` | `/ingest/collections/{name}` | 删除集合 |
| `GET` | `/ingest/collections/{name}/papers` | 集合内论文列表 |
| `DELETE` | `/ingest/collections/{name}/papers/{paper_id}` | 删除论文 |
| `POST` | `/ingest/upload` | 上传文件 |
| `POST` | `/ingest/process` | 触发处理 |
| `GET` | `/ingest/jobs` | 任务列表 |
| `GET` | `/ingest/jobs/{job_id}` | 任务详情 |
| `POST` | `/ingest/jobs/{job_id}/cancel` | 取消任务 |
| `GET` | `/ingest/jobs/{job_id}/events` | 任务事件流 |

## Models 模型管理

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/models/status` | 模型加载状态 |
| `POST` | `/models/sync` | 同步模型 |
| `GET` | `/llm/providers` | 可用 LLM provider 列表 |
