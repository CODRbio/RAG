# API 参考文档

本文档描述 DeepSea RAG 后端所有 HTTP/SSE 接口，按功能模块分组。

完整的交互式 API 文档（Swagger UI）可访问：`http://localhost:9999/docs`

---

## 通用说明

### 认证

所有接口（除 `/auth/login` 外）均需 JWT 认证：

```
Authorization: Bearer <token>
```

### 通用响应格式

```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功"
}
```

错误响应：
```json
{
  "detail": "错误描述",
  "code": "ERROR_CODE"
}
```

### SSE 响应格式

流式接口返回 Server-Sent Events：

```
id: 42
event: chunk
data: {"type": "text", "content": "生成内容..."}

id: 43
event: done
data: {"status": "completed"}
```

客户端重连时传 `?after_id=42` 或 `Last-Event-ID: 42` 实现断点续传。

---

## 1. 认证（`/auth`）

### POST /auth/login

登录并获取 JWT Token。

**请求体**：
```json
{
  "username": "admin",
  "password": "password"
}
```

**响应**：
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### POST /auth/logout

撤销当前 Token（写入 `revoked_tokens` 表）。

---

## 2. 对话（`/chat`）

### POST /chat/stream

提交 Chat 流式请求，返回任务 ID。

**请求体（ChatRequest）**：

```json
{
  "session_id": "uuid",
  "message": "继续展开上面第二点",
  "search_mode": "hybrid",
  "llm_provider": "qwen-thinking",
  "intent_provider": "intent-local",
  "ultra_lite_provider": "openai-mini",
  "model_override": "qwen3.5-plus",
  "local_top_k": 45,
  "step_top_k": 10,
  "write_top_k": 15,
  "reranker_mode": "bge_only",
  "agent_mode": "assist",
  "graphic_abstract_model": "nanobanana 2"
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `search_mode` | string | 检索模式：`local` / `web` / `hybrid` / `none` |
| `llm_provider` | string | 主回答 / 主写作模型 |
| `intent_provider` | string | 意图判断专用模型，仅用于意图检测、多轮追问与上下文复用判断 |
| `ultra_lite_provider` | string | 长文本压缩等超轻量任务模型 |
| `local_top_k` | int | 本地召回上限（推荐 45） |
| `step_top_k` | int | 单次检索输出上限（推荐 10） |
| `write_top_k` | int | 进 LLM 的证据上限（推荐 15） |
| `reranker_mode` | string | `bge_only` / `colbert_only` / `cascade` |
| `agent_mode` | string | `standard` / `assist` / `autonomous` |
| `graphic_abstract_model` | string | 图文摘要图片模型 |

**模型字段职责**：

- `llm_provider`：主回答模型
- `intent_provider`：Chat 意图/追问判断模型；未显式传入时，默认跟随 `llm_provider` 的 lite 版
- `ultra_lite_provider`：压缩与超轻量任务模型
- `graphic_abstract_model`：图像生成模型

**响应**：
```json
{
  "task_id": "uuid",
  "status": "queued"
}
```

### GET /chat/stream/{task_id}

订阅 Chat 流式响应（SSE）。

**查询参数**：
- `after_id`（可选）：从指定事件 ID 后重放

**SSE 事件类型**：

| event | data 格式 | 说明 |
|-------|----------|------|
| `chunk` | `{"type": "text", "content": "..."}` | 生成的文本片段 |
| `citations` | `[{"id": "...", "title": "...", "authors": [...], ...}]` | 引文列表 |
| `evidence` | `[{"chunk_id": "...", "text": "...", "score": 0.9, ...}]` | 检索证据 |
| `diag` | `{"phase": "...", "pool_fusion": {...}, ...}` | 诊断信息 |
| `done` | `{"status": "completed"}` | 任务完成 |
| `error` | `{"message": "...", "code": "..."}` | 错误信息 |

### GET /chat/sessions

获取当前用户的 Chat 会话列表。

### DELETE /chat/sessions/{session_id}

删除指定会话。

### POST /tasks/{task_id}/cancel

取消 Chat 任务（与 Scholar 共用）。

---

## 3. Deep Research（`/deep-research`）

### POST /deep-research/clarify

提交研究主题，获取澄清问题。

**请求体**：
```json
{
  "topic": "深海热液喷口生物群落的演化机制",
  "history": [],
  "llm_provider": "claude",
  "use_preliminary_knowledge": true
}
```

**响应**：
```json
{
  "clarification_questions": [
    "您希望重点关注哪个时间尺度（地质年代还是近现代）？",
    "是否需要包括冷泉生态系统的对比？"
  ],
  "preliminary_knowledge": "热液喷口生态系统..."
}
```

### POST /deep-research/submit

提交 Deep Research 任务（用户已回答澄清问题并确认大纲后调用）。

**请求体（DeepResearchRequest）**：
```json
{
  "topic": "深海热液喷口生物群落的演化机制",
  "clarification_answers": ["关注近现代（百年尺度）", "是，需要冷泉对比"],
  "confirmed_brief": { ... },
  "confirmed_outline": ["引言", "热液喷口理化环境", "微生物群落", "宏生物群落", "结论"],
  "depth": "comprehensive",
  "llm_provider": "claude",
  "filters": {
    "step_top_k": 20,
    "write_top_k": null
  },
  "skip_draft_review": false,
  "canvas_id": null
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `depth` | string | `lite` / `comprehensive`（控制研究深度预算） |
| `filters.step_top_k` | int | 每轮检索上限；影响 write_top_k 推导 |
| `filters.write_top_k` | int | 显式覆盖写作证据上限；null 则由 step_top_k 推导 |
| `skip_draft_review` | bool | 跳过章节审核门控 |

**响应**：
```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

### GET /deep-research/jobs/{job_id}/stream

订阅 Deep Research 进度流（SSE）。

**SSE 事件类型**：

| event | data 格式 | 说明 |
|-------|----------|------|
| `status` | `{"status": "planning", "phase": "scoping", ...}` | 阶段状态更新 |
| `section_start` | `{"section": "引言", "round": 1}` | 章节开始 |
| `section_done` | `{"section": "引言", "text": "..."}` | 章节完成 |
| `review_gate` | `{"sections": [...], "job_id": "..."}` | 等待用户审核 |
| `heartbeat` | `{"ts": 1710000000}` | 心跳（约 10s 间隔） |
| `done` | `{"job_id": "...", "canvas_id": "..."}` | 全文完成 |
| `error` | `{"message": "...", "code": "..."}` | 错误 |

### GET /deep-research/jobs/{job_id}

查询任务状态与进度。

**响应**：
```json
{
  "job_id": "uuid",
  "status": "running",
  "phase": "research",
  "current_section": "微生物群落",
  "sections_done": 2,
  "sections_total": 5,
  "canvas_id": "uuid",
  "created_at": "2026-03-11T10:00:00Z",
  "updated_at": "2026-03-11T10:15:00Z"
}
```

### POST /deep-research/jobs/{job_id}/review

提交章节审核结果。

**请求体**：
```json
{
  "action": "revise",
  "section": "微生物群落",
  "feedback": "需要补充更多关于嗜热古菌的内容",
  "author_notes": "请重点参考 2020 年后的文献"
}
```

`action` 可选值：`approve` / `revise` / `skip`

### POST /deep-research/jobs/{job_id}/cancel

取消 Deep Research 任务。

### GET /deep-research/jobs

列出当前用户的 Deep Research 任务。

---

## 4. 文档入库（`/ingest`）

### POST /ingest/upload

上传 PDF 文件（multipart form-data）。

**请求**：
```
Content-Type: multipart/form-data
file: [binary PDF data]
project_id: uuid（可选）
```

**响应**：
```json
{
  "file_id": "uuid",
  "filename": "paper.pdf",
  "size": 1024000
}
```

### POST /ingest/process

提交 Ingest 任务（处理已上传的文件）。

**请求体**：
```json
{
  "file_ids": ["uuid1", "uuid2"],
  "project_id": "uuid",
  "options": {
    "ocr": false,
    "enrich_tables": true,
    "enrich_figures": true
  }
}
```

**响应**：
```json
{
  "job_id": "uuid",
  "status": "pending",
  "file_count": 2
}
```

### GET /ingest/jobs/{job_id}/events

订阅 Ingest 任务进度（SSE）。

**SSE 事件类型**：

| event | data 格式 | 说明 |
|-------|----------|------|
| `progress` | `{"file": "paper.pdf", "stage": "parsing", "percent": 45}` | 进度更新 |
| `file_done` | `{"file": "paper.pdf", "chunks": 42, "status": "done"}` | 单文件完成 |
| `heartbeat` | `{"ts": 1710000000}` | 心跳（约 5s） |
| `done` | `{"job_id": "...", "total_chunks": 84}` | 全部完成 |
| `error` | `{"file": "paper.pdf", "message": "..."}` | 错误 |

### GET /ingest/jobs/{job_id}

查询 Ingest 任务状态。

### POST /ingest/jobs/{job_id}/cancel

取消 Ingest 任务。

### GET /ingest/documents

列出已入库的文档。

**查询参数**：
- `project_id`：按项目过滤
- `page` / `page_size`：分页

### DELETE /ingest/documents/{doc_id}

删除已入库文档（同时从 Milvus 删除向量）。

---

## 5. 学术搜索（`/scholar`）

### POST /scholar/search

搜索学术文献。

**请求体**：
```json
{
  "query": "deep sea hydrothermal vent microbial community",
  "providers": ["semantic_scholar", "google_scholar"],
  "max_results": 20,
  "year_from": 2015,
  "year_to": 2026
}
```

**响应**：
```json
{
  "results": [
    {
      "title": "...",
      "authors": ["Author A", "Author B"],
      "year": 2023,
      "doi": "10.xxxx/xxx",
      "abstract": "...",
      "pdf_url": "...",
      "source": "semantic_scholar"
    }
  ],
  "total": 47
}
```

### POST /scholar/download

下载单篇论文 PDF。

**请求体**：
```json
{
  "doi": "10.xxxx/xxx",
  "pdf_url": "https://...",
  "title": "Paper Title",
  "ingest_after_download": true,
  "project_id": "uuid"
}
```

**响应**：
```json
{
  "task_id": "uuid",
  "status": "queued"
}
```

### POST /scholar/download/batch

批量下载论文 PDF。

**请求体**：
```json
{
  "papers": [
    {"doi": "10.xxx/1", "title": "..."},
    {"doi": "10.xxx/2", "title": "..."}
  ],
  "ingest_after_download": true,
  "project_id": "uuid",
  "max_concurrent": 3
}
```

### GET /scholar/task/{task_id}/stream

订阅下载/推荐任务进度（SSE，同 Chat 取消共用 `/tasks/{task_id}/cancel`）。

**SSE 事件类型**：

| event | 说明 |
|-------|------|
| `progress` | 下载进度 |
| `paper_done` | 单篇完成 |
| `captcha` | 需要验证码解决（自动处理，仅信息事件） |
| `done` | 全部完成 |
| `error` | 错误 |

### GET /scholar/libraries

获取当前用户的学术文献库列表。

### POST /scholar/libraries

创建学术文献库。

### POST /scholar/libraries/{lib_id}/recommend/start

启动文献推荐任务（基于已有文献推荐相关论文）。

**请求体**：
```json
{
  "max_recommendations": 20,
  "year_from": 2020
}
```

### POST /scholar/libraries/{lib_id}/ingest

将文献库中的 PDF 批量入库。

---

## 6. Canvas（`/canvas`）

### GET /canvas

列出所有 Canvas 文档。

### POST /canvas

创建 Canvas 文档。

**请求体**：
```json
{
  "title": "研究综述草稿",
  "content": "# 引言\n...",
  "project_id": "uuid"
}
```

### GET /canvas/{canvas_id}

获取 Canvas 文档内容。

### PUT /canvas/{canvas_id}

更新 Canvas 文档。

### DELETE /canvas/{canvas_id}

删除 Canvas 文档。

### POST /canvas/{canvas_id}/ai-edit

AI 辅助编辑（扩写/缩写/润色/引文插入）。

**请求体**：
```json
{
  "action": "expand",
  "selection": "热液喷口的理化环境极端。",
  "instruction": "请扩写这段内容，补充具体的温度、压力、化学成分数据",
  "llm_provider": "claude"
}
```

`action` 可选值：`expand` / `condense` / `refine` / `insert_citations`

---

## 7. 学术助手（`/academic-assistant`）

### POST /academic-assistant/papers/summary

单篇论文精读总结。

**请求体**：
```json
{
  "locator": {
    "paper_uid": "doi:10.1000/example"
  },
  "scope": {
    "scope_type": "collection",
    "scope_key": "reef"
  },
  "question": "重点说明图像和实验结果"
}
```

**响应重点字段**：

- `paper_card`
- `summary_md`
- `citations`
- `evidence_summary`

### POST /academic-assistant/papers/qa

针对单篇论文做定向问答。输出结构与 summary 类似，但正文字段为 `answer_md`。

### POST /academic-assistant/papers/compare

按 `paper_uid` 做多篇论文对比，返回：

- `papers`
- `comparison_matrix`
- `narrative`
- `citations`
- `evidence_summary`

### POST /academic-assistant/media-analysis/start

手动触发论文图片解析回填任务。

**请求体**：
```json
{
  "paper_uids": ["doi:10.1000/example"],
  "scope": {
    "scope_type": "collection",
    "scope_key": "reef"
  },
  "force_reparse": false,
  "upsert_vectors": true
}
```

说明：

- 成功后会回写 `enriched.json`
- 同时增量 upsert `image_caption` / `image_analysis` 向量行

### POST /academic-assistant/discovery/{mode}/start

异步发现任务统一入口，`mode` 取值：

- `missing_core`
- `forward_tracking`
- `experts`
- `institutions`

### POST /academic-assistant/annotations

写入或更新标注。

**请求体重点字段**：

- `resource_type`
- `resource_id`
- `paper_uid`
- `target_kind`：`chunk | figure | page_region | canvas_section`
- `target_locator`
- `target_text`
- `directive`
- `status`

### GET /academic-assistant/annotations

按 `paper_uid / resource_type / resource_id / target_kind / status` 过滤查询标注。

### GET /academic-assistant/task/{task_id}

查询 academic assistant 异步任务状态。

### GET /academic-assistant/task/{task_id}/stream

订阅 academic assistant 异步任务 SSE。

**SSE 事件类型**：

- `progress`
- `heartbeat`
- `done`
- `error`
- `cancelled`

---

## 8. 通用资源状态（`/resources`）

### GET /resources/state

读取单个资源的用户态覆盖信息。

**查询参数**：

- `resource_type`
- `resource_id`

### PATCH /resources/state

写入或更新用户态覆盖信息。

**请求体**：
```json
{
  "resource_type": "project",
  "resource_id": "canvas-123",
  "archived": true,
  "favorite": true,
  "read_status": "reading"
}
```

说明：

- `project` 只是 API 别名，后端会规范化为 `canvas`
- `paper.resource_id` 固定使用 `paper_uid`
- `resource_annotation` 不支持该接口

### GET /resources/tags

列出某个资源的自由标签。

### POST /resources/tags

给资源新增一个 tag；重复 tag 会按规范化结果去重。

### DELETE /resources/tags

删除某个 tag。

### GET /resources/notes

列出某个资源的资源级 Markdown 笔记。

### POST /resources/notes

创建资源级 Markdown 笔记。

### PATCH /resources/notes/{note_id}

更新指定笔记。

### DELETE /resources/notes/{note_id}

删除指定笔记。

---

## 9. 多文档比较（`/compare`）

### POST /compare/start

启动多文档比较任务（支持 2-5 篇）。

**请求体**：
```json
{
  "doc_ids": ["uuid1", "uuid2", "uuid3"],
  "dimensions": ["方法论", "结论", "局限性"],
  "llm_provider": "claude"
}
```

**响应**：
```json
{
  "task_id": "uuid",
  "status": "queued"
}
```

### GET /compare/results/{task_id}

获取比较结果。

---

## 10. 项目管理（`/projects`）

### GET /projects

列出当前用户项目；返回字段中的 `archived` 已由通用资源态覆盖。

### POST /projects/{canvas_id}/archive

将项目归档。内部兼容写入 `resource_user_states`，并保持旧 `Canvas.archived` 双写。

### POST /projects/{canvas_id}/unarchive

取消项目归档。

### DELETE /projects/{project_id}

删除项目（不删除文档本身）。

---

## 11. 系统管理（`/admin`）

### GET /admin/users

列出用户（管理员权限）。

### POST /admin/users

创建用户。

### GET /admin/models

获取所有 provider 的可用模型列表（实时从平台 API 拉取）。

### GET /health

基础健康检查。

```json
{"status": "ok", "version": "8.0.0"}
```

### GET /health/detailed

详细健康检查（含各组件状态）。

```json
{
  "status": "ok",
  "components": {
    "database": "ok",
    "milvus": "ok",
    "redis": "ok",
    "llm_default": "ok"
  }
}
```

### GET /metrics

Prometheus Metrics 端点（OpenTelemetry 格式）。

---

## 10. 错误码参考

| HTTP 状态 | 错误码 | 说明 |
|-----------|--------|------|
| 400 | `INVALID_PARAMS` | 请求参数无效 |
| 401 | `UNAUTHORIZED` | 未认证或 Token 已过期 |
| 403 | `FORBIDDEN` | 无权限访问该资源 |
| 404 | `NOT_FOUND` | 资源不存在 |
| 409 | `CONFLICT` | 资源状态冲突（如任务已在运行） |
| 429 | `RATE_LIMITED` | 请求过于频繁 |
| 500 | `INTERNAL_ERROR` | 服务内部错误 |
| 503 | `SERVICE_UNAVAILABLE` | 依赖服务不可用（如 Milvus 未就绪） |

---

## 11. 前端 API 客户端对应关系

| 后端模块 | 前端文件 | 说明 |
|---------|---------|------|
| `/chat` | `frontend/src/api/chat.ts` | Chat + Deep Research |
| `/ingest` | `frontend/src/api/ingest.ts` | 入库任务 |
| `/scholar` | `frontend/src/api/scholar.ts` | 学术搜索下载 |
| `/canvas` | `frontend/src/api/canvas.ts` | Canvas 操作 |
| SSE 公共 | `frontend/src/api/sse.ts` | `streamSSEResumable()` |
| 模型列表 | `frontend/src/api/models.ts` | LLM 模型选择 |
