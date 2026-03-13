# 开发指南

本文档面向后端/前端开发者，覆盖模块职责、代码约定、扩展路径与关键接口说明。

---

## 1. 开发环境快速启动

```bash
# 激活环境
conda activate deepsea-rag

# 启动基础设施（Milvus + Redis + PostgreSQL）
docker compose --profile dev up -d

# 启动后端（开发模式，热重载）
uvicorn src.api.server:app --host 0.0.0.0 --port 9999 --reload

# 启动前端（开发模式，热重载）
cd frontend && npm run dev
```

访问地址：
- 前端：`http://localhost:5173`
- 后端 API：`http://localhost:9999`
- API 文档（Swagger）：`http://localhost:9999/docs`

---

## 2. 模块职责边界

### 2.1 API 层（`src/api/`）

- **职责**：接收 HTTP 请求、参数验证、任务提交、SSE 流式响应
- **约定**：
  - 路由处理函数只做参数组装与任务委托，不含业务逻辑
  - 所有流式接口均返回 `EventSourceResponse`，包含 `id:` 字段
  - 请求参数通过 Pydantic schema 验证，禁止直接访问 `request.json()`

### 2.2 检索层（`src/retrieval/`）

- **职责**：提供 `RetrievalService.search()` 统一检索入口，屏蔽 local/web/hybrid 差异
- **约定**：
  - 所有进入 Section Evidence Pool 的检索必须使用 `pool_only=True`
  - `fuse_pools_with_gap_protection()` 是唯一的跨池融合入口，不允许在外部手动截断 fused pool
  - Web 检索使用软等待（soft-wait），禁止改回硬超时 with-block

### 2.3 LLM 层（`src/llm/`）

- **职责**：多 provider 统一调度，对上层提供 `call_llm(provider, model, messages, ...)` 接口
- **约定**：
  - 新增 provider 只需在 `config/rag_config.json` 的 `llm.platforms` 中添加 key/url，并在 `llm.providers` 中注册逻辑 name
  - Thinking 模型通过 `params` 字段注入特殊参数（如 `reasoning_effort`、`budget_tokens`），不影响主流调用路径
  - 工具定义（tools.py）保持 OpenAI function calling 格式，Anthropic/其他 provider 由 LLMManager 负责格式转换

### 2.4 Research Agent（`src/collaboration/research/agent.py`）

- **职责**：LangGraph 图定义、节点实现、Section Evidence Pool 管理
- **约定**：
  - 新增节点必须遵守 pool_source 命名规范（见架构文档 §3.4）
  - 进入章节池的检索统一经 `_accumulate_section_pool()` 追加，带 pool_source 标签
  - 节点函数不直接调用 LLM，通过 `call_llm(state["request"].llm_provider, ...)` 统一调用
  - LangGraph state 使用 TypedDict，新增字段必须在 state schema 中声明

### 2.5 任务层（`src/tasks/`）

- **职责**：Redis 队列、任务状态持久化、并发槽位管理
- **约定**：
  - Chat/DR 使用全局槽位（`max_active_slots`），Scholar 不占用全局槽位
  - Ingest 在子进程中运行，不允许直接向子进程内存写状态（通过 DB 通信）
  - 取消逻辑通过轮询取消标志实现（`_is_cancel_requested()`），不使用强制 kill

---

## 3. 检索证据流转详解

### 3.1 Chat 证据流

```
step_top_k（UI 传入或 fallback）
    ↓ 1.2 倍软放大 → chat_effective_step_top_k
    ↓
主检索（pool_only=True → 原始 RRF 候选池）
    ↓
[可选] gap 补搜（pool_only=True → 暂存 chat_gap_candidates_hits）
    ↓
§5¾ 统一 BGE Rerank（必须执行，仅此一次）
    main_pool + gap_pool → write_top_k 条（gap 0.2 保护）
    ↓
[可选] agent 工具检索 → agent_extra_chunks
    ↓（仅 agent_extra_chunks 非空时）
Agent 追加 BGE Rerank
    pack.chunks（含 gap）+ agent_extra_chunks → write_top_k 条（agent 0.1 保护）
    gap 传 []（禁止重传！）
    ↓
EvidenceSynthesizer → context_str → LLM 生成
```

**关键不变式**：
- §5¾ 必须在 agent 调用前执行，agent 在 context_str 基础上使用工具
- agent_extra_chunks 只含 §5¾ 之后才产生的新 chunk
- gap 在 agent 追加 fusion 时传 `[]`，否则双重计数

### 3.2 Deep Research 证据流

每个章节独立维护 Section Evidence Pool，三池按 pool_source 区分：

```
research_node 主检索 ──────────────────────── main pool (research_round)
research_node agent 补搜 ───────────────────── agent pool (agent_supplement)
evaluate_node GAP 补搜 ─────────────────────── gap pool (eval_supplement) ← 受 0.2 保护
write_node 兜底补搜 ─────────────────────────── main pool (write_stage)
review_revise_agent 补证 ───────────────────── agent pool (revise_supplement) ← 受 0.25 保护

                        ↓ write_node
        _rerank_section_pool_chunks(write_top_k)
        三池 fuse：gap 0.2 + agent 0.25 保护
                        ↓
             write_chunks（write_top_k 条）
             verify_chunks（verification_k 条）
```

### 3.3 write_top_k 推导逻辑

```python
# 优先级：UI 显式传入 > step_top_k 推导 > preset 基线
if ui_write_top_k > 0:
    effective = max(preset_write_k, ui_write_top_k)
elif ui_step_top_k > 0:
    effective = max(preset_write_k, int(ui_step_top_k * 1.5))
else:
    effective = preset_write_k

# search_top_k_write_max 硬 cap（已接线）
if cap > 0:
    effective = min(effective, max(cap, preset_write_k))
```

### 3.4 verification_k 动态计算

```python
verification_k = max(15, ceil(write_top_k * 0.25))
```

不再使用 preset 固定值（`lite=12, comprehensive=16` 已废弃）。

---

## 4. 提示词工程约定

### 4.1 提示词文件位置

所有提示词模板存放在 `src/prompts/` 目录：

| 文件 | 用途 |
|------|------|
| `chat_gap_queries.txt` | Chat 证据不足时生成 gap query |
| `review_revise_integrate.txt` | Review 阶段章节整合重写 |
| `agent.py` 内联 | Deep Research 各节点系统提示 |

### 4.2 提示词编写原则

1. **明确角色**：开头声明 AI 角色（研究助手、文献综述专家等）
2. **结构化输出**：需要 JSON 输出的提示词必须在 schema 定义和示例中说明格式，并在代码中做 JSON 修复（`json_repair`）
3. **引用占位符**：章节写作提示词中的引文使用 `[ref:xxxx]` 占位，不允许内联完整引用文本
4. **Token 预算意识**：写作节点的提示词不能把上下文窗口压满，必须为 review 阶段补证和整合预留空间
5. **中英文**：面向 LLM 的系统提示以英文为主（效果更好），面向用户的 UI 提示通过 i18n 本地化

### 4.3 新增/修改提示词时必须同步

- 更新本文档（提示词文件列表）
- 更新 `docs/architecture.md`（若涉及架构级数据流变化）
- 若改动影响节点行为，更新 `docs/chat_research_workflow_contract.md`

---

## 5. LLM 工具扩展

### 5.1 新增 Agent 工具

1. 在 `src/llm/tools.py` 中添加工具函数（OpenAI function calling 格式）
2. 在工具函数定义字典中注册
3. 在工具 handler（`handle_tool_call()`）中添加分支处理
4. 若工具创建外部 session（如 aiohttp），**必须**在 `try/finally` 中关闭

```python
# 正确模式：aiohttp session 必须关闭
async def _search_and_close(searcher, query):
    try:
        return await searcher.search(query)
    finally:
        await searcher.close()
```

### 5.2 新增 LLM Provider

1. 在 `config/rag_config.json` 的 `llm.platforms` 中添加平台配置
2. 在 `llm.providers` 中注册逻辑 provider name
3. 若平台 API 格式与 OpenAI 不兼容，在 `llm_manager.py` 中添加格式适配
4. Thinking 模型通过 `params` 字段注入，不需要修改调用逻辑

### 5.3 `run_code` 安全约束

`run_code` 默认关闭，受 `tool_execution.run_code_enabled` 控制。即使开启，也不是通用脚本执行入口——防御层次如下：

| 层次 | 机制 |
|---|---|
| AST 预检 | 禁止危险函数调用（`eval`/`exec`/`open`/`getattr` 等）、禁止导入白名单之外的模块、禁止 `__dunder__` 属性访问、禁止相对导入 |
| 运行时隔离 | `_safe_import` 运行时再次验证导入；`__builtins__` 替换为最小白名单字典 |
| 子进程隔离 | Python `-I -S -B` flags；仅注入两个必要环境变量；`cwd` 为临时目录；`stdin=DEVNULL`；新会话（`start_new_session=True`） |
| 资源限制（POSIX/Linux） | `RLIMIT_CPU`、`RLIMIT_AS`/`RLIMIT_DATA`、`RLIMIT_NOFILE=32`、`RLIMIT_NPROC` 软限制=8、`RLIMIT_CORE=0` |
| 双熔断 | 墙钟超时 + 输出字节上限，任一触发则 `SIGKILL` 整个进程组 |
| 并发控制 | `threading.Semaphore(max_concurrent)`，超限立即拒绝，不排队 |

**macOS 注意**：`RLIMIT_AS`/`RLIMIT_DATA` 在 macOS 上不被内核强制，内存限制无效，启动时会打印 WARNING。测试/开发可以在 macOS 上进行，但生产建议 Linux。

**扩展白名单时**：如需添加 `allowed_modules`，须评估该模块是否有文件 I/O 能力（如 `io.FileIO`）、网络能力或 `ctypes` 绑定，这些模块不应加入白名单。

若面向外部用户开放，优先使用独立容器/沙箱（如 gVisor、Firecracker），不要直接依赖宿主机 Python 进程隔离。

---

## 6. 数据库与迁移

### 6.1 ORM 框架

- 使用 **SQLModel + Alembic** 管理 schema
- 正式数据库后端统一为 PostgreSQL
- ORM 模型定义在各模块的 `*_store.py` 文件中

### 6.2 添加新字段

```bash
# 生成迁移脚本
alembic revision --autogenerate -m "add_xxx_field"

# 检查生成的迁移脚本（必须人工审核）
# 应用迁移
alembic upgrade head
```

> **约束**：迁移脚本必须是幂等的（`IF NOT EXISTS`）。生产环境迁移前必须备份数据库。

### 6.3 Milvus Collection

Milvus collection schema 定义在 `src/indexing/embedder.py`。修改 schema 需要：
1. drop 并重建 collection
2. 重新 ingest 所有文档

---

## 7. 前端开发约定

### 7.1 状态管理

- 全局状态使用 **Zustand**（`frontend/src/stores/`）
- API 返回的任务状态以后端为准（唯一真相源），前端不做乐观更新
- SSE 连接通过 `streamSSEResumable()`（`frontend/src/api/sse.ts`）统一管理，内含指数退避重连

### 7.2 SSE 客户端使用规范

```typescript
// 标准用法
const cleanup = streamSSEResumable({
  url: `/api/chat/stream/${taskId}`,
  afterId: lastEventId,           // 断点续传
  onEvent: (event) => { ... },
  onError: (err) => { ... },
  onDone: () => { ... },
});

// 组件卸载时必须清理
useEffect(() => cleanup, []);
```

### 7.3 国际化

- 所有用户可见文本通过 `i18next` 管理（`frontend/src/i18n/locales/`）
- 新增 UI 文本必须同时在 `en.json` 和 `zh.json` 中添加
- 后端错误消息通过 error code 映射到前端 i18n key

### 7.4 API 客户端

- 所有 API 调用通过 `frontend/src/api/` 下的模块封装，不允许直接调用 axios
- 流式接口使用 `sse.ts`，非流式接口使用各模块的 axios 封装

---

## 8. 测试

### 8.1 运行测试

```bash
# 运行全部测试
pytest

# 运行特定测试文件
pytest tests/test_agent_tool_step_topk.py -v

# 运行特定测试并显示日志
pytest tests/test_llm_manager_claude.py -v -s
```

### 8.2 测试类型

| 目录/文件 | 类型 | 说明 |
|----------|------|------|
| `tests/test_agent_tool_step_topk.py` | 单元 | Agent 工具 top_k 传参验证 |
| `tests/test_chat_agent_refusion.py` | 集成 | Chat agent 追加融合路径 |
| `tests/test_llm_manager_*.py` | 集成 | LLM provider 调用验证 |
| `tests/test_research_pool_only.py` | 集成 | Deep Research pool_only 路径 |
| `tests/test_service_pool_only.py` | 单元 | RetrievalService pool_only 参数 |

### 8.3 Mock 模式

配置 `llm.dry_run: true` 可开启 Mock LLM 模式，测试时不消耗 API token。

---

## 9. 常见扩展场景

### 9.1 添加新的 Web 检索来源

1. 在 `src/retrieval/unified_web_search.py` 中添加新的 searcher 类
2. 在 `unified_web_searcher.search_sync()` 中并行调用
3. 确保结果格式与 `EvidenceChunk` 兼容
4. 在 `config/rag_config.json` 中添加对应配置块

### 9.2 添加新的 Canvas 操作

1. 在 `src/collaboration/canvas/` 中实现操作逻辑
2. 在 `routes_canvas.py` 中添加 API 端点
3. 在 `frontend/src/api/canvas.ts` 中添加前端封装
4. 在 `frontend/src/components/` 中添加 UI 组件

### 9.3 调整检索参数默认值

修改顺序（优先级从高到低）：
1. UI 参数（前端 ChatInput 组件的默认值）
2. `config/rag_config.json` 中的 `search` 块
3. `src/retrieval/service.py` 中的代码默认值

修改后必须验证：
- `fuse_pools_with_gap_protection()` 的 quota 保护仍然生效
- `verify_light_threshold` / `verify_severe_threshold` 的比例关系未被破坏

---

## 10. 日志系统

### 10.1 架构概览

日志系统由三个协作层组成：

```
前端（logger.ts）
     │ X-Correlation-ID 请求头
     ▼
FastAPI CorrelationMiddleware  ←── 生成/透传 correlation_id，注入 ContextVar
     │
     ▼
后端 LogManager（分层路由）
  ├─ src.api.*        → logs/api/YYYY-MM-DD.log
  ├─ src.llm.*        → logs/llm/YYYY-MM-DD.log
  ├─ src.collaboration.* → logs/agent/YYYY-MM-DD.log
  ├─ src.retrieval.*  ┐
  │  src.indexing.*   ├─ logs/service/YYYY-MM-DD.log
  │  src.chunking.*   │
  │  src.parser.*     ┘
  └─ 其他 src.*       → logs/system/YYYY-MM-DD.log

  所有 ERROR+         → logs/error/YYYY-MM-DD.log（跨层聚合）
```

每条日志包含自动注入的 `correlation_id`，可将同一 HTTP 请求在 api / llm / service 多个日志文件中的记录串联起来：

```
2026-03-13 10:23:45.123 | INFO     | req-a1b2c3d4            | src.api.routes_chat | chat request received
2026-03-13 10:23:45.201 | INFO     | req-a1b2c3d4            | src.llm.llm_manager | provider=openai model=gpt-4o
2026-03-13 10:23:46.834 | INFO     | req-a1b2c3d4            | src.retrieval.service | retrieved 12 chunks
```

### 10.2 后端：添加日志

```python
# 统一导入方式（所有后端模块）
from src.log import get_logger

logger = get_logger(__name__)   # 模块路径自动决定路由到哪个日志文件

# 使用
logger.info("操作说明 %s", value)
logger.warning("[模块] 警告内容: %s", detail)
logger.error("[模块] 错误: %s", err, exc_info=True)
```

**命名约定：**
- 使用 `__name__` 作为 logger 名称，保持与模块路径一致
- 禁止直接使用 `import logging` + `logging.getLogger(__name__)`
- 合法例外：`log_manager.py`、`debug_logger.py`、`log_utils.py`、`aiohttp_tls_patch.py`（早期启动工具）、需要 `logging.Logger` 类型注解的文件

**在业务代码中携带 correlation_id（无需手动传递）：**

```python
# correlation_id 由 ContextVar 自动注入，logger 会自动读取
# 如需在非请求上下文（如后台任务）中手动设置：
from src.log import set_correlation_id, reset_correlation_id

def my_background_job(job_id: str):
    token = set_correlation_id(f"job-{job_id[:8]}")
    try:
        logger.info("开始处理后台任务") # 自动带上 job-xxx
        # ...业务逻辑...
    finally:
        reset_correlation_id(token) # 必须重置，防止线程池上下文泄露
```

### 10.3 前端：添加日志

```typescript
import { logger } from '../../utils/logger';   // 按实际相对路径调整

// API 调用层（fetch / SSE 事件）
logger.api.debug('[ComponentName] POST /chat/submit', { url, bodyLen });
logger.api.error('[ComponentName] fetch failed', err);

// 组件交互层（UI 事件、状态变化）
logger.ui.debug('[ComponentName] user action', { detail });
logger.ui.error('[ComponentName] operation failed', err);

// 跨域 error（始终输出）
logger.error('Unhandled exception', err);
```

**作用域选择：**

| 作用域 | 适用场景 |
|--------|----------|
| `logger.api` | `fetch`、`axios`、SSE 流、任何 HTTP 请求 |
| `logger.ui` | 用户点击、表单提交、组件状态变更、弹窗逻辑 |
| `logger.error` | 顶层未捕获异常 |

**环境行为：**
- 开发环境（`import.meta.env.DEV`）：`debug` 及以上全部打印
- 生产环境：仅打印 `warn` 及以上；`error` 级别自动上报 `POST /api/logs/frontend`

**日志工具路径（按文件深度）：**

| 文件位置 | 导入路径 |
|----------|----------|
| `src/pages/` | `'../utils/logger'` |
| `src/components/**/` | `'../../utils/logger'` |
| `src/components/**/**/` | `'../../../utils/logger'` |

### 10.4 关键文件速查

| 文件 | 作用 |
|------|------|
| `src/log/log_manager.py` | LogManager：分层路由、线程安全的自定义跨天 `DailyFileHandler`、error 聚合。高性能 O(N) 自动清理。|
| `src/log/context.py` | ContextVar：`get/set/reset/new_correlation_id()`。**重要**：使用 `set_correlation_id` 会返回 `Token`，在后台任务或并发环境必须用 `finally` 块调用 `reset_correlation_id(token)` 避免线程池上下文泄露串台。 |
| `src/log/__init__.py` | 统一导出 |
| `src/api/middleware/correlation.py` | FastAPI 中间件：生成/透传 `X-Correlation-ID`，并负责 `set/reset` 上下文 Token |
| `frontend/src/utils/logger.ts` | 前端统一 logger |

### 10.5 常见问题

**Q: 如何跨日志文件查询同一请求？**

```bash
grep "req-a1b2c3d4" logs/api/2026-03-13.log logs/llm/2026-03-13.log logs/service/2026-03-13.log
```

**Q: 运行时临时调高日志级别？**

```bash
# 环境变量（重启生效）
RAG_LOG_LEVEL=DEBUG uvicorn src.api.server:app ...

# 或 API 热切换（仅影响 debug logger）
curl -X POST /debug/toggle -d '{"enabled": true}'
```

**Q: 添加新模块时日志路由规则是什么？**

模块前缀到日志目录的映射定义在 `src/log/log_manager.py` 的 `_LAYER_ROUTING` 列表中。如需为新的顶级子包添加路由，在该列表追加 `("src.新包名", "目录名")` 即可，无需修改其他代码。

---

## 11. 代码提交规范

- 提交信息格式：`<type>(<scope>): <description>`
  - type：`feat / fix / refactor / docs / test / chore`
  - 示例：`feat(research): add review_revise_integrate node`
- 修改检索流程核心逻辑时，必须同步更新 `docs/chat_research_workflow_contract.md`
- 修改 API 接口时，必须同步更新 `docs/api_reference.md`
- 修改配置项时，必须同步更新 `docs/configuration.md` 和 `config/rag_config.example.json`
