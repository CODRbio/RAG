# 系统架构

本文档描述 DeepSea RAG 的真实代码架构与核心数据流，便于开发、排障和二次扩展。

## 一、总体分层

```text
Frontend (React + Zustand + i18n)
  → API Layer (FastAPI routes_*.py)
    → Collaboration Layer (workflow / canvas / memory / research / intent / citation / export)
    → Agent Layer (llm_manager + tools + react_loop)
    → Retrieval Layer (hybrid + web + graph + rerank)
  → Persistence / Infra (Milvus + SQLModel/Alembic + SQLite(rag.db) + files + observability)
```

## 二、后端模块映射（`src/`）

### 接口层

| 模块 | 职责 |
|---|---|
| `api/server.py` | FastAPI 应用入口，路由注册，生命周期管理 |
| `api/routes_chat.py` | 聊天、意图检测、Deep Research 全部接口 |
| `api/routes_canvas.py` | 画布 CRUD、大纲、草稿、快照、AI 编辑、引用管理 |
| `api/routes_ingest.py` | 在线入库（上传/Collections/任务管理） |
| `api/routes_compare.py` | 多文档对比 |
| `api/routes_graph.py` | 图谱统计、实体、邻居、chunk 详情 |
| `api/routes_export.py` | 导出（Markdown） |
| `api/routes_auto.py` | 自动补全（一键综述） |
| `api/routes_auth.py` | 认证与用户管理 |
| `api/routes_project.py` | 项目管理（存档/删除） |
| `api/routes_models.py` | 模型状态与同步 |
| `api/schemas.py` | Pydantic 请求/响应模型 |

### LLM / Agent 层

| 模块 | 职责 |
|---|---|
| `llm/llm_manager.py` | 统一 LLM 客户端管理（多 provider、日志、指标、结构化输出、tool-calling） |
| `llm/tools.py` | 工具定义与路由 |
| `llm/react_loop.py` | ReAct Agent 循环 |
| `generation/llm_client.py` | 兼容层（旧 `call_llm()` 接口） |
| `generation/evidence_synthesizer.py` | 证据综合（时间线排序、来源分组、强度分级、交叉验证） |
| `generation/context_packer.py` | 上下文打包 |

### 协作层

| 模块 | 职责 |
|---|---|
| `collaboration/canvas/` | 画布管理（canvas_manager + canvas_store + models） |
| `collaboration/memory/session_memory.py` | 会话级记忆（历史、引用、滚动摘要） |
| `collaboration/memory/working_memory.py` | 画布级工作记忆 |
| `collaboration/memory/persistent_store.py` | 持久用户偏好 |
| `collaboration/intent/parser.py` | 意图解析（Chat vs Deep Research） |
| `collaboration/intent/commands.py` | 命令处理 |
| `collaboration/research/agent.py` | Deep Research LangGraph Agent（核心） |
| `collaboration/research/verifier.py` | 声明验证 |
| `collaboration/research/job_store.py` | 后台任务与事件存储 |
| `collaboration/research/dashboard.py` | 研究进度仪表盘 |
| `collaboration/research/trajectory.py` | 研究轨迹追踪 |
| `collaboration/workflow/` | 状态机（states + transitions + graph） |
| `collaboration/citation/manager.py` | 引用解析（[ref:xxxx] → cite_key） |
| `collaboration/citation/formatter.py` | 引用格式化 |
| `collaboration/export/formatter.py` | 导出格式化 |
| `collaboration/auto_complete.py` | 一键综述服务 |

### 检索层

| 模块 | 职责 |
|---|---|
| `retrieval/hybrid_retriever.py` | 混合检索（dense + sparse + graph + RRF 融合） |
| `retrieval/service.py` | 检索服务编排 |
| `retrieval/unified_web_search.py` | 统一网络检索聚合器 |
| `retrieval/web_search.py` | Tavily 搜索 |
| `retrieval/web_content_fetcher.py` | 网页内容提取 |
| `retrieval/google_search.py` | Google / Scholar 搜索 |
| `retrieval/ncbi_search.py` | NCBI 文献搜索 |
| `retrieval/semantic_scholar.py` | Semantic Scholar API |
| `retrieval/colbert_reranker.py` | ColBERT 重排 |
| `retrieval/query_optimizer.py` | 规则查询优化 |
| `retrieval/smart_query_optimizer.py` | LLM 查询优化 |
| `retrieval/evidence.py` | 证据数据结构（EvidenceChunk / EvidencePack） |
| `retrieval/dedup.py` | 去重与多样化 |

### 数据处理层

| 模块 | 职责 |
|---|---|
| `parser/pdf_parser.py` | PDF 解析（Docling，含表格/图片/公式提取与 LLM 增强） |
| `parser/claim_extractor.py` | 声明提取 |
| `chunking/chunker.py` | 结构化切块（section-aware，表格行切块，句级 overlap） |
| `indexing/embedder.py` | 文本向量化（BGE-M3 dense + sparse，BGE-Reranker） |
| `indexing/milvus_ops.py` | Milvus 向量数据库操作（schema v1/v2，hybrid search） |
| `indexing/paper_store.py` | 论文元数据持久化（SQLModel / `data/rag.db`） |
| `indexing/paper_metadata_store.py` | 扩展元数据存储 |
| `indexing/ingest_job_store.py` | 入库任务追踪 |
| `graph/entity_extractor.py` | 通用实体抽取器（GLiNER zero-shot / 规则 / LLM 三种后端，领域本体由 `config/ontology.json` 驱动） |
| `graph/hippo_rag.py` | HippoRAG 知识图谱（委托 EntityExtractor 抽取实体 + Personalized PageRank 检索 + 图谱持久化） |
| `graphs/ingestion_graph.py` | LangGraph 入库流水线 |
| `pipelines/ingestion_graph.py` | 兼容性桥接（转发至 `graphs/ingestion_graph`，保留旧导入路径） |

### 基础设施层

| 模块 | 职责 |
|---|---|
| `auth/session.py` | JWT 签发/校验与撤销校验（无状态认证） |
| `auth/password.py` | 密码哈希（bcrypt） |
| `observability/metrics.py` | Prometheus 指标（LLM 调用计数/延迟/token 用量） |
| `observability/tracing.py` | OpenTelemetry 分布式追踪 |
| `observability/middleware.py` | FastAPI 可观测中间件 |
| `observability/setup.py` | 可观测性初始化 |
| `mcp/server.py` | MCP Server（对外暴露工具与资源） |
| `evaluation/runner.py` | 评测执行器 |
| `evaluation/metrics.py` | 评测指标 |
| `evaluation/dataset.py` | 评测数据集 |
| `utils/cache.py` | TTL 缓存 |
| `utils/limiter.py` | 并发限流 |
| `utils/storage_cleaner.py` | 存储清理 |
| `utils/prompt_manager.py` | 提示词模板管理 |
| `utils/token_counter.py` | Token 预算估算（tiktoken cl100k_base；含模型上下文窗口注册表与安全预算计算） |
| `utils/task_runner.py` | 后台任务运行器 |
| `utils/model_sync.py` | 模型同步 |
| `log/log_manager.py` | 统一日志管理 |
| `prompts/` | LLM 提示词模板文件（API/检索/解析/协作/评测等），与业务代码解耦 |

## 三、前端结构（`frontend/src/`）

### 页面层

- `pages/ChatPage.tsx`：主聊天界面
- `pages/IngestPage.tsx`：文档入库
- `pages/LoginPage.tsx`：登录认证
- `pages/AdminPage.tsx`：管理后台

### 组件层

| 目录 | 内容 |
|---|---|
| `components/chat/` | ChatWindow、ChatInput、ToolTracePanel、RetrievalDebugPanel |
| `components/canvas/` | CanvasPanel、ExploreStage、OutlineStage、DraftingStage、RefineStage、StageStepper、FloatingToolbar |
| `components/compare/` | CompareView |
| `components/graph/` | GraphExplorer |
| `components/workflow/` | DeepResearchDialog、DeepResearchSettingsPopover、WorkflowStepper、CommandPalette、IntentModeSelector、IntentConfirmPopover |
| `components/research/` | ResearchProgressPanel |
| `components/settings/` | SettingsModal |
| `components/layout/` | Header、Sidebar |
| `components/ui/` | Modal、Toast、PdfViewerModal |

### 状态管理

`stores/`：Zustand 状态管理（useChatStore、useCanvasStore、useConfigStore、useAuthStore、useProjectsStore、useCompareStore、useUIStore、useToastStore）

### 其他

- `api/`：后端接口封装（chat、canvas、compare、graph、ingest、models、auth、projects、auto、health）
- `types/`：TypeScript 类型定义
- `i18n/`：国际化（en.json、zh.json）

## 四、核心数据流

### 1) 离线入库流

```text
raw PDF
 → scripts/02_parse_papers.py
 → src/parser/pdf_parser.py（Docling + LLM 增强）
 → scripts/03_index_papers.py
 → src/chunking/chunker.py（section-aware 切块）
 → src/indexing/embedder.py（BGE-M3 dense + sparse）
 → src/indexing/milvus_ops.py（写入 Milvus）
 → (optional) scripts/03b_build_graph.py
 → src/graph/entity_extractor.py（读取 config/ontology.json，GLiNER / 规则 / LLM 抽取实体）
 → src/graph/hippo_rag.py（构建知识图谱 + Personalized PageRank 索引）
```

### 2) 在线入库流

```text
POST /ingest/upload
 → routes_ingest.py
 → PDF 解析 + 切块 + 向量化 + 写入 Milvus
 → 任务状态追踪（job_store）
 → SSE 事件流上报进度
```

### 3) 在线聊天检索流

```text
POST /chat or /chat/stream
 → routes_chat.py
 → (route decision: chat / rag)
 → build_search_query_from_context()
    - reference gate: 无指代时仅使用当前问题
    - context resolve: 存在指代时用 rolling_summary 解析主语
    - post-validation: 关键词重叠校验，否则回退原问题
 → RetrievalService.search()
    → HybridRetriever（dense + sparse + RRF）
    → UnifiedWebSearcher（Tavily + Google + Scholar + NCBI + Semantic Scholar）
    → HippoRAG（图检索，条件触发）
 → Evidence 综合 + context packing
 → LLMManager client.chat()
 → response / SSE events
```

### 4) Agent ReAct 流

```text
react_loop(messages, tools)
 → llm_manager.chat(tools=...)
 → parse_tool_calls(...)
 → execute_tool_call(...)（检索/画布/图谱/对比等工具）
 → append tool result message
 → next iteration / final_text
```

### 5) Deep Research 流（LangGraph + 后台任务）

```text
Phase 1: /deep-research/start
  Scope → Plan → 返回 brief + outline（前端可确认/编辑）

Phase 2: /deep-research/submit（推荐）
  提交后台任务 → 返回 job_id
  前端轮询 /deep-research/jobs/{job_id} 与 /events
  任务在后端持续运行（不依赖前端连接）

  研究循环（per section）：
    Research → Evaluate → (gap? → Research) → Write → Verify
    - Recall + Precision 双类查询
    - 分层 search_top_k
    - 3 级验证分流
    - 收益曲线早停

  Drafting 审核门：
    - 用户审核各章节（approve / revise）
    - 支持缺口补充（material / direct_info）
    - 审核门指数退避 + 早停

  最终整合：
    Synthesize → Abstract → Limitations → Open Gaps → Global Refine → Citation Guard

兼容模式: /deep-research/confirm（SSE 直连）
```

### Deep Research 前端交互

- **配置前置化**：输入区 `⚙` 设置弹窗，持久化到 `useConfigStore + localStorage`
  - 配置项：`depth`、`outputLanguage`、`stepModels`、`stepModelStrict`
- **启动链路**：澄清问题优先使用 `scope` 步骤模型
- **Drafting 审核区**：
  - 通过 / 修改 / 重新确认 / 一键全部通过并触发整合
  - 章节级缺口补充弹窗
- **运行期监测**：
  - Research Monitor（graph steps、成本状态、coverage 曲线、效率评分）
  - 低效率提示与章节优化提示词模板
- **人工介入**：
  - `user_context`（supporting / direct_injection）
  - 临时材料上传（pdf/md/txt → 不写入持久库）

### Research Depth 双级别

| | lite | comprehensive |
|---|---|---|
| 定位 | 快速但学术可用 | 全面学术综述 |
| 预计耗时 | ~5-15 min | ~20-60 min |
| 迭代预算 | 3 × 章节数 | 6 × 章节数 |
| 每章研究轮次 | max 3 | max 5 |
| 覆盖度阈值 | 0.60 | 0.80 |
| 查询策略 | 2 recall + 2 precision + gaps | 4 recall + 4 precision + gaps |
| search_top_k（首轮/补缺/写作） | 18 / 10 / 8 | 30 / 15 / 10 |
| 写作二次取证 `verification_k` | 12 | 16 |
| 验证（轻/中/重） | 20% / 40% / 45% | 15% / 30% / 35% |
| 审核门（指数退避 + 早停） | 80 轮 / 8 次无变化 | 200 轮 / 12 次无变化 |
| LangGraph 递归上限 | 200 | 500 |
| 成本预警 / 强制摘要步数 | 120 / 180 | 300 / 420 |

阈值定义在 `src/collaboration/research/agent.py` 的 `DEPTH_PRESETS`，可通过 `config/rag_config.json` → `deep_research.depth_presets` 覆盖。

### 循环防护机制

- **动态迭代预算**：`max_iterations = max_iterations_per_section × num_sections`
- **每章研究轮次上限**：`max_section_research_rounds`
- **Self-Correction**：第 3 轮及以后，coverage 达标时自动衰减 `search_top_k_gap`
- **收益曲线早停**：最近两轮 coverage 增益低且接近目标时提前进入写作
- **审核门指数退避 + 早停**：sleep 从 2s 指数增长；连续 N 轮无变化自动放行
- **3 级验证分流**：只有"严重"级才触发回滚
- **LangGraph 递归限制**：编译时显式设置 `recursion_limit`
- **成本步数监控**：达到 `cost_warn_steps` 提示人工介入；达到 `cost_force_summary_steps` 强制摘要
- **成本心跳上报**：每 `cost_tick_interval` 步发出 `cost_monitor_tick`

### 最终整合链路（Synthesize + Global Refine）

1. 生成 `Abstract`
2. 基于 insights + evidence-scarce sections 生成 `Limitations and Future Directions`
3. 聚合 open gaps 生成 `Open Gaps and Future Research Agenda`
4. **Token 预算估算**（`src/utils/token_counter.py`）：使用 tiktoken 精确计算拼接后文档的 token 数，与模型上下文窗口对比，决定下一步策略
5. **全篇连贯性重写**（双路策略）：
   - **Path A — Single-pass**：文档 token 充足时，单次调用 `coherence_refine.txt`，`max_tokens` 由安全预算动态计算（替代原硬编码 3500）
   - **Path B — Sliding Window**：文档过长时，逐章节润色：
     - 每个窗口携带 **Document Blueprint**（话题 + 摘要 + 全章节大纲 + 当前位置标记 `>> [CURRENT] <<`，~500–800 token，零额外 LLM 调用）
     - 本地上下文：前章末尾 ~300 token + 后章预览 ~150 token
     - Prompt 模板：`coherence_refine_window.txt`
     - 每章完成后上报 `coherence_window_done` 进度事件
6. 引用保护（citation guard）：对全文合并结果检测引用/证据标签丢失，任一路径失败均自动回退到整合前版本
7. 写回最终结果并切换 Canvas 到 `refine` 阶段

> **策略选择事件**：`coherence_strategy_selected`（含 `strategy`、`body_tokens`、`prompt_tokens`、`context_window` 字段），可在前端研究进度面板或日志中观察。

## 五、存储与状态

| 存储 | 用途 |
|---|---|
| Milvus | 向量索引与 chunk 检索 |
| SQLite (`data/rag.db`) | 统一业务数据库（会话、画布、用户、项目、任务、元数据等 21 张表） |
| SQLModel + Alembic | ORM 与版本化 schema 迁移（支持后续平滑切换 PostgreSQL） |
| 文件系统 | `data/raw_papers`、`data/parsed`、`artifacts`、`logs` |
| NetworkX (in-memory + JSON) | HippoRAG 知识图谱 |
| 会话记忆 | `rolling_summary` / `summary_at_turn` 用于跨轮主语解析 |

## 六、可观测性

| 端点 | 功能 |
|---|---|
| `GET /metrics` | Prometheus 指标导出 |
| `GET /health` | 基础健康检查 |
| `GET /health/detailed` | 组件级健康状态（检索/LLM/图谱等） |
| `GET /storage/stats` | 存储统计 |
| OTel | 通过环境变量启用 tracing |
| LangSmith | 可选集成（LangGraph 执行追踪） |

## 七、设计约束

- LLM 调用必须通过 `src/llm/llm_manager.py`
- Prompt 必须通过 `src/utils/prompt_manager.py` 从 `src/prompts/*.txt` 加载，避免业务代码硬编码多行提示词
- 实体类型定义必须放 `config/ontology.json`，禁止在 `entity_extractor.py` 或 `hippo_rag.py` 中硬编码领域正则或实体列表
- 常规场景使用 `PromptManager.render()`；需要延迟格式化时使用 `PromptManager.load()`
- 新工具必须在 `src/llm/tools.py` 与 `src/mcp/server.py` 同步注册
- 配置新增字段必须同步 `config/rag_config.json` 与 `config/rag_config.example.json`
- 敏感配置放 `config/rag_config.local.json` 或环境变量
- 依赖方向：上层调用下层，避免反向耦合
- 新 API 路由统一放 `src/api/routes_*.py`，在 `server.py` 注册

## 八、Prompt 资产流

```text
src/prompts/*.txt
  → src/utils/prompt_manager.py
    → render(template, **kwargs)    # 直接渲染
    → load(template)                # 读取原始模板，后续再 format
      → 业务模块（routes_* / retrieval / parser / collaboration / evaluation / graph）
```

这一路径保证：

- 模板可独立于 Python 逻辑快速调优
- 多模块共享统一提示词资产与缓存机制
- 变更审查时可清晰区分“业务逻辑改动”与“提示词改动”

### 关键模板速查（Deep Research 最终整合）

| 模板文件 | 用途 | 使用场景 |
|---|---|---|
| `coherence_refine.txt` | 全篇连贯性重写（单次调用） | Path A：文档 token 充足 |
| `coherence_refine_window.txt` | 单章节连贯性润色（含 Document Blueprint） | Path B：滑动窗口，每章一次调用 |
| `generate_abstract.txt` | 生成摘要 | `synthesize_node` |
| `limitations_section.txt` | 生成不足与未来方向 | `synthesize_node` |
| `open_gaps_agenda.txt` | 生成开放问题与研究议程 | `synthesize_node` |
| `write_section.txt` | 写作单个章节 | `write_node` |
| `verify_claims.txt` | 声明验证 | `verifier.py` |
| `generate_claims.txt` | 提取核心声明 | `generate_claims_node` |
