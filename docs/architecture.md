# 系统架构文档

本文档描述 DeepSea RAG 的整体架构、核心模块职责、关键数据流，以及各流程中必须遵守的设计约束。

---

## 1. 整体架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│                        前端 (React + Vite)                        │
│  ChatPage | ScholarPage | ResearchPage | CanvasPage | ComparePage │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼──────────────────────────────────────┐
│                  后端 API 层 (FastAPI + Uvicorn)                   │
│  routes_chat  routes_ingest  routes_scholar  routes_canvas  ...  │
└──────┬──────────────┬───────────────┬───────────────┬────────────┘
       │              │               │               │
┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐ ┌─────▼──────────┐
│   Chat/DR   │ │  Ingest   │ │   Scholar   │ │  Canvas/Compare │
│   Worker    │ │  Worker   │ │   Worker    │ │    (inline)     │
└──────┬──────┘ └─────┬─────┘ └──────┬──────┘ └────────────────┘
       │              │               │
┌──────▼──────────────▼───────────────▼──────────────────────────┐
│                       核心服务层                                  │
│  ┌──────────────┐  ┌───────────┐  ┌──────────────────────────┐ │
│  │RetrievalSvc  │  │LLMManager │  │  Deep Research Agent      │ │
│  │(hybrid search│  │(multi LLM │  │  (LangGraph stateful)     │ │
│  │ + rerank)    │  │ dispatch) │  │                          │ │
│  └──────────────┘  └───────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
       │              │               │
┌──────▼──────────────▼───────────────▼──────────────────────────┐
│                       基础设施层                                  │
│  Milvus (向量DB)  PostgreSQL (元数据)  Redis (队列/任务状态)      │
│  Playwright (浏览器自动化)  BGE Embedding / Reranker 模型         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
src/
├── api/
│   ├── server.py              # FastAPI 应用初始化、lifespan、中间件
│   ├── routes_chat.py         # Chat 流式 + Deep Research 提交/审核
│   ├── routes_ingest.py       # PDF 入库任务管理
│   ├── routes_scholar.py      # 学术搜索与下载
│   ├── routes_canvas.py       # Canvas 文档 CRUD + AI 编辑
│   ├── routes_compare.py      # 多文档比较
│   └── schemas.py             # Pydantic 请求/响应模型
├── llm/
│   ├── llm_manager.py         # LLM provider 统一调度层
│   ├── tools.py               # Agent 工具定义（search_local/web/scholar 等）
│   └── react_loop.py          # ReAct 循环实现
├── retrieval/
│   ├── service.py             # RetrievalService（混合检索主入口）
│   ├── hybrid_retriever.py    # Dense+Sparse RRF 融合
│   ├── unified_web_search.py  # 多 Web 来源聚合
│   ├── structured_queries.py  # 1+1+1 结构化查询生成
│   └── dedup.py               # 跨源去重
├── collaboration/
│   ├── research/
│   │   ├── agent.py           # LangGraph Deep Research 主引擎
│   │   └── job_store.py       # Deep Research 任务 DB 存储
│   ├── canvas/                # Canvas 文档模型与操作
│   ├── memory/                # Session/Working/Persistent Memory
│   ├── citation/              # 引文解析与格式化
│   └── intent/                # 意图检测与命令路由
├── indexing/
│   ├── ingest_job_store.py    # Ingest 任务管理
│   └── embedder.py            # 向量嵌入 pipeline
├── tasks/
│   ├── dispatcher.py          # 任务路由与执行
│   ├── redis_queue.py         # Redis 队列操作
│   └── task_state.py          # 任务状态追踪
└── utils/
    ├── task_runner.py         # 后台任务 runner（轮询执行）
    └── storage_cleaner.py     # 存储清理工具
config/
├── settings.py                # 配置加载（JSON + env 合并）
├── rag_config.json            # 主配置
└── rag_config.local.json      # 本地覆盖（gitignored）
frontend/src/
├── api/                       # Axios + SSE 客户端
├── components/                # React 组件
├── stores/                    # Zustand 状态管理
├── pages/                     # 页面路由
└── i18n/                      # 中英文本地化
```

---

## 3. 核心模块详解

### 3.1 检索服务（RetrievalService）

**文件**：`src/retrieval/service.py`

检索服务是整个 RAG 流程的证据入口，负责将用户查询转换为高质量的证据候选池。

#### 检索模式

| 模式 | 说明 |
|------|------|
| `local` | 仅本地向量库（Dense + Sparse + RRF） |
| `web` | 仅 Web 来源（Tavily + Scholar + Semantic Scholar + NCBI） |
| `hybrid` | 本地 + Web 并行，跨源去重后融合 |

#### 混合检索内部流程

```
RetrievalService.search(mode="hybrid", top_k=local_top_k, filters={step_top_k, ...})
    │
    ├─ [并行启动]
    │   ├─ HybridRetriever.retrieve(top_k=local_recall_k)
    │   │    ├─ Milvus Dense 召回（BGE-M3 embedding）
    │   │    ├─ Milvus Sparse 召回（BM25）
    │   │    ├─ 加权 RRF 融合（dense=0.6, sparse=0.4）
    │   │    └─ per_doc_cap=3 去重多样化
    │   └─ unified_web_searcher.search_sync(...)
    │        ├─ Tavily（网络搜索）
    │        ├─ Google Scholar（浏览器）
    │        ├─ Semantic Scholar API
    │        └─ NCBI PubMed API
    │
    ├─ 软等待（soft-wait）
    │   ├─ 本地硬超时：默认 60s
    │   └─ Web 软超时：min(timeout_s × 5, 300s)  ← Scholar 约 145s，不可设为硬超时
    │
    ├─ cross_source_dedup（过滤 web 中已在本地库的文献）
    │
    └─ 若 pool_only=False：fuse_pools_with_gap_protection → 截断到 result_limit
       若 pool_only=True：返回原始候选池（不截断，供上层统一 rerank）
```

#### 关键参数语义

| 参数 | 含义 | 取值规则 |
|------|------|---------|
| `local_top_k` | 传给 retriever 的召回上限 | UI 显式传入或配置默认值（推荐 45） |
| `step_top_k` | 单次检索输出上限（result_limit） | UI 传入；未传时继承 local_top_k |
| `write_top_k` | 单个产出单元进 LLM 的证据上限 | Chat 中等于 step_top_k；DR 中由 `_compute_effective_write_k()` 推导 |
| `local_recall_k` | 内部 hybrid 分支传给 retriever 的上限 | `min(actual_recall, max(result_limit × 2, 20))`；`result_limit ≥ 10` 时等于 result_limit |

> **约束**：`local_recall_k` 公式不可随意修改。hybrid 模式需要 local 分支提供充足候选供全局融合；若改为仅用 result_limit 会导致候选池过小。

#### 三池融合算法（fuse_pools_with_gap_protection）

这是整个系统最核心的证据选优算法，Chat 和 Deep Research 共用：

```
输入：main_candidates + gap_candidates + agent_candidates + top_k
      + gap_ratio / agent_ratio / rank_pool_multiplier

Step 1  按 chunk_id 去重（优先级：agent > gap > main）
Step 2  全局 BGE rerank
        rerank_k = min(max(ceil(top_k × multiplier), top_k + N_gap + N_agent), N_total)
Step 3  初始切片 top_k
Step 4  软配额保护
        gap_min_keep  = ceil(top_k × gap_ratio)    [Chat=0.2, Research=0.2]
        agent_min_keep = ceil(top_k × agent_ratio) [Chat=0.1, Research=0.25]
        若当前切片中 gap 数 < gap_min_keep，从 reranked tail 回填直至达标
        若 gap 池本身不足配额，effective_min_keep = min(配额, 实际 gap 数)
Step 5  剥除 pool_tag，返回 top_k 条
```

> **约束**：fuse 之后**不得再做分数阈值过滤**。quota 回填完成后再次过滤会破坏 gap/agent 保护。

---

### 3.2 LLM 管理器（LLMManager）

**文件**：`src/llm/llm_manager.py`

统一封装多 LLM provider，对上层屏蔽 provider 差异。

#### 支持的 Provider

| Provider 标识 | 平台 | 说明 |
|--------------|------|------|
| `openai` / `openai-thinking` / `openai-mini` | OpenAI | GPT 系列；thinking 版注入 `reasoning_effort` |
| `deepseek` / `deepseek-thinking` | DeepSeek | Chat + Reasoner |
| `gemini` / `gemini-thinking` / `gemini-vision` / `gemini-flash` | Google | Pro/Flash 系列；vision 版支持图像 |
| `claude` / `claude-thinking` / `claude-haiku` | Anthropic | Sonnet/Haiku；thinking 版注入 `budget_tokens` |
| `kimi` / `kimi-thinking` / `kimi-vision` | Moonshot | Kimi-K2；vision 版支持图像 |
| `sonar` | Perplexity | 网络搜索增强型；用于 prelim knowledge |
| `qwen` / `qwen-thinking` / `qwen-vision` | Alibaba | 通义千问 |

#### 调用优先级

```
UI 选择的 provider/model
  → config.llm.providers[name].default_model
  → config.llm.default（兜底 provider）
```

#### Thinking 模型注意事项

- Claude thinking：`budget_tokens` 控制思考预算，token 消耗显著高于普通模式
- DeepSeek thinking：`{"thinking": {"type": "enabled"}}` 注入 params
- 前端模型选择器会实时从平台 API 拉取可用模型列表，不依赖 config 中的静态列表

---

### 3.3 Agent 工具（tools.py）

**文件**：`src/llm/tools.py`

定义 Chat Agent 和 Deep Research Agent 可调用的检索工具。

| 工具 | 内部调用 | pool_only | 说明 |
|------|---------|-----------|------|
| `search_local` | `svc.search(mode="local")` | True（Agent 模式） | 本地向量库搜索 |
| `search_web` | `svc.search(mode="web")` | True（Agent 模式） | Web 来源搜索 |
| `search_scholar` | `SemanticScholarSearcher.search()` | True | Semantic Scholar 精确搜索 |
| `canvas_edit` | Canvas 操作 | — | 编辑 Canvas 文档 |

> **约束（关键）**：`search_scholar` 工具每次都会创建新的 `SemanticScholarSearcher` 实例，其内部 `_ensure_session()` 创建 `aiohttp.ClientSession`。**必须**在 `try/finally` 中调用 `await ss.close()`，否则 agent 结束后 event loop 关闭时会触发 `RuntimeError: Event loop is closed`。实现模式：将 `search + close` 封装在同一协程 `_search_and_close()` 中，通过 `asyncio.run()` 在子线程执行。

---

### 3.4 Deep Research Agent（LangGraph）

**文件**：`src/collaboration/research/agent.py`

使用 LangGraph 实现的有状态研究 Agent，支持断点续传和用户审核门控。

#### 状态机节点

```
clarify_node → scoping_node → plan_node
    ↓（用户确认后）
execute_deep_research():
    research_node → evaluate_node → [generate_claims_node] → write_node → verify_node
         ↑                                                                      ↓
         └─────────────── verify severe（未超 rewrite cap）───────────────────┘
                                        ↓（所有章节完成）
                              review_gate_node
                              ↓revise          ↓approve/skip
              review_revise_agent_supplement    synthesize_node
                      ↓
              review_revise_integrate
                      ↓
              review_gate_node（再次确认）
```

#### Section Evidence Pool（章节证据池）

每个章节维护独立的 `state["section_evidence_pool"][section_title]`，所有检索结果按 `pool_source` 标签追加入池：

| pool_source | 写入节点 | 参与 fuse 的池 |
|-------------|---------|---------------|
| `research_round` | research_node 主检索 | main pool |
| `eval_supplement` | evaluate_node GAP 补搜 | **gap pool（受 0.2 保护）** |
| `agent_supplement` | research_node 内部 agent 补搜 | **agent pool（受 0.25 保护）** |
| `write_stage` | write_node 兜底补搜 | main pool |
| `revise_supplement` | review_revise_agent_supplement | agent pool（受保护） |

> **架构铁律**：`eval_supplement` 必须进入 gap pool（`_DR_GAP_POOL_SOURCES` 常量）；`agent_supplement` 必须进入 agent pool；`revise_supplement` 必须入池（保证 synthesize 阶段引文可溯源）；`write_stage` 严禁回灌（防止写作证据污染主池重排权重）。

#### pool_only 原则

凡是检索结果还要进入 Section Evidence Pool 并在后续 `_rerank_section_pool_chunks()` 中统一重排的路径，**必须**设置 `pool_only=True`，禁止提前 rerank/截断：

- `research_round`：pool_only=True
- `eval_supplement`：pool_only=True
- `review_revise_supplement`：pool_only=True

例外（直接消费、不入池）：确认前背景检索、evaluate fallback、write/verify fallback。

---

### 3.5 任务调度层

**文件**：`src/tasks/`、`src/utils/task_runner.py`

#### 并发槽位设计

| 任务类型 | 并发控制 | 槽位共享 |
|---------|---------|---------|
| Deep Research | `max_active_slots`（全局） | 与 Chat 共享 |
| Chat | `max_active_slots`（全局） | 与 DR 共享 |
| Ingest | `ingest_max_concurrent`（独立） | 独立，子进程隔离 |
| Scholar | asyncio.create_task（per-task） | **不占用**全局槽位 |

#### Ingest 子进程隔离

Ingest 任务运行在独立的 `multiprocessing "spawn"` 子进程中：
- 子进程 OOM 或 segfault 不影响主 uvicorn 进程
- 异常退出时 task_runner 自动将 job 标记为 `error`

#### 任务状态枚举对照

| 系统 | 状态枚举 |
|------|---------|
| Redis（Chat/Scholar） | `queued / running / completed / error / cancelled / timeout` |
| IngestJob DB | `pending / running / done / error / cancelled` |
| DeepResearchJob DB | `pending / planning / running / waiting_review / cancelling / done / error / cancelled` |

---

## 4. Chat 核心流程

### 4.1 完整流程

```
用户消息
  ├─ 1. 上下文分析（意图检测：is_deep_research? is_compare?）
  ├─ 2. 查询构建（1+1+1 结构化查询：recall + precision + discovery）
  │       Chat 主检索前对 step_top_k 1.2 倍软放大：
  │       chat_effective_step_top_k = max(step_k, ceil(step_k × 1.2))
  ├─ 3. 并行检索（hybrid: local + web）
  │       结果以 pool_only=True 返回原始候选池（禁止中间 BGE rerank）
  │
  ├─ 4. [可选] 证据不足时 gap 补搜
  │       LLM 生成 1-3 条 gap query → 并行补搜 → 结果暂存 chat_gap_candidates_hits
  │       gap 补搜同样 pool_only=True，不做中间 rerank
  │
  ├─ §5¾ 单次统一 BGE Rerank（系统提示组装前，必须执行）
  │       _fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=chat_gap_candidates_hits, agent=[])
  │       → 全流程唯一一次联合 BGE rerank
  │       → 输出 write_top_k 条（含 gap 配额保护 chat_gap_ratio=0.2）
  │
  ├─ 5. [可选] Agent 工具调用
  │       LLM 在 §5¾ context_str 基础上调用工具（search_local/web/scholar 等）
  │       产生 agent_extra_chunks（§5¾ 之后才存在的新 chunk）
  │
  ├─ ⑧b [仅当 agent_extra_chunks 非空] Agent 追加 BGE Rerank
  │       _fuse_chat_main_gap_agent_candidates(main=pack.chunks, gap=[], agent=agent_extra_chunks)
  │       → 第二次 BGE rerank（gap 传 []，禁止重传，否则双重计数）
  │       → agent 受 chat_agent_ratio=0.1 保护
  │
  └─ 6. LLM 生成流式响应（含引文解析）
```

### 4.2 Chat BGE Rerank 次数约束

| 场景 | 次数 | 说明 |
|------|------|------|
| 无 agent / agent 无新 chunk | 1 次 | 仅 §5¾ main+gap fusion |
| agent 且有新 chunk | 2 次 | §5¾ + ⑧b agent 追加融合 |

> **禁止**：local main pool 单独 rerank、gap 补搜后立即做中间 fusion。超过 2 次的 rerank 均为冗余，会引入排序偏差。

---

## 5. Deep Research 核心流程

### 5.1 确认前流程（Pre-Research）

```
用户提交研究主题
    ↓
clarify_node
    ├─ [可选] Perplexity Sonar 生成 preliminary_knowledge
    └─ 基于 topic + history + prelim 生成澄清问题
    ↓
用户提交 clarification_answers
    ↓
scoping_node → 生成 ResearchBrief
    （scope / success_criteria / key_questions / exclusions / time_range / source_priority）
    ↓
plan_node
    ├─ 背景检索（1+1+1，plan_top_k = step_top_k or 15）
    └─ 生成章节大纲 → 写入 dashboard.sections + Canvas outline
    ↓
用户确认 confirmed_outline / confirmed_brief
    ↓
execute_deep_research()
```

### 5.2 章节研究循环

每个章节按以下顺序执行（深度预算受 `lite` / `comprehensive` preset 控制）：

```
research_node
  ├─ 1+1+1 主检索（step_top_k 条，pool_only=True → pool_source="research_round"）
  └─ agent_supplement（research_node 内部步骤，非独立节点）
       └─ pool_only=True → pool_source="agent_supplement"

evaluate_node
  ├─ BGE rerank #1（evaluate_pool_rerank，top_k=len(pool)，全池排序供覆盖评估）
  └─ 若 coverage_score < threshold 且存在 gaps：
       最多取前 3 个 gaps，每 gap 独立检索（step_top_k 条）
       → pool_source="eval_supplement"（必须进入 gap pool）

generate_claims_node（comprehensive 执行，lite 跳过）
  └─ BGE rerank #2（generate_claims_pool_rerank，top_k=write_top_k）

write_node
  ├─ effective_write_top_k = _compute_effective_write_k(preset, filters)
  ├─ verification_k = max(15, ceil(write_top_k × 0.25))
  └─ BGE rerank #3 + #4（write_pool_rerank + write_verify_pool_rerank）
       三池 fuse（main/gap/agent）：gap 0.2 + agent 0.25 保护
       独立产出 write_chunks 和 verify_chunks

verify_node
  ├─ light（unsupported ratio ≤ verify_light_threshold）：轻量告警，继续
  ├─ medium：记录 gaps，不回退
  └─ severe（> verify_severe_threshold）：回退到 research_node
       若超过 max_verify_rewrite_cycles，则 cap，强制继续完成
```

### 5.3 Deep Research BGE Rerank 次数

每章节 3-4 次，各有合理用途：

| 次序 | 节点 | 目的 | top_k |
|------|------|------|-------|
| #1 | evaluate_node | 全池覆盖评估 | len(pool)（全量） |
| #2 | generate_claims_node（仅 comprehensive） | 主张提取 | write_top_k |
| #3 | write_node | 章节写作 | write_top_k |
| #4 | write_node | 引文验证（独立 quota 窗口） | verification_k |

> 多次 rerank 是合理的：各节点 top_k 不同，gap/agent quota 窗口不同，章节池持续累积，后续节点在更大池上重新排序。这与 Chat 的"最小化 rerank"原则不冲突，因为场景完全不同。

### 5.4 Review / Confirm 阶段

```
review_gate_node（若 skip_draft_review=false 进入）
    ↓revise                        ↓approve/skip
review_revise_agent_supplement     synthesize_node
    │  仅跑 1 轮定向 GAP 补证
    │  review_revise_supplement_k = max(1, ceil((step_top_k or search_top_k_eval) × 0.5))
    │  新证据 → pool_source="revise_supplement" 追加入章节池（必须！）
    ↓
review_revise_integrate
    │  输入：旧章节文本 + 新补证 + 作者补充观点 + review 问题
    │  输出：新版章节（尽量保留有效的 [ref:xxxx] 占位）
    │  不重跑整章 research → evaluate → claims → write
    ↓
review_gate_node（必须回访，不能内部自动通过）
```

> **约束**：`revise` 路径**不重开整章研究循环**，仅做定向补证 + 整合重写。`revise_supplement` 证据必须入 Section Evidence Pool，保证 synthesize 阶段引文溯源。

### 5.5 深度预算参数对照

| 参数 | lite | comprehensive | 说明 |
|------|-----:|-------------:|------|
| `max_section_research_rounds` | 3 | 5 | 每章最多 research 轮次 |
| `max_verify_rewrite_cycles` | 1 | 2 | verify severe 最多打回次数 |
| `coverage_threshold` | 0.60 | 0.80 | evaluate 达标阈值 |
| `search_top_k_write`（write_top_k 基线） | 10 | 12 | 无 UI 覆盖时生效 |
| `verification_k` | 动态 `max(15, ceil(write_top_k×0.25))` | 同左 | 不再是固定值 |
| `verify_light_threshold` | 0.20 | 0.15 | 轻微告警阈值 |
| `verify_severe_threshold` | 0.45 | 0.35 | 严重回退阈值 |
| `global max_iterations` | `4 × N_sections` | `7 × N_sections` | `(rounds + cycles) × sections` |
| `recursion_limit` | 200 | 500 | LangGraph recursion limit |
| `cost_warn_steps` | 120 | 300 | 超过发成本预警 |
| `cost_force_summary_steps` | 180 | 420 | 超过设 force_synthesize |

### 5.6 effective_write_top_k 计算公式

```python
def _compute_effective_write_k(preset, filters):
    preset_write_k = int(preset.get("search_top_k_write", 12))
    ui_write_k     = int(filters.get("write_top_k") or 0)
    ui_step_k      = int(filters.get("step_top_k") or 0)

    if ui_write_k > 0:
        result = max(preset_write_k, ui_write_k)
    elif ui_step_k > 0:
        result = max(preset_write_k, int(ui_step_k * 1.5))
    else:
        result = preset_write_k

    # search_top_k_write_max 已接线，参与裁剪
    cap = int(preset.get("search_top_k_write_max", 0))
    if cap > 0:
        result = min(result, max(cap, preset_write_k))
    return result
```

---

## 6. 长任务 SSE 与断点机制

### 6.1 心跳机制

| 任务类型 | 心跳间隔 | 实现方式 |
|---------|---------|---------|
| Ingest | ~5s | SSE `heartbeat` 事件 |
| Deep Research | ~10s | SSE 含完整 job state 的状态推送 |
| Chat | ~15s | SSE comment 行 |
| Scholar | ~5s | SSE comment 或 heartbeat 事件 |

### 6.2 SSE 断点续传

所有 SSE 响应均包含 `id:` 字段（事件序号），客户端重连时通过 `Last-Event-ID` / `after_id` 参数请求服务端从断点位置重放事件。

### 6.3 优雅关闭

收到 SIGTERM → 等待 Scholar 任务完成（最长 30s） → 取消长任务 → 关闭 workers。

重启后：
- Ingest `running` → 标记 error，清理 checkpoint
- Deep Research `waiting_review` → 保留 checkpoint，下次重启可恢复
- Deep Research 其他非终态 → 标记 error

---

## 7. 配置加载优先级

```
UI/API 请求参数（最高优先）
  > rag_config.local.json（本地覆盖，gitignored）
  > 环境变量（RAG_LLM__{PROVIDER}__API_KEY 等）
  > rag_config.json（主配置）
  > 代码默认值（最低优先）
```

---

## 8. 可观测性

| 能力 | 实现 |
|------|------|
| 结构化日志 | JSON logs → `logs/backend/`，含 phase/section/job_id 上下文 |
| Metrics | OpenTelemetry + Prometheus exporter（`/metrics` 端点） |
| 健康检查 | `GET /health`（基础）/ `GET /health/detailed`（含 DB/Milvus/LLM/Redis 状态） |
| Debug 面板 | 开启后记录当次请求完整 DEBUG 日志到 `logs/debug/` |
| LLM 原始日志 | JSONL 按时间/大小自动轮转，存 `logs/llm_raw/` |
| LangSmith | 可选 Tracing（ReAct + LangGraph 全链路） |

---

## 9. 设计约束速查（禁止随意修改）

| # | 约束 | 原因 |
|---|------|------|
| 1 | `eval_supplement` 必须进入 gap pool（`_DR_GAP_POOL_SOURCES`） | 否则 GAP 证据在写作时与普通证据平等竞争，可能全部被挤出 top-k |
| 2 | `agent_supplement` 必须进入 agent pool | 否则 agent 补搜证据在候选量大时可能被全部挤出 |
| 3 | `revise_supplement` 必须入池 | 保证 synthesize 阶段引文精准溯源 |
| 4 | `write_stage` 严禁回灌 | 防止写作证据污染主池重排权重 |
| 5 | 入章节池的检索统一 pool_only=True | 防止提前 rerank/截断破坏全局排序 |
| 6 | fuse 后不得再做绝对分数阈值过滤 | 破坏 gap/agent 配额保护 |
| 7 | Chat BGE rerank 最多 2 次 | §5¾ 一次（必须）+ 可选 agent 追加一次 |
| 8 | gap 传入 agent 追加 fusion 时传 [] | gap 已在 pack.chunks 中，重传会双重计数 |
| 9 | search_scholar 必须在 finally 关闭 aiohttp session | 否则 event loop 关闭时崩溃 |
| 10 | local_recall_k 公式不可改 | hybrid 模式需要充足候选供全局融合 |
| 11 | soft-wait 不能改回 with-block | with-block 会强制等待所有线程，失去超时控制 |
| 12 | revise 路径不重开整章研究循环 | 定向补证已足够，全章重跑浪费资源且可能覆盖原有有效引用 |
| 13 | review_gate 必须回访（不能内部自动通过） | 用户是唯一的质量确认方，不可绕过 |
