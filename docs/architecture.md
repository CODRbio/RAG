# 系统架构

本文档描述 DeepSea RAG 的真实代码架构与核心数据流，便于开发、排障和二次扩展。

## 一、总体分层

```text
Frontend (React + Zustand)
  -> API Layer (FastAPI routes_*.py)
     -> Collaboration Layer (workflow/canvas/memory/research)
     -> Agent Layer (llm_manager + tools + react_loop)
     -> Retrieval Layer (hybrid + web + graph + rerank)
     -> Persistence/Infra (Milvus + SQLite + files + observability)
```

## 二、后端模块映射（`src/`）

- `api/`：HTTP 接口层，统一在 `server.py` 注册路由
- `auth/`：token 鉴权与密码处理
- `collaboration/`：多轮协作核心（canvas、memory、intent、workflow、research）
- `llm/`：统一 LLM 与 tool-calling 适配
- `retrieval/`：本地检索、网络检索、融合与去重
- `indexing/`：向量化与 Milvus 读写
- `parser/`：PDF 解析、声明提取
- `graph/`：HippoRAG 图检索
- `mcp/`：MCP Server，对外暴露工具/资源
- `observability/`：metrics、tracing、中间件
- `evaluation/`：评测执行与指标计算

## 三、前端结构（`frontend/src/`）

- `pages/`：`ChatPage`、`IngestPage`、`LoginPage`、`AdminPage`
- `components/`：
  - `chat/`（聊天窗口、输入、工具轨迹）
  - `canvas/`（画布）
  - `compare/`（多文档对比）
  - `graph/`（图谱可视化）
  - `workflow/`（研究流程相关交互）
  - `layout/`（头部、侧栏）
- `stores/`：Zustand 状态管理（chat/config/canvas/auth/projects/compare 等）
- `api/`：前端请求封装（按后端路由分模块）

## 四、核心数据流

### 1) 离线入库流

```text
raw PDF
 -> scripts/02_parse_papers.py
 -> src/parser/pdf_parser.py
 -> scripts/03_index_papers.py
 -> src/chunking/chunker.py + src/indexing/embedder.py
 -> src/indexing/milvus_ops.py
 -> (optional) scripts/03b_build_graph.py
```

### 2) 在线聊天检索流

```text
POST /chat or /chat/stream
 -> routes_chat.py
 -> (route decision: chat/rag)
 -> build_search_query_from_context()
    - reference gate: 无代词/指代时仅使用当前问题
    - context resolve: 仅在存在指代时使用 rolling_summary 解析主语
    - post-validation: 生成 query 必须与当前问题保留关键词重叠，否则回退原问题
 -> RetrievalService.search()
 -> HybridRetriever + UnifiedWebSearcher + HippoRAG
 -> Evidence/context packing
 -> LLMManager client.chat()
 -> response / SSE events
```

### 3) Agent ReAct 流

```text
react_loop(messages, tools)
 -> llm_manager.chat(tools=...)
 -> parse_tool_calls(...)
 -> execute_tool_call(...)
 -> append tool result message
 -> next iteration / final_text
```

### 4) Deep Research 流（LangGraph + 后台任务）

```text
Phase 1: /deep-research/start
  Scope -> Plan -> 返回 brief + outline（前端可确认/编辑）

Phase 2: /deep-research/submit（推荐）
  提交后台任务 -> 返回 job_id
  前端轮询 /deep-research/jobs/{job_id} 与 /events
  任务在后端持续运行（不依赖前端连接）
  用户可通过 /deep-research/jobs/{job_id}/cancel 显式停止

兼容模式: /deep-research/confirm（SSE 直连）
  Research -> Evaluate -> Write -> Verify -> Synthesize
```

### Deep Research 前端交互（最新）

- 配置前置化：
  - 在输入区 Deep Research 按钮旁提供 `⚙` 设置弹窗（持久化到 `useConfigStore + localStorage`）。
  - 配置项：`depth`、`outputLanguage`、`stepModels`、`stepModelStrict`。
- 启动链路：
  - 触发 Deep Research 时，澄清问题优先使用 `scope` 步骤模型。
  - 对话框中保留“本次运行覆盖”能力（高级折叠区），默认加载 `⚙` 的持久化配置。
- Drafting 审核区：
  - 支持“通过此章 / 需要修改 / 重新确认”。
  - 支持“一键全部通过并触发整合”。
  - 支持章节级 `gap supplement`（材料线索或直接观点）弹窗提交。
- 运行期监测：
  - 对话框 `running` 阶段内置 `Research Monitor`，实时显示 graph steps、成本状态、章节 coverage 曲线与效率评分（高/中/低）。
  - 对低效率章节给出提示：优先补充证据/约束提示词，再决定是否追加预算。
  - 支持一键生成“章节优化提示词模板”（可复制到 Intervention 作为下一轮输入）。

在 `use_agent=true` 的场景下，研究流程可递归补检索并对声明执行验证。
Deep Research 的检索策略为“按大纲聚焦 + gaps-first”：

- 章节查询先覆盖当前 section 的 gaps，再补 section 级查询
- `plan/research/write` 节点统一 `use_query_optimizer=False`
- 避免通用优化器对已聚焦 query 进行二次稀释

新增的“人工介入”链路：

- 用户可在确认阶段补充文本观点（`user_context`）
- 可选 `user_context_mode`：
  - `supporting`（普通补充）
  - `direct_injection`（强提示直接注入）
- 用户可上传 `pdf/md/txt` 作为临时材料：
  - `POST /deep-research/context-files` 提取文本
  - 文本进入 `user_documents`（仅本次任务）
  - 在 query 生成与写作阶段做“临时材料召回 + 注入”
  - 不写入 Milvus 持久库

### Research Depth 双级别

Deep Research 支持两种研究深度，通过 `depth` 参数选择：

| | lite | comprehensive |
|---|---|---|
| 定位 | 快速但学术可用 | 全面学术综述 |
| 预计耗时 | ~5-15 min | ~20-60 min |
| 迭代预算 | 3 × 章节数 | 6 × 章节数 |
| 每章研究轮次 | max 3 | max 5 |
| 覆盖度阈值 | 0.60 | 0.80 |
| 查询策略 | 2 recall + 2 precision + gaps | 4 recall + 4 precision + gaps |
| search_top_k（首轮/补缺/写作） | 18 / 10 / 10 | 30 / 15 / 12 |
| 写作二次取证 `verification_k` | 12 | 16 |
| 验证（轻/中/重） | 20% / 40% / 45% | 15% / 30% / 35% |
| 审核门（指数退避 + 早停） | 80 轮 / 8 次无变化 | 200 轮 / 12 次无变化 |
| LangGraph 递归上限 | 200 | 500 |
| 成本预警 / 强制摘要步数 | 120 / 180 | 300 / 420 |

所有阈值均定义在 `src/collaboration/research/agent.py` 的 `DEPTH_PRESETS` 常量中，
也可通过 `config/rag_config.json` → `deep_research.depth_presets` 覆盖。

**查询策略：Recall + Precision 双类查询**

每轮研究节点通过 LLM 生成两类查询（替代原来的单一查询列表）：
- **Recall queries**：短（3-6 关键词）、广、包含同义词/缩写/不同命名体系 → 保证"不漏角度"
- **Precision queries**：长（6-12 关键词）、带方法/时间/数据类型/对象约束 → 保证"证据准确"

**分层 search_top_k**

| 阶段 | 目的 | lite | comprehensive |
|------|------|------|--------------|
| 首轮研究 | 广撒网，多视角 | 18 | 30 |
| 补缺/复核 | 定点补证据 | 10 | 15 |
| 写作前检索 | 精选最可引用的 | 10 | 12 |
| 写作二次取证 | 关键数据点/引用复核 | 12 | 16 |

**3 级验证处理**（替代原来的单阈值回滚）：
- **轻微**（< light_threshold）：仅标记到 Insights Ledger，不中断流程
- **中等**（light..severe 区间）：记录 gaps，但**不**回滚到研究阶段（避免无限循环）
- **严重**（> severe_threshold）：回到 research 全面补证据

**信息不足处理策略**（Insufficient Information Policy）：
- **评估失败保守回退**：`evaluate_node` 出错时不再默认“信息充足”，而是低覆盖度+显式 gaps，继续补搜
- **零结果自适应补搜**：`research_node` 当 chunks 过少时自动执行更宽模式/更宽 query 的 fallback 检索
- **证据稀缺标记**：章节级 `evidence_scarce=true` 会贯穿到写作与最终综合阶段
- **写作降级保护**：证据严重不足时输出“受限摘要+缺口列表”，而非强写长段，减少幻觉风险
- **最终限制强化**：`synthesize_node` 会把证据稀缺章节显式写入“Limitations and Future Directions”

**循环防护机制**（防止图执行陷入无限循环）：
- **动态迭代预算**：`max_iterations = max_iterations_per_section × num_sections`，按章节数线性伸缩
- **每章研究轮次上限**：`max_section_research_rounds` 防止单章节消耗全部预算
- **Self-Correction**：当第 3 轮及以后且 coverage 达到阈值时，自动衰减 `search_top_k_gap` 并减半 query 数量
- **收益曲线早停**：若最近两轮 coverage 增益都低于阈值且已接近目标覆盖，提前进入写作
- **审核门指数退避 + 早停**：sleep 从 2s 指数增长到上限；连续 N 轮无变化自动放行
- **3 级验证分流**：只有"严重"级才触发回滚，避免"永远研究、写不出来"
- **LangGraph 递归限制**：编译时显式设置 `recursion_limit`，替代默认的 10000
- **成本步数监控**：达到 `cost_warn_steps` 发出人工介入提示；达到 `cost_force_summary_steps` 触发强制摘要模式
- **成本心跳上报**：每 `cost_tick_interval` 步发出 `cost_monitor_tick`，便于前端实时监测收益/成本曲线

### 最终整合链路（Synthesize + Global Refine）

当前最终阶段不是简单拼接，而是多步整合：

1. 生成 `Abstract`
2. 基于 insights + evidence-scarce sections 生成 `Limitations and Future Directions`
3. 聚合所有 `open gaps`（insights.gap + dashboard.coverage_gaps + section.gaps），生成
   `Open Gaps and Future Research Agenda`
4. 执行全篇连贯性重写（global coherence refine）：
   - 优化跨章节衔接、术语一致性、冗余消除
   - 保持标题层级与事实语义
5. 引用保护（citation guard）：
   - 若检测到引用/证据标签显著丢失（如 `[xxx]` / `[evidence limited]`），自动回退到整合前版本
6. 写回最终结果并切换 Canvas 到 `refine` 阶段

## 五、存储与状态

- Milvus：向量索引与 chunk 检索
- SQLite：会话、画布、用户与项目状态
- SQLite：Deep Research 后台任务与事件（`src/data/deep_research_jobs.db`）
- 会话中维护滚动摘要（`rolling_summary`/`summary_at_turn`）用于跨轮主语解析
- 文件系统：`data/raw_papers`、`data/parsed`、`artifacts`、`logs`

## 六、可观测性

- `/metrics`：Prometheus 指标导出
- `/health`：基础健康检查
- `/health/detailed`：组件级健康状态
- OTel：通过环境变量启用 tracing

## 七、设计约束

- LLM 调用必须通过 `src/llm/llm_manager.py`
- 新工具必须在 `src/llm/tools.py` 与 `src/mcp/server.py` 同步注册
- 配置新增字段必须同步 `config/rag_config.json` 与 `config/rag_config.example.json`
- 敏感配置放 `config/rag_config.local.json` 或环境变量
