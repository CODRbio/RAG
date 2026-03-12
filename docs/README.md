# 文档中心

DeepSea RAG 系统的统一文档入口。

**使用方式原则**：所有功能与选项以 **UI 前端为主**；命令行与脚本为**辅助/运维**手段（配置检查、单点测试、自动化），正式使用请通过前端界面操作。

---

## 按角色快速导航

| 角色 | 推荐路径 |
|------|---------|
| **新成员 / 首次安装** | [`installation_and_migration.md`](installation_and_migration.md) |
| **生产部署 / 迁移升级** | [`installation_and_migration.md`](installation_and_migration.md)（第 3-6 节） |
| **后端开发** | [`architecture.md`](architecture.md) → [`developer_guide.md`](developer_guide.md) → [`api_reference.md`](api_reference.md) |
| **前端开发** | [`api_reference.md`](api_reference.md) → [`developer_guide.md`](developer_guide.md)（第 7 节） |
| **Prompt / 提示词调优** | [`developer_guide.md`](developer_guide.md)（第 4 节）→ [`chat_research_workflow_contract.md`](chat_research_workflow_contract.md) |
| **运维 / 排障** | [`operations_and_troubleshooting.md`](operations_and_troubleshooting.md) → [`configuration.md`](configuration.md) |
| **理解检索证据机制** | [`evidence_retention.md`](evidence_retention.md) → [`budget_evidence_flow_mechanism.md`](budget_evidence_flow_mechanism.md) |
| **理解 Deep Research 流程** | [`architecture.md`](architecture.md)（第 5 节）→ [`chat_research_workflow_contract.md`](chat_research_workflow_contract.md) |
| **长任务管理** | [`task_management.md`](task_management.md) |

---

## 文档清单

### 核心架构与开发

| 文档 | 内容摘要 |
|------|---------|
| [`architecture.md`](architecture.md) | 系统整体架构、目录结构、核心模块详解、Chat/Deep Research 完整流程、**设计约束速查** |
| [`developer_guide.md`](developer_guide.md) | 开发环境启动、模块职责边界、证据流转详解、提示词约定、数据库迁移、测试、扩展场景 |
| [`api_reference.md`](api_reference.md) | 所有 HTTP/SSE 接口（Chat、Deep Research、Ingest、Scholar、Canvas、Compare） |

### 安装与运维

| 文档 | 内容摘要 |
|------|---------|
| [`installation_and_migration.md`](installation_and_migration.md) | **完整安装指南**：Mac 开发环境 + Ubuntu 生产环境 + systemd + Nginx + 版本升级 + 数据迁移 + 回滚方案 |
| [`operations_and_troubleshooting.md`](operations_and_troubleshooting.md) | 服务管理、健康检查、任务监控、存储清理、检索诊断、常见故障处理、紧急处置清单 |
| [`configuration.md`](configuration.md) | 配置文件详解（rag_config.json 所有配置块）、加载优先级、环境变量说明 |

### 核心流程与约束

| 文档 | 内容摘要 |
|------|---------|
| [`chat_research_workflow_contract.md`](chat_research_workflow_contract.md) | Chat + Deep Research **当前实现现状**（节点顺序、参数关系、数量边界、设计约束）—— 代码对齐文档 |
| [`evidence_retention.md`](evidence_retention.md) | 证据保留全流程：step_top_k / write_top_k 语义、pool fusion 算法、gap 保护、soft-wait 超时 |
| [`budget_evidence_flow_mechanism.md`](budget_evidence_flow_mechanism.md) | 预算计算、证据流转、阶段跳转、候选择优机制详解 |
| [`task_management.md`](task_management.md) | 长任务管理契约：并发槽位、心跳、checkpoint 断点续传、SSE、优雅关闭 |

### 专项功能

| 文档 | 内容摘要 |
|------|---------|
| [`information_download_gather.md`](information_download_gather.md) | 文献 PDF 下载策略链、验证码处理、PDF 按钮查找机制 |

---

## 关键设计约束速查

以下约束**禁止随意修改**，每项均有对应的架构级原因：

| 约束 | 详细位置 |
|------|---------|
| `eval_supplement` 必须进 gap pool（`_DR_GAP_POOL_SOURCES`） | `architecture.md` §9，`evidence_retention.md` §七.4 |
| `agent_supplement` 必须进 agent pool | `architecture.md` §9，`evidence_retention.md` §七.5 |
| `revise_supplement` 必须追加写入 Section Evidence Pool | `chat_research_workflow_contract.md` §10.3 |
| `write_stage` 严禁回灌进章节池 | `evidence_retention.md` §七.8 |
| 入章节池的检索统一使用 `pool_only=True` | `evidence_retention.md` §七.6 |
| fuse 后不得再做绝对分数阈值过滤 | `evidence_retention.md` §七.2 |
| Chat BGE rerank 最多 2 次 | `evidence_retention.md` §七.11 |
| agent 追加 fusion 的 gap 传 `[]`，禁止重传 | `evidence_retention.md` §七.11 |
| `search_scholar` 工具必须在 `finally` 关闭 aiohttp session | `evidence_retention.md` §七.13，`architecture.md` §3.3 |
| `review_gate_node` 必须回访，不能内部自动通过 | `architecture.md` §5.4，`chat_research_workflow_contract.md` §6.2 |

---

## Deep Research 参数速查

| 参数 | lite | comprehensive | 说明 |
|------|-----:|-------------:|------|
| `max_section_research_rounds` | 3 | 5 | 每章最多 research 轮次 |
| `max_verify_rewrite_cycles` | 1 | 2 | verify severe 最多打回次数 |
| `coverage_threshold` | 0.60 | 0.80 | evaluate 达标阈值 |
| `search_top_k_write`（write_top_k 基线） | 10 | 12 | 无 UI 覆盖时生效 |
| `search_top_k_write_max`（上限 cap） | 40 | 60 | 已接线参与裁剪 |
| `verification_k` | 动态 `max(15, ceil(write_top_k×0.25))` | 同 | 不再是固定值 |
| `verify_light_threshold` | 0.20 | 0.15 | 轻微告警阈值（继续） |
| `verify_severe_threshold` | 0.45 | 0.35 | 严重回退阈值 |
| 章节 fuse gap 比例 | 0.20 | 0.20 | 仅 `eval_supplement` |
| 章节 fuse agent 比例 | 0.25 | 0.25 | 仅 `agent_supplement` |
| 放大池倍率 | 3.0 | 3.0 | `research_rank_pool_multiplier` |

**effective_write_top_k 计算（按优先级）**：
1. UI 传 `write_top_k > 0`：`max(preset_write_k, write_top_k)`
2. UI 传 `step_top_k > 0`（未传 write_top_k）：`max(preset_write_k, floor(step_top_k × 1.5))`
3. 均未传：`preset_write_k`（lite=10，comprehensive=12）
4. 结果受 `search_top_k_write_max` cap（lite=40，comprehensive=60）

---

## 维护规范

修改代码时，同步更新对应文档：

| 修改内容 | 需同步的文档 |
|---------|------------|
| 新增 / 修改 API 接口 | `api_reference.md` |
| 新增 / 修改配置项 | `configuration.md` + `config/rag_config.example.json` |
| 检索流程变更 | `chat_research_workflow_contract.md` + `evidence_retention.md` + `architecture.md` |
| Deep Research 节点变更 | `chat_research_workflow_contract.md` + `architecture.md` |
| 新增 Agent 工具 | `developer_guide.md`（第 5.1 节）+ `api_reference.md` |
| 新增 / 修改提示词模板 | `developer_guide.md`（第 4 节） |
| 安装 / 部署流程变更 | `installation_and_migration.md` |
| 任务管理机制变更 | `task_management.md` |
| 根目录 README（面向用户）变更 | `../README.md`（中英文版本） |

---

*最后更新：2026-03-11*
