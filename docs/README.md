# 文档中心

本目录是 DeepSea RAG 的统一文档入口。建议按角色阅读。

**使用方式原则**：所有功能与选项以 **UI 前端为主**；命令行与脚本为**默认/辅助**手段（如配置检查、单点测试、自动化），正式使用请通过前端界面操作。

## 按角色阅读

| 角色 | 推荐路径 |
|---|---|
| 新成员 / 首次部署 | `../README.md` → `../install.md` → `scripts_guide.md` |
| 后端开发 | `developer_guide.md` → `architecture.md` → `api_reference.md` |
| Prompt 工程 / 提示词调优 | `developer_guide.md`（新增提示词模板）→ `architecture.md`（Prompt 资产流） |
| 前端开发 | `../frontend/README.md` → `api_reference.md` |
| 运维 / 排障 | `configuration.md` → `operations_and_troubleshooting.md` → `agent_debug.md`（调试面板与日志） |
| 质量保障 | `testing_and_evaluation.md` → `dependency_matrix.md` |
| 生产部署 | `release_migration_ubuntu.md` → `operations_and_troubleshooting.md` |

## Deep Research 相关文档导航

| 方面 | 文档 |
|---|---|
| 前后端流程与节点策略 | `architecture.md` |
| 任务接口、审核/补充接口与事件 | `api_reference.md` |
| 配置项与环境变量 | `configuration.md` |
| 运行与排障（review gate / synthesize / resume queue） | `operations_and_troubleshooting.md` |
| 安装后快速验证（后台任务模式） | `../install.md` |

## 检索与证据相关文档导航

| 方面 | 文档 |
|---|---|
| Chat + DR 证据保留全流程、参数速查、gap 保护算法 | `evidence_retention.md` |
| 证据参数（`step_top_k` / `write_top_k` / `local_top_k`）含义 | `evidence_retention.md` §一 |
| **Chat 证据不足时的 gap 补搜与融合**（gap query 生成 → 补搜 → fuse，配额 0.2） | `evidence_retention.md` §二 2.3 |
| **gap_min_keep 来源**（Chat 0.2 / Research 0.25）与 **fuse 两种 WARNING**（池不足、补插后仍不足） | `evidence_retention.md` §四 4.2、4.3 |
| fuse_pools_with_gap_protection 算法步骤与 quota 保护 | `evidence_retention.md` §四 |
| soft-wait 超时机制与日志解读 | `evidence_retention.md` §五 |
| 勿随意修改的设计约束 | `evidence_retention.md` §七 |

## 文档清单

| 文件 | 内容 |
|---|---|
| `developer_guide.md` | 开发总指南（模块职责、约定、扩展路径） |
| `architecture.md` | 系统架构与关键数据流 |
| `evidence_retention.md` | **Chat + Deep Research 证据保留机制**（pool fusion、gap 保护、soft-wait、参数语义） |
| `api_reference.md` | 按前缀分组的完整 API 参考 |
| `configuration.md` | 配置项与环境变量说明 |
| `scripts_guide.md` | 脚本用途、参数、推荐执行顺序 |
| `operations_and_troubleshooting.md` | 启动、监控、运维、故障处理 |
| `agent_debug.md` | **调试面板**说明与「开面板即开本请求 DEBUG 日志」的联动 |
| `release_migration_ubuntu.md` | Ubuntu 发布与迁移全流程（systemd + Nginx） |
| `testing_and_evaluation.md` | pytest 与评测体系 |
| `dependency_matrix.md` | Python / 前端依赖矩阵与运行时要求 |

## 维护规范

- 新增功能时，至少更新以下任一文档：
  - 新 API → `api_reference.md`
  - 新配置 → `configuration.md`
  - 新脚本 → `scripts_guide.md`
  - 新模块/重要设计变更 → `architecture.md` + `developer_guide.md`
- 涉及提示词改造（`src/prompts/` 或 `PromptManager`）时，至少更新：
  - `developer_guide.md`（开发约定）
  - `architecture.md`（架构约束或数据流）
- 与用户直接相关的变更（启动方式、主要能力）同步更新根目录 `README.md`。
- 依赖版本变更时更新 `dependency_matrix.md`。
