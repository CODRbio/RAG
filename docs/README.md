# 文档中心

本目录是 DeepSea RAG 的统一文档入口。建议按角色阅读。

## 按角色阅读

- 新成员/首次部署：`../README.md` -> `../install.md` -> `scripts_guide.md`
- 后端开发：`developer_guide.md` -> `architecture.md` -> `api_reference.md`
- 前端开发：`../frontend/README.md` -> `api_reference.md`
- 运维/排障：`configuration.md` -> `operations_and_troubleshooting.md`
- 质量保障：`testing_and_evaluation.md`

## Deep Research 最新变更导航

- 前后端流程与节点策略：`architecture.md`
- 任务接口、审核/补充接口与事件：`api_reference.md`
- 配置项与环境变量（含 sonar key）：`configuration.md`
- 运行与排障（review gate / synthesize）：`operations_and_troubleshooting.md`
- 安装后快速验证（后台任务模式）：`../install.md`

## 文档清单

- `developer_guide.md`：开发总指南（模块职责、约定、扩展路径）
- `architecture.md`：系统架构与关键数据流
- `api_reference.md`：按前缀分组的 API 参考
- `configuration.md`：配置项与环境变量说明
- `scripts_guide.md`：脚本用途、参数、推荐执行顺序
- `operations_and_troubleshooting.md`：启动、监控、运维、故障处理
- `release_migration_ubuntu.md`：Ubuntu 发布与迁移全流程（systemd）
- `testing_and_evaluation.md`：pytest 与评测体系
- `dependency_matrix.md`：Python/前端依赖矩阵与运行时要求
- `parser_enhancement_summary.md`：PDF 解析增强专项说明
- `to_be_done_improve`：后续改进追踪（路线图）

## 维护规范

- 新增功能时，至少更新以下任一文档：
  - 新 API -> `api_reference.md`
  - 新配置 -> `configuration.md`
  - 新脚本 -> `scripts_guide.md`
  - 新模块/重要设计变更 -> `architecture.md` + `developer_guide.md`
- 与用户直接相关的变更（启动方式、主要能力）同步更新根目录 `README.md`。
