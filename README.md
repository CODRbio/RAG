# DeepSea RAG

面向科研场景（尤其深海相关文献）的全栈 RAG 系统：支持 PDF 解析与入库、混合检索（向量 + 图谱 + 网络）、多轮对话、Deep Research、画布协作、引用管理、多文档对比，以及可观测性与 MCP 工具化接入。

## 你可以用它做什么

- 把本地 PDF 批量解析为结构化数据并向量入库
- 在聊天中做 `local / web / hybrid` 检索并给出可追溯引用
- 启用 Agent 模式让模型自动调用检索/画布/图谱/对比工具
- 对 2-5 篇论文做结构化比较（对话引文候选 + 本地文库候选）
- 在 Canvas 中持续编辑综述草稿并导出 Markdown
- 通过 Deep Research 完成从选题到综述终稿的全自动化研究流程
- 通过 `/metrics`、`/health/detailed` 监控运行状态
- 多语言支持（中/英）

## 当前技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI + Pydantic + SQLModel/Alembic + SQLite(`data/rag.db`) + Milvus + NetworkX |
| 检索引擎 | Dense/Sparse + RRF + BGE-M3/ColBERT + HippoRAG + Web Search 聚合 |
| LLM 调度 | 统一 `LLMManager`（OpenAI / DeepSeek / Gemini / Claude / Kimi / Sonar） |
| Agent 框架 | ReAct 循环 + 统一 Tool 抽象 + LangGraph Deep Research |
| 前端 | React 19 + TypeScript + Zustand + Vite 7 + Tailwind CSS |
| 国际化 | i18next（中/英） |
| 可观测性 | OpenTelemetry + Prometheus + LangSmith |
| 工具化 | MCP Server（对外暴露工具/资源） |

## 快速开始

### 1) 安装依赖

```bash
conda create -n deepsea-rag python=3.10 -y
conda activate deepsea-rag
conda install -c pytorch -c conda-forge "pytorch>=2.6.0" "torchvision>=0.21.0" timm -y
pip install -r requirements.txt --no-cache-dir
cd frontend && npm install && cd ..
playwright install chromium
```

详细安装与依赖核对见 `install.md`。

### 2) 准备配置

```bash
cp config/rag_config.example.json config/rag_config.json
cp config/rag_config.local.example.json config/rag_config.local.json
cp .env.example .env
bash scripts/verify_dependencies.sh
```

- 敏感字段（API Key）建议放在 `config/rag_config.local.json`
- 或使用环境变量覆盖：`RAG_LLM__{PROVIDER}__API_KEY`
- 数据库默认使用 `database.url = sqlite:///data/rag.db`（可用 `RAG_DATABASE_URL` 覆盖）

### 3) 启动基础服务（Milvus 等）

```bash
bash scripts/00_preflight_check.sh
# 开发环境（Mac CPU）
docker compose --profile dev up -d
# 生产环境（Linux GPU）
# docker compose --profile prod up -d
bash scripts/00_healthcheck_docker.sh
```

### 4) 执行离线入库流水线

```bash
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
```

### 5) 启动全栈

```bash
bash scripts/start.sh
```

- 前端：`http://localhost:5173`
- 后端 Swagger：`http://127.0.0.1:9999/docs`

## 关键接口速览

| 功能域 | 接口 |
|---|---|
| 聊天 | `POST /chat`、`POST /chat/stream` |
| Deep Research | `/deep-research/start\|submit\|jobs/*\|review\|gap-supplement\|insights` |
| 画布 | `/canvas/*`（CRUD + 大纲 + 草稿 + 快照 + AI 编辑 + 引用管理） |
| 导出 | `POST /export` |
| 对比 | `POST /compare`、`GET /compare/candidates`、`GET /compare/papers` |
| 图谱 | `/graph/*`（统计 + 实体 + 邻居 + chunk 详情） |
| 在线入库 | `/ingest/*`（上传 + Collections + 任务管理） |
| 认证与项目 | `/auth/*`、`/admin/*`、`/projects/*` |
| 模型管理 | `GET /models/status`、`POST /models/sync`、`GET /llm/providers` |
| 自动补全 | `POST /auto-complete` |
| 可观测 | `GET /metrics`、`GET /health`、`GET /health/detailed`、`GET /storage/stats` |

完整接口见 `docs/api_reference.md`。

## Deep Research 核心能力

- **启动前 `⚙` 设置**：深度（lite / comprehensive）、输出语言、分步骤模型、strict step model
- **后台任务模式**：提交后前端可关闭/刷新，任务在后端持续运行
- **Drafting 人工审核**：通过 / 修改 / 重新确认 + 一键"全部通过并触发整合"
- **章节缺口补充**：支持"材料线索"与"直接观点"
- **人工介入**：上传临时材料（pdf/md/txt）或文本补充，仅用于本次任务
- **最终整合**：自动生成 Abstract + Limitations + Open Gaps 研究议程 + 全篇连贯性整合
- **引用保护**：若整合后引用/证据标签显著丢失，自动回退安全版本
- **成本监控**：心跳上报 + 预警阈值 + 强制摘要模式
- **循环防护**：动态迭代预算 + 收益曲线早停 + 3 级验证分流

## 目录结构

```text
.
├── README.md                 # 项目入口
├── install.md                # 安装与依赖核对
├── requirements.txt          # Python 依赖
├── docker-compose.yml        # Docker 服务（Milvus/etcd/MinIO）
├── .env.example              # 环境变量模板
│
├── config/                   # 配置中心
│   ├── rag_config.json       #   主配置（结构 + 默认值）
│   ├── rag_config.local.json #   本地覆盖（敏感信息，gitignored）
│   ├── rag_config.example.json
│   ├── rag_config.local.example.json
│   └── settings.py           #   Python 配置加载器
│
├── src/                      # 后端源码
│   ├── api/                  #   FastAPI 路由层
│   ├── llm/                  #   LLMManager + tools + react_loop
│   ├── db/                   #   SQLModel 模型 / 连接引擎 / 历史库迁移
│   ├── retrieval/            #   混合检索 + web 聚合 + 重排
│   ├── collaboration/        #   协作核心
│   │   ├── canvas/           #     画布管理
│   │   ├── memory/           #     会话/工作/持久记忆
│   │   ├── intent/           #     意图解析与命令
│   │   ├── research/         #     Deep Research（LangGraph Agent）
│   │   ├── workflow/         #     状态机
│   │   ├── citation/         #     引用管理与格式化
│   │   └── export/           #     导出格式化
│   ├── indexing/             #   embed + Milvus + paper 管理
│   ├── parser/               #   PDF 解析 + 声明提取
│   ├── chunking/             #   结构化切块
│   ├── generation/           #   证据综合 + 上下文打包 + LLM 兼容层
│   ├── graph/                #   HippoRAG 图检索
│   ├── graphs/               #   LangGraph 流水线（入库图）
│   ├── mcp/                  #   MCP Server
│   ├── observability/        #   metrics + tracing + middleware
│   ├── evaluation/           #   评测执行与指标
│   ├── auth/                 #   认证（session + password）
│   ├── utils/                #   缓存/限流/清理/提示词管理/任务运行器
│   ├── log/                  #   日志管理
│   └── prompts/              #   LLM 提示词模板
│
├── frontend/                 # React 前端
│   ├── src/pages/            #   ChatPage / IngestPage / LoginPage / AdminPage
│   ├── src/components/       #   chat / canvas / compare / graph / workflow / research / settings / layout / ui
│   ├── src/stores/           #   Zustand 状态管理
│   ├── src/api/              #   后端接口封装
│   ├── src/types/            #   TypeScript 类型
│   └── src/i18n/             #   国际化（en/zh）
│
├── scripts/                  # 运行与测试脚本
├── tests/                    # pytest 测试
├── docs/                     # 文档中心
├── data/                     # 数据存储（raw/parsed/metadata/graph）
├── volumes/                  # Docker 持久卷（Milvus/etcd/MinIO）
└── logs/                     # 应用日志（含 LLM 原始响应日志）
```

## 文档导航

| 文档 | 说明 |
|---|---|
| `docs/README.md` | 文档总览与角色阅读路径 |
| `docs/developer_guide.md` | 开发总指南（模块职责、约定、扩展路径） |
| `docs/architecture.md` | 系统架构与关键数据流 |
| `docs/api_reference.md` | 按前缀分组的完整 API 参考 |
| `docs/configuration.md` | 配置项与环境变量说明 |
| `docs/scripts_guide.md` | 脚本用途、参数、推荐执行顺序 |
| `docs/operations_and_troubleshooting.md` | 启动、监控、运维、故障处理 |
| `docs/release_migration_ubuntu.md` | Ubuntu 发布与迁移全流程（systemd + Nginx） |
| `docs/testing_and_evaluation.md` | pytest 与评测体系 |
| `docs/dependency_matrix.md` | Python / 前端依赖矩阵与运行时要求 |

## LLM 调用约定（必须遵守）

统一走 `src/llm/llm_manager.py`：

```python
from src.llm import LLMManager

manager = LLMManager.from_json("config/rag_config.json")
client = manager.get_client("deepseek")
resp = client.chat(messages=[{"role": "user", "content": "你好"}])
text = resp["final_text"]
```

禁止直接在业务代码里实例化各家 SDK 客户端或硬编码密钥。

## Prompt 管理约定（必须遵守）

- 所有业务 prompt 模板统一放在 `src/prompts/`
- 业务代码通过 `src/utils/prompt_manager.py` 读取，不在 Python 逻辑中硬编码多行提示词
- 常规模板使用 `PromptManager.render(template, **kwargs)`，延迟格式化场景使用 `PromptManager.load(template)`

示例：

```python
from src.utils.prompt_manager import PromptManager

_pm = PromptManager()
system_prompt = _pm.render("chat_route_system.txt")
user_prompt = _pm.render("chat_route_classify.txt", history=history, message=message)
```

对于包含 JSON 示例的模板，字面量花括号需使用 `{{` / `}}`，以兼容 `str.format`。

## License

MIT
