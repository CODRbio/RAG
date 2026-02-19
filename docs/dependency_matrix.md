# 依赖矩阵

本文档基于当前仓库依赖文件核对：

- `requirements.txt`
- `frontend/package.json`
- `frontend/package-lock.json`

更新时间：2026-02-19

## 1) Python 依赖（后端）

### 关键框架

| 包 | 版本约束 | 用途 |
|---|---|---|
| FastAPI | `>=0.100.0,<1.0.0` | Web 框架 |
| Uvicorn | `>=0.22.0` | ASGI 服务器 |
| Pydantic | `>=2.0.0,<3.0.0` | 数据验证 |
| SQLModel | `>=0.0.22` | ORM（统一业务数据库访问） |
| Alembic | `>=1.14.0` | 数据库 schema 迁移 |
| LangGraph | `>=0.2.0` | Agent 图编排 |
| MCP SDK | `>=1.26.0` | MCP 协议 |

### LLM 客户端

| 包 | 版本约束 | 用途 |
|---|---|---|
| openai | `>=1.0.0,<3.0.0` | OpenAI / DeepSeek / Kimi 等 |
| anthropic | `>=0.18.0` | Claude |
| tiktoken | `>=0.7.0` | Token 预算估算（cl100k_base 编码，跨 provider 近似计算，用于 coherence 阶段双路策略决策） |

### 检索与模型

| 包 | 版本约束 | 用途 |
|---|---|---|
| pymilvus[model] | `>=2.5.0` | Milvus 向量数据库 |
| transformers | `>=4.42.0,<5.0.0` | HuggingFace 模型 |
| huggingface-hub | `==0.36.0` | 模型下载 |
| sentence-transformers | `>=2.2.0` | 句向量 |
| FlagEmbedding | `>=1.2.0` | BGE 系列模型 |
| ragatouille | `>=0.0.8` | ColBERT 重排 |
| einops | `>=0.7.0` | 张量操作 |

### PDF 解析

| 包 | 版本约束 | 用途 |
|---|---|---|
| docling | `>=2.0.0` | PDF 解析 |
| pymupdf | `>=1.24.0` | PDF 操作 |
| pillow | `>=10.0.0` | 图像处理 |

### 搜索与抓取

| 包 | 版本约束 | 用途 |
|---|---|---|
| tavily-python | `>=0.5.0` | Tavily 搜索 |
| playwright | `>=1.40.0` | 浏览器自动化 |
| playwright-stealth | `>=1.0.0` | 反检测 |
| beautifulsoup4 | `>=4.12.0` | HTML 解析 |
| trafilatura | `>=2.0.0` | 网页提取 |
| aiohttp | `>=3.9.0` | 异步 HTTP |

### 可观测性

| 包 | 版本约束 | 用途 |
|---|---|---|
| prometheus-client | `>=0.20.0` | Prometheus 指标 |
| opentelemetry-api/sdk | — | 分布式追踪 |

### 图与知识

| 包 | 版本约束 | 用途 |
|---|---|---|
| networkx | — | 知识图谱 |
| langgraph-checkpoint-sqlite | `>=0.0.20` | LangGraph 检查点 |

### 测试

| 包 | 版本约束 | 用途 |
|---|---|---|
| pytest | `>=8.0.0` | 测试框架 |

## 2) JavaScript / TypeScript 依赖（前端）

### Runtime dependencies

| 包 | 版本 | 用途 |
|---|---|---|
| react | `^19.2.0` | UI 框架 |
| react-dom | `^19.2.0` | DOM 渲染 |
| react-router-dom | `^7.13.0` | 路由 |
| zustand | `^5.0.11` | 状态管理 |
| axios | `^1.13.4` | HTTP 客户端 |
| react-markdown | `^10.1.0` | Markdown 渲染 |
| react-force-graph-2d | `^1.29.1` | 图谱可视化 |
| react-pdf | `^10.3.0` | PDF 预览 |
| lucide-react | — | 图标库 |
| i18next | — | 国际化 |
| react-i18next | — | React 国际化绑定 |

### Dev dependencies

| 包 | 版本 | 用途 |
|---|---|---|
| vite | `^7.2.4` | 构建工具 |
| typescript | `~5.9.3` | 类型系统 |
| eslint | `^9.39.1` | 代码检查 |
| tailwindcss | `^4.1.18` | CSS 框架 |
| @tailwindcss/vite | — | Tailwind Vite 插件 |

## 3) 运行时要求

| 依赖 | 版本 |
|---|---|
| Python | `3.10+`（推荐 3.10） |
| Node.js | `^20.19.0 \|\| >=22.12.0` |
| PyTorch | `>=2.6.0`（Conda 安装） |
| Torchvision | `>=0.21.0`（Conda 安装） |
| Docker / Compose | 最新稳定版 |

## 3.1) 系统依赖（Ubuntu 推荐）

- `docker.io`、`docker-compose-plugin`
- `git`、`curl`、`wget`、`unzip`
- `build-essential`、`pkg-config`、`python3-dev`
- `libssl-dev`、`libffi-dev`、`libsqlite3-dev`
- 可选：`xvfb`（仅 headful 浏览器场景）

## 3.2) 依赖配置映射

### LLM 相关

- 依赖：`openai`、`anthropic`、`tiktoken`、`mcp`
- 配置：`llm.default`、`llm.providers.*`
- 环境变量：`RAG_LLM__{PROVIDER}__API_KEY`

### Web 搜索相关

- 依赖：`playwright`、`playwright-stealth`、`tavily-python`、`trafilatura`、`aiohttp`
- 配置：`web_search.enabled`、`google_search.enabled`、`semantic_scholar.enabled`、`ncbi.enabled`、`content_fetcher.enabled`

### 检索/索引相关

- 依赖：`pymilvus[model]`、`sentence-transformers`、`FlagEmbedding`、`ragatouille`
- 配置：`search.*`、`MILVUS_HOST`、`MILVUS_PORT`

### API/服务相关

- 依赖：`fastapi`、`uvicorn`、`pydantic`
- 配置：`api.host`、`api.port`、`auth.*`

### 可观测性

- 依赖：`opentelemetry-*`、`prometheus-client`
- 端点：`/metrics`、`/health/detailed`

## 4) 一致性检查建议

- Python：`pip install -r requirements.txt --no-cache-dir` 后跑一次关键导入检查
- 前端：优先 `npm ci` 确保与 lock 一致
- CI / 发布前执行：
  - `pytest -q`
  - `npm run build`（`frontend/`）
  - `bash scripts/verify_dependencies.sh`

## 5) 注意事项

- `torch` / `torchvision` / `timm` 不在 `requirements.txt`，需 Conda 安装
- 密钥相关依赖配置不要写死在代码，统一走 `rag_config.local.json` 或环境变量
- `huggingface-hub` 版本锁定为 `0.36.0`，升级前需验证 `transformers` 与 `FlagEmbedding` 兼容性
