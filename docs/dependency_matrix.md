# 依赖矩阵（核对版）

本文档基于当前仓库依赖文件核对：

- `requirements.txt`
- `frontend/package.json`
- `frontend/package-lock.json`

更新时间：2026-02-11

## 1) Python 依赖（后端）

### 关键框架

- FastAPI: `>=0.100.0,<1.0.0`
- Uvicorn: `>=0.22.0`
- Pydantic: `>=2.0.0,<3.0.0`
- LangGraph: `>=0.2.0`
- MCP SDK: `>=1.26.0`

### 检索与模型

- pymilvus[model]: `>=2.5.0`
- transformers: `>=4.42.0,<5.0.0`
- huggingface-hub: `==0.36.0`
- sentence-transformers: `>=2.2.0`
- FlagEmbedding: `>=1.2.0`
- ragatouille: `>=0.0.8`

### 搜索与抓取

- tavily-python: `>=0.5.0`
- playwright: `>=1.40.0`
- playwright-stealth: `>=1.0.0`
- beautifulsoup4: `>=4.12.0`
- trafilatura: `>=2.0.0`
- aiohttp: `>=3.9.0`

### 可观测性

- prometheus-client: `>=0.20.0`
- opentelemetry-api/sdk/instrumentation/exporter

### 测试

- pytest: `>=8.0.0`

## 2) JavaScript/TypeScript 依赖（前端）

### Runtime dependencies

- react: `^19.2.0`
- react-dom: `^19.2.0`
- react-router-dom: `^7.13.0`
- zustand: `^5.0.11`
- axios: `^1.13.4`
- react-markdown: `^10.1.0`
- react-force-graph-2d: `^1.29.1`

### Dev dependencies

- vite: `^7.2.4`（lock 实际解析到 `7.3.1`）
- typescript: `~5.9.3`
- eslint: `^9.39.1`
- tailwindcss: `^4.1.18`

## 3) 运行时要求

- Python: `3.10+`
- Node: `^20.19.0 || >=22.12.0`（来自 lock 中 `node_modules/vite` 的 engines）

## 3.1) 系统依赖（Ubuntu 推荐）

- `docker.io`、`docker-compose-plugin`
- `git`、`curl`、`wget`、`unzip`
- `build-essential`、`pkg-config`、`python3-dev`
- `libssl-dev`、`libffi-dev`、`libsqlite3-dev`
- 可选：`xvfb`（仅 headful 浏览器场景）

## 3.2) 依赖配置映射（必须核对）

### LLM 相关

- 依赖：`openai`、`anthropic`、`mcp`
- 配置：
  - `config/rag_config.json` -> `llm.default`, `llm.providers.*`
  - 环境变量 -> `RAG_LLM__{PROVIDER}__API_KEY`

### Web 搜索相关

- 依赖：`playwright`、`playwright-stealth`、`tavily-python`、`trafilatura`、`aiohttp`
- 配置：
  - `web_search.enabled`, `web_search.api_key`
  - `google_search.enabled`, `google_search.scholar_enabled`, `google_search.google_enabled`
  - `semantic_scholar.enabled`, `semantic_scholar.api_key`
  - `content_fetcher.enabled`

### 检索/索引相关

- 依赖：`pymilvus[model]`, `sentence-transformers`, `FlagEmbedding`, `ragatouille`
- 配置：
  - `search.*`
  - `collection.*`（通过 `config/settings.py` 环境变量映射）
  - `MILVUS_HOST`, `MILVUS_PORT`

### API/服务相关

- 依赖：`fastapi`, `uvicorn`, `pydantic`, `requests`
- 配置：
  - `api.host`, `api.port`
  - `API_HOST`, `API_PORT`
  - `auth.*`（生产环境必须替换默认 secret）

### 可观测性

- 依赖：`opentelemetry-*`, `prometheus-client`
- 配置：
  - `/metrics`、`/health/detailed`
  - 可选 tracing 环境变量（OTEL）

## 4) 一致性检查建议

- Python：`pip install -r requirements.txt --no-cache-dir` 后跑一次关键导入检查
- 前端：优先 `npm ci` 确保与 lock 一致
- CI/发布前执行：
  - `pytest -q`
  - `npm run build`（`frontend/`）
  - `bash scripts/verify_dependencies.sh`

## 5) 注意事项

- `torch/torchvision/timm` 不在 `requirements.txt`，需 Conda 安装
- 密钥相关依赖配置不要写死在代码，统一走 `rag_config.local.json` 或环境变量

## 6) 本机核对记录（2026-02-11）

在 `deepsea-rag` 环境执行依赖核对：

- 初次核对缺失：`mcp`、`trafilatura`
- 已补装并复核通过：`mcp==1.26.0`、`trafilatura==2.0.0`
- 关键包实测版本：
  - `transformers==4.57.6`
  - `huggingface-hub==0.36.0`
  - `fastapi==0.128.0`
  - `torch==2.10.0`
  - `torchvision==0.25.0`

前端运行时补充：

- 当前命令环境中 `node` 不在 PATH（`command not found`）
- 但前端依赖声明与 lock 正常（`package.json` + `package-lock.json`）
- 请在本机确保 Node 满足 `^20.19.0 || >=22.12.0`
