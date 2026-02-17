# DeepSea RAG

面向科研场景（尤其深海相关文献）的全栈 RAG 系统：支持 PDF 解析与入库、混合检索（向量+图谱+网络）、多轮对话、Deep Research、画布协作、引用管理、多文档对比，以及可观测性与 MCP 工具化接入。

## 你可以用它做什么

- 把本地 PDF 批量解析为结构化数据并向量入库
- 在聊天中做 `local / web / hybrid` 检索并给出可追溯引用
- 启用 Agent 模式让模型自动调用检索/画布/图谱/对比工具
- 对 2-5 篇论文做结构化比较（对话引文候选 + 本地文库候选）
- 在 Canvas 中持续编辑综述草稿并导出 Markdown
- 通过 `/metrics`、`/health/detailed` 监控运行状态

## 当前技术栈（代码实况）

- 后端：FastAPI + Pydantic + SQLite + Milvus + NetworkX
- 检索：Dense/Sparse + RRF + BGE/ColBERT + HippoRAG + Web Search 聚合
- LLM：统一 `LLMManager`（OpenAI/DeepSeek/Gemini/Claude/Kimi）
- Agent：ReAct 循环 + 统一 Tool 抽象 + LangGraph Deep Research
- 前端：React + TypeScript + Zustand + Vite

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
bash scripts/verify_dependencies.sh
```

- 敏感字段（API Key）建议放在 `config/rag_config.local.json`
- 或使用环境变量覆盖：`RAG_LLM__{PROVIDER}__API_KEY`

### 3) 启动基础服务（Milvus 等）

```bash
bash scripts/00_preflight_check.sh
docker compose --profile dev up -d
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

- 聊天：`POST /chat`、`POST /chat/stream`
- Deep Research：`/deep-research/start|submit|jobs/*|review|gap-supplement|insights`
- 画布：`/canvas/*`
- 导出：`POST /export`
- 对比：`POST /compare`、`GET /compare/candidates`、`GET /compare/papers`
- 图谱：`/graph/*`
- 在线入库：`/ingest/*`
- 认证与项目：`/auth/*`、`/admin/*`、`/projects/*`
- 可观测：`GET /metrics`、`GET /health`、`GET /health/detailed`

完整接口见 `docs/api_reference.md`。

## Deep Research 最新能力（摘要）

- 启动前 `⚙` 设置：深度、输出语言、分步骤模型、strict step model
- Drafting 人工审核：通过/修改/重新确认 + 一键“全部通过并触发整合”
- 章节缺口补充：支持“材料线索”与“直接观点”
- 最终阶段：自动生成 open gaps 研究议程，并执行全篇连贯性整合
- 引用保护：若整合后引用/证据标签显著丢失，自动回退安全版本

## 目录结构（精简）

```text
.
├── config/            # 配置（rag_config.json + settings.py）
├── src/
│   ├── api/           # FastAPI 路由
│   ├── llm/           # LLMManager + tools + react_loop
│   ├── retrieval/     # 混合检索 + web 聚合
│   ├── collaboration/ # canvas/memory/intent/research/workflow
│   ├── indexing/      # embed + Milvus
│   ├── parser/        # PDF 解析
│   ├── graph/         # HippoRAG
│   ├── mcp/           # MCP Server
│   └── observability/
├── frontend/          # React 前端
├── scripts/           # 运行与测试脚本
├── tests/             # pytest 测试
└── docs/              # 文档中心
```

## 文档导航

- 文档总览：`docs/README.md`
- 开发总指南：`docs/developer_guide.md`
- 架构说明：`docs/architecture.md`
- API 参考：`docs/api_reference.md`
- 配置说明：`docs/configuration.md`
- 脚本说明：`docs/scripts_guide.md`
- 运维与排障：`docs/operations_and_troubleshooting.md`
- Ubuntu 发布迁移：`docs/release_migration_ubuntu.md`
- 评测与测试：`docs/testing_and_evaluation.md`
- 依赖矩阵：`docs/dependency_matrix.md`

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

## License

MIT
