# 脚本指南

本文档按使用场景组织 `scripts/` 下脚本，帮助快速定位执行顺序与用途。

更新时间：2026-02-19

## 一、环境与基础设施

| 脚本 | 用途 |
|---|---|
| `00_preflight_check.sh` | 环境预检（Python、Docker、端口等） |
| `00_healthcheck_docker.sh` | Milvus / MinIO / etcd 健康检查 |
| `01_init_env.py` | 初始化 Milvus Collections |
| `verify_dependencies.sh` | 一键核对 Python 依赖 + 前端 Node 运行时 |

## 二、离线数据流水线

| 脚本 | 用途 |
|---|---|
| `02_parse_papers.py` | 解析 PDF 生成结构化数据 |
| `03_index_papers.py` | 切块、向量化并写入 Milvus |
| `03b_build_graph.py` | 构建 HippoRAG 图谱 |
| `06_ingest_langgraph.py` | 基于 LangGraph 的入库流程 |

## 三、服务启动

| 脚本 | 用途 |
|---|---|
| `start.sh` | 一键启动后端 + 前端（`--backend-only` / `--frontend-only`） |
| `08_run_api.py` | 只启动 FastAPI（支持 `--host --port --reload`） |

## 四、检索与功能测试脚本

| 脚本 | 用途 |
|---|---|
| `04_test_search.py` | 本地检索测试 |
| `05_test_rag.py` | RAG 端到端测试 |
| `07_test_web_search.py` | Tavily 网络搜索测试 |
| `09_test_google_search.py` | Google / Scholar 搜索测试 |
| `10_test_multiturn.py` | 多轮对话测试 |
| `11_test_llm_providers.py` | LLM provider 切换测试 |
| `12_test_workflow_stage.py` | 工作流状态测试 |
| `13_test_canvas_api.py` | Canvas API 测试 |
| `14_test_memory.py` | 记忆系统测试 |
| `15_test_citations.py` | 引用管理测试 |
| `16_test_export.py` | 导出功能测试 |
| `17_test_chat_stream.py` | SSE 流式聊天测试 |
| `21_test_intent_override.py` | 意图覆盖测试 |
| `23_test_deep_research_e2e.py` | Deep Research 端到端测试 |
| `27_test_year_window_and_citation_style.py` | 年份窗口与引用风格测试 |
| `test_chat_hybrid_optimizer.py` | 混合检索优化测试 |
| `_test_ncbi_integration.py` | NCBI 集成测试 |

## 五、评测与运维脚本

| 脚本 | 用途 |
|---|---|
| `18_eval_rag.py` | RAG 评测入口 |
| `19_cleanup_storage.py` | 存储清理与压缩（`--vacuum`） |
| `20_bootstrap_admin.py` | 首次管理员初始化 |
| `22_test_offline_models.py` | 离线模型检查 |
| `23_sync_local_models.py` | 本地模型同步 |
| `24_generate_eval_dataset.py` | 生成评测数据集 |
| `25_extract_claims.py` | 声明提取 |
| `26_backfill_doi.py` | DOI 元数据回填 |

## 六、调试与演示脚本

| 脚本 | 用途 |
|---|---|
| `08_test_logging.py` | 日志系统测试 |
| `debug_chat.py` | 聊天调试工具 |
| `demo_llm_manager.py` | LLMManager 功能演示 |
| `check_sonar_health.py` | Sonar 健康检查 |

## 七、推荐执行顺序

### 冷启动（首次）

```bash
bash scripts/00_preflight_check.sh
docker compose --profile dev up -d
bash scripts/00_healthcheck_docker.sh
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
python scripts/20_bootstrap_admin.py
bash scripts/start.sh
```

### 功能回归（开发中）

```bash
pytest -q
python scripts/04_test_search.py
python scripts/05_test_rag.py
python scripts/13_test_canvas_api.py
python scripts/17_test_chat_stream.py
python scripts/18_eval_rag.py
cd frontend && npm run build && cd ..
```

### 数据维护

```bash
python scripts/19_cleanup_storage.py --vacuum
python scripts/26_backfill_doi.py
python scripts/23_sync_local_models.py
```

## 八、常见注意事项

- 需要浏览器自动化的脚本（Google / Scholar）先执行：`playwright install chromium`
- 并发或网络不稳定时，优先调小配置中的 timeout / concurrency 参数
- 清理数据前，先备份 `data/` 与 `src/data/sessions.db`
- Deep Research 端到端测试（`23_test_deep_research_e2e.py`）耗时较长，建议在非高峰时段运行
- 脚本编号可能不连续，属于历史遗留，不影响使用
