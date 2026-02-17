# 脚本指南

本文档按使用场景组织 `scripts/` 下脚本，帮助快速定位执行顺序与用途。

## 一、环境与基础设施

- `00_preflight_check.sh`：环境预检（Python、Docker、端口等）
- `00_healthcheck_docker.sh`：Milvus/MinIO/etcd 健康检查
- `01_init_env.py`：初始化 Milvus Collections
- `verify_dependencies.sh`：一键核对 `deepsea-rag` Python 依赖 + 前端 Node 运行时

## 二、离线数据流水线

- `02_parse_papers.py`：解析 PDF 生成结构化数据
- `03_index_papers.py`：切块、向量化并写入 Milvus
- `03b_build_graph.py`：构建 HippoRAG 图谱
- `06_ingest_langgraph.py`：基于 LangGraph 的入库流程

## 三、服务启动

- `start.sh`：一键启动后端 + 前端
  - `--backend-only`：仅后端
  - `--frontend-only`：仅前端
- `08_run_api.py`：只启动 FastAPI（支持 `--host --port --reload`）

## 四、检索与功能测试脚本

- `04_test_search.py`：检索测试
- `05_test_rag.py`：RAG 端到端测试
- `07_test_web_search.py`：Tavily 测试
- `09_test_google_search.py`：Google/Scholar 测试
- `10_test_multiturn.py`：多轮对话
- `11_test_llm_providers.py`：LLM provider 切换
- `12_test_workflow_stage.py`：工作流状态测试
- `13_test_canvas_api.py`：Canvas API
- `14_test_memory.py`：记忆系统
- `15_test_citations.py`：引用管理
- `16_test_export.py`：导出
- `17_test_chat_stream.py`：SSE 流式聊天
- `21_test_intent_override.py`：意图覆盖测试
- `test_chat_hybrid_optimizer.py`：混合优化相关测试

## 五、评测与运维脚本

- `18_eval_rag.py`：评测入口
- `19_cleanup_storage.py`：存储清理与压缩
- `20_bootstrap_admin.py`：首次管理员初始化
- `22_test_offline_models.py`：离线模型检查
- `23_sync_local_models.py`：本地模型同步
- `24_generate_eval_dataset.py`：生成评测数据集
- `25_extract_claims.py`：声明提取
- `08_test_logging.py`：日志测试
- `debug_chat.py`：聊天调试
- `demo_llm_manager.py`：LLMManager 演示

## 六、推荐执行顺序

### 冷启动（首次）

```bash
bash scripts/00_preflight_check.sh
docker compose --profile dev up -d
bash scripts/00_healthcheck_docker.sh
python scripts/01_init_env.py
python scripts/02_parse_papers.py
python scripts/03_index_papers.py
python scripts/03b_build_graph.py
bash scripts/start.sh
```

### 功能回归（开发中）

```bash
python scripts/04_test_search.py
python scripts/05_test_rag.py
python scripts/13_test_canvas_api.py
python scripts/17_test_chat_stream.py
python scripts/18_eval_rag.py
```

## 七、常见注意事项

- 需要浏览器自动化的脚本（Google/Scholar）先执行：`playwright install chromium`
- 并发或网络不稳定时，优先调小配置中的 timeout/concurrency 参数
- 清理数据前，先备份 `data/` 与 `src/data/sessions.db`
