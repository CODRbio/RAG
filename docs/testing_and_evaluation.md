# 测试与评测

本文档说明项目的自动化测试与效果评测入口。

更新时间：2026-02-19

## 一、单元测试（pytest）

测试目录：`tests/`

| 测试文件 | 覆盖范围 |
|---|---|
| `test_chunker.py` | 切块策略 |
| `test_hybrid_retriever.py` | 混合检索 |
| `test_intent_parser.py` | 意图解析 |
| `test_citation_resolution.py` | 引用解析 |
| `test_tools_routing.py` | 工具路由 |
| `test_tracing_modes.py` | 追踪模式 |
| `test_research_meta_analysis_guards.py` | 研究元分析防护 |
| `test_research_agent_state_compaction.py` | Agent 状态压缩 |

执行：

```bash
pytest -q
```

详细输出：

```bash
pytest -v
```

指定测试文件：

```bash
pytest tests/test_hybrid_retriever.py -v
```

## 二、脚本级功能测试

| 脚本 | 覆盖范围 |
|---|---|
| `scripts/04_test_search.py` | 本地检索 |
| `scripts/05_test_rag.py` | RAG 端到端 |
| `scripts/07_test_web_search.py` | Tavily 网络搜索 |
| `scripts/09_test_google_search.py` | Google / Scholar |
| `scripts/10_test_multiturn.py` | 多轮对话 |
| `scripts/11_test_llm_providers.py` | LLM provider 切换 |
| `scripts/12_test_workflow_stage.py` | 工作流状态 |
| `scripts/13_test_canvas_api.py` | Canvas API |
| `scripts/14_test_memory.py` | 记忆系统 |
| `scripts/15_test_citations.py` | 引用管理 |
| `scripts/16_test_export.py` | 导出功能 |
| `scripts/17_test_chat_stream.py` | SSE 流式聊天 |
| `scripts/21_test_intent_override.py` | 意图覆盖 |
| `scripts/22_test_offline_models.py` | 离线模型 |
| `scripts/23_test_deep_research_e2e.py` | Deep Research 端到端 |
| `scripts/27_test_year_window_and_citation_style.py` | 年份窗口与引用风格 |
| `scripts/test_chat_hybrid_optimizer.py` | 混合检索优化 |
| `scripts/_test_ncbi_integration.py` | NCBI 集成 |

## 三、评测体系

评测脚本：`scripts/18_eval_rag.py`

默认数据集：`data/eval_mini.json`

关注指标（由评测模块输出）：

- 检索：Recall / Hit 类指标
- 生成：ROUGE / 文本重叠类指标
- 引用：引用命中与有效性

评测数据集生成：

```bash
python scripts/24_generate_eval_dataset.py
```

## 四、推荐回归流程

每次较大改动后建议执行：

### 最小回归集

```bash
pytest -q
python scripts/04_test_search.py
python scripts/05_test_rag.py
python scripts/13_test_canvas_api.py
python scripts/17_test_chat_stream.py
```

### 效果回归

```bash
python scripts/18_eval_rag.py
```

### 前端构建验证

```bash
cd frontend && npm run build && cd ..
```

### 完整回归（发布前）

```bash
pytest -q
python scripts/04_test_search.py
python scripts/05_test_rag.py
python scripts/10_test_multiturn.py
python scripts/13_test_canvas_api.py
python scripts/15_test_citations.py
python scripts/17_test_chat_stream.py
python scripts/18_eval_rag.py
cd frontend && npm run build && cd ..
```

## 五、评测数据集扩展建议

- 固定领域子集（同主题论文）与开放问题集分开维护
- 对每条样本同时标注：
  - 期望命中文档
  - 关键事实点
  - 可接受答案范围
- 对高价值样本建立长期基线，避免回归退化
- 建议定期从实际使用中提取新样本补充数据集
