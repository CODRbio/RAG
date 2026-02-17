# 测试与评测

本文档说明项目的自动化测试与效果评测入口。

## 一、单元测试（pytest）

测试目录：`tests/`

- `test_chunker.py`
- `test_hybrid_retriever.py`
- `test_intent_parser.py`

执行：

```bash
pytest -q
```

## 二、脚本级功能测试

- 检索：`python scripts/04_test_search.py`
- 端到端：`python scripts/05_test_rag.py`
- 流式聊天：`python scripts/17_test_chat_stream.py`
- 画布 API：`python scripts/13_test_canvas_api.py`
- 多轮对话：`python scripts/10_test_multiturn.py`

## 三、评测体系

评测脚本：`scripts/18_eval_rag.py`

默认数据集：`data/eval_mini.json`

关注指标（由评测模块输出）：

- 检索：Recall/Hit 类指标
- 生成：ROUGE/文本重叠类指标
- 引用：引用命中与有效性

## 四、推荐回归流程

每次较大改动后建议执行：

```bash
pytest -q
python scripts/04_test_search.py
python scripts/05_test_rag.py
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
