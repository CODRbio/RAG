# DeepSea RAG 开发指南

本指南面向接手开发的工程师，聚焦"如何在当前项目里正确开发和扩展"。

## 1. 先读什么

建议阅读顺序：

1. `../README.md`（项目入口）
2. `architecture.md`（架构与数据流）
3. `api_reference.md`（接口边界）
4. `configuration.md`（配置与环境变量）
5. `scripts_guide.md`（脚本与执行顺序）
6. `dependency_matrix.md`（依赖矩阵与运行时要求）

## 2. 代码分层约定

```text
src/
├── api/              # 接口层：HTTP 路由，请求/响应模型
├── collaboration/    # 协作层：canvas / memory / intent / research / workflow / citation / export
├── llm/              # Agent 层：LLMManager / tools / react_loop
├── retrieval/        # 检索层：混合检索 / web 聚合 / 重排 / 去重
├── parser/           # 数据处理：PDF 解析
├── chunking/         #            结构化切块
├── indexing/         #            向量化与 Milvus 读写
├── generation/       # 生成层：证据综合 / 上下文打包 / LLM 兼容层
├── graph/            # 图谱：HippoRAG
├── graphs/           # 流水线：LangGraph 入库图
├── auth/             # 认证：session / password
├── observability/    # 可观测：metrics / tracing / middleware
├── evaluation/       # 评测：runner / metrics / dataset
├── mcp/              # MCP Server
├── utils/            # 工具：缓存 / 限流 / 清理 / 提示词 / 任务运行器
├── log/              # 日志管理
└── prompts/          # LLM 提示词模板
```

保持依赖方向：上层调用下层，避免反向耦合。

## 3. LLM 调用规范（强约束）

所有 LLM 调用必须走 `src/llm/llm_manager.py`。

### 推荐写法

```python
from src.llm import LLMManager

manager = LLMManager.from_json("config/rag_config.json")
client = manager.get_client("deepseek")
resp = client.chat(
    messages=[
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "问题内容"},
    ],
    model=None,       # 可选：覆盖默认模型
    max_tokens=2000,  # 可选：覆盖默认参数
)
text = resp["final_text"]        # 最终回答
reasoning = resp["reasoning_text"]  # 思考过程（thinking 模式）
usage = resp["meta"]["usage"]    # token 用量
```

### 兼容写法（旧代码）

```python
from src.generation.llm_client import call_llm

result = call_llm(
    provider="deepseek",
    system="系统提示",
    user_prompt="用户问题",
)
# result 为字符串（final_text）
```

### 可用 Providers

- `openai` / `openai-thinking`
- `deepseek` / `deepseek-thinking`
- `gemini` / `gemini-thinking` / `gemini-vision`
- `claude` / `claude-thinking`
- `kimi` / `kimi-thinking` / `kimi-vision`
- `sonar`

### 禁止做法

- 直接实例化 `openai.OpenAI()` / `anthropic.Anthropic()`
- 业务代码里硬编码 API Key
- 绕过 LLMManager 直接发 HTTP 请求

## 4. Agent 与工具扩展

核心文件：

- `src/llm/tools.py`：工具定义与路由
- `src/llm/react_loop.py`：ReAct 循环
- `src/mcp/server.py`：MCP Server

新增工具步骤：

1. 在 `tools.py` 定义 `ToolDef`
2. 将工具加入核心工具列表
3. 在 `mcp/server.py` 注册对应 MCP tool
4. 必要时补充前端工具轨迹展示（`ToolTracePanel.tsx`）

## 5. API 开发约定

- 新路由统一放 `src/api/routes_*.py`
- 在 `src/api/server.py` 注册 `include_router`
- 请求/响应模型优先放 `src/api/schemas.py`
- 新增接口后同步更新：
  - `docs/api_reference.md`
  - 前端 `frontend/src/api/*` 客户端
  - 前端 `frontend/src/types/index.ts` 类型

## 6. 配置与密钥

- 公共配置：`config/rag_config.json`
- 本地覆盖：`config/rag_config.local.json`（gitignored）
- 环境变量覆盖：`RAG_LLM__{PROVIDER}__API_KEY`

新增配置字段时必须同步：

- `config/rag_config.json`
- `config/rag_config.example.json`
- `docs/configuration.md`

## 7. 数据与存储

| 路径 | 用途 |
|---|---|
| `data/raw_papers/` | 原始 PDF 文档 |
| `data/parsed/` | 解析后结构化数据 |
| `src/data/sessions.db` | 会话数据库 |
| `src/data/deep_research_jobs.db` | Deep Research 任务数据库 |
| `logs/` | 运行日志 |
| `logs/llm_raw/` | LLM 原始响应日志（JSONL） |
| `artifacts/` | 评测/任务产物 |

上线前确认：

- `storage.max_age_days`
- `storage.max_size_gb`
- `storage.cleanup_on_startup`

## 8. 前端联动约定

当你改后端协议时，至少检查：

- `frontend/src/api/*` 调用参数与返回类型
- `frontend/src/types/index.ts` 类型定义
- `frontend/src/stores/*` 状态处理
- SSE 事件处理组件（聊天与研究相关）
- `frontend/src/i18n/locales/*.json` 翻译文件

## 9. 测试与回归

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

## 10. 常见扩展场景

### 新增 LLM Provider

1. 配置 `rag_config.json` 的 `llm.providers`
2. 若非 OpenAI 兼容协议，在 `llm_manager.py` 增加 provider 适配
3. 补充文档与测试脚本

### 新增检索源

1. `src/retrieval/` 新增 searcher
2. 在 `unified_web_search.py` 注册聚合逻辑
3. 在工具层与 API 层暴露可选参数
4. 更新 `docs/configuration.md`

### 新增 API 业务域

1. 新建 `routes_xxx.py`
2. 在 `server.py` 注册 router
3. schemas 中补齐模型
4. 前端加 API 封装与页面入口
5. 更新文档与测试

### 新增提示词模板

1. 在 `src/prompts/` 新增 `.txt` 模板文件
2. 通过 `src/utils/prompt_manager.py` 加载
3. 在业务代码中引用

## 11. 文档维护职责

任何业务变更至少更新 1 份文档；跨模块变更更新 2 份以上。优先维护：

- `../README.md`
- `api_reference.md`
- `configuration.md`
- `scripts_guide.md`

这可以保证新成员可以在 1 小时内完成项目冷启动与关键功能定位。
