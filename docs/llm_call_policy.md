# LLM 调用策略文档

> 本文档说明系统中所有 LLM 调用的分类、thinking 降级策略、以及 max_tokens 管理机制。

---

## 1. Thinking 智能降级

当用户在前端选择 thinking 变体（如 `claude-thinking`、`openai-thinking`）时，**并非所有调用都需要 extended thinking**。系统通过 `get_lite_client()` 自动将轻量级调用降级到同平台的基础 provider。

### 降级映射规则

| 用户选择 | 重度调用（写作/分析） | 轻度调用（路由/查询/JSON） |
|---------|-------------------|-----------------------|
| `claude-thinking` | `claude-thinking` | `claude` |
| `openai-thinking` | `openai-thinking` | `openai` |
| `deepseek-thinking` | `deepseek-thinking` | `deepseek` |
| `gemini-thinking` | `gemini-thinking` | `gemini` |
| `kimi-thinking` | `kimi-thinking` | `kimi` |
| `qwen-thinking` | `qwen-thinking` | `qwen` |
| `claude`（非 thinking） | `claude` | `claude`（不变） |

---

## 1.5 各厂商 Thinking 参数注入规则（基于官方文档）

Thinking 通过 provider 变体的 `params` 注入，同一个模型 ID 通过参数切换 thinking 开关。

| Provider | 参数格式 | 文档来源 |
|----------|---------|---------|
| **OpenAI** (`openai-thinking`) | `{"reasoning_effort": "high"}` | [OpenAI Reasoning Guide](https://platform.openai.com/docs/guides/reasoning) |
| **Claude** (`claude-thinking`) | `{"thinking": {"type": "enabled", "budget_tokens": 10000}}` | [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) — `anthropic-version: 2023-06-01` 下使用 `enabled` + `budget_tokens`。`adaptive` + `effort` 需要更新的 API 版本 |
| **Gemini** (`gemini-thinking`) | `{"reasoning_effort": "high"}` | [Gemini Thinking](https://ai.google.dev/gemini-api/docs/thinking) — OpenAI-compat 端点用 `reasoning_effort` |
| **DeepSeek** (`deepseek-thinking`) | `{"thinking": {"type": "enabled"}}` | [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode) — `deepseek-chat` + 此参数 = `deepseek-reasoner` 等效 |
| **Kimi** (`kimi-thinking`) | `{"thinking": {"type": "enabled"}}` | [Moonshot API Docs](https://platform.moonshot.ai/docs/api/chat) — `kimi-k2.5` 模型支持 `thinking` 参数 |
| **Qwen** (`qwen-thinking`) | `{"enable_thinking": true}` | [DashScope OpenAI Compatible](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) — qwen3 系列通过 `enable_thinking` 开启 |

### 实现

- `LLMManager.get_lite_client(provider)` — 将 `-thinking` 变体映射到基础 provider
- `_resolve_step_lite_client(state, step)` — Deep Research agent 内部使用

---

## 2. 调用分类清单

### 轻量级调用 — 使用 `lite_client`（自动降级 thinking）

| 文件 | 函数/位置 | 用途 | 说明 |
|------|----------|------|------|
| `routes_chat.py` | `_classify_query()` | Chat/RAG 路由分类 | 输出一个词：chat 或 rag |
| `routes_chat.py` | `detect_intent()` | 独立意图检测 API | 简单二分类 |
| `routes_chat.py` | `IntentParser(lite_client)` | 命令解析 | 解析 /xxx 命令 |
| `agent.py` | `_repair_queries()` | 搜索查询修复 | 修正格式，几行文本 |
| `agent.py` | `_generate_refined_queries()` | 精化查询生成 | 生成关键词短语 |
| `agent.py` | `_generate_search_queries()` | 召回/精确/发现查询 | 生成搜索查询列表 |
| `agent.py` | `scope_node()` | 研究范围界定 | JSON 结构化输出 |
| `agent.py` | `_quick_coverage_check()` | 快速覆盖率检查 | JSON 判断 |
| `agent.py` | `evaluate_node()` | 覆盖率评估 | JSON 结构化输出 |
| `intent/parser.py` | `IntentParser.parse()` | 意图分类 | 简单分类任务 |
| `intent/commands.py` | `generate_focused_query()` | 生成聚焦搜索查询 | 一行查询 |
| `memory/session_memory.py` | `rolling_summary()` | 对话摘要 | 短摘要 |
| `memory/working_memory.py` | `generate_progress()` | 进度摘要 | 短摘要 |
| `research/trajectory.py` | `compress()` | 轨迹压缩 | 上下文管理 |
| `research/verifier.py` | `extract_claims()` / `verify()` | 声明提取与验证 | JSON 结构化 |
| `smart_query_optimizer.py` | `optimize()` / `routing_plan()` | 查询优化 | JSON 结构化 |
| `web_search.py` | `expand_queries()` | 查询扩展 | 短列表 |
| `web_content_fetcher.py` | `fetch_decision()` | 抓取决策 | JSON 二选一 |
| `parser/claim_extractor.py` | `extract()` | 声明提取 | JSON 结构化 |
| `routes_compare.py` | `compare()` | 文档比较 | JSON 结构化 |

### 重度调用 — 保持用户选择的 thinking provider

| 文件 | 函数/位置 | 用途 | 说明 |
|------|----------|------|------|
| `routes_chat.py` | 主 chat 生成 | 直接对话回复 | 用户面对的主输出 |
| `routes_chat.py` | Deep Research 澄清问题 | 生成澄清问题 | 需要深度理解 |
| `routes_canvas.py` | 文档精修 / 段落编辑 | Canvas 编辑 | 长文本生成 |
| `agent.py` | `plan_node()` | 大纲规划 | 创造性规划 |
| `agent.py` | `generate_claims_node()` | 论点提炼 | 带引用的推理 |
| `agent.py` | `write_node()` | 章节写作 | 主内容生成 |
| `agent.py` | `verify_node()` | 验证与修正 | 需要深度推理 |
| `agent.py` | `synthesize_node()` | 全文合成 | 摘要/Limitations/全文连贯性 |
| `react_loop.py` | ReAct 循环 | Agent 工具调用 | 需要推理和工具使用 |
| `auto_complete.py` | 章节/摘要生成 | 自动补全 | 内容生成 |
| `pdf_parser.py` | 表格/图片富化 | PDF 解析 | 内容理解 |

---

## 3. max_tokens 管理策略

### 原则：不在调用方设置 max_tokens

系统不再在各个调用点硬编码 `max_tokens`。由 `llm_manager.py` 中的中央保护层统一处理：

| 场景 | 行为 |
|------|------|
| Anthropic API | 必须有正整数 → 默认 `16384` |
| Anthropic thinking enabled | `max_tokens = max(传入值, budget_tokens + 2000)` — 自动保护 thinking 预算 |
| OpenAI reasoning_effort + 传入值 < 8000 | 自动移除限制，让 API 自主分配 |
| 其他情况 | 不传 `max_tokens`，让 API 用自己的默认值 |

### 例外：保留 max_tokens 的场景

| 位置 | 值 | 原因 |
|------|---|------|
| `routes_canvas.py` 全文精修 | `5000` | 防止完整文档重写失控 |
| `agent.py` coherence 窗口 | 动态 `output_budget` / `win_budget` | 基于 context window 智能计算 |
| `pdf_parser.py` | 从 config 读取 | 用户可配置的解析预算 |

---

## 4. Provider 与 Platform 架构

```
rag_config.json
├── llm.platforms      ← 7 个平台（api_key + base_url）
│   ├── openai
│   ├── deepseek
│   ├── gemini
│   ├── claude
│   ├── kimi
│   ├── perplexity
│   └── qwen
└── llm.providers      ← 变体（继承 platform 的 key/url，定义 params 差异）
    ├── openai          ← platform: openai
    ├── openai-thinking ← platform: openai, params: {reasoning_effort: "high"}
    ├── claude          ← platform: claude
    ├── claude-thinking ← platform: claude, params: {thinking: {type: "enabled", ...}}
    └── ...
```

`rag_config.local.json` 只需覆盖 platforms 的 api_key（7 行），所有变体自动继承。
