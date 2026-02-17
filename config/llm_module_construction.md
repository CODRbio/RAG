```markdown
# Cursor 开发指导（最终版）：单文件 `llm_manager.py` —— Provider 统一封装 + final/reasoning 隔离 + Raw JSON 落库与清理

## English query prompt (for Cursor)
Build a single-file Python module `llm_manager.py` that loads an LLM provider configuration from JSON (same schema as rag_config.example.json), implements a unified Provider base class with subclasses for OpenAI-compatible and Anthropic, deep-merges default params with per-call overrides, resolves model aliases via `models` mapping, supports API key env override, supports dry-run mode, always stores raw JSON responses to JSONL logs, and normalizes outputs by returning `final_text` by default while extracting `reasoning_text` (best-effort) without mixing it into `final_text`. Add log retention: delete logs older than 10 days and keep total log size under 100MB by deleting oldest files.

---

## 0. 背景/输入配置（必须兼容）

配置文件结构参考 `rag_config.example.json`，关键点如下：

- 默认 provider：`llm.default = "claude"` [1]
- dry_run 开关：`llm.dry_run = false` [1]
- providers 里包含多种 profile（如 `openai-thinking`, `claude-thinking`, `gemini-thinking`, `kimi-vision` 等）[1]
- provider 配置字段：
  - `api_key`, `base_url`, `default_model`, `models`（alias -> real model），`params`（默认参数）[1]
- thinking 相关默认参数示例：
  - `openai-thinking.params.reasoning_effort = "high"` [1]
  - `claude-thinking.params.thinking = {"type":"enabled","budget_tokens":16000}` 且 `max_tokens=32000` [1]
  - `gemini-thinking.params.thinkingConfig.thinkingBudget=16000`（走 OpenAI-compatible base_url）[1]
  - `kimi-thinking.params.enable_reasoning=true`, `reasoning_effort="high"` [1]
- 视觉/结构化输出示例：
  - `kimi-vision.params.response_format={"type":"json_object"}` [1]

---

## 1. 总目标（MVP）

做一个统一的 LLM 管理器，后续用于 LangGraph/RAG 节点调用，要求：

1) 单文件实现：`llm_manager.py`  
2) Provider 基类统一封装：不同模型/服务实现 `request()`  
3) 支持 OpenAI-compatible 与 Anthropic 两类 HTTP 客户端  
4) **输出必须规范化**：默认只返回最终答案 `final_text`；如果响应包含思考过程，则尽力抽取到 `reasoning_text`，但绝不混入 `final_text`  
5) **Raw JSON 必须落库**（jsonl），并做定量清理：总大小 >100MB 删除旧文件；超过 10 天删除旧文件  
6) API key 支持环境变量覆盖，且严禁明文打印 key  

---

## 2. 文件交付

- 必须：`llm_manager.py`
- 可选：`demo_llm_manager.py`（简单演示加载配置与 dry_run）

---

## 3. 统一 API 设计

### 3.1 `LLMManager`
必须实现：

- `LLMManager.from_json(path: str) -> LLMManager`
- `get_provider_names() -> list[str]`
- `get_client(provider: str | None = None, api_key: str | None = None) -> BaseChatClient`
- `resolve_model(provider: str, model: str | None) -> str`
  - model=None => provider.default_model [1]
  - model 在 `provider.models` 映射中 => 用映射值 [1]
  - 否则直接用传入 model

### 3.2 `BaseChatClient.chat()`
统一消息格式（OpenAI 风格）：

```python
messages = [{"role": "user", "content": "hello"}]
```

统一调用：

```python
resp = client.chat(
  messages=messages,
  model=None,                 # 可选：alias 或真实模型名
  return_reasoning=False,     # 默认 False：业务层只用 final_text
  **overrides                 # 覆盖 params
)
```

---

## 4. 输出规范化（你必须实现这层）

### 4.1 统一返回结构（所有 client 一致）
`chat()` 返回 dict：

```python
{
  "provider": "claude-thinking",
  "model": "resolved-model-name",
  "final_text": "...",            # 业务层默认使用
  "reasoning_text": None | "...", # 仅旁路信息，不混入 final_text
  "raw": {...},                   # 完整原始 JSON（必须）
  "params": {...},                # 实际使用的 merged params（必须）
  "meta": {
      "usage": {...} | None,
      "latency_ms": int | None,
      "refusal": bool | None
  }
}
```

### 4.2 `normalize_response(provider_name, raw)`（必须）
实现一个 best-effort 的提取器：

```python
def normalize_response(provider_name: str, raw: dict) -> dict:
    """
    Returns:
      final_text: str|None
      reasoning_text: str|None
      usage: dict|None
      refusal: bool|None
    """
```

要求：
- `final_text`：只抽取最终答案（可展示）
- `reasoning_text`：尽力抽取（如果有），但不影响 final_text
- 字段缺失时必须容错，不允许抛 KeyError
- 抽不到 reasoning_text 是允许的，因为 raw 会落库

#### OpenAI-compatible（/chat/completions）建议抽取规则
- `final_text`：优先 `choices[0].message.content`（str）  
  - 如果是 list（多段），只拼接 `type=="text"` 段落
- `reasoning_text`：多候选探测（抽不到也 OK）
  - `choices[0].message.reasoning` / `thoughts`（若存在）
  - content 为 list 时，拼接 `type in ("reasoning","thinking")` 段落

#### Anthropic（/v1/messages）建议抽取规则
- `final_text`：拼接 `raw["content"]` 中 `type=="text"` 的 `.text`
- `reasoning_text`：如果 content 中有 `type=="thinking"` 或类似块，抽取其文本；否则 None

---

## 5. Provider 基类与两类实现（你要求的统一封装）

在单文件中实现：

```python
class Provider:
    def request(self, payload: dict) -> dict:
        raise NotImplementedError
```

子类：
- `OpenAICompatProvider(Provider)`：POST `{base_url}/chat/completions`
  - Header：`Authorization: Bearer {api_key}`
- `AnthropicProvider(Provider)`：POST `{base_url}/v1/messages`
  - Headers：`x-api-key`, `anthropic-version: 2023-06-01`

然后 `ChatClient` 调用 `provider.request()`，拿到 raw，再走 `normalize_response()`。

> 注：`gemini/deepseek/kimi` 在示例里是 OpenAI-compatible base_url，可统一走 `OpenAICompatProvider` [1]。`claude/claude-thinking` 走 `AnthropicProvider` [1]。

---

## 6. 参数合并（必须深合并）

### 6.1 默认参数来源
- provider `params` 作为默认 [1]
- `chat()` 的 `**overrides` 覆盖默认

### 6.2 深合并实现
实现：

```python
def deep_merge(base: dict, override: dict) -> dict:
    # recursive dict merge
```

必须支持嵌套对象：
- `thinking`, `thinkingConfig`, `response_format` 等 [1]

---

## 7. API key 覆盖策略（必须）

优先级：
1) `get_client(..., api_key=...)`
2) 环境变量：`RAG_LLM__{PROVIDER}__API_KEY`
   - provider 名称转大写
   - `-` 替换为 `_`
   - 例：`claude-thinking` => `RAG_LLM__CLAUDE_THINKING__API_KEY`
3) JSON 里的 `api_key` [1]

安全：
- 不允许在 print/log/异常里输出完整 key
- 实现 `mask_secret()` 仅保留前后少量字符

---

## 8. dry_run（必须）

如果 `llm.dry_run == true` [1]：
- `LLMManager.get_client()` 返回 `DryRunChatClient`
- DryRun 返回同样结构，raw 里写：
  - `{"dry_run": true, "note": "..."}`
- 仍然执行参数合并与 model 解析，便于测试工作流一致性

---

## 9. Raw JSON 落库 + 清理策略（你要求必须做）

### 9.1 设计：jsonl 按天写文件
实现一个简单的 `RawLogStore`：

- 默认目录：`logs/llm_raw/`
- 文件名：`YYYY-MM-DD.jsonl`
- 每次 `chat()` 成功或失败都要记录（失败记录 error 信息 + request 摘要）

record 建议字段：
- `timestamp`
- `provider`, `model`
- `params`
- `messages_digest`（role + content 前 N 字符；默认 N=200，可配置）
- `final_text`（可选）
- `reasoning_text`（可选，可能很大；默认也可保存）
- `raw_response`（完整 raw JSON）
- `meta`（usage/latency/refusal）
- `error`（如果异常）

### 9.2 清理策略：超过 10 天 + 总大小 100MB
实现：

```python
cleanup(max_age_days: int = 10, max_total_mb: int = 100)
```

规则：
1) 删除超过 10 天的 log 文件
2) 若总大小 > 100MB，从最旧文件开始删除直到小于阈值

默认值必须是：10 days + 100MB（但允许调用者改）。


---

## 10. 建议的单文件代码骨架（Cursor 按这个顺序写）

1) imports / constants（anthropic version、默认超时等）
2) dataclasses：
   - `ProviderConfig`
   - `LLMConfig`
3) helper funcs：
   - `load_json()`
   - `deep_merge()`
   - `mask_secret()`
   - `provider_env_var()`
   - `now_iso()`
4) `RawLogStore`
5) Provider classes：
   - `Provider`
   - `OpenAICompatProvider`
   - `AnthropicProvider`
6) Response normalization：
   - `normalize_response()`
7) Chat clients：
   - `BaseChatClient`
   - `DryRunChatClient`
   - `HTTPChatClient`（内部持有 Provider、config、logstore）
8) `LLMManager`

---

## 11. 验收标准（我会这样验收）

- 能加载配置，默认 provider 为 `claude` [1]
- `resolve_model()` 能将 `claude-sonnet-4-5` 映射到版本化真实模型名（例如 `claude-sonnet-4-5-20250929`）[1]
- `openai-thinking` 会把 `reasoning_effort="high"` 作为默认参数透传 [1]
- `claude-thinking` 会把 `thinking.budget_tokens=16000` 与 `max_tokens=32000` 作为默认参数透传 [1]
- 返回结构永远包含：
  - `final_text`（不含 reasoning）
  - `reasoning_text`（尽力抽取）
  - `raw`（完整）
- Raw 日志会写 jsonl，且清理策略可用（10 天/100MB）

---

## 12. 备注（重要）
- 不要把任何 API key 写进 Git 仓库。
- 对 response 的 reasoning 字段提取只能 best-effort；抽不到不算失败，因为 raw 已保存，可后续离线解析。
- 若某 provider 的返回结构与预期不一致，请在代码中写 TODO 注释，保持可维护。

```

这份 MD 文档已把你的三条关键诉求都“落到可实现的工程要求”上了：
- Provider 基类统一封装（子类实现 request）
- Raw JSON 落库 + 10days/100MB 清理
- thinking 模式“抽取 final_text + 隔离 reasoning_text”，业务层默认只用 final_text，不污染输出

如果你还希望加一条：**`reasoning_text` 默认不写入 messages_digest（避免泄露敏感推理）**，我也可以帮你补一个更严格的脱敏/采样策略。