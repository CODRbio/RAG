# 配置说明

本文档描述配置文件与环境变量的加载逻辑、关键配置块和推荐实践。

## 一、配置来源与优先级

配置入口：`config/settings.py`

- 主配置：`config/rag_config.json`
- 本地覆盖：`config/rag_config.local.json`（可选）
- 环境变量覆盖（敏感配置优先）

加载顺序：

1. 读取 `rag_config.json`
2. 若存在 `rag_config.local.json`，深度合并覆盖
3. 部分字段再由环境变量覆盖（如 API Key/端口）

## 二、关键配置块（`rag_config.json`）

### `llm`

- `default`：默认 provider
- `providers`：各 provider 的 `api_key/base_url/default_model/models/params`
- 支持 provider：`openai`、`deepseek`、`gemini`、`claude`、`kimi`、`sonar` 及 thinking/vision 变体

### `deep_research`

- `depth_presets`：`lite` / `comprehensive` 的完整阈值集合
  - 迭代预算：`max_iterations_per_section`、`max_section_research_rounds`
  - 覆盖阈值：`coverage_threshold`
  - 查询预算：`recall_queries_per_section`、`precision_queries_per_section`
  - 分层召回：`search_top_k_first` / `search_top_k_gap` / `search_top_k_write`
  - 写作二次取证：`verification_k`（Writing phase 的数据点/引用复核检索窗口）
  - 自校正补检索：`self_correction_trigger_coverage`、`self_correction_min_round`、`search_top_k_gap_decay_factor`、`search_top_k_gap_min`
  - 收益曲线早停：`coverage_plateau_floor`、`coverage_plateau_min_gain`
  - 验证分层：`verify_light_threshold` / `verify_medium_threshold` / `verify_severe_threshold`
  - 审核门：`review_gate_max_rounds`、`review_gate_base_sleep`、`review_gate_max_sleep`、`review_gate_early_stop_unchanged`
  - 图上限：`recursion_limit`
  - 成本监控：`cost_warn_steps`、`cost_force_summary_steps`、`cost_tick_interval`

### `parser`

- PDF 解析相关阈值
- `llm_text_provider`、`llm_vision_provider`
- `enrich_tables`、`enrich_figures`

### `chunk`

- 切块粒度与上下限：`target_chars/min_chars/max_chars`
- `overlap_sentences`

### `search`

- `top_k`、`rerank_top_k`
- Dense/Sparse 召回参数
- `reranker_mode`
- ColBERT 开关与模型配置

### `web_search` / `google_search` / `semantic_scholar`

- 多源网络检索开关、并发、超时、搜索优化

### `content_fetcher`

- 全文抓取策略开关与缓存

### `performance`

- `retrieval`、`llm`、`web_search`、`unified_web_search`、`google_search`
- 用于统一控制超时、重试、并发和缓存

### `storage`

- `max_age_days`、`max_size_gb`
- `cleanup_on_startup`、`cleanup_batch_size`

### `auth`

- `secret_key`
- `token_expire_hours`
- 初始化管理员账户字段

## 三、重要环境变量

- `RAG_LLM__{PROVIDER}__API_KEY`：覆盖对应 provider key（推荐）
- 兼容旧变量：`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`
- `RAG_LLM__SONAR__API_KEY`：Perplexity Sonar key（推荐优先用该变量）
- `API_HOST`、`API_PORT`：API 监听地址
- `MILVUS_HOST`、`MILVUS_PORT`：Milvus 地址
- `HF_LOCAL_FILES_ONLY`：离线加载 HuggingFace 模型
- `MODEL_CACHE_ROOT`、`EMBEDDING_CACHE_DIR`、`RERANKER_CACHE_DIR`、`COLBERT_CACHE_DIR`

## 四、推荐配置实践

- `rag_config.json` 存结构化配置，不写真实密钥
- `rag_config.local.json` 存开发机私密配置，不提交仓库
- CI/服务器使用环境变量注入密钥
- 修改配置后优先检查：
  - `/health/detailed`
  - `GET /llm/providers`
  - `GET /models/status`

## 五、示例（最小可运行）

```json
{
  "llm": {
    "default": "deepseek",
    "providers": {
      "deepseek": {
        "api_key": "sk-xxx",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": {"deepseek-chat": "deepseek-chat"},
        "params": {}
      }
    }
  },
  "api": {"host": "127.0.0.1", "port": 9999},
  "search": {"top_k": 20}
}
```
