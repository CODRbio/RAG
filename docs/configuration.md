# 配置说明

本文档描述配置文件与环境变量的加载逻辑、关键配置块和推荐实践。

更新时间：2026-02-19

## 一、配置来源与优先级

配置入口：`config/settings.py`

- 主配置：`config/rag_config.json`
- 本地覆盖：`config/rag_config.local.json`（可选，gitignored）
- 环境变量覆盖（敏感配置优先）

加载顺序：

1. 读取 `rag_config.json`
2. 若存在 `rag_config.local.json`，深度合并覆盖
3. 部分字段再由环境变量覆盖（如 API Key / 端口）

## 二、关键配置块（`rag_config.json`）

### `llm`

LLM 统一调度配置。

- `default`：默认 provider
- `dry_run`：是否启用 Mock 模式（测试用）
- `providers`：各 provider 的 `api_key / base_url / default_model / models / params`
- 支持 provider：
  - `openai` / `openai-thinking`
  - `deepseek` / `deepseek-thinking`
  - `gemini` / `gemini-thinking` / `gemini-vision`
  - `claude` / `claude-thinking`
  - `kimi` / `kimi-thinking` / `kimi-vision`
  - `sonar`

### `parser`

PDF 解析相关配置。

- `ocr`：OCR 开关
- `detect_columns`：多栏检测
- `caption_pattern` / `table_caption_pattern`：图表标题正则
- `llm_text_provider` / `llm_vision_provider`：解析增强使用的 LLM provider
- `enrich_tables` / `enrich_figures`：表格/图表 LLM 增强开关

### `chunk`

切块粒度配置。

- `target_chars`：目标字符数（默认 1000）
- `min_chars`：最小字符数（默认 200）
- `max_chars`：最大字符数（默认 1800）
- `overlap_sentences`：句级重叠（默认 2）

### `search`

本地检索配置。

- `top_k`：召回数量
- `rerank_top_k`：重排后保留数量
- Dense/Sparse 召回参数
- `reranker_mode`：重排模式（bge / colbert / cascade）
- ColBERT 开关与模型配置

### `web_search`

Tavily 网络搜索配置。

- `enabled`：开关
- `api_key`：Tavily API Key
- `max_results`：最大结果数
- `query_optimizer`：查询优化选项

### `google_search`

Google / Scholar 搜索配置。

- `enabled` / `google_enabled` / `scholar_enabled`：开关
- `browser_headless`：无头模式
- `max_results`：最大结果数

### `semantic_scholar`

Semantic Scholar API 配置。

- `enabled`：开关
- `api_key`：API Key

### `ncbi`

NCBI 文献搜索配置。

- `enabled`：开关
- `email`：NCBI API 邮箱
- `api_key`：API Key

### `content_fetcher`

网页全文抓取配置。

- `enabled`：开关
- 缓存与策略配置

### `api`

API 服务配置。

- `host`：监听地址（默认 `127.0.0.1`）
- `port`：端口（默认 `9999`）

### `auth`

认证配置。

- `secret_key`：JWT 密钥（生产环境必须替换）
- `token_expire_hours`：Token 过期时间（默认 24h）
- 初始化管理员账户字段（`default_admin_*`）

### `logging`

日志配置。

- `level`：日志级别（默认 INFO）
- `max_size`：单文件大小上限
- `retention_days`：保留天数

### `storage`

存储清理配置。

- `max_age_days`：最大保留天数
- `max_size_gb`：最大存储大小
- `cleanup_on_startup`：启动时是否自动清理
- `cleanup_batch_size`：清理批次大小

### `citation`

引用格式配置。

- 引用样式与格式化选项

### `deep_research`

Deep Research 配置。

- `depth_presets`：`lite` / `comprehensive` 的完整阈值集合
  - 迭代预算：`max_iterations_per_section`、`max_section_research_rounds`
  - 覆盖阈值：`coverage_threshold`
  - 查询预算：`recall_queries_per_section`、`precision_queries_per_section`
  - 分层召回：`search_top_k_first` / `search_top_k_gap` / `search_top_k_write`
  - 写作二次取证：`verification_k`
  - 自校正补检索：`self_correction_trigger_coverage`、`self_correction_min_round`、`search_top_k_gap_decay_factor`、`search_top_k_gap_min`
  - 收益曲线早停：`coverage_plateau_floor`、`coverage_plateau_min_gain`
  - 验证分层：`verify_light_threshold` / `verify_medium_threshold` / `verify_severe_threshold`
  - 审核门：`review_gate_max_rounds`、`review_gate_base_sleep`、`review_gate_max_sleep`、`review_gate_early_stop_unchanged`
  - 图上限：`recursion_limit`
  - 成本监控：`cost_warn_steps`、`cost_force_summary_steps`、`cost_tick_interval`

### `auto_complete`

一键综述配置。

- 自动补全相关参数

### `performance`

性能配置（统一控制超时、重试、并发和缓存）。

- `retrieval`：检索超时与并发
- `llm`：LLM 调用超时与重试
- `web_search`：Tavily 超时
- `unified_web_search`：统一网络搜索并发
  - `browser_providers_max_parallel`：浏览器类搜索并发数
- `google_search`：Google 搜索超时

## 三、重要环境变量

### API Key 注入（推荐）

- `RAG_LLM__{PROVIDER}__API_KEY`：覆盖对应 provider key
  - 示例：`RAG_LLM__OPENAI__API_KEY`、`RAG_LLM__DEEPSEEK__API_KEY`
- `RAG_LLM__SONAR__API_KEY`：Perplexity Sonar key
- 兼容旧变量：`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`

### 服务配置

- `API_HOST`、`API_PORT`：API 监听地址
- `MILVUS_HOST`、`MILVUS_PORT`：Milvus 地址
- `RAG_ENV`：运行环境（dev / prod）
- `COMPUTE_DEVICE`：计算设备（mps / cuda / cpu）

### 模型与缓存

- `HF_LOCAL_FILES_ONLY`：离线加载 HuggingFace 模型
- `MODEL_CACHE_ROOT`：模型缓存根目录
- `EMBEDDING_CACHE_DIR`：Embedding 模型缓存
- `RERANKER_CACHE_DIR`：Reranker 模型缓存
- `COLBERT_CACHE_DIR`：ColBERT 模型缓存
- `EMBEDDING_MODEL`：Embedding 模型名（默认 `BAAI/bge-m3`）
- `RERANKER_MODEL`：Reranker 模型名（默认 `BAAI/bge-reranker-v2-m3`）
- `INDEX_TYPE`：索引类型（`IVF_FLAT` / `GPU_IVF_FLAT`）

### 可观测性

- OpenTelemetry 环境变量（`OTEL_*`）
- LangSmith 环境变量（可选）

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
