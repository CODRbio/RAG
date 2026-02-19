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

### `database`

统一数据库配置（当前默认 SQLite，后续可平滑切换 PostgreSQL）。

- `url`：数据库连接字符串（默认 `sqlite:///data/rag.db`）
- `echo`：是否输出 SQL 调试日志

说明：

- 当前后端使用 **SQLModel + Alembic** 管理 schema 与迁移
- 启动时会确保 `data/rag.db` 可用，并尝试自动迁移历史多库数据（若存在）

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

- `secret_key`：JWT 签名密钥（生产环境必须替换；默认值仅用于开发环境）
- `token_expire_hours`：JWT 过期时间（默认 24h）
- 初始化管理员账户字段（`default_admin_*`）

说明：

- 登录后签发的是无状态 JWT（`HS256`），服务端不再依赖进程内 token 字典，支持多 worker / 多副本部署。
- 被显式撤销的 token 会写入数据库表 `revoked_tokens`（保存 token 的 SHA-256 哈希），用于后续校验拦截。
- 服务启动时会自动清理过期的撤销记录，避免 `revoked_tokens` 持续膨胀。

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

**全篇连贯性重写（Coherence Refine）策略**

`synthesize_node` 在最终整合阶段自动选择写作策略，无需额外配置：

| 情况 | 策略 | 说明 |
|---|---|---|
| 文档 token 充足 | Single-pass | 单次调用，`max_tokens` 由 `compute_safe_budget` 动态计算 |
| 文档过长（输出预算 < 1024 token）| Sliding Window | 逐章润色，每章携带 Document Blueprint 全局引导 |

- Token 估算工具：`src/utils/token_counter.py`（tiktoken cl100k_base，不可用时自动降级为字符估算）
- 模型上下文窗口注册在 `MODEL_CONTEXT_WINDOWS`（`token_counter.py`），未知模型默认 64,000 token
- 安全余量：默认保留 10%（`safety_margin=0.10`），可在源码中调整
- Prompt 模板：单次 → `coherence_refine.txt`；分章 → `coherence_refine_window.txt`

### `graph`

知识图谱实体抽取配置，控制 HippoRAG 构图时所使用的 NER 策略。

```json
"graph": {
  "entity_extraction": {
    "strategy": "gliner",
    "fallback": "rule",
    "ontology_path": "config/ontology.json",
    "gliner": {
      "model": "urchade/gliner_base",
      "threshold": 0.4,
      "device": "cpu"
    },
    "llm": {
      "provider": "deepseek",
      "max_tokens": 1000
    }
  }
}
```

- `strategy`：主策略，可选值：
  - `gliner`（默认）：本地轻量 Zero-Shot NER，CPU 友好，~50 MB 模型，无需手写规则
  - `rule`：基于 `config/ontology.json` 中的正则规则，无需外部模型
  - `llm`：调用项目 LLM Provider 按文本抽取，精度最高但速度最慢
- `fallback`：主策略**抛出异常**时自动降级到的备用策略（不因空结果触发，避免噪声）
- `ontology_path`：领域本体配置路径，相对于项目根目录；切换领域只需替换此文件
- `gliner.model`：GLiNER 模型名（HuggingFace Hub ID 或本地路径）
- `gliner.threshold`：实体置信度阈值，越低召回越多，建议 0.3–0.5
- `gliner.device`：推理设备，`cpu` / `cuda` / `mps`
- `llm.provider`：使用 LLM 策略时的 provider（需在 `llm.providers` 中已配置）
- `llm.max_tokens`：LLM 单次抽取最大 token 数

#### 领域本体 `config/ontology.json`

定义实体类型与（可选）规则模式。切换领域只需编辑此文件，无需改动代码：

```json
{
  "entity_types": {
    "ORGANISM": {
      "label": "organism, species, or biological taxon",
      "description": "Species, organisms, biological taxa including Latin binomial names",
      "patterns": ["\\b[A-Z][a-z]{3,} [a-z]{3,}\\b"]
    },
    "SUBSTANCE": { ... },
    ...
  },
  "min_entity_length": 2,
  "_profiles": {
    "deep_sea": { ... },
    "biomedical": { ... }
  }
}
```

- `entity_types`：实体类型字典，key 为类型名称（全大写）
  - `label`：GLiNER 使用的自然语言描述（影响 Zero-Shot 性能，应具体）
  - `description`：LLM prompt 中展示的说明
  - `patterns`：`rule` 策略使用的正则列表（`gliner`/`llm` 策略会忽略此字段）
- `min_entity_length`：最短实体字符数，过短的匹配将被过滤
- `_profiles`：参考用领域 Profile，不会自动激活；若需启用，将对应 patterns 复制到 `entity_types` 对应类型的 `patterns` 字段中

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
- `RAG_DATABASE_URL`：覆盖数据库连接串（高于配置文件，适合 CI/生产）
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
  "database": {
    "url": "sqlite:///data/rag.db",
    "echo": false
  },
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
