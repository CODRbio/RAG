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

**运行时取值优先级（生效顺序）：**

- **UI/请求入参（若有）**：前端或 API 请求里显式传入的参数（如检索超时、全文抓取模式等）优先生效。
- **Config 托底**：上述未提供时，使用本文件（及本地覆盖、环境变量）中的配置。
- **代码默认值**：配置中未书写该项时，使用各模块代码中的默认值。

即：**UI 输入（若有）> config > 代码默认**。新增或扩展配置项时，应保持该优先级，并在使用处合并请求级覆盖后再读 config。

## 二、关键配置块（`rag_config.json`）

### `database`

统一数据库配置。正式后端统一使用 PostgreSQL。

- `url`：数据库连接字符串（推荐 `postgresql+psycopg://rag:change-me@localhost:5433/rag`）
- `echo`：是否输出 SQL 调试日志

说明：

- 当前后端使用 **SQLModel + Alembic** 管理 schema 与迁移
- 生产与开发环境均以 PostgreSQL 为正式后端
- SQLite 仅保留给历史版本迁移/兼容脚本，不再作为当前默认部署方案

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
- `rerank_funnel_floor_k`：重排 funnel 的最小地板（默认 20）
- `reranker_mode`：重排模式（bge / colbert / cascade）
- ColBERT 开关与模型配置
  - `cascade_bge_multiplier`：cascade 中 BGE 中间层倍率（默认 1.5）
- Gap 融合配置（Chat / Research 可独立）
  - `chat_gap_ratio`：Chat gap 最低配额比例（默认 `0.2`）
  - `research_gap_ratio`：Research gap 最低配额比例（默认 `0.25`）
  - `chat_rank_pool_multiplier`：Chat 融合重排池放大倍数（默认 `3.0`）
  - `research_rank_pool_multiplier`：Research 融合重排池放大倍数（默认 `3.0`）

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

### `shared_browser`

共享浏览器与 Playwright context 池配置。启动时根据此配置初始化常驻 CDP 浏览器和 context 池，供 Google/Scholar 搜索、网页全文抓取（content_fetcher）等模块复用。

#### 浏览器进程

- `start_headless`：是否启动无头 CDP 浏览器（默认 true）
- `start_headed`：是否启动有头 CDP 浏览器（默认 true）
- `headless_port`：无头浏览器 CDP 端口（默认 9222）
- `headed_port`：有头浏览器 CDP 端口（默认 9223）

#### Context 池大小

- `headless_context_pool_size`：无头 context 总数（默认 4）
  - 其中一部分为通用 slot，所有模块均可使用
  - 另一部分为搜索预留 slot，仅 Google/Scholar 搜索可优先使用
- `headless_search_reserved_slots`：搜索预留 slot 数量（默认 1）
  - 通用 slot 数 = `headless_context_pool_size - headless_search_reserved_slots`
  - 设为 0 则关闭预留，所有 slot 均为通用
- `headed_context_pool_size`：有头 context 数量（默认 2，无预留）

#### 超时与冷却

- `context_acquire_timeout_seconds`：获取 context 的最大等待时间（默认 30s）
  - 超时后调用方回退到临时浏览器
- `context_cooldown_min_seconds` / `context_cooldown_max_seconds`：归还 context 后的随机冷却区间（默认 1~2s），防止同一 context 被过于频繁复用
- `context_idle_ttl_seconds`：slot 元数据空闲 TTL（默认 300s，仅用于诊断标记，不触发驱逐——context 始终常驻）

#### 调优建议

- 如果并发用户较多（同时多人检索），可适当增大 `headless_context_pool_size`（如 6），并酌情增加 `headless_search_reserved_slots`（如 2）
- 如果搜索功能很少使用，可将 `headless_search_reserved_slots` 设为 0 以释放该 slot 给其他模块
- 有头 context 主要用于文献下载和需要 UI 交互的验证码场景，一般 2 个即可满足需求

### `tool_execution`

Agent 工具执行安全配置。**`run_code` 默认关闭，必须显式启用。**

| 字段 | 默认值 | 说明 |
|---|---|---|
| `run_code_enabled` | `false` | 是否启用 `run_code` 工具 |
| `timeout_seconds` | `5` | 单次代码执行墙钟超时（秒） |
| `cpu_seconds` | `2` | 子进程 CPU 时间硬上限（秒，POSIX 仅限） |
| `max_memory_mb` | `128` | 子进程内存上限（MB，Linux 有效；macOS 不受 RLIMIT 限制，见下方说明） |
| `max_code_chars` | `6000` | 代码字符长度上限 |
| `max_output_chars` | `12000` | stdout/stderr 合并输出上限 |
| `max_concurrent` | `2` | 同时运行的 `run_code` 子进程上限，超限立即返回错误 |
| `allowed_modules` | 见下方 | 允许导入的标准库白名单（字符串列表） |

默认 `allowed_modules`（均为纯计算模块，无文件/网络访问能力）：
`collections`, `datetime`, `decimal`, `fractions`, `functools`, `itertools`,
`json`, `math`, `random`, `re`, `statistics`, `time`

**安全说明：**

- `run_code` 默认关闭；仅建议在受信任的内部环境显式开启
- 开启后执行路径：AST 预检（禁止危险调用与导入）→ 受限 builtins 白名单 → 独立子进程（`-I -S -B`，干净环境）→ POSIX 资源限制（CPU / fd / 进程数）→ 超时 + 输出双熔断 → 进程组强制终止
- **macOS 限制**：`RLIMIT_AS` / `RLIMIT_DATA` 在 macOS 上不被内核强制执行，`max_memory_mb` 配置在 macOS 下无效；启动时会打印 WARNING 日志。生产环境建议部署在 Linux 上
- `max_concurrent` 防止并发聚合攻击，超出槽位的请求立即返回提示，不排队等待
- 若面向外部用户开放，仍建议迁移到容器/沙箱执行后端，不要直接依赖宿主机解释器隔离

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

---

## 仅通过 config 生效的参数（未在 UI 暴露）

以下参数**不在前端 UI 中展示**，只能通过 `config/rag_config.json`（或 `rag_config.local.json`）修改；修改后需重启服务生效。默认值以当前代码为准，下表供查阅与调优。

### `performance.retrieval`（检索层）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timeout_seconds` | `60` | Hybrid 时 **local** 分支硬超时（秒），避免向量/图 DB 卡死拖住整次请求。 |
| `web_soft_wait_seconds` | `500` | Hybrid 时对 **web** 分支（provider 搜索 + content_fetcher 拉正文）的最大等待秒数；超时后使用已完成的 Phase1 片段或部分富文本，不丢弃。 |
| `cache_enabled` | `false` | 检索层内存缓存开关。 |
| `cache_ttl_seconds` | `3600` | 检索层缓存 TTL（秒）。 |
| `parallel_dense_sparse` | `true` | 是否并行执行 Dense / Sparse 召回。 |
| `max_workers` | `4` | 检索并行 worker 数。 |

### `content_fetcher`（URL 全文抓取）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_enabled` | `true` | 内存 L1 缓存开关。 |
| `cache_ttl_seconds` | `3600` | L1 缓存 TTL（秒）。 |
| `disk_cache_enabled` | `true` | 磁盘 L2 缓存（SQLite）开关；跨进程/重启复用。 |
| `disk_cache_ttl_seconds` | `2592000` | L2 普通条目 TTL（秒），默认 30 天。 |
| `disk_cache_promote_threshold` | `3` | 同一 URL 在 TTL 内命中次数达到该值后晋升为永久缓存，不再过期。 |
| `disk_cache_dir` | `"data/cache"` | L2 数据库所在目录，库文件为 `web_content.db`。 |
| `max_concurrent` | `5` | 并发抓取 URL 数。 |
| `timeout_seconds` | `15` | 单 URL 抓取超时（秒）。 |
| `compress_long_fulltext` / `compress_word_threshold` / `compress_max_output_words` | 见 `rag_config.json` | 长文压缩与截断。 |

### 其他 performance 子块

- `performance.llm`：`timeout_seconds`、`max_retries`、`retry_backoff`、`cache_*`、`max_concurrent_per_provider`
- `performance.web_search`：`timeout_seconds`、`cache_*`
- `performance.unified_web_search`：`max_parallel_providers`、`per_provider_timeout_seconds`、`browser_providers_max_parallel`
- `performance.google_search`：`browser_reuse`、`max_idle_seconds`、`max_pages_per_browser`、`cache_*`

以上默认值以 `config/settings.py` 及 `config/rag_config.json` 为准；若需覆盖，在 `performance` / `content_fetcher` 对应块中增加或修改键即可。

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
    "url": "postgresql+psycopg://rag:change-me@localhost:5433/rag",
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
