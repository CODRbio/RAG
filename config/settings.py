"""
统一配置模块
- 配置文件: config/rag_config.json（LLM 等可调参数）
- 本地覆盖: config/rag_config.local.json（本地私密配置）
- 环境变量优先覆盖敏感项（API Key 等）
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# 加载 config/rag_config.json + config/rag_config.local.json（本地覆盖）
_CONFIG_PATH = Path(__file__).parent / "rag_config.json"
_LOCAL_CONFIG_PATH = Path(__file__).parent / "rag_config.local.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_RAW_CONFIG: Dict[str, Any] = _load_json(_CONFIG_PATH)
if _LOCAL_CONFIG_PATH.exists():
    _RAW_CONFIG = _deep_merge(_RAW_CONFIG, _load_json(_LOCAL_CONFIG_PATH))


@dataclass
class MilvusSettings:
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: int = int(os.getenv("MILVUS_PORT", "19530"))

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelSettings:
    device: str = os.getenv("COMPUTE_DEVICE", "mps")
    use_fp16: bool = os.getenv("USE_FP16", "false").lower() == "true"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    _default_cache_root: str = os.path.expanduser(os.getenv("MODEL_CACHE_ROOT", "~/Hug"))
    embedding_cache_dir: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR") or _default_cache_root
    reranker_cache_dir: Optional[str] = os.getenv("RERANKER_CACHE_DIR") or _default_cache_root
    colbert_cache_dir: Optional[str] = os.getenv("COLBERT_CACHE_DIR") or _default_cache_root
    local_files_only: bool = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"


@dataclass
class CollectionSettings:
    """Collection 命名 - 全局统一，迁移时不变"""
    life: str = os.getenv("COLLECTION_LIFE", "deepsea_life")
    ocean: str = os.getenv("COLLECTION_OCEAN", "deepsea_ocean")
    env: str = os.getenv("COLLECTION_ENV", "deepsea_env")
    global_: str = os.getenv("COLLECTION_GLOBAL", "deepsea_global")

    def get(self, domain: str) -> str:
        mapping = {"life": self.life, "ocean": self.ocean, "env": self.env}
        return mapping.get(domain, self.global_)

    def all(self) -> List[str]:
        return [self.life, self.ocean, self.env, self.global_]


@dataclass
class IndexSettings:
    index_type: str = os.getenv("INDEX_TYPE", "IVF_FLAT")
    nlist: int = int(os.getenv("INDEX_NLIST", "256"))

    @property
    def params(self) -> dict:
        return {
            "index_type": self.index_type,
            "metric_type": "COSINE",
            "params": {"nlist": self.nlist}
        }


@dataclass
class ChunkSettings:
    target_chars: int = int(os.getenv("CHUNK_TARGET_CHARS", "1000"))
    min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "200"))
    max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
    overlap_sentences: int = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))
    table_rows_per_chunk: int = int(os.getenv("CHUNK_TABLE_ROWS_PER_CHUNK", "10"))


@dataclass
class SearchSettings:
    top_k: int = int(os.getenv("SEARCH_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    dense_recall_k: int = int(os.getenv("DENSE_RECALL_K", "80"))
    sparse_recall_k: int = int(os.getenv("SPARSE_RECALL_K", "80"))
    rrf_dense_weight: float = float(os.getenv("RRF_DENSE_WEIGHT", "0.6"))
    rrf_sparse_weight: float = float(os.getenv("RRF_SPARSE_WEIGHT", "0.4"))
    rerank_input_k: int = int(os.getenv("RERANK_INPUT_K", "100"))
    rerank_output_k: int = int(os.getenv("RERANK_OUTPUT_K", "20"))
    per_doc_cap: int = int(os.getenv("PER_DOC_CAP", "5"))
    # ColBERT 精排：bge_only | colbert_only | cascade
    reranker_mode: str = os.getenv("RERANKER_MODE", "cascade")
    use_colbert_reranker: bool = os.getenv("USE_COLBERT_RERANKER", "true").lower() == "true"
    colbert_model: str = os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
    colbert_top_k: int = int(os.getenv("COLBERT_TOP_K", "30"))  # cascade 时 BGE 粗排输出条数，再送 ColBERT 精排


@dataclass
class WebSearchConfig:
    """Tavily 网络搜索配置"""
    enabled: bool = True
    provider: str = "tavily"
    api_key: str = ""
    search_depth: str = "advanced"
    max_results: int = 5
    include_answer: bool = True
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    enable_query_optimizer: bool = True
    enable_query_expansion: bool = False
    query_expansion_llm: str = "deepseek"
    max_queries: int = 4


@dataclass
class GoogleSearchConfig:
    """Google Scholar / Google 搜索配置"""
    enabled: bool = True
    scholar_enabled: bool = True
    google_enabled: bool = False
    extension_path: str = "extra_tools/CapSolverExtension"
    headless: Optional[bool] = None
    proxy: Optional[str] = None
    timeout: int = 60000
    max_results: int = 5
    user_data_dir: Optional[str] = None


@dataclass
class SemanticScholarConfig:
    """Semantic Scholar API 配置（通过 ai4scholar 代理）"""
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://ai4scholar.net/graph/v1"
    max_results: int = 5
    timeout_seconds: int = 30


@dataclass
class ContentFetcherConfig:
    """WebContentFetcher 全文抓取配置"""
    enabled: bool = False
    only_academic: bool = False
    max_content_length: int = 8000
    timeout_seconds: int = 15
    brightdata_api_key: str = ""
    brightdata_zone: str = ""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent: int = 5


@dataclass
class ApiSettings:
    """API 服务配置"""
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = int(os.getenv("API_PORT", "9999"))


@dataclass
class CitationSettings:
    """引用格式配置"""
    key_format: str = "author_date"  # numeric | hash | author_date
    hash_length: int = 12
    author_date_max_authors: int = 2
    merge_level: str = "document"  # chunk | document（按 doc_id/URL 合并为文章级）


# LLM 环境变量映射（兼容旧变量名）
_LLM_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "kimi": "KIMI_API_KEY",
}

# 各 provider 默认 base_url / model（config 未填时回退）
_LLM_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta", "default_model": "gemini-1.5-pro"},
    "claude": {"base_url": "https://api.anthropic.com", "default_model": "claude-sonnet-4-20250514"},
}


def _llm_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("llm") or {})


def _llm_provider_raw(name: str) -> Dict[str, Any]:
    return (_llm_from_config().get("providers") or {}).get(name) or {}


class LLMSettings:
    """
    LLM 配置：config/rag_config.json 中 llm.providers 支持 openai / deepseek / gemini / claude。
    环境变量覆盖 api_key：OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY。
    脚本可通过参数 --llm / --model 指定本次使用的 provider 和模型。
    """

    def __init__(self):
        cfg = _llm_from_config()
        self.default: str = os.getenv("DEFAULT_LLM") or cfg.get("default") or "claude"
        self.dry_run: bool = (
            os.getenv("LLM_DRY_RUN", "").lower() == "true" or cfg.get("dry_run") is True
        )

    def get_provider(self, name: str) -> Dict[str, Any]:
        """
        按 provider 名（openai / deepseek / gemini / claude / kimi 等）返回配置：
        - api_key/base_url
        - default_model + models（支持一个 provider 多模型）
        - params（provider 额外参数）
        环境变量优先覆盖 api_key。支持变体名匹配环境变量（如 kimi-thinking 找 KIMI_API_KEY）。
        """
        raw = _llm_provider_raw(name)
        defaults = _LLM_DEFAULTS.get(name) or {}

        # 环境变量查找逻辑：
        # 1) RAG_LLM__{PROVIDER}__API_KEY（优先）
        # 2) 兼容旧变量名（如 OPENAI_API_KEY）
        normalized = name.upper().replace("-", "_")
        env_key = f"RAG_LLM__{normalized}__API_KEY"
        api_key = os.getenv(env_key)
        if not api_key:
            legacy_key = _LLM_ENV_KEYS.get(name)
            if not legacy_key and "-" in name:
                base_name = name.split("-")[0]
                legacy_key = _LLM_ENV_KEYS.get(base_name)
            api_key = (os.getenv(legacy_key) if legacy_key else None)
        api_key = api_key or raw.get("api_key") or ""
        models = raw.get("models") or {}
        if isinstance(models, list):
            models = {m: m for m in models}
        default_model = (
            raw.get("default_model")
            or raw.get("model")  # 兼容旧字段
            or defaults.get("default_model")
            or defaults.get("model")
            or ""
        )
        return {
            "api_key": api_key,
            "base_url": raw.get("base_url") or defaults.get("base_url") or "",
            "default_model": default_model,
            "models": models,
            "params": raw.get("params") or {},
        }

    def resolve_model(self, provider: str, model_override: str | None = None) -> str:
        """
        解析本次实际要调用的模型名：
        - 若传入 model_override：可为 alias（在 models 中）或直接 model 名
        - 若未传：使用 default_model
        """
        cfg = self.get_provider(provider)
        models: Dict[str, str] = cfg.get("models") or {}
        default_model: str = (cfg.get("default_model") or "").strip()

        if model_override:
            m = model_override.strip()
            if not m:
                return default_model
            if m in models:
                return str(models[m]).strip()
            # 允许直接给 model 名（无需事先写入 models）
            return m

        # 未指定 model：走 default_model；若 default_model 是 alias，则映射
        if default_model in models:
            return str(models[default_model]).strip()
        return default_model

    def is_available(self, name: str) -> bool:
        """检查某 provider 是否已配置 api_key（不做占位前缀过滤）"""
        p = self.get_provider(name)
        key = (p.get("api_key") or "").strip()
        return bool(key)


@dataclass
class PathSettings:
    base: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data(self) -> Path:
        return self.base / "data"

    @property
    def raw_papers(self) -> Path:
        return self.data / "raw_papers"

    @property
    def parsed(self) -> Path:
        return self.data / "parsed"

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def artifacts(self) -> Path:
        return self.base / "artifacts"

    def ensure_dirs(self):
        for p in [self.data, self.raw_papers, self.parsed, self.logs, self.artifacts]:
            p.mkdir(parents=True, exist_ok=True)


def _chunk_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("chunk") or {})


def _search_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("search") or {})


def _web_search_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("web_search") or {})


def _google_search_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("google_search") or {})


def _api_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("api") or {})


def _citation_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("citation") or {})


def _semantic_scholar_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("semantic_scholar") or {})


def _content_fetcher_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("content_fetcher") or {})


def _performance_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("performance") or {})


def _storage_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("storage") or {})


def _auth_from_config() -> Dict[str, Any]:
    return (_RAW_CONFIG.get("auth") or {})


@dataclass
class RetrievalPerfSettings:
    """检索层：超时、缓存、并行"""
    timeout_seconds: int = 60
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    parallel_dense_sparse: bool = True
    max_workers: int = 4


@dataclass
class LLMPerfSettings:
    """LLM：超时、重试、并发限流、缓存"""
    timeout_seconds: int = 120
    max_retries: int = 2
    retry_backoff: float = 1.5
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    max_concurrent_per_provider: int = 5


@dataclass
class WebSearchPerfSettings:
    """Tavily 等：超时、缓存"""
    timeout_seconds: int = 30
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600


@dataclass
class UnifiedWebSearchPerfSettings:
    """统一网络搜索：并发、超时。Scholar/Google 因风控单独限流，默认 1，建议最大 2。"""
    max_parallel_providers: int = 3
    per_provider_timeout_seconds: int = 30
    browser_providers_max_parallel: int = 1  # Scholar+Google 同时最多几个，建议不超过 2


@dataclass
class GoogleSearchPerfSettings:
    """Google/Scholar：浏览器复用、缓存"""
    browser_reuse: bool = True
    max_idle_seconds: int = 300
    max_pages_per_browser: int = 10
    cache_enabled: bool = False
    cache_ttl_seconds: int = 1800


@dataclass
class StorageSettings:
    """持久化存储生命周期与容量限制"""
    max_age_days: int = 30          # 数据保留天数，默认 30 天
    max_size_gb: float = 5.0        # 总大小上限（GB），默认 5GB
    cleanup_on_startup: bool = True # 启动时是否自动清理
    cleanup_batch_size: int = 100   # 每批清理记录数


@dataclass
class AuthSettings:
    """认证配置：token 有效期、首次管理员账号（敏感项放 .local.json）"""
    secret_key: str = "change-me-in-local"
    token_expire_hours: float = 24.0
    admin_username: str = "admin"
    admin_default_password: str = "admin123"


class Settings:
    def __init__(self):
        self.env = os.getenv("RAG_ENV", "dev")
        self.milvus = MilvusSettings()
        self.model = ModelSettings()
        self.collection = CollectionSettings()
        self.index = IndexSettings()
        c = _chunk_from_config()
        self.chunk = ChunkSettings(
            target_chars=c.get("target_chars", 1000),
            min_chars=c.get("min_chars", 200),
            max_chars=c.get("max_chars", 1800),
            overlap_sentences=c.get("overlap_sentences", 2),
            table_rows_per_chunk=c.get("table_rows_per_chunk", 10),
        )
        s = _search_from_config()
        self.search = SearchSettings(
            top_k=s.get("top_k", int(os.getenv("SEARCH_TOP_K", "20"))),
            rerank_top_k=s.get("rerank_top_k", int(os.getenv("RERANK_TOP_K", "20"))),
            rrf_k=s.get("rrf_k", int(os.getenv("RRF_K", "60"))),
            dense_recall_k=s.get("dense_recall_k", 80),
            sparse_recall_k=s.get("sparse_recall_k", 80),
            rrf_dense_weight=s.get("rrf_dense_weight", 0.6),
            rrf_sparse_weight=s.get("rrf_sparse_weight", 0.4),
            rerank_input_k=s.get("rerank_input_k", 100),
            rerank_output_k=s.get("rerank_output_k", 20),
            per_doc_cap=s.get("per_doc_cap", 5),
            reranker_mode=s.get("reranker_mode", os.getenv("RERANKER_MODE", "bge_only")),
            use_colbert_reranker=s.get("use_colbert_reranker", os.getenv("USE_COLBERT_RERANKER", "false").lower() == "true"),
            colbert_model=s.get("colbert_model", os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")),
            colbert_top_k=s.get("colbert_top_k", int(os.getenv("COLBERT_TOP_K", "30"))),
        )
        w = _web_search_from_config()
        include = w.get("include_domains") or []
        exclude = w.get("exclude_domains") or []
        if isinstance(include, str):
            include = [x.strip() for x in include.split(",") if x.strip()]
        if isinstance(exclude, str):
            exclude = [x.strip() for x in exclude.split(",") if x.strip()]
        self.web_search = WebSearchConfig(
            enabled=w.get("enabled", True),
            provider=(w.get("provider") or "tavily").strip(),
            api_key=(w.get("api_key") or "").strip(),
            search_depth=(w.get("search_depth") or "advanced").strip(),
            max_results=min(int(w.get("max_results", 5)), 10),
            include_answer=w.get("include_answer", True),
            include_domains=include,
            exclude_domains=exclude,
            enable_query_optimizer=bool(w.get("enable_query_optimizer", True)),
            enable_query_expansion=w.get("enable_query_expansion", False),
            query_expansion_llm=(w.get("query_expansion_llm") or "deepseek").strip(),
            max_queries=min(int(w.get("max_queries", 4)), 8),
        )
        g = _google_search_from_config()
        self.google_search = GoogleSearchConfig(
            enabled=g.get("enabled", True),
            scholar_enabled=g.get("scholar_enabled", True),
            google_enabled=g.get("google_enabled", False),
            extension_path=(g.get("extension_path") or "extra_tools/CapSolverExtension").strip(),
            headless=g.get("headless"),
            proxy=g.get("proxy"),
            timeout=int(g.get("timeout", 60000)),
            max_results=min(int(g.get("max_results", 5)), 20),
            user_data_dir=g.get("user_data_dir"),
        )
        ss = _semantic_scholar_from_config()
        self.semantic_scholar = SemanticScholarConfig(
            enabled=bool(ss.get("enabled", False)),
            api_key=(ss.get("api_key") or "").strip(),
            base_url=(ss.get("base_url") or "https://ai4scholar.net/graph/v1").strip(),
            max_results=min(int(ss.get("max_results", 5)), 20),
            timeout_seconds=int(ss.get("timeout_seconds", 30)),
        )
        cf = _content_fetcher_from_config()
        self.content_fetcher = ContentFetcherConfig(
            enabled=bool(cf.get("enabled", False)),
            only_academic=bool(cf.get("only_academic", False)),
            max_content_length=int(cf.get("max_content_length", 8000)),
            timeout_seconds=int(cf.get("timeout_seconds", 15)),
            brightdata_api_key=(cf.get("brightdata_api_key") or "").strip(),
            brightdata_zone=(cf.get("brightdata_zone") or "").strip(),
            cache_enabled=bool(cf.get("cache_enabled", True)),
            cache_ttl_seconds=int(cf.get("cache_ttl_seconds", 3600)),
            max_concurrent=int(cf.get("max_concurrent", 5)),
        )
        a = _api_from_config()
        self.api = ApiSettings(
            host=str(a.get("host", os.getenv("API_HOST", "127.0.0.1"))),
            port=int(a.get("port", os.getenv("API_PORT", "9999"))),
        )
        ct = _citation_from_config()
        self.citation = CitationSettings(
            key_format=str(ct.get("key_format", os.getenv("CITATION_KEY_FORMAT", "author_date"))),
            hash_length=int(ct.get("hash_length", 12)),
            author_date_max_authors=int(ct.get("author_date_max_authors", 2)),
            merge_level=str(ct.get("merge_level", "document")),
        )
        self.llm = LLMSettings()
        self.path = PathSettings()
        pf = _performance_from_config()
        rp = pf.get("retrieval") or {}
        lp = pf.get("llm") or {}
        wp = pf.get("web_search") or {}
        up = pf.get("unified_web_search") or {}
        gp = pf.get("google_search") or {}
        self.perf_retrieval = RetrievalPerfSettings(
            timeout_seconds=int(rp.get("timeout_seconds", 60)),
            cache_enabled=bool(rp.get("cache_enabled", False)),
            cache_ttl_seconds=int(rp.get("cache_ttl_seconds", 3600)),
            parallel_dense_sparse=bool(rp.get("parallel_dense_sparse", True)),
            max_workers=int(rp.get("max_workers", 4)),
        )
        self.perf_llm = LLMPerfSettings(
            timeout_seconds=int(lp.get("timeout_seconds", 120)),
            max_retries=int(lp.get("max_retries", 2)),
            retry_backoff=float(lp.get("retry_backoff", 1.5)),
            cache_enabled=bool(lp.get("cache_enabled", False)),
            cache_ttl_seconds=int(lp.get("cache_ttl_seconds", 3600)),
            max_concurrent_per_provider=int(lp.get("max_concurrent_per_provider", 5)),
        )
        self.perf_web_search = WebSearchPerfSettings(
            timeout_seconds=int(wp.get("timeout_seconds", 30)),
            cache_enabled=bool(wp.get("cache_enabled", False)),
            cache_ttl_seconds=int(wp.get("cache_ttl_seconds", 3600)),
        )
        browser_max = min(2, int(up.get("browser_providers_max_parallel", 1)))  # 风控建议不超过 2
        self.perf_unified_web = UnifiedWebSearchPerfSettings(
            max_parallel_providers=int(up.get("max_parallel_providers", 3)),
            per_provider_timeout_seconds=int(up.get("per_provider_timeout_seconds", 30)),
            browser_providers_max_parallel=max(1, browser_max),
        )
        self.perf_google_search = GoogleSearchPerfSettings(
            browser_reuse=bool(gp.get("browser_reuse", True)),
            max_idle_seconds=int(gp.get("max_idle_seconds", 300)),
            max_pages_per_browser=int(gp.get("max_pages_per_browser", 10)),
            cache_enabled=bool(gp.get("cache_enabled", False)),
            cache_ttl_seconds=int(gp.get("cache_ttl_seconds", 1800)),
        )
        st = _storage_from_config()
        self.storage = StorageSettings(
            max_age_days=int(st.get("max_age_days", 30)),
            max_size_gb=float(st.get("max_size_gb", 5.0)),
            cleanup_on_startup=bool(st.get("cleanup_on_startup", True)),
            cleanup_batch_size=int(st.get("cleanup_batch_size", 100)),
        )
        au = _auth_from_config()
        self.auth = AuthSettings(
            secret_key=str(au.get("secret_key", "change-me-in-local")),
            token_expire_hours=float(au.get("token_expire_hours", 24)),
            admin_username=str(au.get("admin_username", "admin")),
            admin_default_password=str(au.get("admin_default_password", "admin123")),
        )

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    def print_info(self):
        print(f"""
========================================
  深海科研知识库 RAG 系统
========================================
  环境: {self.env}
  Milvus: {self.milvus.uri}
  设备: {self.model.device}
  FP16: {self.model.use_fp16}
  索引: {self.index.index_type}
  Collections: {', '.join(self.collection.all())}
========================================
        """)


# 全局单例
settings = Settings()
