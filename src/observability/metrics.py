"""
Prometheus metrics 定义。

所有自定义指标集中定义，业务模块通过 `from src.observability import metrics` 引用。
"""

from prometheus_client import Counter, Histogram, Gauge, Info


class _Metrics:
    """集中管理所有 Prometheus 指标"""

    def __init__(self):
        # ── HTTP 请求 ──
        self.http_requests_total = Counter(
            "rag_http_requests_total",
            "HTTP 请求总数",
            ["method", "endpoint", "status_code"],
        )
        self.http_request_duration_seconds = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP 请求延迟 (秒)",
            ["method", "endpoint"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

        # ── 检索 ──
        self.retrieval_total = Counter(
            "rag_retrieval_total",
            "检索请求总数",
            ["mode"],  # local / web / hybrid
        )
        self.retrieval_duration_seconds = Histogram(
            "rag_retrieval_duration_seconds",
            "检索延迟 (秒)",
            ["mode"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self.retrieval_chunks_returned = Histogram(
            "rag_retrieval_chunks_returned",
            "每次检索返回的 chunk 数",
            ["mode"],
            buckets=(0, 1, 3, 5, 10, 20, 50),
        )

        # ── LLM ──
        self.llm_requests_total = Counter(
            "rag_llm_requests_total",
            "LLM 调用总数",
            ["provider", "model"],
        )
        self.llm_duration_seconds = Histogram(
            "rag_llm_duration_seconds",
            "LLM 调用延迟 (秒)",
            ["provider", "model"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.llm_tokens_used = Counter(
            "rag_llm_tokens_total",
            "LLM token 消耗",
            ["provider", "model", "direction"],  # direction: input / output
        )
        self.llm_errors_total = Counter(
            "rag_llm_errors_total",
            "LLM 调用失败数",
            ["provider", "model"],
        )

        # ── 内容抓取 ──
        self.content_fetch_total = Counter(
            "rag_content_fetch_total",
            "全文抓取总数",
            ["strategy", "success"],  # strategy: trafilatura / brightdata / playwright
        )

        # ── Ingest ──
        self.ingest_total = Counter(
            "rag_ingest_total",
            "文档 ingest 总数",
            ["status"],  # success / failed
        )
        self.ingest_duration_seconds = Histogram(
            "rag_ingest_duration_seconds",
            "单文档 ingest 延迟 (秒)",
            buckets=(1, 5, 10, 30, 60, 120, 300),
        )

        # ── 系统 ──
        self.active_connections = Gauge(
            "rag_active_connections",
            "当前活跃 SSE 连接数",
        )
        self.app_info = Info(
            "rag_app",
            "应用元信息",
        )

        # ── 任务队列 (Chat + DR 统一槽位) ──
        self.task_queue_active_slots = Gauge(
            "rag_task_queue_active_slots",
            "当前占用的活跃槽位数 (0..max_slots)",
        )
        self.task_queue_pending_count = Gauge(
            "rag_task_queue_pending_count",
            "排队中的任务数",
        )
        self.task_queue_submitted_total = Counter(
            "rag_task_queue_submitted_total",
            "提交到队列的任务总数",
            ["kind"],  # chat / dr
        )
        self.task_queue_cancelled_total = Counter(
            "rag_task_queue_cancelled_total",
            "取消的任务数",
            ["kind"],
        )
        self.task_queue_timeout_total = Counter(
            "rag_task_queue_timeout_total",
            "超时任务数",
            ["kind"],
        )


# 单例
metrics = _Metrics()
