"""
API 请求/响应 Pydantic 模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""

    session_id: Optional[str] = Field(None, description="会话 ID，不传则创建新会话")
    user_id: Optional[str] = Field(None, description="用户 ID，用于 Persistent Store 偏好")
    canvas_id: Optional[str] = Field(None, description="画布 ID，新建会话时可绑定")
    message: str = Field(..., min_length=1, description="用户消息")
    collection: Optional[str] = Field(
        None,
        description="本地检索目标知识库（Milvus collection），None 表示使用默认库",
    )
    search_mode: str = Field("local", description="检索模式: local | web | hybrid | none")
    web_providers: Optional[List[str]] = Field(
        None,
        description="Web 搜索来源选择: tavily | scholar | google | semantic，可传一个或多个",
    )
    web_source_configs: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="每个搜索源的独立配置 {provider_id: {topK: int, threshold: float, useSerpapi?: bool}}",
    )
    serpapi_ratio: Optional[float] = Field(
        None,
        description="SerpAPI 轮询比例 0-1，控制 google/scholar 中走 SerpAPI 的查询比例",
    )
    use_query_expansion: Optional[bool] = Field(
        None,
        description="仅 Tavily：是否启用多查询扩展，None 表示使用配置默认值",
    )
    use_query_optimizer: Optional[bool] = Field(
        None,
        description="是否启用查询优化器（针对不同搜索引擎优化查询格式），None 表示使用配置默认值",
    )
    query_optimizer_max_queries: Optional[int] = Field(
        None,
        description="查询优化器：每个搜索引擎每种语言查询数（1-5，中文输入时=中文N+英文N），None 表示使用配置默认值",
    )
    local_top_k: Optional[int] = Field(
        None,
        description="本地检索返回的最大文档数，None 表示使用配置默认值",
    )
    local_threshold: Optional[float] = Field(
        None,
        description="本地检索的相似度阈值 (0-1)，低于此阈值的结果会被过滤，None 表示不过滤",
    )
    year_start: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="年份窗口起始（硬过滤，含边界）",
    )
    year_end: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="年份窗口结束（硬过滤，含边界）",
    )
    step_top_k: Optional[int] = Field(
        None,
        description="每步检索保留文档数（local + web 合并重排后），None 表示使用配置默认值",
    )
    llm_provider: Optional[str] = Field(
        None,
        description="LLM 提供商: deepseek | openai | gemini | claude | kimi 等，None 表示使用配置默认值",
    )
    ultra_lite_provider: Optional[str] = Field(
        None,
        description="专门用于长文本压缩等超轻量任务的 LLM 提供商 (如 openai-mini, gemini-flash, deepseek 等)",
    )
    model_override: Optional[str] = Field(
        None,
        description="覆盖 provider 的默认模型，值为 models map 中的 key（如 claude-opus-4-6），None 表示使用 provider 默认模型",
    )
    use_content_fetcher: Optional[str] = Field(
        None,
        description="全文抓取模式: auto | force | off，None 等同于 auto",
    )
    use_agent: Optional[bool] = Field(
        None,
        description="[兼容旧字段] 是否启用 Agent 模式，推荐使用 agent_mode 替代",
    )
    agent_mode: Optional[str] = Field(
        None,
        description=(
            "Agent 执行模式: "
            "standard（标准 RAG，无 Agent）| "
            "assist（辅助：先预检索再 ReAct，Agent 按需补充工具调用）| "
            "autonomous（自主：跳过预检索，Agent 全权自主检索和推理）。"
            "优先级高于 use_agent；为 None 时回退到 use_agent 字段推断。"
        ),
    )
    clarification_answers: Optional[Dict[str, str]] = Field(
        None,
        description="Deep Research 澄清问题回答 {question_id: answer_text}",
    )
    output_language: Optional[str] = Field(
        None,
        description="Deep Research 输出语言: auto | en | zh",
    )
    step_models: Optional[Dict[str, Optional[str]]] = Field(
        None,
        description="Deep Research 各步骤模型覆盖，格式 provider::model，键支持 scope/plan/research/evaluate/write/verify/synthesize",
    )
    reranker_mode: Optional[str] = Field(
        None,
        description="重排序模式: bge_only | colbert_only | cascade；None 表示使用服务端配置默认值",
    )
    mode: Optional[str] = Field(
        "chat",
        description="执行模式: chat（普通对话）| deep_research（多步综述流水线），默认 chat",
    )
    session_preference_local_db: Optional[str] = Field(
        None,
        description=(
            "本会话本地库偏好（由前端在用户选择「换库/本会话不用本地库」后传入）："
            "no_local = 本会话暂不使用本地库；use = 仍使用当前库。"
        ),
    )


class EvidenceSummary(BaseModel):
    """检索证据摘要（用于响应）"""

    query: str = ""
    total_chunks: int = 0
    sources_used: List[str] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
    # 证据综合元数据
    year_range: Optional[List[Optional[int]]] = Field(None, description="证据时间跨度 [earliest, latest]")
    source_breakdown: Optional[Dict[str, int]] = Field(None, description="来源分布（来源级：同 URL/文档只算一次）{local: N, tavily: M, ...}")
    evidence_type_breakdown: Optional[Dict[str, int]] = Field(None, description="证据类型分布 {finding: N, method: M, ...}")
    cross_validated_count: int = Field(0, description="本地+网络交叉验证的文献数")
    total_documents: int = Field(0, description="涉及的独立文献总数")
    # 双层来源统计
    provider_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="双层来源统计: chunk_level（每个信息块计一次）+ citation_level（同网站/文档只计一次）",
    )
    # 证据充分性
    evidence_scarce: bool = Field(False, description="True when pre-retrieval evidence is insufficient (chunks<3 or distinct_docs<2)")
    # 检索诊断信息
    diagnostics: Optional[Dict[str, Any]] = Field(None, description="检索诊断: stages/web_providers/content_fetcher")


class ChatCitation(BaseModel):
    """对话引用项（轻量版）"""

    cite_key: str = ""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doc_id: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    bbox: Optional[List[float]] = Field(None, description="Docling bbox 坐标 [x0,y0,x1,y1]")
    page_num: Optional[int] = Field(None, description="证据所在页码")
    provider: Optional[str] = Field(None, description="来源 provider: local | tavily | scholar | semantic | ncbi | google")


class ChatResponse(BaseModel):
    """对话响应"""

    session_id: str = Field(..., description="会话 ID")
    response: str = Field(..., description="助手回复")
    citations: List[ChatCitation] = Field(default_factory=list, description="引用来源列表")
    evidence_summary: Optional[EvidenceSummary] = Field(None, description="本轮检索摘要")
    prompt_local_db_choice: bool = Field(
        False,
        description="是否提示用户选择：当前查询与本地库可能不符，可换库或本会话不用本地库",
    )
    local_db_mismatch_message: Optional[str] = Field(
        None,
        description="当 prompt_local_db_choice 为 true 时，前端展示的提示文案",
    )


class ChatSubmitResponse(BaseModel):
    """异步 Chat 提交响应"""

    task_id: str = Field(..., description="任务 ID，用于 GET /chat/stream/{task_id} 订阅流式结果")


class TaskStateItem(BaseModel):
    """任务状态项（排队区）"""

    task_id: str = ""
    kind: str = "chat"
    status: str = "queued"
    session_id: str = ""
    user_id: str = ""
    queue_position: int = 0
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error_message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class TaskQueueResponse(BaseModel):
    """排队区快照"""

    active_count: int = 0
    max_slots: int = 2
    active: List[TaskStateItem] = Field(default_factory=list)
    queued: List[Dict[str, Any]] = Field(default_factory=list)


class TaskCancelResponse(BaseModel):
    """取消任务响应"""

    success: bool = Field(..., description="是否取消成功")
    message: str = Field("", description="说明")


class TurnItem(BaseModel):
    """单轮对话项"""

    role: str
    content: str
    sources: List[ChatCitation] = Field(default_factory=list, description="该轮对话的引用来源")


class SessionInfo(BaseModel):
    """会话信息"""

    session_id: str
    canvas_id: str = ""
    stage: str = Field("explore", description="工作流阶段: explore | outline | drafting | refine")
    turn_count: int = 0
    turns: List[TurnItem] = Field(default_factory=list)
    research_dashboard: Optional[Dict[str, Any]] = Field(
        None,
        description="最近一次 Deep Research 的仪表盘数据（用于刷新后恢复进度列表）",
    )


class SessionListItem(BaseModel):
    """会话列表项（用于历史记录）"""

    session_id: str
    title: str = Field(..., description="会话标题（首条消息生成或第一轮用户消息）")
    canvas_id: str = ""
    stage: str = "explore"
    turn_count: int = 0
    session_type: str = Field(default="chat", description="chat | research")
    created_at: str
    updated_at: str


# ---- Canvas ----

class CanvasCreateRequest(BaseModel):
    session_id: str = ""
    topic: str = ""


class CanvasUpdateRequest(BaseModel):
    session_id: Optional[str] = None
    topic: Optional[str] = None
    working_title: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    stage: Optional[str] = None
    refined_markdown: Optional[str] = None
    user_directives: Optional[List[str]] = None
    skip_draft_review: Optional[bool] = None
    skip_refine_review: Optional[bool] = None


class OutlineSectionSchema(BaseModel):
    id: str = ""
    title: str = ""
    level: int = 1
    order: int = 0
    parent_id: Optional[str] = None
    status: str = "todo"
    guidance: Optional[str] = None


class OutlineUpsertRequest(BaseModel):
    sections: List[OutlineSectionSchema] = Field(default_factory=list)


class DraftBlockSchema(BaseModel):
    section_id: str = ""
    content_md: str = ""
    version: int = 1
    used_fragment_ids: List[str] = Field(default_factory=list)
    used_citation_ids: List[str] = Field(default_factory=list)


class DraftUpsertRequest(BaseModel):
    block: DraftBlockSchema


class CitationResponse(BaseModel):
    cite_key: str = ""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    bibtex: Optional[str] = None


class CitationFilterRequest(BaseModel):
    """引用筛选请求：保留或删除指定 cite_key"""

    keep_keys: Optional[List[str]] = Field(None, description="仅保留的 cite_key 列表（与 remove_keys 二选一）")
    remove_keys: Optional[List[str]] = Field(None, description="要删除的 cite_key 列表（与 keep_keys 二选一）")


class CitationFilterResponse(BaseModel):
    """引用筛选响应"""

    removed_count: int = 0
    remaining_keys: List[str] = Field(default_factory=list)


class CanvasAIEditRequest(BaseModel):
    """Canvas AI 段落编辑请求"""

    section_text: str = Field(..., min_length=1, description="选中的段落文本")
    action: str = Field(..., description="操作: rewrite | expand | condense | add_citations | targeted_refine")
    context: str = Field("", description="可选：周围上下文")
    search_mode: str = Field("local", description="是否检索补充资料: local | web | hybrid | none")
    directive: str = Field("", description="可选：定向精炼指令（targeted_refine 推荐）")
    preserve_citations: bool = Field(True, description="是否启用引用保护（默认开启）")


class CanvasAIEditResponse(BaseModel):
    """Canvas AI 段落编辑响应"""

    edited_text: str = Field(..., description="AI 修改后的文本")
    citations_added: List[str] = Field(default_factory=list, description="新增的引用 keys")
    citation_guard_triggered: bool = Field(False, description="是否触发了引用保护兜底")
    citation_guard_message: str = Field("", description="引用保护提示信息")


class ExportRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话 ID")
    canvas_id: Optional[str] = Field(None, description="画布 ID")
    format: str = Field("markdown", description="导出格式: markdown | docx")
    cite_key_format: Optional[str] = Field(
        None,
        description="引用键格式: numeric | hash | author_date（默认从配置读取）"
    )


class ExportResponse(BaseModel):
    format: str = "markdown"
    content: str = ""
    session_id: str = ""
    canvas_id: str = ""


class AnnotationSchema(BaseModel):
    """行内批注"""

    id: str = ""
    section_id: str = ""
    target_text: str = ""
    directive: str = ""
    status: str = "pending"
    created_at: Optional[str] = None


class ResearchBriefSchema(BaseModel):
    """研究简报（Explore 阶段的 Survey Canvas 板块）"""

    scope: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    key_questions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    time_range: str = ""
    source_priority: List[str] = Field(default_factory=list)
    action_plan: str = ""


class CanvasResponse(BaseModel):
    id: str
    session_id: str = ""
    topic: str = ""
    working_title: str = ""
    abstract: str = ""
    keywords: List[str] = Field(default_factory=list)
    stage: str = "explore"
    refined_markdown: str = ""
    outline: List[Dict[str, Any]] = Field(default_factory=list)
    drafts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    citation_pool: List[Dict[str, Any]] = Field(default_factory=list)
    identified_gaps: List[str] = Field(default_factory=list)
    user_directives: List[str] = Field(default_factory=list)
    annotations: List[AnnotationSchema] = Field(default_factory=list)
    research_brief: Optional[ResearchBriefSchema] = None
    research_insights: List[str] = Field(default_factory=list)
    skip_draft_review: bool = False
    skip_refine_review: bool = False
    version: int = 1


class CanvasVersionItem(BaseModel):
    version_number: int
    created_at: str


class CanvasRefineRequest(BaseModel):
    """全文精炼请求（支持多轮迭代）。"""

    content_md: str = Field("", description="当前全文 Markdown；为空时使用画布已保存版本")
    directives: List[str] = Field(default_factory=list, description="本轮附加指令")
    save_snapshot_before: bool = Field(True, description="精炼前是否创建快照")
    locked_ranges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="锁定片段范围列表，元素格式: {start, end, text}",
    )


class CanvasRefineResponse(BaseModel):
    edited_markdown: str = Field("", description="精炼后的全文")
    snapshot_version: Optional[int] = Field(None, description="精炼前快照版本号（若启用）")
    locked_applied: int = Field(0, description="本轮成功应用的锁定片段数")
    locked_skipped: int = Field(0, description="本轮跳过的锁定片段数（越界/失配/重叠）")
    lock_guard_triggered: bool = Field(False, description="是否触发锁定保护兜底")
    lock_guard_message: str = Field("", description="锁定保护提示")


# ---- Intent Detection ----

class IntentDetectRequest(BaseModel):
    """意图检测请求"""

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: Optional[str] = Field(None, description="会话 ID，用于获取历史上下文")
    current_stage: str = Field("explore", description="当前工作流阶段")
    llm_provider: Optional[str] = Field(None, description="LLM 提供商，用于意图解析；无则使用 config 默认")


class IntentDetectResponse(BaseModel):
    """意图检测响应（简化版：chat vs deep_research）"""

    mode: str = Field("chat", description="执行模式: chat | deep_research")
    confidence: float = Field(..., description="置信度 0-1")
    suggested_topic: str = Field("", description="如果为 deep_research，建议的综述主题")
    params: Dict[str, Any] = Field(default_factory=dict, description="解析出的参数")
    # 兼容旧字段
    intent_type: str = Field("chat", description="[兼容] 意图类型")
    needs_retrieval: bool = Field(True, description="[兼容] 是否需要检索（现由 UI search_mode 决定）")
    suggested_search_mode: str = Field("hybrid", description="[兼容] 建议的检索模式")


# ---- Model Sync ----

class ModelSyncRequest(BaseModel):
    """模型同步请求"""

    force_update: bool = Field(False, description="是否强制更新模型缓存")
    local_files_only: Optional[bool] = Field(None, description="仅使用本地缓存（不联网）")


class ModelStatusItem(BaseModel):
    name: str
    model_id: str
    cache_dir: str
    exists: bool
    local_files_only: bool
    error: Optional[str] = None


class ModelSyncItem(BaseModel):
    name: str
    model_id: str
    cache_dir: str
    local_files_only: bool
    updated: bool
    status: str
    message: Optional[str] = None
    error: Optional[str] = None
    resolved_path: Optional[str] = None


class ModelStatusResponse(BaseModel):
    items: List[ModelStatusItem]


class ModelSyncResponse(BaseModel):
    items: List[ModelSyncItem]


class AutoCompleteRequest(BaseModel):
    """自动完成综述请求"""

    topic: str = Field(..., min_length=1, description="综述主题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    canvas_id: Optional[str] = Field(None, description="画布 ID")
    search_mode: str = Field("hybrid", description="检索模式: local | web | hybrid")
    max_sections: int = Field(4, ge=2, le=6, description="最大章节数")


class AutoCompleteResponse(BaseModel):
    """自动完成综述响应"""

    session_id: str = ""
    canvas_id: str = ""
    markdown: str = ""
    outline: List[str] = Field(default_factory=list, description="大纲章节列表")
    citations: List[str] = Field(default_factory=list, description="引用键列表")
    total_time_ms: float = 0.0


# ---- Deep Research ----

class ClarifyRequest(BaseModel):
    """Deep Research 澄清问题生成请求"""

    message: str = Field(..., min_length=1, description="用户的主题描述")
    session_id: Optional[str] = Field(None, description="会话 ID（用于获取 chat 历史上下文）")
    search_mode: str = Field("hybrid", description="检索模式，用于预检索领域资料辅助生成问题")
    llm_provider: Optional[str] = Field(None, description="LLM 提供商")
    ultra_lite_provider: Optional[str] = Field(None, description="超轻量级 LLM 提供商（长文本压缩等）")
    model_override: Optional[str] = Field(None, description="覆盖 provider 默认模型")


class ClarifyQuestion(BaseModel):
    """单个澄清问题"""

    id: str = Field(..., description="问题唯一标识")
    text: str = Field(..., description="问题文本")
    question_type: str = Field("text", description="问题类型: text | choice | multi_choice")
    options: List[str] = Field(default_factory=list, description="选项（choice/multi_choice 类型时）")
    default: str = Field("", description="默认回答/建议值")


class ClarifyResponse(BaseModel):
    """Deep Research 澄清问题响应"""

    questions: List[ClarifyQuestion] = Field(default_factory=list, description="澄清问题列表（1-6个）")
    suggested_topic: str = Field("", description="系统建议的综述主题")
    suggested_outline: List[str] = Field(default_factory=list, description="初步建议的大纲章节")
    research_brief: Optional[Dict[str, Any]] = Field(None, description="结构化研究简报（Scoping Agent 输出）")
    used_fallback: bool = Field(False, description="是否触发了澄清问题降级回退")
    fallback_reason: str = Field("", description="回退原因（若有）")
    llm_provider_used: str = Field("", description="本次实际使用的 provider")
    llm_model_used: str = Field("", description="本次实际使用的模型（尽力标注）")


class DeepResearchRequest(BaseModel):
    """Deep Research 执行请求（携带澄清回答和完整检索参数）"""

    topic: str = Field(..., min_length=1, description="综述主题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    canvas_id: Optional[str] = Field(None, description="画布 ID")
    user_id: Optional[str] = Field(None, description="用户 ID")
    search_mode: str = Field("hybrid", description="检索模式: local | web | hybrid")
    max_sections: int = Field(4, ge=2, le=6, description="最大章节数")
    clarification_answers: Dict[str, str] = Field(
        default_factory=dict,
        description="澄清问题回答 {question_id: answer_text}",
    )
    # ---- 完整检索参数（与 ChatRequest 保持一致）----
    web_providers: Optional[List[str]] = Field(None, description="Web 搜索来源")
    web_source_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="每个搜索源配置")
    serpapi_ratio: Optional[float] = Field(None, description="SerpAPI 轮询比例 0-1")
    use_query_optimizer: Optional[bool] = Field(None, description="启用查询优化器")
    query_optimizer_max_queries: Optional[int] = Field(None, description="每个搜索引擎查询数")
    local_top_k: Optional[int] = Field(None, description="本地检索 top_k")
    local_threshold: Optional[float] = Field(None, description="本地检索阈值")
    year_start: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口起始（硬过滤）")
    year_end: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口结束（硬过滤）")
    step_top_k: Optional[int] = Field(None, description="每步检索保留文档数（合并重排后）")
    write_top_k: Optional[int] = Field(None, description="Deep Research 写作阶段保留文档数")
    llm_provider: Optional[str] = Field(None, description="LLM 提供商")
    ultra_lite_provider: Optional[str] = Field(None, description="超轻量级 LLM 提供商（长文本压缩等）")
    model_override: Optional[str] = Field(None, description="覆盖 provider 默认模型")
    use_agent: bool = Field(False, description="是否使用递归研究 Agent（LangGraph 引擎）")
    output_language: str = Field("auto", description="输出语言: auto | en | zh")
    step_models: Optional[Dict[str, Optional[str]]] = Field(
        None,
        description="各步骤模型覆盖：{step: 'provider::model' | null}",
    )


class DeepResearchStartRequest(BaseModel):
    """Deep Research 第一阶段请求：生成 Brief + Outline（待用户确认）"""

    topic: str = Field(..., min_length=1, description="综述主题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    canvas_id: Optional[str] = Field(None, description="画布 ID")
    user_id: Optional[str] = Field(None, description="用户 ID")
    search_mode: str = Field("hybrid", description="检索模式: local | web | hybrid")
    max_sections: int = Field(4, ge=2, le=6, description="最大章节数")
    clarification_answers: Optional[Dict[str, str]] = Field(
        None,
        description="澄清问题回答 {question_id: answer_text}",
    )
    output_language: str = Field("auto", description="输出语言: auto | en | zh")
    step_models: Optional[Dict[str, Optional[str]]] = Field(
        None,
        description="各步骤模型覆盖：{step: 'provider::model' | null}",
    )
    step_model_strict: bool = Field(
        False,
        description="步骤模型解析是否严格模式：true=任一步骤 provider 解析失败直接报错；false=自动回退默认模型并记录告警",
    )
    web_providers: Optional[List[str]] = Field(None, description="Web 搜索来源")
    web_source_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="每个搜索源配置")
    serpapi_ratio: Optional[float] = Field(None, description="SerpAPI 轮询比例 0-1")
    use_query_optimizer: Optional[bool] = Field(None, description="启用查询优化器")
    query_optimizer_max_queries: Optional[int] = Field(None, description="每个搜索引擎查询数")
    local_top_k: Optional[int] = Field(None, description="本地检索 top_k")
    local_threshold: Optional[float] = Field(None, description="本地检索阈值")
    year_start: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口起始（硬过滤）")
    year_end: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口结束（硬过滤）")
    step_top_k: Optional[int] = Field(None, description="每步检索保留文档数")
    write_top_k: Optional[int] = Field(None, description="Deep Research 写作阶段保留文档数")
    llm_provider: Optional[str] = Field(None, description="LLM 提供商")
    ultra_lite_provider: Optional[str] = Field(None, description="超轻量级 LLM 提供商（长文本压缩等）")
    model_override: Optional[str] = Field(None, description="覆盖 provider 默认模型")
    collection: Optional[str] = Field(None, description="本地检索目标 collection")
    use_content_fetcher: Optional[str] = Field(None, description="全文抓取模式: auto | force | off")
    gap_query_intent: Optional[str] = Field(
        None,
        description="Gap 查询意图: broad | review_pref | reviews_only",
    )
    reranker_mode: Optional[str] = Field(
        None,
        description="重排序模式: bge_only | colbert_only | cascade；研究过程强制 bge_only，写作阶段使用此值",
    )


class DeepResearchStartResponse(BaseModel):
    """Deep Research 第一阶段响应"""

    session_id: str = Field("", description="会话 ID")
    canvas_id: str = Field("", description="画布 ID")
    brief: Dict[str, Any] = Field(default_factory=dict, description="研究简报")
    outline: List[str] = Field(default_factory=list, description="建议大纲")
    initial_stats: Dict[str, Any] = Field(default_factory=dict, description="初始检索统计")


class DeepResearchConfirmRequest(BaseModel):
    """Deep Research 第二阶段请求：确认后执行研究"""

    topic: str = Field(..., min_length=1, description="综述主题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    canvas_id: Optional[str] = Field(None, description="画布 ID")
    user_id: Optional[str] = Field(None, description="用户 ID")
    search_mode: str = Field("hybrid", description="检索模式: local | web | hybrid")
    confirmed_outline: List[str] = Field(default_factory=list, description="用户确认后的大纲")
    confirmed_brief: Optional[Dict[str, Any]] = Field(None, description="用户确认后的简报（可选）")
    depth: str = Field(
        "comprehensive",
        description=(
            "研究深度: lite（快速探索，~3-10 min）| comprehensive（全面学术综述，~15-40 min）。"
            "控制每章研究轮次、覆盖度阈值、搜索量和审核超时等全部循环上限。"
        ),
    )
    output_language: str = Field("auto", description="输出语言: auto | en | zh")
    step_models: Optional[Dict[str, Optional[str]]] = Field(
        None,
        description="各步骤模型覆盖：{step: 'provider::model' | null}",
    )
    step_model_strict: bool = Field(
        False,
        description="步骤模型解析是否严格模式：true=任一步骤 provider 解析失败直接报错；false=自动回退默认模型并记录告警",
    )
    web_providers: Optional[List[str]] = Field(None, description="Web 搜索来源")
    web_source_configs: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="每个搜索源配置")
    serpapi_ratio: Optional[float] = Field(None, description="SerpAPI 轮询比例 0-1")
    use_query_optimizer: Optional[bool] = Field(None, description="启用查询优化器")
    query_optimizer_max_queries: Optional[int] = Field(None, description="每个搜索引擎查询数")
    local_top_k: Optional[int] = Field(None, description="本地检索 top_k")
    local_threshold: Optional[float] = Field(None, description="本地检索阈值")
    year_start: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口起始（硬过滤）")
    year_end: Optional[int] = Field(None, ge=1900, le=2100, description="年份窗口结束（硬过滤）")
    step_top_k: Optional[int] = Field(None, description="每步检索保留文档数")
    write_top_k: Optional[int] = Field(None, description="Deep Research 写作阶段保留文档数")
    llm_provider: Optional[str] = Field(None, description="LLM 提供商")
    ultra_lite_provider: Optional[str] = Field(None, description="超轻量级 LLM 提供商（长文本压缩等）")
    model_override: Optional[str] = Field(None, description="覆盖 provider 默认模型")
    collection: Optional[str] = Field(None, description="本地检索目标 collection")
    use_content_fetcher: Optional[str] = Field(None, description="全文抓取模式: auto | force | off")
    gap_query_intent: Optional[str] = Field(
        None,
        description="Gap 查询意图: broad | review_pref | reviews_only",
    )
    reranker_mode: Optional[str] = Field(
        None,
        description="重排序模式: bge_only | colbert_only | cascade；研究过程强制 bge_only，写作阶段使用此值",
    )
    user_context: Optional[str] = Field(
        None,
        description="用户补充的观点/约束（仅本次 Deep Research 使用，不写入持久知识库）",
    )
    user_context_mode: Optional[str] = Field(
        "supporting",
        description="用户文本上下文模式: supporting | direct_injection",
    )
    user_documents: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="用户上传的临时材料摘要列表 [{name, content}]，仅本次任务使用",
    )
    # ---- 阶段跳过控制 ----
    skip_draft_review: bool = Field(
        False,
        description="跳过逐章审阅（Drafting 阶段 Agent 连续写完所有章节）",
    )
    skip_refine_review: bool = Field(
        False,
        description="跳过精炼审阅（Refine 阶段直接展示最终结果）",
    )
    skip_claim_generation: bool = Field(
        False,
        description="跳过前置论点提炼（Claim Generation）阶段，直接进入写作",
    )
    max_sections: int = Field(4, ge=2, le=6, description="最大章节数")


class DeepResearchSubmitResponse(BaseModel):
    """Deep Research 后台任务提交响应"""

    ok: bool = True
    job_id: str = Field(..., description="后台任务 ID")
    session_id: str = Field("", description="会话 ID")
    canvas_id: str = Field("", description="画布 ID")


class DeepResearchRestartPhaseRequest(BaseModel):
    """Restart deep-research from a major phase."""

    phase: str = Field(
        ...,
        description="重启阶段: plan | research | generate_claims | write | verify | review_gate | synthesize",
    )


class DeepResearchRestartSectionRequest(BaseModel):
    """Restart deep-research for a specific outline section."""

    section_title: str = Field(..., min_length=1, description="章节标题（与 confirmed_outline 一致）")
    action: str = Field("research", description="重启动作: research | write")


class DeepResearchJobInfo(BaseModel):
    """Deep Research 后台任务状态"""

    job_id: str
    topic: str
    session_id: str = ""
    canvas_id: str = ""
    status: str
    current_stage: str = ""
    message: str = ""
    error_message: str = ""
    result_markdown: str = ""
    result_citations: List[Dict[str, Any]] = Field(default_factory=list)
    result_dashboard: Dict[str, Any] = Field(default_factory=dict)
    total_time_ms: float = 0.0
    created_at: float
    updated_at: float
    finished_at: Optional[float] = None


class DeepResearchContextExtractResponse(BaseModel):
    """Deep Research 临时上下文文件提取结果"""

    documents: List[Dict[str, str]] = Field(default_factory=list, description="提取后的临时文档 [{name, content}]")


# ---- Auth ----

class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户名/用户 ID")
    password: str = Field(..., min_length=1, description="密码")


class LoginResponse(BaseModel):
    token: str = Field(..., description="登录后使用的 token")
    user_id: str = ""
    role: str = "user"


class CreateUserRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="用户名/用户 ID")
    password: str = Field(..., min_length=1, description="密码")
    role: str = Field("user", description="角色: user | admin")


class UserItem(BaseModel):
    user_id: str = ""
    role: str = "user"
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""
