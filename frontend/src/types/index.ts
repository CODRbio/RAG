// ============================================================
// User & Auth
// ============================================================

export interface User {
  user_id: string;
  username?: string;
  role: 'user' | 'admin';
  avatar?: string;
}

export interface LoginRequest {
  user_id: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user_id: string;
  role: string;
}

export interface UserItem {
  user_id: string;
  role: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

// ============================================================
// Chat & Messages
// ============================================================

export interface Source {
  id: string | number;
  cite_key: string;
  title: string;
  authors: string[];
  year?: number | null;
  doc_id?: string | null;
  url?: string | null;
  pdf_url?: string | null;
  doi?: string | null;
  score?: number;
  snippet?: string;
  path?: string;
  type?: 'local' | 'web';
  provider?: string;  // local | tavily | google | scholar | semantic | semantic_snippet | semantic_bulk | ncbi | serpapi | serpapi_scholar | serpapi_google
  bbox?: number[];
  page_num?: number | null;
}

export interface Message {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp?: string;
  providerStats?: {
    chunk_level: Record<string, number>;
    citation_level: Record<string, number>;
  };
  agentDebug?: AgentDebugData | null;
}

export interface ActiveResponseState {
  kind: 'chat' | 'research';
  taskOrJobId: string;
  surface: 'chat' | 'research';
  stepKey: string | null;
  stepLabel: string;
  streamPhase: 'thinking' | 'streaming' | 'paused';
  targetMessageId?: string | null;
  hasVisibleOutput: boolean;
}

export interface RetrievalStageDiag {
  count: number;
  time_ms: number;
}

export interface PoolFusionDiag {
  main_in?: number;
  gap_in?: number;
  agent_in?: number;
  rank_pool_k?: number;
  rank_pool_multiplier?: number;
  gap_ratio?: number | null;
  agent_ratio?: number | null;
  gap_min_keep?: number;
  gap_in_output?: number;
  agent_min_keep?: number;
  agent_in_output?: number;
  output_count?: number;
}

export interface RetrievalDiagnostics {
  optimized_queries?: string[];
  stages?: Record<string, RetrievalStageDiag>;
  web_providers?: Record<string, RetrievalStageDiag>;
  content_fetcher?: { enriched: number; total: number };
  cross_source_dedup?: { removed: number; remaining: number };
  cache_hit?: boolean;
  pool_fusion?: PoolFusionDiag;
  final_fusion?: PoolFusionDiag;
  agent_refusion?: {
    agent_extra_chunks?: number;
    gap_candidates?: number;
    main_candidates?: number;
    output_count?: number;
  };
}

export interface EvidenceSummary {
  query: string;
  total_chunks: number;
  sources_used: string[];
  retrieval_time_ms: number;
  // P0 证据综合元数据
  year_range?: [number | null, number | null];
  source_breakdown?: Record<string, number>;
  evidence_type_breakdown?: Record<string, number>;
  cross_validated_count?: number;
  total_documents?: number;
  provider_stats?: {
    chunk_level: Record<string, number>;
    citation_level: Record<string, number>;
  };
  // P1 检索诊断
  diagnostics?: RetrievalDiagnostics;
}

/** Pre-Research (Sonar) strength: off | model id from Perplexity (e.g. sonar | sonar-pro | sonar-reasoning-pro | sonar-deep-research) */
export type SonarStrength = 'off' | 'sonar' | 'sonar-pro' | 'sonar-reasoning-pro' | 'sonar-deep-research' | (string & {});

export interface ChatRequest {
  session_id?: string;
  user_id?: string;
  canvas_id?: string;
  message: string;
  collection?: string;
  collections?: string[];
  search_mode: 'local' | 'web' | 'hybrid' | 'none';
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number; useSerpapi?: boolean }>;
  serpapi_ratio?: number;  // SerpAPI 轮询比例 0-1（仅当 google/scholar 启用 useSerpapi 时生效）
  use_query_expansion?: boolean;  // 兼容字段（已弃用）
  local_top_k?: number;  // 本地检索返回的最大文档数
  pool_score_thresholds?: Record<string, number>;
  /** 合并池分数阈值 (0-1)，仅作用于融合后的最终池。默认 0.35。 */
  fused_pool_score_threshold?: number;
  /** @deprecated 请用 fused_pool_score_threshold */
  local_threshold?: number;
  year_start?: number;  // 年份窗口起始（硬过滤）
  year_end?: number;  // 年份窗口结束（硬过滤）
  step_top_k?: number;  // 每步检索保留的文档数（local + web 合并重排后）
  write_top_k?: number;
  graph_top_k?: number;
  llm_provider?: string;  // LLM 提供商: deepseek | openai | gemini | claude | kimi 等
  ultra_lite_provider?: string;  // 长文本压缩等超轻量任务用的 provider（如 openai-mini, gemini-flash）
  model_override?: string;  // 覆盖默认模型，如 claude-opus-4-6
  mode?: ChatMode;  // 执行模式: chat（默认）| deep_research
  use_content_fetcher?: 'auto' | 'force' | 'off';  // 是否对网络搜索结果做全文抓取（None 用后端默认）
  use_agent?: boolean;  // [兼容旧字段] 是否启用 Agent，推荐使用 agent_mode
  agent_mode?: 'standard' | 'assist' | 'autonomous';  // Agent 执行模式
  /** Pre-Research 强度：off | sonar | sonar-pro | sonar-reasoning-pro */
  sonar_strength?: SonarStrength;
  /** @deprecated 使用 sonar_strength，后端兼容 */
  use_sonar_prelim?: boolean;
  /** @deprecated 使用 sonar_strength，后端兼容 */
  sonar_model?: string;
  /** Sonar 检索工具模型（仅当 web_providers 含 sonar 时生效）：sonar | sonar-pro */
  agent_sonar_model?: string;
  max_iterations?: number;  // Agent ReAct 最大迭代轮数，默认 2，仅 assist/autonomous 时生效
  clarification_answers?: Record<string, string>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
  reranker_mode?: 'bge_only' | 'colbert_only' | 'cascade';  // 重排序模式，None 使用服务端默认
  /** 本会话本地库偏好：no_local=本会话不用本地库，use=仍使用当前库（用于回复「查询与本地库范围不符」的提示） */
  session_preference_local_db?: 'no_local' | 'use';
  /** 前端调试面板开启时传 true，本请求期间后端临时开启 DEBUG 级别日志 */
  agent_debug_mode?: boolean;
  enable_graphic_abstract?: boolean;
  graphic_abstract_model?: string;
}

export interface ChatCitation {
  cite_key: string;
  title: string;
  authors: string[];
  year?: number | null;
  doc_id?: string | null;
  url?: string | null;
  doi?: string | null;
  bbox?: number[];
  page_num?: number | null;
  provider?: string | null;
}

export interface ChatResponse {
  session_id: string;
  response: string;
  citations: ChatCitation[];
  evidence_summary?: EvidenceSummary;
  /** 是否提示用户选择：当前查询与本地库可能不符 */
  prompt_local_db_choice?: boolean;
  /** 当 prompt_local_db_choice 为 true 时的提示文案 */
  local_db_mismatch_message?: string | null;
}

/** 异步 Chat 提交响应 */
export interface ChatSubmitResponse {
  task_id: string;
}

/** 任务状态（排队区） */
export type TaskStatus = 'queued' | 'running' | 'pausing' | 'paused' | 'completed' | 'error' | 'cancelled' | 'timeout';

export interface TaskStateItem {
  task_id: string;
  kind: string;
  status: TaskStatus;
  session_id: string;
  user_id: string;
  queue_position: number;
  created_at?: number;
  started_at?: number;
  finished_at?: number;
  pause_started_at?: number;
  paused_total_seconds?: number;
  error_message?: string;
  payload?: Record<string, unknown>;
}

/** 排队区快照 */
export interface TaskQueueResponse {
  active_count: number;
  max_slots: number;
  active: TaskStateItem[];
  queued: Array<{ task_id: string; kind: string; session_id: string; user_id: string; queue_position: number; state?: TaskStateItem | null }>;
}

export interface SessionInfo {
  session_id: string;
  canvas_id: string;
  stage: string;
  turn_count: number;
  turns: { role: string; content: string; sources?: ChatCitation[]; timestamp?: string | number }[];
  research_dashboard?: ResearchDashboardData | null;
}

export interface SessionListItem {
  session_id: string;
  title: string;
  canvas_id: string;
  stage: string;
  turn_count: number;
  /** chat | research */
  session_type?: string;
  created_at: string;
  updated_at: string;
}

// ============================================================
// Intent / Mode（简化版：Chat vs Deep Research）
// ============================================================

/**
 * 执行模式（与后端 IntentType 对齐）
 */
export type ChatMode = 'chat' | 'deep_research';

/**
 * 兼容旧的 IntentType（旧值在运行时会被后端映射为 chat/deep_research）
 */
export type IntentType = ChatMode | string;

// 兼容旧代码的类型别名
export type IntentMode = 'auto' | 'search' | 'write' | 'chat';  // deprecated, kept for type compatibility

/**
 * 显式命令定义（用于命令面板）
 */
export interface CommandDefinition {
  command: string;
  label: string;
  description: string;
  mode: ChatMode;  // 命令触发的模式
  example?: string;
}

export interface IntentDetectRequest {
  message: string;
  session_id?: string;
  current_stage?: string;
}

export interface IntentDetectResponse {
  mode: ChatMode;
  confidence: number;
  suggested_topic: string;
  params: Record<string, unknown>;
  // 兼容旧字段
  intent_type?: string;
  needs_retrieval?: boolean;
  suggested_search_mode?: string;
}

export interface IntentInfo {
  mode: ChatMode;
  confidence: number;
  from_command: boolean;
}

// ============================================================
// Deep Research
// ============================================================

export interface ClarifyQuestion {
  id: string;
  text: string;
  question_type: 'text' | 'choice' | 'multi_choice';
  options: string[];
  default: string;
}

export interface ClarifyResponse {
  questions: ClarifyQuestion[];
  session_id?: string;
  suggested_topic: string;
  suggested_outline: string[];
  preliminary_knowledge?: string;
  used_fallback?: boolean;
  fallback_reason?: string;
  llm_provider_used?: string;
  llm_model_used?: string;
}

export interface DeepResearchRequest {
  topic: string;
  session_id?: string;
  canvas_id?: string;
  user_id?: string;
  search_mode: 'local' | 'web' | 'hybrid';
  max_sections?: number;
  clarification_answers?: Record<string, string>;
  // 完整检索参数
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number }>;
  use_query_optimizer?: boolean;
  query_optimizer_max_queries?: number;
  gap_query_intent?: 'broad' | 'review_pref' | 'reviews_only';
  local_top_k?: number;
  pool_score_thresholds?: Record<string, number>;
  fused_pool_score_threshold?: number;
  local_threshold?: number;
  year_start?: number;
  year_end?: number;
  step_top_k?: number;
  llm_provider?: string;
  ultra_lite_provider?: string;
  model_override?: string;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
}

export interface DeepResearchStartRequest {
  topic: string;
  session_id?: string;
  canvas_id?: string;
  user_id?: string;
  collection?: string;
  collections?: string[];
  search_mode: 'local' | 'web' | 'hybrid';
  max_sections?: number;
  clarification_answers?: Record<string, string>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
  step_model_strict?: boolean;
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number; useSerpapi?: boolean }>;
  serpapi_ratio?: number;
  use_query_optimizer?: boolean;
  query_optimizer_max_queries?: number;
  use_content_fetcher?: 'auto' | 'force' | 'off';
  gap_query_intent?: 'broad' | 'review_pref' | 'reviews_only';
  local_top_k?: number;
  pool_score_thresholds?: Record<string, number>;
  fused_pool_score_threshold?: number;
  local_threshold?: number;
  year_start?: number;
  year_end?: number;
  step_top_k?: number;
  write_top_k?: number;
  graph_top_k?: number;
  llm_provider?: string;
  ultra_lite_provider?: string;
  model_override?: string;
  reranker_mode?: 'bge_only' | 'colbert_only' | 'cascade';
  agent_sonar_model?: string;
  enable_graphic_abstract?: boolean;
  graphic_abstract_model?: string;
}

/** Returned immediately from POST /deep-research/start – poll status endpoint for result. */
export interface DeepResearchStartAsyncResponse {
  job_id: string;
  session_id: string;
}

/** Polling response from GET /deep-research/start/{job_id}/status */
export interface DeepResearchStartStatusResponse {
  job_id: string;
  status: 'running' | 'done' | 'error';
  session_id: string;
  error?: string;
  canvas_id: string;
  brief: Record<string, unknown>;
  outline: string[];
  initial_stats: Record<string, unknown>;
  current_stage?: string;
  progress?: number;
}

/** @deprecated kept for internal compatibility – use DeepResearchStartAsyncResponse */
export interface DeepResearchStartResponse {
  session_id: string;
  canvas_id: string;
  brief: Record<string, unknown>;
  outline: string[];
  initial_stats: Record<string, unknown>;
  used_fallback?: boolean;
  fallback_reason?: string;
}

export interface DeepResearchConfirmRequest {
  topic: string;
  planning_job_id?: string;
  session_id?: string;
  canvas_id?: string;
  user_id?: string;
  collection?: string;
  collections?: string[];
  search_mode: 'local' | 'web' | 'hybrid';
  confirmed_outline: string[];
  confirmed_brief?: Record<string, unknown>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
  step_model_strict?: boolean;
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number; useSerpapi?: boolean }>;
  serpapi_ratio?: number;
  use_query_optimizer?: boolean;
  query_optimizer_max_queries?: number;
  use_content_fetcher?: 'auto' | 'force' | 'off';
  gap_query_intent?: 'broad' | 'review_pref' | 'reviews_only';
  local_top_k?: number;
  pool_score_thresholds?: Record<string, number>;
  fused_pool_score_threshold?: number;
  local_threshold?: number;
  year_start?: number;
  year_end?: number;
  step_top_k?: number;
  write_top_k?: number;
  graph_top_k?: number;
  llm_provider?: string;
  ultra_lite_provider?: string;
  model_override?: string;
  reranker_mode?: 'bge_only' | 'colbert_only' | 'cascade';
  agent_sonar_model?: string;
  user_context?: string;
  user_context_mode?: 'supporting' | 'direct_injection';
  user_documents?: Array<{ name: string; content: string }>;
  // 研究深度
  depth?: 'lite' | 'comprehensive';
  // 阶段跳过控制
  skip_draft_review?: boolean;
  skip_refine_review?: boolean;
  skip_claim_generation?: boolean;
  // 大纲章节数
  max_sections?: number;
  enable_graphic_abstract?: boolean;
  graphic_abstract_model?: string;
}

export interface DeepResearchSubmitResponse {
  ok: boolean;
  job_id: string;
  session_id: string;
  canvas_id: string;
}

export interface DeepResearchRestartPhaseRequest {
  phase: 'plan' | 'research' | 'generate_claims' | 'write' | 'verify' | 'review_gate' | 'synthesize';
}

export interface DeepResearchRestartSectionRequest {
  section_title: string;
  action: 'research' | 'write';
}

export interface DeepResearchRestartIncompleteSectionsRequest {
  section_titles: string[];
  action: 'research' | 'write';
}

export interface DeepResearchRestartWithOutlineRequest {
  new_outline: string[];
  action: 'research' | 'write';
}

export interface DeepResearchJobInfo {
  job_id: string;
  topic: string;
  session_id: string;
  canvas_id: string;
  status: 'planning' | 'pending' | 'running' | 'pausing' | 'paused' | 'cancelling' | 'waiting_review' | 'done' | 'error' | 'cancelled' | string;
  current_stage: string;
  message: string;
  error_message: string;
  result_markdown: string;
  result_citations: ChatCitation[];
  result_dashboard: Record<string, unknown>;
  total_time_ms: number;
  created_at: number;
  updated_at: number;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface DeepResearchJobEvent {
  event_id: number;
  event: string;
  created_at: number;
  data: Record<string, unknown>;
}

// ============================================================
// Model Sync
// ============================================================

export interface ModelStatusItem {
  name: string;
  model_id: string;
  cache_dir: string;
  exists: boolean;
  local_files_only: boolean;
  error?: string | null;
}

export interface ModelSyncItem {
  name: string;
  model_id: string;
  cache_dir: string;
  local_files_only: boolean;
  updated: boolean;
  status: string;
  message?: string | null;
  error?: string | null;
  resolved_path?: string | null;
}

export interface ModelStatusResponse {
  items: ModelStatusItem[];
}

export interface ModelSyncRequest {
  force_update?: boolean;
  local_files_only?: boolean;
}

export interface ModelSyncResponse {
  items: ModelSyncItem[];
}

// ============================================================
// Canvas
// ============================================================

export type CanvasStage = 'explore' | 'outline' | 'drafting' | 'refine';

export interface OutlineSection {
  id: string;
  title: string;
  level: number;
  order: number;
  parent_id?: string;
  status: string;
  guidance?: string;
}

export interface DraftBlock {
  section_id: string;
  content_md: string;
  version: number;
  used_fragment_ids: string[];
  used_citation_ids: string[];
  updated_at?: string;
}

export interface Annotation {
  id: string;
  section_id: string;
  target_text: string;
  directive: string;
  status: 'pending' | 'applied' | 'rejected';
  created_at?: string;
}

export interface CanvasResearchBrief {
  scope: string;
  success_criteria: string[];
  key_questions: string[];
  exclusions: string[];
  time_range: string;
  source_priority: string[];
  action_plan: string;
}

export interface Canvas {
  id: string;
  session_id: string;
  topic: string;
  working_title: string;
  abstract: string;
  preliminary_knowledge?: string;
  keywords: string[];
  stage: CanvasStage;
  refined_markdown: string;
  outline: OutlineSection[];
  drafts: Record<string, DraftBlock>;
  citation_pool: Citation[];
  identified_gaps: string[];
  user_directives: string[];
  annotations: Annotation[];
  research_brief: CanvasResearchBrief | null;
  research_insights: string[];
  skip_draft_review: boolean;
  skip_refine_review: boolean;
  version: number;
}

// ── Gap Supplement Types ──

export type GapSupplementStatus = 'pending' | 'consumed';

export interface GapSupplement {
  id: number;
  job_id: string;
  section_id: string;
  gap_text: string;
  supplement_type: 'material' | 'direct_info';
  content: Record<string, unknown>;
  status: GapSupplementStatus;
  created_at: number;
  consumed_at?: number | null;
}

// ── Research Insight Types ──

export type InsightType = 'gap' | 'conflict' | 'limitation' | 'future_direction';
export type InsightStatus = 'open' | 'addressed' | 'deferred';

export interface ResearchInsight {
  id: number;
  job_id: string;
  section_id: string;
  insight_type: InsightType;
  text: string;
  source_context: string;
  status: InsightStatus;
  created_at: number;
}

export interface Citation {
  id?: string;
  cite_key: string;
  title: string;
  authors: string[];
  year?: number;
  doi?: string;
  url?: string;
  bibtex?: string;
}

// ============================================================
// Projects / History
// ============================================================

export interface Project {
  id: string;
  title: string;           // 后端返回 working_title || topic
  topic?: string;
  working_title?: string;
  stage?: string;
  created_at: string;
  updated_at: string;
  archived: boolean;
  session_id?: string;
}

// ============================================================
// Config
// ============================================================

export interface WebSource {
  id: string;
  name: string;
  enabled: boolean;
  topK: number;
  threshold: number;
  useSerpapi?: boolean;
}

export interface PoolScoreThresholds {
  main: number;
  gap: number;
  agent: number;
}

export interface RagConfig {
  enabled: boolean;  // 是否启用本地 RAG 检索
  localTopK: number;
  poolScoreThresholds?: {
    chat: PoolScoreThresholds;
    research: PoolScoreThresholds;
  };
  /** 合并池分数阈值 (0-1)，仅作用于 local+web 融合后的最终池。默认 0.35。 */
  fusedPoolScoreThreshold: number;
  /** @deprecated 已由 fusedPoolScoreThreshold 取代，保留仅作持久化兼容 */
  localThreshold?: number;
  stepTopK: number;  // 每步检索保留的文档数（local + web 合并重排后）
  writeTopK: number;  // 最终撰写阶段证据保留数量（Deep Research）
  yearStart?: number | null; // 全局年份过滤起始
  yearEnd?: number | null; // 全局年份过滤结束
  enableHippoRAG: boolean;
  graphTopK: number;  // 进入候选池的最大图检索结果数（仅 HippoRAG 开启时生效）
  enableReranker: boolean;
  agentMode: 'standard' | 'assist' | 'autonomous';  // Agent 执行模式
  /** Pre-Research 强度：off 或 Perplexity/Sonar 模型 id（与 LLM 选择同源），默认 sonar-reasoning-pro */
  sonarStrength: SonarStrength;
  /** Sonar 检索工具使用的模型（仅当 Web 来源勾选 Sonar 时生效），与预研究分离；选项来自 /llm/models */
  agentSonarModel?: string;
  maxIterations: number;  // Agent ReAct 最大迭代轮数，默认 2
  agentDebugMode: boolean;  // 是否显示 Agent 详细调试面板
  enableGraphicAbstract?: boolean; // 是否在末尾生成 Graphic Abstract
  graphicAbstractModel?: string; // 生成 Graphic Abstract 所用的模型提供商（gemini, openai, kimi）
}

export interface WebSearchConfig {
  enabled: boolean;
  sources: WebSource[];
  contentFetcherMode: 'auto' | 'force' | 'off';  // 全文抓取模式
  serpapiRatio: number; // SerpAPI 轮询比例 0-100（仅当 google/scholar 开启 useSerpapi 时生效）
}

// ============================================================
// Deep Research Defaults (persistent settings via ⚙ popover)
// ============================================================

export interface DeepResearchDefaults {
  depth: 'lite' | 'comprehensive';
  outputLanguage: 'auto' | 'en' | 'zh';
  yearStart: number | null;
  yearEnd: number | null;
  stepModelStrict: boolean;
  stepModels: Record<string, string>;
  /** 初步研究专用模型，仅从 Perplexity 中选择；无空选项 */
  preliminaryModel: string;
  /** 提问模型，用于生成澄清问题；空则用全局选中模型 */
  questionModel: string;
  skipClaimGeneration: boolean;
  maxSections: number;
  gapQueryIntent: 'broad' | 'review_pref' | 'reviews_only';
  /** 长文本压缩等超轻量任务用的 provider（高级配置中选择，与拉取模型列表比对） */
  ultra_lite_provider?: string | null;
}

export type ScholarDownloadStrategyId =
  | 'direct_download'
  | 'playwright_download'
  | 'browser_lookup'
  | 'sci_hub'
  | 'brightdata'
  | 'anna';

export interface ScholarDownloaderDefaults {
  includeAcademia: boolean;
  assistLlmEnabled: boolean;
  assistLlmMode: 'ultra-lite' | 'lite' | 'auto-upgrade';
  browserMode: 'headed' | 'headless';
  strategyOrder: ScholarDownloadStrategyId[];
}

// ============================================================
// Workflow
// ============================================================

export type WorkflowStep = 'idle' | 'explore' | 'outline' | 'drafting' | 'refine';

/**
 * 工作流阶段详细信息（用于 UI 显示）
 */
export interface WorkflowStageInfo {
  id: WorkflowStep;
  label: string;
  description: string;
  icon: string;
  color: string;
}

/**
 * 预定义的工作流阶段配置
 * label / description 使用 i18n key，在组件中通过 t() 渲染。
 */
export const WORKFLOW_STAGES: WorkflowStageInfo[] = [
  { id: 'explore', label: 'workflow.explore', description: 'workflow.exploreDesc', icon: '🔍', color: 'blue' },
  { id: 'outline', label: 'workflow.outline', description: 'workflow.outlineDesc', icon: '📋', color: 'purple' },
  { id: 'drafting', label: 'workflow.drafting', description: 'workflow.draftingDesc', icon: '✍️', color: 'orange' },
  { id: 'refine', label: 'workflow.refine', description: 'workflow.refineDesc', icon: '✨', color: 'green' },
];

/**
 * 简化命令列表（/auto 触发 Deep Research，其余为 Chat 内 prompt hints）
 * label / description 使用 i18n key，在组件中通过 t() 渲染。
 */
export const COMMAND_LIST: CommandDefinition[] = [
  { command: '/auto', label: 'commands.deepResearch', description: 'commands.deepResearchDesc', mode: 'deep_research', example: '/auto 深海冷泉生态系统' },
  { command: '/search', label: 'commands.search', description: 'commands.searchDesc', mode: 'chat', example: '/search deep sea cold seep' },
  { command: '/outline', label: 'commands.generateOutline', description: 'commands.generateOutlineDesc', mode: 'chat', example: '/outline' },
  { command: '/draft', label: 'commands.draftChapter', description: 'commands.draftChapterDesc', mode: 'chat', example: '/draft introduction' },
  { command: '/export', label: 'commands.exportDoc', description: 'commands.exportDocDesc', mode: 'chat', example: '/export' },
  { command: '/status', label: 'commands.viewStatus', description: 'commands.viewStatusDesc', mode: 'chat', example: '/status' },
];

// ============================================================
// Tool Trace (Agent ReAct Loop)
// ============================================================

export interface ToolTraceItem {
  iteration: number;
  tool: string;
  arguments: Record<string, unknown>;
  result: string;
  is_error: boolean;
  tool_latency_ms?: number;
  llm_latency_ms?: number;
}

export interface AgentStats {
  total_iterations: number;
  total_tool_calls: number;
  tools_used_summary: Record<string, number>;
  total_tool_time_ms: number;
  total_llm_time_ms: number;
  total_agent_time_ms: number;
  error_count: number;
}

export interface AgentDebugData {
  agent_stats: AgentStats;
  tool_trace: ToolTraceItem[];
  tools_contributed: boolean;
  pre_retrieval_chunks: number;
  agent_added_chunks: number;
  cited_from_agent: number;
}

// ============================================================
// Research Dashboard (Deep Research Agent Progress)
// ============================================================

export interface ResearchSectionStatus {
  title: string;
  status: 'pending' | 'researching' | 'writing' | 'reviewing' | 'done';
  coverage_score: number;
  source_count: number;
  gaps: string[];
  evidence_scarce?: boolean;
}

export interface ResearchDashboardData {
  topic: string;
  scope: string;
  progress: number;       // 0-1
  coverage: number;       // 0-1
  confidence: 'low' | 'medium' | 'high';
  total_sources: number;
  total_iterations: number;
  sections: ResearchSectionStatus[];
  coverage_gaps: string[];
  conflict_notes: string[];
}

// ============================================================
// Toast
// ============================================================

export interface Toast {
  id: number;
  msg: string;
  type: 'info' | 'success' | 'error' | 'warning';
}
