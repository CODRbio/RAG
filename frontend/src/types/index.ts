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
  doi?: string | null;
  score?: number;
  snippet?: string;
  path?: string;
  type?: 'local' | 'web';
}

export interface Message {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp?: string;
}

export interface RetrievalStageDiag {
  count: number;
  time_ms: number;
}

export interface RetrievalDiagnostics {
  optimized_queries?: string[];
  stages?: Record<string, RetrievalStageDiag>;
  web_providers?: Record<string, RetrievalStageDiag>;
  content_fetcher?: { enriched: number; total: number };
  cache_hit?: boolean;
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
  // P1 检索诊断
  diagnostics?: RetrievalDiagnostics;
}

export interface ChatRequest {
  session_id?: string;
  user_id?: string;
  canvas_id?: string;
  message: string;
  collection?: string;
  search_mode: 'local' | 'web' | 'hybrid' | 'none';
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number }>;  // 每个搜索源的独立配置
  use_query_optimizer?: boolean;  // 是否启用查询优化器
  query_optimizer_max_queries?: number; // 每个搜索引擎最多生成的查询数
  use_query_expansion?: boolean;  // 兼容字段（已弃用）
  local_top_k?: number;  // 本地检索返回的最大文档数
  local_threshold?: number;  // 本地检索的相似度阈值 (0-1)
  final_top_k?: number;  // 最终保留的文档数（local + web 合并重排后）
  llm_provider?: string;  // LLM 提供商: deepseek | openai | gemini | claude | kimi 等
  model_override?: string;  // 覆盖默认模型，如 claude-opus-4-6
  mode?: ChatMode;  // 执行模式: chat（默认）| deep_research
  use_content_fetcher?: boolean;  // 是否对网络搜索结果做全文抓取（None 用后端默认）
  use_agent?: boolean;  // 是否启用 Agent 模式（ReAct 循环 / LangGraph 引擎）
  clarification_answers?: Record<string, string>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
}

export interface ChatCitation {
  cite_key: string;
  title: string;
  authors: string[];
  year?: number | null;
  doc_id?: string | null;
  url?: string | null;
  doi?: string | null;
}

export interface ChatResponse {
  session_id: string;
  response: string;
  citations: ChatCitation[];
  evidence_summary?: EvidenceSummary;
}

export interface SessionInfo {
  session_id: string;
  canvas_id: string;
  stage: string;
  turn_count: number;
  turns: { role: string; content: string; sources?: ChatCitation[] }[];
  research_dashboard?: ResearchDashboardData | null;
}

export interface SessionListItem {
  session_id: string;
  title: string;
  canvas_id: string;
  stage: string;
  turn_count: number;
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
  suggested_topic: string;
  suggested_outline: string[];
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
  local_top_k?: number;
  local_threshold?: number;
  final_top_k?: number;
  llm_provider?: string;
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
  search_mode: 'local' | 'web' | 'hybrid';
  clarification_answers?: Record<string, string>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
  step_model_strict?: boolean;
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number }>;
  use_query_optimizer?: boolean;
  query_optimizer_max_queries?: number;
  local_top_k?: number;
  local_threshold?: number;
  final_top_k?: number;
  llm_provider?: string;
  model_override?: string;
}

export interface DeepResearchStartResponse {
  session_id: string;
  canvas_id: string;
  brief: Record<string, unknown>;
  outline: string[];
  initial_stats: Record<string, unknown>;
  used_fallback?: boolean;
  fallback_reason?: string;
  llm_provider_used?: string;
  llm_model_used?: string;
}

export interface DeepResearchConfirmRequest {
  topic: string;
  session_id?: string;
  canvas_id?: string;
  user_id?: string;
  collection?: string;
  search_mode: 'local' | 'web' | 'hybrid';
  confirmed_outline: string[];
  confirmed_brief?: Record<string, unknown>;
  output_language?: 'auto' | 'en' | 'zh';
  step_models?: Record<string, string | null | undefined>;
  step_model_strict?: boolean;
  web_providers?: string[];
  web_source_configs?: Record<string, { topK: number; threshold: number }>;
  use_query_optimizer?: boolean;
  query_optimizer_max_queries?: number;
  local_top_k?: number;
  local_threshold?: number;
  final_top_k?: number;
  llm_provider?: string;
  model_override?: string;
  user_context?: string;
  user_context_mode?: 'supporting' | 'direct_injection';
  user_documents?: Array<{ name: string; content: string }>;
  // 研究深度
  depth?: 'lite' | 'comprehensive';
  // 阶段跳过控制
  skip_draft_review?: boolean;
  skip_refine_review?: boolean;
}

export interface DeepResearchSubmitResponse {
  ok: boolean;
  job_id: string;
  session_id: string;
  canvas_id: string;
}

export interface DeepResearchJobInfo {
  job_id: string;
  topic: string;
  session_id: string;
  canvas_id: string;
  status: 'pending' | 'running' | 'cancelling' | 'done' | 'error' | 'cancelled' | string;
  current_stage: string;
  message: string;
  error_message: string;
  result_markdown: string;
  result_citations: ChatCitation[];
  result_dashboard: Record<string, unknown>;
  total_time_ms: number;
  created_at: number;
  updated_at: number;
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
}

export interface RagConfig {
  enabled: boolean;  // 是否启用本地 RAG 检索
  localTopK: number;
  localThreshold: number;  // 相似度阈值 (0-1)
  finalTopK: number;  // 最终保留的文档数（local + web 合并重排后）
  enableHippoRAG: boolean;
  enableReranker: boolean;
  enableAgent: boolean;  // 是否启用 Agent 模式（ReAct / LangGraph）
}

export interface WebSearchConfig {
  enabled: boolean;
  sources: WebSource[];
  queryOptimizer: boolean;   // 查询优化器（针对不同搜索引擎优化查询格式）
  maxQueriesPerProvider: number; // 每个搜索引擎每种语言的查询数
  enableContentFetcher: boolean;  // 是否对网络搜索结果做全文抓取
}

// ============================================================
// Deep Research Defaults (persistent settings via ⚙ popover)
// ============================================================

export interface DeepResearchDefaults {
  depth: 'lite' | 'comprehensive';
  outputLanguage: 'auto' | 'en' | 'zh';
  stepModelStrict: boolean;
  stepModels: Record<string, string>;
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
