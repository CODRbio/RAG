export const DEEP_RESEARCH_JOB_KEY = 'deep_research_active_job_id';
export const DEEP_RESEARCH_PENDING_CONTEXT_KEY = 'deep_research_pending_user_context';
export const DEEP_RESEARCH_ARCHIVED_JOBS_KEY = 'deep_research_archived_job_ids';

export type ProgressPayload = {
  type?: string;
  section?: string;
  message?: string;
  [key: string]: unknown;
};

export type BriefDraft = Record<string, unknown>;

export type InitialStats = {
  total_sources?: number;
  total_iterations?: number;
  [key: string]: unknown;
};

export type ResearchMonitorState = {
  graphSteps: number;
  warnSteps: number | null;
  forceSteps: number | null;
  lastNode: string;
  costState: 'normal' | 'warn' | 'force';
  selfCorrectionCount: number;
  plateauEarlyStopCount: number;
  verificationContextCount: number;
  sectionCoverage: Record<string, number[]>;
  sectionSteps: Record<string, number[]>;
};

export const createEmptyMonitor = (): ResearchMonitorState => ({
  graphSteps: 0,
  warnSteps: null,
  forceSteps: null,
  lastNode: '',
  costState: 'normal',
  selfCorrectionCount: 0,
  plateauEarlyStopCount: 0,
  verificationContextCount: 0,
  sectionCoverage: {},
  sectionSteps: {},
});

export type EfficiencyRow = {
  section: string;
  firstCoverage: number;
  lastCoverage: number;
  rounds: number;
  avgDelta: number;
  lastDelta: number;
  per10Steps: number | null;
  score: number;
  level: 'high' | 'medium' | 'low';
};
