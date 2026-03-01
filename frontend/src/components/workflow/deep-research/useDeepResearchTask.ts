import { useState, useEffect, useRef } from 'react';
import { useChatStore, useConfigStore, useToastStore, useCanvasStore, useUIStore } from '../../../stores';
import {
  clarifyForDeepResearch,
  deepResearchStart,
  getDeepResearchStartStatus,
  deepResearchSubmit,
  restartDeepResearchPhase,
  restartDeepResearchSection,
  getDeepResearchJob,
  streamDeepResearchEvents,
  cancelDeepResearchJob,
} from '../../../api/chat';
import { exportCanvas, getCanvas } from '../../../api/canvas';
import type {
  DeepResearchConfirmRequest,
  DeepResearchStartRequest,
  DeepResearchJobEvent,
  Source,
  ResearchDashboardData,
  ChatCitation,
} from '../../../types';
import {
  DEEP_RESEARCH_JOB_KEY,
  DEEP_RESEARCH_ARCHIVED_JOBS_KEY,
  type ProgressPayload,
  type BriefDraft,
  type InitialStats,
  type ResearchMonitorState,
  createEmptyMonitor,
} from './types';

export type Phase = 'clarify' | 'confirm' | 'running';

export type GeneratePlanParams = {
  topic: string;
  answers: Record<string, string>;
  outputLanguage: 'auto' | 'en' | 'zh';
  stepModels: Record<string, string>;
  stepModelStrict: boolean;
  maxSections: number;
};

export type ConfirmAndRunParams = GeneratePlanParams & {
  outlineDraft: string[];
  briefDraft: BriefDraft | null;
  userContext: string;
  userContextMode: 'supporting' | 'direct_injection';
  tempDocuments: Array<{ name: string; content: string }>;
  depth: 'lite' | 'comprehensive';
  skipDraftReview: boolean;
  skipRefineReview: boolean;
  skipClaimGeneration: boolean;
  keepPreviousJobId: boolean;
};

export type StalledJobInfo = {
  jobId: string;
  topic: string;
  status: string;
  canvas_id: string;
};

export type UseDeepResearchTaskReturn = {
  phase: Phase;
  setPhase: (p: Phase) => void;
  activeJobId: string | null;
  planningJobId: string | null;
  stalledJob: StalledJobInfo | null;
  isStopping: boolean;
  isClarifying: boolean;
  progressLogs: string[];
  researchMonitor: ResearchMonitorState;
  startPhaseProgress: { stage: string; percent: number };
  outlineDraft: string[];
  setOutlineDraft: React.Dispatch<React.SetStateAction<string[]>>;
  briefDraft: BriefDraft | null;
  initialStats: InitialStats | null;
  optimizationPromptDraft: string;
  setOptimizationPromptDraft: React.Dispatch<React.SetStateAction<string>>;
  runClarify: (topic: string) => Promise<void>;
  generatePlan: (params: GeneratePlanParams) => Promise<void>;
  confirmAndRun: (params: ConfirmAndRunParams) => Promise<void>;
  stopJob: () => Promise<void>;
  restartFromPhase: (
    phase: 'plan' | 'research' | 'generate_claims' | 'write' | 'verify' | 'review_gate' | 'synthesize',
    sourceJobId?: string,
  ) => Promise<void>;
  restartSection: (
    sectionTitle: string,
    action: 'research' | 'write',
    sourceJobId?: string,
  ) => Promise<void>;
  handleInsertOptimizationPrompt: (text: string, setUserContext: React.Dispatch<React.SetStateAction<string>>) => void;
  clearStalledJob: () => void;
  openCanvasForCurrentJob: () => Promise<void>;
};

export function useDeepResearchTask(): UseDeepResearchTaskReturn {
  const {
    showDeepResearchDialog,
    setShowDeepResearchDialog,
    setDeepResearchTopic,
    setClarificationQuestions,
    sessionId,
    canvasId,
    addMessage,
    updateLastMessage,
    appendToLastMessage,
    setLastMessageSources,
    setSessionId,
    setCanvasId,
    setWorkflowStep,
    setIsStreaming,
    setDeepResearchActive,
  } = useChatStore();
  const setResearchDashboard = useChatStore((s) => s.setResearchDashboard);
  const {
    webSearchConfig,
    ragConfig,
    selectedProvider,
    selectedModel,
    currentCollection,
    deepResearchDefaults,
  } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const { setCanvas, setCanvasContent, setIsLoading: setCanvasLoading, setActiveStage } = useCanvasStore();
  const { setCanvasOpen, requestSessionListRefresh } = useUIStore();

  const [phase, setPhase] = useState<Phase>('clarify');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [stalledJob, setStalledJob] = useState<StalledJobInfo | null>(null);
  const [isStopping, setIsStopping] = useState(false);
  const [isClarifying, setIsClarifying] = useState(false);
  const [progressLogs, setProgressLogs] = useState<string[]>([]);
  const [researchMonitor, setResearchMonitor] = useState<ResearchMonitorState>(createEmptyMonitor);
  const [outlineDraft, setOutlineDraft] = useState<string[]>([]);
  const [briefDraft, setBriefDraft] = useState<BriefDraft | null>(null);
  const [initialStats, setInitialStats] = useState<InitialStats | null>(null);
  const [optimizationPromptDraft, setOptimizationPromptDraft] = useState('');
  const [startPhaseProgress, setStartPhaseProgress] = useState<{ stage: string; percent: number }>({
    stage: '正在启动...',
    percent: 0,
  });
  const [planningJobId, setPlanningJobId] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const lastEventIdRef = useRef(0);
  const canvasRefreshCounterRef = useRef(0);
  const latestJobIdRef = useRef<string | null>(null);

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const mapCitationsToSources = (citations: ChatCitation[]): Source[] => (
    citations.map((cite, idx: number) => ({
      id: idx + 1,
      cite_key: cite.cite_key,
      title: cite.title || cite.cite_key,
      authors: cite.authors || [],
      year: cite.year,
      doc_id: cite.doc_id,
      url: cite.url,
      doi: cite.doi,
      type: cite.url ? 'web' : 'local',
    }))
  );

  const archiveJobId = (jobId: string) => {
    try {
      const raw = localStorage.getItem(DEEP_RESEARCH_ARCHIVED_JOBS_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      const ids: string[] = Array.isArray(parsed) ? parsed.filter((x) => typeof x === 'string') : [];
      const next = [jobId, ...ids.filter((id) => id !== jobId)].slice(0, 20);
      localStorage.setItem(DEEP_RESEARCH_ARCHIVED_JOBS_KEY, JSON.stringify(next));
    } catch (err) {
      console.debug('[DeepResearch] archive job id failed:', err);
    }
  };

  const appendProgressEvents = (events: DeepResearchJobEvent[]) => {
    if (!events.length) return;
    setResearchMonitor((prev) => {
      const next: ResearchMonitorState = {
        ...prev,
        sectionCoverage: { ...prev.sectionCoverage },
        sectionSteps: { ...prev.sectionSteps },
      };
      const toNum = (v: unknown): number | null => {
        if (v === null || v === undefined || v === '') return null;
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
      };
      events.forEach((evt) => {
        const payload = (evt.data || {}) as ProgressPayload;
        if (evt.event !== 'progress') return;
        const pType = String(payload?.type || 'progress');
        if (pType === 'section_evaluate_done') {
          const section = String(payload?.section || '').trim();
          const cov = toNum(payload?.coverage);
          const graphSteps = toNum(payload?.graph_steps);
          if (section && cov !== null) {
            const hist = [...(next.sectionCoverage[section] || []), cov];
            next.sectionCoverage[section] = hist.slice(-8);
            if (graphSteps !== null) {
              const stepHist = [...(next.sectionSteps[section] || []), graphSteps];
              next.sectionSteps[section] = stepHist.slice(-8);
            }
          }
        } else if (pType === 'search_self_correction') {
          next.selfCorrectionCount += 1;
        } else if (pType === 'coverage_plateau_early_stop') {
          next.plateauEarlyStopCount += 1;
        } else if (pType === 'write_verification_context') {
          next.verificationContextCount += 1;
        } else if (pType === 'cost_monitor_tick' || pType === 'cost_monitor_warn' || pType === 'cost_monitor_force_summary') {
          const steps = toNum(payload?.steps);
          const warn = toNum(payload?.warn_steps);
          const force = toNum(payload?.force_steps);
          const node = String(payload?.node || '').trim();
          if (steps !== null) next.graphSteps = Math.max(next.graphSteps, steps);
          if (warn !== null) next.warnSteps = warn;
          if (force !== null) next.forceSteps = force;
          if (node) next.lastNode = node;
          if (pType === 'cost_monitor_warn' && next.costState === 'normal') next.costState = 'warn';
          if (pType === 'cost_monitor_force_summary') next.costState = 'force';
        }
      });
      return next;
    });

    setProgressLogs((prev) => {
      const next = [...prev];
      events.forEach((evt) => {
        const payload = (evt.data || {}) as ProgressPayload;
        if (evt.event === 'progress') {
          const pType = String(payload?.type || 'progress');
          let line = payload?.section
            ? `[${pType}] ${payload.section}`
            : `[${pType}] ${JSON.stringify(payload)}`;
          if (pType === 'section_evaluate_done') {
            const section = String(payload?.section || '');
            const parseNum = (v: unknown): number | null => {
              if (v === null || v === undefined || v === '') return null;
              const n = Number(v);
              return Number.isFinite(n) ? n : null;
            };
            const coverage = parseNum(payload?.coverage);
            const gain = parseNum(payload?.coverage_gain);
            const round = parseNum(payload?.research_round);
            const coverageText = coverage !== null ? coverage.toFixed(3) : '?';
            const gainText = gain !== null ? gain.toFixed(3) : '?';
            line = `[section_evaluate_done] ${section || 'section'}: coverage=${coverageText}, gain=${gainText}, round=${round !== null ? Math.trunc(round) : '?'}`;
          } else if (pType === 'evidence_insufficient') {
            line = payload?.section
              ? `[evidence_insufficient] ${payload.section}: ${payload.message || 'Evidence remains insufficient after fallback search.'}`
              : `[evidence_insufficient] ${payload.message || JSON.stringify(payload)}`;
            addToast(payload?.message ? String(payload.message) : 'Evidence insufficient for current section', 'warning');
          } else if (pType === 'section_degraded') {
            line = payload?.section
              ? `[section_degraded] ${payload.section}: ${payload.message || 'Section downgraded due to sparse evidence.'}`
              : `[section_degraded] ${payload.message || JSON.stringify(payload)}`;
            addToast(payload?.message ? String(payload.message) : 'Section downgraded due to sparse evidence', 'info');
          } else if (pType === 'search_self_correction') {
            line = payload?.section
              ? `[search_self_correction] ${payload.section}: top_k ${String(payload?.top_k_from || '?')} -> ${String(payload?.top_k_to || '?')}`
              : `[search_self_correction] ${payload?.message || JSON.stringify(payload)}`;
          } else if (pType === 'coverage_plateau_early_stop') {
            line = payload?.section
              ? `[coverage_plateau_early_stop] ${payload.section}: coverage ${String(payload?.coverage || '?')}`
              : `[coverage_plateau_early_stop] ${payload?.message || JSON.stringify(payload)}`;
            addToast(payload?.message ? String(payload.message) : 'Coverage gain flattened, early stop applied', 'info');
          } else if (pType === 'write_verification_context') {
            line = payload?.section
              ? `[write_verification_context] ${payload.section}: write_k=${String(payload?.write_top_k || '?')}, verify_k=${String(payload?.verification_k || '?')}`
              : `[write_verification_context] ${payload?.message || JSON.stringify(payload)}`;
          } else if (pType === 'cost_monitor_tick') {
            line = `[cost_monitor_tick] steps=${String(payload?.steps || '?')} (${String(payload?.node || 'node')})`;
          } else if (pType === 'step_model_fallback') {
            line = payload?.step
              ? `[step_model_fallback] ${payload.step}: ${payload.message || 'Fallback to default model'}`
              : `[step_model_fallback] ${payload.message || JSON.stringify(payload)}`;
            addToast(payload?.message ? String(payload.message) : 'Step model fallback occurred', 'warning');
          } else if (pType === 'step_model_resolved') {
            const step = String(payload?.step || '');
            const provider = String(payload?.actual_provider || 'default');
            const model = String(payload?.actual_model || '');
            line = `[step_model_resolved] ${step || 'step'} -> ${provider}${model ? `::${model}` : ''}`;
          } else if (pType === 'all_reviews_approved') {
            line = `[all_reviews_approved] approved ${String(payload?.approved || 0)}/${String(payload?.total || 0)}; entering synthesize`;
            addToast('所有章节已通过，开始最终整合...', 'success');
          } else if (pType === 'global_refine_done') {
            line = `[global_refine_done] ${payload?.message || 'Global coherence refinement completed.'}`;
            addToast('全文连贯性整合完成', 'success');
          } else if (pType === 'citation_guard_fallback') {
            line = `[citation_guard_fallback] ${payload?.message || 'Citation guard fallback to pre-refine version.'}`;
            addToast('检测到引用丢失风险，已自动回退到安全版本', 'warning');
          } else if (pType === 'cost_monitor_warn') {
            line = `[cost_monitor_warn] ${payload?.message || 'Cost monitor warning triggered.'}`;
            addToast(payload?.message ? String(payload.message) : '研究步数较高，建议人工介入', 'warning');
          } else if (pType === 'cost_monitor_force_summary') {
            line = `[cost_monitor_force_summary] ${payload?.message || 'Forced summary mode activated.'}`;
            addToast('已触发强制摘要模式以控制成本', 'warning');
          }
          next.push(line);
        } else if (evt.event === 'warning') {
          const line = payload?.message
            ? `[warning] ${payload.message}`
            : `[warning] ${JSON.stringify(payload)}`;
          next.push(line);
        } else if (evt.event === 'cancel_requested') {
          next.push('[info] 收到停止请求，任务正在终止...');
        }
      });
      return next.slice(-300);
    });
  };

  const finalizeRunningJob = async (jobId: string) => {
    latestJobIdRef.current = jobId;
    let wasCancelled = false;
    try {
      const job = await getDeepResearchJob(jobId);
      if (job.session_id) setSessionId(job.session_id);
      if (job.canvas_id) setCanvasId(job.canvas_id);
      if (job.result_dashboard && Object.keys(job.result_dashboard).length > 0) {
        setResearchDashboard(job.result_dashboard as unknown as ResearchDashboardData);
      }
      const citations = (job.result_citations || []) as ChatCitation[];
      if (citations.length > 0) {
        setLastMessageSources(mapCitationsToSources(citations));
      }
      if (job.result_markdown) {
        updateLastMessage(job.result_markdown);
        setCanvasContent(job.result_markdown);
      }
      if (job.canvas_id) {
        setCanvasLoading(true);
        Promise.all([
          getCanvas(job.canvas_id).catch((err) => {
            console.error('[DeepResearch] Canvas data load failed:', err);
            return null;
          }),
          exportCanvas(job.canvas_id, 'markdown').catch((err) => {
            console.error('[DeepResearch] Canvas markdown load failed:', err);
            return null;
          }),
        ])
          .then(([canvasData, exportResp]) => {
            if (canvasData) setCanvas(canvasData);
            if (exportResp?.content && !job.result_markdown) {
              setCanvasContent(exportResp.content);
            }
            if (canvasData || exportResp?.content || job.result_markdown) setCanvasOpen(true);
          })
          .finally(() => setCanvasLoading(false));
      }
      if (job.status === 'done') {
        setActiveStage('refine');
        setWorkflowStep('refine');
        addToast('Deep Research 完成', 'success');
        setShowDeepResearchDialog(false);
        setTimeout(() => setWorkflowStep('idle'), 1000);
      } else if (job.status === 'cancelled') {
        // Preserve the stopped task so the user can go to canvas or restart,
        // rather than resetting to the confirm/clarify phase.
        wasCancelled = true;
        addToast('Deep Research 已停止，可进入画布查看或重启', 'info');
        setStalledJob({
          jobId,
          topic: String(job.topic || ''),
          status: 'cancelled',
          canvas_id: String(job.canvas_id || ''),
        });
      } else {
        appendToLastMessage(`\n\n[错误] Deep Research 失败：${job.error_message || job.message || 'unknown error'}`);
        addToast('Deep Research 失败，请重试', 'error');
      }
    } finally {
      stopStreaming();
      const finishedJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
      if (finishedJobId) {
        archiveJobId(finishedJobId);
      }
      localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
      setActiveJobId(null);
      setIsStopping(false);
      setIsStreaming(false);
      setDeepResearchActive(false);
      // For cancelled jobs keep the dialog open showing the stopped state (stalledJob banner)
      // so the user can inspect progress or restart. For done/error, go back to confirm.
      if (!wasCancelled) {
        setPhase('confirm');
      }
    }
  };

  const startStreamingJob = (jobId: string, resetEvents: boolean) => {
    stopStreaming();
    if (resetEvents) {
      lastEventIdRef.current = 0;
      canvasRefreshCounterRef.current = 0;
      setProgressLogs([]);
      setResearchMonitor(createEmptyMonitor());
      setOptimizationPromptDraft('');
    }
    setActiveJobId(jobId);
    latestJobIdRef.current = jobId;
    localStorage.setItem(DEEP_RESEARCH_JOB_KEY, jobId);

    const ac = new AbortController();
    abortControllerRef.current = ac;

    (async () => {
      let retryDelay = 1000;
      while (!ac.signal.aborted) {
        try {
          for await (const { event, data } of streamDeepResearchEvents(jobId, ac.signal, lastEventIdRef.current)) {
            if (ac.signal.aborted) break;

            if (event === 'heartbeat' || event === 'job_status') {
              // Update research dashboard if available
              const dashboard = data.result_dashboard as Record<string, unknown> | undefined;
              if (dashboard && Object.keys(dashboard).length > 0) {
                setResearchDashboard(dashboard as unknown as ResearchDashboardData);
              }
              // Update status message log
              const message = typeof data.message === 'string' ? data.message : '';
              if (message) {
                setProgressLogs((prev) =>
                  prev[prev.length - 1] === `[status] ${message}`
                    ? prev
                    : [...prev, `[status] ${message}`].slice(-300),
                );
              }
              if (event === 'heartbeat') {
                // Periodic canvas refresh on every other heartbeat (~10s cadence)
                canvasRefreshCounterRef.current += 1;
                if (canvasRefreshCounterRef.current % 2 === 0) {
                  const runningCanvasId =
                    (typeof data.canvas_id === 'string' ? data.canvas_id : '') ||
                    useChatStore.getState().canvasId ||
                    '';
                  const status = typeof data.status === 'string' ? data.status : '';
                  if (runningCanvasId) {
                    getCanvas(runningCanvasId)
                      .then((canvasData) => {
                        if (!canvasData) return;
                        setCanvas(canvasData);
                        if (
                          (status === 'running' || status === 'cancelling') &&
                          (canvasData.stage === 'drafting' || canvasData.stage === 'refine')
                        ) {
                          setCanvasOpen(true);
                        }
                      })
                      .catch((err) => {
                        console.debug('[DeepResearch] periodic canvas refresh failed:', err);
                      });
                  }
                }
              }
              if (event === 'job_status') {
                const status = typeof data.status === 'string' ? data.status : '';
                if (status === 'done' || status === 'cancelled' || status === 'error') {
                  await finalizeRunningJob(jobId);
                  return;
                }
              }
            } else if (event === 'error') {
              console.error('[DeepResearch] Backend emitted error:', data);
              if (String(data.message).includes('不存在')) {
                await finalizeRunningJob(jobId);
                return;
              }
            } else {
              // Regular DB-backed job event — track event_id for reconnect, then process
              const eid = typeof data._event_id === 'number' ? data._event_id : null;
              if (eid !== null) {
                lastEventIdRef.current = Math.max(lastEventIdRef.current, eid);
              }
              appendProgressEvents([
                {
                  event_id: eid ?? lastEventIdRef.current,
                  event,
                  created_at: typeof data.created_at === 'number' ? data.created_at : Date.now() / 1000,
                  data,
                },
              ]);
            }
          }
          // If we reach here without returning, the stream was closed by the proxy or network (e.g. proxy timeout).
          // We must NOT break the outer retry loop! Otherwise it stays stuck forever.
          if (ac.signal.aborted) break;
          console.warn('[DeepResearch] SSE stream closed before job completion, reconnecting in 2s...');
          await new Promise<void>((resolve) => setTimeout(resolve, 2000));
        } catch (err) {
          if (ac.signal.aborted) break;
          console.error('[DeepResearch] SSE stream error, retrying in', retryDelay, 'ms:', err);
          await new Promise<void>((resolve) => setTimeout(resolve, retryDelay));
          retryDelay = Math.min(retryDelay * 2, 10000);
        }
      }
    })();
  };

  // Restore job state when dialog opens; set stalledJob when job exists but is not runnable (半途死了)
  useEffect(() => {
    if (!showDeepResearchDialog) return;
    const savedJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (!savedJobId || activeJobId) return;

    let cancelled = false;
    (async () => {
      try {
        const job = await getDeepResearchJob(savedJobId);
        if (cancelled) return;
        const isRunnable = job.status === 'planning'
          || job.status === 'pending'
          || job.status === 'running'
          || job.status === 'cancelling'
          || job.status === 'waiting_review';
        const sameSession = !sessionId || !job.session_id || job.session_id === sessionId;
        const requestedTopic = (useChatStore.getState().deepResearchTopic || '').trim();
        const runningTopic = String(job.topic || '').trim();
        const sameTopic = !requestedTopic || !runningTopic || requestedTopic === runningTopic;
        if (!isRunnable) {
          setStalledJob({
            jobId: savedJobId,
            topic: String(job.topic || ''),
            status: job.status,
            canvas_id: String(job.canvas_id || ''),
          });
          archiveJobId(savedJobId);
          localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
          return;
        }
        if (!sameSession) return;
        if (!sameTopic) {
          addToast('检测到其他进行中的 Deep Research 任务；当前按新主题进入大纲确认流程。', 'info');
          return;
        }
        setPhase('running');
        setIsStreaming(true);
        setDeepResearchActive(true);
        startStreamingJob(savedJobId, false);
        addToast('已恢复 Deep Research 后台任务状态', 'info');
      } catch (err) {
        console.debug('[DeepResearch] skip stale saved job:', err);
        setStalledJob({
          jobId: savedJobId,
          topic: (useChatStore.getState().deepResearchTopic || '').trim(),
          status: 'error',
          canvas_id: '',
        });
        archiveJobId(savedJobId);
        localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDeepResearchDialog, activeJobId, sessionId]);

  // Cleanup SSE stream on unmount
  useEffect(() => () => {
    stopStreaming();
  }, []);

  const buildCommonRequestParams = () => {
    const enabledProviders = webSearchConfig.sources
      .filter((s) => s.enabled)
      .map((s) => s.id);

    const webSourceConfigs: Record<string, { topK: number; threshold: number; useSerpapi?: boolean }> = {};
    webSearchConfig.sources.forEach((source) => {
      if (source.enabled) {
        const cfg: { topK: number; threshold: number; useSerpapi?: boolean } = { topK: source.topK, threshold: source.threshold };
        if (source.useSerpapi) cfg.useSerpapi = true;
        webSourceConfigs[source.id] = cfg;
      }
    });
    const hasAnySerpapi = webSearchConfig.sources.some((s) => s.enabled && s.useSerpapi);

    const localEnabled = ragConfig.enabled ?? true;
    const webEnabled = webSearchConfig.enabled && enabledProviders.length > 0;

    let searchMode: 'local' | 'web' | 'hybrid';
    if (localEnabled && webEnabled) {
      searchMode = 'hybrid';
    } else if (webEnabled) {
      searchMode = 'web';
    } else {
      searchMode = 'local';
    }

    const queryOptimizerEnabled = Boolean(webSearchConfig.queryOptimizer ?? true);
    const maxQueries = Math.min(5, Math.max(1, Number(webSearchConfig.maxQueriesPerProvider ?? 3)));
    const deepStepTopK = Math.max(ragConfig.stepTopK ?? 10, 40);
    const deepWriteTopK = Math.max(
      ragConfig.writeTopK ?? 15,
      Math.ceil((ragConfig.stepTopK ?? 10) * 1.5),
    );
    const gapQueryIntent = (deepResearchDefaults.gapQueryIntent || 'broad') as 'broad' | 'review_pref' | 'reviews_only';

    const serpapiRatio = hasAnySerpapi ? (webSearchConfig.serpapiRatio ?? 50) / 100 : undefined;

    return {
      searchMode,
      enabledProviders,
      webSourceConfigs,
      webEnabled,
      localEnabled,
      queryOptimizerEnabled,
      maxQueries,
      deepStepTopK,
      deepWriteTopK,
      gapQueryIntent,
      serpapiRatio,
    };
  };

  const normalizeStepModels = (stepModels: Record<string, string>) => {
    const out: Record<string, string | null> = {};
    Object.entries(stepModels).forEach(([k, v]) => {
      out[k] = (v || '').trim() || null;
    });
    return out;
  };

  const runClarify = async (topic: string) => {
    const trimTopic = (topic || '').trim();
    if (!trimTopic) return;
    
    const { deepResearchDefaults } = useConfigStore.getState();
    const prelimModel = (deepResearchDefaults.preliminaryModel ?? '').trim();
    const questModel = (deepResearchDefaults.questionModel ?? '').trim();
    
    let resolvedProvider: string | undefined = selectedProvider || undefined;
    let resolvedModel: string | undefined = selectedModel || undefined;
    if (questModel) {
      const idx = questModel.indexOf('::');
      if (idx > 0) {
        resolvedProvider = questModel.slice(0, idx).trim() || undefined;
        resolvedModel = questModel.slice(idx + 2).trim() || undefined;
      }
    }
    
    let prelimProvider: string | undefined = undefined;
    let prelimModelOverride: string | undefined = undefined;
    if (prelimModel) {
      const idx = prelimModel.indexOf('::');
      if (idx > 0) {
        prelimProvider = prelimModel.slice(0, idx).trim() || undefined;
        prelimModelOverride = prelimModel.slice(idx + 2).trim() || undefined;
      }
    }

    setIsClarifying(true);
    try {
      const result = await clarifyForDeepResearch({
        message: trimTopic,
        session_id: useChatStore.getState().sessionId || undefined,
        search_mode: 'hybrid',
        llm_provider: resolvedProvider,
        model_override: resolvedModel,
        prelim_provider: prelimProvider,
        prelim_model: prelimModelOverride,
      });
      const qs = result.questions || [];
      if (result.session_id) setSessionId(result.session_id);
      requestSessionListRefresh();
      setClarificationQuestions(qs);
      const suggestedTopic = (result.suggested_topic || '').trim();
      if (suggestedTopic) {
        setDeepResearchTopic(suggestedTopic);
      }
      if (result.used_fallback) {
        addToast(`澄清阶段触发回退：${result.fallback_reason || 'unknown'}`, 'warning');
      } else {
        addToast(
          `澄清模型：${result.llm_provider_used || resolvedProvider || 'default'}`
            + `${result.llm_model_used ? `::${result.llm_model_used}` : ''}`,
          'info',
        );
      }
    } catch (err) {
      console.error('[DeepResearchDialog] Clarify failed:', err);
      setClarificationQuestions([
        {
          id: 'q1',
          text: '请确认本次研究最关键的目标与范围边界',
          question_type: 'text',
          options: [],
          default: trimTopic,
        },
      ]);
      addToast('澄清问题生成失败，已回退到最小问题集', 'warning');
    } finally {
      setIsClarifying(false);
    }
  };

  const generatePlan = async (params: GeneratePlanParams) => {
    const { topic, answers, outputLanguage, stepModels, stepModelStrict, maxSections } = params;
    if (!topic.trim()) {
      addToast('请输入研究主题', 'error');
      return;
    }
    const {
      searchMode,
      enabledProviders,
      webSourceConfigs,
      webEnabled,
      localEnabled,
      queryOptimizerEnabled,
      maxQueries,
      deepStepTopK,
      deepWriteTopK,
      gapQueryIntent,
      serpapiRatio,
    } = buildCommonRequestParams();
    const hasNonEmptyAnswers = Object.values(answers).some((v) => v.trim().length > 0);
    const startRequest: DeepResearchStartRequest = {
      topic,
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      collection: currentCollection || undefined,
      search_mode: searchMode,
      max_sections: maxSections,
      llm_provider: selectedProvider || undefined,
      ultra_lite_provider: deepResearchDefaults.ultra_lite_provider || undefined,
      model_override: selectedModel || undefined,
      web_providers: webEnabled ? enabledProviders : undefined,
      web_source_configs: (webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      serpapi_ratio: (webEnabled && serpapiRatio !== undefined) ? serpapiRatio : undefined,
      use_query_optimizer: webEnabled ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: webEnabled ? maxQueries : undefined,
      use_content_fetcher: webEnabled ? webSearchConfig.contentFetcherMode : undefined,
      gap_query_intent: gapQueryIntent,
      local_top_k: localEnabled ? ragConfig.localTopK : undefined,
      local_threshold: localEnabled ? (ragConfig.localThreshold ?? undefined) : undefined,
      year_start: ragConfig.yearStart ?? undefined,
      year_end: ragConfig.yearEnd ?? undefined,
      step_top_k: deepStepTopK,
      write_top_k: deepWriteTopK,
      reranker_mode: ragConfig.enableReranker
        ? ((localStorage.getItem('adv_reranker_mode') || 'cascade') as 'bge_only' | 'colbert_only' | 'cascade')
        : 'bge_only',
      clarification_answers: hasNonEmptyAnswers ? answers : undefined,
      output_language: outputLanguage,
      step_models: normalizeStepModels(stepModels),
      step_model_strict: stepModelStrict,
    };
    setIsStreaming(true);
    setStartPhaseProgress({ stage: '正在启动...', percent: 0 });
    try {
      // 1. Submit the start job – returns immediately with a job_id.
      //    The backend now creates a DB entry with status='planning' so
      //    it appears in the sidebar immediately.
      const jobResp = await deepResearchStart(startRequest);
      if (jobResp.session_id) setSessionId(jobResp.session_id);
      const jobId = jobResp.job_id;
      setPlanningJobId(jobId);
      requestSessionListRefresh();
      console.log('[DeepResearch] Start job submitted (planning):', jobId);

      // 2. Poll until scope+plan finishes.
      //
      // Bidirectional timeout contract:
      //   - Backend: heartbeat writes DB updated_at every 15s; hard ceiling ~30min.
      //   - Frontend: waits 35min (slightly longer than backend) and checks DB
      //     liveness before declaring failure.  Individual poll errors are tolerated
      //     as long as the DB job is still alive (updated_at recent).
      //
      // The key invariant: we ONLY report failure when the DB job is confirmed
      // to be in error state, or the backend has been silent for > LIVENESS_GAP.
      const POLL_INTERVAL_MS = 4000;
      const MAX_WAIT_MS = 35 * 60 * 1000; // frontend waits 5min longer than backend's 30min ceiling
      const LIVENESS_GAP_MS = 90_000;      // if DB hasn't been touched for 90s, consider backend dead
      const deadline = Date.now() + MAX_WAIT_MS;
      let consecutivePollErrors = 0;
      let lastProgressLog = '';

      const logProgress = (msg: string) => {
        if (msg && msg !== lastProgressLog) {
          lastProgressLog = msg;
          setProgressLogs((prev) => [...prev, `[planning] ${msg}`].slice(-300));
        }
      };

      while (Date.now() < deadline) {
        await new Promise<void>((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));

        // ── Primary path: fast poll endpoint ──
        let status: Awaited<ReturnType<typeof getDeepResearchStartStatus>> | null = null;
        try {
          status = await getDeepResearchStartStatus(jobId);
          consecutivePollErrors = 0;
        } catch (pollErr) {
          consecutivePollErrors++;
          const httpStatus = (pollErr as { response?: { status?: number } })?.response?.status;
          console.warn(
            `[DeepResearch] Start poll error #${consecutivePollErrors}`,
            `(HTTP ${httpStatus ?? 'network'})`,
          );
          logProgress(`轮询异常 #${consecutivePollErrors}，正在检查后台状态...`);

          // ── Fallback path: check DB job liveness ──
          // The backend heartbeat keeps updated_at fresh every 15s.
          // If updated_at is still recent, the backend is alive — keep waiting.
          try {
            const dbJob = await getDeepResearchJob(jobId);
            const dbStatus = dbJob.status;
            if (dbStatus === 'error') {
              throw new Error(dbJob.error_message || dbJob.message || '研究规划生成失败');
            }
            if (dbStatus === 'cancelled') {
              throw new Error('任务已取消');
            }
            // Job still alive in DB ('planning' status)
            const updatedAt = dbJob.updated_at ?? 0;
            const ageMs = (Date.now() / 1000 - updatedAt) * 1000;
            if (ageMs < LIVENESS_GAP_MS) {
              // Backend is alive: show heartbeat info and keep waiting
              logProgress(dbJob.message || dbJob.current_stage || `后台仍在运行 (${Math.round(ageMs / 1000)}s)`);
              setStartPhaseProgress({
                stage: dbJob.message || dbJob.current_stage || '后台运行中...',
                percent: Math.min(90, Math.max(0, (status?.progress ?? 0))),
              });
              continue;
            }
            // DB hasn't been updated for too long — backend might have crashed
            if (consecutivePollErrors < 15) {
              logProgress(`后台心跳超时 ${Math.round(ageMs / 1000)}s，继续等待...`);
              continue;
            }
            // Too many errors AND stale heartbeat — give up
            throw new Error(
              `后台长时间无响应（心跳超时 ${Math.round(ageMs / 1000)}s），` +
              '请检查后端日志或重试',
            );
          } catch (dbErr) {
            // If the DB check itself also failed, only give up after many retries
            if ((dbErr as Error).message?.includes('规划生成失败') ||
                (dbErr as Error).message?.includes('已取消') ||
                (dbErr as Error).message?.includes('无响应')) {
              throw dbErr;
            }
            if (consecutivePollErrors >= 20) {
              throw new Error(
                '前后端连接全部中断，请检查网络和后端状态后重试',
              );
            }
            logProgress(`连接异常 #${consecutivePollErrors}，稍后重试...`);
            continue;
          }
        }

        // ── Normal poll response handling ──
        if (status.session_id) setSessionId(status.session_id);
        if (status.status === 'running') {
          const stageMsg = status.current_stage ?? '正在准备...';
          setStartPhaseProgress({
            stage: stageMsg,
            percent: Math.min(100, Math.max(0, status.progress ?? 0)),
          });
          logProgress(stageMsg);
        }

        if (status.status === 'done') {
          if (status.canvas_id) setCanvasId(status.canvas_id);
          setOutlineDraft(status.outline?.length ? status.outline : [topic]);
          setBriefDraft(status.brief || null);
          setInitialStats(status.initial_stats || null);
          setPhase('confirm');
          return;
        }

        if (status.status === 'error') {
          throw new Error(status.error || '研究规划生成失败');
        }
        // status === 'running': keep polling
      }

      throw new Error('研究规划生成超时（超过35分钟），请缩短主题或减少检索范围后重试');
    } catch (error) {
      console.error('[DeepResearch] Start error:', error);
      const msg = (error as Error)?.message || '研究规划生成失败';
      addToast(msg, 'error');
      setProgressLogs((prev) => [...prev, `[error] ${msg}`].slice(-300));
    } finally {
      setIsStreaming(false);
    }
  };

  const confirmAndRun = async (params: ConfirmAndRunParams) => {
    const {
      topic, outlineDraft: rawOutline, briefDraft: brief, outputLanguage,
      stepModels, stepModelStrict, maxSections, userContext, userContextMode, tempDocuments,
      depth, skipDraftReview, skipRefineReview, skipClaimGeneration, keepPreviousJobId,
    } = params;
    if (!topic.trim()) return;
    const filteredOutline = rawOutline.map((x) => x.trim()).filter(Boolean);
    if (!filteredOutline.length) {
      addToast('请至少保留一个大纲章节', 'error');
      return;
    }
    const {
      searchMode,
      enabledProviders,
      webSourceConfigs,
      webEnabled,
      localEnabled,
      queryOptimizerEnabled,
      maxQueries,
      deepStepTopK,
      deepWriteTopK,
      gapQueryIntent,
      serpapiRatio,
    } = buildCommonRequestParams();

    setDeepResearchActive(true);
    setPhase('running');
    setProgressLogs([]);
    addMessage({ role: 'user', content: `[Deep Research] ${topic}` });
    addMessage({ role: 'assistant', content: '' });
    setWorkflowStep('explore');

    const confirmRequest: DeepResearchConfirmRequest = {
      topic,
      planning_job_id: planningJobId || undefined,
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      collection: currentCollection || undefined,
      search_mode: searchMode,
      confirmed_outline: filteredOutline,
      confirmed_brief: brief || undefined,
      output_language: outputLanguage,
      step_models: normalizeStepModels(stepModels),
      step_model_strict: stepModelStrict,
      llm_provider: selectedProvider || undefined,
      ultra_lite_provider: deepResearchDefaults.ultra_lite_provider || undefined,
      model_override: selectedModel || undefined,
      web_providers: webEnabled ? enabledProviders : undefined,
      web_source_configs: (webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      serpapi_ratio: (webEnabled && serpapiRatio !== undefined) ? serpapiRatio : undefined,
      use_query_optimizer: webEnabled ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: webEnabled ? maxQueries : undefined,
      use_content_fetcher: webEnabled ? webSearchConfig.contentFetcherMode : undefined,
      gap_query_intent: gapQueryIntent,
      local_top_k: localEnabled ? ragConfig.localTopK : undefined,
      local_threshold: localEnabled ? (ragConfig.localThreshold ?? undefined) : undefined,
      year_start: ragConfig.yearStart ?? undefined,
      year_end: ragConfig.yearEnd ?? undefined,
      step_top_k: deepStepTopK,
      write_top_k: deepWriteTopK,
      reranker_mode: ragConfig.enableReranker
        ? ((localStorage.getItem('adv_reranker_mode') || 'cascade') as 'bge_only' | 'colbert_only' | 'cascade')
        : 'bge_only',
      user_context: userContext.trim() || undefined,
      user_context_mode: userContext.trim() ? userContextMode : undefined,
      user_documents: tempDocuments.length ? tempDocuments : undefined,
      depth,
      skip_draft_review: skipDraftReview,
      skip_refine_review: skipRefineReview,
      skip_claim_generation: skipClaimGeneration,
      max_sections: maxSections,
    };

    let submitted = false;
    const previousActiveJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    try {
      const submitResp = await deepResearchSubmit(confirmRequest);
      if (submitResp.session_id) setSessionId(submitResp.session_id);
      if (submitResp.canvas_id) setCanvasId(submitResp.canvas_id);
      if (keepPreviousJobId && previousActiveJobId && previousActiveJobId !== submitResp.job_id) {
        archiveJobId(previousActiveJobId);
      }
      setPlanningJobId(null);
      requestSessionListRefresh();
      setWorkflowStep('drafting');
      startStreamingJob(submitResp.job_id, true);
      setShowDeepResearchDialog(false);
      submitted = true;
      addToast('已转为后台执行，可安全关闭当前前端页面', 'info');
    } catch (error) {
      console.error('[DeepResearch] Error:', error);
      addToast('Deep Research 失败，请重试', 'error');
      appendToLastMessage('\n\n[错误] Deep Research 请求失败。');
      stopStreaming();
      const failedJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
      if (failedJobId) archiveJobId(failedJobId);
      localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
      setActiveJobId(null);
      setDeepResearchActive(false);
      setPhase('confirm');
    } finally {
      if (!submitted) {
        setIsStreaming(false);
      }
    }
  };

  const stopJob = async () => {
    if (!activeJobId || isStopping) return;
    setIsStopping(true);
    try {
      await cancelDeepResearchJob(activeJobId);
      addToast('已请求停止任务，正在终止...', 'info');
    } catch (err) {
      console.error('[DeepResearch] Cancel failed:', err);
      setIsStopping(false);
      addToast('停止任务失败，请重试', 'error');
    }
  };

  const resolveSourceJobId = (provided?: string): string | null => {
    if (provided && provided.trim()) return provided.trim();
    if (activeJobId) return activeJobId;
    if (latestJobIdRef.current) return latestJobIdRef.current;
    const saved = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (saved && saved.trim()) return saved.trim();
    try {
      const raw = localStorage.getItem(DEEP_RESEARCH_ARCHIVED_JOBS_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'string') {
        return parsed[0];
      }
    } catch {
      // noop
    }
    return null;
  };

  const restartFromPhase = async (
    phaseToRestart: 'plan' | 'research' | 'generate_claims' | 'write' | 'verify' | 'review_gate' | 'synthesize',
    sourceJobId?: string,
  ) => {
    const source = resolveSourceJobId(sourceJobId);
    if (!source) {
      addToast('未找到可重启的 Deep Research 任务', 'warning');
      return;
    }
    setDeepResearchActive(true);
    setPhase('running');
    setIsStreaming(true);
    try {
      const resp = await restartDeepResearchPhase(source, { phase: phaseToRestart });
      if (resp.session_id) setSessionId(resp.session_id);
      if (resp.canvas_id) setCanvasId(resp.canvas_id);
      startStreamingJob(resp.job_id, true);
      requestSessionListRefresh();
      addToast('已提交重启任务，进入后台执行', 'success');
    } catch (err) {
      console.error('[DeepResearch] restart phase failed:', err);
      setIsStreaming(false);
      setDeepResearchActive(false);
      setPhase('confirm');
      addToast('重启任务失败，请重试', 'error');
    }
  };

  const restartSection = async (
    sectionTitle: string,
    action: 'research' | 'write',
    sourceJobId?: string,
  ) => {
    const source = resolveSourceJobId(sourceJobId);
    if (!source) {
      addToast('未找到可重启的 Deep Research 任务', 'warning');
      return;
    }
    if (!sectionTitle.trim()) {
      addToast('章节标题不能为空', 'warning');
      return;
    }
    setDeepResearchActive(true);
    setPhase('running');
    setIsStreaming(true);
    try {
      const resp = await restartDeepResearchSection(source, {
        section_title: sectionTitle.trim(),
        action,
      });
      if (resp.session_id) setSessionId(resp.session_id);
      if (resp.canvas_id) setCanvasId(resp.canvas_id);
      startStreamingJob(resp.job_id, true);
      requestSessionListRefresh();
      addToast(`已提交章节重启：${sectionTitle}`, 'success');
    } catch (err) {
      console.error('[DeepResearch] restart section failed:', err);
      setIsStreaming(false);
      setDeepResearchActive(false);
      setPhase('confirm');
      addToast('章节重启失败，请重试', 'error');
    }
  };

  const handleInsertOptimizationPrompt = (
    text: string,
    setUserContext: React.Dispatch<React.SetStateAction<string>>,
  ) => {
    if (!text) return;
    setUserContext((prev) => {
      const current = (prev || '').trim();
      if (!current) return text;
      if (current.includes(text)) return current;
      return `${current}\n\n${text}`;
    });
    addToast('已写入 Intervention，可在下一轮直接使用', 'success');
  };

  const clearStalledJob = () => {
    setStalledJob(null);
  };

  const openCanvasForCurrentJob = async () => {
    const cid = canvasId || stalledJob?.canvas_id || '';
    if (!cid.trim()) {
      addToast('无画布可打开', 'warning');
      return;
    }
    // 让画布内各阶段能识别为「激活任务」：写入 localStorage
    const jobIdToActivate = activeJobId || stalledJob?.jobId || '';
    if (jobIdToActivate) {
      localStorage.setItem(DEEP_RESEARCH_JOB_KEY, jobIdToActivate);
    }
    if (stalledJob?.canvas_id) {
      setCanvasId(stalledJob.canvas_id);
    }
    setCanvasLoading(true);
    try {
      const [canvasData, exportResp] = await Promise.all([
        getCanvas(cid).catch(() => null),
        exportCanvas(cid, 'markdown').catch(() => null),
      ]);
      if (canvasData) {
        setCanvas(canvasData);
        setActiveStage(canvasData.stage || 'explore');
      }
      if (exportResp?.content) setCanvasContent(exportResp.content);
      setCanvasOpen(true);
    } catch (err) {
      console.error('[DeepResearch] openCanvasForCurrentJob failed:', err);
      addToast('打开画布失败', 'error');
    } finally {
      setCanvasLoading(false);
    }
  };

  return {
    phase,
    setPhase,
    activeJobId,
    planningJobId,
    stalledJob,
    isStopping,
    isClarifying,
    progressLogs,
    researchMonitor,
    startPhaseProgress,
    outlineDraft,
    setOutlineDraft,
    briefDraft,
    initialStats,
    optimizationPromptDraft,
    setOptimizationPromptDraft,
    runClarify,
    generatePlan,
    confirmAndRun,
    stopJob,
    restartFromPhase,
    restartSection,
    handleInsertOptimizationPrompt,
    clearStalledJob,
    openCanvasForCurrentJob,
  };
}
