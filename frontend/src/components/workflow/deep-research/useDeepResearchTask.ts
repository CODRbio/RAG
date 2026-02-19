import { useState, useEffect, useRef } from 'react';
import { useChatStore, useConfigStore, useToastStore, useCanvasStore, useUIStore } from '../../../stores';
import {
  clarifyForDeepResearch,
  deepResearchStart,
  deepResearchSubmit,
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
  yearStart: number | null;
  yearEnd: number | null;
  stepModels: Record<string, string>;
  stepModelStrict: boolean;
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

export type UseDeepResearchTaskReturn = {
  phase: Phase;
  setPhase: (p: Phase) => void;
  activeJobId: string | null;
  isStopping: boolean;
  isClarifying: boolean;
  progressLogs: string[];
  researchMonitor: ResearchMonitorState;
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
  handleInsertOptimizationPrompt: (text: string, setUserContext: React.Dispatch<React.SetStateAction<string>>) => void;
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
  } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const { setCanvas, setCanvasContent, setIsLoading: setCanvasLoading, setActiveStage } = useCanvasStore();
  const setCanvasOpen = useUIStore((s) => s.setCanvasOpen);

  const [phase, setPhase] = useState<Phase>('clarify');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);
  const [isClarifying, setIsClarifying] = useState(false);
  const [progressLogs, setProgressLogs] = useState<string[]>([]);
  const [researchMonitor, setResearchMonitor] = useState<ResearchMonitorState>(createEmptyMonitor);
  const [outlineDraft, setOutlineDraft] = useState<string[]>([]);
  const [briefDraft, setBriefDraft] = useState<BriefDraft | null>(null);
  const [initialStats, setInitialStats] = useState<InitialStats | null>(null);
  const [optimizationPromptDraft, setOptimizationPromptDraft] = useState('');

  const abortControllerRef = useRef<AbortController | null>(null);
  const lastEventIdRef = useRef(0);
  const canvasRefreshCounterRef = useRef(0);

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
        appendToLastMessage('\n\n[提示] Deep Research 已停止。');
        addToast('Deep Research 已停止', 'info');
      } else {
        appendToLastMessage(`\n\n[错误] Deep Research 失败：${job.error_message || job.message || 'unknown error'}`);
        addToast('Deep Research 失败，请重试', 'error');
      }
    } finally {
      stopStreaming();
      localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
      setActiveJobId(null);
      setIsStopping(false);
      setIsStreaming(false);
      setDeepResearchActive(false);
      setPhase('confirm');
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
          // Stream ended normally (server closed after job_status or job not found)
          break;
        } catch (err) {
          if (ac.signal.aborted) break;
          console.error('[DeepResearch] SSE stream error, retrying in', retryDelay, 'ms:', err);
          await new Promise<void>((resolve) => setTimeout(resolve, retryDelay));
          retryDelay = Math.min(retryDelay * 2, 10000);
        }
      }
    })();
  };

  // Restore job state when dialog opens
  useEffect(() => {
    if (!showDeepResearchDialog) return;
    const savedJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (!savedJobId || activeJobId) return;
    if (!sessionId) return;

    let cancelled = false;
    (async () => {
      try {
        const job = await getDeepResearchJob(savedJobId);
        if (cancelled) return;
        const isRunnable = job.status === 'running' || job.status === 'cancelling';
        const sameSession = !job.session_id || job.session_id === sessionId;
        const requestedTopic = (useChatStore.getState().deepResearchTopic || '').trim();
        const runningTopic = String(job.topic || '').trim();
        const sameTopic = !requestedTopic || !runningTopic || requestedTopic === runningTopic;
        if (!isRunnable) {
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

    const webSourceConfigs: Record<string, { topK: number; threshold: number }> = {};
    webSearchConfig.sources.forEach((source) => {
      if (source.enabled) {
        webSourceConfigs[source.id] = { topK: source.topK, threshold: source.threshold };
      }
    });

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
    const deepFinalTopK = Math.max(ragConfig.finalTopK ?? 10, 30);

    return {
      searchMode,
      enabledProviders,
      webSourceConfigs,
      webEnabled,
      localEnabled,
      queryOptimizerEnabled,
      maxQueries,
      deepFinalTopK,
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
    const defaults = useConfigStore.getState().deepResearchDefaults;
    const scopeValue = (defaults.stepModels.scope || '').trim();
    let resolvedProvider: string | undefined;
    let resolvedModel: string | undefined;
    if (scopeValue && scopeValue.includes('::')) {
      const [provider, model] = scopeValue.split('::', 2);
      resolvedProvider = provider || undefined;
      resolvedModel = model || undefined;
    } else if (scopeValue) {
      resolvedProvider = selectedProvider || undefined;
      resolvedModel = scopeValue || undefined;
    } else {
      resolvedProvider = selectedProvider || undefined;
      resolvedModel = selectedModel || undefined;
    }
    setIsClarifying(true);
    try {
      const result = await clarifyForDeepResearch({
        message: trimTopic,
        session_id: useChatStore.getState().sessionId || undefined,
        search_mode: 'hybrid',
        llm_provider: resolvedProvider,
        model_override: resolvedModel,
      });
      const qs = result.questions || [];
      setClarificationQuestions(qs);
      if (result.suggested_topic) {
        setDeepResearchTopic(result.suggested_topic);
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
    const { topic, answers, outputLanguage, yearStart, yearEnd, stepModels, stepModelStrict } = params;
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
      deepFinalTopK,
    } = buildCommonRequestParams();
    const hasNonEmptyAnswers = Object.values(answers).some((v) => v.trim().length > 0);
    const startRequest: DeepResearchStartRequest = {
      topic,
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      collection: currentCollection || undefined,
      search_mode: searchMode,
      llm_provider: selectedProvider || undefined,
      model_override: selectedModel || undefined,
      web_providers: webEnabled ? enabledProviders : undefined,
      web_source_configs: (webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      use_query_optimizer: webEnabled ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: webEnabled ? maxQueries : undefined,
      local_top_k: localEnabled ? Math.max(ragConfig.localTopK, 15) : undefined,
      local_threshold: localEnabled ? (ragConfig.localThreshold ?? undefined) : undefined,
      year_start: yearStart ?? undefined,
      year_end: yearEnd ?? undefined,
      final_top_k: deepFinalTopK,
      clarification_answers: hasNonEmptyAnswers ? answers : undefined,
      output_language: outputLanguage,
      step_models: normalizeStepModels(stepModels),
      step_model_strict: stepModelStrict,
    };
    setIsStreaming(true);
    try {
      const startResp = await deepResearchStart(startRequest);
      if (startResp.session_id) setSessionId(startResp.session_id);
      if (startResp.canvas_id) setCanvasId(startResp.canvas_id);
      setOutlineDraft(startResp.outline?.length ? startResp.outline : [topic]);
      setBriefDraft(startResp.brief || null);
      setInitialStats(startResp.initial_stats || null);
      if (startResp.used_fallback) {
        addToast(`澄清阶段触发回退：${startResp.fallback_reason || 'unknown'}`, 'warning');
      }
      setPhase('confirm');
    } catch (error) {
      console.error('[DeepResearch] Start error:', error);
      addToast('研究规划生成失败，请重试', 'error');
    } finally {
      setIsStreaming(false);
    }
  };

  const confirmAndRun = async (params: ConfirmAndRunParams) => {
    const {
      topic, outlineDraft: rawOutline, briefDraft: brief, outputLanguage, yearStart, yearEnd,
      stepModels, stepModelStrict, userContext, userContextMode, tempDocuments,
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
      deepFinalTopK,
    } = buildCommonRequestParams();

    setDeepResearchActive(true);
    setPhase('running');
    setProgressLogs([]);
    addMessage({ role: 'user', content: `[Deep Research] ${topic}` });
    addMessage({ role: 'assistant', content: '' });
    setWorkflowStep('explore');

    const confirmRequest: DeepResearchConfirmRequest = {
      topic,
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
      model_override: selectedModel || undefined,
      web_providers: webEnabled ? enabledProviders : undefined,
      web_source_configs: (webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      use_query_optimizer: webEnabled ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: webEnabled ? maxQueries : undefined,
      local_top_k: localEnabled ? Math.max(ragConfig.localTopK, 15) : undefined,
      local_threshold: localEnabled ? (ragConfig.localThreshold ?? undefined) : undefined,
      year_start: yearStart ?? undefined,
      year_end: yearEnd ?? undefined,
      final_top_k: deepFinalTopK,
      user_context: userContext.trim() || undefined,
      user_context_mode: userContext.trim() ? userContextMode : undefined,
      user_documents: tempDocuments.length ? tempDocuments : undefined,
      depth,
      skip_draft_review: skipDraftReview,
      skip_refine_review: skipRefineReview,
      skip_claim_generation: skipClaimGeneration,
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

  return {
    phase,
    setPhase,
    activeJobId,
    isStopping,
    isClarifying,
    progressLogs,
    researchMonitor,
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
    handleInsertOptimizationPrompt,
  };
}
