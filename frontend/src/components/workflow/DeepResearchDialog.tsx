import { useState, useEffect, useMemo, useRef } from 'react';
import { Telescope, Loader2, X, ChevronRight, Plus, Trash2, GripVertical, Square, Paperclip, Copy } from 'lucide-react';
import { useChatStore, useConfigStore, useToastStore, useCanvasStore, useUIStore } from '../../stores';
import {
  clarifyForDeepResearch,
  deepResearchStart,
  deepResearchSubmit,
  getDeepResearchJob,
  listDeepResearchJobEvents,
  cancelDeepResearchJob,
  extractDeepResearchContextFiles,
} from '../../api/chat';
import { exportCanvas, getCanvas } from '../../api/canvas';
import type {
  DeepResearchConfirmRequest,
  DeepResearchStartRequest,
  DeepResearchJobEvent,
  Source,
  ResearchDashboardData,
  ChatCitation,
} from '../../types';

/**
 * Deep Research 澄清对话框
 * 在用户触发 Deep Research 时弹出，显示 LLM 生成的澄清问题，
 * 用户回答后发送 Deep Research 请求。
 */
export function DeepResearchDialog() {
  const DEEP_RESEARCH_JOB_KEY = 'deep_research_active_job_id';
  const DEEP_RESEARCH_PENDING_CONTEXT_KEY = 'deep_research_pending_user_context';
  const DEEP_RESEARCH_ARCHIVED_JOBS_KEY = 'deep_research_archived_job_ids';
  type ProgressPayload = {
    type?: string;
    section?: string;
    message?: string;
    [key: string]: unknown;
  };
  type BriefDraft = Record<string, unknown>;
  type InitialStats = {
    total_sources?: number;
    total_iterations?: number;
    [key: string]: unknown;
  };
  type ResearchMonitorState = {
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

  const createEmptyMonitor = (): ResearchMonitorState => ({
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

  const {
    showDeepResearchDialog,
    setShowDeepResearchDialog,
    deepResearchTopic,
    setDeepResearchTopic,
    clarificationQuestions,
    setClarificationQuestions,
    sessionId,
    canvasId,
    isStreaming,
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
  const { webSearchConfig, ragConfig, selectedProvider, selectedModel, currentCollection } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const { setCanvas, setCanvasContent, setIsLoading: setCanvasLoading, setActiveStage } = useCanvasStore();
  const setCanvasOpen = useUIStore((s) => s.setCanvasOpen);

  // 用户对每个问题的回答
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [phase, setPhase] = useState<'clarify' | 'confirm' | 'running'>('clarify');
  const [outlineDraft, setOutlineDraft] = useState<string[]>([]);
  const [briefDraft, setBriefDraft] = useState<BriefDraft | null>(null);
  const [initialStats, setInitialStats] = useState<InitialStats | null>(null);
  const [progressLogs, setProgressLogs] = useState<string[]>([]);
  const [researchMonitor, setResearchMonitor] = useState<ResearchMonitorState>(createEmptyMonitor);
  const [outputLanguage, setOutputLanguage] = useState<'auto' | 'en' | 'zh'>('auto');
  const [showAdvancedModels, setShowAdvancedModels] = useState(false);
  const [stepModelStrict, setStepModelStrict] = useState(false);
  const [draggingOutlineIndex, setDraggingOutlineIndex] = useState<number | null>(null);
  const [dragOverOutlineIndex, setDragOverOutlineIndex] = useState<number | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);
  const [userContext, setUserContext] = useState('');
  const [userContextMode, setUserContextMode] = useState<'supporting' | 'direct_injection'>('supporting');
  const [tempDocuments, setTempDocuments] = useState<Array<{ name: string; content: string }>>([]);
  const [isExtractingContextFiles, setIsExtractingContextFiles] = useState(false);
  const [depth, setDepth] = useState<'lite' | 'comprehensive'>('comprehensive');
  const [skipDraftReview, setSkipDraftReview] = useState(false);
  const [skipRefineReview, setSkipRefineReview] = useState(false);
  const [keepPreviousJobId, setKeepPreviousJobId] = useState(true);
  const [optimizationPromptDraft, setOptimizationPromptDraft] = useState('');
  const contextFileInputRef = useRef<HTMLInputElement | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const isPollingRef = useRef(false);
  const lastEventIdRef = useRef(0);
  const canvasRefreshCounterRef = useRef(0);
  const [stepModels, setStepModels] = useState<Record<string, string>>({
    scope: 'sonar::sonar-pro',
    plan: '',
    research: '',
    evaluate: '',
    write: '',
    verify: '',
    synthesize: '',
  });

  // 初始化默认回答
  useEffect(() => {
    if (clarificationQuestions.length > 0) {
      const defaults: Record<string, string> = {};
      clarificationQuestions.forEach((q) => {
        defaults[q.id] = q.default || '';
      });
      setAnswers(defaults);
    }
  }, [clarificationQuestions]);

  const [isClarifying, setIsClarifying] = useState(false);

  /** Resolve scope model from persistent defaults for clarify calls */
  const resolveScopeModel = () => {
    const defaults = useConfigStore.getState().deepResearchDefaults;
    const scopeValue = (defaults.stepModels.scope || '').trim();
    if (scopeValue && scopeValue.includes('::')) {
      const [provider, model] = scopeValue.split('::', 2);
      return { llm_provider: provider || undefined, model_override: model || undefined };
    }
    if (scopeValue) {
      return { llm_provider: selectedProvider || undefined, model_override: scopeValue || undefined };
    }
    return { llm_provider: selectedProvider || undefined, model_override: selectedModel || undefined };
  };

  const runClarify = async (topic: string) => {
    const trimTopic = (topic || '').trim();
    if (!trimTopic) return;
    const resolved = resolveScopeModel();
    setIsClarifying(true);
    try {
      const result = await clarifyForDeepResearch({
        message: trimTopic,
        session_id: useChatStore.getState().sessionId || undefined,
        search_mode: 'hybrid',
        llm_provider: resolved.llm_provider,
        model_override: resolved.model_override,
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
          `澄清模型：${result.llm_provider_used || resolved.llm_provider || 'default'}`
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

  useEffect(() => {
    if (showDeepResearchDialog) {
      if (activeJobId) {
        setPhase('running');
      } else {
        setPhase('clarify');
        setOutlineDraft([]);
        setBriefDraft(null);
        setInitialStats(null);
        setProgressLogs([]);
        setResearchMonitor(createEmptyMonitor());
        setOptimizationPromptDraft('');
        const pendingUserContext = localStorage.getItem(DEEP_RESEARCH_PENDING_CONTEXT_KEY) || '';
        if (pendingUserContext.trim()) {
          setUserContext(pendingUserContext.trim());
          setUserContextMode('direct_injection');
          localStorage.removeItem(DEEP_RESEARCH_PENDING_CONTEXT_KEY);
        } else {
          setUserContext('');
          setUserContextMode('supporting');
        }
        setTempDocuments([]);

        // Initialize local overrides from persistent defaults (configured via gear popover)
        const defaults = useConfigStore.getState().deepResearchDefaults;
        setDepth(defaults.depth);
        setOutputLanguage(defaults.outputLanguage);
        setStepModelStrict(defaults.stepModelStrict);
        setStepModels({ ...defaults.stepModels });
        setAnswers({});
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDeepResearchDialog, activeJobId]);

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    isPollingRef.current = false;
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
        // Prefer synthesized final markdown as refine content.
        setCanvasContent(job.result_markdown);
      }
      if (job.canvas_id) {
        setCanvasLoading(true);
        // 加载结构化 canvas 数据（用于 stage views）+ Markdown 内容（用于 Refine 预览）
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
      stopPolling();
      localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
      setActiveJobId(null);
      setIsStopping(false);
      setIsStreaming(false);
      setDeepResearchActive(false);
      setPhase('confirm');
    }
  };

  const pollJobOnce = async (jobId: string) => {
    if (isPollingRef.current) return;
    isPollingRef.current = true;
    try {
      const events = await listDeepResearchJobEvents(jobId, lastEventIdRef.current, 200);
      if (events.length > 0) {
        const maxId = events[events.length - 1].event_id;
        lastEventIdRef.current = Math.max(lastEventIdRef.current, maxId);
        appendProgressEvents(events);
      }
      const job = await getDeepResearchJob(jobId);
      if (job.result_dashboard && Object.keys(job.result_dashboard).length > 0) {
        setResearchDashboard(job.result_dashboard as unknown as ResearchDashboardData);
      }
      if (job.message) {
        setProgressLogs((prev) => (prev[prev.length - 1] === `[status] ${job.message}` ? prev : [...prev, `[status] ${job.message}`].slice(-300)));
      }
      canvasRefreshCounterRef.current += 1;
      const runningCanvasId = job.canvas_id || canvasId;
      if (runningCanvasId && canvasRefreshCounterRef.current % 3 === 0) {
        getCanvas(runningCanvasId)
          .then((canvasData) => {
            if (!canvasData) return;
            setCanvas(canvasData);
            if ((job.status === 'running' || job.status === 'cancelling') && (canvasData.stage === 'drafting' || canvasData.stage === 'refine')) {
              setCanvasOpen(true);
            }
          })
          .catch((err) => {
            console.debug('[DeepResearch] periodic canvas refresh failed:', err);
          });
      }
      if (job.status === 'done' || job.status === 'cancelled' || job.status === 'error') {
        await finalizeRunningJob(jobId);
      }
    } catch (err) {
      console.error('[DeepResearch] poll failed:', err);
    } finally {
      isPollingRef.current = false;
    }
  };

  const startPollingJob = (jobId: string, resetEvents: boolean) => {
    stopPolling();
    if (resetEvents) {
      lastEventIdRef.current = 0;
      canvasRefreshCounterRef.current = 0;
      setProgressLogs([]);
      setResearchMonitor(createEmptyMonitor());
      setOptimizationPromptDraft('');
    }
    setActiveJobId(jobId);
    localStorage.setItem(DEEP_RESEARCH_JOB_KEY, jobId);
    pollJobOnce(jobId);
    pollTimerRef.current = window.setInterval(() => {
      pollJobOnce(jobId);
    }, 2000);
  };

  useEffect(() => {
    if (!showDeepResearchDialog) return;
    const savedJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (!savedJobId || activeJobId) return;
    // 新对话未绑定 session 时，不自动恢复旧任务，避免“新对话无法启动新任务”
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
        // If user is starting a NEW topic, don't auto-resume an old running job.
        // This preserves clarify -> outline confirm flow for the new request.
        if (!sameTopic) {
          addToast('检测到其他进行中的 Deep Research 任务；当前按新主题进入大纲确认流程。', 'info');
          return;
        }
        setPhase('running');
        setIsStreaming(true);
        setDeepResearchActive(true);
        startPollingJob(savedJobId, false);
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

  useEffect(() => () => {
    stopPolling();
  }, []);

  const modelOptions = useMemo(() => {
    const current = selectedModel
      ? `${selectedProvider}::${selectedModel}`
      : '';
    return [
      { value: '', label: 'Default (current global model)' },
      ...(current ? [{ value: current, label: `Current: ${current}` }] : []),
      { value: 'sonar::sonar', label: 'sonar (search)' },
      { value: 'sonar::sonar-pro', label: 'sonar-pro (search)' },
      { value: 'sonar::sonar-reasoning-pro', label: 'sonar-reasoning-pro' },
    ];
  }, [selectedProvider, selectedModel]);

  const monitorSectionEntries = useMemo(
    () => Object.entries(researchMonitor.sectionCoverage),
    [researchMonitor.sectionCoverage],
  );
  const sectionEfficiencyRows = useMemo(() => {
    const rows = monitorSectionEntries
      .map(([section, coverageValues]) => {
        if (!coverageValues || coverageValues.length < 2) return null;
        const stepValues = researchMonitor.sectionSteps[section] || [];
        const firstCoverage = coverageValues[0];
        const lastCoverage = coverageValues[coverageValues.length - 1];
        const rounds = coverageValues.length - 1;
        const deltaCoverage = lastCoverage - firstCoverage;
        const avgDelta = deltaCoverage / Math.max(1, rounds);
        const lastDelta = coverageValues[coverageValues.length - 1] - coverageValues[coverageValues.length - 2];
        let per10Steps: number | null = null;
        if (stepValues.length >= 2) {
          const stepSpan = stepValues[stepValues.length - 1] - stepValues[0];
          if (stepSpan > 0) {
            per10Steps = deltaCoverage / (stepSpan / 10);
          }
        }
        const score = avgDelta * 100;
        let level: 'high' | 'medium' | 'low' = 'low';
        if (avgDelta >= 0.08) level = 'high';
        else if (avgDelta >= 0.03) level = 'medium';
        return {
          section,
          firstCoverage,
          lastCoverage,
          rounds,
          avgDelta,
          lastDelta,
          per10Steps,
          score,
          level,
        };
      })
      .filter((row): row is NonNullable<typeof row> => Boolean(row));
    return rows.sort((a, b) => b.score - a.score);
  }, [monitorSectionEntries, researchMonitor.sectionSteps]);
  const targetCoverage = depth === 'lite' ? 0.6 : 0.8;
  const highEfficiencyRows = useMemo(
    () => sectionEfficiencyRows.filter((row) => row.level === 'high').slice(0, 3),
    [sectionEfficiencyRows],
  );
  const lowEfficiencyRows = useMemo(
    () => sectionEfficiencyRows.filter((row) => row.level === 'low').slice(0, 3),
    [sectionEfficiencyRows],
  );
  const handleGenerateOptimizationPrompt = () => {
    const targets = lowEfficiencyRows.length > 0 ? lowEfficiencyRows : sectionEfficiencyRows.slice(0, 3);
    if (!targets.length) {
      addToast('当前样本不足，请至少完成两轮 section evaluate', 'info');
      return;
    }
    const lines: string[] = [];
    lines.push(`# Deep Research Section Optimization Template`);
    lines.push(`Topic: ${deepResearchTopic || '(fill topic)'}`);
    lines.push(`Target coverage: ${targetCoverage.toFixed(2)}`);
    lines.push('');
    lines.push(`Usage: paste selected blocks into "Intervention" before next run.`);
    lines.push('');
    targets.forEach((row, idx) => {
      lines.push(`## ${idx + 1}. ${row.section}`);
      lines.push(`Current signal:`);
      lines.push(`- Coverage: ${row.lastCoverage.toFixed(2)} (target ${targetCoverage.toFixed(2)})`);
      lines.push(`- Avg gain/round: ${row.avgDelta.toFixed(3)}`);
      if (row.per10Steps !== null) {
        lines.push(`- Gain per 10 steps: ${row.per10Steps.toFixed(3)}`);
      }
      lines.push(`Optimization prompt skeleton:`);
      lines.push(`- Scope constraints:`);
      lines.push(`  - Focus only on "${row.section}"`);
      lines.push(`  - Exclude adjacent sections and generic narrative`);
      lines.push(`- Retrieval directives:`);
      lines.push(`  - Expand terminology variants, abbreviations, and mechanism synonyms`);
      lines.push(`  - Prioritize primary studies and data-bearing sources`);
      lines.push(`- Evidence directives:`);
      lines.push(`  - Provide explicit support for each major claim with citation tags`);
      lines.push(`  - Flag weak claims as evidence-limited instead of asserting`);
      lines.push(`- My supplemental evidence:`);
      lines.push(`  - [Paste your materials, notes, or constraints here]`);
      lines.push('');
    });
    const next = lines.join('\n');
    setOptimizationPromptDraft(next);
    addToast('已生成章节优化提示词模板，可复制后用于 Intervention', 'success');
  };
  const handleCopyOptimizationPrompt = async () => {
    if (!optimizationPromptDraft.trim()) return;
    try {
      await navigator.clipboard.writeText(optimizationPromptDraft);
      addToast('已复制优化提示词模板', 'success');
    } catch (err) {
      console.error('[DeepResearch] copy optimization prompt failed:', err);
      addToast('复制失败，请手动复制文本', 'warning');
    }
  };
  const handleInsertOptimizationPrompt = () => {
    const text = optimizationPromptDraft.trim();
    if (!text) return;
    setUserContext((prev) => {
      const current = (prev || '').trim();
      if (!current) return text;
      if (current.includes(text)) return current;
      return `${current}\n\n${text}`;
    });
    addToast('已写入 Intervention，可在下一轮直接使用', 'success');
  };

  if (!showDeepResearchDialog) return null;

  const handleClose = () => {
    setShowDeepResearchDialog(false);
    if (!activeJobId) {
      setDeepResearchActive(false);
    }
  };

  const handleAnswerChange = (questionId: string, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

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

  const normalizeStepModels = () => {
    const out: Record<string, string | null> = {};
    Object.entries(stepModels).forEach(([k, v]) => {
      out[k] = (v || '').trim() || null;
    });
    return out;
  };

  const handleGeneratePlan = async () => {
    if (!deepResearchTopic.trim()) {
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
      topic: deepResearchTopic,
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
      final_top_k: deepFinalTopK,
      clarification_answers: hasNonEmptyAnswers ? answers : undefined,
      output_language: outputLanguage,
      step_models: normalizeStepModels(),
      step_model_strict: stepModelStrict,
    };
    setIsStreaming(true);
    try {
      const startResp = await deepResearchStart(startRequest);
      if (startResp.session_id) setSessionId(startResp.session_id);
      if (startResp.canvas_id) setCanvasId(startResp.canvas_id);
      setOutlineDraft(startResp.outline?.length ? startResp.outline : [deepResearchTopic]);
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

  const handleSkipClarificationAndGenerate = async () => {
    setAnswers({});
    await handleGeneratePlan();
  };

  const handleConfirmAndRun = async () => {
    if (!deepResearchTopic.trim()) return;
    const filteredOutline = outlineDraft.map((x) => x.trim()).filter(Boolean);
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
    addMessage({ role: 'user', content: `[Deep Research] ${deepResearchTopic}` });
    addMessage({ role: 'assistant', content: '' });
    setWorkflowStep('explore');

    const confirmRequest: DeepResearchConfirmRequest = {
      topic: deepResearchTopic,
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      collection: currentCollection || undefined,
      search_mode: searchMode,
      confirmed_outline: filteredOutline,
      confirmed_brief: briefDraft || undefined,
      output_language: outputLanguage,
      step_models: normalizeStepModels(),
      step_model_strict: stepModelStrict,
      llm_provider: selectedProvider || undefined,
      model_override: selectedModel || undefined,
      web_providers: webEnabled ? enabledProviders : undefined,
      web_source_configs: (webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      use_query_optimizer: webEnabled ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: webEnabled ? maxQueries : undefined,
      local_top_k: localEnabled ? Math.max(ragConfig.localTopK, 15) : undefined,
      local_threshold: localEnabled ? (ragConfig.localThreshold ?? undefined) : undefined,
      final_top_k: deepFinalTopK,
      user_context: userContext.trim() || undefined,
      user_context_mode: userContext.trim() ? userContextMode : undefined,
      user_documents: tempDocuments.length ? tempDocuments : undefined,
      depth,
      skip_draft_review: skipDraftReview,
      skip_refine_review: skipRefineReview,
    };

    let submitted = false;
    const previousActiveJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    try {
      const submitResp = await deepResearchSubmit(confirmRequest);
      if (submitResp.session_id) setSessionId(submitResp.session_id);
      if (submitResp.canvas_id) setCanvasId(submitResp.canvas_id);
      if (
        keepPreviousJobId
        && previousActiveJobId
        && previousActiveJobId !== submitResp.job_id
      ) {
        archiveJobId(previousActiveJobId);
      }
      setWorkflowStep('drafting');
      startPollingJob(submitResp.job_id, true);
      setShowDeepResearchDialog(false);
      submitted = true;
      addToast('已转为后台执行，可安全关闭当前前端页面', 'info');
    } catch (error) {
      console.error('[DeepResearch] Error:', error);
      addToast('Deep Research 失败，请重试', 'error');
      appendToLastMessage('\n\n[错误] Deep Research 请求失败。');
      stopPolling();
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

  const handleStopRunningJob = async () => {
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

  const handleSelectContextFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    setIsExtractingContextFiles(true);
    try {
      const docs = await extractDeepResearchContextFiles(files);
      if (!docs.length) {
        addToast('未从文件提取到有效文本（支持 pdf/md/txt）', 'info');
        return;
      }
      setTempDocuments((prev) => {
        const merged = [...prev];
        docs.forEach((d) => {
          const exists = merged.some((x) => x.name === d.name && x.content === d.content);
          if (!exists) merged.push(d);
        });
        return merged.slice(0, 10);
      });
      addToast(`已添加 ${docs.length} 份临时材料`, 'success');
    } catch (err) {
      console.error('[DeepResearch] context extract failed:', err);
      addToast('临时材料提取失败，请重试', 'error');
    } finally {
      setIsExtractingContextFiles(false);
      if (contextFileInputRef.current) contextFileInputRef.current.value = '';
    }
  };

  const updateOutlineItem = (idx: number, value: string) => {
    setOutlineDraft((prev) => prev.map((item, i) => (i === idx ? value : item)));
  };

  const moveOutlineItem = (fromIdx: number, toIdx: number) => {
    if (fromIdx === toIdx) return;
    setOutlineDraft((prev) => {
      if (fromIdx < 0 || toIdx < 0 || fromIdx >= prev.length || toIdx >= prev.length) {
        return prev;
      }
      const next = [...prev];
      const [moved] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, moved);
      return next;
    });
  };

  const getQuestionRationale = (questionText: string) => {
    const text = questionText.toLowerCase();
    if (text.includes('范围') || text.includes('scope')) {
      return '用于锁定研究边界，避免大纲发散。';
    }
    if (text.includes('受众') || text.includes('风格') || text.includes('audience') || text.includes('style')) {
      return '用于匹配表达方式和写作深度。';
    }
    if (text.includes('篇幅') || text.includes('深度') || text.includes('字数') || text.includes('length')) {
      return '用于控制章节粒度和信息密度。';
    }
    if (text.includes('排除') || text.includes('exclude')) {
      return '用于减少无关检索和错误扩展。';
    }
    if (text.includes('语言') || text.includes('language')) {
      return '用于提前确定文献与输出语言策略。';
    }
    return '用于减少歧义，让后续大纲更贴合目标。';
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col animate-in slide-in-from-bottom-4 duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Telescope size={20} className="text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Deep Research</h2>
              <p className="text-xs text-gray-500">多步深度研究 - 可确认大纲并跟踪进度</p>
            </div>
          </div>
          <button onClick={handleClose} className="p-1 hover:bg-gray-100 rounded-lg transition-colors">
            <X size={18} className="text-gray-400" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
            <div className="text-xs text-gray-600">
              <span className={phase === 'clarify' ? 'font-semibold text-indigo-600' : ''}>1. 澄清问题</span>
              <span className="mx-2 text-gray-300">→</span>
              <span className={phase === 'confirm' ? 'font-semibold text-indigo-600' : ''}>2. 确认大纲</span>
              <span className="mx-2 text-gray-300">→</span>
              <span className={phase === 'running' ? 'font-semibold text-indigo-600' : ''}>3. 执行研究</span>
            </div>
          </div>

          {/* 主题 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">研究主题</label>
            <input
              type="text"
              value={deepResearchTopic}
              onChange={(e) => setDeepResearchTopic(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
              placeholder="输入综述主题..."
            />
          </div>

          {phase === 'clarify' && clarificationQuestions.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-600">
                请补充以下信息（可选，共 {clarificationQuestions.length} 题）
              </h3>
              {clarificationQuestions.map((q) => (
                <div key={q.id} className="bg-gray-50 rounded-lg p-3">
                  <label className="block text-sm text-gray-700 mb-1.5">{q.text}</label>
                  {q.question_type === 'choice' && q.options.length > 0 ? (
                    <select
                      value={answers[q.id] || ''}
                      onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                      className="w-full border border-gray-200 rounded-md px-2.5 py-1.5 text-sm bg-white focus:ring-2 focus:ring-indigo-500 outline-none"
                    >
                      {q.options.map((opt) => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={answers[q.id] || ''}
                      onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                      placeholder={q.default || '输入回答...'}
                      className="w-full border border-gray-200 rounded-md px-2.5 py-1.5 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                    />
                  )}
                  <div className="mt-1.5 text-xs text-gray-500">
                    为什么问：{getQuestionRationale(q.text)}
                  </div>
                </div>
              ))}
            </div>
          )}

          {phase === 'clarify' && clarificationQuestions.length === 0 && (
            <div className="text-center py-4 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
              {isClarifying ? (
                <div className="flex items-center justify-center gap-2">
                  <Loader2 size={14} className="animate-spin text-indigo-500" />
                  <span>正在生成澄清问题...</span>
                </div>
              ) : (
                'Click "Regenerate" to generate clarification questions, or proceed directly to outline.'
              )}
            </div>
          )}

          {/* Clarify phase: compact settings summary + regenerate button */}
          {phase === 'clarify' && (
            <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg px-3 py-2.5">
              <div className="text-[11px] text-gray-500 flex items-center gap-1.5 flex-wrap">
                <span className="font-medium text-gray-700">{depth === 'lite' ? 'Lite' : 'Comprehensive'}</span>
                <span className="text-gray-300">|</span>
                <span>Scope: <span className="font-medium">{stepModels.scope || 'default'}</span></span>
                <span className="text-gray-300">|</span>
                <span>Lang: {outputLanguage === 'auto' ? 'Auto' : outputLanguage}</span>
                <span className="text-gray-300">|</span>
                <span className="text-[10px] text-gray-400">via input &#9881;</span>
              </div>
              <button
                onClick={() => runClarify(deepResearchTopic)}
                disabled={isClarifying || !deepResearchTopic.trim()}
                className="inline-flex items-center gap-1 px-2 py-1 border rounded-md text-[11px] text-indigo-600 hover:bg-indigo-50 disabled:opacity-50 shrink-0 ml-2"
              >
                {isClarifying ? <Loader2 size={10} className="animate-spin" /> : <ChevronRight size={10} />}
                Regenerate
              </button>
            </div>
          )}

          {/* Confirm phase: output language (can override per-run) */}
          {phase === 'confirm' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Output Language</label>
              <select
                value={outputLanguage}
                onChange={(e) => setOutputLanguage(e.target.value as 'auto' | 'en' | 'zh')}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
              >
                <option value="auto">Auto (follow topic language)</option>
                <option value="en">English</option>
                <option value="zh">中文</option>
              </select>
            </div>
          )}

          {/* Confirm phase: collapsed per-step model override */}
          {phase === 'confirm' && (
            <div className="space-y-2">
              <button
                onClick={() => setShowAdvancedModels((v) => !v)}
                className="text-xs text-indigo-600 hover:text-indigo-700"
              >
                {showAdvancedModels ? '\u25BE Hide' : '\u25B8 Override'} Per-step Models
              </button>
              {showAdvancedModels && (
                <div className="space-y-2 border border-gray-200 rounded-lg p-3">
                  <div className="text-[10px] text-gray-400 pb-1.5 border-b border-gray-100">
                    Loaded from &#9881; defaults. Changes here apply to this run only.
                  </div>
                  <label className="flex items-center justify-between text-xs text-gray-600">
                    <span className="flex items-center gap-1">
                      Strict step model resolution
                      <span className="text-gray-400 cursor-help" title="OFF: model failure falls back to default silently. ON: model failure aborts the research immediately.">?</span>
                    </span>
                    <input
                      type="checkbox"
                      checked={stepModelStrict}
                      onChange={(e) => setStepModelStrict(e.target.checked)}
                      className="accent-indigo-500"
                    />
                  </label>
                  {['scope', 'plan', 'research', 'evaluate', 'write', 'verify', 'synthesize'].map((step) => (
                    <div key={step} className="grid grid-cols-3 items-center gap-2">
                      <div className="text-xs font-medium text-gray-600 uppercase">{step}</div>
                      <select
                        value={stepModels[step] || ''}
                        onChange={(e) => setStepModels((prev) => ({ ...prev, [step]: e.target.value }))}
                        className="col-span-2 border border-gray-200 rounded-md px-2 py-1 text-xs"
                      >
                        {modelOptions.map((opt) => (
                          <option key={`${step}-${opt.value || 'default'}`} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {phase === 'confirm' && (
            <>
              <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3 text-xs text-indigo-800">
                <div>Initial sources: {initialStats?.total_sources ?? 0}</div>
                <div>Iterations: {initialStats?.total_iterations ?? 0}</div>
                <div>Tip: edit and reorder outline before execution.</div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">Outline Confirmation</div>
                {outlineDraft.map((item, idx) => (
                  <div key={`outline-${idx}`} className="space-y-1">
                    {dragOverOutlineIndex === idx && draggingOutlineIndex !== null && (
                      <div className="h-0.5 bg-indigo-500 rounded-full" />
                    )}
                    <div
                      className={`flex gap-2 ${draggingOutlineIndex === idx ? 'opacity-60' : ''}`}
                      draggable
                      onDragStart={() => setDraggingOutlineIndex(idx)}
                      onDragEnd={() => {
                        setDraggingOutlineIndex(null);
                        setDragOverOutlineIndex(null);
                      }}
                      onDragOver={(e) => {
                        e.preventDefault();
                        setDragOverOutlineIndex(idx);
                      }}
                      onDrop={() => {
                        if (draggingOutlineIndex === null) return;
                        moveOutlineItem(draggingOutlineIndex, idx);
                        setDraggingOutlineIndex(null);
                        setDragOverOutlineIndex(null);
                      }}
                    >
                    <button
                      type="button"
                      className="px-2 py-1 border rounded-md text-gray-400 hover:bg-gray-50 cursor-grab active:cursor-grabbing"
                      title="拖拽排序"
                    >
                      <GripVertical size={14} />
                    </button>
                    <input
                      type="text"
                      value={item}
                      onChange={(e) => updateOutlineItem(idx, e.target.value)}
                      placeholder="New section title..."
                      className="flex-1 border border-gray-200 rounded-md px-2.5 py-1.5 text-sm"
                    />
                    <button
                      onClick={() => setOutlineDraft((prev) => prev.filter((_, i) => i !== idx))}
                      className="px-2 py-1 border rounded-md text-gray-500 hover:bg-gray-50"
                    >
                      <Trash2 size={14} />
                    </button>
                    </div>
                  </div>
                ))}
                <button
                  onClick={() => setOutlineDraft((prev) => [...prev, ''])}
                  className="inline-flex items-center gap-1 px-2.5 py-1 border rounded-md text-xs text-gray-600 hover:bg-gray-50"
                >
                  <Plus size={12} /> Add section
                </button>
                <div className="text-xs text-gray-500">可拖拽左侧图标调整章节顺序。</div>
              </div>

              {/* 研究深度选择 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2">
                <div className="text-sm font-medium text-gray-700">Research Depth (研究深度)</div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setDepth('lite')}
                    className={`flex flex-col items-start p-2.5 rounded-lg border text-left transition-all ${
                      depth === 'lite'
                        ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <span className="text-xs font-semibold text-gray-800">Lite</span>
                    <span className="text-[10px] text-gray-500 leading-tight mt-0.5">
                      Quick but academically usable, ~5-15 min. 4 queries/section (recall+precision), tiered top_k 18/10/10, coverage &ge; 60%.
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setDepth('comprehensive')}
                    className={`flex flex-col items-start p-2.5 rounded-lg border text-left transition-all ${
                      depth === 'comprehensive'
                        ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <span className="text-xs font-semibold text-gray-800">Comprehensive</span>
                    <span className="text-[10px] text-gray-500 leading-tight mt-0.5">
                      Thorough academic review, ~20-60 min. 8 queries/section (recall+precision), tiered top_k 30/15/12, coverage &ge; 80%.
                    </span>
                  </button>
                </div>
              </div>

              {/* 阶段介入控制 — 放在大纲确认之后，位置更显眼 */}
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2.5">
                <div className="text-sm font-medium text-gray-700">Stage Intervention (阶段介入)</div>
                <div className="space-y-1.5">
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-not-allowed opacity-70">
                    <input type="checkbox" checked disabled className="accent-indigo-500" />
                    <span>澄清意图 (Clarify)</span>
                    <span className="ml-auto text-[10px] text-indigo-500 font-medium">必须</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-not-allowed opacity-70">
                    <input type="checkbox" checked disabled className="accent-indigo-500" />
                    <span>确认大纲 (Confirm Outline)</span>
                    <span className="ml-auto text-[10px] text-indigo-500 font-medium">必须</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={!skipDraftReview}
                      onChange={(e) => setSkipDraftReview(!e.target.checked)}
                      className="accent-indigo-500"
                    />
                    <span>逐章审阅 (Review Each Section)</span>
                    <span className="ml-auto text-[10px] text-gray-400">{skipDraftReview ? '已跳过' : '可选'}</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={!skipRefineReview}
                      onChange={(e) => setSkipRefineReview(!e.target.checked)}
                      className="accent-indigo-500"
                    />
                    <span>精炼修改 (Refine with Directives)</span>
                    <span className="ml-auto text-[10px] text-gray-400">{skipRefineReview ? '已跳过' : '可选'}</span>
                  </label>
                </div>
                <label className="flex items-center gap-2 text-xs text-gray-500 pt-1 border-t border-gray-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={skipDraftReview && skipRefineReview}
                    onChange={(e) => {
                      setSkipDraftReview(e.target.checked);
                      setSkipRefineReview(e.target.checked);
                    }}
                    className="accent-gray-400"
                  />
                  <span>最小化人工介入（仅保留必须步骤）</span>
                </label>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-1.5">
                <div className="text-sm font-medium text-gray-700">任务 ID 策略</div>
                <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={keepPreviousJobId}
                    onChange={(e) => setKeepPreviousJobId(e.target.checked)}
                    className="accent-indigo-500"
                  />
                  <span>开始新任务时保留旧任务 ID（便于后续恢复）</span>
                </label>
                <div className="text-[11px] text-gray-500">
                  当前任务将继续在后台运行；新任务会使用新的 job id。
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700">Intervention (补充上下文，可选)</div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">文本介入模式</label>
                  <select
                    value={userContextMode}
                    onChange={(e) => setUserContextMode(e.target.value as 'supporting' | 'direct_injection')}
                    className="w-full border border-gray-200 rounded-md px-2.5 py-1.5 text-xs"
                  >
                    <option value="supporting">作为补充上下文（默认）</option>
                    <option value="direct_injection">作为强提示直接注入（我对内容非常自信）</option>
                  </select>
                </div>
                <textarea
                  value={userContext}
                  onChange={(e) => setUserContext(e.target.value)}
                  placeholder={userContextMode === 'direct_injection'
                    ? '输入高置信观点/约束，系统会作为高优先级提示并要求显式验证...'
                    : '可补充新观点、反例、约束条件、重点文献线索...'}
                  className="w-full min-h-20 border border-gray-200 rounded-md px-2.5 py-2 text-sm"
                />
                <input
                  ref={contextFileInputRef}
                  type="file"
                  accept=".pdf,.md,.txt"
                  multiple
                  className="hidden"
                  onChange={(e) => handleSelectContextFiles(e.target.files)}
                />
                <button
                  onClick={() => contextFileInputRef.current?.click()}
                  disabled={isExtractingContextFiles}
                  className="inline-flex items-center gap-1 px-2.5 py-1 border rounded-md text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-50"
                >
                  {isExtractingContextFiles ? <Loader2 size={12} className="animate-spin" /> : <Paperclip size={12} />}
                  上传临时材料 (pdf/md/txt)
                </button>
                {tempDocuments.length > 0 && (
                  <div className="space-y-1">
                    {tempDocuments.map((doc, idx) => (
                      <div key={`${doc.name}-${idx}`} className="flex items-center justify-between text-xs bg-gray-50 border border-gray-200 rounded px-2 py-1.5">
                        <span className="truncate pr-2">{doc.name}</span>
                        <button
                          onClick={() => setTempDocuments((prev) => prev.filter((_, i) => i !== idx))}
                          className="text-gray-500 hover:text-red-500"
                        >
                          移除
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="text-xs text-gray-500">这些材料仅用于本次任务，不写入持久本地知识库。</div>
              </div>

            </>
          )}

          {phase === 'running' && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-gray-700">Research Progress</div>
              <div className="text-xs text-gray-500">Executing confirmed plan...</div>
              <div className="border border-gray-200 rounded-lg p-2.5 bg-white space-y-2">
                <div className="text-xs font-medium text-gray-700">Research Monitor</div>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
                    <div className="text-gray-500">Graph steps</div>
                    <div className={`font-medium ${
                      researchMonitor.costState === 'force'
                        ? 'text-red-600'
                        : researchMonitor.costState === 'warn'
                          ? 'text-amber-600'
                          : 'text-gray-800'
                    }`}
                    >
                      {researchMonitor.graphSteps > 0 ? researchMonitor.graphSteps : '--'}
                    </div>
                    <div className="text-[10px] text-gray-400">
                      warn {researchMonitor.warnSteps ?? '--'} / force {researchMonitor.forceSteps ?? '--'}
                    </div>
                  </div>
                  <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
                    <div className="text-gray-500">Cost status</div>
                    <div className={`font-medium ${
                      researchMonitor.costState === 'force'
                        ? 'text-red-600'
                        : researchMonitor.costState === 'warn'
                          ? 'text-amber-600'
                          : 'text-emerald-600'
                    }`}
                    >
                      {researchMonitor.costState.toUpperCase()}
                    </div>
                    <div className="text-[10px] text-gray-400">node: {researchMonitor.lastNode || '--'}</div>
                  </div>
                  <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
                    <div className="text-gray-500">Self-correction</div>
                    <div className="font-medium text-gray-800">{researchMonitor.selfCorrectionCount}</div>
                  </div>
                  <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5">
                    <div className="text-gray-500">Plateau early-stop</div>
                    <div className="font-medium text-gray-800">{researchMonitor.plateauEarlyStopCount}</div>
                  </div>
                </div>
                <div className="rounded border border-gray-200 bg-gray-50 px-2 py-1.5 text-[11px]">
                  <span className="text-gray-500">Write verification passes: </span>
                  <span className="font-medium text-gray-800">{researchMonitor.verificationContextCount}</span>
                </div>
                {monitorSectionEntries.length > 0 && (
                  <div className="space-y-1">
                    <div className="text-[11px] text-gray-600">Section coverage curve</div>
                    {monitorSectionEntries.map(([section, values]) => (
                      <div key={`cov-${section}`} className="text-[11px] text-gray-700 border border-gray-200 rounded px-2 py-1 bg-gray-50">
                        <span className="font-medium">{section}</span>
                        <span className="text-gray-400"> : </span>
                        {values.map((v, idx) => (
                          <span key={`${section}-${idx}`} className="font-mono text-[10px]">
                            {idx > 0 ? ' -> ' : ''}
                            {v.toFixed(2)}
                          </span>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
                {sectionEfficiencyRows.length > 0 && (
                  <div className="space-y-1.5">
                    <div className="text-[11px] text-gray-600">Efficiency insights</div>
                    {sectionEfficiencyRows.slice(0, 5).map((row) => (
                      <div key={`eff-${row.section}`} className="text-[11px] border border-gray-200 rounded px-2 py-1.5 bg-gray-50">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-700">{row.section}</span>
                          <span className={`font-medium ${
                            row.level === 'high'
                              ? 'text-emerald-600'
                              : row.level === 'medium'
                                ? 'text-amber-600'
                                : 'text-red-600'
                          }`}
                          >
                            {row.level.toUpperCase()} ({row.score.toFixed(1)})
                          </span>
                        </div>
                        <div className="text-[10px] text-gray-500 mt-0.5">
                          cov {row.firstCoverage.toFixed(2)} {'->'} {row.lastCoverage.toFixed(2)}
                          {' | '}avg gain {row.avgDelta.toFixed(3)}/round
                          {row.per10Steps !== null ? ` | gain/10 steps ${row.per10Steps.toFixed(3)}` : ''}
                        </div>
                      </div>
                    ))}
                    <div className="rounded border border-gray-200 bg-white px-2 py-1.5 text-[11px] text-gray-600">
                      {highEfficiencyRows.length > 0 && (
                        <div>
                          Continue deepening: {highEfficiencyRows.map((r) => r.section).join(' / ')}.
                        </div>
                      )}
                      {lowEfficiencyRows.length > 0 && (
                        <div>
                          Optimize first (prompt/evidence): {lowEfficiencyRows.map((r) => r.section).join(' / ')}.
                        </div>
                      )}
                      <div>
                        Action hint: if low-efficiency section is below target coverage ({targetCoverage.toFixed(2)}), add section-specific constraints,
                        terminology variants, or temporary materials in Intervention before next run.
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleGenerateOptimizationPrompt}
                        className="px-2 py-1 border rounded-md text-[11px] text-indigo-600 hover:bg-indigo-50"
                      >
                        Generate optimization prompt template
                      </button>
                      {optimizationPromptDraft.trim() && (
                        <>
                          <button
                            onClick={handleInsertOptimizationPrompt}
                            className="px-2 py-1 border rounded-md text-[11px] text-emerald-700 hover:bg-emerald-50"
                          >
                            Insert to Intervention
                          </button>
                          <button
                            onClick={handleCopyOptimizationPrompt}
                            className="inline-flex items-center gap-1 px-2 py-1 border rounded-md text-[11px] text-gray-600 hover:bg-gray-50"
                          >
                            <Copy size={11} />
                            Copy
                          </button>
                        </>
                      )}
                    </div>
                    {optimizationPromptDraft.trim() && (
                      <textarea
                        readOnly
                        value={optimizationPromptDraft}
                        className="w-full min-h-32 border border-gray-200 rounded-md px-2 py-1.5 text-[11px] font-mono bg-gray-50"
                      />
                    )}
                  </div>
                )}
              </div>
              <div className="max-h-56 overflow-auto border border-gray-200 rounded-lg p-2 bg-gray-50 text-xs space-y-1">
                {progressLogs.length === 0 ? (
                  <div className="text-gray-400">Waiting for progress events...</div>
                ) : (
                  progressLogs.map((line, idx) => <div key={`log-${idx}`}>{line}</div>)
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-gray-50 rounded-b-2xl">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
            disabled={phase === 'running'}
          >
            取消
          </button>
          <div className="flex items-center gap-2">
            {phase === 'clarify' && clarificationQuestions.length > 0 && (
              <button
                onClick={handleSkipClarificationAndGenerate}
                disabled={isStreaming || !deepResearchTopic.trim()}
                className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
              >
                跳过澄清
              </button>
            )}
            {phase === 'clarify' && (
              <button
                onClick={handleGeneratePlan}
                disabled={isStreaming || !deepResearchTopic.trim()}
                className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isStreaming ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
                生成大纲
              </button>
            )}
            {phase === 'confirm' && (
              <>
                <button
                  onClick={() => setPhase('clarify')}
                  disabled={isStreaming}
                  className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
                >
                  返回澄清
                </button>
                <button
                  onClick={handleConfirmAndRun}
                  disabled={isStreaming || !deepResearchTopic.trim()}
                  className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isStreaming ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
                  确认并开始研究
                </button>
              </>
            )}
            {phase === 'running' && (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 text-sm font-medium rounded-lg">
                  <Loader2 size={16} className="animate-spin" />
                  后台研究中
                </div>
                <button
                  onClick={handleStopRunningJob}
                  disabled={isStopping || !activeJobId}
                  className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 text-sm font-medium rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Square size={14} />
                  {isStopping ? '停止中...' : '停止任务'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
