import { useState, useRef, useCallback, useEffect, useMemo, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowRight, Loader2, Telescope, Settings, Image as ImageIcon, Pause, Play, Square } from 'lucide-react';
import { useChatStore, useConfigStore, useToastStore, useCanvasStore, useUIStore } from '../../stores';
import {
  submitChat,
  streamChatByTaskId,
  clarifyForDeepResearch,
  getChatSuggestions,
  cancelTask,
  pauseTask,
  resumeTask,
  cancelDeepResearchJob,
  pauseDeepResearchJob,
  resumeDeepResearchJob,
} from '../../api/chat';
import { exportCanvas, getCanvas } from '../../api/canvas';
import { CommandPalette, DeepResearchSettingsPopover, GraphicAbstractPopover } from '../workflow';
import { COMMAND_LIST } from '../../types';
import type { ChatRequest, Source, EvidenceSummary, IntentInfo, CommandDefinition, WorkflowStep } from '../../types';

/** 1 分钟未选择时默认「本会话不用本地库」的间隔（毫秒） */
const LOCAL_DB_CHOICE_TIMEOUT_MS = 60_000;

/** Chat input history (localStorage) for cross-session suggestions */
const CHAT_INPUT_HISTORY_KEY = 'chat_input_history_v1';
const CHAT_INPUT_HISTORY_MAX = 30;
const SUGGESTIONS_MIN_PREFIX = 2;
const SUGGESTIONS_MAX = 8;

function loadChatInputHistory(): string[] {
  try {
    const raw = localStorage.getItem(CHAT_INPUT_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item: unknown) => String(item ?? '').trim()).filter(Boolean);
  } catch {
    return [];
  }
}

function saveChatInputHistory(message: string) {
  const trimmed = message.trim();
  if (!trimmed) return;
  try {
    const prev = loadChatInputHistory();
    const next = prev.filter((s) => s !== trimmed);
    next.push(trimmed);
    localStorage.setItem(
      CHAT_INPUT_HISTORY_KEY,
      JSON.stringify(next.slice(-CHAT_INPUT_HISTORY_MAX)),
    );
  } catch {
    // ignore quota errors
  }
}

export function ChatInput() {
  const [inputValue, setInputValue] = useState('');
  const [showDRSettings, setShowDRSettings] = useState(false);
  const [showGASettings, setShowGASettings] = useState(false);
  const [suggestionIndex, setSuggestionIndex] = useState(-1);
  const [dismissedSuggestions, setDismissedSuggestions] = useState(false);
  const [backendSuggestions, setBackendSuggestions] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsDropdownRef = useRef<HTMLDivElement>(null);
  const suggestionsDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** 用 ref 保存 runChoice，确保 1 分钟定时器触发时一定能执行，不依赖 store 状态 */
  const localDbChoiceRunRef = useRef<((choice: 'no_local' | 'use') => Promise<void>) | null>(null);
  const localDbChoiceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toggleDRSettings = useCallback(() => setShowDRSettings((v) => !v), []);
  const closeDRSettings = useCallback(() => setShowDRSettings(false), []);
  const toggleGASettings = useCallback(() => setShowGASettings((v) => !v), []);
  const closeGASettings = useCallback(() => setShowGASettings(false), []);

  useEffect(() => {
    return () => {
      if (localDbChoiceTimeoutRef.current) {
        clearTimeout(localDbChoiceTimeoutRef.current);
        localDbChoiceTimeoutRef.current = null;
      }
      localDbChoiceRunRef.current = null;
    };
  }, []);
  const {
    sessionId,
    canvasId,
    messages,
    addMessage,
    appendToMessageById,
    setMessageSourcesById,
    setSessionId,
    setWorkflowStep,
    setLastEvidenceSummary,
    setShowDeepResearchDialog,
    setDeepResearchTopic,
    setClarificationQuestions,
    setStreamingTask,
    clearStreamingTask,
    setStreamingStep,
    setActiveResponse,
    patchActiveResponse,
    activeResponse,
    streamingTasks,
    setMessageAgentDebugById,
    updateMessageTimestampById,
    setPendingLocalDbChoice,
    clearPendingLocalDbChoice,
    setLocalDbChoiceHandler,
    setLocalDbChoiceTimeoutId,
  } = useChatStore();
  const {
    webSearchConfig,
    ragConfig,
    selectedProvider,
    selectedModel,
    currentCollection,
    selectedCollections,
    collections: validCollectionNames,
    deepResearchDefaults,
  } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const { setCanvas, setCanvasContent, setIsLoading: setCanvasLoading } = useCanvasStore();
  const { setCanvasOpen, requestSessionListRefresh } = useUIStore();
  const { t } = useTranslation();
  const isCurrentSessionBusy = Boolean(
    sessionId &&
      Object.values(streamingTasks).some(
        (task) => task.sessionId === sessionId && ['queued', 'running', 'pausing', 'paused'].includes(task.status)
      )
  );
  const controlTaskId = activeResponse?.taskOrJobId?.trim() || '';
  const isResearchControl = activeResponse?.kind === 'research';
  const isPaused = activeResponse?.streamPhase === 'paused';
  const isPauseTransitioning = activeResponse?.stepKey === 'pausing';
  const hasControllableTask = Boolean(controlTaskId);

  const applyControlState = useCallback((status: 'pausing' | 'paused' | 'running', label?: string) => {
    if (!activeResponse) return;
    const nextPhase = status === 'paused' ? 'paused' : 'thinking';
    patchActiveResponse({
      streamPhase: nextPhase,
      stepKey: status,
      stepLabel:
        label
        || (status === 'paused'
          ? t('chat.taskPaused', 'Paused')
          : status === 'pausing'
            ? t('chat.taskPausing', 'Pausing...')
            : t('chat.taskResumed', 'Resumed')),
    });
    if (controlTaskId && !isResearchControl) {
      const currentSessionId = useChatStore.getState().sessionId;
      setStreamingTask(controlTaskId, currentSessionId ?? null, status);
    }
    if (status === 'paused') {
      setStreamingStep(null);
    }
  }, [activeResponse, controlTaskId, isResearchControl, patchActiveResponse, setStreamingStep, setStreamingTask, t]);

  const handleStopActiveTask = useCallback(async () => {
    if (!controlTaskId) return;
    try {
      if (isResearchControl) {
        await cancelDeepResearchJob(controlTaskId);
      } else {
        await cancelTask(controlTaskId);
      }
      patchActiveResponse({
        stepKey: 'cancelling',
        stepLabel: t('chat.taskStopping', 'Stopping...'),
        streamPhase: 'thinking',
      });
      addToast(t('chat.stopRequested', 'Stop request sent, task is terminating'), 'info');
    } catch (err) {
      console.error('[ChatInput] stop task failed:', err);
      addToast(t('chat.stopFailed', 'Stop failed'), 'error');
    }
  }, [addToast, controlTaskId, isResearchControl, patchActiveResponse, t]);

  const handleTogglePauseActiveTask = useCallback(async () => {
    if (!controlTaskId || isPauseTransitioning) return;
    try {
      if (isPaused) {
        if (isResearchControl) {
          await resumeDeepResearchJob(controlTaskId);
        } else {
          await resumeTask(controlTaskId);
        }
        applyControlState('running');
        addToast(t('chat.taskResumed', 'Resumed'), 'info');
      } else {
        if (isResearchControl) {
          await pauseDeepResearchJob(controlTaskId);
        } else {
          await pauseTask(controlTaskId);
        }
        applyControlState('pausing');
        addToast(t('chat.taskPauseRequested', 'Pause requested'), 'info');
      }
    } catch (err) {
      console.error('[ChatInput] toggle pause failed:', err);
      addToast(t('chat.taskControlFailed', 'Task control failed'), 'error');
    }
  }, [addToast, applyControlState, controlTaskId, isPauseTransitioning, isPaused, isResearchControl, t]);

  const prefix = inputValue.trim();
  const prefixLower = prefix.toLowerCase();
  const isCommandMode = inputValue.startsWith('/');

  const suggestions = useMemo(() => {
    if (isCommandMode || prefix.length < SUGGESTIONS_MIN_PREFIX) return [];
    const sessionContents = messages
      .filter((m) => m.role === 'user' && typeof m.content === 'string' && (m.content as string).trim())
      .map((m) => (m.content as string).trim());
    const fromSession = sessionContents
      .slice()
      .reverse()
      .filter(
        (s) => s.toLowerCase().startsWith(prefixLower) || s.toLowerCase().includes(prefixLower),
      );
    const fromStorage = loadChatInputHistory().filter(
      (s) => s.toLowerCase().startsWith(prefixLower) || s.toLowerCase().includes(prefixLower),
    );
    const seen = new Set<string>();
    const merged: string[] = [];
    for (const s of fromSession) {
      if (!seen.has(s)) {
        seen.add(s);
        merged.push(s);
      }
    }
    for (const s of fromStorage) {
      if (!seen.has(s)) {
        seen.add(s);
        merged.push(s);
      }
    }
    for (const s of backendSuggestions) {
      if (!seen.has(s)) {
        seen.add(s);
        merged.push(s);
      }
    }
    return merged.slice(0, SUGGESTIONS_MAX);
  }, [messages, prefix, prefixLower, isCommandMode, backendSuggestions]);

  const showSuggestionsDropdown =
    suggestions.length > 0 && prefix.length >= SUGGESTIONS_MIN_PREFIX && !isCommandMode && !dismissedSuggestions;

  useEffect(() => {
    if (!showSuggestionsDropdown) {
      setSuggestionIndex(-1);
      return;
    }
    setSuggestionIndex((i) => (i < 0 || i >= suggestions.length ? 0 : Math.min(i, suggestions.length - 1)));
  }, [showSuggestionsDropdown, suggestions.length]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        suggestionsDropdownRef.current &&
        !suggestionsDropdownRef.current.contains(event.target as Node)
      ) {
        setDismissedSuggestions(true);
        setSuggestionIndex(-1);
      }
    }
    if (showSuggestionsDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showSuggestionsDropdown]);

  useEffect(() => {
    if (prefix.length < SUGGESTIONS_MIN_PREFIX || isCommandMode || !sessionId) {
      setBackendSuggestions([]);
      if (suggestionsDebounceRef.current) {
        clearTimeout(suggestionsDebounceRef.current);
        suggestionsDebounceRef.current = null;
      }
      return;
    }
    suggestionsDebounceRef.current = setTimeout(() => {
      suggestionsDebounceRef.current = null;
      getChatSuggestions({ prefix, session_id: sessionId, limit: 5 })
        .then((r) => setBackendSuggestions(r.suggestions || []))
        .catch(() => setBackendSuggestions([]));
    }, 250);
    return () => {
      if (suggestionsDebounceRef.current) {
        clearTimeout(suggestionsDebounceRef.current);
        suggestionsDebounceRef.current = null;
      }
    };
  }, [prefix, isCommandMode, sessionId]);

  const acceptSuggestion = useCallback(
    (suggestion: string) => {
      setInputValue(suggestion);
      setDismissedSuggestions(true);
      setSuggestionIndex(-1);
      inputRef.current?.focus();
    },
    [],
  );

  const handleTaskControlEvent = useCallback((event: string, data: unknown) => {
    const message = typeof data === 'object' && data && 'message' in data ? String((data as { message?: unknown }).message || '') : '';
    if (event === 'pause_requested') {
      applyControlState('pausing', message || t('chat.taskPausing', 'Pausing...'));
      return true;
    }
    if (event === 'paused') {
      applyControlState('paused', message || t('chat.taskPaused', 'Paused'));
      return true;
    }
    if (event === 'resumed') {
      applyControlState('running', message || t('chat.taskResumed', 'Resumed'));
      return true;
    }
    return false;
  }, [applyControlState, t]);

  const toWorkflowStep = (step?: string | null): WorkflowStep => {
    switch (step) {
      case 'idle':
      case 'explore':
      case 'outline':
      case 'drafting':
      case 'refine':
        return step;
      default:
        return 'explore';
    }
  };

  /**
   * 触发 Deep Research 流程（打开澄清对话框）
   * 澄清模型使用当前 UI 选中的 provider/model，避免被 step scope 覆盖。
   *
   * @param topic 必须显式传入主题，不从 inputValue 闭包读取，避免 React 并发模式下的值捕获问题
   */
  const handleDeepResearch = async (topic: string) => {
    const researchTopic = topic.trim();
    if (!researchTopic) {
      addToast(t('chatInput.enterTopic'), 'error');
      return;
    }
    setInputValue('');
    closeDRSettings(); // 关闭设置弹窗（如果打开的话）
    setDeepResearchTopic(researchTopic);

    // Clarify uses Deep Research setting "提问 模型" and "初步研究 模型".
    const { deepResearchDefaults } = useConfigStore.getState();
    
    const questModel = (deepResearchDefaults.questionModel ?? '').trim();
    let llmProvider: string | undefined = selectedProvider || undefined;
    let modelOverride: string | undefined = selectedModel || undefined;
    if (questModel) {
      const idx = questModel.indexOf('::');
      if (idx > 0) {
        llmProvider = questModel.slice(0, idx).trim() || undefined;
        modelOverride = questModel.slice(idx + 2).trim() || undefined;
      }
    }
    
    const prelimModel = (deepResearchDefaults.preliminaryModel ?? '').trim();
    let prelimProvider: string | undefined = undefined;
    let prelimModelOverride: string | undefined = undefined;
    if (prelimModel) {
      const idx = prelimModel.indexOf('::');
      if (idx > 0) {
        prelimProvider = prelimModel.slice(0, idx).trim() || undefined;
        prelimModelOverride = prelimModel.slice(idx + 2).trim() || undefined;
      }
    }

    // 调用澄清问题生成 API
    try {
      addToast(t('chatInput.generatingQuestions'), 'info');
      const result = await clarifyForDeepResearch({
        message: researchTopic,
        session_id: sessionId || undefined,
        search_mode: 'hybrid',
        llm_provider: llmProvider,
        model_override: modelOverride,
        prelim_provider: prelimProvider,
        prelim_model: prelimModelOverride,
      });
      const questions = result.questions || [];
      setClarificationQuestions(questions);
      // 仅当 suggested_topic 非空时才覆盖（trim 避免空白字符串误判为有效值）
      const suggestedTopic = (result.suggested_topic || '').trim();
      if (suggestedTopic) {
        setDeepResearchTopic(suggestedTopic);
      }
      if (questions.length === 0) {
        addToast(t('chatInput.topicClear'), 'info');
      }
      setShowDeepResearchDialog(true);
    } catch (err) {
      console.error('[ChatInput] Clarify failed:', err);
      setClarificationQuestions([
        {
          id: 'q1',
          text: t('chatInput.defaultQuestion'),
          question_type: 'text',
          options: [],
          default: researchTopic,
        },
      ]);
      addToast(t('chatInput.clarifyFailed'), 'info');
      setShowDeepResearchDialog(true);
    }
  };

  /**
   * 发送 Chat 消息（检索由 UI 开关决定，不再做意图检测）
   */
  const handleSend = async (messageOverride?: string, modeOverride?: 'chat' | 'deep_research') => {
    const messageToSend = messageOverride || inputValue.trim();
    if (!messageToSend) return;

    // /auto 命令 → 触发 Deep Research 流程
    if (messageToSend.startsWith('/auto')) {
      const topic = messageToSend.replace(/^\/auto\s*/, '').trim();
      if (topic) {
        handleDeepResearch(topic);
      } else {
        addToast(t('chatInput.enterTopic'), 'error');
      }
      return;
    }

    const userMessage = messageToSend;
    setInputValue('');
    const assistantMessageId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    addMessage({ role: 'user', content: userMessage });
    addMessage({ id: assistantMessageId, role: 'assistant', content: '' });
    setActiveResponse({
      kind: 'chat',
      taskOrJobId: '',
      surface: 'chat',
      stepKey: 'queued',
      stepLabel: t('chat.thinking', 'Thinking'),
      streamPhase: 'thinking',
      targetMessageId: assistantMessageId,
      hasVisibleOutput: false,
    });
    setWorkflowStep('explore');

    try {
      // 构建检索参数
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

    // search_mode 纯由 UI 开关决定
    let searchMode: 'local' | 'web' | 'hybrid' | 'none';
    if (localEnabled && webEnabled) {
      searchMode = 'hybrid';
    } else if (webEnabled) {
      searchMode = 'web';
    } else if (localEnabled) {
      searchMode = 'local';
    } else {
      searchMode = 'none';
    }

    const mode = modeOverride || 'chat';
    const sonarStrength = ragConfig.sonarStrength ?? 'sonar-reasoning-pro';

    const request: ChatRequest = {
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      message: userMessage,
      collection: currentCollection || undefined,
      collections: (() => {
        const valid = selectedCollections.filter((c) => validCollectionNames.includes(c));
        return valid.length > 1 ? valid : undefined;
      })(),
      search_mode: searchMode,
      mode,
      llm_provider: selectedProvider || undefined,
      ultra_lite_provider: deepResearchDefaults.ultra_lite_provider || undefined,
      model_override: selectedModel || undefined,
      web_providers: (searchMode !== 'none' && webEnabled) ? enabledProviders : undefined,
      web_source_configs: (searchMode !== 'none' && webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      serpapi_ratio: (searchMode !== 'none' && webEnabled && hasAnySerpapi) ? (webSearchConfig.serpapiRatio ?? 50) / 100 : undefined,
      local_top_k: (searchMode !== 'none' && localEnabled) ? ragConfig.localTopK : undefined,
      pool_score_thresholds: (searchMode !== 'none' && ragConfig.poolScoreThresholds?.chat) ? (ragConfig.poolScoreThresholds.chat as unknown as Record<string, number>) : undefined,
      fused_pool_score_threshold: (searchMode !== 'none') ? (ragConfig.fusedPoolScoreThreshold ?? 0.35) : undefined,
      year_start: ragConfig.yearStart ?? undefined,
      year_end: ragConfig.yearEnd ?? undefined,
      step_top_k: (searchMode !== 'none') ? (ragConfig.stepTopK ?? 10) : undefined,
      write_top_k: (searchMode !== 'none') ? (ragConfig.writeTopK ?? 15) : undefined,
      graph_top_k: (searchMode !== 'none' && ragConfig.enableHippoRAG) ? ragConfig.graphTopK : undefined,
      use_content_fetcher: (searchMode !== 'none' && webEnabled) ? webSearchConfig.contentFetcherMode : undefined,
      agent_mode: ragConfig.agentMode ?? 'standard',
      sonar_strength: sonarStrength,
      use_sonar_prelim: sonarStrength !== 'off',
      sonar_model: sonarStrength !== 'off' ? sonarStrength : undefined,
      agent_sonar_model: (searchMode !== 'none' && webEnabled && enabledProviders.includes('sonar'))
        ? (ragConfig.agentSonarModel ?? 'sonar-pro')
        : undefined,
      max_iterations: ragConfig.maxIterations ?? 2,
      output_language: deepResearchDefaults.outputLanguage ?? 'auto',
      reranker_mode: searchMode !== 'none'
        ? (ragConfig.enableReranker
          ? ((localStorage.getItem('adv_reranker_mode') || 'cascade') as 'bge_only' | 'colbert_only' | 'cascade')
          : 'bge_only')
        : undefined,
      agent_debug_mode: ragConfig.agentDebugMode ?? false,
      enable_graphic_abstract: ragConfig.enableGraphicAbstract ?? false,
      graphic_abstract_model: ragConfig.graphicAbstractModel ?? 'gemini',
    };

    if (import.meta.env.DEV) {
      console.info('[ChatInput] request.web_source_configs', {
        search_mode: request.search_mode,
        web_source_configs: request.web_source_configs,
        semantic: request.web_source_configs?.semantic,
      });
      const base = import.meta.env.VITE_API_BASE_URL || '/api';
      console.log('[ChatInput] POST', `${base}/chat/submit`, JSON.stringify(request, null, 2));
    }

    // ── 引用类型（供本轮所有流处理函数共享）─────────────────────────────────────
    interface CitationData {
      cite_key: string; title: string; authors: string[];
      year?: number | null; doc_id?: string | null; url?: string | null; pdf_url?: string | null;
      doi?: string | null; bbox?: number[]; page_num?: number | null; provider?: string;
    }

    // ── 激活本地库选择弹窗（幂等，重复调用无效）──────────────────────────────────
    let localDbChoiceActivated = false;
    const activateLocalDbChoice = () => {
      if (localDbChoiceActivated) return;
      localDbChoiceActivated = true;
      console.log('[ChatInput] activateLocalDbChoice: showing dialog, msg=', assistantMessageId);

      setPendingLocalDbChoice(true, assistantMessageId, userMessage);

      const runChoice = async (choice: 'no_local' | 'use') => {
        // 防止重复执行（按钮点击 + 超时同时触发）
        if (localDbChoiceTimeoutRef.current) {
          clearTimeout(localDbChoiceTimeoutRef.current);
          localDbChoiceTimeoutRef.current = null;
        }
        localDbChoiceRunRef.current = null;
        setLocalDbChoiceTimeoutId(null);

        // 始终从 store 读取最新 session_id（第一轮完成后 meta 事件可能已更新它）
        const latestSessionId = useChatStore.getState().sessionId;
        console.log('[ChatInput] runChoice:', choice, 'latestSessionId=', latestSessionId);

        let secondTaskId: string | null = null;
        try {
          // 1. 提交偏好任务（让后端记录用户选择并准备第二轮检索）
          const preq = {
            ...request,
            session_id: latestSessionId || request.session_id,
            session_preference_local_db: choice,
            message: '（设置偏好）',
          };
          const { task_id: tid1 } = await submitChat(preq);
          setStreamingTask(tid1, latestSessionId ?? null, 'running');
          patchActiveResponse({ kind: 'chat', taskOrJobId: tid1, surface: 'chat', targetMessageId: assistantMessageId });
          const s1 = streamChatByTaskId(tid1);
          for await (const { event: ev, data: d } of s1) {
            if (handleTaskControlEvent(ev, d)) continue;
            if (ev === 'meta' && (d as { session_id?: string }).session_id)
              setSessionId((d as { session_id: string }).session_id);
            if (ev === 'delta') {
              appendToMessageById(assistantMessageId, (d as { delta: string }).delta);
              patchActiveResponse({ hasVisibleOutput: true, streamPhase: 'streaming' });
            }
          }
          clearStreamingTask(tid1);

          // 2. 清除选择 UI，开始真实的第二轮问答
          clearPendingLocalDbChoice();
          setLocalDbChoiceHandler(null);
          const newAssistantId = `assistant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          addMessage({ id: newAssistantId, role: 'assistant', content: '' });
          setActiveResponse({
            kind: 'chat',
            taskOrJobId: '',
            surface: 'chat',
            stepKey: 'queued',
            stepLabel: t('chat.thinking', 'Thinking'),
            streamPhase: 'thinking',
            targetMessageId: newAssistantId,
            hasVisibleOutput: false,
          });

          // 再次读取最新 session_id（偏好任务可能返回了新的 session_id）
          const sessionIdForReq2 = useChatStore.getState().sessionId;
          const req2 = {
            ...request,
            session_id: sessionIdForReq2 || latestSessionId || request.session_id,
            message: userMessage,
          };
          delete (req2 as Record<string, unknown>).session_preference_local_db;
          const { task_id: tid2 } = await submitChat(req2);
          secondTaskId = tid2;
          setStreamingTask(tid2, sessionIdForReq2 ?? null, 'queued');
          patchActiveResponse({ taskOrJobId: tid2, targetMessageId: newAssistantId });
          requestSessionListRefresh();

          const s2 = streamChatByTaskId(tid2);
          for await (const { event: ev, data: d } of s2) {
            if (handleTaskControlEvent(ev, d)) continue;
            if (ev === 'meta') {
              const m = d as { session_id?: string; canvas_id?: string; citations?: CitationData[]; evidence_summary?: EvidenceSummary | null; current_stage?: string };
              if (m.session_id) setSessionId(m.session_id);
              if (m.current_stage) setWorkflowStep(toWorkflowStep(m.current_stage));
              if (m.evidence_summary) setLastEvidenceSummary(m.evidence_summary);
              if (m.citations?.length) {
                const srcs: Source[] = m.citations.map((c, i) => ({
                  id: i + 1, cite_key: c.cite_key, title: c.title || c.cite_key, authors: c.authors || [], year: c.year,
                  doc_id: c.doc_id, url: c.url, pdf_url: c.pdf_url, doi: c.doi, bbox: c.bbox, page_num: c.page_num,
                  type: c.url || c.pdf_url ? 'web' : 'local', provider: c.provider || (c.url || c.pdf_url ? 'web' : 'local'),
                }));
                setMessageSourcesById(newAssistantId, srcs, m.evidence_summary?.provider_stats);
              }
            }
            if (ev === 'step') {
              const stepPayload = d as { step?: string | null; label?: string };
              setStreamingStep(stepPayload?.step ? { step: stepPayload.step, label: stepPayload.label ?? stepPayload.step } : null);
              patchActiveResponse({
                stepKey: stepPayload?.step ?? null,
                stepLabel: stepPayload?.label ?? stepPayload?.step ?? t('chat.thinking', 'Thinking'),
                streamPhase: 'thinking',
              });
            }
            if (ev === 'delta') {
              appendToMessageById(newAssistantId, (d as { delta: string }).delta);
              patchActiveResponse({ hasVisibleOutput: true, streamPhase: 'streaming' });
            }
            if (ev === 'done') {
              const donePayload = d as { final_text?: string };
              if (donePayload?.final_text) {
                useChatStore.getState().updateMessageById(newAssistantId, donePayload.final_text);
              }
              setStreamingStep(null);
              setActiveResponse(null);
              updateMessageTimestampById(newAssistantId);
              setWorkflowStep('idle');
            }
            if (ev === 'error' || ev === 'cancelled' || ev === 'timeout') {
              setStreamingStep(null);
              setActiveResponse(null);
              updateMessageTimestampById(newAssistantId);
              break;
            }
          }
        } catch (e) {
          console.error('[ChatInput] localDbChoice follow-up error:', e);
          addToast(t('chat.sendFailed'), 'error');
        } finally {
          setStreamingStep(null);
          setActiveResponse(null);
          if (secondTaskId) clearStreamingTask(secondTaskId);
        }
      };

      setLocalDbChoiceHandler((choice) => runChoice(choice));
      localDbChoiceRunRef.current = runChoice;
      if (localDbChoiceTimeoutRef.current) clearTimeout(localDbChoiceTimeoutRef.current);
      localDbChoiceTimeoutRef.current = setTimeout(() => {
        console.log('[ChatInput] localDbChoice 60s timeout → auto no_local');
        localDbChoiceRunRef.current?.('no_local');
      }, LOCAL_DB_CHOICE_TIMEOUT_MS);
      setLocalDbChoiceTimeoutId(localDbChoiceTimeoutRef.current);
    };

    // ── 提交任务并订阅 SSE ────────────────────────────────────────────────────
    let taskId: string | null = null;
    try {
      const { task_id } = await submitChat(request);
      taskId = task_id;
      setStreamingTask(task_id, sessionId ?? null, 'queued');
      patchActiveResponse({ taskOrJobId: task_id });
      requestSessionListRefresh();
      const stream = streamChatByTaskId(task_id);

      for await (const { event, data } of stream) {
        console.log('[ChatInput] SSE event:', event, typeof data === 'object' ? JSON.stringify(data).slice(0, 200) : data);
        if (handleTaskControlEvent(event, data)) continue;

        if (event === 'meta') {
          const meta = data as {
            session_id: string; canvas_id?: string;
            citations: CitationData[]; evidence_summary: EvidenceSummary | null;
            intent?: IntentInfo; current_stage?: string;
            prompt_local_db_choice?: boolean; local_db_mismatch_message?: string | null;
          };

          // 若本地库范围不符，弹出选择弹窗
          if (meta.prompt_local_db_choice) activateLocalDbChoice();

          if (meta.session_id) {
            setSessionId(meta.session_id);
            if (taskId) setStreamingTask(taskId, meta.session_id, 'running');
          }
          if (meta.current_stage) setWorkflowStep(toWorkflowStep(meta.current_stage));
          if (meta.evidence_summary) {
            setLastEvidenceSummary(meta.evidence_summary);
            if (!meta.current_stage) setWorkflowStep('outline');
          }
          if (meta.citations && meta.citations.length > 0) {
            const sources: Source[] = meta.citations.map((cite, idx) => ({
              id: idx + 1, cite_key: cite.cite_key, title: cite.title || cite.cite_key,
              authors: cite.authors || [], year: cite.year, doc_id: cite.doc_id,
              url: cite.url, pdf_url: cite.pdf_url, doi: cite.doi, bbox: cite.bbox, page_num: cite.page_num,
              type: cite.url || cite.pdf_url ? 'web' : 'local', provider: cite.provider || (cite.url || cite.pdf_url ? 'web' : 'local'),
            }));
            setMessageSourcesById(assistantMessageId, sources, meta.evidence_summary?.provider_stats || undefined);
          }
          if (meta.canvas_id) {
            setCanvasLoading(true);
            Promise.all([
              getCanvas(meta.canvas_id).catch((err) => { console.error('[ChatInput] Canvas load failed:', err); return null; }),
              exportCanvas(meta.canvas_id, 'markdown').catch((err) => { console.error('[ChatInput] Canvas export failed:', err); return null; }),
            ])
              .then(([canvasData, exportResp]) => {
                if (canvasData) setCanvas(canvasData);
                if (exportResp?.content) { setCanvasContent(exportResp.content); setCanvasOpen(true); }
              })
              .finally(() => setCanvasLoading(false));
          }

        } else if (event === 'local_db_choice') {
          // 专属事件：后端检测到范围不符时单独推送，比 meta 字段更可靠
          console.log('[ChatInput] local_db_choice event received');
          activateLocalDbChoice();

        } else if (event === 'dashboard') {
          useChatStore.getState().setResearchDashboard(data as import('../../types').ResearchDashboardData);
        } else if (event === 'tool_trace') {
          useChatStore.getState().setToolTrace(data as import('../../types').ToolTraceItem[]);
        } else if (event === 'agent_debug') {
          setMessageAgentDebugById(assistantMessageId, data as import('../../types').AgentDebugData);

        } else if (event === 'step') {
          const stepPayload = data as { step?: string | null; label?: string };
          setStreamingStep(
            stepPayload?.step
              ? { step: stepPayload.step, label: stepPayload.label ?? stepPayload.step }
              : null
          );
          patchActiveResponse({
            stepKey: stepPayload?.step ?? null,
            stepLabel: stepPayload?.label ?? stepPayload?.step ?? t('chat.thinking', 'Thinking'),
            streamPhase: 'thinking',
          });

        } else if (event === 'delta') {
          const delta = (data as { delta: string }).delta;
          appendToMessageById(assistantMessageId, delta);
          patchActiveResponse({ hasVisibleOutput: true, streamPhase: 'streaming' });
          setWorkflowStep('drafting');
          // 兜底：若 meta / local_db_choice 事件均未触发，根据消息内容识别范围不符关键词
          if (!useChatStore.getState().pendingLocalDbChoice) {
            const accContent = (useChatStore.getState().messages.find(m => m.id === assistantMessageId)?.content ?? '') + delta;
            if (accContent.includes('当前问题与本地知识库') && accContent.includes('您可以选择')) {
              console.log('[ChatInput] delta fallback: activating localDbChoice');
              activateLocalDbChoice();
            }
          }

        } else if (event === 'done') {
          const donePayload = data as { final_text?: string };
          if (donePayload?.final_text) {
            useChatStore.getState().updateMessageById(assistantMessageId, donePayload.final_text);
          }
          setStreamingStep(null);
          setActiveResponse(null);
          updateMessageTimestampById(assistantMessageId);
          setWorkflowStep('refine');
          setTimeout(() => setWorkflowStep('idle'), 1000);
          saveChatInputHistory(userMessage);
        } else if (event === 'error' || event === 'cancelled' || event === 'timeout') {
          setStreamingStep(null);
          setActiveResponse(null);
          updateMessageTimestampById(assistantMessageId);
          if (event === 'error') {
            appendToMessageById(assistantMessageId, '\n\n' + (t('chat.requestError') || 'Error'));
            addToast(t('chat.sendFailed'), 'error');
          }
          break;
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      const resStatus = error && typeof error === 'object' && 'response' in error
        ? (error as { response?: { status?: number } }).response?.status
        : 0;
      const is500 = resStatus === 500 ||
        (error instanceof Error && (error.message.includes('500') || error.message.includes('Internal Server Error')));
      if (is500) {
        addToast(t('chat.serverError', '服务器错误(500)，请稍后重试'), 'error');
      } else {
        addToast(t('chat.sendFailed'), 'error');
      }
      updateMessageTimestampById(assistantMessageId);
      appendToMessageById(assistantMessageId, '\n\n' + (t('chat.requestError') || '请求失败'));
    } finally {
      setStreamingStep(null);
      setActiveResponse(null);
      if (taskId) clearStreamingTask(taskId);
    }
    } catch (err) {
      console.error('[ChatInput] handleSend error (e.g. missing store field):', err);
      addToast(t('chat.sendFailed'), 'error');
    }
  };

  const filteredCommands = inputValue.startsWith('/')
    ? COMMAND_LIST.filter(cmd =>
        cmd.command.toLowerCase().includes(inputValue.slice(1).toLowerCase()) ||
        cmd.label.toLowerCase().includes(inputValue.slice(1).toLowerCase())
      )
    : [];

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (showSuggestionsDropdown && suggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSuggestionIndex((i) => Math.min(i + 1, suggestions.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSuggestionIndex((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Tab' || e.key === 'Enter') {
        if (suggestionIndex >= 0 && suggestions[suggestionIndex]) {
          e.preventDefault();
          acceptSuggestion(suggestions[suggestionIndex]);
          return;
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setDismissedSuggestions(true);
        setSuggestionIndex(-1);
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === 'Tab' && inputValue.startsWith('/')) {
      e.preventDefault();
      if (filteredCommands.length > 0) {
        handleSelectCommand(filteredCommands[0]);
      }
    }
  };

  const handleSelectCommand = (cmd: CommandDefinition) => {
    if (cmd.mode === 'deep_research') {
      // 从当前输入框中提取主题（去除命令前缀，如 "/auto "）
      const rawInput = inputValue.trim();
      const match = rawInput.match(/^\/\S+\s+(.*)/);
      const topic = match ? match[1].trim() : '';
      setInputValue('');
      if (topic) {
        handleDeepResearch(topic);
      } else {
        // 没有额外输入主题时直接打开对话框，topic 字段留给用户填写
        setDeepResearchTopic('');
        setShowDeepResearchDialog(true);
      }
      return;
    }
    setInputValue(cmd.command + ' ');
    inputRef.current?.focus();
  };

  return (
    <div className="p-4 glass-header border-t border-slate-700/50 z-30">
      <div className="max-w-3xl mx-auto relative">
        {/* 命令面板 */}
        <CommandPalette
          inputValue={inputValue}
          onSelectCommand={handleSelectCommand}
        />

        {/* 顶部控制栏 (语言, 图文摘要, Deep Research) 与 输入区域分两行显示，让输入框占满宽度 */}
        <div className="flex flex-col gap-2">
          {/* 顶部控制栏 */}
          <div className="flex items-center justify-end gap-2">
            {/* 输出语言 */}
            <select
              value={deepResearchDefaults.outputLanguage ?? 'auto'}
              onChange={(e) => useConfigStore.getState().updateDeepResearchDefaults({ outputLanguage: e.target.value as 'auto' | 'en' | 'zh' })}
              title={t('sidebar.outputLanguage')}
              className="flex-shrink-0 h-[36px] px-2.5 rounded-lg border border-slate-700 bg-slate-800/60 text-slate-200 text-sm focus:ring-1 focus:ring-sky-500 focus:border-sky-500 outline-none cursor-pointer"
            >
              <option value="auto">{t('chatInput.langAuto', 'Auto')}</option>
              <option value="en">EN</option>
              <option value="zh">中文</option>
            </select>

            {/* Graphic Abstract 设置按钮 */}
            <div className="relative flex items-center">
              <button
                onClick={toggleGASettings}
                className={`
                  flex items-center px-2 py-2 rounded-lg border text-sm
                  transition-all cursor-pointer
                  ${showGASettings
                    ? 'bg-pink-900/30 text-pink-400 border-pink-500/30'
                    : 'bg-slate-800/60 text-slate-400 border-slate-700/60 hover:bg-slate-700 hover:text-slate-300'
                  }
                `}
                title="Graphic Abstract 设置"
              >
                <ImageIcon size={16} />
              </button>
              <GraphicAbstractPopover open={showGASettings} onClose={closeGASettings} />
            </div>

            {/* Deep Research 按钮组：⚙ 设置 + 🔭 启动 */}
            <div className="relative flex items-center">
              <button
                onClick={toggleDRSettings}
                className={`
                  flex items-center px-2 py-2 rounded-l-lg border border-r-0 text-sm
                  transition-all cursor-pointer
                  ${showDRSettings
                    ? 'bg-indigo-900/30 text-indigo-400 border-indigo-500/30'
                    : 'bg-slate-800/60 text-slate-400 border-slate-700/60 hover:bg-slate-700 hover:text-slate-300'
                  }
                `}
                title={t('chatInput.drSettingsTitle')}
              >
                <Settings size={14} />
              </button>
              <button
                onClick={() => handleDeepResearch(inputValue)}
                disabled={!inputValue.trim()}
                className={`
                  flex items-center gap-1.5 px-3 py-2 rounded-r-lg border text-sm font-medium
                  transition-all cursor-pointer whitespace-nowrap
                  ${!inputValue.trim()
                    ? 'bg-slate-800/60 text-slate-500 border-slate-700/60 cursor-not-allowed'
                    : 'bg-gradient-to-r from-indigo-900/30 to-purple-900/30 text-indigo-400 border-indigo-500/30 hover:border-indigo-400 hover:shadow-sm'
                  }
                `}
                title="Deep Research - 多步深度研究"
              >
                <Telescope size={16} />
                <span className="hidden sm:inline">Deep Research</span>
              </button>

              {/* Settings popover (positioned above the button group) */}
              <DeepResearchSettingsPopover open={showDRSettings} onClose={closeDRSettings} />
            </div>
          </div>

          {/* 输入框 + 历史建议下拉 */}
          <div ref={suggestionsDropdownRef} className="w-full relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                setDismissedSuggestions(false);
              }}
              onKeyDown={handleKeyDown}
              placeholder={t('chatInput.placeholder')}
              className="w-full bg-slate-900/60 border border-slate-700 text-slate-200 rounded-xl py-3 pl-4 pr-32 shadow-sm focus:ring-1 focus:ring-sky-500 focus:border-sky-500 placeholder-slate-500 outline-none transition-all"
              aria-autocomplete="list"
              aria-controls="chat-suggestions-listbox"
              aria-expanded={showSuggestionsDropdown}
              aria-activedescendant={
                showSuggestionsDropdown && suggestionIndex >= 0
                  ? `chat-suggestion-${suggestionIndex}`
                  : undefined
              }
            />
            {showSuggestionsDropdown && (
              <ul
                id="chat-suggestions-listbox"
                role="listbox"
                aria-label={t('chatInput.suggestionsFromHistory')}
                className="absolute top-full left-0 right-0 mt-1 max-h-56 overflow-y-auto rounded-xl border border-slate-700/50 bg-slate-900/95 backdrop-blur-md shadow-xl z-50 py-1 animate-in fade-in slide-in-from-top-1 duration-150"
              >
                {suggestions.map((s, i) => (
                  <li
                    key={`${i}-${s.slice(0, 40)}`}
                    id={`chat-suggestion-${i}`}
                    role="option"
                    aria-selected={i === suggestionIndex}
                    className={`px-4 py-2.5 text-sm text-left cursor-pointer truncate max-w-full ${
                      i === suggestionIndex ? 'bg-sky-900/30 text-sky-200' : 'text-slate-300 hover:bg-slate-800/60'
                    }`}
                    onClick={() => acceptSuggestion(s)}
                  >
                    {s}
                  </li>
                ))}
              </ul>
            )}
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
              <button
                onClick={() => void handleTogglePauseActiveTask()}
                disabled={!hasControllableTask || isPauseTransitioning}
                title={isPaused ? t('chat.resumeTask', 'Resume task') : t('chat.pauseTask', 'Pause task')}
                className="w-8 h-8 bg-slate-800/90 text-slate-200 rounded-lg flex items-center justify-center hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                {isPaused ? <Play size={16} /> : <Pause size={16} />}
              </button>
              <button
                onClick={() => void handleStopActiveTask()}
                disabled={!hasControllableTask}
                title={t('chat.stopTask', 'Stop task')}
                className="w-8 h-8 bg-rose-900/80 text-rose-100 rounded-lg flex items-center justify-center hover:bg-rose-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
              >
                <Square size={14} />
              </button>
              <button
                onClick={() => handleSend()}
                disabled={isCurrentSessionBusy || !inputValue.trim()}
                className="w-8 h-8 bg-sky-600 text-white rounded-lg flex items-center justify-center hover:bg-sky-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                {isCurrentSessionBusy ? (
                  <Loader2 size={18} className={isPaused ? '' : 'animate-spin'} />
                ) : (
                  <ArrowRight size={18} />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* 快捷提示 */}
        <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
          <span>
            {t('chatInput.inputCommand')} <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">/</kbd> {t('chatInput.viewCommands')}
            {' | '}
            <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">/auto</kbd> {t('chatInput.deepResearch')}
            {' | '}
            <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">Tab</kbd> {t('chatInput.tabToConfirmSuggestion')}
          </span>
          <span>
            {t('chatInput.pressEnter')} <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">Enter</kbd> {t('chatInput.toSend')}
          </span>
        </div>
      </div>
    </div>
  );
}
