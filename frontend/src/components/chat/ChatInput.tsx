import { useState, useRef, useCallback, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowRight, Loader2, Telescope, Settings } from 'lucide-react';
import { useChatStore, useConfigStore, useToastStore, useCanvasStore, useUIStore } from '../../stores';
import { chatStream, clarifyForDeepResearch } from '../../api/chat';
import { exportCanvas, getCanvas } from '../../api/canvas';
import { CommandPalette, DeepResearchSettingsPopover } from '../workflow';
import { COMMAND_LIST } from '../../types';
import type { ChatRequest, Source, EvidenceSummary, IntentInfo, CommandDefinition } from '../../types';

export function ChatInput() {
  const [inputValue, setInputValue] = useState('');
  const [showDRSettings, setShowDRSettings] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const toggleDRSettings = useCallback(() => setShowDRSettings((v) => !v), []);
  const closeDRSettings = useCallback(() => setShowDRSettings(false), []);
  const {
    sessionId,
    canvasId,
    isStreaming,
    addMessage,
    appendToLastMessage,
    setLastMessageSources,
    setSessionId,
    setWorkflowStep,
    setIsStreaming,
    setLastEvidenceSummary,
    setShowDeepResearchDialog,
    setDeepResearchTopic,
    setClarificationQuestions,
  } = useChatStore();
  const {
    webSearchConfig,
    ragConfig,
    selectedProvider,
    selectedModel,
    currentCollection,
    deepResearchDefaults,
  } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const { setCanvas, setCanvasContent, setIsLoading: setCanvasLoading } = useCanvasStore();
  const setCanvasOpen = useUIStore((s) => s.setCanvasOpen);
  const { t } = useTranslation();

  /**
   * 触发 Deep Research 流程（打开澄清对话框）
   * 使用 ⚙ 弹窗中持久化的 scope 模型来生成澄清问题。
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

    // Resolve clarify model from Deep Research persistent defaults
    const { deepResearchDefaults } = useConfigStore.getState();
    const scopeModel = (deepResearchDefaults.stepModels.scope || '').trim();
    let llmProvider: string | undefined;
    let modelOverride: string | undefined;

    if (scopeModel && scopeModel.includes('::')) {
      const [p, m] = scopeModel.split('::', 2);
      llmProvider = p || undefined;
      modelOverride = m || undefined;
    } else if (scopeModel) {
      llmProvider = selectedProvider || undefined;
      modelOverride = scopeModel || undefined;
    } else {
      llmProvider = selectedProvider || undefined;
      modelOverride = selectedModel || undefined;
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
    if (!messageToSend || isStreaming) return;

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

    addMessage({ role: 'user', content: userMessage });
    addMessage({ role: 'assistant', content: '' });
    setIsStreaming(true);
    setWorkflowStep('explore');

    // 构建检索参数
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
    const queryOptimizerEnabled = Boolean(webSearchConfig.queryOptimizer ?? true);
    const maxQueries = Math.min(5, Math.max(1, Number(webSearchConfig.maxQueriesPerProvider ?? 3)));

    const request: ChatRequest = {
      session_id: sessionId || undefined,
      canvas_id: canvasId || undefined,
      message: userMessage,
      collection: currentCollection || undefined,
      search_mode: searchMode,
      mode,
      llm_provider: selectedProvider || undefined,
      model_override: selectedModel || undefined,
      web_providers: (searchMode !== 'none' && webEnabled) ? enabledProviders : undefined,
      web_source_configs: (searchMode !== 'none' && webEnabled && Object.keys(webSourceConfigs).length > 0) ? webSourceConfigs : undefined,
      use_query_optimizer: (searchMode !== 'none' && webEnabled) ? queryOptimizerEnabled : undefined,
      query_optimizer_max_queries: (searchMode !== 'none' && webEnabled) ? maxQueries : undefined,
      local_top_k: (searchMode !== 'none' && localEnabled) ? ragConfig.localTopK : undefined,
      local_threshold: (searchMode !== 'none' && localEnabled) ? (ragConfig.localThreshold ?? undefined) : undefined,
      year_start: ragConfig.yearStart ?? undefined,
      year_end: ragConfig.yearEnd ?? undefined,
      final_top_k: (searchMode !== 'none') ? (ragConfig.finalTopK ?? 10) : undefined,
      use_content_fetcher: (searchMode !== 'none' && webEnabled) ? webSearchConfig.contentFetcherMode : undefined,
      agent_mode: ragConfig.agentMode ?? 'standard',
    };

    if (import.meta.env.DEV) {
      const base = import.meta.env.VITE_API_BASE_URL || '/api';
      console.log('[ChatInput] POST', `${base}/chat/stream`, JSON.stringify(request, null, 2));
    }

    try {
      const stream = chatStream(request);

      for await (const { event, data } of stream) {
        if (event === 'meta') {
          interface CitationData {
            cite_key: string;
            title: string;
            authors: string[];
            year?: number | null;
            doc_id?: string | null;
            url?: string | null;
            doi?: string | null;
            bbox?: number[];
            page_num?: number | null;
            provider?: string;
          }
          const meta = data as {
            session_id: string;
            canvas_id?: string;
            citations: CitationData[];
            evidence_summary: EvidenceSummary | null;
            intent?: IntentInfo;
            current_stage?: string;
          };

          if (meta.session_id) setSessionId(meta.session_id);
          if (meta.current_stage) setWorkflowStep(meta.current_stage as any);
          if (meta.evidence_summary) {
            setLastEvidenceSummary(meta.evidence_summary);
            if (!meta.current_stage) setWorkflowStep('outline');
          }

          if (meta.citations && meta.citations.length > 0) {
            const sources: Source[] = meta.citations.map((cite, idx) => ({
              id: idx + 1,
              cite_key: cite.cite_key,
              title: cite.title || cite.cite_key,
              authors: cite.authors || [],
              year: cite.year,
              doc_id: cite.doc_id,
              url: cite.url,
              doi: cite.doi,
              bbox: cite.bbox,
              page_num: cite.page_num,
              type: cite.url ? 'web' : 'local',
              provider: cite.provider || (cite.url ? 'web' : 'local'),
            }));
            const pStats = meta.evidence_summary?.provider_stats;
            setLastMessageSources(sources, pStats || undefined);
          }

          if (meta.canvas_id) {
            setCanvasLoading(true);
            Promise.all([
              getCanvas(meta.canvas_id).catch((err) => {
                console.error('[ChatInput] Canvas data load failed:', err);
                return null;
              }),
              exportCanvas(meta.canvas_id, 'markdown').catch((err) => {
                console.error('[ChatInput] Canvas markdown load failed:', err);
                return null;
              }),
            ])
              .then(([canvasData, exportResp]) => {
                if (canvasData) setCanvas(canvasData);
                if (exportResp?.content) {
                  setCanvasContent(exportResp.content);
                  setCanvasOpen(true);
                }
              })
              .finally(() => setCanvasLoading(false));
          }
        } else if (event === 'dashboard') {
          // Deep Research 进度更新
          const dashboardData = data as import('../../types').ResearchDashboardData;
          useChatStore.getState().setResearchDashboard(dashboardData);
        } else if (event === 'tool_trace') {
          // Agent 工具调用轨迹
          const traceData = data as import('../../types').ToolTraceItem[];
          useChatStore.getState().setToolTrace(traceData);
        } else if (event === 'agent_debug') {
          // Agent debug 详情（含 stats + tools_contributed）
          const debugData = data as import('../../types').AgentDebugData;
          useChatStore.getState().setLastMessageAgentDebug(debugData);
        } else if (event === 'delta') {
          appendToLastMessage((data as { delta: string }).delta);
          setWorkflowStep('drafting');
        } else if (event === 'done') {
          setWorkflowStep('refine');
          setTimeout(() => setWorkflowStep('idle'), 1000);
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      addToast(t('chat.sendFailed'), 'error');
      appendToLastMessage('\n\n' + t('chat.requestError'));
    } finally {
      setIsStreaming(false);
    }
  };

  const filteredCommands = inputValue.startsWith('/')
    ? COMMAND_LIST.filter(cmd =>
        cmd.command.toLowerCase().includes(inputValue.slice(1).toLowerCase()) ||
        cmd.label.toLowerCase().includes(inputValue.slice(1).toLowerCase())
      )
    : [];

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
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

        {/* 输入区域 */}
        <div className="flex items-center gap-2">
          {/* Deep Research 按钮组：⚙ 设置 + 🔭 启动 */}
          <div className="relative flex items-center">
            <button
              onClick={toggleDRSettings}
              className={`
                flex items-center px-2 py-2.5 rounded-l-lg border border-r-0 text-sm
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
              disabled={isStreaming || !inputValue.trim()}
              className={`
                flex items-center gap-1.5 px-3 py-2.5 rounded-r-lg border text-sm font-medium
                transition-all cursor-pointer whitespace-nowrap
                ${isStreaming || !inputValue.trim()
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

          {/* 输入框 */}
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('chatInput.placeholder')}
              className="w-full bg-slate-900/60 border border-slate-700 text-slate-200 rounded-xl py-3 pl-4 pr-12 shadow-sm focus:ring-1 focus:ring-sky-500 focus:border-sky-500 placeholder-slate-500 outline-none transition-all"
              disabled={isStreaming}
            />
            <button
              onClick={() => handleSend()}
              disabled={isStreaming || !inputValue.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-sky-600 text-white rounded-lg flex items-center justify-center hover:bg-sky-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              {isStreaming ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <ArrowRight size={18} />
              )}
            </button>
          </div>
        </div>

        {/* 快捷提示 */}
        <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
          <span>
            {t('chatInput.inputCommand')} <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">/</kbd> {t('chatInput.viewCommands')}
            {' | '}
            <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">/auto</kbd> {t('chatInput.deepResearch')}
          </span>
          <span>
            {t('chatInput.pressEnter')} <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-400">Enter</kbd> {t('chatInput.toSend')}
          </span>
        </div>
      </div>
    </div>
  );
}
