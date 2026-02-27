import { create } from 'zustand';
import type { Message, WorkflowStep, EvidenceSummary, Source, ClarifyQuestion, ResearchDashboardData, ToolTraceItem, AgentDebugData } from '../types';
import { getSession } from '../api/chat';

interface ChatState {
  sessionId: string | null;
  canvasId: string | null;
  messages: Message[];
  workflowStep: WorkflowStep;
  isStreaming: boolean;
  lastEvidenceSummary: EvidenceSummary | null;
  isLoadingSession: boolean;

  // Deep Research 状态
  deepResearchActive: boolean;          // 是否正在进行 Deep Research
  deepResearchTopic: string;            // Deep Research 主题
  clarificationQuestions: ClarifyQuestion[]; // 澄清问题列表
  showDeepResearchDialog: boolean;      // 是否显示澄清对话框
  researchDashboard: ResearchDashboardData | null;  // 研究进度仪表盘
  toolTrace: ToolTraceItem[] | null;                // Agent 工具调用轨迹

  // 命令面板
  showCommandPalette: boolean;

  /** 查询与本地库范围不符时：是否显示「本会话不用本地库 / 仍使用当前库」选择 */
  pendingLocalDbChoice: boolean;
  pendingLocalDbChoiceMessageId: string | null;
  pendingLocalDbChoiceOriginalMessage: string;
  /** 弹出时的时间戳（ms），用于弹窗显示剩余倒计时 */
  pendingLocalDbChoiceStartedAt: number | null;
  /** 用户选择或 1 分钟超时后执行的 handler，由 ChatInput 注入 */
  localDbChoiceHandler: ((choice: 'no_local' | 'use') => Promise<void>) | null;
  localDbChoiceTimeoutId: ReturnType<typeof setTimeout> | null;

  // 按任务/会话的流式状态（taskId -> { sessionId, status, queuePosition? }）
  streamingTasks: Record<string, { sessionId: string | null; status: string; queuePosition?: number }>;

  // Actions
  setStreamingTask: (taskId: string, sessionId: string | null, status: string, queuePosition?: number) => void;
  clearStreamingTask: (taskId: string) => void;
  setSessionId: (id: string | null) => void;
  setCanvasId: (id: string | null) => void;
  addMessage: (msg: Message) => void;
  updateMessageById: (id: string, content: string) => void;
  appendToMessageById: (id: string, delta: string) => void;
  setMessageSourcesById: (id: string, sources: Message['sources'], providerStats?: Message['providerStats']) => void;
  setMessageAgentDebugById: (id: string, debug: AgentDebugData) => void;
  updateLastMessage: (content: string) => void;
  appendToLastMessage: (delta: string) => void;
  setLastMessageSources: (sources: Message['sources'], providerStats?: Message['providerStats']) => void;
  clearMessages: () => void;
  setWorkflowStep: (step: WorkflowStep) => void;
  setIsStreaming: (streaming: boolean) => void;
  setLastEvidenceSummary: (summary: EvidenceSummary | null) => void;
  newChat: () => void;
  loadSession: (sessionId: string) => Promise<void>;

  // Deep Research Actions
  setDeepResearchActive: (active: boolean) => void;
  setDeepResearchTopic: (topic: string) => void;
  setClarificationQuestions: (questions: ClarifyQuestion[]) => void;
  setShowDeepResearchDialog: (show: boolean) => void;
  setShowCommandPalette: (show: boolean) => void;
  setResearchDashboard: (dashboard: ResearchDashboardData | null) => void;
  setToolTrace: (trace: ToolTraceItem[] | null) => void;
  setLastMessageAgentDebug: (debug: AgentDebugData) => void;

  setPendingLocalDbChoice: (show: boolean, messageId?: string | null, originalMessage?: string, startedAt?: number | null) => void;
  clearPendingLocalDbChoice: () => void;
  setLocalDbChoiceHandler: (handler: ((choice: 'no_local' | 'use') => Promise<void>) | null) => void;
  setLocalDbChoiceTimeoutId: (id: ReturnType<typeof setTimeout> | null) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  sessionId: null,
  canvasId: null,
  messages: [],
  workflowStep: 'idle',
  isStreaming: false,
  lastEvidenceSummary: null,
  isLoadingSession: false,

  // Deep Research 初始状态
  deepResearchActive: false,
  deepResearchTopic: '',
  clarificationQuestions: [],
  showDeepResearchDialog: false,
  researchDashboard: null,
  toolTrace: null,
  showCommandPalette: false,
  pendingLocalDbChoice: false,
  pendingLocalDbChoiceMessageId: null,
  pendingLocalDbChoiceOriginalMessage: '',
  pendingLocalDbChoiceStartedAt: null,
  localDbChoiceHandler: null,
  localDbChoiceTimeoutId: null,
  streamingTasks: {},

  setStreamingTask: (taskId, sessionId, status, queuePosition) =>
    set((state) => ({
      streamingTasks: {
        ...state.streamingTasks,
        [taskId]: { sessionId, status, ...(queuePosition !== undefined ? { queuePosition } : {}) },
      },
    })),
  clearStreamingTask: (taskId) =>
    set((state) => {
      const next = { ...state.streamingTasks };
      delete next[taskId];
      return { streamingTasks: next };
    }),

  setSessionId: (id) => set({ sessionId: id }),
  setCanvasId: (id) => set({ canvasId: id }),

  addMessage: (msg) =>
    set((state) => ({
      messages: [...state.messages, { ...msg, timestamp: new Date().toISOString() }],
    })),
  updateMessageById: (id, content) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === id ? { ...m, content } : m)),
    })),
  appendToMessageById: (id, delta) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === id ? { ...m, content: (m.content || '') + delta } : m)),
    })),
  setMessageSourcesById: (id, sources, providerStats) =>
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, sources, ...(providerStats ? { providerStats } : {}) } : m
      ),
    })),
  setMessageAgentDebugById: (id, debug) =>
    set((state) => ({
      messages: state.messages.map((m) => (m.id === id ? { ...m, agentDebug: debug } : m)),
    })),

  updateLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content,
        };
      }
      return { messages };
    }),

  appendToLastMessage: (delta) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content: messages[messages.length - 1].content + delta,
        };
      }
      return { messages };
    }),

  setLastMessageSources: (sources, providerStats) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          sources,
          ...(providerStats ? { providerStats } : {}),
        };
      }
      return { messages };
    }),

  clearMessages: () => set({ messages: [] }),

  setWorkflowStep: (step) => set({ workflowStep: step }),
  setIsStreaming: (streaming) => set({ isStreaming: streaming }),
  setLastEvidenceSummary: (summary) => set({ lastEvidenceSummary: summary }),

  newChat: () =>
    set(() => ({
      sessionId: null,
      canvasId: null,
      messages: [],
      workflowStep: 'idle',
      lastEvidenceSummary: null,
      // 新对话 = 全新页面；旧 Deep Research 任务仍在后台运行，
      // 加载旧会话时会重新恢复 dashboard。
      deepResearchActive: false,
      deepResearchTopic: '',
      clarificationQuestions: [],
      showDeepResearchDialog: false,
      researchDashboard: null,
      toolTrace: null,
    })),

  // Deep Research Actions
  setDeepResearchActive: (active) => set({ deepResearchActive: active }),
  setDeepResearchTopic: (topic) => set({ deepResearchTopic: topic }),
  setClarificationQuestions: (questions) => set({ clarificationQuestions: questions }),
  setShowDeepResearchDialog: (show) => set({ showDeepResearchDialog: show }),
  setShowCommandPalette: (show) => set({ showCommandPalette: show }),
  setResearchDashboard: (dashboard) => set({ researchDashboard: dashboard }),
  setToolTrace: (trace) => set({ toolTrace: trace }),
  setLastMessageAgentDebug: (debug) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          agentDebug: debug,
        };
      }
      return { messages };
    }),

  setPendingLocalDbChoice: (show, messageId, originalMessage, startedAt) =>
    set({
      pendingLocalDbChoice: show,
      pendingLocalDbChoiceMessageId: messageId ?? null,
      pendingLocalDbChoiceOriginalMessage: originalMessage ?? '',
      pendingLocalDbChoiceStartedAt: startedAt ?? (show ? Date.now() : null),
    }),
  clearPendingLocalDbChoice: () =>
    set({
      pendingLocalDbChoice: false,
      pendingLocalDbChoiceMessageId: null,
      pendingLocalDbChoiceOriginalMessage: '',
      pendingLocalDbChoiceStartedAt: null,
      localDbChoiceHandler: null,
    }),
  setLocalDbChoiceHandler: (handler) => set({ localDbChoiceHandler: handler }),
  setLocalDbChoiceTimeoutId: (id) => set({ localDbChoiceTimeoutId: id }),

  loadSession: async (sessionId: string) => {
    set({ isLoadingSession: true });
    try {
      const sessionInfo = await getSession(sessionId);
      const turns = Array.isArray(sessionInfo.turns) ? sessionInfo.turns : [];
      const messages: Message[] = turns.map((turn, index) => {
        const sources: Source[] = (Array.isArray(turn.sources) ? turn.sources : []).map((s, sIndex) => ({
          id: s.cite_key || `${sessionId}-${index}-${sIndex}`,
          cite_key: s.cite_key || '',
          title: s.title || '',
          authors: s.authors || [],
          year: s.year,
          doc_id: s.doc_id,
          url: s.url,
          doi: s.doi,
          bbox: s.bbox,
          page_num: s.page_num,
          type: s.url ? 'web' : 'local',
        }));
        return {
          id: `${sessionId}-${index}`,
          role: (turn.role === 'assistant' ? 'assistant' : 'user') as 'user' | 'assistant',
          content: typeof turn.content === 'string' ? turn.content : '',
          timestamp: new Date().toISOString(),
          sources,
        };
      });
      set({
        sessionId: sessionInfo.session_id,
        canvasId: sessionInfo.canvas_id || null,
        messages,
        workflowStep: 'idle',
        lastEvidenceSummary: null,
        deepResearchActive: false,
        deepResearchTopic: '',
        clarificationQuestions: [],
        showDeepResearchDialog: false,
        researchDashboard: sessionInfo.research_dashboard || null,
        showCommandPalette: false,
        isLoadingSession: false,
      });
    } catch (error) {
      set({ isLoadingSession: false });
      throw error;
    }
  },
}));
