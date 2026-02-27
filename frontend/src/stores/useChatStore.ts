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

  // 按任务/会话的流式状态（taskId -> { sessionId, status, queuePosition? }）
  streamingTasks: Record<string, { sessionId: string | null; status: string; queuePosition?: number }>;

  // Actions
  setStreamingTask: (taskId: string, sessionId: string | null, status: string, queuePosition?: number) => void;
  clearStreamingTask: (taskId: string) => void;
  setSessionId: (id: string | null) => void;
  setCanvasId: (id: string | null) => void;
  addMessage: (msg: Message) => void;
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

  loadSession: async (sessionId: string) => {
    set({ isLoadingSession: true });
    try {
      const sessionInfo = await getSession(sessionId);
      const messages: Message[] = sessionInfo.turns.map((turn, index) => {
        const sources: Source[] = (turn.sources || []).map((s, sIndex) => ({
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
          role: turn.role as 'user' | 'assistant',
          content: turn.content,
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
