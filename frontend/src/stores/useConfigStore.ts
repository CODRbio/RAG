import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { RagConfig, WebSearchConfig, WebSource, DeepResearchDefaults } from '../types';

interface ConfigState {
  // 服务连接
  dbAddress: string;
  dbStatus: 'disconnected' | 'connecting' | 'connected';
  currentCollection: string;
  collections: string[];

  // RAG 配置
  ragConfig: RagConfig;

  // Web 搜索配置
  webSearchConfig: WebSearchConfig;

  // LLM 模型选择
  selectedProvider: string;  // 当前选中的 LLM provider
  selectedModel: string;     // 当前选中的具体模型（空字符串表示使用 provider 默认）

  // Deep Research 默认设置（通过 ⚙ 弹窗配置，跨会话持久化）
  deepResearchDefaults: DeepResearchDefaults;

  // Actions
  setDbAddress: (addr: string) => void;
  setDbStatus: (status: 'disconnected' | 'connecting' | 'connected') => void;
  setCurrentCollection: (name: string) => void;
  setCollections: (list: string[]) => void;
  addCollection: (name: string) => void;

  updateRagConfig: (update: Partial<RagConfig>) => void;

  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;

  updateDeepResearchDefaults: (update: Partial<DeepResearchDefaults>) => void;
  setDeepResearchStepModel: (step: string, value: string) => void;

  setWebSearchEnabled: (enabled: boolean) => void;
  toggleWebSource: (sourceId: string) => void;
  updateWebSourceParam: (
    sourceId: string,
    field: 'topK' | 'threshold',
    value: number
  ) => void;
  setQueryOptimizer: (enabled: boolean) => void;
  setMaxQueriesPerProvider: (value: number) => void;
  setContentFetcherEnabled: (enabled: boolean) => void;
  setAgentEnabled: (enabled: boolean) => void;
}

const defaultWebSources: WebSource[] = [
  { id: 'tavily', name: 'Tavily API', enabled: true, topK: 5, threshold: 0.5 },
  { id: 'google', name: 'Google Search', enabled: false, topK: 5, threshold: 0.4 },
  { id: 'scholar', name: 'Google Scholar', enabled: false, topK: 3, threshold: 0.6 },
  { id: 'semantic', name: 'Semantic Scholar', enabled: false, topK: 3, threshold: 0.7 },
];

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      dbAddress: 'localhost:19530',
      dbStatus: 'disconnected',
      currentCollection: 'deepsea_research_v1',
      collections: ['deepsea_research_v1', 'general_ocean_v2'],

      ragConfig: {
        enabled: true,  // 默认启用本地 RAG
        localTopK: 5,
        localThreshold: 0.5,  // 默认相似度阈值
        finalTopK: 10,  // 默认最终保留10条
        enableHippoRAG: false,
        enableReranker: true,
        enableAgent: true,  // Agent 模式默认开启
      },

      webSearchConfig: {
        enabled: true,
        sources: defaultWebSources,
        queryOptimizer: true,   // 默认开启
        maxQueriesPerProvider: 3,
        enableContentFetcher: false,  // 全文抓取默认关闭
      },

      selectedProvider: 'deepseek',  // 默认 provider
      selectedModel: '',              // 空=使用 provider 默认模型

      deepResearchDefaults: {
        depth: 'comprehensive',
        outputLanguage: 'auto',
        stepModelStrict: false,
        stepModels: {
          scope: 'sonar::sonar-pro',
          plan: '',
          research: '',
          evaluate: '',
          write: '',
          verify: '',
          synthesize: '',
        },
      },

      setDbAddress: (addr) => set({ dbAddress: addr }),
      setDbStatus: (status) => set({ dbStatus: status }),
      setCurrentCollection: (name) => set({ currentCollection: name }),
      setCollections: (list) => set({ collections: list }),
      addCollection: (name) =>
        set((state) => ({
          collections: [...state.collections, name],
          currentCollection: name,
        })),

      updateRagConfig: (update) =>
        set((state) => ({
          ragConfig: { ...state.ragConfig, ...update },
        })),

      setSelectedProvider: (provider) => set({ selectedProvider: provider, selectedModel: '' }),
      setSelectedModel: (model) => set({ selectedModel: model }),

      updateDeepResearchDefaults: (update) =>
        set((state) => ({
          deepResearchDefaults: { ...state.deepResearchDefaults, ...update },
        })),
      setDeepResearchStepModel: (step, value) =>
        set((state) => ({
          deepResearchDefaults: {
            ...state.deepResearchDefaults,
            stepModels: { ...state.deepResearchDefaults.stepModels, [step]: value },
          },
        })),

      setWebSearchEnabled: (enabled) =>
        set((state) => ({
          webSearchConfig: { ...state.webSearchConfig, enabled },
        })),

      toggleWebSource: (sourceId) =>
        set((state) => ({
          webSearchConfig: {
            ...state.webSearchConfig,
            sources: state.webSearchConfig.sources.map((s) =>
              s.id === sourceId ? { ...s, enabled: !s.enabled } : s
            ),
          },
        })),

      updateWebSourceParam: (sourceId, field, value) =>
        set((state) => ({
          webSearchConfig: {
            ...state.webSearchConfig,
            sources: state.webSearchConfig.sources.map((s) =>
              s.id === sourceId ? { ...s, [field]: value } : s
            ),
          },
        })),

      setQueryOptimizer: (enabled) =>
        set((state) => ({
          webSearchConfig: { ...state.webSearchConfig, queryOptimizer: enabled },
        })),

      setMaxQueriesPerProvider: (value) =>
        set((state) => ({
          webSearchConfig: {
            ...state.webSearchConfig,
            maxQueriesPerProvider: Math.min(Math.max(1, value), 5),
          },
        })),

      setContentFetcherEnabled: (enabled) =>
        set((state) => ({
          webSearchConfig: { ...state.webSearchConfig, enableContentFetcher: enabled },
        })),

      setAgentEnabled: (enabled) =>
        set((state) => ({
          ragConfig: { ...state.ragConfig, enableAgent: enabled },
        })),
    }),
    {
      name: 'config-storage',
      partialize: (state) => ({
        dbAddress: state.dbAddress,
        currentCollection: state.currentCollection,
        collections: state.collections,
        ragConfig: state.ragConfig,
        webSearchConfig: state.webSearchConfig,
        selectedProvider: state.selectedProvider,
        selectedModel: state.selectedModel,
        deepResearchDefaults: state.deepResearchDefaults,
      }),
      // 合并存储数据与默认值，确保新增字段有默认值
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<ConfigState>;
        return {
          ...currentState,
          ...persisted,
          ragConfig: {
            ...currentState.ragConfig,
            ...(persisted.ragConfig || {}),
            // 确保新字段有默认值
            enabled: persisted.ragConfig?.enabled ?? true,
            localThreshold: persisted.ragConfig?.localThreshold ?? 0.5,
            finalTopK: persisted.ragConfig?.finalTopK ?? 10,
            enableAgent: persisted.ragConfig?.enableAgent ?? true,
          },
          webSearchConfig: {
            ...currentState.webSearchConfig,
            ...(persisted.webSearchConfig || {}),
            // 确保新字段有默认值
            queryOptimizer: persisted.webSearchConfig?.queryOptimizer ?? true,
            maxQueriesPerProvider: persisted.webSearchConfig?.maxQueriesPerProvider ?? 3,
            enableContentFetcher: persisted.webSearchConfig?.enableContentFetcher ?? false,
          },
          deepResearchDefaults: {
            ...currentState.deepResearchDefaults,
            ...(persisted.deepResearchDefaults || {}),
            stepModels: {
              ...currentState.deepResearchDefaults.stepModels,
              ...(persisted.deepResearchDefaults?.stepModels || {}),
            },
          },
        };
      },
    }
  )
);
