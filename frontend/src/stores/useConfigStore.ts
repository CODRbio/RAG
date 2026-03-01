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
  setContentFetcherMode: (mode: 'auto' | 'force' | 'off') => void;
  toggleSourceSerpapi: (sourceId: string) => void;
  setSerpapiRatio: (value: number) => void;
  setAgentMode: (mode: 'standard' | 'assist' | 'autonomous') => void;
}

const defaultWebSources: WebSource[] = [
  { id: 'tavily', name: 'Tavily API', enabled: true, topK: 5, threshold: 0.5 },
  { id: 'google', name: 'Google Search', enabled: false, topK: 5, threshold: 0.4, useSerpapi: false },
  { id: 'scholar', name: 'Google Scholar', enabled: false, topK: 3, threshold: 0.6, useSerpapi: false },
  { id: 'semantic', name: 'Semantic Scholar', enabled: false, topK: 3, threshold: 0.7 },
  { id: 'ncbi', name: 'NCBI PubMed', enabled: false, topK: 5, threshold: 0.6 },
];

const normalizeOptionalYear = (value: unknown): number | null => {
  if (value === null || value === undefined || value === '') return null;
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1900 || n > 2100) return null;
  return n;
};

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
        stepTopK: 10,  // 默认每步保留10条
        writeTopK: 15,  // 默认撰写阶段 Top-K（>= stepTopK * 1.5）
        yearStart: null,
        yearEnd: null,
        enableHippoRAG: false,
        enableReranker: true,
        agentMode: 'assist' as const,
        maxIterations: 2,
        agentDebugMode: false,
      },

      webSearchConfig: {
        enabled: true,
        sources: defaultWebSources,
        queryOptimizer: true,   // 默认开启
        maxQueriesPerProvider: 3,
        contentFetcherMode: 'auto',  // 全文抓取默认智能模式
        serpapiRatio: 50,
      },

      selectedProvider: 'deepseek',  // 默认 provider
      selectedModel: '',              // 空=使用 provider 默认模型

      deepResearchDefaults: {
        depth: 'comprehensive',
        outputLanguage: 'auto',
        yearStart: null,
        yearEnd: null,
        stepModelStrict: false,
        preliminaryModel: 'sonar::sonar-reasoning-pro',
        questionModel: '',
        skipClaimGeneration: false,
        maxSections: 4,
        gapQueryIntent: 'broad',
        ultra_lite_provider: null as string | null | undefined,
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

      setContentFetcherMode: (mode) =>
        set((state) => ({
          webSearchConfig: { ...state.webSearchConfig, contentFetcherMode: mode },
        })),

      toggleSourceSerpapi: (sourceId) =>
        set((state) => ({
          webSearchConfig: {
            ...state.webSearchConfig,
            sources: state.webSearchConfig.sources.map((s) =>
              s.id === sourceId ? { ...s, useSerpapi: !s.useSerpapi } : s
            ),
          },
        })),

      setSerpapiRatio: (value) => {
        const VALID = [0, 25, 33, 50, 67, 75, 100];
        const snapped = VALID.reduce((best, v) => Math.abs(v - value) < Math.abs(best - value) ? v : best, VALID[3]);
        set((state) => ({
          webSearchConfig: { ...state.webSearchConfig, serpapiRatio: snapped },
        }));
      },

      setAgentMode: (mode) =>
        set((state) => ({
          ragConfig: { ...state.ragConfig, agentMode: mode },
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
            stepTopK: persisted.ragConfig?.stepTopK ?? (persisted.ragConfig as any)?.finalTopK ?? 10,
            writeTopK: persisted.ragConfig?.writeTopK
              ?? Math.ceil((persisted.ragConfig?.stepTopK ?? (persisted.ragConfig as any)?.finalTopK ?? 10) * 1.5),
            yearStart: normalizeOptionalYear(persisted.ragConfig?.yearStart),
            yearEnd: normalizeOptionalYear(persisted.ragConfig?.yearEnd),
            // Migrate old enableAgent boolean to agentMode tri-state
            agentMode: (persisted.ragConfig as any)?.agentMode
              ?? ((persisted.ragConfig as any)?.enableAgent === false ? 'standard' : 'assist'),
            maxIterations: (persisted.ragConfig as any)?.maxIterations ?? 2,
            agentDebugMode: persisted.ragConfig?.agentDebugMode ?? false,
          },
          webSearchConfig: {
            ...currentState.webSearchConfig,
            ...(persisted.webSearchConfig || {}),
            queryOptimizer: persisted.webSearchConfig?.queryOptimizer ?? true,
            maxQueriesPerProvider: persisted.webSearchConfig?.maxQueriesPerProvider ?? 3,
            contentFetcherMode: (persisted.webSearchConfig as any)?.contentFetcherMode ?? ((persisted.webSearchConfig as any)?.enableContentFetcher === true ? 'force' : 'auto'),
            serpapiRatio: (persisted.webSearchConfig as any)?.serpapiRatio ?? 50,
            sources: (() => {
              const saved = (persisted.webSearchConfig?.sources || [])
                .filter((s) => s.id !== 'serpapi')
                .map((s) => {
                  if ((s.id === 'google' || s.id === 'scholar') && s.useSerpapi === undefined) {
                    return { ...s, useSerpapi: false };
                  }
                  return s;
                });
              const savedIds = new Set(saved.map((s) => s.id));
              const missing = defaultWebSources.filter((d) => !savedIds.has(d.id));
              return [...saved, ...missing];
            })(),
          },
          deepResearchDefaults: {
            ...currentState.deepResearchDefaults,
            ...(persisted.deepResearchDefaults || {}),
            gapQueryIntent: ((): 'broad' | 'review_pref' | 'reviews_only' => {
              const raw = (persisted.deepResearchDefaults as any)?.gapQueryIntent;
              return (raw === 'review_pref' || raw === 'reviews_only' || raw === 'broad') ? raw : 'broad';
            })(),
            yearStart: normalizeOptionalYear(persisted.deepResearchDefaults?.yearStart),
            yearEnd: normalizeOptionalYear(persisted.deepResearchDefaults?.yearEnd),
            stepModels: {
              ...currentState.deepResearchDefaults.stepModels,
              ...(persisted.deepResearchDefaults?.stepModels || {}),
            },
            ultra_lite_provider: (persisted.deepResearchDefaults as any)?.ultra_lite_provider ?? currentState.deepResearchDefaults.ultra_lite_provider,
          },
        };
      },
    }
  )
);

// After hydration, ensure any newly added default sources are present.
// This covers cases where the merge callback didn't fire (e.g. HMR).
useConfigStore.persist.onFinishHydration?.(() => {
  const { webSearchConfig } = useConfigStore.getState();
  const filtered = webSearchConfig.sources.filter((s) => s.id !== 'serpapi');
  const existingIds = new Set(filtered.map((s) => s.id));
  const missing = defaultWebSources.filter((d) => !existingIds.has(d.id));
  const needsUpdate = missing.length > 0 || filtered.length !== webSearchConfig.sources.length;
  if (needsUpdate) {
    useConfigStore.setState({
      webSearchConfig: {
        ...webSearchConfig,
        sources: [...filtered, ...missing],
      },
    });
  }

  // Sync collection list from backend so Chat page always has real Milvus collections.
  import('../api/ingest').then(({ listCollections }) => {
    listCollections()
      .then((cols) => {
        const names = cols.map((c) => c.name);
        if (names.length > 0) {
          const { currentCollection } = useConfigStore.getState();
          useConfigStore.setState({
            collections: names,
            dbStatus: 'connected',
            ...(names.includes(currentCollection) ? {} : { currentCollection: names[0] }),
          });
        }
      })
      .catch(() => { /* Milvus unavailable; keep persisted values */ });
  });
});
