import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  RagConfig,
  WebSearchConfig,
  WebSource,
  DeepResearchDefaults,
  ScholarDownloaderDefaults,
} from '../types';
import type { CollectionInfo } from '../api/ingest';

export const DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS: ScholarDownloaderDefaults = {
  includeAcademia: false,
  assistLlmEnabled: false,
  assistLlmMode: 'ultra-lite',
  browserMode: 'headed',
  strategyOrder: ['direct_download', 'playwright_download', 'browser_lookup', 'sci_hub', 'brightdata', 'anna'],
};

const LEGACY_SCHOLAR_DOWNLOADER_DEFAULT_ORDER: ScholarDownloaderDefaults['strategyOrder'] = [
  'anna',
  'playwright_download',
  'sci_hub',
  'brightdata',
  'direct_download',
];

interface ConfigState {
  // 服务连接
  dbAddress: string;
  dbStatus: 'disconnected' | 'connecting' | 'connected';
  currentCollection: string;
  selectedCollections: string[];
  collections: string[];
  collectionInfos: CollectionInfo[];

  // RAG 配置
  ragConfig: RagConfig;

  // Web 搜索配置
  webSearchConfig: WebSearchConfig;

  // LLM 模型选择
  selectedProvider: string;  // 当前选中的 LLM provider
  selectedModel: string;     // 当前选中的具体模型（空字符串表示使用 provider 默认）

  // Deep Research 默认设置（通过 ⚙ 弹窗配置，跨会话持久化）
  deepResearchDefaults: DeepResearchDefaults;

  // Scholar 下载高级设置默认值（浏览器本地持久化）
  scholarDownloaderDefaults: ScholarDownloaderDefaults;

  // Actions
  setDbAddress: (addr: string) => void;
  setDbStatus: (status: 'disconnected' | 'connecting' | 'connected') => void;
  setCurrentCollection: (name: string) => void;
  setSelectedCollections: (names: string[]) => void;
  toggleCollection: (name: string) => void;
  setCollections: (list: string[]) => void;
  setCollectionInfos: (list: CollectionInfo[]) => void;
  addCollection: (name: string) => void;

  updateRagConfig: (update: Partial<RagConfig>) => void;

  setSelectedProvider: (provider: string) => void;
  setSelectedModel: (model: string) => void;

  updateDeepResearchDefaults: (update: Partial<DeepResearchDefaults>) => void;
  setDeepResearchStepModel: (step: string, value: string) => void;
  updateScholarDownloaderDefaults: (update: Partial<ScholarDownloaderDefaults>) => void;
  resetScholarDownloaderDefaults: (defaults?: Partial<ScholarDownloaderDefaults>) => void;

  setWebSearchEnabled: (enabled: boolean) => void;
  toggleWebSource: (sourceId: string) => void;
  updateWebSourceParam: (
    sourceId: string,
    field: 'topK' | 'threshold',
    value: number
  ) => void;
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
  { id: 'sonar', name: 'Sonar (检索工具)', enabled: false, topK: 1, threshold: 0.5 },  // 独立于预研究；整体一次调用，无数量限制；模型由 agentSonarModel 指定
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
      selectedCollections: ['deepsea_research_v1'],
      collections: ['deepsea_research_v1', 'general_ocean_v2'],
      collectionInfos: [
        { name: 'deepsea_research_v1', count: -1, associated_library_id: null, associated_library_name: null, binding_ready: false },
        { name: 'general_ocean_v2', count: -1, associated_library_id: null, associated_library_name: null, binding_ready: false },
      ],

      ragConfig: {
        enabled: true,  // 默认启用本地 RAG
        localTopK: 45,  // hybrid chat 推荐更高本地召回预算，避免本地候选池过小
        poolScoreThresholds: {
          chat: { main: 0.3, gap: 0, agent: 0.1 },
          research: { main: 0.35, gap: 0.05, agent: 0.15 }
        },
        fusedPoolScoreThreshold: 0.35,  // 合并池分数阈值，仅作用于融合后的最终池
        localThreshold: 0.5,  // 已废弃，兼容持久化
        stepTopK: 10,  // 默认每步保留10条
        writeTopK: 15,  // 默认撰写阶段 Top-K（>= stepTopK * 1.5）
        yearStart: null,
        yearEnd: null,
        enableHippoRAG: false,
        graphTopK: 20,
        enableReranker: true,
        agentMode: 'assist' as const,
        sonarStrength: 'sonar-reasoning-pro' as const,
        agentSonarModel: 'sonar-pro' as const,  // Sonar 检索工具用模型（仅当 Web 来源勾选 Sonar 时生效），与预研究分离
        maxIterations: 2,
        agentDebugMode: false,
        enableGraphicAbstract: false,
        graphicAbstractModel: 'nanobanana 2',
      },

      webSearchConfig: {
        enabled: true,
        sources: defaultWebSources,
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

      scholarDownloaderDefaults: DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS,

      setDbAddress: (addr) => set({ dbAddress: addr }),
      setDbStatus: (status) => set({ dbStatus: status }),
      setCurrentCollection: (name) =>
        set((state) => ({
          currentCollection: name,
          selectedCollections: state.selectedCollections.includes(name)
            ? state.selectedCollections
            : [name],
        })),
      setSelectedCollections: (names) =>
        set({
          selectedCollections: names,
          currentCollection: names[0] ?? '',
        }),
      toggleCollection: (name) =>
        set((state) => {
          const current = state.selectedCollections;
          const next = current.includes(name)
            ? current.filter((n) => n !== name)
            : [...current, name];
          const final = next.length === 0 ? current : next;
          return {
            selectedCollections: final,
            currentCollection: final[0] ?? state.currentCollection,
          };
        }),
      setCollections: (list) => set({ collections: list }),
      setCollectionInfos: (list) => set({ collectionInfos: list }),
      addCollection: (name) =>
        set((state) => ({
          collections: [...state.collections, name],
          collectionInfos: state.collectionInfos.some((c) => c.name === name)
            ? state.collectionInfos
            : [...state.collectionInfos, { name, count: -1, associated_library_id: null, associated_library_name: null, binding_ready: false }],
          currentCollection: name,
          selectedCollections: [name],
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
      updateScholarDownloaderDefaults: (update) =>
        set((state) => ({
          scholarDownloaderDefaults: {
            ...state.scholarDownloaderDefaults,
            ...update,
          },
        })),
      resetScholarDownloaderDefaults: (defaults) =>
        set({
          scholarDownloaderDefaults: {
            ...DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS,
            ...defaults,
            strategyOrder: defaults?.strategyOrder
              ? [...defaults.strategyOrder]
              : [...DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS.strategyOrder],
          },
        }),

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
        selectedCollections: state.selectedCollections,
        collections: state.collections,
        collectionInfos: state.collectionInfos,
        ragConfig: state.ragConfig,
        webSearchConfig: state.webSearchConfig,
        selectedProvider: state.selectedProvider,
        selectedModel: state.selectedModel,
        deepResearchDefaults: state.deepResearchDefaults,
        scholarDownloaderDefaults: state.scholarDownloaderDefaults,
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
            localTopK: persisted.ragConfig?.localTopK ?? currentState.ragConfig.localTopK,
            poolScoreThresholds: persisted.ragConfig?.poolScoreThresholds ?? currentState.ragConfig.poolScoreThresholds,
            fusedPoolScoreThreshold: (persisted.ragConfig as any)?.fusedPoolScoreThreshold ?? 0.35,
            localThreshold: persisted.ragConfig?.localThreshold ?? 0.5,
            stepTopK: persisted.ragConfig?.stepTopK ?? (persisted.ragConfig as any)?.finalTopK ?? 10,
            writeTopK: persisted.ragConfig?.writeTopK
              ?? Math.ceil((persisted.ragConfig?.stepTopK ?? (persisted.ragConfig as any)?.finalTopK ?? 10) * 1.5),
            yearStart: normalizeOptionalYear(persisted.ragConfig?.yearStart),
            yearEnd: normalizeOptionalYear(persisted.ragConfig?.yearEnd),
            enableHippoRAG: persisted.ragConfig?.enableHippoRAG ?? false,
            graphTopK: persisted.ragConfig?.graphTopK ?? 20,
            // Migrate old enableAgent boolean to agentMode tri-state
            agentMode: (persisted.ragConfig as any)?.agentMode
              ?? ((persisted.ragConfig as any)?.enableAgent === false ? 'standard' : 'assist'),
            // Migrate useSonarPrelim + sonarModel -> sonarStrength（与 LLM 选择同源，支持 sonar-deep-research 等 Perplexity 模型 id）
            sonarStrength: ((): import('../types').SonarStrength => {
              const strength = (persisted.ragConfig as any)?.sonarStrength;
              if (strength === 'off') return 'off';
              if (typeof strength === 'string' && strength.trim()) return strength as import('../types').SonarStrength;
              const usePrelim = (persisted.ragConfig as any)?.useSonarPrelim ?? false;
              const model = (persisted.ragConfig as any)?.sonarModel;
              if (!usePrelim) return 'off';
              return (typeof model === 'string' && model.trim() ? model : 'sonar-reasoning-pro') as import('../types').SonarStrength;
            })(),
            maxIterations: (persisted.ragConfig as any)?.maxIterations ?? 2,
            agentDebugMode: persisted.ragConfig?.agentDebugMode ?? false,
            enableGraphicAbstract: persisted.ragConfig?.enableGraphicAbstract ?? false,
            graphicAbstractModel: persisted.ragConfig?.graphicAbstractModel ?? 'nanobanana 2',
          },
          webSearchConfig: {
            ...currentState.webSearchConfig,
            ...(persisted.webSearchConfig || {}),
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
          collectionInfos: persisted.collectionInfos || currentState.collectionInfos,
          selectedCollections: (() => {
            const saved = (persisted as any).selectedCollections;
            if (Array.isArray(saved) && saved.length > 0) return saved as string[];
            // Migrate: if no selectedCollections, derive from currentCollection
            const col = persisted.currentCollection || currentState.currentCollection;
            return col ? [col] : currentState.selectedCollections;
          })(),
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
          scholarDownloaderDefaults: {
            ...currentState.scholarDownloaderDefaults,
            ...(persisted.scholarDownloaderDefaults || {}),
            includeAcademia: persisted.scholarDownloaderDefaults?.includeAcademia ?? currentState.scholarDownloaderDefaults.includeAcademia,
            assistLlmEnabled: persisted.scholarDownloaderDefaults?.assistLlmEnabled ?? currentState.scholarDownloaderDefaults.assistLlmEnabled,
            assistLlmMode: (() => {
              const saved = (persisted.scholarDownloaderDefaults as any)?.assistLlmMode;
              if (saved === 'ultra-lite' || saved === 'lite' || saved === 'auto-upgrade') return saved;
              return currentState.scholarDownloaderDefaults.assistLlmMode;
            })(),
            browserMode: persisted.scholarDownloaderDefaults?.browserMode === 'headless' ? 'headless' : 'headed',
            strategyOrder: (() => {
              const saved = persisted.scholarDownloaderDefaults?.strategyOrder || [];
              const allowed = new Set(DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS.strategyOrder);
              const filtered = saved.filter((id) => allowed.has(id));
              const missing = DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS.strategyOrder.filter((id) => !filtered.includes(id));
              const normalized = filtered.length > 0 ? [...filtered, ...missing] : currentState.scholarDownloaderDefaults.strategyOrder;
              const matchesLegacyDefault = normalized.length === LEGACY_SCHOLAR_DOWNLOADER_DEFAULT_ORDER.length
                && normalized.every((id, index) => id === LEGACY_SCHOLAR_DOWNLOADER_DEFAULT_ORDER[index]);
              return matchesLegacyDefault ? [...DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS.strategyOrder] : normalized;
            })(),
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
          const { currentCollection, selectedCollections } = useConfigStore.getState();
          const validSelected = selectedCollections.filter((n) => names.includes(n));
          const finalSelected = validSelected.length > 0 ? validSelected : [names[0]];
          const finalCurrent = names.includes(currentCollection) ? currentCollection : names[0];
          useConfigStore.setState({
            collections: names,
            collectionInfos: cols,
            dbStatus: 'connected',
            currentCollection: finalCurrent,
            selectedCollections: finalSelected,
          });
        }
      })
      .catch(() => { /* Milvus unavailable; keep persisted values */ });
  });
});
