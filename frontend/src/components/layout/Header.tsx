import {
  MoreHorizontal,
  MessageSquare,
  UploadCloud,
  Users,
  GitBranch,
  MessageSquarePlus,
  History,
  PlugZap,
  PanelRightClose,
  PanelRightOpen,
  Sparkles,
  Network,
  GitCompareArrows,
  Globe,
  Telescope,
  RefreshCw,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuthStore, useConfigStore, useChatStore, useUIStore, useToastStore } from '../../stores';
import { checkHealth } from '../../api/health';
import { listLLMProviders, listAllLiveModels, type LLMProviderInfo } from '../../api/ingest';

const LLM_PROVIDER_CACHE_KEY = 'llm_provider_cache_v1';
const LLM_PROVIDER_BOOTSTRAP_FLAG = 'llm_provider_bootstrap_refreshed_v1';

type LlmProviderCachePayload = {
  ts: number;
  providers: LLMProviderInfo[];
};

type HeaderTabId = 'chat' | 'ingest' | 'users' | 'graph' | 'compare';

interface HeaderTabConfig {
  id: HeaderTabId;
  labelKey: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  activeClass: string;
}

const GROUP_COLORS: Record<string, string> = {
  openai: 'text-emerald-400',
  deepseek: 'text-sky-400',
  claude: 'text-amber-400',
  gemini: 'text-purple-400',
  kimi: 'text-cyan-400',
  qwen: 'text-rose-400',
  perplexity: 'text-teal-400',
  sonar: 'text-teal-400',
};

function baseProviderName(id: string): string {
  return id.split('-')[0].toLowerCase();
}

function groupLabel(id: string): string {
  const base = baseProviderName(id);
  const labels: Record<string, string> = {
    openai: 'OpenAI',
    deepseek: 'DeepSeek',
    claude: 'Claude',
    gemini: 'Gemini',
    kimi: 'Kimi',
    qwen: 'Qwen',
    perplexity: 'Perplexity',
    sonar: 'Perplexity',
  };
  return labels[base] || id;
}

function providerColor(id: string): string {
  const base = baseProviderName(id);
  return GROUP_COLORS[base] || 'text-slate-400';
}

/** "openai-thinking" → "OpenAI · thinking", "gemini-vision" → "Gemini · vision" */
function providerOptionLabel(id: string): string {
  const base = groupLabel(id);
  const suffix = id.split('-').slice(1).join('-');
  return suffix ? `${base} · ${suffix}` : base;
}

// ── Model filter combobox ──────────────────────────────────────────────────
interface ModelComboboxProps {
  models: string[];
  value: string;
  defaultModel: string;
  onChange: (model: string) => void;
  compact: boolean;
}

function ModelCombobox({ models, value, defaultModel, onChange, compact }: ModelComboboxProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = filter
    ? models.filter((m) => m.toLowerCase().includes(filter.toLowerCase()))
    : models;

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setFilter('');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleSelect = (model: string) => {
    onChange(model);
    setOpen(false);
    setFilter('');
  };

  const displayValue = open ? filter : (value || `${defaultModel} (default)`);

  return (
    <div ref={containerRef} className="relative">
      <input
        ref={inputRef}
        value={displayValue}
        onChange={(e) => setFilter(e.target.value)}
        onFocus={() => { setOpen(true); setFilter(''); }}
        placeholder="filter models..."
        className={`bg-transparent text-xs font-medium text-slate-200 focus:outline-none placeholder:text-slate-500 ${compact ? 'w-[120px]' : 'w-[180px]'}`}
      />
      {open && (
        <div className="absolute top-[calc(100%+6px)] left-0 min-w-[220px] max-h-52 overflow-y-auto bg-slate-900/98 border border-slate-600/80 rounded-lg shadow-2xl z-[100] py-1 backdrop-blur-sm">
          {/* Default option */}
          <button
            onMouseDown={(e) => { e.preventDefault(); handleSelect(''); }}
            className={`w-full px-3 py-1.5 text-left text-[11px] flex items-center gap-1.5 transition-colors ${
              value === '' ? 'text-sky-400 bg-sky-900/30' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            <span className="text-[9px] px-1 py-0.5 rounded border border-slate-600 text-slate-500">DEF</span>
            {defaultModel}
          </button>
          <div className="border-t border-slate-700/60 my-0.5" />
          {filtered.length === 0 ? (
            <div className="px-3 py-2 text-[11px] text-slate-500">no models match</div>
          ) : (
            filtered.map((m) => (
              <button
                key={m}
                onMouseDown={(e) => { e.preventDefault(); handleSelect(m); }}
                className={`w-full px-3 py-1.5 text-left text-[11px] transition-colors truncate ${
                  m === value ? 'text-sky-400 bg-sky-900/30' : 'text-slate-200 hover:bg-slate-800'
                }`}
                title={m}
              >
                {m}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

/** Map raw API error text to a concise user-facing hint. */
function _classifyApiError(error: string): string {
  const e = error.toLowerCase();
  if (e.includes('401') || e.includes('invalid authentication') || e.includes('unauthorized'))
    return 'API Key 无效或已过期 (401)';
  if (e.includes('403') || e.includes('forbidden'))
    return 'API Key 权限不足 (403)';
  if (e.includes('api_key_invalid') || e.includes('key not found') || e.includes('400'))
    return 'API Key 未找到或无效 (400)';
  if (e.includes('429') || e.includes('rate limit') || e.includes('quota'))
    return '请求频率超限，稍后重试 (429)';
  if (e.includes('timeout'))
    return '请求超时';
  if (e.includes('connection') || e.includes('network'))
    return '网络连接失败';
  return error.slice(0, 80);
}

export function Header() {
  const { t, i18n } = useTranslation();
  const currentLang = i18n.language?.startsWith('en') ? 'en' : 'zh';
  const headerRef = useRef<HTMLElement>(null);
  const moreMenuRef = useRef<HTMLDivElement>(null);
  const [compactHeader, setCompactHeader] = useState(false);
  const [headerWidth, setHeaderWidth] = useState(0);
  const [showMoreTabs, setShowMoreTabs] = useState(false);
  const user = useAuthStore((s) => s.user);
  const { dbStatus, setDbStatus, selectedProvider, setSelectedProvider, selectedModel, setSelectedModel } = useConfigStore();
  const { workflowStep, deepResearchActive, setShowDeepResearchDialog, newChat } = useChatStore();
  const addToast = useToastStore((s) => s.addToast);
  const [hasArchivedJob, setHasArchivedJob] = useState(false);
  const [llmProviders, setLlmProviders] = useState<LLMProviderInfo[]>([]);
  const [isRefreshingModels, setIsRefreshingModels] = useState(false);
  const loadSeqRef = useRef(0);

  const readProviderCache = useCallback((): LLMProviderInfo[] | null => {
    try {
      const raw = localStorage.getItem(LLM_PROVIDER_CACHE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as LlmProviderCachePayload;
      if (!Array.isArray(parsed?.providers) || parsed.providers.length === 0) return null;
      return parsed.providers;
    } catch {
      return null;
    }
  }, []);

  const writeProviderCache = useCallback((providers: LLMProviderInfo[]) => {
    try {
      const payload: LlmProviderCachePayload = { ts: Date.now(), providers };
      localStorage.setItem(LLM_PROVIDER_CACHE_KEY, JSON.stringify(payload));
    } catch {
      // ignore cache write failures
    }
  }, []);

  const loadProviders = useCallback(async (opts?: { forceLive?: boolean; toastOnFinish?: boolean }) => {
    const seq = ++loadSeqRef.current;
    if (opts?.forceLive) {
      setIsRefreshingModels(true);
    }
    try {
      const data = await listLLMProviders();
      if (!data.providers?.length) return;
      if (seq !== loadSeqRef.current) return;

      let mergedProviders = data.providers;
      setLlmProviders((prev) => {
        if (prev.length === 0) return data.providers;
        const liveModelMap = new Map<string, string[]>();
        for (const p of prev) {
          if (p.models.length > 1) liveModelMap.set(p.id, p.models);
        }
        if (liveModelMap.size === 0) return data.providers;
        mergedProviders = data.providers.map((p) => {
          const kept = liveModelMap.get(p.id);
          return kept ? { ...p, models: kept } : p;
        });
        return mergedProviders;
      });
      writeProviderCache(mergedProviders);

      const live = await listAllLiveModels(Boolean(opts?.forceLive));
      if (seq !== loadSeqRef.current) return;
      const platformModels = live.platforms ?? {};

      const apiErrors: string[] = [];
      for (const [platform, entry] of Object.entries(platformModels)) {
        if (entry.error) {
          const hint = _classifyApiError(entry.error);
          apiErrors.push(`${platform}: ${hint}`);
        }
      }
      if (apiErrors.length > 0) {
        addToast(
          `部分平台模型列表拉取失败，已用配置默认值代替。\n${apiErrors.join('\n')}`,
          'warning',
        );
      }

      setLlmProviders((prev) => {
        const merged = prev.map((p) => {
          const platform = p.platform ?? p.id.split('-')[0];
          const liveEntry = platformModels[platform];
          if (liveEntry?.models?.length) {
            return { ...p, models: liveEntry.models };
          }
          return p;
        });
        writeProviderCache(merged);
        return merged;
      });

      if (opts?.toastOnFinish) {
        addToast(currentLang === 'zh' ? '模型列表已刷新' : 'Model list refreshed', 'success');
      }
    } catch (err) {
      console.warn('[Header] load providers failed:', err);
      if (opts?.toastOnFinish) {
        addToast(currentLang === 'zh' ? '模型列表刷新失败，已保留缓存' : 'Model refresh failed, cached list kept', 'warning');
      }
    } finally {
      if (opts?.forceLive) {
        setIsRefreshingModels(false);
      }
    }
  }, [addToast, currentLang, writeProviderCache]);

  useEffect(() => {
    // 先使用上次缓存，避免 Header 首次展示卡在网络请求上
    const cached = readProviderCache();
    if (cached?.length) {
      setLlmProviders(cached);
    }

    // 每次浏览器启动（session）首次进入页面时，后台强制刷新一次
    let shouldBootstrapRefresh = true;
    try {
      shouldBootstrapRefresh = sessionStorage.getItem(LLM_PROVIDER_BOOTSTRAP_FLAG) !== '1';
    } catch {
      shouldBootstrapRefresh = true;
    }
    if (shouldBootstrapRefresh) {
      loadProviders({ forceLive: true });
      try {
        sessionStorage.setItem(LLM_PROVIDER_BOOTSTRAP_FLAG, '1');
      } catch {
        // ignore sessionStorage errors
      }
    } else if (!cached?.length) {
      // 无缓存时兜底拉取一次（走后端缓存）
      loadProviders();
    }
  }, [loadProviders, readProviderCache]);


  useEffect(() => {
    try {
      const archived = localStorage.getItem('deep_research_archived_job_ids');
      if (archived) {
        const parsed = JSON.parse(archived);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setHasArchivedJob(true);
        }
      }
    } catch {
      // ignore
    }
  }, [deepResearchActive]); // Check when deepResearchActive changes (e.g. finishes)
  const {
    activeTab,
    setActiveTab,
    toggleSidebar,
    isCanvasOpen,
    toggleCanvas,
    isHistoryOpen,
    toggleHistory,
  } = useUIStore();

  const toggleLanguage = () => {
    const next = currentLang === 'zh' ? 'en' : 'zh';
    i18n.changeLanguage(next);
  };

  const handleTabChange = (tab: HeaderTabId) => {
    setActiveTab(tab);
    setShowMoreTabs(false);
  };

  const handleConnect = async () => {
    setDbStatus('connecting');
    addToast(t('header.connectService'), 'info');
    try {
      await checkHealth();
      setDbStatus('connected');
      addToast(t('header.serviceConnected'), 'success');
    } catch {
      setDbStatus('disconnected');
      addToast(t('header.connectFailed'), 'error');
    }
  };

  const handleRefresh = async () => {
    if (dbStatus !== 'connected') return;
    addToast(t('header.syncingStatus'), 'info');
    try {
      await checkHealth();
      addToast(t('header.syncOk'), 'success');
    } catch {
      setDbStatus('disconnected');
      addToast(t('header.disconnectedStatus'), 'error');
    }
  };

  const handleNewChat = () => {
    newChat();
    addToast(t('header.newChatCreated'), 'success');
  };

  useEffect(() => {
    const measure = () => {
      const width = headerRef.current?.clientWidth || 0;
      setHeaderWidth(width);
      setCompactHeader(width > 0 && width < 1320);
    };

    measure();
    const ro = new ResizeObserver(measure);
    if (headerRef.current) ro.observe(headerRef.current);
    window.addEventListener('resize', measure);

    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  useEffect(() => {
    if (!showMoreTabs) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (moreMenuRef.current && !moreMenuRef.current.contains(e.target as Node)) {
        setShowMoreTabs(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showMoreTabs]);

  const tabConfigs: HeaderTabConfig[] = [
    {
      id: 'chat',
      labelKey: 'header.chat',
      icon: MessageSquare,
      activeClass: 'border-sky-400 text-sky-400 shadow-[0_4px_12px_-4px_rgba(56,189,248,0.5)]',
    },
    {
      id: 'ingest',
      labelKey: 'header.ingest',
      icon: UploadCloud,
      activeClass: 'border-sky-400 text-sky-400 shadow-[0_4px_12px_-4px_rgba(56,189,248,0.5)]',
    },
    {
      id: 'graph',
      labelKey: 'header.graph',
      icon: Network,
      activeClass: 'border-emerald-400 text-emerald-400 shadow-[0_4px_12px_-4px_rgba(52,211,153,0.5)]',
    },
    {
      id: 'compare',
      labelKey: 'header.compare',
      icon: GitCompareArrows,
      activeClass: 'border-amber-400 text-amber-400 shadow-[0_4px_12px_-4px_rgba(251,191,36,0.5)]',
    },
  ];
  if (user?.role === 'admin') {
    tabConfigs.push({
      id: 'users',
      labelKey: 'header.users',
      icon: Users,
      activeClass: 'border-purple-400 text-purple-400 shadow-[0_4px_12px_-4px_rgba(192,132,252,0.5)]',
    });
  }

  const maxVisibleTabs = headerWidth >= 1320
    ? tabConfigs.length
    : headerWidth >= 1200
      ? Math.min(tabConfigs.length, 4)
      : headerWidth >= 1040
        ? Math.min(tabConfigs.length, 3)
        : Math.min(tabConfigs.length, 2);

  const tabPriority: HeaderTabId[] = ['chat', 'ingest', 'graph', 'compare', 'users'];
  const visibleTabIds: HeaderTabId[] = [];
  const addVisible = (id: HeaderTabId) => {
    if (!tabConfigs.some((tab) => tab.id === id)) return;
    if (!visibleTabIds.includes(id) && visibleTabIds.length < maxVisibleTabs) {
      visibleTabIds.push(id);
    }
  };
  addVisible('chat');
  addVisible(activeTab as HeaderTabId);
  tabPriority.forEach(addVisible);

  const visibleTabs = tabConfigs.filter((tab) => visibleTabIds.includes(tab.id));
  const hiddenTabs = tabConfigs.filter((tab) => !visibleTabIds.includes(tab.id));
  const showCompactRightText = headerWidth >= 1180;
  const onlineLabel = currentLang === 'zh' ? '在线' : 'Online';
  const offlineLabel = currentLang === 'zh' ? '离线' : 'Offline';
  const canvasLabel = currentLang === 'zh' ? '画布' : 'Canvas';

  const renderTabButton = (tab: HeaderTabConfig) => {
    const Icon = tab.icon;
    const isActive = activeTab === tab.id;
    return (
      <button
        key={tab.id}
        onClick={() => handleTabChange(tab.id)}
        className={`h-16 flex items-center ${compactHeader ? 'gap-1 px-1.5' : 'gap-2 px-1'} border-b-2 font-medium text-sm transition-all focus:outline-none whitespace-nowrap ${
          isActive
            ? tab.activeClass
            : 'border-transparent text-slate-400 hover:text-slate-200'
        }`}
        title={t(tab.labelKey)}
      >
        <Icon size={18} />
        <span className="text-[11px]">{t(tab.labelKey)}</span>
      </button>
    );
  };

  return (
    <header ref={headerRef} className="glass-header px-4 h-16 flex items-center justify-between gap-3 flex-shrink-0 z-30 transition-colors duration-300 overflow-visible">
      <div className="flex gap-3 min-w-0 items-center">
        <button
          onClick={toggleSidebar}
          className="text-slate-400 hover:text-sky-400 p-1 transition-colors"
          title={currentLang === 'zh' ? '切换侧边栏' : 'Toggle Sidebar'}
        >
          <MoreHorizontal size={20} />
        </button>
        {!compactHeader && <div className="h-8 w-[1px] bg-slate-700/50"></div>}

        {visibleTabs.map(renderTabButton)}

        {hiddenTabs.length > 0 && (
          <div ref={moreMenuRef} className="relative">
            <button
              onClick={() => setShowMoreTabs((prev) => !prev)}
              className={`h-16 flex items-center gap-1 px-1.5 border-b-2 font-medium text-sm transition-all focus:outline-none whitespace-nowrap ${
                hiddenTabs.some((tab) => tab.id === activeTab)
                  ? 'border-sky-400 text-sky-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
              title={currentLang === 'zh' ? '更多' : 'More'}
            >
              <MoreHorizontal size={18} />
            </button>
            {showMoreTabs && (
              <div className="absolute top-[56px] left-0 min-w-[170px] bg-slate-900/95 border border-slate-700 rounded-lg shadow-xl py-1.5 z-[60]">
                {hiddenTabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={`more-${tab.id}`}
                      onClick={() => handleTabChange(tab.id)}
                      className={`w-full px-3 py-1.5 text-xs flex items-center gap-2 transition-colors ${
                        isActive
                          ? 'text-sky-300 bg-sky-900/40'
                          : 'text-slate-300 hover:bg-slate-800'
                      }`}
                      title={t(tab.labelKey)}
                    >
                      <Icon size={14} />
                      {t(tab.labelKey)}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Center: Model Selector + Workflow */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Two-step: Provider → Model */}
        <div className={`flex items-center gap-1.5 ${compactHeader ? 'px-2 py-1' : 'px-2.5 py-1'} bg-slate-800/60 rounded-lg border border-slate-700/60 backdrop-blur-sm shadow-sm hover:border-sky-500/30 transition-all`}>
          <Sparkles size={13} className={providerColor(selectedProvider)} />

          {/* Step 1: Provider selector */}
          <select
            value={selectedProvider}
            onChange={(e) => {
              setSelectedProvider(e.target.value);
              setSelectedModel('');
            }}
            className={`bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer appearance-none pr-3 ${compactHeader ? 'max-w-[90px]' : 'max-w-[130px]'}`}
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 0px center',
            }}
          >
            {Array.from(
              llmProviders.reduce((acc, p) => {
                const g = groupLabel(p.id);
                if (!acc.has(g)) acc.set(g, []);
                acc.get(g)!.push(p);
                return acc;
              }, new Map<string, typeof llmProviders>()),
            ).map(([group, providers]) => (
              <optgroup key={group} label={group} className="bg-slate-800">
                {providers.map((p) => (
                  <option key={p.id} value={p.id} className="bg-slate-800 text-slate-200">
                    {providerOptionLabel(p.id)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>

          {/* Divider */}
          <span className="text-slate-600 text-xs select-none">/</span>

          {/* Step 2: Model combobox with filter */}
          {(() => {
            const currentProvider = llmProviders.find((p) => p.id === selectedProvider);
            const models = currentProvider?.models ?? [];
            const defaultModel = currentProvider?.default_model ?? '';
            return (
              <ModelCombobox
                models={models}
                value={selectedModel}
                defaultModel={defaultModel}
                onChange={setSelectedModel}
                compact={compactHeader}
              />
            );
          })()}

          <button
            onClick={() => loadProviders({ forceLive: true, toastOnFinish: true })}
            disabled={isRefreshingModels}
            className="text-slate-400 hover:text-sky-300 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
            title={currentLang === 'zh' ? '刷新模型列表' : 'Refresh model list'}
          >
            <RefreshCw size={12} className={isRefreshingModels ? 'animate-spin' : ''} />
          </button>
        </div>

        {!compactHeader && workflowStep !== 'idle' && (
          <div className="flex items-center gap-2 px-4 py-1.5 bg-sky-900/30 text-sky-400 rounded-full text-xs font-bold border border-sky-500/30 animate-pulse-glow">
            <GitBranch size={14} />
            <span className="uppercase tracking-wide">{t('header.workflow')}: {workflowStep}</span>
          </div>
        )}
        {!compactHeader && deepResearchActive ? (
          <button
            onClick={() => setShowDeepResearchDialog(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-indigo-900/30 text-indigo-400 rounded-full text-xs font-bold border border-indigo-500/30 hover:bg-indigo-900/50 transition-colors"
            title={currentLang === 'zh' ? '打开 Deep Research 面板' : 'Open Deep Research Panel'}
          >
            <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-pulse" />
            Deep Research Running
          </button>
        ) : (!compactHeader && hasArchivedJob && !deepResearchActive && (
          <button
            onClick={() => setShowDeepResearchDialog(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-900/10 text-indigo-300 rounded-full text-xs font-medium border border-indigo-500/20 hover:bg-indigo-900/30 transition-colors"
            title={currentLang === 'zh' ? '找回并继续已完成的 Deep Research 项目' : 'Resume completed Deep Research project'}
          >
            <Telescope size={14} />
            <span className="hidden sm:inline">{currentLang === 'zh' ? '找回 Deep Research' : 'Resume Deep Research'}</span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {activeTab === 'chat' && dbStatus === 'connected' && (
          <>
            <button
              onClick={handleNewChat}
              className={`flex items-center ${compactHeader ? 'gap-1 px-2 py-1.5' : 'gap-2 px-3 py-1.5'} bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 rounded-lg text-xs font-medium transition-colors border border-slate-700/60 hover:border-sky-500/30 hover:text-sky-300`}
              title={t('header.newChat')}
            >
              <MessageSquarePlus size={14} />
              <span className="text-[11px]">{showCompactRightText || !compactHeader ? t('header.newChat') : (currentLang === 'zh' ? '新建' : 'New')}</span>
            </button>
            <button
              onClick={toggleHistory}
              className={`p-2 rounded-lg transition-colors cursor-pointer ${
                isHistoryOpen
                  ? 'bg-sky-900/30 text-sky-400 border border-sky-500/30'
                  : 'hover:bg-slate-800/60 text-slate-400'
              }`}
              title={t('header.toggleHistory')}
            >
              <History size={18} />
            </button>
          </>
        )}

        {dbStatus === 'connected' ? (
          <button
            onClick={handleRefresh}
            className={`flex items-center ${compactHeader ? 'gap-1 px-2 py-1' : 'gap-2 px-3 py-1'} bg-emerald-900/20 text-emerald-400 rounded-full text-xs font-medium border border-emerald-500/30 shadow-sm cursor-pointer hover:bg-emerald-900/30 transition-colors whitespace-nowrap`}
            title={t('header.systemOnline')}
          >
            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.6)]"></div>
            <span className="text-[11px]">{showCompactRightText || !compactHeader ? t('header.systemOnline') : onlineLabel}</span>
          </button>
        ) : (
          <button
            onClick={handleConnect}
            className={`flex items-center ${compactHeader ? 'gap-1 px-2 py-1' : 'gap-2 px-3 py-1'} bg-red-900/20 text-red-400 hover:bg-red-900/30 rounded-full text-xs font-medium border border-red-500/30 shadow-sm transition-colors cursor-pointer animate-pulse whitespace-nowrap`}
            title={t('header.disconnected')}
          >
            <PlugZap size={14} />
            <span className="text-[11px]">{showCompactRightText || !compactHeader ? t('header.disconnected') : offlineLabel}</span>
          </button>
        )}

        {/* Language Switcher */}
        <button
          onClick={toggleLanguage}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border border-slate-700/60 bg-slate-800/60 text-slate-300 hover:border-sky-500/30 hover:text-sky-300"
          title={currentLang === 'zh' ? 'Switch to English' : '切换到中文'}
        >
          <Globe size={14} />
          <span>{currentLang === 'zh' ? 'EN' : '中文'}</span>
        </button>

        <button
          onClick={toggleCanvas}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
            isCanvasOpen
              ? 'bg-sky-900/30 border-sky-500/30 text-sky-400 shadow-[0_0_10px_rgba(56,189,248,0.2)]'
              : 'bg-transparent border-slate-700/60 text-slate-400 hover:bg-slate-800/60'
          } whitespace-nowrap`}
          title={currentLang === 'zh' ? '切换画布面板' : 'Toggle Canvas Panel'}
        >
          {isCanvasOpen ? (
            <PanelRightClose size={14} />
          ) : (
            <PanelRightOpen size={14} />
          )}
          <span>{showCompactRightText || !compactHeader ? canvasLabel : 'CV'}</span>
        </button>
      </div>
    </header>
  );
}
