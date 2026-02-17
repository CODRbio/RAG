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
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuthStore, useConfigStore, useChatStore, useUIStore, useToastStore } from '../../stores';
import { checkHealth } from '../../api/health';

interface ModelEntry {
  provider: string;
  model: string;
  label: string;
  group: string;
  color: string;
}
type HeaderTabId = 'chat' | 'ingest' | 'users' | 'graph' | 'compare';

interface HeaderTabConfig {
  id: HeaderTabId;
  labelKey: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  activeClass: string;
}

const MODEL_LIST: ModelEntry[] = [
  { provider: 'deepseek',          model: 'deepseek-chat',       label: 'deepseek-chat',       group: 'DeepSeek',       color: 'text-sky-400' },
  { provider: 'deepseek-thinking', model: 'deepseek-reasoner',   label: 'deepseek-reasoner (thinking)', group: 'DeepSeek', color: 'text-sky-500' },
  { provider: 'openai',            model: 'gpt-5-mini',          label: 'gpt-5-mini',          group: 'OpenAI',         color: 'text-emerald-400' },
  { provider: 'openai',            model: 'gpt-5.2',             label: 'gpt-5.2',             group: 'OpenAI',         color: 'text-emerald-400' },
  { provider: 'openai-thinking',   model: 'gpt-5.2',             label: 'gpt-5.2 (thinking)',   group: 'OpenAI',         color: 'text-emerald-500' },
  { provider: 'claude',            model: 'claude-sonnet-4-5',   label: 'claude-sonnet-4.5',   group: 'Claude',         color: 'text-amber-400' },
  { provider: 'claude',            model: 'claude-haiku-4-5',    label: 'claude-haiku-4.5',    group: 'Claude',         color: 'text-amber-500' },
  { provider: 'claude',            model: 'claude-opus-4-6',     label: 'claude-opus-4.6',     group: 'Claude',         color: 'text-amber-600' },
  { provider: 'claude-thinking',   model: 'claude-sonnet-4-5',   label: 'claude-sonnet-4.5 (thinking)', group: 'Claude', color: 'text-amber-400' },
  { provider: 'claude-thinking',   model: 'claude-haiku-4-5',    label: 'claude-haiku-4.5 (thinking)',  group: 'Claude', color: 'text-amber-500' },
  { provider: 'claude-thinking',   model: 'claude-opus-4-6',     label: 'claude-opus-4.6 (thinking)',   group: 'Claude', color: 'text-amber-600' },
  { provider: 'gemini',            model: 'gemini-pro-latest',   label: 'gemini-pro',          group: 'Gemini',         color: 'text-purple-400' },
  { provider: 'gemini',            model: 'gemini-flash-latest', label: 'gemini-flash',        group: 'Gemini',         color: 'text-purple-300' },
  { provider: 'gemini-thinking',   model: 'gemini-pro-latest',   label: 'gemini-pro (thinking)',   group: 'Gemini',     color: 'text-purple-500' },
  { provider: 'gemini-thinking',   model: 'gemini-flash-latest', label: 'gemini-flash (thinking)', group: 'Gemini',     color: 'text-purple-500' },
  { provider: 'gemini-vision',     model: 'gemini-2.5-flash',    label: 'gemini-2.5-flash (vision)', group: 'Gemini',   color: 'text-purple-300' },
  { provider: 'kimi',              model: 'kimi-k2.5',           label: 'kimi-k2.5',           group: 'Kimi',           color: 'text-cyan-400' },
  { provider: 'kimi-thinking',     model: 'kimi-k2.5',           label: 'kimi-k2.5 (thinking)', group: 'Kimi',          color: 'text-cyan-500' },
  { provider: 'kimi-vision',       model: 'kimi-k2.5',           label: 'kimi-k2.5 (vision)',   group: 'Kimi',          color: 'text-cyan-300' },
  { provider: 'sonar',             model: 'sonar',               label: 'sonar (search)',      group: 'Perplexity',    color: 'text-teal-400' },
  { provider: 'sonar',             model: 'sonar-pro',           label: 'sonar-pro (search)',  group: 'Perplexity',    color: 'text-teal-400' },
  { provider: 'sonar',             model: 'sonar-reasoning-pro', label: 'sonar-reasoning-pro', group: 'Perplexity',    color: 'text-teal-500' },
];

function encodeModelValue(provider: string, model: string): string {
  return `${provider}::${model}`;
}
function decodeModelValue(val: string): { provider: string; model: string } {
  const [provider, ...rest] = val.split('::');
  return { provider, model: rest.join('::') };
}

function groupedModels(): Map<string, ModelEntry[]> {
  const map = new Map<string, ModelEntry[]>();
  for (const entry of MODEL_LIST) {
    if (!map.has(entry.group)) map.set(entry.group, []);
    map.get(entry.group)!.push(entry);
  }
  return map;
}

export function Header() {
  const { t, i18n } = useTranslation();
  const headerRef = useRef<HTMLElement>(null);
  const moreMenuRef = useRef<HTMLDivElement>(null);
  const [compactHeader, setCompactHeader] = useState(false);
  const [headerWidth, setHeaderWidth] = useState(0);
  const [showMoreTabs, setShowMoreTabs] = useState(false);
  const user = useAuthStore((s) => s.user);
  const { dbStatus, setDbStatus, selectedProvider, setSelectedProvider, selectedModel, setSelectedModel } = useConfigStore();
  const { workflowStep, deepResearchActive, setShowDeepResearchDialog, newChat } = useChatStore();
  const {
    activeTab,
    setActiveTab,
    toggleSidebar,
    isCanvasOpen,
    toggleCanvas,
    isHistoryOpen,
    toggleHistory,
  } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);

  const currentLang = i18n.language?.startsWith('en') ? 'en' : 'zh';

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
        <div className={`flex items-center gap-1.5 ${compactHeader ? 'px-2 py-1' : 'px-2.5 py-1'} bg-slate-800/60 rounded-lg border border-slate-700/60 backdrop-blur-sm shadow-sm hover:border-sky-500/30 transition-all`}>
          <Sparkles
            size={13}
            className={
              MODEL_LIST.find(
                (m) => m.provider === selectedProvider && m.model === selectedModel
              )?.color || 'text-slate-500'
            }
          />
          <select
            value={encodeModelValue(selectedProvider, selectedModel)}
            onChange={(e) => {
              const { provider, model } = decodeModelValue(e.target.value);
              setSelectedProvider(provider);
              setSelectedModel(model);
            }}
            className={`bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer pr-4 appearance-none ${compactHeader ? 'max-w-[140px]' : 'max-w-[220px]'}`}
            title={currentLang === 'zh' ? '选择模型' : 'Select Model'}
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 0px center',
            }}
          >
            {Array.from(groupedModels().entries()).map(([group, entries]) => (
              <optgroup key={group} label={group} className="bg-slate-800 text-slate-300">
                {entries.map((entry) => (
                  <option
                    key={encodeModelValue(entry.provider, entry.model)}
                    value={encodeModelValue(entry.provider, entry.model)}
                    className="bg-slate-800 text-slate-300"
                  >
                    {entry.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>

        {!compactHeader && workflowStep !== 'idle' && (
          <div className="flex items-center gap-2 px-4 py-1.5 bg-sky-900/30 text-sky-400 rounded-full text-xs font-bold border border-sky-500/30 animate-pulse-glow">
            <GitBranch size={14} />
            <span className="uppercase tracking-wide">{t('header.workflow')}: {workflowStep}</span>
          </div>
        )}
        {!compactHeader && deepResearchActive && (
          <button
            onClick={() => setShowDeepResearchDialog(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-indigo-900/30 text-indigo-400 rounded-full text-xs font-bold border border-indigo-500/30 hover:bg-indigo-900/50 transition-colors"
            title={currentLang === 'zh' ? '打开 Deep Research 面板' : 'Open Deep Research Panel'}
          >
            <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-pulse" />
            Deep Research Running
          </button>
        )}
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
