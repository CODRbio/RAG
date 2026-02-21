import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Layers,
  Shield,
  Cpu,
  Network,
  Filter,
  Globe,
  Database,
  Server,
  PlugZap,
  Loader2,
  History,
  Clock,
  Trash2,
  Settings,
  LogOut,
  GripVertical,
  DownloadCloud,
  CloudCheck,
  HelpCircle,
} from 'lucide-react';
import { useAuthStore, useConfigStore, useUIStore, useToastStore, useChatStore, useCanvasStore } from '../../stores';
import { checkHealth } from '../../api/health';
import { listSessions, deleteSession } from '../../api/chat';
import { getModelStatus, syncModels } from '../../api/models';
import { exportCanvas, getCanvas } from '../../api/canvas';
import type { SessionListItem } from '../../types';

interface SidebarProps {
  onStartResize: () => void;
}

/** 悬停约 1s 后显示小弹窗说明，移出后短暂延迟关闭 */
function HelpTooltip({
  content,
  delayMs = 1000,
  children,
}: {
  content: string;
  delayMs?: number;
  children: React.ReactNode;
}) {
  const [visible, setVisible] = useState(false);
  const showRef = useRef<number | null>(null);
  const hideRef = useRef<number | null>(null);

  const handleEnter = () => {
    if (hideRef.current) {
      window.clearTimeout(hideRef.current);
      hideRef.current = null;
    }
    showRef.current = window.setTimeout(() => setVisible(true), delayMs);
  };

  const handleLeave = () => {
    if (showRef.current) {
      window.clearTimeout(showRef.current);
      showRef.current = null;
    }
    hideRef.current = window.setTimeout(() => setVisible(false), 150);
  };

  return (
    <span
      className="relative inline-flex items-center text-slate-500 hover:text-sky-400 transition-colors cursor-help shrink-0"
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      {children}
      {visible && (
        <span
          className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-3 py-2 text-xs text-slate-200 bg-slate-900/95 backdrop-blur-md rounded-lg shadow-[0_4px_20px_rgba(0,0,0,0.5)] border border-slate-700/50 max-w-[260px] whitespace-normal z-[100] pointer-events-none animate-in fade-in zoom-in-95 duration-200"
          role="tooltip"
        >
          {content}
          <div className="absolute left-1/2 -translate-x-1/2 top-full w-2 h-2 bg-slate-900 border-r border-b border-slate-700/50 transform rotate-45 -mt-1"></div>
        </span>
      )}
    </span>
  );
}

export function Sidebar({ onStartResize }: SidebarProps) {
  const { t } = useTranslation();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const addToast = useToastStore((s) => s.addToast);

  const {
    dbAddress,
    setDbAddress,
    dbStatus,
    setDbStatus,
    currentCollection,
    collections,
    setCurrentCollection,
    ragConfig,
    updateRagConfig,
    webSearchConfig,
    setWebSearchEnabled,
    toggleWebSource,
    updateWebSourceParam,
    setQueryOptimizer,
    setMaxQueriesPerProvider,
    setAgentMode,
  } = useConfigStore();

  const {
    sidebarWidth,
    isSidebarOpen,
    setShowSettingsModal,
  } = useUIStore();

  const { loadSession, isLoadingSession, sessionId: currentSessionId } = useChatStore();
  const { setCanvas, setCanvasContent, clearCanvas, setIsLoading: setCanvasLoading } = useCanvasStore();

  const [showQuerySettings, setShowQuerySettings] = useState(false);
  const [chatHistory, setChatHistory] = useState<SessionListItem[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isSyncingModels, setIsSyncingModels] = useState(false);
  const [modelStatusSummary, setModelStatusSummary] = useState<string | null>(null);

  // 加载会话历史
  useEffect(() => {
    const loadSessions = async () => {
      if (dbStatus !== 'connected') return;
      setIsLoadingSessions(true);
      try {
        const sessions = await listSessions(50);
        setChatHistory(sessions);
      } catch (error) {
        console.error('Failed to load sessions:', error);
      } finally {
        setIsLoadingSessions(false);
      }
    };
    loadSessions();
  }, [dbStatus]);

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      setChatHistory((prev) => prev.filter((s) => s.session_id !== sessionId));
      addToast(t('sidebar.sessionDeleted'), 'success');
    } catch (error) {
      addToast(t('sidebar.deleteFailed'), 'error');
    }
  };

  const handleLoadSession = async (sessionId: string) => {
    if (isLoadingSession || sessionId === currentSessionId) return;
    try {
      await loadSession(sessionId);
      addToast(t('sidebar.sessionLoaded'), 'success');
      
      // 加载 Canvas 内容（如果有）
      const loadedCanvasId = useChatStore.getState().canvasId;
      if (loadedCanvasId) {
        clearCanvas();
        setCanvasLoading(true);
        try {
          const [canvasData, exportResp] = await Promise.all([
            getCanvas(loadedCanvasId).catch(() => null),
            exportCanvas(loadedCanvasId, 'markdown').catch(() => null),
          ]);
          if (canvasData) setCanvas(canvasData);
          if (exportResp?.content) setCanvasContent(exportResp.content);
        } catch (err) {
          console.error('[Sidebar] Failed to load canvas:', err);
        } finally {
          setCanvasLoading(false);
        }
      } else {
        clearCanvas();
      }
    } catch (error) {
      addToast(t('sidebar.loadSessionFailed'), 'error');
    }
  };

  const handleConnect = async () => {
    setDbStatus('connecting');
    addToast(t('sidebar.connectingTo', { address: dbAddress }), 'info');
    try {
      await checkHealth();
      setDbStatus('connected');
      addToast(t('sidebar.connected'), 'success');
    } catch {
      setDbStatus('disconnected');
      addToast(t('sidebar.connectFailed'), 'error');
    }
  };

  const handleLogout = () => {
    logout();
    addToast(t('sidebar.loggedOut'), 'info');
  };

  const handleSyncModels = async () => {
    if (isSyncingModels) return;
    setIsSyncingModels(true);
    setModelStatusSummary(null);
    try {
      // 默认仅在发现新版本时升级；已是最新版本则跳过。
      const result = await syncModels({ force_update: false });
      const upgraded = result.items.filter((i) => i.updated).length;
      const skipped = result.items.filter((i) => i.status === 'already_latest').length;
      const failed = result.items.filter((i) => i.status === 'failed').length;
      setModelStatusSummary(t('sidebar.modelSyncComplete', { upgraded, skipped, failed }));
      addToast(
        t('sidebar.modelSyncComplete', { upgraded, skipped, failed }),
        failed > 0 ? 'error' : 'success',
      );
    } catch (error) {
      addToast(t('sidebar.modelSyncFailed'), 'error');
    } finally {
      setIsSyncingModels(false);
    }
  };

  const handleCheckModels = async () => {
    if (isSyncingModels) return;
    setIsSyncingModels(true);
    setModelStatusSummary(null);
    try {
      const result = await getModelStatus();
      const ok = result.items.filter((i) => i.exists).length;
      const failed = result.items.filter((i) => !i.exists).length;
      setModelStatusSummary(t('sidebar.modelStatusReady', { ok, failed }));
      addToast(t('sidebar.modelStatusReady', { ok, failed }), failed > 0 ? 'error' : 'success');
    } catch (error) {
      addToast(t('sidebar.modelStatusFailed'), 'error');
    } finally {
      setIsSyncingModels(false);
    }
  };

  if (!user) return null;

  return (
    <div
      className="glass-sidebar flex flex-col relative flex-shrink-0 z-40 transition-[width] duration-100 ease-linear shadow-[5px_0_30px_rgba(0,0,0,0.3)]"
      style={{ width: isSidebarOpen ? sidebarWidth : 80 }}
    >
      {/* Logo */}
      <div className="p-6 flex items-center gap-3 h-20 overflow-hidden relative">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-sky-500/10 to-transparent pointer-events-none"></div>
        <div className="bg-gradient-to-br from-sky-500 to-blue-600 p-2 rounded-xl text-white flex-shrink-0 shadow-[0_0_15px_rgba(14,165,233,0.5)] border border-sky-400/30 animate-pulse-glow">
          <Layers size={24} />
        </div>
        {isSidebarOpen && (
          <span className="font-bold text-xl tracking-tight whitespace-nowrap text-sky-100 drop-shadow-md">
            RAG Lab
          </span>
        )}
      </div>

      {/* 内容区 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-8 scrollbar-thin">
        {/* 用户信息 */}
        <div
          className={`flex items-center gap-3 p-3 bg-slate-800/40 rounded-xl border border-slate-700/50 hover:border-sky-500/30 transition-colors ${
            !isSidebarOpen && 'justify-center'
          }`}
        >
          <img
            src={user.avatar}
            alt="Avatar"
            className="w-10 h-10 rounded-full bg-slate-900 border border-slate-600 shadow-sm"
          />
          {isSidebarOpen && (
            <div className="flex-1 min-w-0">
              <div className="font-bold text-sm truncate text-slate-200">
                {user.username || user.user_id}
              </div>
              <div className="text-xs text-slate-400 flex items-center gap-1">
                <Shield size={10} className="text-sky-400" />
                <span className="capitalize">{user.role}</span>
              </div>
            </div>
          )}
        </div>

        {/* 检索策略配置 */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <Cpu size={14} /> {t('sidebar.retrievalConfig')}
            </div>
            <div className="bg-slate-900/30 rounded-xl p-3 border border-slate-700/50 space-y-3 shadow-inner">
              {/* Local RAG Toggle */}
              <div className="flex items-center justify-between pb-2 border-b border-slate-700/50">
                <div className="flex items-center gap-2">
                  <Database size={14} className="text-sky-500" />
                  <span className="text-sm font-medium text-slate-300">{t('sidebar.localKnowledge')}</span>
                  <HelpTooltip content={t('sidebar.localKnowledgeHelp')}>
                    <HelpCircle size={12} />
                  </HelpTooltip>
                </div>
                <input
                  id="rag-enabled"
                  name="rag-enabled"
                  type="checkbox"
                  checked={ragConfig.enabled ?? true}
                  onChange={(e) =>
                    updateRagConfig({ enabled: e.target.checked })
                  }
                  className="accent-sky-500 w-4 h-4 cursor-pointer"
                />
              </div>

              {/* Top-K Slider - 只在启用时显示 */}
              {ragConfig.enabled && (
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span>{t('sidebar.queryCollection')}</span>
                </div>
                <select
                  id="query-collection"
                  name="query-collection"
                  value={currentCollection || ''}
                  onChange={(e) => setCurrentCollection(e.target.value)}
                  className="w-full text-xs bg-slate-950 border border-slate-700 text-slate-300 rounded px-2 py-1.5 mb-2 focus:border-sky-500 focus:outline-none"
                >
                  {collections.length === 0 ? (
                    <option value="">{t('sidebar.noCollection')}</option>
                  ) : (
                    collections.map((name) => (
                      <option key={`query-collection-${name}`} value={name}>
                        {name}
                      </option>
                    ))
                  )}
                </select>
              </div>
              )}

              {/* Top-K Slider - 只在启用时显示 */}
              {ragConfig.enabled && <div>
                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span>Local RAG Top-K</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-700 text-sky-400">
                      {ragConfig.localTopK}
                    </span>
                    <input
                      id="local-topk-num"
                      name="local-topk-num"
                      type="number"
                      min="1"
                      max="200"
                      step="1"
                      value={ragConfig.localTopK}
                      onChange={(e) =>
                        updateRagConfig({ localTopK: Math.max(1, Number(e.target.value)) })
                      }
                      className="w-16 text-[10px] bg-slate-950 border border-slate-700 text-slate-300 rounded px-1 py-0.5 focus:border-sky-500 focus:outline-none"
                      title={t('sidebar.localTopKTitle')}
                    />
                  </div>
                </div>
                <input
                  id="local-topk-range"
                  name="local-topk-range"
                  type="range"
                  min="1"
                  max="60"
                  step="1"
                  value={Math.min(ragConfig.localTopK, 60)}
                  onChange={(e) =>
                    updateRagConfig({ localTopK: Number(e.target.value) })
                  }
                  className="w-full accent-sky-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>}

              {/* HippoRAG Toggle - 只在启用时显示 */}
              {ragConfig.enabled && <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Network size={14} className="text-purple-400" />
                  <span className="text-sm text-slate-300">{t('sidebar.hippoRAG')}</span>
                  <HelpTooltip content={t('sidebar.hippoRAGHelp')}>
                    <HelpCircle size={12} />
                  </HelpTooltip>
                </div>
                <input
                  id="hippo-rag"
                  name="hippo-rag"
                  type="checkbox"
                  checked={ragConfig.enableHippoRAG}
                  onChange={(e) =>
                    updateRagConfig({ enableHippoRAG: e.target.checked })
                  }
                  className="accent-purple-500 w-4 h-4 cursor-pointer"
                />
              </div>}

              {/* Reranker Toggle - 只在启用时显示 */}
              {ragConfig.enabled && <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Filter size={14} className="text-amber-400" />
                  <span className="text-sm text-slate-300">{t('sidebar.colbertReranker')}</span>
                </div>
                <input
                  id="reranker"
                  name="reranker"
                  type="checkbox"
                  checked={ragConfig.enableReranker}
                  onChange={(e) =>
                    updateRagConfig({ enableReranker: e.target.checked })
                  }
                  className="accent-amber-500 w-4 h-4 cursor-pointer"
                />
              </div>}
            </div>
          </section>
        )}

        {/* 合并检索参数（独立于 Local / Web，始终可见） */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <Layers size={14} /> {t('sidebar.mergeParams')}
            </div>
            <div className="bg-slate-900/30 rounded-xl p-3 border border-slate-700/50 space-y-3 shadow-inner">
              {/* Final Top-K */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span>{t('sidebar.finalTopK')}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-700 text-sky-400">
                      {ragConfig.finalTopK ?? 10}
                    </span>
                    <input
                      id="final-topk-num"
                      name="final-topk-num"
                      type="number"
                      min="1"
                      max="200"
                      step="1"
                      value={ragConfig.finalTopK ?? 10}
                      onChange={(e) =>
                        updateRagConfig({ finalTopK: Math.max(1, Number(e.target.value)) })
                      }
                      className="w-16 text-[10px] bg-slate-950 border border-slate-700 text-slate-300 rounded px-1 py-0.5 focus:border-sky-500 focus:outline-none"
                      title={t('sidebar.finalTopKDesc')}
                    />
                  </div>
                </div>
                <input
                  id="final-topk-range"
                  name="final-topk-range"
                  type="range"
                  min="1"
                  max="60"
                  step="1"
                  value={Math.min(ragConfig.finalTopK ?? 10, 60)}
                  onChange={(e) =>
                    updateRagConfig({ finalTopK: Number(e.target.value) })
                  }
                  className="w-full accent-amber-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="text-[9px] text-slate-600 mt-0.5">
                  {t('sidebar.finalTopKDesc')}
                </div>
              </div>

              {/* 相似度阈值 */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
                  <span>{t('sidebar.threshold')}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-700 text-sky-400">
                      {ragConfig.localThreshold?.toFixed(2) ?? '0.50'}
                    </span>
                    <input
                      id="threshold-num"
                      name="threshold-num"
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={ragConfig.localThreshold ?? 0.5}
                      onChange={(e) =>
                        updateRagConfig({ localThreshold: Math.min(1, Math.max(0, Number(e.target.value))) })
                      }
                      className="w-16 text-[10px] bg-slate-950 border border-slate-700 text-slate-300 rounded px-1 py-0.5 focus:border-sky-500 focus:outline-none"
                      title={t('sidebar.thresholdTitle')}
                    />
                  </div>
                </div>
                <input
                  id="threshold-range"
                  name="threshold-range"
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={ragConfig.localThreshold ?? 0.5}
                  onChange={(e) =>
                    updateRagConfig({ localThreshold: Number(e.target.value) })
                  }
                  className="w-full accent-emerald-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="text-[9px] text-slate-600 mt-0.5">
                  {t('sidebar.thresholdDesc')}
                </div>
              </div>

              {/* 年份过滤 */}
              <div className="flex gap-2">
                <div className="flex-1">
                  <div className="text-[10px] text-slate-500 mb-1">{t('sidebar.yearStart')}</div>
                  <input
                    type="number"
                    min="1900"
                    max="2100"
                    value={ragConfig.yearStart || ''}
                    placeholder="1900"
                    onChange={(e) => {
                      const val = e.target.value;
                      updateRagConfig({ yearStart: val ? Number(val) : null });
                    }}
                    className="w-full text-xs bg-slate-950 border border-slate-700 text-slate-300 rounded px-2 py-1.5 focus:border-sky-500 focus:outline-none"
                  />
                </div>
                <div className="flex-1">
                  <div className="text-[10px] text-slate-500 mb-1">{t('sidebar.yearEnd')}</div>
                  <input
                    type="number"
                    min="1900"
                    max="2100"
                    value={ragConfig.yearEnd || ''}
                    placeholder="2026"
                    onChange={(e) => {
                      const val = e.target.value;
                      updateRagConfig({ yearEnd: val ? Number(val) : null });
                    }}
                    className="w-full text-xs bg-slate-950 border border-slate-700 text-slate-300 rounded px-2 py-1.5 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Web Search Config */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <Globe size={14} /> {t('sidebar.webSearch')}
            </div>
            <div className="space-y-3">
              {/* 总开关 */}
              <div className="border border-slate-700/50 rounded-xl p-3 bg-slate-900/30 hover:border-sky-500/30 hover:bg-slate-800/50 transition-all cursor-pointer shadow-sm">
                <label className="flex items-center justify-between cursor-pointer w-full">
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-2 rounded-lg shadow-sm transition-colors ${
                        webSearchConfig.enabled
                          ? 'bg-indigo-900/40 text-indigo-400 shadow-[0_0_10px_rgba(99,102,241,0.2)]'
                          : 'bg-slate-800 text-slate-500'
                      }`}
                    >
                      <Globe size={16} />
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium text-slate-200">{t('sidebar.webSearchToggle')}</span>
                        <HelpTooltip content={t('sidebar.webSearchHelp')}>
                          <HelpCircle size={12} />
                        </HelpTooltip>
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {t('sidebar.webSearchDesc')}
                      </div>
                    </div>
                  </div>
                  <input
                    id="web-search-enabled"
                    name="web-search-enabled"
                    type="checkbox"
                    checked={webSearchConfig.enabled}
                    onChange={(e) => {
                      setWebSearchEnabled(e.target.checked);
                      addToast(
                        e.target.checked ? t('sidebar.webSearchEnabled') : t('sidebar.webSearchDisabled'),
                        'info'
                      );
                    }}
                    className="accent-indigo-500 w-4 h-4 cursor-pointer"
                  />
                </label>
              </div>

              {webSearchConfig.enabled && (
                <div className="bg-slate-900/30 border border-slate-700/50 rounded-xl overflow-hidden animate-in slide-in-from-top-2 duration-200">
                  <div className="bg-slate-800/50 px-3 py-2 border-b border-slate-700/50 text-[10px] font-bold text-slate-500 uppercase">
                    Search Sources
                  </div>
                  <div className="divide-y divide-slate-700/30">
                    {webSearchConfig.sources.map((source) => (
                      <div
                        key={source.id}
                        className="p-3 hover:bg-slate-800/30 transition-colors"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span
                            className={`text-sm font-medium ${
                              source.enabled ? 'text-slate-200' : 'text-slate-500'
                            }`}
                          >
                            {source.name}
                          </span>
                          <input
                            id={`web-source-${source.id}`}
                            name={`web-source-${source.id}`}
                            type="checkbox"
                            checked={source.enabled}
                            onChange={() => toggleWebSource(source.id)}
                            className="accent-sky-500 w-4 h-4 cursor-pointer"
                          />
                        </div>
                        {source.enabled && (
                          <div className="space-y-2 animate-in slide-in-from-top-1 duration-200">
                            {/* Top-K 配置 */}
                            <div className="flex items-center justify-between text-[10px] text-slate-500">
                              <span>Top-K</span>
                              <div className="flex items-center gap-2">
                                <span className="font-mono bg-slate-950 px-1.5 py-0.5 rounded border border-slate-700 text-sky-400">
                                  {source.topK}
                                </span>
                                <input
                                  id={`web-source-topk-num-${source.id}`}
                                  name={`web-source-topk-num-${source.id}`}
                                  type="number"
                                  min="1"
                                  max="100"
                                  step="1"
                                  value={source.topK}
                                  onChange={(e) =>
                                    updateWebSourceParam(
                                      source.id,
                                      'topK',
                                      Math.max(1, Number(e.target.value))
                                    )
                                  }
                                  className="w-14 text-[10px] bg-slate-950 border border-slate-700 text-slate-300 rounded px-1 py-0.5 focus:border-sky-500 focus:outline-none"
                                />
                              </div>
                            </div>
                            <input
                              id={`web-source-topk-range-${source.id}`}
                              name={`web-source-topk-range-${source.id}`}
                              type="range"
                              min="1"
                              max="60"
                              step="1"
                              value={Math.min(source.topK, 60)}
                              onChange={(e) =>
                                updateWebSourceParam(
                                  source.id,
                                  'topK',
                                  Number(e.target.value)
                                )
                              }
                              className="w-full accent-sky-500 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="text-[9px] text-slate-600 mt-1">
                              {t('sidebar.webTopKDesc')}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 查询增强选项 */}
              {webSearchConfig.enabled && (
                <div className="bg-slate-900/30 border border-slate-700/50 rounded-xl p-3 animate-in slide-in-from-top-2 duration-200 space-y-3">
                  <div className="text-[10px] font-bold text-slate-500 uppercase">
                    Query Enhancement
                  </div>
                  
                  {/* 查询优化器 */}
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium text-slate-300">{t('sidebar.queryOptimizer')}</span>
                        <HelpTooltip content={t('sidebar.queryOptimizerHelp')}>
                          <HelpCircle size={12} />
                        </HelpTooltip>
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {t('sidebar.queryOptimizerDesc')}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setShowQuerySettings((v) => !v)}
                        className="text-[10px] text-sky-400 px-2 py-1 border border-slate-700/50 rounded-md hover:bg-slate-800 transition-colors"
                        title={t('sidebar.settingsQueryTitle')}
                      >
                        {t('common.settings')}
                      </button>
                      <input
                        id="query-optimizer"
                        name="query-optimizer"
                        type="checkbox"
                        checked={webSearchConfig.queryOptimizer ?? true}
                        onChange={(e) => {
                          setQueryOptimizer(e.target.checked);
                          addToast(
                            e.target.checked
                              ? t('sidebar.queryOptimizerEnabled')
                              : t('sidebar.queryOptimizerDisabled'),
                            'info'
                          );
                        }}
                        className="accent-sky-500 w-4 h-4 cursor-pointer"
                      />
                    </div>
                  </div>

                  {showQuerySettings && (
                    <div className="pt-2 border-t border-slate-700/50">
                      <label className="flex items-center justify-between">
                        <span className="text-xs text-slate-400">{t('sidebar.queriesPerProvider')}</span>
                        <select
                          id="max-queries-per-provider"
                          name="max-queries-per-provider"
                          value={webSearchConfig.maxQueriesPerProvider ?? 3}
                          onChange={(e) => {
                            setMaxQueriesPerProvider(Number(e.target.value));
                            addToast(t('sidebar.queryCountSet', { count: Number(e.target.value) }), 'info');
                          }}
                          className="text-xs border border-slate-700 rounded-md px-2 py-1 bg-slate-950 text-slate-300 focus:outline-none focus:border-sky-500"
                        >
                          {[1, 2, 3, 4, 5].map((n) => (
                            <option key={n} value={n}>
                              {n}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  )}
                </div>
              )}

              {/* 全文抓取 */}
              {webSearchConfig.enabled && (
                <div className="bg-slate-900/30 border border-slate-700/50 rounded-xl p-3 animate-in slide-in-from-top-2 duration-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium text-slate-300">{t('sidebar.contentFetcher')}</span>
                        <HelpTooltip content={t('sidebar.contentFetcherHelp')}>
                          <HelpCircle size={12} />
                        </HelpTooltip>
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {t('sidebar.contentFetcherDesc')}
                      </div>
                    </div>
                    <select
                      id="content-fetcher-mode"
                      name="content-fetcher-mode"
                      value={webSearchConfig.contentFetcherMode ?? 'auto'}
                      onChange={(e) => {
                        const mode = e.target.value as 'auto' | 'force' | 'off';
                        (useConfigStore.getState() as any).setContentFetcherMode(mode);
                        addToast(
                          mode === 'force' ? t('sidebar.contentFetcherForce')
                            : mode === 'off' ? t('sidebar.contentFetcherDisabled')
                            : t('sidebar.contentFetcherAuto'),
                          'info'
                        );
                      }}
                      className="text-xs border border-slate-700 rounded-md px-2 py-1 bg-slate-950 text-slate-300 focus:outline-none focus:border-sky-500"
                    >
                      <option value="auto">Auto</option>
                      <option value="force">Force</option>
                      <option value="off">Off</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Agent 模式 */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <Cpu size={14} /> {t('sidebar.agentMode')}
            </div>
            <div className="bg-slate-900/30 border border-slate-700/50 rounded-xl p-3 shadow-inner space-y-2">
              {/* 三模式选择器 */}
              {(['standard', 'assist', 'autonomous'] as const).map((mode) => {
                const isSelected = (ragConfig.agentMode ?? 'assist') === mode;
                const labels: Record<string, string> = {
                  standard: t('sidebar.agentModeStandard'),
                  assist: t('sidebar.agentModeAssist'),
                  autonomous: t('sidebar.agentModeAutonomous'),
                };
                const descs: Record<string, string> = {
                  standard: t('sidebar.agentModeStandardDesc'),
                  assist: t('sidebar.agentModeAssistDesc'),
                  autonomous: t('sidebar.agentModeAutonomousDesc'),
                };
                const helps: Record<string, string> = {
                  standard: t('sidebar.agentModeStandardHelp'),
                  assist: t('sidebar.agentModeAssistHelp'),
                  autonomous: t('sidebar.agentModeAutonomousHelp'),
                };
                return (
                  <button
                    key={mode}
                    onClick={() => {
                      setAgentMode(mode);
                      addToast(t('sidebar.agentModeChanged', { mode: labels[mode] }), 'info');
                    }}
                    className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg border text-left transition-colors ${
                      isSelected
                        ? 'bg-indigo-600/20 border-indigo-500/60 text-slate-200'
                        : 'bg-transparent border-slate-700/40 text-slate-400 hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="flex-1 min-w-0 pr-2">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-medium">{labels[mode]}</span>
                        <HelpTooltip content={helps[mode]}>
                          <HelpCircle size={10} />
                        </HelpTooltip>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">{descs[mode]}</div>
                    </div>
                    <div className={`w-3.5 h-3.5 rounded-full border-2 flex-shrink-0 ${
                      isSelected ? 'bg-indigo-500 border-indigo-400' : 'border-slate-600'
                    }`} />
                  </button>
                );
              })}
              {/* Agent Debug Mode sub-toggle — only for assist / autonomous */}
              {(ragConfig.agentMode ?? 'assist') !== 'standard' && (
                <div className="flex items-center justify-between border-t border-slate-700/30 pt-2 mt-1">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-slate-400">{t('sidebar.agentDebugMode', 'Debug Panel')}</span>
                      <HelpTooltip content={t('sidebar.agentDebugModeHelp', 'Show detailed Agent stats, tool timing, and contribution analysis in chat')}>
                        <HelpCircle size={10} />
                      </HelpTooltip>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {t('sidebar.agentDebugModeDesc', 'Timeline + stats')}
                    </div>
                  </div>
                  <input
                    id="agent-debug-mode"
                    name="agent-debug-mode"
                    type="checkbox"
                    checked={ragConfig.agentDebugMode ?? false}
                    onChange={(e) => {
                      updateRagConfig({ agentDebugMode: e.target.checked });
                    }}
                    className="accent-purple-500 w-3.5 h-3.5 cursor-pointer"
                  />
                </div>
              )}
            </div>
          </section>
        )}

        {/* 服务连接 */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <Database size={14} /> {t('sidebar.serviceConnection')}
            </div>
            <div className="space-y-3">
              <div>
                <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
                  {t('sidebar.serviceAddress')}
                  <HelpTooltip content={t('sidebar.serviceAddressHelp')}>
                    <HelpCircle size={12} />
                  </HelpTooltip>
                </label>
                <div className="relative">
                  <input
                    id="service-address"
                    name="service-address"
                    type="text"
                    disabled={user.role !== 'admin'}
                    className={`w-full bg-slate-950 border border-slate-700 rounded-md p-2 text-sm pl-8 text-slate-300 focus:border-sky-500 focus:outline-none ${
                      dbStatus === 'connected'
                        ? 'text-emerald-400 border-emerald-500/30 bg-emerald-900/10'
                        : ''
                    } disabled:opacity-60 disabled:cursor-not-allowed`}
                    value={dbAddress}
                    onChange={(e) => setDbAddress(e.target.value)}
                  />
                  <Server
                    size={14}
                    className="absolute left-2.5 top-2.5 text-slate-500"
                  />
                </div>
              </div>
              {dbStatus === 'connected' ? (
                <div className="p-3 bg-emerald-900/10 rounded-lg border border-emerald-500/30">
                  <div className="flex justify-between items-center text-xs text-emerald-400/80">
                    <span>Status:</span>
                    <span className="text-emerald-400 flex items-center gap-1 font-medium">
                      <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]"></div>
                      Online
                    </span>
                  </div>
                </div>
              ) : (
                <button
                  onClick={handleConnect}
                  disabled={dbStatus === 'connecting'}
                  className="w-full text-xs text-sky-400 font-medium py-2 border border-sky-500/30 rounded-md hover:bg-sky-500/10 flex justify-center items-center gap-2 transition-colors cursor-pointer"
                >
                  {dbStatus === 'connecting' ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <PlugZap size={14} />
                  )}
                  {dbStatus === 'connecting' ? t('sidebar.connecting') : t('sidebar.connectService')}
                </button>
              )}
            </div>
          </section>
        )}

        {/* 本地模型管理 */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <DownloadCloud size={14} /> {t('sidebar.localModels')}
            </div>
            <div className="bg-slate-900/30 rounded-lg p-3 border border-slate-700/50 space-y-2">
              <button
                onClick={handleSyncModels}
                disabled={isSyncingModels}
                className="w-full text-xs text-sky-400 font-medium py-2 border border-sky-500/30 rounded-md hover:bg-sky-500/10 flex justify-center items-center gap-2 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
                title={t('sidebar.syncModels')}
              >
                {isSyncingModels ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <DownloadCloud size={14} />
                )}
                {isSyncingModels ? t('sidebar.syncing') : t('sidebar.syncModels')}
              </button>
              <button
                onClick={handleCheckModels}
                disabled={isSyncingModels}
                className="w-full text-xs text-slate-400 font-medium py-2 border border-slate-700/50 rounded-md hover:bg-slate-800 flex justify-center items-center gap-2 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
                title={t('sidebar.checkModelStatus')}
              >
                <CloudCheck size={14} />
                {t('sidebar.checkModelStatus')}
              </button>
              {modelStatusSummary && (
                <div className="text-[10px] text-slate-400 text-center">
                  {modelStatusSummary}
                </div>
              )}
              <div className="text-[10px] text-slate-600 text-center">
                {t('sidebar.defaultCacheDir')}: <code className="font-mono text-slate-500">~/Hug</code>
              </div>
            </div>
          </section>
        )}

        {/* 历史项目 */}
        {isSidebarOpen && (
          <section>
            <div className="flex items-center gap-2 mb-4 text-sky-500/80 font-semibold text-xs uppercase tracking-wider pl-1">
              <History size={14} /> {t('sidebar.history')}
              {isLoadingSessions && <Loader2 size={12} className="animate-spin" />}
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto pr-1 scrollbar-thin">
              {chatHistory.length === 0 && !isLoadingSessions && (
                <div className="text-xs text-slate-600 text-center py-4">{t('sidebar.noHistory')}</div>
              )}
              {chatHistory.map((h) => (
                <div
                  key={h.session_id}
                  className={`group p-2 rounded-lg hover:bg-slate-800/50 cursor-pointer text-sm relative transition-colors border border-transparent ${
                    currentSessionId === h.session_id 
                      ? 'bg-sky-900/20 border-sky-500/30 shadow-[0_0_10px_rgba(56,189,248,0.1)]' 
                      : ''
                  }`}
                  onClick={() => handleLoadSession(h.session_id)}
                >
                  <div className="flex justify-between items-start">
                    <span className={`font-medium truncate flex-1 transition-colors ${
                      currentSessionId === h.session_id ? 'text-sky-400' : 'text-slate-300 group-hover:text-sky-200'
                    }`}>
                      {h.title}
                    </span>
                    {isLoadingSession && currentSessionId !== h.session_id && (
                      <Loader2 size={12} className="animate-spin text-sky-500 flex-shrink-0" />
                    )}
                  </div>
                  <div className="text-[10px] text-slate-500 flex items-center gap-1 mt-1">
                    <Clock size={10} /> {new Date(h.updated_at).toLocaleString('zh-CN', {
                      month: '2-digit',
                      day: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                    <span className="ml-2 text-slate-700">|</span>
                    <span className="ml-1">{h.turn_count} {t('sidebar.turns')}</span>
                  </div>

                  {/* Hover Actions */}
                  <div className="absolute right-1 top-1 hidden group-hover:flex gap-1 bg-slate-900 border border-slate-700 p-1 rounded shadow-lg z-10">
                    <button
                      className="p-1 hover:text-red-400 text-slate-500 transition-colors"
                      title={t('common.delete')}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(t('sidebar.confirmDeleteSession'))) {
                          handleDeleteSession(h.session_id);
                        }
                      }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700/50 bg-slate-900/50 space-y-2">
        {user.role === 'admin' && (
          <button
            onClick={() => setShowSettingsModal(true)}
            className={`w-full flex items-center justify-center gap-2 bg-slate-800 text-slate-200 py-3 rounded-xl text-sm font-medium hover:bg-slate-700 hover:text-white transition-all border border-slate-700/50 ${
              !isSidebarOpen && 'px-0'
            }`}
          >
            <Settings size={18} />
            {isSidebarOpen && t('sidebar.advancedConfig')}
          </button>
        )}
        <button
          onClick={handleLogout}
          className={`w-full flex items-center justify-center gap-2 text-red-400 hover:bg-red-900/20 py-3 rounded-xl text-sm font-medium transition-all ${
            !isSidebarOpen && 'px-0'
          }`}
        >
          <LogOut size={18} />
          {isSidebarOpen && t('sidebar.logout')}
        </button>
      </div>

      {/* 拖拽把手 */}
      {isSidebarOpen && (
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            onStartResize();
          }}
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-sky-500/50 z-50 group transition-colors"
        >
          <div className="absolute top-1/2 -right-3 w-6 h-8 bg-slate-800 border border-slate-600 rounded shadow-sm flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            <GripVertical size={12} className="text-slate-400" />
          </div>
        </div>
      )}
    </div>
  );
}
