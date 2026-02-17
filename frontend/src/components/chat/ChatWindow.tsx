import { useRef, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { MessageSquare, FileSearch, Copy, Download, ExternalLink, FileText, User, Calendar, GitCompareArrows } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChatStore, useToastStore, useCompareStore } from '../../stores';
import type { Source, DeepResearchJobInfo } from '../../types';
import { cancelDeepResearchJob, getDeepResearchJob, listDeepResearchJobEvents } from '../../api/chat';
import { RetrievalDebugPanel } from './RetrievalDebugPanel';
import { ToolTracePanel } from './ToolTracePanel';
import { ResearchProgressPanel } from '../research/ResearchProgressPanel';

export function ChatWindow() {
  const { t } = useTranslation();
  const messages = useChatStore((s) => s.messages);
  const lastEvidenceSummary = useChatStore((s) => s.lastEvidenceSummary);
  const deepResearchActive = useChatStore((s) => s.deepResearchActive);
  const researchDashboard = useChatStore((s) => s.researchDashboard);
  const toolTrace = useChatStore((s) => s.toolTrace);
  const setShowDeepResearchDialog = useChatStore((s) => s.setShowDeepResearchDialog);
  const setDeepResearchTopic = useChatStore((s) => s.setDeepResearchTopic);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const addToast = useToastStore((s) => s.addToast);
  const addComparePreselected = useCompareStore((s) => s.addComparePreselected);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [backgroundJob, setBackgroundJob] = useState<DeepResearchJobInfo | null>(null);
  const [stoppingJobId, setStoppingJobId] = useState<string | null>(null);
  const [showBackgroundLogs, setShowBackgroundLogs] = useState(false);
  const [backgroundEventLines, setBackgroundEventLines] = useState<string[]>([]);
  const lastBackgroundEventIdRef = useRef(0);
  const trackedBackgroundJobIdRef = useRef<string | null>(null);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 新对话场景下：如果后台仍有 Deep Research 任务，显示窄条提示并允许停止。
  useEffect(() => {
    let cancelled = false;

    const toEventLine = (eventName: string, data: Record<string, unknown>): string => {
      const section = typeof data.section === 'string' ? data.section : '';
      const message = typeof data.message === 'string' ? data.message : '';
      const typ = typeof data.type === 'string' ? data.type : eventName;
      if (section) return `[${typ}] ${section}`;
      if (message) return `[${eventName}] ${message}`;
      return `[${eventName}] ${JSON.stringify(data)}`;
    };

    const refreshBackgroundJob = async () => {
      const activeJobId = localStorage.getItem('deep_research_active_job_id');
      if (!activeJobId) {
        if (!cancelled) {
          setBackgroundJob(null);
          setBackgroundEventLines([]);
          setShowBackgroundLogs(false);
        }
        trackedBackgroundJobIdRef.current = null;
        lastBackgroundEventIdRef.current = 0;
        return;
      }
      try {
        const job = await getDeepResearchJob(activeJobId);
        if (cancelled) return;
        if (trackedBackgroundJobIdRef.current !== activeJobId) {
          trackedBackgroundJobIdRef.current = activeJobId;
          lastBackgroundEventIdRef.current = 0;
          setBackgroundEventLines([]);
          setShowBackgroundLogs(false);
        }
        const running = job.status === 'pending' || job.status === 'running' || job.status === 'cancelling';
        if (running) {
          setBackgroundJob(job);
          const events = await listDeepResearchJobEvents(activeJobId, lastBackgroundEventIdRef.current, 30);
          if (events.length > 0) {
            const maxId = events[events.length - 1].event_id;
            lastBackgroundEventIdRef.current = Math.max(lastBackgroundEventIdRef.current, maxId);
            setBackgroundEventLines((prev) => {
              const next = [...prev];
              events.forEach((evt) => {
                if (evt.event === 'progress' || evt.event === 'warning' || evt.event === 'waiting_review') {
                  next.push(toEventLine(evt.event, evt.data || {}));
                }
              });
              return next.slice(-10);
            });
          }
        } else {
          setBackgroundJob(null);
          setBackgroundEventLines([]);
          setShowBackgroundLogs(false);
          localStorage.removeItem('deep_research_active_job_id');
        }
      } catch {
        if (!cancelled) setBackgroundJob(null);
      }
    };

    refreshBackgroundJob();
    const timer = window.setInterval(refreshBackgroundJob, 10000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const handleCopy = (content: string) => {
    navigator.clipboard.writeText(content);
    addToast(t('chat.copied'), 'info');
  };

  const handleExport = (content: string) => {
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chat_response.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addToast(t('chat.exported'), 'success');
  };

  const handleOpenSource = (source: Source) => {
    if (source.url) {
      window.open(source.url, '_blank');
    } else if (source.doi) {
      window.open(`https://doi.org/${source.doi}`, '_blank');
    } else if (source.doc_id) {
      addToast(t('chat.localDoc') + ': ' + source.doc_id, 'info');
    } else {
      addToast(`引用: [${source.cite_key}] ${source.title}`, 'info');
    }
  };

  const handleForceStopBackgroundJob = async () => {
    if (!backgroundJob || stoppingJobId === backgroundJob.job_id) return;
    const ok = window.confirm(t('chat.confirmForceStop'));
    if (!ok) return;
    try {
      setStoppingJobId(backgroundJob.job_id);
      await cancelDeepResearchJob(backgroundJob.job_id);
      addToast(t('chat.stopRequested'), 'info');
    } catch {
      addToast(t('chat.stopFailed'), 'error');
    } finally {
      setStoppingJobId(null);
    }
  };

  const formatAuthors = (authors: string[] | undefined): string => {
    if (!authors || authors.length === 0) return '佚名';
    if (authors.length === 1) return authors[0];
    if (authors.length === 2) return authors.join(' & ');
    return `${authors[0]} et al.`;
  };

  const showBackgroundBanner = Boolean(backgroundJob) && !deepResearchActive && !researchDashboard;

  const backgroundBanner = showBackgroundBanner ? (
    <div className="border rounded-lg bg-sky-900/30 border-sky-500/30 px-3 py-2 text-xs text-sky-300 shadow-[0_0_10px_rgba(14,165,233,0.1)]">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="font-medium truncate">
            {backgroundJob?.status === 'cancelling' ? t('chat.bgResearchStopping') : t('chat.bgResearchRunning')}
          </div>
          <div className="text-[11px] text-sky-400 truncate">
            {backgroundJob?.topic || t('chat.unnamed')} · {t('chat.stage')}: {backgroundJob?.current_stage || 'unknown'}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setShowBackgroundLogs((v) => !v)}
            className="px-2 py-1 rounded-md border border-sky-500/30 text-sky-400 hover:bg-sky-500/10 transition-colors"
          >
            {showBackgroundLogs ? t('chat.collapseLogs') : t('chat.recentLogs')}
          </button>
          <button
            onClick={() => {
              if (!backgroundJob) return;
              setDeepResearchTopic(backgroundJob.topic || '');
              setShowDeepResearchDialog(true);
            }}
            className="px-2 py-1 rounded-md border border-sky-500/30 text-sky-400 hover:bg-sky-500/10 transition-colors"
          >
            {t('chat.viewTask')}
          </button>
          <button
            onClick={handleForceStopBackgroundJob}
            disabled={!backgroundJob || backgroundJob.status === 'cancelling' || stoppingJobId === backgroundJob?.job_id}
            className="px-2 py-1 rounded-md bg-red-900/40 border border-red-500/30 text-red-400 hover:bg-red-900/60 disabled:opacity-50 transition-colors"
          >
            {t('chat.forceStop')}
          </button>
        </div>
      </div>
      {showBackgroundLogs && (
        <div className="mt-2 border-t border-sky-500/30 pt-2 space-y-1">
          {(backgroundEventLines.length > 0 ? backgroundEventLines.slice(-3) : [t('chat.noNewLogs')]).map((line, idx) => (
            <div key={`${idx}-${line}`} className="text-[11px] text-sky-500 truncate font-mono">
              {line}
            </div>
          ))}
        </div>
      )}
    </div>
  ) : null;

  if (messages.length === 0) {
    return (
      <div className="max-w-3xl mx-auto space-y-4 pb-24">
        {backgroundBanner}
        {(deepResearchActive || researchDashboard) && (
          <ResearchProgressPanel dashboard={researchDashboard} isActive={deepResearchActive} />
        )}
        <div className="flex flex-col items-center justify-center h-64 text-slate-500">
          <div className="w-16 h-16 bg-slate-800/50 rounded-2xl flex items-center justify-center mb-4 shadow-[0_0_20px_rgba(56,189,248,0.1)] border border-slate-700/50 animate-float">
            <MessageSquare size={32} className="opacity-60 text-sky-400" />
          </div>
          <p className="font-medium text-slate-400">{t('chat.newConversation')}</p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="max-w-3xl mx-auto space-y-6 pb-24 overflow-y-auto px-4"
    >
      {backgroundBanner}
      {/* Agent 工具调用轨迹 */}
      {toolTrace && toolTrace.length > 0 && (
        <ToolTracePanel trace={toolTrace} />
      )}
      {/* Deep Research 进度面板 */}
      {(deepResearchActive || researchDashboard) && (
        <ResearchProgressPanel dashboard={researchDashboard} isActive={deepResearchActive} />
      )}
      {!deepResearchActive && researchDashboard && researchDashboard.coverage_gaps.length > 0 && (
        <div className="border rounded-lg bg-amber-900/20 border-amber-500/30 p-3 text-sm text-amber-400">
          <div className="font-medium mb-1 flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
            {t('chat.coverageGaps')}
          </div>
          <div className="text-xs mb-2 text-amber-400/80">{t('chat.supplementHint')}</div>
          <button
            onClick={() => setShowDeepResearchDialog(true)}
            className="px-3 py-1.5 text-xs rounded-md bg-amber-600/20 border border-amber-500/30 text-amber-300 hover:bg-amber-600/40 transition-colors"
          >
            {t('chat.supplementAndContinue')}
          </button>
        </div>
      )}
      {/* 检索诊断面板 */}
      {lastEvidenceSummary && lastEvidenceSummary.total_chunks > 0 && (
        <RetrievalDebugPanel summary={lastEvidenceSummary} />
      )}
      {messages.map((msg, idx) => {
        const displayContent =
          isStreaming && idx === messages.length - 1
            ? msg.content.replace(/\[([a-fA-F0-9]{8})\]/g, '[⏳...]')
            : msg.content;
        return (
        <div
          key={idx}
          className={`flex ${
            msg.role === 'user' ? 'justify-end' : 'justify-start'
          } animate-in slide-in-from-bottom-2 duration-300`}
        >
          <div
            className={`max-w-[90%] rounded-2xl p-5 shadow-lg group backdrop-blur-sm ${
              msg.role === 'user'
                ? 'bg-[var(--bg-bubble-user)] text-white border-transparent shadow-sky-500/10'
                : 'bg-[var(--bg-bubble-ai)] text-slate-200 border border-slate-700/50 shadow-black/20'
            }`}
          >
            {/* 消息内容 - 使用 Markdown 渲染 */}
            {msg.role === 'assistant' ? (
              <div className="prose prose-invert prose-sm max-w-none 
                prose-headings:text-sky-300 
                prose-h1:text-xl prose-h2:text-lg prose-h3:text-base 
                prose-p:text-slate-300 prose-p:leading-relaxed 
                prose-li:text-slate-300 
                prose-strong:text-sky-200 
                prose-a:text-sky-400 prose-a:no-underline hover:prose-a:underline
                prose-code:bg-slate-900/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sky-300 prose-code:border prose-code:border-slate-700/50
                prose-pre:bg-slate-950/80 prose-pre:border prose-pre:border-slate-800 prose-pre:shadow-inner
                prose-blockquote:border-l-sky-500/50 prose-blockquote:bg-slate-800/20 prose-blockquote:py-1 prose-blockquote:px-4
                prose-table:border-collapse prose-th:border-b prose-th:border-slate-700 prose-td:border-b prose-td:border-slate-800/50">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {displayContent}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
                {msg.content}
              </p>
            )}

            {/* 助手消息工具栏 */}
            {msg.role === 'assistant' && msg.content && (
              <div className="mt-3 pt-2 border-t border-slate-700/30 flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleCopy(msg.content)}
                  className="p-1.5 text-slate-500 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
                  title={t('chat.copyContent')}
                >
                  <Copy size={14} />
                </button>
                <button
                  onClick={() => handleExport(msg.content)}
                  className="p-1.5 text-slate-500 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
                  title={t('chat.exportMarkdown')}
                >
                  <Download size={14} />
                </button>
              </div>
            )}

            {/* 引用来源 - 显示完整信息 */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-slate-700/30 space-y-2">
                <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5 mb-3">
                  <FileSearch size={12} className="text-sky-500" /> {t('chat.references')} ({msg.sources.length})
                </div>
                <div className="space-y-2">
                  {msg.sources.map((src) => (
                    <div
                      key={src.id}
                      onClick={() => handleOpenSource(src)}
                      className="bg-slate-800/40 hover:bg-slate-700/60 border border-slate-700/50 hover:border-sky-500/30 rounded-lg p-3 transition-all cursor-pointer group/ref relative overflow-hidden"
                    >
                      {/* Glow effect on hover */}
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-sky-500/5 to-transparent -translate-x-full group-hover/ref:translate-x-full transition-transform duration-1000 pointer-events-none"></div>

                      {/* 第一行：cite_key + 链接图标 */}
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <span className="text-xs font-mono font-bold text-sky-400 bg-slate-900/50 border border-slate-700/50 px-1.5 py-0.5 rounded shadow-sm">
                          [{src.cite_key}]
                        </span>
                        {(src.url || src.doi) && (
                          <ExternalLink size={12} className="text-slate-500 group-hover/ref:text-sky-400 flex-shrink-0 mt-0.5 transition-colors" />
                        )}
                      </div>
                      
                      {/* 第二行：标题 */}
                      {src.title && (
                        <div className="text-sm font-medium text-slate-200 leading-snug mb-1.5 group-hover/ref:text-sky-100 transition-colors">
                          {src.title}
                        </div>
                      )}
                      
                      {/* 第三行：作者 + 年份 */}
                      <div className="flex items-center gap-3 text-xs text-slate-400">
                        {src.authors && src.authors.length > 0 && (
                          <span className="flex items-center gap-1">
                            <User size={10} className="opacity-60" />
                            {formatAuthors(src.authors)}
                          </span>
                        )}
                        {src.year && (
                          <span className="flex items-center gap-1">
                            <Calendar size={10} className="opacity-60" />
                            {src.year}
                          </span>
                        )}
                        {src.doc_id && !src.url && (
                          <span className="flex items-center gap-1 text-slate-500">
                            <FileText size={10} />
                            {t('chat.localDoc')}
                          </span>
                        )}
                      </div>
                      {/* 加入对比：仅当有 doc_id（本地）时可用 */}
                      <div className="mt-2 pt-2 border-t border-slate-700/30 flex justify-end">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (src.doc_id) {
                              addComparePreselected(src.doc_id);
                              addToast(t('chat.addedToCompare'), 'info');
                            } else {
                              addToast(t('chat.onlyLocalCompare'), 'info');
                            }
                          }}
                          className="text-xs text-sky-500 hover:text-sky-300 flex items-center gap-1 font-medium transition-colors"
                        >
                          <GitCompareArrows size={10} />
                          {t('chat.addToCompare')}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
      })}
    </div>
  );
}
