import { useRef, useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { MessageSquare, FileSearch, Copy, Download, ExternalLink, FileText, User, Calendar, GitCompareArrows, BookOpen, Database, Globe, BookmarkPlus, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChatStore, useConfigStore, useToastStore, useCompareStore, useScholarStore } from '../../stores';
import type { Source, DeepResearchJobInfo, ResearchDashboardData, CitationAnchor } from '../../types';
import { cancelDeepResearchJob, getDeepResearchJob, streamDeepResearchEvents, listDeepResearchJobs } from '../../api/chat';
import { DEEP_RESEARCH_JOB_KEY } from '../workflow/deep-research/types';
import { getChunkDetail } from '../../api/graph';
import { RetrievalDebugPanel } from './RetrievalDebugPanel';
import { ToolTracePanel } from './ToolTracePanel';
import { AgentDebugPanel } from './AgentDebugPanel';
import { ResearchProgressPanel } from '../research/ResearchProgressPanel';
import { PdfViewerModal } from '../ui/PdfViewerModal';
import { ScholarLibrarySelectModal } from '../scholar/ScholarLibrarySelectModal';
import { logger } from '../../utils/logger';
import { transformMarkdownMediaUrl } from '../../utils/mediaUrl';

/** 格式化消息时间，便于查找：同天只显示时分，否则显示日期+时分 */
function formatMessageTime(iso?: string | null): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const now = new Date();
  const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
  const sameYear = d.getFullYear() === now.getFullYear();
  const pad = (n: number) => String(n).padStart(2, '0');
  if (sameDay) {
    return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
  if (sameYear) {
    return `${d.getMonth() + 1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const PROVIDER_META: Record<string, { label: string; color: string; icon: 'db' | 'globe' }> = {
  local:    { label: 'Local RAG',        color: '#38bdf8', icon: 'db' },
  tavily:   { label: 'Tavily',           color: '#a78bfa', icon: 'globe' },
  google:   { label: 'Google',           color: '#f97316', icon: 'globe' },
  scholar:  { label: 'Google Scholar',   color: '#34d399', icon: 'globe' },
  semantic: { label: 'Semantic Scholar', color: '#facc15', icon: 'globe' },
  ncbi:     { label: 'NCBI PubMed',      color: '#f472b6', icon: 'globe' },
  sonar:    { label: 'Sonar',            color: '#06b6d4', icon: 'globe' },
  web:      { label: 'Google Fetched',   color: '#94a3b8', icon: 'globe' },
};

const UNKNOWN_PROVIDER_LABEL = 'Unknown';

function MiniPieChart({ data, size = 48 }: { data: { label: string; value: number; color: string }[]; size?: number }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return null;
  const r = size / 2;
  const cx = r;
  const cy = r;
  const ir = r * 0.55;
  const paths = data.map((d, idx) => {
    const startAngle = data
      .slice(0, idx)
      .reduce((sum, item) => sum + (item.value / total) * 2 * Math.PI, -Math.PI / 2);
    const angle = (d.value / total) * 2 * Math.PI;
    const endAngle = startAngle + angle;
    const startOuter = { x: cx + r * Math.cos(startAngle), y: cy + r * Math.sin(startAngle) };
    const endOuter = { x: cx + r * Math.cos(endAngle), y: cy + r * Math.sin(endAngle) };
    const startInner = { x: cx + ir * Math.cos(endAngle), y: cy + ir * Math.sin(endAngle) };
    const endInner = { x: cx + ir * Math.cos(startAngle), y: cy + ir * Math.sin(startAngle) };
    const large = angle > Math.PI ? 1 : 0;
    const path = [
      `M ${startOuter.x} ${startOuter.y}`,
      `A ${r} ${r} 0 ${large} 1 ${endOuter.x} ${endOuter.y}`,
      `L ${startInner.x} ${startInner.y}`,
      `A ${ir} ${ir} 0 ${large} 0 ${endInner.x} ${endInner.y}`,
      'Z',
    ].join(' ');
    return <path key={d.label} d={path} fill={d.color} opacity={0.85} />;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="flex-shrink-0">
      {paths}
      <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" className="fill-slate-300 text-[9px] font-bold">
        {total}
      </text>
    </svg>
  );
}

function ProviderRow({ label, counts, pieSize = 40 }: { label: string; counts: Record<string, number>; pieSize?: number }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, c]) => s + c, 0);
  if (total === 0) return null;
  const pieData = entries.map(([prov, count]) => ({
    label: PROVIDER_META[prov]?.label || UNKNOWN_PROVIDER_LABEL,
    value: count,
    color: PROVIDER_META[prov]?.color || '#64748b',
  }));

  return (
    <div className="flex items-center gap-2.5">
      <MiniPieChart data={pieData} size={pieSize} />
      <div className="flex-1 min-w-0">
        <div className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-1">{label}</div>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5">
          {entries.map(([prov, count]) => {
            const meta = PROVIDER_META[prov] || { label: UNKNOWN_PROVIDER_LABEL, color: '#64748b', icon: 'globe' as const };
            const pct = Math.round((count / total) * 100);
            return (
              <span key={prov} className="flex items-center gap-1 text-[10px] text-slate-400 whitespace-nowrap">
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: meta.color }} />
                {meta.icon === 'db' ? <Database size={8} className="opacity-60" /> : <Globe size={8} className="opacity-60" />}
                <span className="text-slate-300 font-medium">{meta.label}</span>
                <span className="text-slate-500">{count} ({pct}%)</span>
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function SourceBreakdownBar({ sources, providerStats }: {
  sources: Source[];
  providerStats?: { chunk_level: Record<string, number>; citation_level: Record<string, number> };
}) {
  if (!sources || sources.length === 0) return null;

  const citeCounts: Record<string, number> = providerStats?.citation_level
    ? { ...providerStats.citation_level }
    : {};
  if (!providerStats?.citation_level) {
    for (const s of sources) {
      const p = s.provider || (s.url ? 'web' : 'local');
      citeCounts[p] = (citeCounts[p] || 0) + 1;
    }
  }

  const chunkCounts = providerStats?.chunk_level;

  return (
    <div className="mb-3 p-2.5 bg-slate-800/60 rounded-lg border border-slate-700/40 space-y-2.5">
      {chunkCounts && Object.keys(chunkCounts).length > 0 && (
        <ProviderRow label="Chunks (Evidence Fragments)" counts={chunkCounts} pieSize={38} />
      )}
      <ProviderRow label="Citations (Articles / Pages)" counts={citeCounts} pieSize={38} />
    </div>
  );
}

function getSourceAnchors(source: Source): CitationAnchor[] {
  const seen = new Set<string>();
  const anchors: CitationAnchor[] = [];
  for (const anchor of source.anchors || []) {
    const chunkId = String(anchor?.chunk_id || '').trim();
    if (!chunkId || seen.has(chunkId)) continue;
    seen.add(chunkId);
    anchors.push({
      chunk_id: chunkId,
      page_num: anchor?.page_num ?? null,
      bbox: Array.isArray(anchor?.bbox) ? anchor.bbox : null,
      snippet: typeof anchor?.snippet === 'string' ? anchor.snippet : null,
    });
  }
  if (anchors.length === 0 && source.chunk_id) {
    anchors.push({
      chunk_id: source.chunk_id,
      page_num: source.page_num ?? null,
      bbox: Array.isArray(source.bbox) ? source.bbox : null,
      snippet: null,
    });
  }
  return anchors;
}

function getPrimaryAnchor(source: Source): CitationAnchor | null {
  return getSourceAnchors(source)[0] || null;
}

export function ChatWindow() {
  const { t } = useTranslation();
  const messages = useChatStore((s) => s.messages);
  const lastEvidenceSummary = useChatStore((s) => s.lastEvidenceSummary);
  const deepResearchActive = useChatStore((s) => s.deepResearchActive);
  const researchDashboard = useChatStore((s) => s.researchDashboard);
  const toolTrace = useChatStore((s) => s.toolTrace);
  const agentDebugMode = useConfigStore((s) => s.ragConfig.agentDebugMode);
  const currentCollection = useConfigStore((s) => s.currentCollection);
  const sessionId = useChatStore((s) => s.sessionId);
  const setShowDeepResearchDialog = useChatStore((s) => s.setShowDeepResearchDialog);
  const setDeepResearchTopic = useChatStore((s) => s.setDeepResearchTopic);
  const setSessionId = useChatStore((s) => s.setSessionId);
  const setCanvasId = useChatStore((s) => s.setCanvasId);
  const setResearchDashboard = useChatStore((s) => s.setResearchDashboard);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const activeResponse = useChatStore((s) => s.activeResponse);
  const setStreamingStep = useChatStore((s) => s.setStreamingStep);
  // LocalDbChoiceDialog 由 App.tsx 全局挂载，ChatWindow 不再处理内联按钮
  const addToast = useToastStore((s) => s.addToast);
  const addComparePreselected = useCompareStore((s) => s.addComparePreselected);
  const libraries = useScholarStore((s) => s.libraries);
  const activeLibraryId = useScholarStore((s) => s.activeLibraryId);
  const loadLibraries = useScholarStore((s) => s.loadLibraries);
  const addSourcesToLibrary = useScholarStore((s) => s.addSourcesToLibrary);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [backgroundJob, setBackgroundJob] = useState<DeepResearchJobInfo | null>(null);
  const [stoppingJobId, setStoppingJobId] = useState<string | null>(null);
  const [showBackgroundLogs, setShowBackgroundLogs] = useState(false);
  const [backgroundEventLines, setBackgroundEventLines] = useState<string[]>([]);
  const [showLibrarySelectModal, setShowLibrarySelectModal] = useState(false);
  const [pendingImportSources, setPendingImportSources] = useState<Source[]>([]);
  const [loadingLibraryChoices, setLoadingLibraryChoices] = useState(false);
  const [importingSources, setImportingSources] = useState(false);
  const [importingMessageId, setImportingMessageId] = useState<string | null>(null);
  const lastBackgroundEventIdRef = useRef(0);
  const trackedBackgroundJobIdRef = useRef<string | null>(null);

  // PDF 溯源 Modal 状态
  const [pdfModal, setPdfModal] = useState<{
    open: boolean;
    pdfUrl: string;
    pageNumber: number;
    bbox?: number[];
    title?: string;
  }>({ open: false, pdfUrl: '', pageNumber: 1 });

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 后台 Deep Research 任务：用 SSE 实时订阅，替代轮询
  useEffect(() => {
    const activeJobId = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (!activeJobId) {
      setBackgroundJob(null);
      setBackgroundEventLines([]);
      setShowBackgroundLogs(false);
      trackedBackgroundJobIdRef.current = null;
      lastBackgroundEventIdRef.current = 0;
      return;
    }

    let cancelled = false;
    const ac = new AbortController();

    const toEventLine = (eventName: string, data: Record<string, unknown>): string => {
      const section = typeof data.section === 'string' ? data.section : '';
      const message = typeof data.message === 'string' ? data.message : '';
      const typ = typeof data.type === 'string' ? data.type : eventName;
      if (section) return `[${typ}] ${section}`;
      if (message) return `[${eventName}] ${message}`;
      return `[${eventName}] ${JSON.stringify(data)}`;
    };

    const mergeJobFromPayload = (prev: DeepResearchJobInfo | null, data: Record<string, unknown>): DeepResearchJobInfo => ({
      ...(prev || {
        job_id: activeJobId,
        topic: '',
        session_id: '',
        canvas_id: '',
        status: 'running',
        current_stage: '',
        message: '',
        error_message: '',
        result_markdown: '',
        result_citations: [],
        result_dashboard: {},
        total_time_ms: 0,
        created_at: 0,
        updated_at: 0,
      }),
      job_id: String(data.job_id || prev?.job_id || activeJobId),
      topic: String(data.topic ?? prev?.topic ?? ''),
      status: String(data.status ?? prev?.status ?? 'running'),
      current_stage: String(data.current_stage ?? prev?.current_stage ?? ''),
      message: String(data.message ?? prev?.message ?? ''),
      canvas_id: String(data.canvas_id ?? prev?.canvas_id ?? ''),
    });

    const clearState = () => {
      if (!cancelled) {
        setBackgroundJob(null);
        setBackgroundEventLines([]);
        setShowBackgroundLogs(false);
        setStreamingStep(null);
      }
      trackedBackgroundJobIdRef.current = null;
      lastBackgroundEventIdRef.current = 0;
      localStorage.removeItem(DEEP_RESEARCH_JOB_KEY);
    };

    (async () => {
      try {
        const job = await getDeepResearchJob(activeJobId);
        if (cancelled || ac.signal.aborted) return;
        const running = ['pending', 'running', 'pausing', 'paused', 'cancelling', 'waiting_review'].includes(job.status);
        if (!running) {
          clearState();
          return;
        }
        trackedBackgroundJobIdRef.current = activeJobId;
        setBackgroundJob(job);

        for await (const { event, data } of streamDeepResearchEvents(
          activeJobId,
          ac.signal,
          lastBackgroundEventIdRef.current,
        )) {
          if (cancelled || ac.signal.aborted) break;
          const eid = typeof data._event_id === 'number' ? data._event_id : null;
          if (eid != null) lastBackgroundEventIdRef.current = Math.max(lastBackgroundEventIdRef.current, eid);

          if (event === 'heartbeat' || event === 'job_status') {
            setBackgroundJob((prev) => mergeJobFromPayload(prev, data));
            const status = String(data.status || '');
            if (status === 'done' || status === 'error' || status === 'cancelled') {
              setStreamingStep(null);
              clearState();
              break;
            }
          } else if (event === 'step') {
            const stepPayload = data as { step?: string | null; label?: string };
            setStreamingStep(
              stepPayload?.step
                ? { step: stepPayload.step, label: stepPayload.label ?? stepPayload.step }
                : null
            );
          } else if (event === 'progress' || event === 'warning' || event === 'waiting_review') {
            setBackgroundEventLines((prev) => [...prev.slice(-9), toEventLine(event, data)]);
          }
        }
      } catch (err) {
        if (!ac.signal.aborted && !cancelled) {
          logger.ui.debug('[ChatWindow] Background job SSE ended', err);
          clearState();
        }
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [setStreamingStep]);

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
    if (source.pdf_url) {
      window.open(source.pdf_url, '_blank');
    } else if (source.url) {
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
    // Always use force=true so a stuck "cancelling" job is terminated immediately
    const isStuck = backgroundJob.status === 'cancelling';
    try {
      setStoppingJobId(backgroundJob.job_id);
      await cancelDeepResearchJob(backgroundJob.job_id, isStuck);
      addToast(t('chat.stopRequested'), 'info');
    } catch {
      addToast(t('chat.stopFailed'), 'error');
    } finally {
      setStoppingJobId(null);
    }
  };

  const handleWakeDeepResearch = useCallback(async (topic?: string) => {
    const topicTrim = (topic || '').trim();
    setDeepResearchTopic(topicTrim || '');
    try {
      const jobs = await listDeepResearchJobs(15);
      const runnable = ['pending', 'running', 'pausing', 'paused', 'cancelling', 'waiting_review'];
      const match = (j: DeepResearchJobInfo) =>
        (sessionId && j.session_id === sessionId) ||
        (topicTrim && String(j.topic || '').trim() === topicTrim);
      const candidates = jobs.filter(match);
      const best = candidates.sort((a, b) => {
        const aRun = runnable.includes(a.status);
        const bRun = runnable.includes(b.status);
        if (aRun !== bRun) return aRun ? -1 : 1;
        return (b.updated_at || 0) - (a.updated_at || 0);
      })[0];
      if (best) {
        localStorage.setItem(DEEP_RESEARCH_JOB_KEY, best.job_id);
        setDeepResearchTopic(best.topic || topicTrim || '');
        if (best.session_id) setSessionId(best.session_id);
        if (best.canvas_id) setCanvasId(best.canvas_id);
        if (best.result_dashboard && Object.keys(best.result_dashboard).length > 0) {
          setResearchDashboard(best.result_dashboard as unknown as ResearchDashboardData);
        }
      }
    } catch (e) {
      logger.ui.debug('[ChatWindow] list jobs for wake failed', e);
    }
    setShowDeepResearchDialog(true);
  }, [sessionId, setDeepResearchTopic, setSessionId, setCanvasId, setResearchDashboard, setShowDeepResearchDialog]);

  const formatAuthors = (authors: string[] | undefined): string => {
    if (!authors || authors.length === 0) return '佚名';
    if (authors.length === 1) return authors[0];
    if (authors.length === 2) return authors.join(' & ');
    return `${authors[0]} et al.`;
  };

  const handleOpenPdfTrace = useCallback(async (src: Source, anchor?: CitationAnchor | null) => {
    if (!src.doc_id) return;
    const targetAnchor = anchor || getPrimaryAnchor(src);
    try {
      // 先尝试从 source 本身获取 bbox/page，如已有则直接使用
      let bbox = targetAnchor?.bbox ?? src.bbox;
      let page = targetAnchor?.page_num ?? src.page_num ?? undefined;
      const chunkId = targetAnchor?.chunk_id || src.chunk_id || String(src.id);

      // 若锚点缺少 bbox 或 page，通过 chunk API 补取
      if ((!bbox || bbox.length < 4 || !page) && chunkId) {
        const detail = await getChunkDetail({
          chunk_id: chunkId,
          collection: currentCollection || undefined,
          paper_id: src.doc_id,
        });
        const rawBbox = detail.bbox;
        if (Array.isArray(rawBbox) && rawBbox.length > 0) {
          if (Array.isArray(rawBbox[0]) && (rawBbox[0] as number[]).length >= 4) {
            // nested format: [[x0,y0,x1,y1], ...]
            bbox = rawBbox[0] as number[];
          } else if (typeof rawBbox[0] === 'number' && rawBbox.length >= 4) {
            // flat format: [x0,y0,x1,y1]
            bbox = rawBbox as number[];
          }
        }
        page = detail.page ?? undefined;
      }

      const apiBase = import.meta.env.VITE_API_BASE_URL || '/api';
      const collectionQuery = currentCollection ? `?collection=${encodeURIComponent(currentCollection)}` : '';
      const pdfUrl = `${apiBase}/graph/pdf/${encodeURIComponent(src.doc_id)}${collectionQuery}`;

      setPdfModal({
        open: true,
        pdfUrl,
        pageNumber: page || 1,
        bbox: bbox || undefined,
        title: src.title || src.doc_id,
      });
    } catch {
      const apiBase = import.meta.env.VITE_API_BASE_URL || '/api';
      const collectionQuery = currentCollection ? `?collection=${encodeURIComponent(currentCollection)}` : '';
      const pdfUrl = `${apiBase}/graph/pdf/${encodeURIComponent(src.doc_id)}${collectionQuery}`;
      setPdfModal({
        open: true,
        pdfUrl,
        pageNumber: (targetAnchor?.page_num || src.page_num || 1),
        bbox: targetAnchor?.bbox || src.bbox || undefined,
        title: src.title || src.doc_id,
      });
      addToast('未获取到精确 chunk 坐标，已打开原文 PDF', 'info');
    }
  }, [addToast, currentCollection]);

  const getDefaultScholarLibraryId = useCallback((): number | null => {
    if (activeLibraryId != null && libraries.some((lib) => lib.id === activeLibraryId)) {
      return activeLibraryId;
    }
    const permanent = libraries.filter((lib) => !lib.is_temporary);
    if (permanent.length > 0) return permanent[0].id;
    return libraries[0]?.id ?? null;
  }, [activeLibraryId, libraries]);

  const handleImportSourcesClick = useCallback(async (messageKey: string, sources: Source[]) => {
    const candidateSources = (sources || []).filter((s) => s && (s.title || s.doi || s.url || s.pdf_url || s.doc_id));
    if (candidateSources.length === 0) {
      addToast('当前回答没有可导入的文献信息', 'info');
      return;
    }
    setImportingMessageId(messageKey);
    setLoadingLibraryChoices(true);
    try {
      await loadLibraries();
      setPendingImportSources(candidateSources);
      setShowLibrarySelectModal(true);
    } catch {
      addToast('加载文献库失败，请稍后重试', 'error');
    } finally {
      setLoadingLibraryChoices(false);
      setImportingMessageId(null);
    }
  }, [addToast, loadLibraries]);

  const handleConfirmImportSources = useCallback(async (libraryId: number) => {
    if (!pendingImportSources.length) return;
    setImportingSources(true);
    try {
      const res = await addSourcesToLibrary(pendingImportSources, libraryId);
      if (!res) {
        addToast('导入失败，请重试', 'error');
        return;
      }
      const ignored = Math.max(0, (res.requested || 0) - res.added);
      if (ignored > 0) {
        addToast(`导入完成：新增 ${res.added}，忽略重复 ${ignored}`, 'success');
      } else {
        addToast(`导入完成：新增 ${res.added}`, 'success');
      }
      setShowLibrarySelectModal(false);
      setPendingImportSources([]);
    } finally {
      setImportingSources(false);
    }
  }, [addSourcesToLibrary, addToast, pendingImportSources]);

  const showBackgroundBanner = Boolean(backgroundJob) && !deepResearchActive && !researchDashboard;

  const backgroundBanner = showBackgroundBanner ? (
    <div className="border rounded-lg bg-sky-900/30 border-sky-500/30 px-3 py-2 text-xs text-sky-300 shadow-[0_0_10px_rgba(14,165,233,0.1)]">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="font-medium truncate">
            {backgroundJob?.status === 'cancelling'
              ? t('chat.bgResearchStopping')
              : backgroundJob?.status === 'paused'
                ? t('chat.bgResearchPaused')
                : backgroundJob?.status === 'pausing'
                  ? t('chat.bgResearchPausing')
                  : t('chat.bgResearchRunning')}
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
              handleWakeDeepResearch(backgroundJob.topic || '');
            }}
            className="px-2 py-1 rounded-md border border-sky-500/30 text-sky-400 hover:bg-sky-500/10 transition-colors"
          >
            {t('chat.wakeTask')}
          </button>
          <button
            onClick={handleForceStopBackgroundJob}
            disabled={!backgroundJob || stoppingJobId === backgroundJob?.job_id}
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
      {/* Agent 工具调用轨迹 / Debug 面板 */}
      {(() => {
        const lastMsg = messages[messages.length - 1];
        const debugData = lastMsg?.agentDebug;
        if (agentDebugMode && debugData) {
          return <AgentDebugPanel data={debugData} />;
        }
        if (toolTrace && toolTrace.length > 0) {
          return <ToolTracePanel trace={toolTrace} />;
        }
        return null;
      })()}
      {/* Deep Research 进度面板 */}
      {(deepResearchActive || researchDashboard) && (
        <ResearchProgressPanel dashboard={researchDashboard} isActive={deepResearchActive} />
      )}
      {!deepResearchActive && researchDashboard && (
        <div className="border rounded-lg bg-indigo-900/20 border-indigo-500/30 p-3 text-sm text-indigo-200">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-indigo-200/90">{t('chat.bgResearchRunning')}</div>
            <button
              onClick={() => handleWakeDeepResearch(researchDashboard?.topic || '')}
              className="px-3 py-1.5 text-xs rounded-md bg-indigo-600/20 border border-indigo-500/30 text-indigo-200 hover:bg-indigo-600/40 transition-colors"
            >
              {t('chat.wakeTask')}
            </button>
          </div>
        </div>
      )}
      {!deepResearchActive && researchDashboard && (researchDashboard.coverage_gaps?.length ?? 0) > 0 && (
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
        const isInlineStatusMessage =
          msg.role === 'assistant'
          && activeResponse?.surface === 'chat'
          && activeResponse?.targetMessageId === msg.id
          && !activeResponse?.hasVisibleOutput;
        const displayContent =
          isStreaming && idx === messages.length - 1
            ? msg.content.replace(/\[([a-fA-F0-9]{8})\]/g, '[⏳...]')
            : msg.content;
        if (isInlineStatusMessage) {
          return (
            <div
              key={msg.id || idx}
              className="flex justify-start animate-in slide-in-from-bottom-2 duration-300"
            >
              <div className="rounded-full px-3 py-1.5 text-xs font-medium text-slate-400 bg-slate-800/45 border border-slate-700/50 inline-flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
                {activeResponse.stepLabel || t('chat.thinking', 'Thinking')}
              </div>
            </div>
          );
        }
        return (
        <div
          key={msg.id || idx}
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
            {/* 每条消息的提问/回复时间 */}
            {msg.timestamp && (
              <div
                className={`text-[11px] mb-2 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} ${
                  msg.role === 'user' ? 'text-white/70' : 'text-slate-500'
                }`}
                title={msg.timestamp}
              >
                {formatMessageTime(msg.timestamp)}
              </div>
            )}
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
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  urlTransform={transformMarkdownMediaUrl}
                >
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
                <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 flex items-center justify-between gap-2 mb-3">
                  <span className="flex items-center gap-1.5">
                    <FileSearch size={12} className="text-sky-500" /> {t('chat.references')} ({msg.sources.length})
                  </span>
                  <button
                    type="button"
                    onClick={() => handleImportSourcesClick(msg.id || `${idx}`, msg.sources || [])}
                    disabled={importingSources || importingMessageId === (msg.id || `${idx}`)}
                    className="text-[10px] normal-case px-2 py-1 rounded-md border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50 inline-flex items-center gap-1"
                    title="将本轮引用文献批量导入文献库"
                  >
                    {importingMessageId === (msg.id || `${idx}`) ? <Loader2 size={11} className="animate-spin" /> : <BookmarkPlus size={11} />}
                    导入文献库
                  </button>
                </div>
                <SourceBreakdownBar sources={msg.sources} providerStats={msg.providerStats} />
                <div className="space-y-2">
                  {msg.sources.map((src) => {
                    const anchors = getSourceAnchors(src);
                    const primaryAnchor = anchors[0] || null;
                    return (
                      <div
                        key={src.id}
                        onClick={() => handleOpenSource(src)}
                        className="bg-slate-800/40 hover:bg-slate-700/60 border border-slate-700/50 hover:border-sky-500/30 rounded-lg p-3 transition-all cursor-pointer group/ref relative overflow-hidden"
                      >
                        {/* Glow effect on hover */}
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-sky-500/5 to-transparent -translate-x-full group-hover/ref:translate-x-full transition-transform duration-1000 pointer-events-none"></div>

                      {/* 第一行：cite_key + provider badge + 链接图标 */}
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span className="text-xs font-mono font-bold text-sky-400 bg-slate-900/50 border border-slate-700/50 px-1.5 py-0.5 rounded shadow-sm">
                            [{src.cite_key}]
                          </span>
                          {(() => {
                            const prov = src.provider || (src.url ? 'web' : 'local');
                            const meta = PROVIDER_META[prov] || { label: UNKNOWN_PROVIDER_LABEL, color: '#64748b' };
                            return (
                              <span
                                className="text-[9px] font-medium px-1.5 py-0.5 rounded-full border"
                                style={{ color: meta.color, borderColor: meta.color + '40', backgroundColor: meta.color + '15' }}
                              >
                                {meta.label}
                              </span>
                            );
                          })()}
                        </div>
                        {(src.pdf_url || src.url || src.doi) && (
                          <ExternalLink size={12} className="text-slate-500 group-hover/ref:text-sky-400 flex-shrink-0 mt-0.5 transition-colors" />
                        )}
                      </div>
                      
                      {/* 第二行：标题 */}
                      {src.title && (
                        <div className="text-sm font-medium text-slate-200 leading-snug mb-1.5 group-hover/ref:text-sky-100 transition-colors">
                          {src.title}
                        </div>
                      )}
                      
                      {/* 第三行：作者 + 年份 + DOI */}
                      <div className="flex items-center gap-3 text-xs text-slate-400 flex-wrap">
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
                        {src.doi && (
                          <span className="flex items-center gap-1 text-slate-500 font-mono">
                            DOI: {src.doi}
                          </span>
                        )}
                        {src.doc_id && !src.url && !src.pdf_url && (
                          <span className="flex items-center gap-1 text-slate-500">
                            <FileText size={10} />
                            {t('chat.localDoc')}
                          </span>
                        )}
                      </div>
                      {src.type === 'local' && src.doc_id && anchors.length >= 1 && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {anchors.map((anchor, anchorIdx) => (
                            <button
                              key={anchor.chunk_id}
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleOpenPdfTrace(src, anchor);
                              }}
                              className="px-2 py-1 rounded-md border border-amber-500/30 bg-amber-500/10 text-[10px] text-amber-200 hover:bg-amber-500/20 transition-colors"
                              title={anchor.snippet || undefined}
                            >
                              {anchors.length > 1 ? `片段 ${anchorIdx + 1}` : '原文定位'}{anchor.page_num ? ` · P${anchor.page_num}` : ''}
                            </button>
                          ))}
                        </div>
                      )}
                      {/* 操作栏：溯源原文 + 加入对比 */}
                      <div className="mt-2 pt-2 border-t border-slate-700/30 flex justify-end gap-3">
                        {/* 溯源原文：仅本地文档且有 doc_id 时可用 */}
                        {src.type === 'local' && src.doc_id && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleOpenPdfTrace(src, primaryAnchor);
                            }}
                            className="text-xs text-amber-500 hover:text-amber-300 flex items-center gap-1 font-medium transition-colors"
                          >
                            <BookOpen size={10} />
                            📄 溯源原文
                          </button>
                        )}
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
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      );
      })}

      {/* PDF 溯源 Modal */}
      <PdfViewerModal
        key={`${pdfModal.pdfUrl}|${pdfModal.pageNumber}|${pdfModal.bbox?.join(',') || ''}`}
        open={pdfModal.open}
        onClose={() => setPdfModal((prev) => ({ ...prev, open: false }))}
        pdfUrl={pdfModal.pdfUrl}
        pageNumber={pdfModal.pageNumber}
        bbox={pdfModal.bbox}
        title={pdfModal.title}
      />
      <ScholarLibrarySelectModal
        open={showLibrarySelectModal}
        libraries={libraries}
        defaultLibraryId={getDefaultScholarLibraryId()}
        importCount={pendingImportSources.length}
        loading={loadingLibraryChoices || importingSources}
        timeoutSeconds={10}
        onClose={() => {
          if (importingSources) return;
          setShowLibrarySelectModal(false);
          setPendingImportSources([]);
        }}
        onConfirm={handleConfirmImportSources}
      />
    </div>
  );
}
