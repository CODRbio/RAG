import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowRight,
  GitCompareArrows,
  Loader2,
  Network,
  SearchCheck,
  Sparkles,
  Telescope,
  Trash2,
  UserRoundSearch,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAcademicAssistantStore, useAnalysisPoolStore, useConfigStore, useToastStore } from '../stores';
import { transformMarkdownMediaUrl } from '../utils/mediaUrl';
import type { AcademicAssistantTaskState, AssistantScope, DiscoveryMode } from '../types';

function proseClassName() {
  return 'prose prose-invert prose-sm max-w-none prose-headings:text-sky-300 prose-p:text-slate-300 prose-li:text-slate-300 prose-strong:text-sky-200 prose-a:text-sky-400 prose-a:no-underline hover:prose-a:underline prose-code:bg-slate-900/60 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sky-300 prose-code:border prose-code:border-slate-700/50 prose-pre:bg-slate-950/80 prose-pre:border prose-pre:border-slate-800';
}

const DISCOVERY_ACTIONS: Array<{
  mode: DiscoveryMode;
  title: string;
  icon: typeof Telescope;
  minItems: number;
  description: string;
}> = [
  {
    mode: 'missing-core',
    title: 'Missing Core',
    icon: SearchCheck,
    minItems: 1,
    description: 'Find missing core papers around the current pool.',
  },
  {
    mode: 'forward-tracking',
    title: 'Forward Tracking',
    icon: ArrowRight,
    minItems: 1,
    description: 'Trace newer citing work built on the selected papers.',
  },
  {
    mode: 'experts',
    title: 'Experts',
    icon: UserRoundSearch,
    minItems: 1,
    description: 'Discover high-signal authors connected to the pool.',
  },
  {
    mode: 'institutions',
    title: 'Institutions',
    icon: Telescope,
    minItems: 1,
    description: 'Discover institutions collaborating around the pool.',
  },
];

export function AnalysisWorkspacePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const currentCollection = useConfigStore((s) => s.currentCollection);
  const addToast = useToastStore((s) => s.addToast);
  const poolItems = useAnalysisPoolStore((s) => s.items);
  const removeItem = useAnalysisPoolStore((s) => s.removeItem);
  const clearPool = useAnalysisPoolStore((s) => s.clear);

  const comparisons = useAcademicAssistantStore((s) => s.comparisons);
  const assistantTasks = useAcademicAssistantStore((s) => s.tasks);
  const assistantLoading = useAcademicAssistantStore((s) => s.loadingKeys);
  const comparePapers = useAcademicAssistantStore((s) => s.comparePapers);
  const startDiscoveryTask = useAcademicAssistantStore((s) => s.startDiscoveryTask);
  const streamTask = useAcademicAssistantStore((s) => s.streamTask);

  const [trackedTaskIds, setTrackedTaskIds] = useState<string[]>([]);
  const [activeResult, setActiveResult] = useState<'compare' | string | null>(null);

  const paperUids = useMemo(() => poolItems.map((item) => item.paper_uid).filter(Boolean), [poolItems]);
  const defaultScope: AssistantScope = useMemo(() => {
    if (currentCollection) {
      return { scope_type: 'collection', scope_key: currentCollection };
    }
    return { scope_type: 'global', scope_key: 'global' };
  }, [currentCollection]);

  const compareKey = useMemo(() => [...paperUids].sort().join('|'), [paperUids]);
  const compareResult = compareKey ? comparisons[compareKey] : undefined;

  const handleCompare = async () => {
    if (paperUids.length < 2) return;
    try {
      await comparePapers({
        paper_uids: paperUids,
        scope: defaultScope,
      });
      setActiveResult('compare');
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.compareFailed', 'Compare failed'), 'error');
    }
  };

  const handleDiscovery = async (mode: DiscoveryMode) => {
    try {
      const task = await startDiscoveryTask({
        mode,
        paper_uids: paperUids,
        scope: defaultScope,
      });
      setTrackedTaskIds((prev) => [...new Set([...prev, task.task_id])]);
      setActiveResult(task.task_id);
      void streamTask(task.task_id).catch((error) => {
        addToast(error instanceof Error ? error.message : t('academicAssistant.discoveryFailed', 'Discovery failed'), 'error');
      });
    } catch (error) {
      addToast(error instanceof Error ? error.message : t('academicAssistant.discoveryFailed', 'Discovery failed'), 'error');
    }
  };

  const activeTask = activeResult && activeResult !== 'compare' ? assistantTasks[activeResult] : undefined;
  const visibleTasks = trackedTaskIds
    .map((taskId) => assistantTasks[taskId])
    .filter((task): task is AcademicAssistantTaskState => task != null)
    .sort((left, right) => right.updated_at - left.updated_at);

  return (
    <div className="flex h-full min-h-0 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.12),transparent_34%),linear-gradient(180deg,rgba(15,23,42,0.94),rgba(2,6,23,0.99))]">
      <aside className="w-[320px] shrink-0 border-r border-slate-800/80 bg-slate-950/55 backdrop-blur-md">
        <div className="border-b border-slate-800/80 px-4 py-4">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-teal-300/80">
            <GitCompareArrows size={14} />
            Analysis Workspace
          </div>
          <h1 className="mt-2 text-xl font-semibold text-slate-100">
            {t('analysisWorkspace.title', 'Compare & Discovery')}
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            {t('analysisWorkspace.subtitle', 'Pick a pool of papers once, then run compare, discovery, and graph exploration from the same object set.')}
          </p>
          <div className="mt-4 flex items-center gap-2">
            <button
              type="button"
              onClick={() => navigate('/workspace/graph')}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800"
            >
              <Network size={13} />
              {t('analysisWorkspace.openGraph', 'Open Graph Workspace')}
            </button>
            <button
              type="button"
              onClick={() => clearPool()}
              disabled={poolItems.length === 0}
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-40"
            >
              <Trash2 size={13} />
              {t('analysisWorkspace.clearPool', 'Clear pool')}
            </button>
          </div>
        </div>

        <div className="border-b border-slate-800/80 px-4 py-3">
          <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
            {t('analysisWorkspace.poolLabel', 'Object Pool')}
          </div>
          <div className="mt-2 text-sm text-slate-300">
            {poolItems.length > 0
              ? t('analysisWorkspace.poolCount', {
                  defaultValue: '{{count}} papers selected',
                  count: poolItems.length,
                })
              : t('analysisWorkspace.poolEmpty', 'No papers in the analysis pool yet.')}
          </div>
        </div>

        <div className="h-[calc(100%-11rem)] overflow-y-auto px-4 py-4">
          {poolItems.length === 0 ? (
            <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-500">
              {t('analysisWorkspace.emptyHint', 'Add papers from Scholar or Paper Workspace to start multi-paper analysis.')}
            </div>
          ) : (
            <div className="space-y-2">
              {poolItems.map((item) => (
                <div key={item.paper_uid} className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="line-clamp-2 text-sm font-medium text-slate-100">{item.title}</div>
                  {item.subtitle && <div className="mt-1 text-xs text-slate-500">{item.subtitle}</div>}
                  <div className="mt-3 flex items-center justify-between gap-2">
                    <span className="truncate text-[11px] text-slate-500">{item.paper_uid}</span>
                    <button
                      type="button"
                      onClick={() => removeItem(item.paper_uid)}
                      className="text-xs text-red-300 hover:text-red-200"
                    >
                      {t('common.remove', 'Remove')}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      <main className="flex min-w-0 flex-1">
        <section className="w-[320px] shrink-0 border-r border-slate-800/70 px-4 py-5">
          <div className="space-y-4">
            <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                <GitCompareArrows size={14} />
                {t('analysisWorkspace.compareTitle', 'Compare')}
              </div>
              <p className="text-xs text-slate-500">
                {t('analysisWorkspace.compareHint', 'Generate a narrative summary plus aspect matrix from the current pool.')}
              </p>
              <button
                type="button"
                onClick={() => void handleCompare()}
                disabled={paperUids.length < 2 || assistantLoading[`compare:${compareKey}`]}
                className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
              >
                {assistantLoading[`compare:${compareKey}`] ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
                {t('academicAssistant.compare', 'Compare')}
              </button>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                <Telescope size={14} />
                {t('analysisWorkspace.discoveryTitle', 'Discovery')}
              </div>
              <div className="space-y-2">
                {DISCOVERY_ACTIONS.map((action) => {
                  const Icon = action.icon;
                  return (
                    <button
                      key={action.mode}
                      type="button"
                      onClick={() => void handleDiscovery(action.mode)}
                      disabled={paperUids.length < action.minItems}
                      className="w-full rounded-2xl border border-slate-800 bg-slate-900/70 px-3 py-3 text-left transition hover:border-slate-700 hover:bg-slate-900 disabled:opacity-40"
                    >
                      <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                        <Icon size={14} />
                        {action.title}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">{action.description}</div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="flex min-w-0 flex-1 flex-col">
          <div className="border-b border-slate-800/70 px-6 py-4">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
              {t('analysisWorkspace.outputLabel', 'Output')}
            </div>
            <div className="mt-2 text-lg font-semibold text-slate-100">
              {activeResult === 'compare'
                ? t('analysisWorkspace.compareOutput', 'Compare Result')
                : activeTask?.mode || t('analysisWorkspace.waiting', 'Choose an action')}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-5">
            {activeResult === 'compare' && compareResult ? (
              <div className="space-y-5">
                <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-5">
                  <div className={proseClassName()}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformMarkdownMediaUrl}>
                      {compareResult.narrative}
                    </ReactMarkdown>
                  </div>
                </div>
                {Object.keys(compareResult.comparison_matrix || {}).length > 0 && (
                  <div className="overflow-x-auto rounded-3xl border border-slate-800 bg-slate-950/60 p-4">
                    <table className="w-full text-left text-sm text-slate-300">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-500">
                          <th className="px-3 py-2">{t('academicAssistant.aspect', 'Aspect')}</th>
                          {compareResult.papers.map((paper) => (
                            <th key={paper.paper_uid || paper.paper_id} className="px-3 py-2">
                              {paper.title}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(compareResult.comparison_matrix).map(([aspect, values]) => (
                          <tr key={aspect} className="border-b border-slate-900/70 align-top">
                            <td className="px-3 py-3 font-medium text-slate-100">{aspect}</td>
                            {compareResult.papers.map((paper) => (
                              <td key={`${aspect}-${paper.paper_uid || paper.paper_id}`} className="px-3 py-3">
                                {values[paper.paper_uid || ''] || '—'}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : activeTask ? (
              <div className="space-y-4">
                <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-5">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <div className="text-sm font-medium text-slate-100">
                      {activeTask.task_type}
                      {activeTask.mode ? ` · ${activeTask.mode}` : ''}
                    </div>
                    <span className="text-xs text-slate-400">{activeTask.status}</span>
                  </div>
                  {activeTask.message && <div className="text-xs text-slate-500">{activeTask.message}</div>}
                  {activeTask.error_message && <div className="mt-3 text-sm text-red-300">{activeTask.error_message}</div>}
                  {activeTask.result && (
                    <>
                      {(activeTask.result as { summary_md?: string }).summary_md && (
                        <div className={`mt-4 ${proseClassName()}`}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformMarkdownMediaUrl}>
                            {(activeTask.result as { summary_md?: string }).summary_md || ''}
                          </ReactMarkdown>
                        </div>
                      )}
                      {Array.isArray((activeTask.result as { items?: Array<Record<string, unknown>> }).items) && (
                        <div className="mt-4 grid gap-3 md:grid-cols-2">
                          {((activeTask.result as { items?: Array<Record<string, unknown>> }).items || []).map((item, idx) => (
                            <div key={`${activeTask.task_id}-${idx}`} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-300">
                              <div className="font-medium text-slate-100">
                                {String(item.title || item.paper_uid || item.author || item.institution || `Item ${idx + 1}`)}
                              </div>
                              {'reason' in item && Boolean(item.reason) && (
                                <div className="mt-2 text-xs text-slate-500">{String(item.reason)}</div>
                              )}
                              {'year' in item && Boolean(item.year) && (
                                <div className="mt-2 text-xs text-slate-500">{String(item.year)}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {visibleTasks.length > 1 && (
                  <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-5">
                    <div className="mb-3 text-sm font-medium text-slate-100">{t('analysisWorkspace.recentTasks', 'Recent discovery tasks')}</div>
                    <div className="space-y-2">
                      {visibleTasks
                        .filter((task) => task.task_id !== activeTask.task_id)
                        .map((task) => (
                          <button
                            key={task.task_id}
                            type="button"
                            onClick={() => setActiveResult(task.task_id)}
                            className="block w-full rounded-2xl border border-slate-800 bg-slate-900/70 px-3 py-3 text-left hover:border-slate-700"
                          >
                            <div className="text-sm font-medium text-slate-100">{task.mode || task.task_type}</div>
                            <div className="mt-1 text-xs text-slate-500">{task.status}</div>
                          </button>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-6 text-sm text-slate-500">
                {t('analysisWorkspace.noOutputYet', 'Choose Compare or a Discovery action to render results here.')}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
