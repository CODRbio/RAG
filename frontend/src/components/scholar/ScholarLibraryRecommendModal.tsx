import { useState, useMemo, useRef, useEffect } from 'react';
import { FileSearch, Loader2, ExternalLink, MapPin, BookOpen, AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Modal } from '../ui/Modal';
import {
  submitRecommendTask,
  streamScholarTaskEvents,
  type LibraryRecommendResponse,
  type LibraryRecommendItem,
  type ScholarLibraryPaper,
} from '../../api/scholar';

interface Props {
  open: boolean;
  onClose: () => void;
  libraryId: number;
  collection: string;
  libraryPapers: ScholarLibraryPaper[];
  filteredPaperIds: number[];
  onLocatePaper?: (paperId: number) => void;
}

type Scope = 'all' | 'filtered';

export function ScholarLibraryRecommendModal({
  open,
  onClose,
  libraryId,
  collection,
  libraryPapers,
  filteredPaperIds,
  onLocatePaper,
}: Props) {
  const { t } = useTranslation();
  const [question, setQuestion] = useState('');
  const [scope, setScope] = useState<Scope>('all');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LibraryRecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progressLogs, setProgressLogs] = useState<string[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  // Compute candidate paper ids for the selected scope
  const candidateIds = useMemo((): number[] | undefined => {
    if (scope === 'all') return undefined;
    return filteredPaperIds;
  }, [scope, filteredPaperIds]);

  // Pre-compute stats: total in scope, eligible (in_collection for current collection), excluded
  const scopeStats = useMemo(() => {
    const scopePapers =
      scope === 'all'
        ? libraryPapers
        : libraryPapers.filter((p) => filteredPaperIds.includes(p.id));
    const eligible = scopePapers.filter((p) => p.in_collection === true).length;
    return {
      total: scopePapers.length,
      eligible,
      excluded: scopePapers.length - eligible,
    };
  }, [scope, libraryPapers, filteredPaperIds]);

  const handleSubmit = async () => {
    if (!question.trim() || loading) return;
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);
    setProgressLogs([]);
    setTaskId(null);
    try {
      const startRes = await submitRecommendTask(libraryId, {
        question: question.trim(),
        collection,
        candidate_library_paper_ids: candidateIds,
        top_k: 10,
      });
      const tid = startRes.task_id;
      setTaskId(tid);
      setProgressLogs((prev) => [...prev, t('scholar.recommendProgressStarted', '任务已提交，正在检索…')]);

      for await (const { event, data } of streamScholarTaskEvents(
        tid,
        abortRef.current.signal,
        '-',
      )) {
        if (event === 'progress') {
          const msg = (data?.message as string) || (data?.stage as string) || '…';
          setProgressLogs((prev) => [...prev, msg]);
        }
        if (event === 'done') {
          const payload = data as unknown as LibraryRecommendResponse;
          setResult(payload);
          setProgressLogs((prev) => [...prev, t('scholar.recommendProgressDone', '完成')]);
          break;
        }
        if (event === 'error') {
          const msg = (data?.message as string) || t('scholar.recommendError', '推荐失败');
          setError(msg);
          setProgressLogs((prev) => [...prev, msg]);
          break;
        }
      }
    } catch (e: unknown) {
      if ((e as { name?: string })?.name === 'AbortError') return;
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
      setTaskId(null);
      abortRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t('scholar.recommendTitle')}
      icon={<FileSearch size={18} className="text-amber-400" />}
      maxWidth="max-w-2xl"
      variant="dark"
    >
      {/* Question input */}
      <div className="mt-1 mb-3">
        <textarea
          className="w-full rounded-lg bg-slate-800 border border-slate-600 text-slate-100 placeholder-slate-500 px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/70"
          rows={3}
          placeholder={t('scholar.recommendPlaceholder')}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
      </div>

      {/* Scope toggle + stats */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-1 rounded-lg bg-slate-800 p-0.5 border border-slate-700">
          <button
            type="button"
            onClick={() => setScope('all')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
              scope === 'all'
                ? 'bg-amber-600/80 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t('scholar.recommendScopeAll')}
          </button>
          <button
            type="button"
            onClick={() => setScope('filtered')}
            disabled={filteredPaperIds.length === 0}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-colors disabled:opacity-40 ${
              scope === 'filtered'
                ? 'bg-amber-600/80 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t('scholar.recommendScopeFiltered')}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span>{t('scholar.recommendStatsTotal', { total: scopeStats.total })}</span>
          <span className="text-teal-400">
            {t('scholar.recommendStatsEligible', { count: scopeStats.eligible })}
          </span>
          {scopeStats.excluded > 0 && (
            <span className="text-slate-500">
              {t('scholar.recommendStatsExcluded', { count: scopeStats.excluded })}
            </span>
          )}
        </div>
      </div>

      {/* Submit */}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={loading || !question.trim() || scopeStats.eligible === 0}
        className="w-full flex items-center justify-center gap-2 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2 transition-colors mb-3"
      >
        {loading ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <FileSearch size={14} />
        )}
        {t('scholar.recommendRun')}
      </button>

      {/* Progress / 信息流 */}
      {loading && progressLogs.length > 0 && (
        <div className="mb-3 rounded-lg bg-slate-800/80 border border-slate-700 max-h-24 overflow-y-auto px-3 py-2">
          <p className="text-xs font-medium text-slate-400 mb-1">{t('scholar.recommendProgressTitle', '进度')}</p>
          <ul className="space-y-0.5 text-xs text-slate-300">
            {progressLogs.map((line, idx) => (
              <li key={idx}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-900/30 border border-red-700/50 p-3 mb-3 text-sm text-red-300">
          <AlertCircle size={15} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          {result.recommendations.length === 0 ? (
            <p className="text-center text-slate-500 text-sm py-4">{t('scholar.recommendEmpty')}</p>
          ) : (
            <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
              {result.recommendations.map((item, idx) => (
                <RecommendResultItem
                  key={item.library_paper_id}
                  item={item}
                  rank={idx + 1}
                  isTop={idx === 0}
                  onLocate={onLocatePaper}
                />
              ))}
            </div>
          )}

          {/* Footer stats */}
          <p className="mt-3 text-xs text-slate-600 text-right">
            {t('scholar.recommendStatsTotal', { total: result.total_candidates })} ·{' '}
            {t('scholar.recommendStatsEligible', { count: result.eligible_candidates })} ·{' '}
            {t('scholar.recommendStatsExcluded', { count: result.excluded_not_ingested })}
          </p>
        </div>
      )}
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Single result item
// ---------------------------------------------------------------------------

interface ResultItemProps {
  item: LibraryRecommendItem;
  rank: number;
  isTop: boolean;
  onLocate?: (paperId: number) => void;
}

function RecommendResultItem({ item, rank, isTop, onLocate }: ResultItemProps) {
  const { t } = useTranslation();

  return (
    <div
      className={`rounded-lg border p-3 ${
        isTop
          ? 'border-amber-500/50 bg-amber-900/20'
          : 'border-slate-700/60 bg-slate-800/50'
      }`}
    >
      <div className="flex items-start gap-2">
        {/* Rank badge */}
        <span
          className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold mt-0.5 ${
            isTop ? 'bg-amber-500 text-white' : 'bg-slate-700 text-slate-300'
          }`}
        >
          {rank}
        </span>

        <div className="flex-1 min-w-0">
          {/* Title row */}
          <div className="flex items-start justify-between gap-2">
            <p className={`text-sm font-medium leading-snug ${isTop ? 'text-amber-100' : 'text-slate-200'}`}>
              {isTop && (
                <span className="inline-flex items-center gap-1 text-xs text-amber-400 font-normal mr-1.5 align-middle">
                  <BookOpen size={11} />
                  {t('scholar.recommendTopMatch')}
                </span>
              )}
              {item.title}
            </p>
            {/* Score badge */}
            <span className="shrink-0 text-xs text-slate-400 bg-slate-700/70 rounded px-1.5 py-0.5 font-mono">
              {(item.best_chunk_score * 100).toFixed(0)}
            </span>
          </div>

          {/* Metadata row */}
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-0.5 text-xs text-slate-500">
            {item.year && <span>{item.year}</span>}
            {item.venue && <span className="truncate max-w-[200px]">{item.venue}</span>}
            {item.impact_factor != null && (
              <span className="text-sky-400">IF {item.impact_factor.toFixed(1)}</span>
            )}
            <span className="text-slate-600">
              {t('scholar.recommendMatchedChunks', { count: item.matched_chunks })}
            </span>
          </div>

          {/* Snippets */}
          {item.snippets.length > 0 && (
            <div className="mt-1.5 space-y-1">
              {item.snippets.slice(0, 2).map((snippet, si) => (
                <p
                  key={si}
                  className="text-xs text-slate-500 line-clamp-2 italic border-l-2 border-slate-700 pl-2"
                >
                  {snippet}
                </p>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-2">
            {onLocate && (
              <button
                type="button"
                onClick={() => onLocate(item.library_paper_id)}
                className="inline-flex items-center gap-1 text-xs text-teal-400 hover:text-teal-300 transition-colors"
              >
                <MapPin size={11} />
                {t('scholar.recommendLocate')}
              </button>
            )}
            {item.doi && (
              <a
                href={`https://doi.org/${item.doi}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300 transition-colors"
              >
                <ExternalLink size={11} />
                DOI
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
