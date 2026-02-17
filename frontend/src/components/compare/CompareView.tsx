import { useCallback, useEffect, useState } from 'react';
import { Loader2, Plus, X, GitCompareArrows, AlertCircle, BookOpen, Library } from 'lucide-react';
import {
  listAvailablePapers,
  listCompareCandidates,
  comparePapers,
  type PaperSummary,
  type CompareCandidate,
  type CompareResponse,
} from '../../api/compare';
import { useChatStore } from '../../stores/useChatStore';
import { useCompareStore } from '../../stores/useCompareStore';

const PAGE_SIZE = 12;
type Mode = 'citation' | 'library';

export function CompareView() {
  const sessionId = useChatStore((s) => s.sessionId);
  const { comparePreselectedPaperIds, clearComparePreselected } = useCompareStore();

  const [mode, setMode] = useState<Mode>('citation');
  const [candidates, setCandidates] = useState<CompareCandidate[]>([]);
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [papersTotal, setPapersTotal] = useState(0);
  const [libraryOffset, setLibraryOffset] = useState(0);
  const [libraryQ, setLibraryQ] = useState('');
  const [librarySearchInput, setLibrarySearchInput] = useState('');

  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [loadingPapers, setLoadingPapers] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState('');

  // Merge preselected into selected when user lands with preselected IDs (e.g. from chat quick-add), then clear
  useEffect(() => {
    if (comparePreselectedPaperIds.length === 0) return;
    setSelected((prev) => {
      const next = new Set([...prev, ...comparePreselectedPaperIds]);
      return [...next].slice(0, 5);
    });
    clearComparePreselected();
  }, [comparePreselectedPaperIds.length, clearComparePreselected]);

  const fetchCandidates = useCallback(() => {
    if (!sessionId) {
      setCandidates([]);
      setLoadingCandidates(false);
      return;
    }
    setLoadingCandidates(true);
    listCompareCandidates(sessionId, { scope: 'session', limit: 200, offset: 0 })
      .then((res) => {
        setCandidates(res.candidates);
      })
      .catch(() => {
        setCandidates([]);
      })
      .finally(() => setLoadingCandidates(false));
  }, [sessionId]);

  useEffect(() => {
    if (mode === 'citation') fetchCandidates();
  }, [mode, fetchCandidates]);

  const fetchPapers = useCallback(() => {
    setLoadingPapers(true);
    listAvailablePapers({ limit: PAGE_SIZE, offset: libraryOffset, q: libraryQ || undefined })
      .then((res) => {
        setPapers(res.papers);
        setPapersTotal(res.total);
      })
      .catch(() => {
        setPapers([]);
        setPapersTotal(0);
      })
      .finally(() => setLoadingPapers(false));
  }, [libraryOffset, libraryQ]);

  useEffect(() => {
    if (mode === 'library') fetchPapers();
  }, [mode, fetchPapers]);

  const togglePaper = (pid: string, selectable: boolean) => {
    if (!selectable) return;
    setSelected((prev) =>
      prev.includes(pid) ? prev.filter((p) => p !== pid) : prev.length < 5 ? [...prev, pid] : prev,
    );
  };

  const runCompare = async () => {
    if (selected.length < 2 || selected.length > 5) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await comparePapers({ paper_ids: selected });
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || '对比失败');
    } finally {
      setLoading(false);
    }
  };

  const applyLibrarySearch = () => {
    setLibraryQ(librarySearchInput.trim());
    setLibraryOffset(0);
  };

  const canAddMore = selected.length < 5;
  const canRun = selected.length >= 2 && selected.length <= 5;

  // Empty state when no session and citation mode, or no papers in library
  if (mode === 'citation' && !sessionId && !loadingCandidates) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
        <BookOpen size={40} />
        <p className="text-sm">暂无对话引文，请先在对话中产生引用后再使用「对话引文」</p>
        <button
          type="button"
          onClick={() => setMode('library')}
          className="text-sm text-blue-600 hover:underline"
        >
          切换到本地文库选择论文
        </button>
      </div>
    );
  }

  if (mode === 'library' && !loadingPapers && papers.length === 0 && papersTotal === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
        <AlertCircle size={40} />
        <p className="text-sm">无可用论文，请先通过 Ingest 导入并解析论文</p>
        <button
          type="button"
          onClick={() => setMode('citation')}
          className="text-sm text-blue-600 hover:underline"
        >
          切换到对话引文
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* Mode tabs */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white px-6 pt-3">
        <div className="flex gap-1 mb-3">
          <button
            type="button"
            onClick={() => setMode('citation')}
            className={`px-3 py-1.5 text-sm rounded-t-lg flex items-center gap-1.5 ${
              mode === 'citation'
                ? 'bg-gray-100 text-gray-800 font-medium border border-b-0 border-gray-200'
                : 'text-gray-500 hover:bg-gray-50 border border-transparent'
            }`}
          >
            <BookOpen size={14} />
            对话引文
          </button>
          <button
            type="button"
            onClick={() => setMode('library')}
            className={`px-3 py-1.5 text-sm rounded-t-lg flex items-center gap-1.5 ${
              mode === 'library'
                ? 'bg-gray-100 text-gray-800 font-medium border border-b-0 border-gray-200'
                : 'text-gray-500 hover:bg-gray-50 border border-transparent'
            }`}
          >
            <Library size={14} />
            本地文库
          </button>
        </div>
      </div>

      {/* Top bar: selection + run + card list — 限制高度并内部滚动，保证下方结果区可见 */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white px-6 py-4 flex flex-col max-h-[45vh] min-h-0">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-gray-700 font-medium">
            <GitCompareArrows size={18} />
            <span>多文档对比</span>
            <span className="text-xs text-gray-400 ml-1">选择 2-5 篇论文</span>
          </div>
          <button
            onClick={runCompare}
            disabled={!canRun || loading}
            className="px-4 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <GitCompareArrows size={14} />}
            生成对比
          </button>
        </div>

        {/* Selected chips */}
        <div className="flex flex-wrap gap-2 mb-3">
          {selected.map((pid) => (
            <span
              key={pid}
              className="inline-flex items-center gap-1 px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-full"
            >
              {pid.length > 30 ? pid.slice(0, 28) + '…' : pid}
              <button type="button" onClick={() => togglePaper(pid, true)} className="hover:text-red-500">
                <X size={12} />
              </button>
            </span>
          ))}
        </div>

        {/* Library: search + pagination */}
        {mode === 'library' && (
          <div className="flex flex-wrap items-center gap-2 mb-3">
            <input
              type="text"
              placeholder="搜索标题或 paper_id"
              value={librarySearchInput}
              onChange={(e) => setLibrarySearchInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && applyLibrarySearch()}
              className="px-2.5 py-1.5 border border-gray-200 rounded text-sm w-48"
            />
            <button
              type="button"
              onClick={applyLibrarySearch}
              className="px-2.5 py-1.5 bg-gray-100 text-gray-700 text-sm rounded hover:bg-gray-200"
            >
              搜索
            </button>
            <span className="text-xs text-gray-400">
              共 {papersTotal} 篇 · 第 {libraryOffset + 1}-{Math.min(libraryOffset + PAGE_SIZE, papersTotal)} 条
            </span>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={() => setLibraryOffset((o) => Math.max(0, o - PAGE_SIZE))}
                disabled={libraryOffset === 0}
                className="px-2 py-1 text-sm rounded border border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                上一页
              </button>
              <button
                type="button"
                onClick={() => setLibraryOffset((o) => o + PAGE_SIZE)}
                disabled={libraryOffset + PAGE_SIZE >= papersTotal}
                className="px-2 py-1 text-sm rounded border border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                下一页
              </button>
            </div>
          </div>
        )}

        {/* Card list: citation candidates or library papers — 可滚动 */}
        {mode === 'citation' && (
          <>
            {loadingCandidates ? (
              <div className="flex justify-center py-4">
                <Loader2 size={20} className="animate-spin text-gray-400" />
              </div>
            ) : (
              <div className="flex gap-2 flex-wrap pb-1 overflow-y-auto min-h-0">
                {candidates.map((c) => {
                  const isSelected = selected.includes(c.paper_id);
                  const selectable = c.is_local_ready && canAddMore;
                  return (
                    <button
                      key={c.paper_id}
                      type="button"
                      onClick={() => togglePaper(c.paper_id, selectable)}
                      disabled={!selectable && !isSelected}
                      title={!c.is_local_ready ? '未在本地文库，无法加入对比' : undefined}
                      className={`flex-shrink-0 w-48 text-left p-2.5 rounded-lg border text-xs transition-all ${
                        isSelected
                          ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-200'
                          : !c.is_local_ready
                            ? 'border-gray-200 bg-gray-50 opacity-75 cursor-not-allowed'
                            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="font-medium text-gray-800 truncate">{c.title || c.paper_id}</div>
                      <div className="flex items-center gap-1.5 mt-0.5 text-gray-500">
                        {c.year && <span>{c.year}</span>}
                        <span>引用 {c.citation_count} 次</span>
                      </div>
                      <div className="text-gray-500 mt-1 line-clamp-2">{c.abstract || '无摘要'}</div>
                      {!c.is_local_ready && (
                        <div className="mt-1 text-amber-600">未在本地，无法对比</div>
                      )}
                      {c.is_local_ready && !isSelected && canAddMore && (
                        <div className="mt-1.5 text-blue-500 flex items-center gap-0.5">
                          <Plus size={10} /> 选择
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </>
        )}

        {mode === 'library' && (
          <>
            {loadingPapers ? (
              <div className="flex justify-center py-4">
                <Loader2 size={20} className="animate-spin text-gray-400" />
              </div>
            ) : (
              <div className="flex gap-2 flex-wrap pb-1 overflow-y-auto min-h-0">
                {papers.map((p) => {
                  const isSelected = selected.includes(p.paper_id);
                  return (
                    <button
                      key={p.paper_id}
                      type="button"
                      onClick={() => togglePaper(p.paper_id, canAddMore)}
                      className={`flex-shrink-0 w-48 text-left p-2.5 rounded-lg border text-xs transition-all ${
                        isSelected
                          ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-200'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="font-medium text-gray-800 truncate">{p.title || p.paper_id}</div>
                      {p.year && <div className="text-gray-400 mt-0.5">{p.year}</div>}
                      <div className="text-gray-500 mt-1 line-clamp-2">{p.abstract || '无摘要'}</div>
                      {!isSelected && canAddMore && (
                        <div className="mt-1.5 text-blue-500 flex items-center gap-0.5">
                          <Plus size={10} /> 选择
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>

      {/* Result area — 始终占据剩余空间并独立滚动 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4 bg-gray-50">
        {error && (
          <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg">{error}</div>
        )}

        {result && (
          <>
            {result.narrative && (
              <div className="mb-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
                <h3 className="text-sm font-medium text-gray-700 mb-2">综合分析</h3>
                <p className="text-sm text-gray-600 leading-relaxed">{result.narrative}</p>
              </div>
            )}

            {(result.papers?.length ?? 0) > 0 && (
              <div className="mb-4 p-3 bg-white rounded-lg border border-gray-200 text-sm text-gray-600">
                <span className="font-medium text-gray-700">已对比论文：</span>
                {result.papers!.map((p) => p.title || p.paper_id).join('、')}
              </div>
            )}

            {Object.keys(result.comparison_matrix || {}).length > 0 && (
              <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-4 py-2.5 font-medium text-gray-600 w-32">维度</th>
                      {result.papers.map((p) => (
                        <th key={p.paper_id} className="text-left px-4 py-2.5 font-medium text-gray-700">
                          <div className="truncate max-w-[200px]">{p.title || p.paper_id}</div>
                          {p.year && <span className="text-xs text-gray-400 font-normal ml-1">{p.year}</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.comparison_matrix).map(([aspect, cells], idx) => (
                      <tr
                        key={aspect}
                        className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}
                      >
                        <td className="px-4 py-3 font-medium text-gray-600 align-top capitalize border-r border-gray-100">
                          {aspect.replace(/_/g, ' ')}
                        </td>
                        {result.papers.map((p) => (
                          <td key={p.paper_id} className="px-4 py-3 text-gray-600 align-top">
                            {cells[p.paper_id] || '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            选择 2-5 篇论文后点击「生成对比」查看结果
          </div>
        )}
      </div>
    </div>
  );
}
