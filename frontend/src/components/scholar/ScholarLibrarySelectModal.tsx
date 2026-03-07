import { useEffect, useMemo, useState } from 'react';
import { Clock3, Library, Loader2, Search } from 'lucide-react';
import { Modal } from '../ui/Modal';
import type { ScholarLibrary } from '../../api/scholar';

interface ScholarLibrarySelectModalProps {
  open: boolean;
  libraries: ScholarLibrary[];
  defaultLibraryId: number | null;
  importCount?: number;
  loading?: boolean;
  timeoutSeconds?: number;
  onClose: () => void;
  onConfirm: (libraryId: number) => Promise<void> | void;
}

const SCHOLAR_LIBRARY_SEARCH_HISTORY_KEY = 'scholar_library_search_history_v1';
const SCHOLAR_LIBRARY_SEARCH_HISTORY_MAX = 10;

function loadLibrarySearchHistory(): string[] {
  try {
    const raw = localStorage.getItem(SCHOLAR_LIBRARY_SEARCH_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => String(item ?? '').trim())
      .filter(Boolean)
      .slice(0, SCHOLAR_LIBRARY_SEARCH_HISTORY_MAX);
  } catch {
    return [];
  }
}

function saveLibrarySearchHistory(history: string[]) {
  try {
    localStorage.setItem(
      SCHOLAR_LIBRARY_SEARCH_HISTORY_KEY,
      JSON.stringify(history.slice(0, SCHOLAR_LIBRARY_SEARCH_HISTORY_MAX)),
    );
  } catch {
    // ignore quota errors
  }
}

export function ScholarLibrarySelectModal({
  open,
  libraries,
  defaultLibraryId,
  importCount = 0,
  loading = false,
  timeoutSeconds = 10,
  onClose,
  onConfirm,
}: ScholarLibrarySelectModalProps) {
  const [searchText, setSearchText] = useState('');
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const selectableLibraries = useMemo(
    () => libraries.filter((lib) => typeof lib.id === 'number'),
    [libraries],
  );
  const visibleLibraries = useMemo(() => {
    const q = searchText.trim().toLowerCase();
    if (!q) return selectableLibraries;
    return selectableLibraries.filter((lib) => lib.name.toLowerCase().includes(q));
  }, [searchText, selectableLibraries]);
  const resolvedDefaultId = useMemo(() => {
    if (defaultLibraryId != null && visibleLibraries.some((lib) => lib.id === defaultLibraryId)) {
      return defaultLibraryId;
    }
    return visibleLibraries[0]?.id ?? null;
  }, [defaultLibraryId, visibleLibraries]);

  const [selectedLibraryId, setSelectedLibraryId] = useState<number | null>(resolvedDefaultId);
  const [remainingSeconds, setRemainingSeconds] = useState(timeoutSeconds);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    setSearchHistory(loadLibrarySearchHistory());
    setSearchText('');
    setSelectedLibraryId(resolvedDefaultId);
    setRemainingSeconds(timeoutSeconds);
    setIsSubmitting(false);
  }, [open, resolvedDefaultId, timeoutSeconds]);

  useEffect(() => {
    if (!open) return;
    if (visibleLibraries.length === 0) {
      setSelectedLibraryId(null);
      return;
    }
    setSelectedLibraryId((prev) => {
      if (prev != null && visibleLibraries.some((lib) => lib.id === prev)) return prev;
      return visibleLibraries[0].id;
    });
  }, [open, visibleLibraries]);

  const persistSearchTerm = (value: string) => {
    const term = value.trim();
    if (!term) return;
    setSearchHistory((prev) => {
      const next = [term, ...prev.filter((item) => item.toLowerCase() !== term.toLowerCase())].slice(
        0,
        SCHOLAR_LIBRARY_SEARCH_HISTORY_MAX,
      );
      saveLibrarySearchHistory(next);
      return next;
    });
  };

  useEffect(() => {
    if (!open || loading || isSubmitting || selectedLibraryId == null) return;
    if (remainingSeconds <= 0) {
      setIsSubmitting(true);
      persistSearchTerm(searchText);
      Promise.resolve(onConfirm(selectedLibraryId))
        .finally(() => {
          setIsSubmitting(false);
        });
      return;
    }
    const timer = window.setTimeout(() => setRemainingSeconds((s) => s - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [open, loading, isSubmitting, remainingSeconds, selectedLibraryId, onConfirm]);

  const handleConfirm = async () => {
    if (selectedLibraryId == null) return;
    setIsSubmitting(true);
    try {
      persistSearchTerm(searchText);
      await onConfirm(selectedLibraryId);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={isSubmitting ? () => undefined : onClose}
      title="导入到文献库"
      icon={<Library size={18} />}
      maxWidth="max-w-lg"
      variant="dark"
    >
      <div className="space-y-4">
        <p className="text-sm text-slate-300">
          本次将批量导入
          <span className="mx-1 font-semibold text-sky-300">{importCount}</span>
          条文献（按 DOI 去重）。
        </p>
        <p className="text-sm text-slate-300">
          选择要导入的文献库。若未操作，将在
          <span className="mx-1 font-semibold text-amber-300">{remainingSeconds}s</span>
          后自动导入默认库。
        </p>

        <div className="rounded-xl border border-slate-700 bg-slate-900/50 px-3 py-2">
          <label className="text-xs text-slate-400 mb-1 block">搜索文献库（支持最近搜索提示）</label>
          <div className="flex items-center rounded-lg border border-slate-600 bg-slate-800 px-2.5">
            <Search size={14} className="text-slate-500 shrink-0" />
            <input
              type="text"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              list="scholar-library-search-history"
              placeholder="输入库名关键词..."
              className="w-full bg-transparent px-2 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none"
            />
            <datalist id="scholar-library-search-history">
              {searchHistory.map((item) => (
                <option key={item} value={item} />
              ))}
            </datalist>
          </div>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-950/40 max-h-64 overflow-auto">
          {loading ? (
            <div className="px-4 py-8 text-center text-slate-400 text-sm">
              <Loader2 size={18} className="animate-spin mx-auto mb-2" />
              正在加载文献库...
            </div>
          ) : selectableLibraries.length === 0 ? (
            <div className="px-4 py-8 text-center text-slate-400 text-sm">
              当前没有可用文献库，请先在文献检索页面创建一个库。
            </div>
          ) : visibleLibraries.length === 0 ? (
            <div className="px-4 py-8 text-center text-slate-400 text-sm">
              没有匹配的文献库，请尝试其他关键词。
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {visibleLibraries.map((lib) => (
                <label key={lib.id} className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-slate-800/50">
                  <input
                    type="radio"
                    name="target-library"
                    checked={selectedLibraryId === lib.id}
                    onChange={() => setSelectedLibraryId(lib.id)}
                    className="accent-sky-500"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-slate-100 truncate">{lib.name}</div>
                    <div className="text-xs text-slate-400">
                      {lib.paper_count} 篇文献
                      {lib.is_temporary ? ' · 临时库' : ''}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between">
          <div className="text-xs text-slate-400 flex items-center gap-1.5">
            <Clock3 size={12} />
            默认库：{resolvedDefaultId != null ? selectableLibraries.find((l) => l.id === resolvedDefaultId)?.name || '-' : '-'}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-3 py-1.5 rounded-md border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              disabled={isSubmitting || loading || selectedLibraryId == null}
              className="px-3 py-1.5 rounded-md bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 inline-flex items-center gap-1.5"
            >
              {isSubmitting ? <Loader2 size={14} className="animate-spin" /> : null}
              导入
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
