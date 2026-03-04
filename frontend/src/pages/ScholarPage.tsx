import { useEffect, useCallback, useState } from 'react';
import {
  Search,
  Download,
  Loader2,
  BookOpen,
  ChevronDown,
  ChevronUp,
  CheckSquare,
  Square,
  X,
  FileText,
  AlertCircle,
  ExternalLink,
  Sparkles,
  BookmarkPlus,
  BookmarkCheck,
  Trash2,
  FolderPlus,
  FolderOpen,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useScholarStore } from '../stores/useScholarStore';
import { useConfigStore } from '../stores/useConfigStore';
import { useToastStore } from '../stores/useToastStore';
import { PdfViewerModal } from '../components/ui/PdfViewerModal';
import { FolderBrowserModal } from '../components/ui/FolderBrowserModal';
import { getPdfViewUrl } from '../api/scholar';
import type { ScholarSource, ScholarLibraryPaper } from '../api/scholar';

const SOURCE_OPTIONS: { value: ScholarSource; labelKey: string }[] = [
  { value: 'google_scholar', labelKey: 'scholar.sourceGoogleScholar' },
  { value: 'google', labelKey: 'scholar.sourceGoogle' },
  { value: 'semantic_relevance', labelKey: 'scholar.sourceSemanticRelevance' },
  { value: 'semantic_bulk', labelKey: 'scholar.sourceSemanticBulk' },
  { value: 'ncbi', labelKey: 'scholar.sourcePubMed' },
  { value: 'annas_archive', labelKey: 'scholar.sourceAnnasArchive' },
];

export function ScholarPage() {
  const { t } = useTranslation();
  const currentCollection = useConfigStore((s) => s.currentCollection);
  const addToast = useToastStore((s) => s.addToast);

  const {
    query,
    source,
    yearStart,
    yearEnd,
    limit,
    results,
    isSearching,
    searchError,
    selectedIndices,
    downloadTasks,
    scholarHealth,
    scholarHealthError,
    libraries,
    activeLibraryId,
    libraryPapers,
    libraryLoading,
    libraryError,
    setQuery,
    setSource,
    setYearStart,
    setYearEnd,
    setLimit,
    setSearchError,
    smartOptimize,
    setSmartOptimize,
    useSerpapi,
    serpapiRatio,
    setUseSerpapi,
    setSerpapiRatio,
    search,
    downloadOne,
    downloadSelected,
    toggleSelect,
    selectAll,
    clearSelection,
    checkHealth,
    removeTask,
    loadLibraries,
    createLibrary,
    deleteLibrary,
    clearTemporaryLibrary,
    setActiveLibrary,
    loadLibraryPapers,
    addResultsToLibrary,
    removeFromLibrary,
    downloadLibraryBatch,
  } = useScholarStore();

  const [tasksPanelOpen, setTasksPanelOpen] = useState(true);
  const [libraryPanelOpen, setLibraryPanelOpen] = useState(true);
  const [pdfView, setPdfView] = useState<{ paperId: string; title: string } | null>(null);
  const [downloadingIndex, setDownloadingIndex] = useState<number | null>(null);
  const [downloadingBatch, setDownloadingBatch] = useState(false);
  const [showNewLibraryModal, setShowNewLibraryModal] = useState(false);
  const [newLibraryName, setNewLibraryName] = useState('');
  const [newLibraryDesc, setNewLibraryDesc] = useState('');
  const [newLibraryFolderPath, setNewLibraryFolderPath] = useState('');
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);
  const [addingToLibrary, setAddingToLibrary] = useState(false);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    if (scholarHealth?.enabled) loadLibraries();
  }, [scholarHealth?.enabled, loadLibraries]);

  const isResultInLibrary = useCallback(
    (title: string, doi: string | null, libPapers: ScholarLibraryPaper[]) => {
      const t = title.trim().toLowerCase();
      const d = (doi || '').trim();
      return libPapers.some(
        (p) =>
          (d && p.doi && p.doi.toLowerCase() === d.toLowerCase()) ||
          (!d && p.title.trim().toLowerCase() === t)
      );
    },
    []
  );

  const taskIds = Object.keys(downloadTasks);
  const activeTaskCount = taskIds.length;

  const handleSearch = useCallback(() => {
    setSearchError(null);
    search();
  }, [search, setSearchError]);

  const handleDownloadOne = useCallback(
    async (index: number) => {
      setDownloadingIndex(index);
      try {
        const taskId = await downloadOne(index, currentCollection, true);
        if (taskId) {
          addToast(t('scholar.downloadStarted'), 'success');
        } else {
          addToast(t('scholar.downloadFailed'), 'error');
        }
      } catch {
        addToast(t('scholar.downloadFailed'), 'error');
      } finally {
        setDownloadingIndex(null);
      }
    },
    [downloadOne, currentCollection, addToast, t]
  );

  const handleDownloadSelected = useCallback(async () => {
    if (selectedIndices.length === 0) return;
    setDownloadingBatch(true);
    try {
      const taskId = await downloadSelected(currentCollection);
      if (taskId) {
        addToast(t('scholar.batchDownloadStarted', { count: selectedIndices.length }), 'success');
      } else {
        addToast(t('scholar.downloadFailed'), 'error');
      }
    } catch {
      addToast(t('scholar.downloadFailed'), 'error');
    } finally {
      setDownloadingBatch(false);
    }
  }, [selectedIndices.length, downloadSelected, currentCollection, addToast, t]);

  const handleAddSelectedToLibrary = useCallback(async () => {
    if (selectedIndices.length === 0 || activeLibraryId == null) return;
    setAddingToLibrary(true);
    try {
      const res = await addResultsToLibrary(selectedIndices);
      if (res && res.added > 0) {
        addToast(t('scholar.libraryAddSelected') + ` (${res.added})`, 'success');
        clearSelection();
      }
    } finally {
      setAddingToLibrary(false);
    }
  }, [selectedIndices.length, activeLibraryId, addResultsToLibrary, addToast, t, clearSelection]);

  const handleAddOneToLibrary = useCallback(
    async (index: number) => {
      if (activeLibraryId == null) return;
      const res = await addResultsToLibrary([index]);
      if (res && res.added > 0) addToast(t('scholar.libraryAddOne'), 'success');
    },
    [activeLibraryId, addResultsToLibrary, addToast, t]
  );

  const allSelected = results.length > 0 && selectedIndices.length === results.length;

  if (scholarHealth && !scholarHealth.enabled) {
    return (
      <div className="p-6 max-w-2xl mx-auto">
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-6 flex items-start gap-4">
          <AlertCircle className="text-amber-400 shrink-0 mt-0.5" size={24} />
          <div>
            <h3 className="font-semibold text-amber-200">{t('scholar.serviceDisabled')}</h3>
            <p className="text-slate-400 text-sm mt-1">{t('scholar.serviceDisabledHint')}</p>
            {scholarHealthError === '404' && (
              <p className="text-amber-200/90 text-sm mt-2">{t('scholar.service404Hint')}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-shrink-0 p-4 space-y-4">
        {/* Search bar */}
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.queryLabel')}</label>
            <div className="flex rounded-lg border border-slate-600/80 bg-slate-800/60 focus-within:border-sky-500/50">
              <Search className="text-slate-500 shrink-0 self-center ml-3" size={18} />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder={t('scholar.queryPlaceholder')}
                className="flex-1 bg-transparent px-3 py-2.5 text-slate-200 placeholder:text-slate-500 focus:outline-none text-sm"
              />
            </div>
          </div>
          <div className="w-[180px]">
            <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.sourceLabel')}</label>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value as ScholarSource)}
              className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-3 py-2.5 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500/50"
            >
              {SOURCE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {t(opt.labelKey)}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <div className="w-24">
              <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.yearStart')}</label>
              <input
                type="number"
                min={1900}
                max={2100}
                value={yearStart ?? ''}
                onChange={(e) => setYearStart(e.target.value === '' ? null : parseInt(e.target.value, 10) || null)}
                placeholder="—"
                className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-3 py-2.5 text-slate-200 text-sm focus:outline-none"
              />
            </div>
            <div className="w-24">
              <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.yearEnd')}</label>
              <input
                type="number"
                min={1900}
                max={2100}
                value={yearEnd ?? ''}
                onChange={(e) => setYearEnd(e.target.value === '' ? null : parseInt(e.target.value, 10) || null)}
                placeholder="—"
                className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-3 py-2.5 text-slate-200 text-sm focus:outline-none"
              />
            </div>
          </div>
          <div className="w-28">
            <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.limitLabel')}</label>
            <div className="flex rounded-lg border border-slate-600/80 bg-slate-800/60 overflow-hidden">
              <input
                type="number"
                min={1}
                step={10}
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value, 10) || 30)}
                className="w-14 flex-1 min-w-0 bg-transparent px-2 py-2.5 text-slate-200 text-sm focus:outline-none"
              />
              <div className="flex flex-col border-l border-slate-600/80">
                <button
                  type="button"
                  onClick={() => setLimit(limit + 10)}
                  className="flex items-center justify-center p-0.5 text-slate-400 hover:bg-slate-600/60 hover:text-slate-200 transition-colors"
                  title={t('scholar.limitIncrease')}
                  aria-label={t('scholar.limitIncrease')}
                >
                  <ChevronUp size={16} />
                </button>
                <button
                  type="button"
                  onClick={() => setLimit(Math.max(1, limit - 10))}
                  className="flex items-center justify-center p-0.5 text-slate-400 hover:bg-slate-600/60 hover:text-slate-200 transition-colors"
                  title={t('scholar.limitDecrease')}
                  aria-label={t('scholar.limitDecrease')}
                >
                  <ChevronDown size={16} />
                </button>
              </div>
            </div>
          </div>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={smartOptimize}
              onChange={(e) => setSmartOptimize(e.target.checked)}
              className="rounded border-slate-500 bg-slate-800 text-teal-500 focus:ring-teal-500/50"
            />
            <Sparkles size={16} className="text-teal-400 shrink-0" />
            <span className="text-xs text-slate-400 whitespace-nowrap">{t('scholar.smartOptimize')}</span>
          </label>
          {(source === 'google_scholar' || source === 'google') && (
            <>
              <label className="flex items-center gap-2 cursor-pointer select-none" title={t('sidebar.useSerpapiHelp')}>
                <input
                  type="checkbox"
                  checked={useSerpapi}
                  onChange={(e) => setUseSerpapi(e.target.checked)}
                  className="rounded border-slate-500 bg-slate-800 text-teal-500 focus:ring-teal-500/50"
                />
                <span className="text-xs text-slate-400 whitespace-nowrap">{t('sidebar.useSerpapi')}</span>
              </label>
              {useSerpapi && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400 whitespace-nowrap">{t('sidebar.serpapiRatio')}</span>
                  <select
                    value={serpapiRatio}
                    onChange={(e) => setSerpapiRatio(Number(e.target.value))}
                    className="rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:ring-1 focus:ring-sky-500/50"
                  >
                    {[0, 25, 33, 50, 67, 75, 100].map((v) => (
                      <option key={v} value={v}>{v}%</option>
                    ))}
                  </select>
                </div>
              )}
            </>
          )}
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-medium text-sm transition-colors"
          >
            {isSearching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
            {t('common.search')}
          </button>

          <div className="flex items-center gap-2">
            <div className="w-[160px]">
              <label className="block text-xs font-medium text-slate-400 mb-1">{t('scholar.libraryLabel')}</label>
              <select
                value={activeLibraryId ?? ''}
                onChange={(e) => setActiveLibrary(e.target.value === '' ? null : Number(e.target.value))}
                className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-3 py-2.5 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500/50"
              >
                <option value="">{t('scholar.libraryTemporary')}</option>
                {libraries.map((lib) => (
                  <option key={lib.id} value={lib.id}>
                    {lib.name}
                    {lib.is_temporary ? ` (${t('scholar.libraryTemporaryBadge')})` : ''} ({lib.paper_count})
                  </option>
                ))}
              </select>
            </div>
            <div className="pt-6">
              <button
                type="button"
                onClick={() => setShowNewLibraryModal(true)}
                className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-slate-600/80 bg-slate-800/60 text-slate-300 hover:bg-slate-700/60 hover:text-slate-200 text-sm"
              >
                <FolderPlus size={18} />
                {t('scholar.libraryNew')}
              </button>
            </div>
          </div>
        </div>

        {searchError && (
          <div className="flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2 text-red-200 text-sm">
            <AlertCircle size={18} />
            {searchError}
          </div>
        )}
        {libraryError && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-amber-200 text-sm">
            <AlertCircle size={18} />
            {libraryError}
          </div>
        )}
      </div>

      {/* New library modal */}
      {showNewLibraryModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setShowNewLibraryModal(false)}>
          <div
            className="rounded-xl border border-slate-600 bg-slate-800 p-6 w-full max-w-md shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="font-medium text-slate-200 mb-4">{t('scholar.libraryNew')}</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">{t('scholar.libraryLabel')}</label>
                <input
                  type="text"
                  value={newLibraryName}
                  onChange={(e) => setNewLibraryName(e.target.value)}
                  placeholder={t('scholar.libraryNew')}
                  className="w-full rounded-lg border border-slate-600 bg-slate-800/60 px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500/50"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Description</label>
                <input
                  type="text"
                  value={newLibraryDesc}
                  onChange={(e) => setNewLibraryDesc(e.target.value)}
                  placeholder="Optional"
                  className="w-full rounded-lg border border-slate-600 bg-slate-800/60 px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500/50"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">{t('scholar.libraryFolderPath')}</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newLibraryFolderPath}
                    onChange={(e) => setNewLibraryFolderPath(e.target.value)}
                    placeholder={t('scholar.libraryFolderPathPlaceholder')}
                    className="flex-1 rounded-lg border border-slate-600 bg-slate-800/60 px-3 py-2 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500/50"
                  />
                  <button
                    type="button"
                    onClick={() => setShowFolderBrowser(true)}
                    className="flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg border border-sky-500/60 bg-sky-500/20 text-sky-300 hover:bg-sky-500/30 hover:text-sky-200 text-sm"
                    title={t('scholar.libraryFolderPathPickHint')}
                  >
                    <FolderOpen size={16} />
                    {t('scholar.libraryFolderPathPick')}
                  </button>
                </div>
                <p className="mt-1 text-[11px] text-slate-500">{t('scholar.libraryFolderPathPickHint')}</p>
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button
                type="button"
                onClick={() => {
                  setShowNewLibraryModal(false);
                  setNewLibraryFolderPath('');
                }}
                className="px-4 py-2 rounded-lg text-slate-400 hover:bg-slate-700/60 text-sm"
              >
                {t('common.cancel')}
              </button>
              <button
                type="button"
                onClick={async () => {
                  const name = newLibraryName.trim();
                  if (!name) return;
                  const path = newLibraryFolderPath.trim();
                  if (!path) {
                    addToast(t('scholar.libraryFolderPathPlaceholder'), 'error');
                    return;
                  }
                  const lib = await createLibrary(name, newLibraryDesc.trim() || undefined, path, false);
                  if (lib) {
                    setActiveLibrary(lib.id);
                    setShowNewLibraryModal(false);
                    setNewLibraryName('');
                    setNewLibraryDesc('');
                    setNewLibraryFolderPath('');
                    addToast(t('scholar.libraryCreate'), 'success');
                  }
                }}
                className="px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-sm"
              >
                {t('scholar.libraryCreate')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results + Library panel + Tasks layout */}
      <div className="flex-1 flex min-h-0 gap-4 px-4 pb-4">
        {/* Library panel (left) */}
        {activeLibraryId != null && (
          <div className="w-80 flex-shrink-0 flex flex-col rounded-xl border border-slate-700/60 bg-slate-800/30 overflow-hidden">
            <button
              onClick={() => setLibraryPanelOpen((o) => !o)}
              className="flex items-center justify-between px-4 py-2 border-b border-slate-700/60 bg-slate-800/50 text-slate-200 text-sm font-medium"
            >
              <span className="flex items-center gap-2 truncate">
                <BookOpen size={16} />
                {t('scholar.libraryPanelTitle')} ({libraryPapers.length})
              </span>
              {libraryPanelOpen ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
            </button>
            {libraryPanelOpen && (
              <>
                <div className="flex items-center gap-2 p-2 border-b border-slate-700/60">
                  <button
                    type="button"
                    onClick={async () => {
                      const taskId = await downloadLibraryBatch(currentCollection);
                      if (taskId) {
                        addToast(t('scholar.batchDownloadStarted', { count: libraryPapers.length }), 'success');
                      } else {
                        addToast(t('scholar.downloadFailed'), 'error');
                      }
                    }}
                    disabled={libraryPapers.length === 0 || libraryLoading}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60 text-white text-sm font-medium"
                  >
                    <Download size={16} />
                    {t('scholar.libraryDownloadAll')}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const activeLib = libraries.find((l) => l.id === activeLibraryId);
                      const isTemp = activeLib?.is_temporary ?? false;
                      const confirmKey = isTemp ? 'scholar.libraryClearConfirm' : 'scholar.libraryDeleteConfirm';
                      if (window.confirm(t(confirmKey))) {
                        if (isTemp) {
                          clearTemporaryLibrary(activeLibraryId);
                        } else {
                          deleteLibrary(activeLibraryId);
                        }
                      }
                    }}
                    className="p-2 rounded-lg text-slate-400 hover:bg-red-500/20 hover:text-red-400"
                    title={
                      libraries.find((l) => l.id === activeLibraryId)?.is_temporary
                        ? t('scholar.libraryClear')
                        : t('scholar.libraryDelete')
                    }
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 min-h-0">
                  {libraryLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 size={24} className="animate-spin text-sky-400" />
                    </div>
                  ) : libraryPapers.length === 0 ? (
                    <p className="text-slate-500 text-sm py-6 text-center">{t('scholar.libraryEmpty')}</p>
                  ) : (
                    <ul className="space-y-2">
                      {libraryPapers.map((p) => (
                        <li
                          key={p.id}
                          className="flex items-start gap-2 rounded-lg border border-slate-600/60 bg-slate-800/50 p-2 text-sm group"
                        >
                          <div className="flex-1 min-w-0">
                            <p className="text-slate-200 line-clamp-2" title={p.title}>
                              {p.title}
                            </p>
                            <p className="text-xs text-slate-500 mt-0.5">
                              {p.authors?.join(', ')} {p.year != null ? ` · ${p.year}` : ''}
                            </p>
                            {p.doi && (
                              <a
                                href={`https://doi.org/${p.doi}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs text-sky-500/80 hover:text-sky-400 hover:underline truncate block mt-0.5"
                                title={`DOI: ${p.doi}`}
                                onClick={(e) => e.stopPropagation()}
                              >
                                {p.doi}
                              </a>
                            )}
                          </div>
                          <button
                            type="button"
                            onClick={() => removeFromLibrary(activeLibraryId, p.id)}
                            className="p-1 rounded text-slate-400 hover:bg-red-500/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                            title={t('scholar.libraryDelete')}
                          >
                            <Trash2 size={14} />
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        <div className="flex-1 min-w-0 flex flex-col">
          {/* Results table / cards */}
          <div className="flex-1 rounded-xl border border-slate-700/60 bg-slate-800/30 overflow-hidden flex flex-col min-h-0">
            {results.length === 0 && !isSearching && (
              <div className="flex-1 flex flex-col items-center justify-center py-16 px-4 text-center">
                <BookOpen className="text-slate-500 mb-4" size={48} />
                <p className="text-slate-400 font-medium">{t('scholar.emptyState')}</p>
                <p className="text-slate-500 text-sm mt-1">{t('scholar.emptyStateHint')}</p>
              </div>
            )}

            {isSearching && (
              <div className="flex-1 flex items-center justify-center py-16">
                <Loader2 size={32} className="animate-spin text-sky-400" />
              </div>
            )}

            {results.length > 0 && !isSearching && (
              <>
                {/* Table header with select all */}
                <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-slate-700/60 bg-slate-800/50">
                  <button
                    onClick={() => (allSelected ? clearSelection() : selectAll())}
                    className="p-1 rounded text-slate-400 hover:text-sky-400 transition-colors"
                    title={allSelected ? t('scholar.clearSelection') : t('scholar.selectAll')}
                  >
                    {allSelected ? <CheckSquare size={20} /> : <Square size={20} />}
                  </button>
                  <span className="text-xs text-slate-500">
                    {results.length} {t('scholar.resultsCount')}
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto">
                  {/* Desktop table */}
                  <table className="w-full hidden md:table text-left text-sm">
                    <thead className="sticky top-0 bg-slate-800/90 border-b border-slate-700/60 z-10">
                      <tr className="text-slate-400">
                        <th className="w-10 py-2 px-2"></th>
                        <th className="py-2 px-3 font-medium">{t('scholar.colTitle')}</th>
                        <th className="py-2 px-3 font-medium w-32">{t('scholar.colAuthors')}</th>
                        <th className="py-2 px-3 font-medium w-16">{t('scholar.colYear')}</th>
                        <th className="py-2 px-3 font-medium w-24">{t('scholar.colSource')}</th>
                        <th className="py-2 px-3 font-medium w-28">{t('scholar.colActions')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((item, index) => {
                        const m = item.metadata;
                        const selected = selectedIndices.includes(index);
                        const isDownloading = downloadingIndex === index;
                        return (
                          <tr
                            key={`${m.title}-${index}`}
                            className="border-b border-slate-700/40 hover:bg-slate-800/50 transition-colors"
                          >
                            <td className="py-2 px-2">
                              <button
                                onClick={() => toggleSelect(index)}
                                className="p-1 rounded text-slate-400 hover:text-sky-400"
                              >
                                {selected ? <CheckSquare size={18} /> : <Square size={18} />}
                              </button>
                            </td>
                            <td className="py-2 px-3">
                              <span className="text-slate-200 line-clamp-2" title={m.title}>
                                {m.title}
                              </span>
                              {m.doi && (
                                <a
                                  href={`https://doi.org/${m.doi}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-sky-500/80 hover:text-sky-400 hover:underline block mt-0.5 truncate"
                                  title={`DOI: ${m.doi}`}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  {m.doi}
                                </a>
                              )}
                            </td>
                            <td className="py-2 px-3 text-slate-400 text-xs line-clamp-2">{m.authors?.join(', ') || '—'}</td>
                            <td className="py-2 px-3 text-slate-400">{m.year ?? '—'}</td>
                            <td className="py-2 px-3 text-slate-500 text-xs">{m.source}</td>
                            <td className="py-2 px-3">
                              <div className="flex items-center gap-1">
                                {activeLibraryId != null && (
                                  <button
                                    onClick={() => handleAddOneToLibrary(index)}
                                    disabled={isResultInLibrary(m.title, m.doi ?? null, libraryPapers)}
                                    className="p-1.5 rounded-lg text-slate-400 hover:bg-teal-500/20 hover:text-teal-400 disabled:opacity-50 disabled:cursor-default"
                                    title={
                                      isResultInLibrary(m.title, m.doi ?? null, libraryPapers)
                                        ? t('scholar.libraryInLibrary')
                                        : t('scholar.libraryAddOne')
                                    }
                                  >
                                    {isResultInLibrary(m.title, m.doi ?? null, libraryPapers) ? (
                                      <BookmarkCheck size={16} />
                                    ) : (
                                      <BookmarkPlus size={16} />
                                    )}
                                  </button>
                                )}
                                <button
                                  onClick={() => handleDownloadOne(index)}
                                  disabled={isDownloading}
                                  className="p-1.5 rounded-lg text-slate-400 hover:bg-sky-500/20 hover:text-sky-400 disabled:opacity-50"
                                  title={t('scholar.downloadOne')}
                                >
                                  {isDownloading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                                </button>
                                {m.url && (
                                  <a
                                    href={m.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-600 hover:text-slate-200"
                                    title={t('scholar.openLink')}
                                  >
                                    <ExternalLink size={16} />
                                  </a>
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>

                  {/* Mobile cards */}
                  <div className="md:hidden divide-y divide-slate-700/40">
                    {results.map((item, index) => {
                      const m = item.metadata;
                      const selected = selectedIndices.includes(index);
                      const isDownloading = downloadingIndex === index;
                      return (
                        <div key={`${m.title}-${index}`} className="p-4 hover:bg-slate-800/50">
                          <div className="flex gap-2">
                            <button
                              onClick={() => toggleSelect(index)}
                              className="shrink-0 p-1 text-slate-400 hover:text-sky-400"
                            >
                              {selected ? <CheckSquare size={20} /> : <Square size={20} />}
                            </button>
                            <div className="flex-1 min-w-0">
                              <p className="text-slate-200 font-medium line-clamp-2">{m.title}</p>
                              <p className="text-xs text-slate-500 mt-1">{m.authors?.join(', ') || '—'}</p>
                              {m.doi && (
                                <a
                                  href={`https://doi.org/${m.doi}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-sky-500/80 hover:text-sky-400 hover:underline truncate block mt-0.5"
                                  title={`DOI: ${m.doi}`}
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  {m.doi}
                                </a>
                              )}
                              <div className="flex items-center gap-2 mt-2 flex-wrap">
                                {m.year && <span className="text-xs text-slate-400">{m.year}</span>}
                                {m.source && <span className="text-xs text-slate-500">{m.source}</span>}
                                {activeLibraryId != null && (
                                  <button
                                    onClick={() => handleAddOneToLibrary(index)}
                                    disabled={isResultInLibrary(m.title, m.doi ?? null, libraryPapers)}
                                    className="inline-flex items-center gap-1 px-2 py-1 rounded border border-slate-600 text-slate-400 hover:bg-teal-500/20 hover:text-teal-400 disabled:opacity-50 text-xs"
                                  >
                                    {isResultInLibrary(m.title, m.doi ?? null, libraryPapers) ? (
                                      <BookmarkCheck size={12} />
                                    ) : (
                                      <BookmarkPlus size={12} />
                                    )}
                                    {t('scholar.libraryAddOne')}
                                  </button>
                                )}
                                <button
                                  onClick={() => handleDownloadOne(index)}
                                  disabled={isDownloading}
                                  className="inline-flex items-center gap-1 px-2 py-1 rounded bg-sky-600/80 hover:bg-sky-500 text-white text-xs"
                                >
                                  {isDownloading ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
                                  {t('scholar.downloadOne')}
                                </button>
                                {m.url && (
                                  <a
                                    href={m.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-sky-400 text-xs"
                                  >
                                    <ExternalLink size={12} /> {t('scholar.openLink')}
                                  </a>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Batch action bar */}
          {selectedIndices.length > 0 && (
            <div className="flex-shrink-0 flex items-center justify-between gap-4 mt-3 px-4 py-3 rounded-xl border border-sky-500/30 bg-sky-900/20">
              <span className="text-sky-200 text-sm font-medium">
                {t('scholar.selectedCount', { count: selectedIndices.length })}
              </span>
              <div className="flex items-center gap-2">
                {activeLibraryId != null && (
                  <button
                    onClick={handleAddSelectedToLibrary}
                    disabled={addingToLibrary}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg border border-teal-500/50 bg-teal-900/20 hover:bg-teal-800/30 text-teal-200 font-medium text-sm disabled:opacity-60"
                  >
                    {addingToLibrary ? <Loader2 size={18} className="animate-spin" /> : <BookmarkPlus size={18} />}
                    {t('scholar.libraryAddSelected')}
                  </button>
                )}
                <button
                  onClick={clearSelection}
                  className="px-3 py-1.5 rounded-lg text-slate-400 hover:bg-slate-700/60 text-sm"
                >
                  {t('scholar.clearSelection')}
                </button>
                <button
                  onClick={handleDownloadSelected}
                  disabled={downloadingBatch}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60 text-white font-medium text-sm"
                >
                  {downloadingBatch ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
                  {t('scholar.downloadSelected', { count: selectedIndices.length })}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Download progress panel */}
        {activeTaskCount > 0 && (
          <div className="w-80 flex-shrink-0 flex flex-col rounded-xl border border-slate-700/60 bg-slate-800/30 overflow-hidden">
            <button
              onClick={() => setTasksPanelOpen((o) => !o)}
              className="flex items-center justify-between px-4 py-2 border-b border-slate-700/60 bg-slate-800/50 text-slate-200 text-sm font-medium"
            >
              <span className="flex items-center gap-2">
                <FileText size={16} />
                {t('scholar.downloadTasks')} ({activeTaskCount})
              </span>
              {tasksPanelOpen ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
            </button>
            {tasksPanelOpen && (
              <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {taskIds.map((taskId) => {
                  const task = downloadTasks[taskId];
                  if (!task) return null;
                  const payload = task.payload as { total?: number; completed?: number; failed?: number };
                  const total = payload?.total ?? 1;
                  const completed = payload?.completed ?? 0;
                  const isDone = task.status === 'completed' || task.status === 'error';
                  const paperId = (task.payload as { paper_id?: string })?.paper_id;
                  return (
                    <div
                      key={taskId}
                      className="rounded-lg border border-slate-600/60 bg-slate-800/50 p-3 text-sm"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <p className="text-slate-400 text-xs font-mono truncate" title={taskId}>
                            {taskId}
                          </p>
                          <p className="text-slate-200 mt-0.5">
                            {t(`scholar.taskStatus.${task.status}`)}
                            {total > 1 && ` · ${completed}/${total}`}
                          </p>
                          {task.error_message && (
                            <p className="text-red-400 text-xs mt-1">{task.error_message}</p>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          {isDone && paperId && (
                            <button
                              onClick={() => setPdfView({ paperId, title: '' })}
                              className="p-1.5 rounded text-sky-400 hover:bg-sky-500/20"
                              title={t('scholar.viewPdf')}
                            >
                              <FileText size={16} />
                            </button>
                          )}
                          <button
                            onClick={() => removeTask(taskId)}
                            className="p-1.5 rounded text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                            title={t('common.close')}
                          >
                            <X size={16} />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {pdfView && (
        <PdfViewerModal
          open={!!pdfView}
          onClose={() => setPdfView(null)}
          pdfUrl={getPdfViewUrl(pdfView.paperId)}
          title={pdfView.title}
        />
      )}

      <FolderBrowserModal
        open={showFolderBrowser}
        onClose={() => setShowFolderBrowser(false)}
        onSelect={(path) => setNewLibraryFolderPath(path)}
        initialPath={newLibraryFolderPath || null}
      />
    </div>
  );
}
