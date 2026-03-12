import { useEffect, useCallback, useMemo, useRef, useState } from 'react';
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
  RefreshCw,
  RefreshCcw,
  FileSearch,
  Upload,
  Settings2,
  PanelLeftOpen,
  PanelLeftClose,
  Layers,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useScholarStore } from '../stores/useScholarStore';
import { useConfigStore } from '../stores/useConfigStore';
import { useToastStore } from '../stores/useToastStore';
import { useAuthStore } from '../stores/useAuthStore';
import { login as apiLogin } from '../api/auth';
import { Modal } from '../components/ui/Modal';
import { PdfViewerModal } from '../components/ui/PdfViewerModal';
import { ScholarAdvancedSettingsModal } from '../components/scholar/ScholarAdvancedSettingsModal';
import { ScholarLibraryRecommendModal } from '../components/scholar/ScholarLibraryRecommendModal';
import {
  getPdfViewUrl,
  getLibraryPdfViewUrl,
  fetchPdfAsBlobUrl,
  uploadLibraryPaperPdf,
  deleteLibraryPaperPdf,
  importLibraryPdfs,
  getHeadedBrowserWindowState,
  showHeadedBrowserWindow,
  parkHeadedBrowserWindow,
} from '../api/scholar';
import type { ScholarSearchResult } from '../api/scholar';
import type {
  ScholarSource,
  ScholarLibraryPaper,
  HeadedBrowserWindowState,
  LibraryImportPdfSummary,
} from '../api/scholar';

const SOURCE_OPTIONS: { value: ScholarSource; labelKey: string }[] = [
  { value: 'google_scholar', labelKey: 'scholar.sourceGoogleScholar' },
  { value: 'google', labelKey: 'scholar.sourceGoogle' },
  { value: 'semantic_relevance', labelKey: 'scholar.sourceSemanticRelevance' },
  { value: 'semantic_bulk', labelKey: 'scholar.sourceSemanticBulk' },
  { value: 'ncbi', labelKey: 'scholar.sourcePubMed' },
  { value: 'annas_archive', labelKey: 'scholar.sourceAnnasArchive' },
];

/** Backend may store different source strings: SerpAPI uses serpapi_scholar/serpapi_google, Playwright uses scholar/google, normalize uses google_scholar/google. */
const SOURCE_FILTER_EQUIVALENTS: Record<string, string[]> = {
  google_scholar: ['google_scholar', 'serpapi_scholar', 'scholar'],
  google: ['google', 'serpapi_google'],
  semantic_relevance: ['semantic', 'semantic_relevance', 'semantic_snippet'],
  semantic_bulk: ['semantic_bulk'],
  ncbi: ['ncbi'],
  annas_archive: ['annas_archive'],
};

function sourceMatchesFilter(paperSource: string, filterValue: string): boolean {
  const equivalents = SOURCE_FILTER_EQUIVALENTS[filterValue];
  if (equivalents) return equivalents.includes(paperSource);
  return paperSource === filterValue;
}

const LIBRARY_PANEL_WIDTH_KEY = 'scholar_library_panel_width_v1';
const LIBRARY_PANEL_MIN_WIDTH = 260;
const LIBRARY_PANEL_MAX_WIDTH = 640;
const RESULTS_PANEL_MIN_WIDTH = 420;
const SCHOLAR_QUERY_HISTORY_KEY = 'scholar_query_history_v1';
const SCHOLAR_QUERY_HISTORY_MAX = 12;

function loadScholarQueryHistory(): string[] {
  try {
    const raw = localStorage.getItem(SCHOLAR_QUERY_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => String(item ?? '').trim())
      .filter(Boolean)
      .slice(0, SCHOLAR_QUERY_HISTORY_MAX);
  } catch {
    return [];
  }
}

function saveScholarQueryHistory(history: string[]) {
  try {
    localStorage.setItem(
      SCHOLAR_QUERY_HISTORY_KEY,
      JSON.stringify(history.slice(0, SCHOLAR_QUERY_HISTORY_MAX)),
    );
  } catch {
    // ignore quota errors
  }
}

export function ScholarPage() {
  const { t } = useTranslation();
  const currentCollection = useConfigStore((s) => s.currentCollection);
  const collectionInfos = useConfigStore((s) => s.collectionInfos);
  const scholarDownloaderDefaults = useConfigStore((s) => s.scholarDownloaderDefaults);
  const addToast = useToastStore((s) => s.addToast);
  const user = useAuthStore((s) => s.user);

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
    batchAllSources,
    setBatchAllSources,
    batchSourceCounts,
    useSerpapi,
    serpapiRatio,
    setUseSerpapi,
    setSerpapiRatio,
    search,
    downloadOne,
    downloadSelected,
    toggleSelect,
    setSelectedIndices,
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
    extractDoiAndDedup,
    pdfRenameDedup,
    refreshMetadataFromCrossref,
    downloadLibraryPaperAndOpen,
    openPdfAfterDownload,
    clearOpenPdfAfterDownload,
    pendingDownloadAndOpen,
    downloadAndOpenFailures,
    clearDownloadAndOpenFailure,
    applyScholarDownloaderDefaults,
    hydrateRunningDownloadTasks,
  } = useScholarStore();

  const [tasksPanelOpen, setTasksPanelOpen] = useState(true);
  const [libraryPanelOpen, setLibraryPanelOpen] = useState(true);
  const [libraryPanelWidth, setLibraryPanelWidth] = useState(320);
  const [isResizingLibraryPanel, setIsResizingLibraryPanel] = useState(false);
  const [pdfView, setPdfView] = useState<{ paperId: string; title: string; pdfUrl?: string } | null>(null);
  const [downloadingIndex, setDownloadingIndex] = useState<number | null>(null);
  const [downloadingBatch, setDownloadingBatch] = useState(false);
  const [extractDoiDedupLoading, setExtractDoiDedupLoading] = useState(false);
  const [refreshMetadataLoading, setRefreshMetadataLoading] = useState(false);
  const [pdfRenameDedupLoading, setPdfRenameDedupLoading] = useState(false);
  const [showNewLibraryModal, setShowNewLibraryModal] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [showRecommendModal, setShowRecommendModal] = useState(false);
  const [deleteLibraryModal, setDeleteLibraryModal] = useState<{ libId: number; name: string } | null>(null);
  const [deleteLibraryPassword, setDeleteLibraryPassword] = useState('');
  const [deleteLibraryVerifying, setDeleteLibraryVerifying] = useState(false);
  const [headedBrowserWindow, setHeadedBrowserWindow] = useState<HeadedBrowserWindowState | null>(null);
  const [headedBrowserWindowBusy, setHeadedBrowserWindowBusy] = useState(false);
  const [newLibraryName, setNewLibraryName] = useState('');
  const [newLibraryDesc, setNewLibraryDesc] = useState('');
  const [importingLibraryPdfs, setImportingLibraryPdfs] = useState(false);
  const [libraryImportSummary, setLibraryImportSummary] = useState<LibraryImportPdfSummary | null>(null);
  const [addingToLibrary, setAddingToLibrary] = useState(false);
  const [uploadingPapers, setUploadingPapers] = useState<Record<number, boolean>>({});
  const [, setUploadFailures] = useState<Record<number, string>>({});
  const [deletingPdfId, setDeletingPdfId] = useState<number | null>(null);
  const [redownloadingId, setRedownloadingId] = useState<number | null>(null);
  const resultsLayoutRef = useRef<HTMLDivElement | null>(null);
  const importLibraryInputRef = useRef<HTMLInputElement | null>(null);

  // Filter/sort state (shared for search results and library list)
  const [sortField, setSortField] = useState<'score' | 'year' | 'impact_factor'>('score');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [filterJournal, setFilterJournal] = useState('');
  const [filterYearMin, setFilterYearMin] = useState<number | null>(null);
  const [filterYearMax, setFilterYearMax] = useState<number | null>(null);
  const [filterStatus, setFilterStatus] = useState<'all' | 'downloaded' | 'not_downloaded' | 'not_in_collection'>('all');
  const [filterIfMin, setFilterIfMin] = useState<number | null>(null);
  const [filterSource, setFilterSource] = useState<ScholarSource | 'all'>('all');
  const [filterDownloadableOnly, setFilterDownloadableOnly] = useState(false);
  const [filterBarCollapsed, setFilterBarCollapsed] = useState(true);
  const [catalogFilterBarCollapsed, setCatalogFilterBarCollapsed] = useState(true);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const currentCollectionInfo = useMemo(
    () => collectionInfos.find((item) => item.name === currentCollection),
    [collectionInfos, currentCollection],
  );

  useEffect(() => {
    setQueryHistory(loadScholarQueryHistory());
  }, []);

  // Recover running batch-download tasks from backend after reload
  useEffect(() => {
    hydrateRunningDownloadTasks();
  }, [hydrateRunningDownloadTasks]);

  const FILTER_STORAGE_KEY = 'scholar_filter_prefs';
  useEffect(() => {
    try {
      const raw = localStorage.getItem(FILTER_STORAGE_KEY);
      if (!raw) return;
      const p = JSON.parse(raw) as Record<string, unknown>;
      if (p.filterJournal != null) setFilterJournal(String(p.filterJournal));
      if (typeof p.filterYearMin === 'number') setFilterYearMin(p.filterYearMin);
      if (typeof p.filterYearMax === 'number') setFilterYearMax(p.filterYearMax);
      if (p.filterStatus === 'downloaded' || p.filterStatus === 'not_downloaded' || p.filterStatus === 'not_in_collection') setFilterStatus(p.filterStatus);
      if (typeof p.filterIfMin === 'number') setFilterIfMin(p.filterIfMin);
      if (p.filterSource && p.filterSource !== 'all') setFilterSource(p.filterSource as ScholarSource);
      if (p.filterDownloadableOnly === true) setFilterDownloadableOnly(true);
      if (p.sortField === 'year' || p.sortField === 'impact_factor') setSortField(p.sortField);
      if (p.sortDir === 'asc') setSortDir('asc');
    } catch {
      // ignore invalid stored prefs
    }
  }, []);
  useEffect(() => {
    const prefs = {
      filterJournal,
      filterYearMin,
      filterYearMax,
      filterStatus,
      filterIfMin,
      filterSource,
      filterDownloadableOnly,
      sortField,
      sortDir,
    };
    try {
      localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(prefs));
    } catch {
      // ignore quota etc.
    }
  }, [
    filterJournal,
    filterYearMin,
    filterYearMax,
    filterStatus,
    filterIfMin,
    filterSource,
    filterDownloadableOnly,
    sortField,
    sortDir,
  ]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(LIBRARY_PANEL_WIDTH_KEY);
      if (!raw) return;
      const parsed = Number(raw);
      if (Number.isFinite(parsed)) {
        setLibraryPanelWidth(Math.max(LIBRARY_PANEL_MIN_WIDTH, Math.min(LIBRARY_PANEL_MAX_WIDTH, parsed)));
      }
    } catch {
      // ignore invalid stored width
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(LIBRARY_PANEL_WIDTH_KEY, String(libraryPanelWidth));
    } catch {
      // ignore quota etc.
    }
  }, [libraryPanelWidth]);

  useEffect(() => {
    if (!isResizingLibraryPanel) return undefined;

    const updateWidth = (clientX: number) => {
      const container = resultsLayoutRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const maxWidth = Math.min(LIBRARY_PANEL_MAX_WIDTH, rect.width - RESULTS_PANEL_MIN_WIDTH);
      const nextWidth = Math.min(Math.max(clientX - rect.left, LIBRARY_PANEL_MIN_WIDTH), maxWidth);
      setLibraryPanelWidth(nextWidth);
    };

    const handlePointerMove = (event: PointerEvent) => {
      updateWidth(event.clientX);
    };

    const stopResize = () => {
      setIsResizingLibraryPanel(false);
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', stopResize);
    window.addEventListener('pointercancel', stopResize);

    return () => {
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', stopResize);
      window.removeEventListener('pointercancel', stopResize);
    };
  }, [isResizingLibraryPanel]);

  const filteredResults = useMemo((): { result: ScholarSearchResult; originalIndex: number }[] => {
    let list = results.map((result, originalIndex) => ({ result, originalIndex }));
    const meta = (r: ScholarSearchResult) => r.metadata;
    if (filterJournal.trim()) {
      const q = filterJournal.trim().toLowerCase();
      list = list.filter(({ result }) => {
        const v = (meta(result).venue ?? meta(result).normalized_journal_name ?? '').toLowerCase();
        return v.includes(q);
      });
    }
    if (filterYearMin != null) {
      list = list.filter(({ result }) => (meta(result).year ?? 0) >= filterYearMin);
    }
    if (filterYearMax != null) {
      list = list.filter(({ result }) => (meta(result).year ?? 9999) <= filterYearMax);
    }
    if (filterSource !== 'all') {
      list = list.filter(({ result }) => sourceMatchesFilter(meta(result).source || '', filterSource));
    }
    if (filterIfMin != null) {
      list = list.filter(({ result }) => (meta(result).impact_factor ?? 0) >= filterIfMin);
    }
    if (filterDownloadableOnly) {
      list = list.filter(({ result }) => !!(meta(result).pdf_url ?? meta(result).downloadable));
    }
    const mult = sortDir === 'asc' ? 1 : -1;
    list = [...list].sort((a, b) => {
      const ma = meta(a.result);
      const mb = meta(b.result);
      if (sortField === 'year') {
        const ya = ma.year ?? 0;
        const yb = mb.year ?? 0;
        return mult * (ya - yb);
      }
      if (sortField === 'impact_factor') {
        const ia = ma.impact_factor ?? 0;
        const ib = mb.impact_factor ?? 0;
        return mult * (ia - ib);
      }
      return mult * (a.result.score - b.result.score);
    });
    return list;
  }, [
    results,
    filterJournal,
    filterYearMin,
    filterYearMax,
    filterSource,
    filterIfMin,
    filterDownloadableOnly,
    sortField,
    sortDir,
  ]);

  const filteredLibraryPapers = useMemo((): ScholarLibraryPaper[] => {
    let list = [...libraryPapers];
    if (filterJournal.trim()) {
      const q = filterJournal.trim().toLowerCase();
      list = list.filter(
        (p) =>
          (p.venue ?? p.normalized_journal_name ?? '').toLowerCase().includes(q)
      );
    }
    if (filterYearMin != null) {
      list = list.filter((p) => (p.year ?? 0) >= filterYearMin);
    }
    if (filterYearMax != null) {
      list = list.filter((p) => (p.year ?? 9999) <= filterYearMax);
    }
    if (filterSource !== 'all') {
      list = list.filter((p) => sourceMatchesFilter(p.source || '', filterSource));
    }
    if (filterStatus === 'downloaded') {
      list = list.filter((p) => p.is_downloaded ?? !!p.downloaded_at);
    } else if (filterStatus === 'not_downloaded') {
      list = list.filter((p) => !(p.is_downloaded ?? p.downloaded_at));
    } else if (filterStatus === 'not_in_collection') {
      list = list.filter((p) => !p.in_collection);
    }
    if (filterIfMin != null) {
      list = list.filter((p) => (p.impact_factor ?? 0) >= filterIfMin);
    }
    const mult = sortDir === 'asc' ? 1 : -1;
    list.sort((a, b) => {
      if (sortField === 'year') {
        return mult * ((a.year ?? 0) - (b.year ?? 0));
      }
      if (sortField === 'impact_factor') {
        return mult * ((a.impact_factor ?? 0) - (b.impact_factor ?? 0));
      }
      return mult * (a.score - b.score);
    });
    return list;
  }, [
    libraryPapers,
    filterJournal,
    filterYearMin,
    filterYearMax,
    filterSource,
    filterStatus,
    filterIfMin,
    sortField,
    sortDir,
  ]);

  const activeFilterCount = useMemo(() => {
    let n = 0;
    if (filterJournal.trim()) n++;
    if (filterYearMin != null) n++;
    if (filterYearMax != null) n++;
    if (filterSource !== 'all') n++;
    if (filterIfMin != null) n++;
    if (filterDownloadableOnly) n++;
    if (filterStatus !== 'all') n++;
    return n;
  }, [
    filterJournal,
    filterYearMin,
    filterYearMax,
    filterSource,
    filterIfMin,
    filterDownloadableOnly,
    filterStatus,
  ]);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const loadHeadedBrowserWindowState = useCallback(async () => {
    try {
      const state = await getHeadedBrowserWindowState();
      setHeadedBrowserWindow(state);
    } catch {
      setHeadedBrowserWindow(null);
    }
  }, []);

  useEffect(() => {
    if (scholarHealth?.enabled) loadLibraries();
  }, [scholarHealth?.enabled, loadLibraries]);

  useEffect(() => {
    const boundLibId = currentCollectionInfo?.associated_library_id;
    if (boundLibId == null) return;
    if (!libraries.some((l) => l.id === boundLibId)) return;
    if (activeLibraryId === boundLibId) return;
    setActiveLibrary(boundLibId);
  }, [activeLibraryId, currentCollectionInfo?.associated_library_id, libraries, setActiveLibrary]);

  useEffect(() => {
    if (!scholarHealth?.enabled) return;
    loadHeadedBrowserWindowState();
  }, [scholarHealth?.enabled, loadHeadedBrowserWindowState]);

  useEffect(() => {
    applyScholarDownloaderDefaults(scholarDownloaderDefaults);
  }, [applyScholarDownloaderDefaults, scholarDownloaderDefaults]);

  useEffect(() => {
    if (!openPdfAfterDownload) return;
    const url =
      openPdfAfterDownload.libId != null
        ? getLibraryPdfViewUrl(openPdfAfterDownload.libId, openPdfAfterDownload.paperId)
        : getPdfViewUrl(openPdfAfterDownload.paperId);
    const isLibraryPdf = url.includes('/scholar/libraries/');
    if (isLibraryPdf) {
      fetchPdfAsBlobUrl(url)
        .then((blobUrl) => {
          setPdfView({
            paperId: openPdfAfterDownload.paperId,
            title: openPdfAfterDownload.title,
            pdfUrl: blobUrl,
          });
        })
        .catch(() => {})
        .finally(() => clearOpenPdfAfterDownload());
    } else {
      setPdfView({
        paperId: openPdfAfterDownload.paperId,
        title: openPdfAfterDownload.title,
        pdfUrl: url,
      });
      clearOpenPdfAfterDownload();
    }
  }, [openPdfAfterDownload, clearOpenPdfAfterDownload]);

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
    const q = query.trim();
    if (q) {
      setQueryHistory((prev) => {
        const next = [q, ...prev.filter((item) => item.toLowerCase() !== q.toLowerCase())].slice(
          0,
          SCHOLAR_QUERY_HISTORY_MAX,
        );
        saveScholarQueryHistory(next);
        return next;
      });
    }
    setSearchError(null);
    search();
  }, [query, search, setSearchError]);

  const handleUploadPdf = useCallback(
    async (p: ScholarLibraryPaper, file?: File) => {
      if (!file || activeLibraryId == null || activeLibraryId < 0) return;
      setUploadingPapers((prev) => ({ ...prev, [p.id]: true }));
      setUploadFailures((prev) => {
        const next = { ...prev };
        delete next[p.id];
        return next;
      });
      try {
        await uploadLibraryPaperPdf(activeLibraryId, p.id, file);
        addToast(t('scholar.uploadPdfSuccess'), 'success');
        await loadLibraryPapers(activeLibraryId);
      } catch {
        setUploadFailures((prev) => ({ ...prev, [p.id]: 'error' }));
        addToast(t('scholar.uploadPdfError'), 'error');
      } finally {
        setUploadingPapers((prev) => ({ ...prev, [p.id]: false }));
      }
    },
    [activeLibraryId, loadLibraryPapers, addToast, t]
  );

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

  const handleShowHeadedBrowser = useCallback(async () => {
    setHeadedBrowserWindowBusy(true);
    try {
      const state = await showHeadedBrowserWindow();
      setHeadedBrowserWindow(state);
      addToast(t('scholar.headedBrowserSummoned'), 'success');
    } catch {
      addToast(t('scholar.headedBrowserControlError'), 'error');
    } finally {
      setHeadedBrowserWindowBusy(false);
    }
  }, [addToast, t]);

  const handleParkHeadedBrowser = useCallback(async () => {
    setHeadedBrowserWindowBusy(true);
    try {
      const state = await parkHeadedBrowserWindow();
      setHeadedBrowserWindow(state);
      addToast(t('scholar.headedBrowserParked'), 'success');
    } catch {
      addToast(t('scholar.headedBrowserControlError'), 'error');
    } finally {
      setHeadedBrowserWindowBusy(false);
    }
  }, [addToast, t]);

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

  const handlePickLibraryPdfFiles = useCallback(() => {
    if (activeLibraryId == null || activeLibraryId < 0) {
      addToast(t('scholar.libraryImportRequirePermanent'), 'info');
      return;
    }
    importLibraryInputRef.current?.click();
  }, [activeLibraryId, addToast, t]);

  const handleLibraryPdfImportChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const picked = Array.from(e.target.files ?? []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
      e.target.value = '';
      if (picked.length === 0) {
        addToast(t('scholar.libraryImportNoPdfFound'), 'info');
        return;
      }
      if (activeLibraryId == null || activeLibraryId < 0) {
        addToast(t('scholar.libraryImportRequirePermanent'), 'info');
        return;
      }
      if (importingLibraryPdfs) return;
      setImportingLibraryPdfs(true);
      try {
        const summary = await importLibraryPdfs(activeLibraryId, picked);
        setLibraryImportSummary(summary);
        await loadLibraryPapers(activeLibraryId);
        await loadLibraries();
        addToast(
          t('scholar.libraryImportSuccessSummary', {
            imported: summary.imported,
            linked: summary.linked_existing,
            skipped: summary.skipped_duplicates,
          }),
          'success',
        );
        if (summary.errors.length > 0) {
          addToast(t('scholar.libraryImportPartialErrors', { count: summary.errors.length }), 'error');
        }
      } catch (err) {
        addToast((err as Error)?.message || t('scholar.libraryImportFailed'), 'error');
      } finally {
        setImportingLibraryPdfs(false);
      }
    },
    [activeLibraryId, importingLibraryPdfs, addToast, t, loadLibraryPapers, loadLibraries],
  );

  const allSelected =
    filteredResults.length > 0 &&
    filteredResults.every(({ originalIndex }) => selectedIndices.includes(originalIndex));

  const handleConfirmDeleteLibraryWithPassword = async () => {
    if (!deleteLibraryModal || !deleteLibraryPassword.trim() || !user) return;
    setDeleteLibraryVerifying(true);
    try {
      await apiLogin({ user_id: user.user_id, password: deleteLibraryPassword });
      const { libId, name } = deleteLibraryModal;
      await deleteLibrary(libId);
      addToast(`已删除文献库「${name}」`, 'success');
      setDeleteLibraryModal(null);
      setDeleteLibraryPassword('');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`删除失败: ${msg}`, 'error');
    } finally {
      setDeleteLibraryVerifying(false);
    }
  };

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
      <input
        ref={importLibraryInputRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={handleLibraryPdfImportChange}
      />
      <div className="flex-shrink-0 p-3 space-y-2">
        {/* Search bar area with dynamic font sizing */}
        <div className="rounded-xl border border-slate-700/60 bg-slate-800/30 p-3 space-y-3 text-[10px] sm:text-xs 2xl:text-sm">
          {/* Row 1: Search Inputs */}
          <div className="flex flex-wrap items-end gap-2 sm:gap-3">
            <div className="flex-[3] min-w-[200px]">
              <label className="block font-medium text-slate-400 mb-1">{t('scholar.queryLabel')}</label>
              <div className="flex rounded-lg border border-slate-600/80 bg-slate-800/60 focus-within:border-sky-500/50">
                <Search className="text-slate-500 shrink-0 self-center ml-2 sm:ml-3" size={14} />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  list="scholar-query-history"
                  placeholder={t('scholar.queryPlaceholder')}
                  className="flex-1 bg-transparent px-2 sm:px-3 py-1.5 sm:py-2 text-slate-200 placeholder:text-slate-500 focus:outline-none"
                />
                <datalist id="scholar-query-history">
                  {queryHistory.map((item) => (
                    <option key={item} value={item} />
                  ))}
                </datalist>
              </div>
            </div>
            <div className="flex-1 min-w-[120px]">
              <label className="block font-medium text-slate-400 mb-1">{t('scholar.sourceLabel')}</label>
              <select
                value={source}
                onChange={(e) => setSource(e.target.value as ScholarSource)}
                className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 sm:px-3 py-1.5 sm:py-2 text-slate-200 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
              >
                {SOURCE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {t(opt.labelKey)}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex gap-1.5 sm:gap-2">
              <div className="w-16 sm:w-20">
                <label className="block font-medium text-slate-400 mb-1 truncate">{t('scholar.yearStart')}</label>
                <input
                  type="number"
                  min={1900}
                  max={2100}
                  value={yearStart ?? ''}
                  onChange={(e) => setYearStart(e.target.value === '' ? null : parseInt(e.target.value, 10) || null)}
                  placeholder="—"
                  className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 sm:py-2 text-slate-200 focus:outline-none"
                />
              </div>
              <div className="w-16 sm:w-20">
                <label className="block font-medium text-slate-400 mb-1 truncate">{t('scholar.yearEnd')}</label>
                <input
                  type="number"
                  min={1900}
                  max={2100}
                  value={yearEnd ?? ''}
                  onChange={(e) => setYearEnd(e.target.value === '' ? null : parseInt(e.target.value, 10) || null)}
                  placeholder="—"
                  className="w-full rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 sm:py-2 text-slate-200 focus:outline-none"
                />
              </div>
            </div>
            <div className="w-20 sm:w-24">
              <label className="block font-medium text-slate-400 mb-1">{t('scholar.limitLabel')}</label>
              <div className="flex rounded-lg border border-slate-600/80 bg-slate-800/60 overflow-hidden">
                <input
                  type="number"
                  min={1}
                  step={10}
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value, 10) || 30)}
                  className="flex-1 min-w-0 bg-transparent px-2 py-1.5 sm:py-2 text-slate-200 focus:outline-none"
                />
                <div className="flex flex-col border-l border-slate-600/80">
                  <button
                    type="button"
                    onClick={() => setLimit(limit + 10)}
                    className="flex items-center justify-center p-0.5 text-slate-400 hover:bg-slate-600/60 hover:text-slate-200 transition-colors"
                  >
                    <ChevronUp size={12} />
                  </button>
                  <button
                    type="button"
                    onClick={() => setLimit(Math.max(1, limit - 10))}
                    className="flex items-center justify-center p-0.5 text-slate-400 hover:bg-slate-600/60 hover:text-slate-200 transition-colors"
                  >
                    <ChevronDown size={12} />
                  </button>
                </div>
              </div>
            </div>
            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-medium transition-colors"
            >
              {isSearching ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
              <span>{t('common.search')}</span>
            </button>
          </div>

          {/* Row 2: Library, Settings & Options */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <div className="flex items-center gap-2">
              <label className="font-medium text-slate-400 whitespace-nowrap">{t('scholar.libraryLabel')}:</label>
              <select
                value={activeLibraryId ?? ''}
                onChange={(e) => setActiveLibrary(e.target.value === '' ? null : Number(e.target.value))}
                className="w-[140px] sm:w-[160px] rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1 sm:py-1.5 text-slate-200 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
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
            
            <button
              type="button"
              onClick={() => setShowAdvancedSettings(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1 sm:py-1.5 font-medium text-slate-200 hover:border-slate-500 hover:bg-slate-800 transition-colors"
            >
              <Settings2 size={14} className="text-sky-300" />
              <span>{t('scholar.advancedSettingsTitle')}</span>
            </button>

            <button
              type="button"
              onClick={() => setShowRecommendModal(true)}
              disabled={!activeLibraryId || activeLibraryId < 0}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-600/80 bg-slate-800/60 px-2 py-1 sm:py-1.5 font-medium text-slate-200 hover:border-slate-500 hover:bg-slate-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <FileSearch size={14} className="text-amber-300" />
              <span>{t('scholar.recommendButton')}</span>
            </button>

            <div className="h-4 w-px bg-slate-700/60 mx-1 hidden sm:block"></div>

            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={handleShowHeadedBrowser}
                disabled={headedBrowserWindowBusy || !headedBrowserWindow?.available}
                className={
                  headedBrowserWindow?.mode === 'parked'
                    ? 'p-1 rounded border transition-colors text-slate-200 disabled:opacity-30 ring-2 ring-teal-400/60 bg-teal-900/30 border-teal-600/60'
                    : 'p-1 rounded border transition-colors text-slate-200 disabled:opacity-30 bg-slate-700/50 border-slate-600/80 hover:bg-slate-600'
                }
                title={t('scholar.headedBrowserSummon')}
              >
                <PanelLeftOpen size={14} className="text-teal-300" />
              </button>
              <button
                type="button"
                onClick={handleParkHeadedBrowser}
                disabled={headedBrowserWindowBusy || !headedBrowserWindow?.available}
                className={
                  headedBrowserWindow?.mode === 'visible'
                    ? 'p-1 rounded border transition-colors text-slate-200 disabled:opacity-30 ring-2 ring-slate-400/60 bg-slate-700/80 border-slate-500/60'
                    : 'p-1 rounded border transition-colors text-slate-200 disabled:opacity-30 bg-slate-700/50 border-slate-600/80 hover:bg-slate-600'
                }
                title={t('scholar.headedBrowserPark')}
              >
                <PanelLeftClose size={14} className="text-slate-300" />
              </button>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={smartOptimize}
                  onChange={(e) => setSmartOptimize(e.target.checked)}
                  className="rounded border-slate-500 bg-slate-800 text-teal-500 focus:ring-teal-500/50 w-3.5 h-3.5"
                />
                <Sparkles size={14} className="text-teal-400 shrink-0" />
                <span className="text-slate-400 whitespace-nowrap">{t('scholar.smartOptimize')}</span>
              </label>
              
              {smartOptimize && (
                <label className="flex items-center gap-1.5 cursor-pointer select-none" title={t('scholar.batchAllSourcesHint')}>
                  <input
                    type="checkbox"
                    checked={batchAllSources}
                    onChange={(e) => setBatchAllSources(e.target.checked)}
                    className="rounded border-slate-500 bg-slate-800 text-purple-500 focus:ring-purple-500/50 w-3.5 h-3.5"
                  />
                  <Layers size={14} className="text-purple-400 shrink-0" />
                  <span className="text-slate-400 whitespace-nowrap">{t('scholar.batchAllSources')}</span>
                </label>
              )}

              {!(smartOptimize && batchAllSources) && (source === 'google_scholar' || source === 'google') && (
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1.5 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={useSerpapi}
                      onChange={(e) => setUseSerpapi(e.target.checked)}
                      className="rounded border-slate-500 bg-slate-800 text-teal-500 focus:ring-teal-500/50 w-3.5 h-3.5"
                    />
                    <span className="text-slate-400 whitespace-nowrap">{t('sidebar.useSerpapi')}</span>
                  </label>
                  {useSerpapi && (
                    <select
                      value={serpapiRatio}
                      onChange={(e) => setSerpapiRatio(Number(e.target.value))}
                      className="rounded border border-slate-600/80 bg-slate-800/60 px-1.5 py-0.5 text-slate-200 focus:outline-none"
                    >
                      {[0, 25, 33, 50, 67, 75, 100].map((v) => (
                        <option key={v} value={v}>{v}%</option>
                      ))}
                    </select>
                  )}
                </div>
              )}
            </div>

            <div className="flex-1"></div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setShowNewLibraryModal(true)}
                className="flex items-center gap-1.5 px-2.5 py-1 sm:py-1.5 rounded-lg border border-slate-600/80 bg-slate-800/60 text-slate-300 hover:bg-slate-700 hover:text-slate-200 transition-colors"
              >
                <FolderPlus size={14} />
                <span>{t('scholar.libraryNew')}</span>
              </button>
              <button
                type="button"
                onClick={handlePickLibraryPdfFiles}
                disabled={importingLibraryPdfs}
                className="flex items-center gap-1.5 px-2.5 py-1 sm:py-1.5 rounded-lg border border-slate-600/80 bg-slate-800/60 text-slate-300 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 transition-colors"
              >
                {importingLibraryPdfs ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
                <span>{t('scholar.libraryImportPdf')}</span>
              </button>
            </div>
          </div>
        </div>
      <ScholarAdvancedSettingsModal
        open={showAdvancedSettings}
        onClose={() => setShowAdvancedSettings(false)}
      />

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
        {libraryImportSummary && (
          <div className="rounded-lg border border-slate-600/70 bg-slate-800/50 px-4 py-3 text-xs text-slate-300">
            {t('scholar.libraryImportResultLine', {
              total: libraryImportSummary.total_files,
              imported: libraryImportSummary.imported,
              linked: libraryImportSummary.linked_existing,
              renamed: libraryImportSummary.renamed,
              skipped: libraryImportSummary.skipped_duplicates,
              invalid: libraryImportSummary.invalid_pdf,
              noDoi: libraryImportSummary.no_doi,
            })}
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
            </div>
            <div className="flex justify-end gap-2 mt-6">
              <button
                type="button"
                onClick={() => {
                  setShowNewLibraryModal(false);
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
                  const lib = await createLibrary(name, newLibraryDesc.trim() || undefined, undefined, false);
                  if (lib) {
                    setActiveLibrary(lib.id);
                    setShowNewLibraryModal(false);
                    setNewLibraryName('');
                    setNewLibraryDesc('');
                    addToast(t('scholar.libraryCreate'), 'success');
                  } else {
                    const err = useScholarStore.getState().libraryError;
                    addToast(err || t('scholar.libraryCreate'), 'error');
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
      <div ref={resultsLayoutRef} className="flex-1 flex min-h-0 gap-4 px-4 pb-4">
        {/* Library panel (left) */}
        {activeLibraryId != null && (
          <>
            <div
              className="flex-shrink-0 flex flex-col rounded-xl border border-slate-700/60 bg-slate-800/30 overflow-hidden"
              style={{ width: libraryPanelWidth }}
            >
              <button
                onClick={() => setLibraryPanelOpen((o) => !o)}
                className="flex items-center justify-between px-4 py-2 border-b border-slate-700/60 bg-slate-800/50 text-slate-200 text-sm font-medium"
              >
                <span className="flex items-center gap-2 truncate">
                  <BookOpen size={16} />
                  {t('scholar.libraryPanelTitle')}{' '}
                  {activeFilterCount > 0 && libraryPapers.length > 0 ? (
                    <span className="text-slate-400">
                      ({filteredLibraryPapers.length} / {libraryPapers.length})
                    </span>
                  ) : (
                    <span className="text-slate-400">({libraryPapers.length})</span>
                  )}
                </span>
                {libraryPanelOpen ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
              </button>
              {libraryPanelOpen && (
                <>
                  {activeFilterCount > 0 && (
                    <p className="px-4 py-1 text-xs text-slate-500 border-b border-slate-700/40" title={t('scholar.filtersApplyToCatalog')}>
                      {t('scholar.filtersApplyToCatalog')}
                    </p>
                  )}
                  <div className="flex items-center gap-1.5 p-1.5 border-b border-slate-700/60">
                    <button
                      type="button"
                      onClick={async () => {
                        const result = await downloadLibraryBatch(currentCollection);
                        if (result == null) {
                          addToast(t('scholar.downloadFailed'), 'error');
                          return;
                        }
                        if (result.taskId) {
                          addToast(t('scholar.batchDownloadStarted', { count: result.submittedCount }), 'success');
                          return;
                        }
                        if (result.skippedReason === 'all_downloaded') {
                          addToast(t('scholar.allAlreadyDownloaded'), 'info');
                        }
                      }}
                      disabled={libraryPapers.length === 0 || libraryLoading}
                      className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-60 text-white text-xs font-medium"
                    >
                      <Download size={14} />
                      <span className="truncate">{t('scholar.libraryDownloadAll')}</span>
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        setExtractDoiDedupLoading(true);
                        try {
                          const stats = await extractDoiAndDedup();
                          if (stats != null) {
                            addToast(
                              t('scholar.libraryExtractDoiDedupSuccess', {
                                extracted: stats.extracted_count,
                                removed: stats.removed_count,
                              }),
                              'success',
                            );
                          } else {
                            addToast(t('scholar.downloadFailed'), 'error');
                          }
                        } finally {
                          setExtractDoiDedupLoading(false);
                        }
                      }}
                      disabled={libraryPapers.length === 0 || libraryLoading || extractDoiDedupLoading}
                      className="flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg border border-slate-500/60 bg-slate-700/50 hover:bg-slate-600/50 disabled:opacity-60 text-slate-200 text-xs font-medium"
                      title={t('scholar.libraryExtractDoiDedup')}
                    >
                      {extractDoiDedupLoading ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <FileSearch size={14} />
                      )}
                      <span>{t('scholar.libraryExtractDoiDedup')}</span>
                    </button>
                    {activeLibraryId != null && activeLibraryId >= 0 && (
                      <button
                        type="button"
                        onClick={async () => {
                          setRefreshMetadataLoading(true);
                          try {
                            const stats = await refreshMetadataFromCrossref();
                            if (stats != null) {
                              addToast(
                                t('scholar.libraryRefreshMetadataSuccess', {
                                  updated: stats.updated,
                                  skipped: stats.skipped_no_doi,
                                  failed: stats.failed,
                                }),
                                'success',
                              );
                            } else {
                              addToast(t('scholar.downloadFailed'), 'error');
                            }
                          } finally {
                            setRefreshMetadataLoading(false);
                          }
                        }}
                        disabled={libraryPapers.length === 0 || libraryLoading || refreshMetadataLoading}
                        className="flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg border border-slate-500/60 bg-slate-700/50 hover:bg-slate-600/50 disabled:opacity-60 text-slate-200 text-xs font-medium"
                        title={t('scholar.libraryRefreshMetadata')}
                      >
                        {refreshMetadataLoading ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <RefreshCcw size={14} />
                        )}
                        <span>{t('scholar.libraryRefreshMetadata')}</span>
                      </button>
                    )}
                    {activeLibraryId != null && activeLibraryId >= 0 && (
                      <button
                        type="button"
                        onClick={async () => {
                          setPdfRenameDedupLoading(true);
                          try {
                            const stats = await pdfRenameDedup();
                            if (stats != null) {
                              addToast(
                                t('scholar.pdfRenameDedupSuccess', {
                                  renamed: stats.renamed,
                                  removed: stats.removed,
                                  no_doi: stats.no_doi,
                                  synced_downloaded: stats.synced_downloaded ?? 0,
                                }),
                                'success',
                              );
                            } else {
                              addToast(t('scholar.downloadFailed'), 'error');
                            }
                          } finally {
                            setPdfRenameDedupLoading(false);
                          }
                        }}
                        disabled={libraryLoading || pdfRenameDedupLoading}
                        className="flex items-center justify-center gap-1 px-2 py-1.5 rounded-lg border border-slate-500/60 bg-slate-700/50 hover:bg-slate-600/50 disabled:opacity-60 text-slate-200 text-xs font-medium"
                        title={t('scholar.pdfRenameDedup')}
                      >
                        {pdfRenameDedupLoading ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <RefreshCw size={14} />
                        )}
                        <span>{t('scholar.pdfRenameDedup')}</span>
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => {
                        const activeLib = libraries.find((l) => l.id === activeLibraryId);
                        if (!activeLib) return;
                        if (activeLib.is_temporary) {
                          if (window.confirm(t('scholar.libraryClearConfirm'))) {
                            clearTemporaryLibrary(activeLibraryId);
                          }
                        } else {
                          setDeleteLibraryModal({ libId: activeLib.id, name: activeLib.name });
                          setDeleteLibraryPassword('');
                        }
                      }}
                      className="p-1.5 rounded-lg text-slate-400 hover:bg-red-500/20 hover:text-red-400"
                      title={
                        libraries.find((l) => l.id === activeLibraryId)?.is_temporary
                          ? t('scholar.libraryClear')
                          : t('scholar.libraryDelete')
                      }
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  {/* 文献目录内排序与筛选栏 */}
                  {libraryPapers.length > 0 && (
                    <div className="flex-shrink-0 border-b border-slate-700/60 bg-slate-800/40">
                      <button
                        type="button"
                        onClick={() => setCatalogFilterBarCollapsed((c) => !c)}
                        className="w-full flex items-center justify-between px-3 py-2 text-left text-xs text-slate-400 hover:text-slate-200"
                      >
                        <span>
                          {t('scholar.sortBy')}: {t(sortField === 'score' ? 'scholar.sortRelevance' : sortField === 'year' ? 'scholar.sortYear' : 'scholar.sortImpactFactor')}{' '}
                          · {sortDir === 'asc' ? t('scholar.sortAsc') : t('scholar.sortDesc')}
                          {activeFilterCount > 0 && ` · ${activeFilterCount} ${t('scholar.filtersActive')}`}
                        </span>
                        {catalogFilterBarCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                      </button>
                      {!catalogFilterBarCollapsed && (
                        <div className="px-3 pb-3 pt-0 flex flex-wrap items-end gap-2">
                          <input
                            type="text"
                            value={filterJournal}
                            onChange={(e) => setFilterJournal(e.target.value)}
                            placeholder={t('scholar.filterJournal')}
                            className="w-24 rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          />
                          <input
                            type="number"
                            value={filterYearMin ?? ''}
                            onChange={(e) => setFilterYearMin(e.target.value ? parseInt(e.target.value, 10) : null)}
                            placeholder={t('scholar.filterYearMin')}
                            className="w-14 rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          />
                          <input
                            type="number"
                            value={filterYearMax ?? ''}
                            onChange={(e) => setFilterYearMax(e.target.value ? parseInt(e.target.value, 10) : null)}
                            placeholder={t('scholar.filterYearMax')}
                            className="w-14 rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          />
                          <input
                            type="number"
                            step={0.1}
                            value={filterIfMin ?? ''}
                            onChange={(e) => setFilterIfMin(e.target.value ? parseFloat(e.target.value) : null)}
                            placeholder={t('scholar.filterImpactFactor')}
                            className="w-14 rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          />
                          <select
                            value={filterSource}
                            onChange={(e) => setFilterSource(e.target.value as ScholarSource | 'all')}
                            className="rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs min-w-0"
                          >
                            <option value="all">{t('scholar.statusAll')}</option>
                            {SOURCE_OPTIONS.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {t(opt.labelKey)}
                              </option>
                            ))}
                          </select>
                          <select
                            value={filterStatus}
                            onChange={(e) => setFilterStatus(e.target.value as 'all' | 'downloaded' | 'not_downloaded' | 'not_in_collection')}
                            className="rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          >
                            <option value="all">{t('scholar.statusAll')}</option>
                            <option value="downloaded">{t('scholar.statusDownloaded')}</option>
                            <option value="not_downloaded">{t('scholar.statusNotDownloaded')}</option>
                            <option value="not_in_collection">{t('scholar.statusNotInCollection')}</option>
                          </select>
                          <select
                            value={sortField}
                            onChange={(e) => setSortField(e.target.value as 'score' | 'year' | 'impact_factor')}
                            className="rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1 text-slate-200 text-xs"
                          >
                            <option value="score">{t('scholar.sortRelevance')}</option>
                            <option value="year">{t('scholar.sortYear')}</option>
                            <option value="impact_factor">{t('scholar.sortImpactFactor')}</option>
                          </select>
                          <button
                            type="button"
                            onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
                            className="px-2 py-1 rounded border border-slate-600/80 bg-slate-800/60 text-slate-400 hover:text-slate-200 text-xs"
                            title={sortDir === 'asc' ? t('scholar.sortDesc') : t('scholar.sortAsc')}
                          >
                            {sortDir === 'asc' ? '↑' : '↓'}
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setFilterJournal('');
                              setFilterYearMin(null);
                              setFilterYearMax(null);
                              setFilterIfMin(null);
                              setFilterSource('all');
                              setFilterStatus('all');
                            }}
                            className="px-2 py-1 rounded border border-slate-600/80 bg-slate-700/50 text-slate-400 hover:text-slate-200 text-xs"
                          >
                            {t('scholar.resetFilters')}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  <div className="flex-1 overflow-y-auto p-2 min-h-0">
                    {libraryLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 size={24} className="animate-spin text-sky-400" />
                      </div>
                    ) : libraryPapers.length === 0 ? (
                      <p className="text-slate-500 text-sm py-6 text-center">{t('scholar.libraryEmpty')}</p>
                    ) : filteredLibraryPapers.length === 0 ? (
                      <p className="text-slate-500 text-sm py-6 text-center">{t('scholar.noMatchFilters')}</p>
                    ) : (
                      <ul className="space-y-2">
                        {filteredLibraryPapers.map((p) => (
                          <li
                            key={p.id}
                            className="flex items-start gap-2 rounded-lg border border-slate-600/60 bg-slate-800/50 p-2 text-sm group"
                          >
                            <div className="flex-1 min-w-0">
                              {p.url ? (
                                <a
                                  href={p.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-slate-200 line-clamp-2 hover:text-sky-300 hover:underline block"
                                  title={p.title}
                                >
                                  {p.title}
                                </a>
                              ) : (
                                <p className="text-slate-200 line-clamp-2" title={p.title}>
                                  {p.title}
                                </p>
                              )}
                              <p className="text-xs text-slate-500 mt-0.5">
                                {p.authors?.join(', ')} {p.year != null ? ` · ${p.year}` : ''}
                              </p>
                              {(p.venue || p.impact_factor != null) && (
                                <p className="text-xs text-slate-400 mt-0.5">
                                  {p.venue ?? ''}
                                  {p.impact_factor != null
                                    ? ` · IF ${p.impact_factor}${p.jif_quartile ? ` ${p.jif_quartile}` : ''}`
                                    : (p.venue ? ` · IF ${t('scholar.impactFactorNA')}` : '')}
                                </p>
                              )}
                              <div className="flex flex-wrap items-center gap-1.5 mt-1">
                                {p.in_collection && (
                                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-600/50 text-emerald-200" title={p.collection_paper_id ?? undefined}>
                                    {t('scholar.inCurrentCollection')}
                                  </span>
                                )}
                                {p.source && (
                                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/60 text-slate-300">
                                    {p.source.replace(/_/g, ' ')}
                                  </span>
                                )}
                                {p.pdf_url && (
                                  <a
                                    href={p.pdf_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/50 text-sky-400 hover:bg-slate-600/70 hover:text-sky-300"
                                    title={t('scholar.openPdfLink')}
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    <FileText size={10} />
                                    PDF
                                  </a>
                                )}
                                {p.downloaded_at && p.paper_id ? (
                                  <>
                                    <button
                                      type="button"
                                      onClick={async () => {
                                        const url =
                                          activeLibraryId != null && activeLibraryId >= 0
                                            ? getLibraryPdfViewUrl(activeLibraryId, p.paper_id!)
                                            : getPdfViewUrl(p.paper_id!);
                                        const pdfUrl = url.includes('/scholar/libraries/')
                                          ? await fetchPdfAsBlobUrl(url).catch(() => url)
                                          : url;
                                        setPdfView({ paperId: p.paper_id!, title: p.title, pdfUrl });
                                      }}
                                      className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/40 text-green-500 hover:bg-slate-600/60 cursor-pointer"
                                      title={t('scholar.openPdf')}
                                    >
                                      {t('scholar.libraryPaperDownloaded')}
                                    </button>
                                    {activeLibraryId != null && activeLibraryId >= 0 && (
                                      <>
                                        <button
                                          type="button"
                                          disabled={deletingPdfId === p.id}
                                          onClick={async (e) => {
                                            e.stopPropagation();
                                            if (activeLibraryId == null) return;
                                            setDeletingPdfId(p.id);
                                            try {
                                              const r = await deleteLibraryPaperPdf(activeLibraryId, p.id);
                                              await loadLibraryPapers(activeLibraryId);
                                              addToast(
                                                r.removed_from_collection
                                                  ? `${t('scholar.pdfDeleted')}（已同步移除向量）`
                                                  : t('scholar.pdfDeleted'),
                                                'success',
                                              );
                                            } catch (err) {
                                              addToast((err as Error)?.message || t('scholar.downloadFailed'), 'error');
                                            } finally {
                                              setDeletingPdfId(null);
                                            }
                                          }}
                                          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/50 text-slate-300 hover:bg-red-500/20 hover:text-red-400 disabled:opacity-70"
                                          title={t('scholar.deletePdf')}
                                        >
                                          {deletingPdfId === p.id ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
                                          {t('scholar.deletePdf')}
                                        </button>
                                        <button
                                          type="button"
                                          disabled={redownloadingId === p.id}
                                          onClick={async (e) => {
                                            e.stopPropagation();
                                            if (activeLibraryId == null) return;
                                            setRedownloadingId(p.id);
                                            try {
                                              await deleteLibraryPaperPdf(activeLibraryId, p.id);
                                              await loadLibraryPapers(activeLibraryId);
                                              await downloadLibraryPaperAndOpen(p);
                                            } catch (err) {
                                              addToast((err as Error)?.message || t('scholar.downloadFailed'), 'error');
                                            } finally {
                                              setRedownloadingId(null);
                                            }
                                          }}
                                          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/50 text-sky-400 hover:bg-slate-600/70 disabled:opacity-70"
                                          title={t('scholar.redownload')}
                                        >
                                          {redownloadingId === p.id ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
                                          {t('scholar.redownload')}
                                        </button>
                                      </>
                                    )}
                                  </>
                                ) : !p.downloaded_at && p.paper_id ? (
                                  (() => {
                                    const isPending = pendingDownloadAndOpen?.paperId === p.paper_id;
                                    const isFailed = !isPending && !!downloadAndOpenFailures[p.paper_id!];
                                    return (
                                      <button
                                        type="button"
                                        onClick={() => {
                                          if (isFailed) clearDownloadAndOpenFailure(p.paper_id!);
                                          downloadLibraryPaperAndOpen(p);
                                        }}
                                        disabled={isPending}
                                        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium disabled:opacity-70 ${
                                          isFailed
                                            ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 hover:text-red-300'
                                            : 'bg-slate-600/50 text-sky-400 hover:bg-slate-600/70 hover:text-sky-300'
                                        }`}
                                        title={isFailed ? t('scholar.downloadAndOpenFailed') : t('scholar.downloadAndOpen')}
                                      >
                                        {isPending ? (
                                          <Loader2 size={10} className="animate-spin" />
                                        ) : isFailed ? (
                                          <AlertCircle size={10} />
                                        ) : (
                                          <Download size={10} />
                                        )}
                                        {isFailed ? t('scholar.downloadAndOpenFailed') : t('scholar.downloadAndOpen')}
                                      </button>
                                    );
                                  })()
                                ) : (
                                  <span
                                    className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/40 ${p.downloaded_at ? 'text-green-500' : 'text-slate-400'}`}
                                  >
                                    {p.downloaded_at ? t('scholar.libraryPaperDownloaded') : t('scholar.libraryPaperNotDownloaded')}
                                  </span>
                                )}
                                {activeLibraryId != null && activeLibraryId >= 0 && !p.downloaded_at && (
                                  <>
                                    <input
                                      type="file"
                                      accept=".pdf"
                                      className="hidden"
                                      id={`upload-pdf-${p.id}`}
                                      onChange={(e) => {
                                        const f = e.target.files?.[0];
                                        if (f) handleUploadPdf(p, f);
                                        e.target.value = '';
                                      }}
                                    />
                                    <button
                                      type="button"
                                      onClick={() => document.getElementById(`upload-pdf-${p.id}`)?.click()}
                                      disabled={!!uploadingPapers[p.id]}
                                      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-600/50 text-amber-400 hover:bg-slate-600/70 disabled:opacity-70"
                                      title={t('scholar.uploadPdf')}
                                    >
                                      {uploadingPapers[p.id] ? (
                                        <Loader2 size={10} className="animate-spin" />
                                      ) : (
                                        <Upload size={10} />
                                      )}
                                      {t('scholar.uploadPdf')}
                                    </button>
                                  </>
                                )}
                              </div>
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
                              className="flex-shrink-0 p-1.5 rounded text-slate-400 hover:bg-red-500/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                              title={t('scholar.libraryRemoveFromCatalog')}
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
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize library panel"
              onPointerDown={(e) => {
                e.preventDefault();
                setIsResizingLibraryPanel(true);
              }}
              className={`hidden md:flex w-3 -mx-1 flex-shrink-0 flex-col items-center justify-center gap-2 cursor-col-resize select-none touch-none ${isResizingLibraryPanel ? 'opacity-100' : 'opacity-70 hover:opacity-100'}`}
            >
              <div className="h-full w-px bg-slate-700/80" />
              <div className="h-14 w-1.5 rounded-full bg-slate-500/60" />
            </div>
          </>
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
                {/* Filter/sort bar (collapsible) */}
                <div className="flex-shrink-0 border-b border-slate-700/60 bg-slate-800/40">
                  <button
                    type="button"
                    onClick={() => setFilterBarCollapsed((c) => !c)}
                    className="w-full flex items-center justify-between px-4 py-2 text-left text-sm text-slate-400 hover:text-slate-200"
                  >
                    <span>
                      {t('scholar.sortBy')}: {t(sortField === 'score' ? 'scholar.sortRelevance' : sortField === 'year' ? 'scholar.sortYear' : 'scholar.sortImpactFactor')}{' '}
                      · {sortDir === 'asc' ? t('scholar.sortAsc') : t('scholar.sortDesc')}
                      {activeFilterCount > 0 && ` · ${activeFilterCount} ${t('scholar.filtersActive')}`}
                    </span>
                    {filterBarCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                  </button>
                  {!filterBarCollapsed && (
                    <div className="px-4 pb-3 pt-0 flex flex-wrap items-end gap-3">
                      <div className="w-32">
                        <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterJournal')}</label>
                        <input
                          type="text"
                          value={filterJournal}
                          onChange={(e) => setFilterJournal(e.target.value)}
                          placeholder=""
                          className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        />
                      </div>
                      <div className="w-20">
                        <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterYearMin')}</label>
                        <input
                          type="number"
                          value={filterYearMin ?? ''}
                          onChange={(e) => setFilterYearMin(e.target.value ? parseInt(e.target.value, 10) : null)}
                          placeholder="—"
                          className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        />
                      </div>
                      <div className="w-20">
                        <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterYearMax')}</label>
                        <input
                          type="number"
                          value={filterYearMax ?? ''}
                          onChange={(e) => setFilterYearMax(e.target.value ? parseInt(e.target.value, 10) : null)}
                          placeholder="—"
                          className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        />
                      </div>
                      <div className="w-24">
                        <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterImpactFactor')}</label>
                        <input
                          type="number"
                          step={0.1}
                          value={filterIfMin ?? ''}
                          onChange={(e) => setFilterIfMin(e.target.value ? parseFloat(e.target.value) : null)}
                          placeholder="—"
                          className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        />
                      </div>
                      <div className="w-28">
                        <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterSource')}</label>
                        <select
                          value={filterSource}
                          onChange={(e) => setFilterSource(e.target.value as ScholarSource | 'all')}
                          className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        >
                          <option value="all">{t('scholar.statusAll')}</option>
                          {SOURCE_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {t(opt.labelKey)}
                            </option>
                          ))}
                        </select>
                      </div>
                      {activeLibraryId != null && (
                        <div className="w-28">
                          <label className="block text-xs text-slate-500 mb-0.5">{t('scholar.filterStatus')}</label>
                          <select
                            value={filterStatus}
                            onChange={(e) => setFilterStatus(e.target.value as 'all' | 'downloaded' | 'not_downloaded' | 'not_in_collection')}
                            className="w-full rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                          >
                            <option value="all">{t('scholar.statusAll')}</option>
                            <option value="downloaded">{t('scholar.statusDownloaded')}</option>
                            <option value="not_downloaded">{t('scholar.statusNotDownloaded')}</option>
                            <option value="not_in_collection">{t('scholar.statusNotInCollection')}</option>
                          </select>
                        </div>
                      )}
                      <label className="flex items-center gap-2 text-sm text-slate-400">
                        <input
                          type="checkbox"
                          checked={filterDownloadableOnly}
                          onChange={(e) => setFilterDownloadableOnly(e.target.checked)}
                          className="rounded border-slate-500"
                        />
                        {t('scholar.filterDownloadableOnly')}
                      </label>
                      <div className="flex items-center gap-2">
                        <label className="text-xs text-slate-500">{t('scholar.sortBy')}</label>
                        <select
                          value={sortField}
                          onChange={(e) => setSortField(e.target.value as 'score' | 'year' | 'impact_factor')}
                          className="rounded border border-slate-600/80 bg-slate-800/60 px-2 py-1.5 text-slate-200 text-sm"
                        >
                          <option value="score">{t('scholar.sortRelevance')}</option>
                          <option value="year">{t('scholar.sortYear')}</option>
                          <option value="impact_factor">{t('scholar.sortImpactFactor')}</option>
                        </select>
                        <button
                          type="button"
                          onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
                          className="px-2 py-1.5 rounded border border-slate-600/80 bg-slate-800/60 text-slate-400 hover:text-slate-200 text-sm"
                          title={sortDir === 'asc' ? t('scholar.sortDesc') : t('scholar.sortAsc')}
                        >
                          {sortDir === 'asc' ? '↑' : '↓'}
                        </button>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setFilterJournal('');
                          setFilterYearMin(null);
                          setFilterYearMax(null);
                          setFilterIfMin(null);
                          setFilterSource('all');
                          setFilterStatus('all');
                          setFilterDownloadableOnly(false);
                        }}
                        className="px-3 py-1.5 rounded border border-slate-600/80 bg-slate-700/50 text-slate-400 hover:text-slate-200 text-sm"
                      >
                        {t('scholar.resetFilters')}
                      </button>
                    </div>
                  )}
                </div>
                {/* Table header with select all */}
                <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-slate-700/60 bg-slate-800/50">
                  <button
                    onClick={() =>
                      allSelected
                        ? clearSelection()
                        : setSelectedIndices(filteredResults.map(({ originalIndex }) => originalIndex))
                    }
                    className="p-1 rounded text-slate-400 hover:text-sky-400 transition-colors"
                    title={allSelected ? t('scholar.clearSelection') : t('scholar.selectAll')}
                  >
                    {allSelected ? <CheckSquare size={20} /> : <Square size={20} />}
                  </button>
                  <span className="font-medium text-slate-200">{t('scholar.searchResultsTitle')}</span>
                  <span className="text-slate-500 text-sm">
                    {filteredResults.length}
                    {filteredResults.length !== results.length && ` / ${results.length}`}{' '}
                    {t('scholar.resultsCount')}
                  </span>
                  {batchSourceCounts && (
                    <span className="flex items-center gap-1 text-xs text-purple-400" title={t('scholar.batchSourceCountsHint')}>
                      <Layers size={12} />
                      {Object.entries(batchSourceCounts)
                        .filter(([, n]) => n > 0)
                        .map(([src, n]) => `${src.replace('_', ' ')}: ${n}`)
                        .join(' · ')}
                    </span>
                  )}
                  {activeFilterCount > 0 && (
                    <span className="text-xs text-sky-400" title={t('scholar.filtersActive')}>
                      ({activeFilterCount})
                    </span>
                  )}
                </div>

                <div className="flex-1 overflow-y-auto">
                  {filteredResults.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center py-16 px-4 text-center">
                      <AlertCircle className="text-slate-500 mb-3" size={28} />
                      <p className="text-slate-300 text-sm">{t('scholar.noMatchFilters')}</p>
                      <button
                        type="button"
                        onClick={() => {
                          setFilterJournal('');
                          setFilterYearMin(null);
                          setFilterYearMax(null);
                          setFilterIfMin(null);
                          setFilterSource('all');
                          setFilterStatus('all');
                          setFilterDownloadableOnly(false);
                        }}
                        className="mt-3 px-3 py-1.5 rounded border border-slate-600/80 bg-slate-700/50 text-slate-300 hover:text-slate-100 text-xs"
                      >
                        {t('scholar.resetFilters')}
                      </button>
                    </div>
                  )}
                  {/* Desktop table */}
                  <table className={`w-full text-left text-sm ${filteredResults.length === 0 ? 'hidden' : 'hidden md:table'}`}>
                    <thead className="sticky top-0 bg-slate-800/90 border-b border-slate-700/60 z-10">
                      <tr className="text-slate-400">
                        <th className="w-10 py-2 px-2"></th>
                        <th className="py-2 px-3 font-medium">{t('scholar.colTitle')}</th>
                        <th className="py-2 px-3 font-medium w-32">{t('scholar.colAuthors')}</th>
                        <th className="py-2 px-3 font-medium w-16">{t('scholar.colYear')}</th>
                        <th className="py-2 px-3 font-medium w-28">{t('scholar.colJournal')}</th>
                        <th className="py-2 px-3 font-medium w-20">{t('scholar.colImpactFactor')}</th>
                        <th className="py-2 px-3 font-medium w-24">{t('scholar.colSource')}</th>
                        <th className="py-2 px-3 font-medium w-28">{t('scholar.colActions')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredResults.map(({ result: item, originalIndex: index }) => {
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
                            <td className="py-2 px-3 text-slate-400 text-xs max-w-[140px] truncate" title={m.venue ?? ''}>
                              {m.venue ?? '—'}
                            </td>
                            <td className="py-2 px-3">
                              {m.impact_factor != null ? (
                                <span
                                  className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${
                                    m.jif_quartile === 'Q1'
                                      ? 'bg-emerald-500/20 text-emerald-400'
                                      : m.jif_quartile === 'Q2'
                                        ? 'bg-sky-500/20 text-sky-400'
                                        : m.jif_quartile === 'Q3'
                                          ? 'bg-amber-500/20 text-amber-400'
                                          : 'bg-slate-500/20 text-slate-400'
                                  }`}
                                  title={m.jif_quartile ?? ''}
                                >
                                  {m.impact_factor}
                                  {m.jif_quartile ? ` ${m.jif_quartile}` : ''}
                                </span>
                              ) : (
                                t('scholar.impactFactorNA')
                              )}
                            </td>
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
                  <div className={`divide-y divide-slate-700/40 ${filteredResults.length === 0 ? 'hidden' : 'md:hidden'}`}>
                    {filteredResults.map(({ result: item, originalIndex: index }) => {
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
                              {(m.venue || m.impact_factor != null) && (
                                <p className="text-xs text-slate-400 mt-0.5">
                                  {m.venue ?? ''}
                                  {m.impact_factor != null
                                    ? ` · IF ${m.impact_factor}${m.jif_quartile ? ` ${m.jif_quartile}` : ''}`
                                    : (m.venue ? ` · IF ${t('scholar.impactFactorNA')}` : '')}
                                </p>
                              )}
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
                  const payload = task.payload as { total?: number; completed?: number; failed?: number; stage?: string; paper_id?: string };
                  const total = payload?.total ?? 1;
                  const completed = payload?.completed ?? 0;
                  const isDone = task.status === 'completed' || task.status === 'error';
                  const paperId = payload?.paper_id;
                  const stage = payload?.stage;
                  const runningLabel =
                    task.status === 'running' && (stage === 'INGEST_QUEUED' || stage === 'INGESTING')
                      ? t('scholar.taskStage.ingesting')
                      : t(`scholar.taskStatus.${task.status}`);
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
                            {runningLabel}
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
          onClose={() => {
            if (pdfView?.pdfUrl?.startsWith?.('blob:')) URL.revokeObjectURL(pdfView.pdfUrl);
            setPdfView(null);
          }}
          pdfUrl={'pdfUrl' in pdfView && pdfView.pdfUrl ? pdfView.pdfUrl : getPdfViewUrl(pdfView.paperId)}
          title={pdfView.title}
        />
      )}

      {showRecommendModal && activeLibraryId && activeLibraryId > 0 && (
        <ScholarLibraryRecommendModal
          open={showRecommendModal}
          onClose={() => setShowRecommendModal(false)}
          libraryId={activeLibraryId}
          collection={currentCollection ?? ''}
          libraryPapers={libraryPapers}
          filteredPaperIds={filteredLibraryPapers.map((p) => p.id)}
        />
      )}

      {/* 删除文献库：密码确认 Modal */}
      <Modal
        open={deleteLibraryModal !== null}
        onClose={() => { setDeleteLibraryModal(null); setDeleteLibraryPassword(''); }}
        title="删除文献库（危险操作）"
        maxWidth="max-w-md"
      >
        {deleteLibraryModal && (
          <div className="space-y-4">
            <p className="text-sm text-gray-700">
              将永久删除文献库「<strong>{deleteLibraryModal.name}</strong>」及其所有文献记录与本地 PDF 目录，此操作不可恢复。
            </p>
            <p className="text-sm text-amber-700">
              请输入当前登录账号密码以确认：
            </p>
            <input
              type="password"
              value={deleteLibraryPassword}
              onChange={(e) => setDeleteLibraryPassword(e.target.value)}
              placeholder="密码"
              className="w-full border rounded-md p-2 text-sm focus:ring-2 focus:ring-red-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleConfirmDeleteLibraryWithPassword()}
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => { setDeleteLibraryModal(null); setDeleteLibraryPassword(''); }}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
              >
                取消
              </button>
              <button
                onClick={handleConfirmDeleteLibraryWithPassword}
                disabled={!deleteLibraryPassword.trim() || deleteLibraryVerifying}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 text-sm flex items-center gap-2"
              >
                {deleteLibraryVerifying ? <Loader2 size={14} className="animate-spin" /> : null}
                确认删除
              </button>
            </div>
          </div>
        )}
      </Modal>

    </div>
  );
}
