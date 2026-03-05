import { create } from 'zustand';
import type {
  ScholarSearchResult,
  ScholarSource,
  ScholarHealth,
  DownloadTaskStatus,
  ScholarLibrary,
  ScholarLibraryPaper,
} from '../api/scholar';
import {
  searchScholar,
  downloadPaper,
  batchDownloadPapers,
  getDownloadTaskStatus,
  getScholarHealth,
  isSubmittedTask,
  listLibraries,
  createLibrary as createLibraryApi,
  deleteLibrary as deleteLibraryApi,
  getLibraryPapers,
  addPapersToLibrary as addPapersToLibraryApi,
  removePaperFromLibrary as removePaperFromLibraryApi,
  extractDoiAndDedupLibrary,
  extractDoiAndDedupPapers,
  pdfRenameDedup as pdfRenameDedupApi,
} from '../api/scholar';

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES = new Set(['completed', 'error', 'cancelled']);

// ─── Temp library localStorage helpers ───────────────────────────────────────

const LS_TEMP_LIBS_KEY = 'scholar_temp_libs_v1';
const LS_TEMP_LIB_SEQ_KEY = 'scholar_temp_lib_seq_v1';
const LS_TEMP_PAPER_SEQ_KEY = 'scholar_temp_paper_seq_v1';
const TEMP_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

interface TempLibData {
  id: number; // negative integer — distinguishes from server permanent libs
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  created_at_ts: number; // Date.now() at creation — used for TTL eviction
  papers: ScholarLibraryPaper[];
}

function _nextSeq(key: string): number {
  try {
    const seq = parseInt(localStorage.getItem(key) || '0') - 1;
    localStorage.setItem(key, String(seq));
    return seq;
  } catch {
    return -(Math.abs(Date.now()) % 1_000_000_000);
  }
}

function _loadTempLibs(): TempLibData[] {
  try {
    const raw = localStorage.getItem(LS_TEMP_LIBS_KEY);
    if (!raw) return [];
    const libs: TempLibData[] = JSON.parse(raw);
    const now = Date.now();
    const alive = libs.filter((l) => now - l.created_at_ts < TEMP_TTL_MS);
    if (alive.length !== libs.length) _saveTempLibs(alive);
    return alive;
  } catch {
    return [];
  }
}

function _saveTempLibs(libs: TempLibData[]): void {
  try {
    localStorage.setItem(LS_TEMP_LIBS_KEY, JSON.stringify(libs));
  } catch {
    // localStorage quota exceeded — silently ignore
  }
}

function _tempToScholarLibrary(lib: TempLibData): ScholarLibrary {
  return {
    id: lib.id,
    name: lib.name,
    description: lib.description,
    paper_count: lib.papers.length,
    created_at: lib.created_at,
    updated_at: lib.updated_at,
    is_temporary: true,
    folder_path: null,
  };
}

function _searchResultToTempPaper(item: ScholarSearchResult, libId: number): ScholarLibraryPaper {
  const m = item.metadata;
  return {
    id: _nextSeq(LS_TEMP_PAPER_SEQ_KEY),
    library_id: libId,
    title: (m.title || '').trim() || '(无标题)',
    authors: m.authors || [],
    year: typeof m.year === 'number' ? m.year : null,
    doi: m.doi || null,
    pdf_url: m.pdf_url || null,
    url: m.url || null,
    source: m.source || '',
    score: item.score || 0,
    annas_md5: m.annas_md5 || null,
    added_at: new Date().toISOString(),
  };
}

/**
 * Append items to a temp lib with DOI/title deduplication.
 * Mutates lib in-place; caller must call _saveTempLibs after.
 * Returns count of actually added papers.
 */
function _appendToTempLib(lib: TempLibData, items: ScholarSearchResult[]): number {
  const existingDois = new Set(
    lib.papers.map((p) => (p.doi || '').trim().toLowerCase()).filter(Boolean),
  );
  const existingTitles = new Set(lib.papers.map((p) => p.title.trim().toLowerCase()));
  let added = 0;
  for (const item of items) {
    const paper = _searchResultToTempPaper(item, lib.id);
    const doi = (paper.doi || '').trim().toLowerCase();
    const title = paper.title.trim().toLowerCase();
    if (doi && existingDois.has(doi)) continue;
    if (!doi && existingTitles.has(title)) continue;
    lib.papers.push(paper);
    if (doi) existingDois.add(doi);
    existingTitles.add(title);
    added++;
  }
  if (added > 0) lib.updated_at = new Date().toISOString();
  return added;
}

// ─── State interface ──────────────────────────────────────────────────────────

type PendingDownloadAndOpen = { taskId: string; libId: number | null; paperId: string; title: string };

interface ScholarState {
  /** paper_id -> true for papers whose download-and-open task ended in error */
  downloadAndOpenFailures: Record<string, true>;
  // Search
  query: string;
  source: ScholarSource;
  yearStart: number | null;
  yearEnd: number | null;
  limit: number;
  smartOptimize: boolean;
  useSerpapi: boolean;
  serpapiRatio: number; // 0–100, used when source is google_scholar or google and useSerpapi is true
  results: ScholarSearchResult[];
  isSearching: boolean;
  searchError: string | null;

  // Selection (indices into results)
  selectedIndices: number[];

  // Download tasks (task_id -> status)
  downloadTasks: Record<string, DownloadTaskStatus>;
  pollIntervals: Record<string, ReturnType<typeof setInterval>>;

  // Health (null = not loaded or network error; 404 sets enabled: false + scholarHealthError)
  scholarHealth: ScholarHealth | null;
  scholarHealthError: '404' | null;

  // Sub-libraries (named candidate lists)
  libraries: ScholarLibrary[];
  activeLibraryId: number | null;
  libraryPapers: ScholarLibraryPaper[];
  libraryLoading: boolean;
  libraryError: string | null;

  // LLM for downloader assist (provider + model; used when calling download API)
  selectedScholarLlmProvider: string;
  selectedScholarLlmModel: string;
  setScholarLlm: (provider: string, model: string) => void;

  // 有头/无头：default=配置默认, headed=有头, headless=无头
  scholarBrowserMode: 'default' | 'headed' | 'headless';
  setScholarBrowserMode: (mode: 'default' | 'headed' | 'headless') => void;

  // Academia.edu 下载开关（默认关闭：跳过 Academia 避免卡住）
  includeAcademia: boolean;
  setIncludeAcademia: (v: boolean) => void;

  // Actions
  setQuery: (q: string) => void;
  setSource: (s: ScholarSource) => void;
  setYearStart: (y: number | null) => void;
  setYearEnd: (y: number | null) => void;
  setLimit: (n: number) => void;
  setSearchError: (err: string | null) => void;
  setSmartOptimize: (v: boolean) => void;
  setUseSerpapi: (v: boolean) => void;
  setSerpapiRatio: (v: number) => void;

  search: () => Promise<void>;
  downloadOne: (index: number, collection?: string, autoIngest?: boolean) => Promise<string | null>;
  downloadSelected: (collection?: string) => Promise<string | null>;
  pollTask: (taskId: string) => Promise<void>;
  startPolling: (taskId: string) => void;
  stopPolling: (taskId: string) => void;
  removeTask: (taskId: string) => void;

  toggleSelect: (index: number) => void;
  selectAll: () => void;
  clearSelection: () => void;

  checkHealth: () => Promise<void>;

  loadLibraries: () => Promise<void>;
  createLibrary: (
    name: string,
    description?: string,
    folderPath?: string | null,
    isTemporary?: boolean,
  ) => Promise<ScholarLibrary | null>;
  deleteLibrary: (libId: number) => Promise<void>;
  /** Clear a temporary library (same as delete; use for UI "Clear" label). */
  clearTemporaryLibrary: (libId: number) => Promise<void>;
  setActiveLibrary: (libId: number | null) => void;
  loadLibraryPapers: (libId: number) => Promise<void>;
  addResultsToLibrary: (indices: number[]) => Promise<{ added: number } | null>;
  removeFromLibrary: (libId: number, paperId: number) => Promise<void>;
  /** Batch download all papers in the active library; returns task_id if submitted. */
  downloadLibraryBatch: (collection?: string) => Promise<string | null>;
  /** Extract DOI + dedupe for active library; returns stats or null. */
  extractDoiAndDedup: () => Promise<{ extracted_count: number; removed_count: number } | null>;
  /** PDF rename and dedup for permanent library folder; returns stats or null. */
  pdfRenameDedup: () => Promise<{ renamed: number; removed: number; no_doi: number; synced_downloaded?: number } | null>;
  /** Download one library paper then open PDF when done. Sets openPdfAfterDownload on completion. */
  downloadLibraryPaperAndOpen: (paper: ScholarLibraryPaper) => Promise<string | null>;
  /** Set when a download-and-open task completes; page opens modal then clears. */
  openPdfAfterDownload: { libId: number | null; paperId: string; title: string } | null;
  clearOpenPdfAfterDownload: () => void;

  /** Internal: set by downloadLibraryPaperAndOpen, consumed by pollTask. */
  pendingDownloadAndOpen: PendingDownloadAndOpen | null;

  /** Clear the failure flag for a paper (called before retry). */
  clearDownloadAndOpenFailure: (paperId: string) => void;
}

export const useScholarStore = create<ScholarState>()((set, get) => ({
  query: '',
  source: 'google_scholar',
  yearStart: null,
  yearEnd: null,
  limit: 30,
  smartOptimize: false,
  useSerpapi: false,
  serpapiRatio: 50,
  results: [],
  isSearching: false,
  searchError: null,
  selectedIndices: [],
  downloadTasks: {},
  pollIntervals: {},
  scholarHealth: null,
  scholarHealthError: null,
  libraries: [],
  activeLibraryId: null,
  libraryPapers: [],
  libraryLoading: false,
  libraryError: null,

  openPdfAfterDownload: null,
  pendingDownloadAndOpen: null as PendingDownloadAndOpen | null,
  downloadAndOpenFailures: {} as Record<string, true>,

  selectedScholarLlmProvider: '',
  selectedScholarLlmModel: '',
  setScholarLlm: (provider, model) => set({ selectedScholarLlmProvider: provider, selectedScholarLlmModel: model }),

  scholarBrowserMode: 'default' as const,
  setScholarBrowserMode: (mode) => set({ scholarBrowserMode: mode }),

  includeAcademia: false,
  setIncludeAcademia: (v) => set({ includeAcademia: v }),

  setQuery: (q) => set({ query: q }),
  setSource: (s) => set({ source: s }),
  setYearStart: (y) => set({ yearStart: y }),
  setYearEnd: (y) => set({ yearEnd: y }),
  setLimit: (n) => {
    const v = Number(n);
    if (v <= 1) return set({ limit: Math.max(1, Math.floor(v)) });
    set({ limit: Math.round(v / 10) * 10 });
  },
  setSearchError: (err) => set({ searchError: err }),
  setSmartOptimize: (v) => set({ smartOptimize: v }),
  setUseSerpapi: (v) => set({ useSerpapi: v }),
  setSerpapiRatio: (v) => set({ serpapiRatio: Math.max(0, Math.min(100, v)) }),

  search: async () => {
    const { query, source, yearStart, yearEnd, limit, smartOptimize, useSerpapi, serpapiRatio } = get();
    if (!query.trim()) {
      set({ searchError: 'Query is required' });
      return;
    }
    set({ isSearching: true, searchError: null });
    try {
      const results = await searchScholar({
        query: query.trim(),
        source,
        limit,
        year_start: yearStart ?? undefined,
        year_end: yearEnd ?? undefined,
        optimize: smartOptimize,
        use_serpapi: source === 'google_scholar' || source === 'google' ? useSerpapi : undefined,
        serpapi_ratio: source === 'google_scholar' || source === 'google' ? serpapiRatio / 100 : undefined,
      });
      // Default: select all; user can deselect then click 加入子库 to add only selected (DOI dedup on server)
      set({ results, selectedIndices: results.map((_, i) => i) });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ results: [], searchError: message });
    } finally {
      set({ isSearching: false });
    }
  },

  downloadOne: async (index, collection, autoIngest = false) => {
    const { results, selectedScholarLlmProvider, selectedScholarLlmModel, scholarBrowserMode, includeAcademia } = get();
    const item = results[index];
    if (!item?.metadata) return null;
    const m = item.metadata;
    const showBrowser = scholarBrowserMode === 'default' ? undefined : scholarBrowserMode === 'headed';
    try {
      const res = await downloadPaper({
        title: m.title,
        doi: m.doi ?? undefined,
        pdf_url: m.pdf_url ?? undefined,
        annas_md5: m.annas_md5 ?? undefined,
        authors: m.authors?.length ? m.authors : undefined,
        year: m.year ?? undefined,
        collection,
        auto_ingest: autoIngest,
        llm_provider: selectedScholarLlmProvider || undefined,
        model_override: selectedScholarLlmModel || undefined,
        show_browser: showBrowser,
        include_academia: includeAcademia,
      });
      if (isSubmittedTask(res)) {
        set((s) => ({
          downloadTasks: { ...s.downloadTasks, [res.task_id]: { task_id: res.task_id, status: 'queued', payload: {} } },
        }));
        get().startPolling(res.task_id);
        return res.task_id;
      }
      return null;
    } catch {
      return null;
    }
  },

  downloadSelected: async (collection) => {
    const { results, selectedIndices, selectedScholarLlmProvider, selectedScholarLlmModel, scholarBrowserMode, includeAcademia } = get();
    if (selectedIndices.length === 0) return null;
    const papers = selectedIndices
      .map((i) => results[i]?.metadata)
      .filter(Boolean)
      .map((m) => ({
        title: m!.title,
        doi: m!.doi ?? undefined,
        pdf_url: m!.pdf_url ?? undefined,
        annas_md5: m!.annas_md5 ?? undefined,
        authors: m!.authors?.length ? m!.authors : undefined,
        year: m!.year ?? undefined,
      }));
    if (papers.length === 0) return null;
    const showBrowser = scholarBrowserMode === 'default' ? undefined : scholarBrowserMode === 'headed';
    try {
      const res = await batchDownloadPapers(papers, {
        collection,
        max_concurrent: 3,
        llm_provider: selectedScholarLlmProvider || undefined,
        model_override: selectedScholarLlmModel || undefined,
        show_browser: showBrowser,
        include_academia: includeAcademia,
      });
      set((s) => ({
        downloadTasks: {
          ...s.downloadTasks,
          [res.task_id]: { task_id: res.task_id, status: 'running', payload: { total: res.total, completed: 0, failed: 0 } },
        },
        selectedIndices: [],
      }));
      get().startPolling(res.task_id);
      return res.task_id;
    } catch {
      return null;
    }
  },

  pollTask: async (taskId) => {
    try {
      const status = await getDownloadTaskStatus(taskId);
      set((s) => ({
        downloadTasks: { ...s.downloadTasks, [taskId]: status },
      }));
      if (TERMINAL_STATUSES.has(status.status)) {
        get().stopPolling(taskId);
        const { pendingDownloadAndOpen, activeLibraryId } = get();
        if (activeLibraryId != null) {
          get().loadLibraryPapers(activeLibraryId);
        }
        if (pendingDownloadAndOpen && pendingDownloadAndOpen.taskId === taskId) {
          if (status.status === 'completed') {
            set({
              openPdfAfterDownload: {
                libId: pendingDownloadAndOpen.libId,
                paperId: pendingDownloadAndOpen.paperId,
                title: pendingDownloadAndOpen.title,
              },
              pendingDownloadAndOpen: null,
            });
          } else {
            // error or cancelled — restore button and mark as failed
            set((s) => ({
              pendingDownloadAndOpen: null,
              downloadAndOpenFailures: {
                ...s.downloadAndOpenFailures,
                [pendingDownloadAndOpen.paperId]: true,
              },
            }));
          }
        }
      }
    } catch {
      get().stopPolling(taskId);
    }
  },

  startPolling: (taskId) => {
    const { pollIntervals } = get();
    if (pollIntervals[taskId]) return;
    const id = setInterval(() => get().pollTask(taskId), POLL_INTERVAL_MS);
    set((s) => ({ pollIntervals: { ...s.pollIntervals, [taskId]: id } }));
  },

  stopPolling: (taskId) => {
    const { pollIntervals } = get();
    const id = pollIntervals[taskId];
    if (id) clearInterval(id);
    set((s) => {
      const next = { ...s.pollIntervals };
      delete next[taskId];
      return { pollIntervals: next };
    });
  },

  removeTask: (taskId) => {
    get().stopPolling(taskId);
    set((s) => {
      const next = { ...s.downloadTasks };
      delete next[taskId];
      return { downloadTasks: next };
    });
  },

  toggleSelect: (index) => {
    set((s) => {
      const has = s.selectedIndices.includes(index);
      const selectedIndices = has
        ? s.selectedIndices.filter((i) => i !== index)
        : [...s.selectedIndices, index].sort((a, b) => a - b);
      return { selectedIndices };
    });
  },

  selectAll: () => {
    const { results } = get();
    set({ selectedIndices: results.map((_, i) => i) });
  },

  clearSelection: () => set({ selectedIndices: [] }),

  checkHealth: async () => {
    try {
      const health = await getScholarHealth();
      set({ scholarHealth: health, scholarHealthError: null });
    } catch (e: unknown) {
      const status = (e as { response?: { status?: number } })?.response?.status;
      if (status === 404) {
        set({
          scholarHealth: { enabled: false, adapter_ready: false, download_dir: '' },
          scholarHealthError: '404',
        });
      } else {
        set({ scholarHealth: null, scholarHealthError: null });
      }
    }
  },

  // ─── Library operations ────────────────────────────────────────────────────

  loadLibraries: async () => {
    set({ libraryError: null });
    try {
      // Permanent libs come from the server; filter out any old server-side temp libs (negative IDs)
      const serverLibs = (await listLibraries()).filter((l) => l.id >= 0);
      // Temp libs live in localStorage
      const tempLibs = _loadTempLibs().map(_tempToScholarLibrary);
      set({ libraries: [...tempLibs, ...serverLibs] });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      // Even if server fails, show localStorage temp libs
      const tempLibs = _loadTempLibs().map(_tempToScholarLibrary);
      set({ libraryError: message, libraries: tempLibs });
    }
  },

  createLibrary: async (name, description, folderPath, isTemporary = false) => {
    set({ libraryError: null });

    if (isTemporary) {
      // Temp library: localStorage only
      const now = new Date().toISOString();
      const lib: TempLibData = {
        id: _nextSeq(LS_TEMP_LIB_SEQ_KEY),
        name: name.trim(),
        description: (description || '').trim(),
        created_at: now,
        updated_at: now,
        created_at_ts: Date.now(),
        papers: [],
      };
      const libs = _loadTempLibs();
      libs.push(lib);
      _saveTempLibs(libs);
      const scholLib = _tempToScholarLibrary(lib);
      set((s) => ({ libraries: [scholLib, ...s.libraries] }));
      return scholLib;
    }

    // Permanent library: persist to server DB
    try {
      const created = await createLibraryApi({
        name: name.trim(),
        description: description ?? '',
        folder_path: folderPath ?? undefined,
        is_temporary: false,
      });
      await get().loadLibraries();
      return {
        ...created,
        paper_count: 0,
        updated_at: created.created_at ?? '',
        is_temporary: false,
        folder_path: created.folder_path ?? null,
      } as ScholarLibrary;
    } catch (e: unknown) {
      const ax = e as { response?: { data?: { detail?: string } }; message?: string };
      const message = ax.response?.data?.detail ?? ax.message ?? String(e);
      set({ libraryError: message });
      return null;
    }
  },

  deleteLibrary: async (libId) => {
    set({ libraryError: null });

    if (libId < 0) {
      // Temp library: remove from localStorage
      const libs = _loadTempLibs().filter((l) => l.id !== libId);
      _saveTempLibs(libs);
      set((s) => ({
        libraries: s.libraries.filter((l) => l.id !== libId),
        activeLibraryId: s.activeLibraryId === libId ? null : s.activeLibraryId,
        libraryPapers: s.activeLibraryId === libId ? [] : s.libraryPapers,
      }));
      return;
    }

    // Permanent library: remove from server DB
    try {
      await deleteLibraryApi(libId);
      set((s) => ({
        libraries: s.libraries.filter((l) => l.id !== libId),
        activeLibraryId: s.activeLibraryId === libId ? null : s.activeLibraryId,
        libraryPapers: s.activeLibraryId === libId ? [] : s.libraryPapers,
      }));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ libraryError: message });
    }
  },

  clearTemporaryLibrary: async (libId) => {
    await get().deleteLibrary(libId);
  },

  setActiveLibrary: (libId) => {
    set({ activeLibraryId: libId, libraryPapers: [], libraryError: null });
    if (libId != null) get().loadLibraryPapers(libId);
  },

  loadLibraryPapers: async (libId) => {
    set({ libraryLoading: true, libraryError: null });

    if (libId < 0) {
      // Temp library: read from localStorage
      const libs = _loadTempLibs();
      const lib = libs.find((l) => l.id === libId);
      set((s) =>
        s.activeLibraryId === libId
          ? { libraryPapers: lib?.papers ?? [], libraryLoading: false }
          : { libraryLoading: false },
      );
      return;
    }

    // Permanent library: read from server
    try {
      const papers = await getLibraryPapers(libId);
      set((s) => (s.activeLibraryId === libId ? { libraryPapers: papers } : {}));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set((s) => (s.activeLibraryId === libId ? { libraryError: message, libraryPapers: [] } : {}));
    } finally {
      set({ libraryLoading: false });
    }
  },

  addResultsToLibrary: async (indices) => {
    const { results, activeLibraryId } = get();
    if (activeLibraryId == null) return null;
    const toAdd = indices.map((i) => results[i]).filter(Boolean) as ScholarSearchResult[];
    if (toAdd.length === 0) return null;

    if (activeLibraryId < 0) {
      // Temp library: write to localStorage
      const libs = _loadTempLibs();
      const lib = libs.find((l) => l.id === activeLibraryId);
      if (!lib) return null;
      const added = _appendToTempLib(lib, toAdd);
      _saveTempLibs(libs);
      // Refresh in-memory state
      set((s) => ({
        libraryPapers:
          s.activeLibraryId === activeLibraryId ? [...lib.papers] : s.libraryPapers,
        libraries: s.libraries.map((l) =>
          l.id === activeLibraryId ? { ...l, paper_count: lib.papers.length, updated_at: lib.updated_at } : l,
        ),
      }));
      return { added };
    }

    // Permanent library: write to server DB
    try {
      const res = await addPapersToLibraryApi(activeLibraryId, toAdd);
      await get().loadLibraryPapers(activeLibraryId);
      await get().loadLibraries();
      return { added: res.added };
    } catch {
      return null;
    }
  },

  removeFromLibrary: async (libId, paperId) => {
    if (libId < 0) {
      // Temp library: remove from localStorage
      const libs = _loadTempLibs();
      const lib = libs.find((l) => l.id === libId);
      if (lib) {
        lib.papers = lib.papers.filter((p) => p.id !== paperId);
        lib.updated_at = new Date().toISOString();
        _saveTempLibs(libs);
        set((s) => ({
          libraryPapers: s.libraryPapers.filter((p) => p.id !== paperId),
          libraries: s.libraries.map((l) =>
            l.id === libId ? { ...l, paper_count: lib.papers.length, updated_at: lib.updated_at } : l,
          ),
        }));
      }
      return;
    }

    // Permanent library: remove from server DB
    try {
      await removePaperFromLibraryApi(libId, paperId);
      set((s) => ({
        libraryPapers: s.libraryPapers.filter((p) => p.id !== paperId),
      }));
      await get().loadLibraries();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ libraryError: message });
    }
  },

  extractDoiAndDedup: async () => {
    const { activeLibraryId, libraryPapers } = get();
    if (activeLibraryId == null) return null;
    if (activeLibraryId >= 0) {
      try {
        const stats = await extractDoiAndDedupLibrary(activeLibraryId);
        await get().loadLibraryPapers(activeLibraryId);
        await get().loadLibraries();
        return stats;
      } catch {
        return null;
      }
    }
    // Client temp library: call papers endpoint and update localStorage
    try {
      const res = await extractDoiAndDedupPapers(libraryPapers);
      const libs = _loadTempLibs();
      const lib = libs.find((l) => l.id === activeLibraryId);
      if (lib) {
        lib.papers = res.papers;
        lib.updated_at = new Date().toISOString();
        _saveTempLibs(libs);
        set((s) => ({
          libraryPapers: s.activeLibraryId === activeLibraryId ? res.papers : s.libraryPapers,
          libraries: s.libraries.map((l) =>
            l.id === activeLibraryId ? { ...l, paper_count: res.papers.length, updated_at: lib.updated_at } : l,
          ),
        }));
      }
      return { extracted_count: res.extracted_count, removed_count: res.removed_count };
    } catch {
      return null;
    }
  },

  pdfRenameDedup: async () => {
    const { activeLibraryId } = get();
    if (activeLibraryId == null || activeLibraryId < 0) return null;
    try {
      const stats = await pdfRenameDedupApi(activeLibraryId);
      await get().loadLibraryPapers(activeLibraryId);
      await get().loadLibraries();
      return stats;
    } catch {
      return null;
    }
  },

  downloadLibraryPaperAndOpen: async (paper) => {
    const { activeLibraryId, selectedScholarLlmProvider, selectedScholarLlmModel, scholarBrowserMode, includeAcademia } = get();
    if (activeLibraryId == null || !paper.paper_id) return null;
    const libId = activeLibraryId >= 0 ? activeLibraryId : null;
    const showBrowser = scholarBrowserMode === 'default' ? undefined : scholarBrowserMode === 'headed';
    try {
      const res = await batchDownloadPapers(
        [
          {
            title: paper.title,
            doi: paper.doi ?? undefined,
            pdf_url: paper.pdf_url ?? undefined,
            annas_md5: paper.annas_md5 ?? undefined,
            authors: paper.authors?.length ? paper.authors : undefined,
            year: paper.year ?? undefined,
            library_paper_id: paper.id,
          },
        ],
        {
          library_id: libId ?? undefined,
          llm_provider: selectedScholarLlmProvider || undefined,
          model_override: selectedScholarLlmModel || undefined,
          show_browser: showBrowser,
          include_academia: includeAcademia,
        },
      );
      set((s) => ({
        downloadTasks: {
          ...s.downloadTasks,
          [res.task_id]: {
            task_id: res.task_id,
            status: 'running',
            payload: { total: 1, completed: 0, failed: 0 },
          },
        },
        pendingDownloadAndOpen: {
          taskId: res.task_id,
          libId,
          paperId: paper.paper_id,
          title: paper.title,
        },
      }));
      get().startPolling(res.task_id);
      return res.task_id;
    } catch {
      return null;
    }
  },

  clearOpenPdfAfterDownload: () => set({ openPdfAfterDownload: null }),

  clearDownloadAndOpenFailure: (paperId) =>
    set((s) => {
      const next = { ...s.downloadAndOpenFailures };
      delete next[paperId];
      return { downloadAndOpenFailures: next };
    }),

  downloadLibraryBatch: async (collection) => {
    const { libraryPapers: papers, activeLibraryId, selectedScholarLlmProvider, selectedScholarLlmModel, scholarBrowserMode, includeAcademia } = get();
    if (papers.length === 0) return null;
    const req = papers.map((p) => ({
      title: p.title,
      doi: p.doi ?? undefined,
      pdf_url: p.pdf_url ?? undefined,
      annas_md5: p.annas_md5 ?? undefined,
      authors: p.authors?.length ? p.authors : undefined,
      year: p.year ?? undefined,
      library_paper_id: p.id,
    }));
    const showBrowser = scholarBrowserMode === 'default' ? undefined : scholarBrowserMode === 'headed';
    try {
      const res = await batchDownloadPapers(req, {
        collection,
        max_concurrent: 3,
        library_id: activeLibraryId != null && activeLibraryId >= 0 ? activeLibraryId : undefined,
        llm_provider: selectedScholarLlmProvider || undefined,
        model_override: selectedScholarLlmModel || undefined,
        show_browser: showBrowser,
        include_academia: includeAcademia,
      });
      set((s) => ({
        downloadTasks: {
          ...s.downloadTasks,
          [res.task_id]: {
            task_id: res.task_id,
            status: 'running',
            payload: { total: res.total, completed: 0, failed: 0 },
          },
        },
      }));
      get().startPolling(res.task_id);
      return res.task_id;
    } catch {
      return null;
    }
  },
}));
