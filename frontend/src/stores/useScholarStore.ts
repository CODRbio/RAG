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
} from '../api/scholar';

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES = new Set(['completed', 'error', 'cancelled']);

interface ScholarState {
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
  createLibrary: (name: string, description?: string) => Promise<ScholarLibrary | null>;
  deleteLibrary: (libId: number) => Promise<void>;
  setActiveLibrary: (libId: number | null) => void;
  loadLibraryPapers: (libId: number) => Promise<void>;
  addResultsToLibrary: (indices: number[]) => Promise<{ added: number } | null>;
  removeFromLibrary: (libId: number, paperId: number) => Promise<void>;
  /** Batch download all papers in the active library; returns task_id if submitted. */
  downloadLibraryBatch: (collection?: string) => Promise<string | null>;
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
      set({ results, selectedIndices: [] });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ results: [], searchError: message });
    } finally {
      set({ isSearching: false });
    }
  },

  downloadOne: async (index, collection, autoIngest = true) => {
    const { results } = get();
    const item = results[index];
    if (!item?.metadata) return null;
    const m = item.metadata;
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
    const { results, selectedIndices } = get();
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
    try {
      const res = await batchDownloadPapers(papers, { collection, max_concurrent: 3 });
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

  loadLibraries: async () => {
    set({ libraryError: null });
    try {
      const libraries = await listLibraries();
      set({ libraries });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ libraryError: message, libraries: [] });
    }
  },

  createLibrary: async (name, description) => {
    set({ libraryError: null });
    try {
      const created = await createLibraryApi({ name: name.trim(), description });
      await get().loadLibraries();
      return { ...created, paper_count: 0, updated_at: created.created_at ?? '' } as ScholarLibrary;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      set({ libraryError: message });
      return null;
    }
  },

  deleteLibrary: async (libId) => {
    set({ libraryError: null });
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

  setActiveLibrary: (libId) => {
    set({ activeLibraryId: libId, libraryPapers: [], libraryError: null });
    if (libId != null) get().loadLibraryPapers(libId);
  },

  loadLibraryPapers: async (libId) => {
    set({ libraryLoading: true, libraryError: null });
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

  downloadLibraryBatch: async (collection) => {
    const { libraryPapers: papers } = get();
    if (papers.length === 0) return null;
    const req = papers.map((p) => ({
      title: p.title,
      doi: p.doi ?? undefined,
      pdf_url: p.pdf_url ?? undefined,
      annas_md5: p.annas_md5 ?? undefined,
      authors: p.authors?.length ? p.authors : undefined,
      year: p.year ?? undefined,
    }));
    try {
      const res = await batchDownloadPapers(req, { collection, max_concurrent: 3 });
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
