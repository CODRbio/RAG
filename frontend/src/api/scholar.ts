import client from './client';
import type { ScholarDownloadStrategyId } from '../types';

export interface ScholarResultMetadata {
  source: string;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  pdf_url?: string | null;
  url?: string | null;
  downloadable?: boolean;
  annas_md5?: string;
  venue?: string | null;
  normalized_journal_name?: string | null;
  impact_factor?: number | null;
  jif_quartile?: string | null;
  jif_5year?: number | null;
  journal_name_matched?: string | null;
}

export interface ScholarSearchResult {
  content: string;
  score: number;
  metadata: ScholarResultMetadata;
}

export interface DownloadResult {
  success: boolean;
  paper_id: string | null;
  filepath: string | null;
  message: string;
}

export interface SubmittedTask {
  status: 'submitted';
  task_id: string;
  message: string;
}

export function isSubmittedTask(
  r: SubmittedTask | DownloadResult,
): r is SubmittedTask {
  return (r as SubmittedTask).task_id !== undefined;
}

export interface DownloadTaskStatus {
  task_id: string;
  status: string;
  error_message?: string | null;
  payload: Record<string, unknown>;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface ScholarHealth {
  enabled: boolean;
  adapter_ready: boolean;
  download_dir: string;
  default_strategy_order?: ScholarDownloadStrategyId[];
}

export interface HeadedBrowserWindowState {
  available: boolean;
  running: boolean;
  mode: 'parked' | 'visible' | 'unavailable' | string;
  cdp_url?: string | null;
  bounds?: Record<string, number> | null;
}

/** Scholar sub-library (named candidate list for download). */
export interface ScholarLibrary {
  id: number;
  name: string;
  description: string;
  paper_count: number;
  created_at: string;
  updated_at: string;
  is_temporary?: boolean;
  folder_path?: string | null;
}

/** A paper saved in a scholar library (literature catalog). */
export interface ScholarLibraryPaper {
  id: number;
  library_id: number;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  pdf_url: string | null;
  url: string | null;
  source: string;
  score: number;
  annas_md5: string | null;
  added_at: string;
  /** ISO timestamp when PDF was downloaded to this library; null if not yet downloaded. */
  downloaded_at?: string | null;
  /** DOI-based filename stem for opening PDF from library folder (permanent lib only). */
  paper_id?: string | null;
  venue?: string | null;
  normalized_journal_name?: string | null;
  impact_factor?: number | null;
  jif_quartile?: string | null;
  jif_5year?: number | null;
  /** Derived: true when downloaded_at is set. */
  is_downloaded?: boolean;
}

export type ScholarSource = 'google_scholar' | 'google' | 'semantic' | 'semantic_relevance' | 'semantic_bulk' | 'ncbi' | 'annas_archive';

export async function searchScholar(params: {
  query: string;
  source?: ScholarSource;
  limit?: number;
  year_start?: number;
  year_end?: number;
  optimize?: boolean;
  use_serpapi?: boolean;
  serpapi_ratio?: number;
}): Promise<ScholarSearchResult[]> {
  const res = await client.post<{ results?: ScholarSearchResult[] }>(
    '/scholar/search',
    {
      query: params.query,
      source: params.source ?? 'google_scholar',
      limit: params.limit ?? 30,
      year_start: params.year_start,
      year_end: params.year_end,
      optimize: params.optimize ?? false,
      use_serpapi: params.use_serpapi,
      serpapi_ratio: params.serpapi_ratio,
    },
    { timeout: 300000 },
  );
  return Array.isArray(res?.data?.results) ? res.data.results : [];
}

export async function downloadPaper(params: {
  title: string;
  doi?: string;
  pdf_url?: string;
  url?: string;
  annas_md5?: string;
  authors?: string[];
  year?: number;
  collection?: string;
  auto_ingest?: boolean;
  library_paper_id?: number;
  library_id?: number;
  llm_provider?: string | null;
  model_override?: string | null;
  assist_llm_enabled?: boolean;
  show_browser?: boolean | null;
  include_academia?: boolean;
  strategy_order?: ScholarDownloadStrategyId[];
}): Promise<SubmittedTask | DownloadResult> {
  const res = await client.post('/scholar/download', params);
  return res.data;
}

export async function batchDownloadPapers(
  papers: Array<{
    title: string;
    doi?: string;
    pdf_url?: string;
    url?: string;
    annas_md5?: string;
    authors?: string[];
    year?: number;
    library_paper_id?: number;
  }>,
  options?: {
    collection?: string;
    max_concurrent?: number;
    library_id?: number;
    llm_provider?: string | null;
    model_override?: string | null;
    assist_llm_enabled?: boolean;
    show_browser?: boolean | null;
    include_academia?: boolean;
    strategy_order?: ScholarDownloadStrategyId[];
  },
): Promise<{ status: string; task_id: string; total: number; message: string }> {
  const res = await client.post('/scholar/download/batch', {
    papers,
    collection: options?.collection,
    max_concurrent: options?.max_concurrent ?? 3,
    library_id: options?.library_id,
    llm_provider: options?.llm_provider ?? undefined,
    model_override: options?.model_override ?? undefined,
    assist_llm_enabled: options?.assist_llm_enabled,
    show_browser: options?.show_browser ?? undefined,
    include_academia: options?.include_academia ?? false,
    strategy_order: options?.strategy_order,
  });
  return res.data;
}

export async function getDownloadTaskStatus(taskId: string): Promise<DownloadTaskStatus> {
  const res = await client.get<DownloadTaskStatus>(`/scholar/task/${encodeURIComponent(taskId)}`);
  return res.data;
}

export async function getScholarHealth(): Promise<ScholarHealth> {
  const res = await client.get<ScholarHealth>('/scholar/health');
  return res.data;
}

export async function getHeadedBrowserWindowState(): Promise<HeadedBrowserWindowState> {
  const res = await client.get<HeadedBrowserWindowState>('/scholar/browser/headed');
  return res.data;
}

export async function showHeadedBrowserWindow(): Promise<HeadedBrowserWindowState> {
  const res = await client.post<HeadedBrowserWindowState>('/scholar/browser/headed/show');
  return res.data;
}

export async function parkHeadedBrowserWindow(): Promise<HeadedBrowserWindowState> {
  const res = await client.post<HeadedBrowserWindowState>('/scholar/browser/headed/park');
  return res.data;
}

/** PDF 查看 URL（供 PdfViewerModal 使用，全局 raw_papers） */
export function getPdfViewUrl(paperId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || '/api';
  return `${base}/graph/pdf/${encodeURIComponent(paperId)}`;
}

/** PDF 查看 URL（文献库本地 PDF，永久库 pdfs 目录） */
export function getLibraryPdfViewUrl(libId: number, paperId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || '/api';
  return `${base}/scholar/libraries/${libId}/pdf/${encodeURIComponent(paperId)}`;
}

/**
 * Fetch a PDF URL with auth (Bearer) and return a blob URL for use in <Document> / iframe.
 * Caller must revoke the returned URL with URL.revokeObjectURL when done (e.g. when modal closes).
 * Accepts full URL (from getLibraryPdfViewUrl); normalizes to path for axios baseURL.
 */
export async function fetchPdfAsBlobUrl(url: string): Promise<string> {
  let path = url;
  if (url.startsWith('http')) path = new URL(url).pathname;
  if (path.startsWith('/api')) path = path.slice(4) || '/';
  const res = await client.get<Blob>(path, { responseType: 'blob' });
  return URL.createObjectURL(res.data);
}

// ─── Scholar sub-libraries ───────────────────────────────────────────────────

export async function listLibraries(): Promise<ScholarLibrary[]> {
  const res = await client.get<ScholarLibrary[]>('/scholar/libraries');
  return res.data;
}

export async function getLibraryByCollection(collection: string, autoCreate = false): Promise<{
  collection: string;
  bound: boolean;
  library: ScholarLibrary | null;
}> {
  const res = await client.get<{
    collection: string;
    bound: boolean;
    library: ScholarLibrary | null;
  }>(`/scholar/libraries/by-collection/${encodeURIComponent(collection)}`, {
    params: { auto_create: autoCreate },
  });
  return res.data;
}

export async function createLibrary(params: {
  name: string;
  description?: string;
  folder_path?: string | null;
  is_temporary?: boolean;
}): Promise<ScholarLibrary & { id: number }> {
  const res = await client.post<ScholarLibrary & { id: number }>('/scholar/libraries', {
    name: params.name,
    description: params.description ?? '',
    folder_path: params.folder_path ?? undefined,
    is_temporary: params.is_temporary ?? false,
  });
  return res.data;
}

export async function deleteLibrary(libId: number): Promise<void> {
  await client.delete(`/scholar/libraries/${libId}`);
}

export async function getLibraryPapers(libId: number): Promise<ScholarLibraryPaper[]> {
  const res = await client.get<ScholarLibraryPaper[]>(`/scholar/libraries/${libId}/papers`);
  return res.data;
}

export async function addPapersToLibrary(
  libId: number,
  papers: ScholarSearchResult[],
): Promise<{ added: number; total_requested: number }> {
  const res = await client.post<{ added: number; total_requested: number }>(
    `/scholar/libraries/${libId}/papers`,
    { papers },
  );
  return res.data;
}

export async function removePaperFromLibrary(
  libId: number,
  paperId: number,
): Promise<void> {
  await client.delete(`/scholar/libraries/${libId}/papers/${paperId}`);
}

/** Delete the downloaded PDF file for a library paper and clear downloaded_at. Paper remains in library so user can re-download. */
export async function deleteLibraryPaperPdf(
  libId: number,
  recordId: number,
): Promise<{ ok: boolean; paper_id: string }> {
  const res = await client.delete<{ ok: boolean; paper_id: string }>(
    `/scholar/libraries/${libId}/papers/${recordId}/pdf`,
  );
  return res.data;
}

export async function uploadLibraryPaperPdf(
  libId: number,
  recordId: number,
  file: File,
): Promise<{ success: boolean; paper_id: string; filename: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await client.post<{ success: boolean; paper_id: string; filename: string }>(
    `/scholar/libraries/${libId}/papers/${recordId}/upload-pdf`,
    form,
    {
      headers: { 'Content-Type': null as unknown as string },
      timeout: 120000,
    },
  );
  return res.data;
}

export interface ExtractDoiDedupResult {
  extracted_count: number;
  removed_count: number;
}

export async function extractDoiAndDedupLibrary(
  libId: number,
): Promise<ExtractDoiDedupResult> {
  const res = await client.post<ExtractDoiDedupResult>(
    `/scholar/libraries/${libId}/extract-doi-dedup`,
  );
  return res.data;
}

export async function extractDoiAndDedupPapers(
  papers: ScholarLibraryPaper[],
): Promise<{ papers: ScholarLibraryPaper[] } & ExtractDoiDedupResult> {
  const res = await client.post<{ papers: ScholarLibraryPaper[] } & ExtractDoiDedupResult>(
    '/scholar/papers/extract-doi-dedup',
    { papers },
  );
  return res.data;
}

export interface PdfRenameDedupResult {
  renamed: number;
  removed: number;
  no_doi: number;
  /** Count of library papers marked as downloaded (badge sync). */
  synced_downloaded?: number;
}

export async function pdfRenameDedup(
  libId: number,
): Promise<PdfRenameDedupResult> {
  const res = await client.post<PdfRenameDedupResult>(
    `/scholar/libraries/${libId}/pdf-rename-dedup`,
  );
  return res.data;
}

export interface LibraryImportPdfSummary {
  total_files: number;
  imported: number;
  linked_existing: number;
  renamed: number;
  skipped_duplicates: number;
  invalid_pdf: number;
  no_doi: number;
  errors: string[];
}

export async function importLibraryPdfs(
  libId: number,
  files: File[],
): Promise<LibraryImportPdfSummary> {
  const form = new FormData();
  for (const file of files) {
    form.append('files', file);
  }
  const res = await client.post<LibraryImportPdfSummary>(
    `/scholar/libraries/${libId}/import-pdfs`,
    form,
    {
      headers: { 'Content-Type': null as unknown as string },
      timeout: 600000,
    },
  );
  return res.data;
}

export interface ScholarLibraryIngestResult {
  ok: boolean;
  job_id: string;
  collection: string;
  library_id: number;
  total_library_papers: number;
  pdf_ready_count: number;
  missing_pdf_count: number;
  attempted_downloads: number;
  downloaded_now: number;
  failed_downloads: number;
}

export async function ingestScholarLibrary(
  libId: number,
  payload: {
    collection: string;
    skip_duplicate_doi?: boolean;
    skip_unchanged?: boolean;
    auto_download_missing?: boolean;
    max_auto_download?: number;
  },
): Promise<ScholarLibraryIngestResult> {
  const res = await client.post<ScholarLibraryIngestResult>(
    `/scholar/libraries/${libId}/ingest`,
    payload,
  );
  return res.data;
}
