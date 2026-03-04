import client from './client';

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
  const res = await client.post<{ results: ScholarSearchResult[] }>(
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
  );
  return res.data.results;
}

export async function downloadPaper(params: {
  title: string;
  doi?: string;
  pdf_url?: string;
  annas_md5?: string;
  authors?: string[];
  year?: number;
  collection?: string;
  auto_ingest?: boolean;
  llm_provider?: string | null;
  model_override?: string | null;
}): Promise<SubmittedTask | DownloadResult> {
  const res = await client.post('/scholar/download', params);
  return res.data;
}

export async function batchDownloadPapers(
  papers: Array<{
    title: string;
    doi?: string;
    pdf_url?: string;
    annas_md5?: string;
    authors?: string[];
    year?: number;
  }>,
  options?: {
    collection?: string;
    max_concurrent?: number;
    library_id?: number;
    llm_provider?: string | null;
    model_override?: string | null;
  },
): Promise<{ status: string; task_id: string; total: number; message: string }> {
  const res = await client.post('/scholar/download/batch', {
    papers,
    collection: options?.collection,
    max_concurrent: options?.max_concurrent ?? 3,
    library_id: options?.library_id,
    llm_provider: options?.llm_provider ?? undefined,
    model_override: options?.model_override ?? undefined,
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

/** PDF 查看 URL（供 PdfViewerModal 使用） */
export function getPdfViewUrl(paperId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || '/api';
  return `${base}/graph/pdf/${encodeURIComponent(paperId)}`;
}

// ─── Scholar sub-libraries ───────────────────────────────────────────────────

export async function listLibraries(): Promise<ScholarLibrary[]> {
  const res = await client.get<ScholarLibrary[]>('/scholar/libraries');
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
