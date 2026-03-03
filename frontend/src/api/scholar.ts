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

export type ScholarSource = 'google_scholar' | 'semantic' | 'ncbi' | 'annas_archive';

export async function searchScholar(params: {
  query: string;
  source?: ScholarSource;
  limit?: number;
  year_start?: number;
  year_end?: number;
}): Promise<ScholarSearchResult[]> {
  const res = await client.post<{ results: ScholarSearchResult[] }>(
    '/scholar/search',
    {
      query: params.query,
      source: params.source ?? 'google_scholar',
      limit: params.limit ?? 10,
      year_start: params.year_start,
      year_end: params.year_end,
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
}): Promise<
  | { status: string; task_id?: string; message?: string }
  | DownloadResult
> {
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
  options?: { collection?: string; max_concurrent?: number },
): Promise<{ status: string; task_id: string; total: number; message: string }> {
  const res = await client.post('/scholar/download/batch', {
    papers,
    collection: options?.collection,
    max_concurrent: options?.max_concurrent ?? 3,
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
