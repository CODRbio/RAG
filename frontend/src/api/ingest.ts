import client from './client';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// ---- Collections ----

export interface CollectionInfo {
  name: string;
  count: number;
}

export async function listCollections(): Promise<CollectionInfo[]> {
  const res = await client.get<{ collections: CollectionInfo[] }>('/ingest/collections');
  return res.data.collections;
}

export async function createCollection(name: string, recreate = false): Promise<void> {
  await client.post('/ingest/collections', { name, recreate });
}

export async function deleteCollection(name: string): Promise<void> {
  await client.delete(`/ingest/collections/${encodeURIComponent(name)}`);
}

/** 获取集合的覆盖范围摘要（用于「查询与库是否匹配」判断） */
export interface CollectionScopeInfo {
  ok: boolean;
  name: string;
  scope_summary: string | null;
  updated_at: string | null;
}

export async function getCollectionScope(name: string): Promise<CollectionScopeInfo> {
  const res = await client.get<CollectionScopeInfo>(
    `/ingest/collections/${encodeURIComponent(name)}/scope`,
  );
  return res.data;
}

/** 编辑并保存集合的覆盖范围摘要 */
export async function updateCollectionScope(
  name: string,
  scopeSummary: string,
): Promise<CollectionScopeInfo> {
  const res = await client.put<CollectionScopeInfo>(
    `/ingest/collections/${encodeURIComponent(name)}/scope`,
    { scope_summary: scopeSummary },
  );
  return res.data;
}

/** 刷新集合的覆盖范围摘要（LLM 根据库名与可选样本重新生成） */
export async function refreshCollectionScope(name: string): Promise<{ ok: boolean; name: string; scope_summary: string }> {
  const res = await client.post<{ ok: boolean; name: string; scope_summary: string }>(
    `/ingest/collections/${encodeURIComponent(name)}/scope-refresh`,
    {},
  );
  return res.data;
}

// ---- Papers (文件级管理) ----

export interface PaperInfo {
  paper_id: string;
  filename: string;
  file_size: number;
  chunk_count: number;
  row_count: number;
  enrich_tables_enabled?: number;
  enrich_figures_enabled?: number;
  table_count?: number;
  figure_count?: number;
  table_success?: number;
  figure_success?: number;
  status: string;
  error_message: string;
  created_at: number;
  content_hash?: string;
}

export async function listPapers(collection: string): Promise<PaperInfo[]> {
  const res = await client.get<{ papers: PaperInfo[] }>(
    `/ingest/collections/${encodeURIComponent(collection)}/papers`,
  );
  return res.data.papers;
}

export async function deletePaper(
  collection: string,
  paperId: string,
): Promise<{ deleted_chunks: number }> {
  const res = await client.delete<{ deleted_chunks: number }>(
    `/ingest/collections/${encodeURIComponent(collection)}/papers/${encodeURIComponent(paperId)}`,
  );
  return res.data;
}

// ---- Upload ----

export interface UploadedFile {
  filename: string;
  path: string;
  size: number;
  content_hash?: string;
}

export async function uploadFiles(
  files: File[],
  collection: string,
): Promise<UploadedFile[]> {
  const fd = new FormData();
  for (const f of files) {
    fd.append('files', f);
  }
  fd.append('collection', collection);

  console.log('[ingest] uploadFiles called, files:', files.length, 'collection:', collection);

  // 使用独立 axios 实例发送 multipart，避免默认 Content-Type: application/json 干扰
  const token = localStorage.getItem('token');
  const res = await client.post<{ uploaded: UploadedFile[]; count: number }>(
    '/ingest/upload',
    fd,
    {
      // axios 1.x: 传 null 才能真正移除默认 header，让浏览器自动加 multipart boundary
      headers: {
        'Content-Type': null as unknown as string,
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      timeout: 300000,
    },
  );
  console.log('[ingest] uploadFiles response:', res.data);
  return res.data.uploaded;
}

// ---- Process (SSE) ----

export interface IngestProgressEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface IngestJobInfo {
  job_id: string;
  collection: string;
  status: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  total_chunks: number;
  total_upserted: number;
  current_file: string;
  current_stage: string;
  message: string;
  error_message: string;
  created_at: number;
  updated_at: number;
  finished_at?: number | null;
}

export interface EnrichmentOptions {
  enrich_tables: boolean;
  enrich_figures: boolean;
  llm_text_provider?: string | null;
  llm_text_model?: string | null;
  llm_text_concurrency?: number | null;
  llm_vision_provider?: string | null;
  llm_vision_model?: string | null;
  llm_vision_concurrency?: number | null;
}

export interface LLMProviderInfo {
  id: string;
  platform?: string;
  default_model: string;
  models: string[];
  supports_image?: boolean;
  registry_key?: string | null;
  label?: string;
}

export interface LLMProvidersResponse {
  default: string;
  providers: LLMProviderInfo[];
  parser_defaults?: {
    llm_text_provider?: string | null;
    llm_text_model?: string | null;
    llm_text_concurrency?: number | null;
    llm_vision_provider?: string | null;
    llm_vision_model?: string | null;
    llm_vision_concurrency?: number | null;
  };
}

export interface ProviderTemplate {
  name: string;
  label: string;
  default_base_url: string;
  supports_image: boolean;
  env_key_hint: string;
}

export interface RemoteModelInfo {
  id: string;
  owned_by?: string;
  supports_image?: boolean;
  extra?: Record<string, unknown>;
}

export interface FetchModelsResponse {
  provider: string;
  registry_key: string;
  models: RemoteModelInfo[];
  count: number;
}

export async function listLLMProviders(): Promise<LLMProvidersResponse> {
  const res = await client.get<LLMProvidersResponse>('/llm/providers');
  return res.data;
}

export interface LiveModelsResponse {
  platforms: Record<string, { models: string[]; count: number; error?: string }>;
}

export async function listAllLiveModels(noCache = false): Promise<LiveModelsResponse> {
  const url = noCache ? '/llm/models?no_cache=true' : '/llm/models';
  const res = await client.get<LiveModelsResponse>(url);
  return res.data;
}

export async function listProviderRegistry(): Promise<{ providers: ProviderTemplate[] }> {
  const res = await client.get<{ providers: ProviderTemplate[] }>('/llm/providers/registry');
  return res.data;
}

export interface UltraLiteProviderOption {
  id: string;
  label: string;
  default_model: string;
  platform: string;
}

export interface UltraLiteProvidersResponse {
  providers: UltraLiteProviderOption[];
  default: string | null;
}

export async function listUltraLiteProviders(noCache = false): Promise<UltraLiteProvidersResponse> {
  const url = noCache ? '/llm/ultra_lite_providers?no_cache=true' : '/llm/ultra_lite_providers';
  const res = await client.get<UltraLiteProvidersResponse>(url);
  return res.data;
}

export async function fetchProviderModels(
  providerName: string,
  options?: { apiKey?: string; baseUrl?: string; noCache?: boolean },
): Promise<FetchModelsResponse> {
  const params = new URLSearchParams();
  if (options?.apiKey) params.set('api_key', options.apiKey);
  if (options?.baseUrl) params.set('base_url', options.baseUrl);
  if (options?.noCache) params.set('no_cache', 'true');
  const qs = params.toString();
  const url = `/llm/providers/${encodeURIComponent(providerName)}/models${qs ? `?${qs}` : ''}`;
  const res = await client.get<FetchModelsResponse>(url);
  return res.data;
}

export async function listIngestJobs(limit = 20, status?: string): Promise<IngestJobInfo[]> {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (status) params.set('status', status);
  const res = await client.get<{ jobs: IngestJobInfo[] }>(`/ingest/jobs?${params.toString()}`);
  return res.data.jobs || [];
}

export async function getIngestJob(jobId: string): Promise<IngestJobInfo> {
  const res = await client.get<{ job: IngestJobInfo }>(`/ingest/jobs/${encodeURIComponent(jobId)}`);
  return res.data.job;
}

export async function cancelIngestJob(jobId: string): Promise<{ ok: boolean; status: string }> {
  const res = await client.post<{ ok: boolean; status: string }>(
    `/ingest/jobs/${encodeURIComponent(jobId)}/cancel`,
  );
  return res.data;
}

export async function startIngestJob(
  filePaths: string[],
  collection: string,
  enrichment: EnrichmentOptions,
  content_hashes?: Record<string, string>,
): Promise<{ job_id: string }> {
  const token = localStorage.getItem('token');
  const skipEnrichment = !enrichment.enrich_tables && !enrichment.enrich_figures;
  const response = await fetch(`${BASE_URL}/ingest/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      file_paths: filePaths,
      collection,
      skip_enrichment: skipEnrichment,
      enrich_tables: enrichment.enrich_tables,
      enrich_figures: enrichment.enrich_figures,
      llm_text_provider: enrichment.llm_text_provider ?? null,
      llm_text_model: enrichment.llm_text_model ?? null,
      llm_text_concurrency: enrichment.llm_text_concurrency ?? null,
      llm_vision_provider: enrichment.llm_vision_provider ?? null,
      llm_vision_model: enrichment.llm_vision_model ?? null,
      llm_vision_concurrency: enrichment.llm_vision_concurrency ?? null,
      content_hashes: content_hashes || {},
    }),
  });
  if (!response.ok) {
    const errBody = await response.text().catch(() => '');
    throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errBody}`);
  }
  const body = (await response.json()) as { job_id?: string };
  if (!body.job_id) throw new Error('No job_id returned from /ingest/process');
  return { job_id: body.job_id };
}

export async function* streamIngestJobEvents(
  jobId: string,
  signal?: AbortSignal,
  afterId = 0,
): AsyncGenerator<IngestProgressEvent> {
  const token = localStorage.getItem('token');
  const response = await fetch(
    `${BASE_URL}/ingest/jobs/${encodeURIComponent(jobId)}/events?after_id=${afterId}`,
    {
      method: 'GET',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      signal,
    },
  );
  if (!response.ok) {
    const errBody = await response.text().catch(() => '');
    throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errBody}`);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);
          yield { event: currentEvent, data };
        } catch {
          yield { event: currentEvent, data: { raw: dataStr } };
        }
      }
    }
  }
}

/**
 * 调用 /ingest/process，以 SSE 方式返回进度事件。
 * content_hashes: path -> SHA256 hex，用于 paper 元数据与去重。
 */
export async function* processFiles(
  filePaths: string[],
  collection: string,
  enrichment: EnrichmentOptions,
  signal?: AbortSignal,
  content_hashes?: Record<string, string>,
): AsyncGenerator<IngestProgressEvent> {
  const started = await startIngestJob(filePaths, collection, enrichment, content_hashes);
  yield { event: 'job_created', data: { job_id: started.job_id, collection } };
  for await (const evt of streamIngestJobEvents(started.job_id, signal, 0)) {
    yield evt;
  }
}
