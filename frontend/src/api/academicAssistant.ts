import client from './client';
import { streamSSEResumable } from './sse';
import type {
  AcademicAssistantTaskInfo,
  AcademicAssistantTaskStartResponse,
  AssistantScope,
  DiscoveryMode,
  PaperCompareResult,
  PaperLocator,
  PaperQaResult,
  PaperSummaryResult,
  ResourceAnnotation,
  ResourceAnnotationUpsert,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export async function summarizePaper(params: {
  locator: PaperLocator;
  scope?: AssistantScope;
  question?: string;
  llm_provider?: string;
  model_override?: string;
}): Promise<PaperSummaryResult> {
  const res = await client.post<PaperSummaryResult>('/academic-assistant/papers/summary', params, {
    timeout: 180000,
  });
  return res.data;
}

export async function askPaper(params: {
  locator: PaperLocator;
  question: string;
  scope?: AssistantScope;
  llm_provider?: string;
  model_override?: string;
}): Promise<PaperQaResult> {
  const res = await client.post<PaperQaResult>('/academic-assistant/papers/qa', params, {
    timeout: 180000,
  });
  return res.data;
}

export async function compareAssistantPapers(params: {
  paper_uids: string[];
  aspects?: string[];
  scope?: AssistantScope;
  llm_provider?: string;
  model_override?: string;
}): Promise<PaperCompareResult> {
  const res = await client.post<PaperCompareResult>('/academic-assistant/papers/compare', params, {
    timeout: 180000,
  });
  return res.data;
}

export async function startMediaAnalysis(params: {
  paper_uids: string[];
  scope?: AssistantScope;
  force_reparse?: boolean;
  upsert_vectors?: boolean;
  llm_text_provider?: string;
  llm_vision_provider?: string;
  llm_text_model?: string;
  llm_vision_model?: string;
}): Promise<AcademicAssistantTaskStartResponse> {
  const res = await client.post<AcademicAssistantTaskStartResponse>('/academic-assistant/media-analysis/start', params);
  return res.data;
}

export async function startDiscovery(params: {
  mode: DiscoveryMode;
  paper_uids?: string[];
  node_ids?: string[];
  scope?: AssistantScope;
  question?: string;
  limit?: number;
}): Promise<AcademicAssistantTaskStartResponse> {
  const { mode, ...body } = params;
  const res = await client.post<AcademicAssistantTaskStartResponse>(
    `/academic-assistant/discovery/${encodeURIComponent(mode)}/start`,
    body,
  );
  return res.data;
}

export async function getAcademicAssistantTask(taskId: string): Promise<AcademicAssistantTaskInfo> {
  const res = await client.get<AcademicAssistantTaskInfo>(`/academic-assistant/task/${encodeURIComponent(taskId)}`);
  return res.data;
}

export async function* streamAcademicAssistantTask(
  taskId: string,
  signal?: AbortSignal,
  afterId: string = '-',
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  const token = localStorage.getItem('token');
  for await (const { event, data } of streamSSEResumable({
    getUrl: (lastEventId) => {
      const aid = lastEventId || afterId;
      return `${BASE_URL}/academic-assistant/task/${encodeURIComponent(taskId)}/stream?after_id=${encodeURIComponent(aid || '-')}`;
    },
    getHeaders: () => {
      const headers: Record<string, string> = {};
      if (token) headers.Authorization = `Bearer ${token}`;
      return headers;
    },
    terminalEvents: ['done', 'error', 'cancelled', 'timeout', 'completed'],
    signal,
    maxRetries: 5,
    baseMs: 1000,
    maxMs: 30000,
  })) {
    yield { event, data: (data || {}) as Record<string, unknown> };
  }
}

export async function listAnnotations(params: {
  paper_uid?: string;
  resource_type?: string;
  resource_id?: string;
  target_kind?: string;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<{ items: ResourceAnnotation[] }> {
  const res = await client.get<{ items: ResourceAnnotation[] }>('/academic-assistant/annotations', {
    params,
  });
  return res.data;
}

export async function upsertAnnotation(body: ResourceAnnotationUpsert): Promise<ResourceAnnotation> {
  const res = await client.post<ResourceAnnotation>('/academic-assistant/annotations', body);
  return res.data;
}
