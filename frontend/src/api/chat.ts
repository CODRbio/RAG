import client, { streamChat } from './client';
import type {
  ChatRequest,
  ChatResponse,
  ChatSubmitResponse,
  SessionInfo,
  SessionListItem,
  IntentDetectRequest,
  IntentDetectResponse,
  ClarifyResponse,
  DeepResearchStartRequest,
  DeepResearchStartResponse,
  DeepResearchConfirmRequest,
  DeepResearchSubmitResponse,
  DeepResearchRestartPhaseRequest,
  DeepResearchRestartSectionRequest,
  DeepResearchJobInfo,
  DeepResearchJobEvent,
  GapSupplement,
  ResearchInsight,
  TaskQueueResponse,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export async function chat(data: ChatRequest): Promise<ChatResponse> {
  const res = await client.post<ChatResponse>('/chat', data);
  return res.data;
}

/**
 * 意图检测 API（简化版）：Chat vs Deep Research
 */
export async function detectIntent(data: IntentDetectRequest): Promise<IntentDetectResponse> {
  const res = await client.post<IntentDetectResponse>('/intent/detect', data);
  return res.data;
}

/**
 * Deep Research 澄清问题生成
 */
export async function clarifyForDeepResearch(data: {
  message: string;
  session_id?: string;
  search_mode?: string;
  llm_provider?: string;
  model_override?: string;
}): Promise<ClarifyResponse> {
  const res = await client.post<ClarifyResponse>('/deep-research/clarify', data);
  return res.data;
}

export function chatStream(data: ChatRequest, signal?: AbortSignal) {
  return streamChat('/chat/stream', data, signal);
}

/** 异步提交 Chat，返回 task_id；配合 streamChatByTaskId 使用 */
export async function submitChat(data: ChatRequest): Promise<ChatSubmitResponse> {
  const res = await client.post<ChatSubmitResponse>('/chat/submit', data);
  return res.data;
}

/** SSE 订阅任务流（GET /chat/stream/{taskId}） */
export async function* streamChatByTaskId(
  taskId: string,
  signal?: AbortSignal
): AsyncGenerator<{ event: string; data: unknown }> {
  const token = localStorage.getItem('token');
  const url = `${BASE_URL}/chat/stream/${encodeURIComponent(taskId)}`;
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal,
  });
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
    // SSE 事件以双换行分隔，按 \n\n 切分再解析，避免 data 被截断导致 JSON 解析失败
    const blocks = buffer.split(/\n\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      let eventType = 'message';
      const dataLines: string[] = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          dataLines.push(line.slice(6));
        }
      }
      const dataStr = dataLines.join('\n');
      if (dataStr === '') {
        yield { event: eventType, data: {} };
        continue;
      }
      try {
        const data = JSON.parse(dataStr) as unknown;
        yield { event: eventType, data };
      } catch {
        yield { event: eventType, data: { raw: dataStr } };
      }
    }
  }
  // 剩余 buffer 若含完整 event+data 也解析一次
  if (buffer.trim()) {
    let eventType = 'message';
    const dataLines: string[] = [];
    for (const line of buffer.split('\n')) {
      if (line.startsWith('event: ')) eventType = line.slice(7).trim();
      else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
    }
    const dataStr = dataLines.join('\n');
    if (dataStr) {
      try {
        const data = JSON.parse(dataStr) as unknown;
        yield { event: eventType, data };
      } catch {
        yield { event: eventType, data: { raw: dataStr } };
      }
    }
  }
}

/** 排队区快照 */
export async function getTaskQueue(): Promise<TaskQueueResponse> {
  const res = await client.get<TaskQueueResponse>('/tasks/queue');
  return res.data;
}

/** 取消任务（排队或运行中） */
export async function cancelTask(taskId: string): Promise<{ success: boolean; message: string }> {
  const res = await client.post<{ success: boolean; message: string }>(
    `/tasks/${encodeURIComponent(taskId)}/cancel`
  );
  return res.data;
}

export async function deepResearchStart(data: DeepResearchStartRequest): Promise<DeepResearchStartResponse> {
  const res = await client.post<DeepResearchStartResponse>(
    '/deep-research/start',
    data,
    {
      // Phase-1 deep research may run hybrid retrieval + planning,
      // which can exceed the global 60s default timeout.
      timeout: 300000,
    }
  );
  return res.data;
}

export function deepResearchConfirmStream(data: DeepResearchConfirmRequest, signal?: AbortSignal) {
  return streamChat('/deep-research/confirm', data, signal);
}

export async function deepResearchSubmit(data: DeepResearchConfirmRequest): Promise<DeepResearchSubmitResponse> {
  const res = await client.post<DeepResearchSubmitResponse>('/deep-research/submit', data, { timeout: 30000 });
  return res.data;
}

export async function getDeepResearchJob(jobId: string): Promise<DeepResearchJobInfo> {
  const res = await client.get<DeepResearchJobInfo>(`/deep-research/jobs/${encodeURIComponent(jobId)}`);
  return res.data;
}

/**
 * 列出 Deep Research 后台任务（用于恢复视图）
 */
export async function listDeepResearchJobs(
  limit: number = 20,
  status?: string,
): Promise<DeepResearchJobInfo[]> {
  const params: Record<string, string | number> = { limit };
  if (status) params.status = status;
  const res = await client.get<DeepResearchJobInfo[]>('/deep-research/jobs', { params });
  return res.data || [];
}

export async function listDeepResearchJobEvents(
  jobId: string,
  afterId: number,
  limit: number = 200,
): Promise<DeepResearchJobEvent[]> {
  const res = await client.get<{ job_id: string; events: DeepResearchJobEvent[] }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/events`,
    { params: { after_id: afterId, limit } },
  );
  return res.data.events || [];
}

/**
 * SSE stream for Deep Research job progress.
 *
 * Replaces the setInterval polling of /events + /jobs/{id}.
 * Yields all job progress events in real-time. The backend also emits
 * periodic `heartbeat` events (with job status) and a final `job_status`
 * event when the job reaches a terminal state.
 */
export async function* streamDeepResearchEvents(
  jobId: string,
  signal?: AbortSignal,
  afterId = 0,
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  const token = localStorage.getItem('token');
  const url = `${BASE_URL}/deep-research/jobs/${encodeURIComponent(jobId)}/stream?after_id=${afterId}`;
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal,
  });
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
          const data = JSON.parse(dataStr) as Record<string, unknown>;
          yield { event: currentEvent, data };
        } catch {
          yield { event: currentEvent, data: { raw: dataStr } };
        }
      }
    }
  }
}

export async function cancelDeepResearchJob(jobId: string): Promise<{ ok: boolean; job_id: string; status: string }> {
  const res = await client.post<{ ok: boolean; job_id: string; status: string }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/cancel`,
  );
  return res.data;
}

/** 删除已终态的后台调研任务，服务端会清空该任务及关联数据（events、reviews 等）。 */
export async function deleteDeepResearchJob(jobId: string): Promise<{ ok: boolean; job_id: string; deleted: boolean }> {
  const res = await client.delete<{ ok: boolean; job_id: string; deleted: boolean }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}`,
  );
  return res.data;
}

export async function restartDeepResearchPhase(
  jobId: string,
  data: DeepResearchRestartPhaseRequest,
): Promise<DeepResearchSubmitResponse> {
  const res = await client.post<DeepResearchSubmitResponse>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/restart-phase`,
    data,
    { timeout: 30000 },
  );
  return res.data;
}

export async function restartDeepResearchSection(
  jobId: string,
  data: DeepResearchRestartSectionRequest,
): Promise<DeepResearchSubmitResponse> {
  const res = await client.post<DeepResearchSubmitResponse>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/restart-section`,
    data,
    { timeout: 30000 },
  );
  return res.data;
}

export async function optimizeSectionEvidence(
  jobId: string,
  data: {
    section_title: string;
    web_providers?: string[];
    web_source_configs?: Record<string, { topK: number; threshold: number }>;
    use_content_fetcher?: 'auto' | 'force' | 'off';
  },
): Promise<{ section_title: string; new_chunks: number; sources_used: string[] }> {
  const res = await client.post<{ section_title: string; new_chunks: number; sources_used: string[] }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/optimize-evidence`,
    data,
    { timeout: 120000 },
  );
  return res.data;
}

export async function submitSectionReview(
  jobId: string,
  data: { section_id: string; action: 'approve' | 'revise'; feedback?: string },
): Promise<{ ok: boolean; job_id: string; section_id: string; action: string }> {
  const res = await client.post<{ ok: boolean; job_id: string; section_id: string; action: string }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/review`,
    data,
  );
  return res.data;
}

export async function listSectionReviews(
  jobId: string,
): Promise<Array<{ job_id: string; section_id: string; action: string; feedback?: string; created_at?: number }>> {
  const res = await client.get<{
    job_id: string;
    reviews: Array<{ job_id: string; section_id: string; action: string; feedback?: string; created_at?: number }>;
  }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/reviews`,
  );
  return res.data.reviews || [];
}

// ── Gap Supplement APIs ──

export async function submitGapSupplement(
  jobId: string,
  data: {
    section_id: string;
    gap_text: string;
    supplement_type: 'material' | 'direct_info';
    content: Record<string, unknown>;
  },
): Promise<{ ok: boolean; id: number; job_id: string; section_id: string; status: string }> {
  const res = await client.post<{ ok: boolean; id: number; job_id: string; section_id: string; status: string }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/gap-supplement`,
    data,
  );
  return res.data;
}

export async function listGapSupplements(
  jobId: string,
  sectionId?: string,
): Promise<GapSupplement[]> {
  const params: Record<string, string> = {};
  if (sectionId) params.section_id = sectionId;
  const res = await client.get<{ job_id: string; supplements: GapSupplement[] }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/gap-supplements`,
    { params },
  );
  return res.data.supplements || [];
}

// ── Research Insights APIs ──

export async function listInsights(
  jobId: string,
  insightType?: string,
  status?: string,
): Promise<ResearchInsight[]> {
  const params: Record<string, string> = {};
  if (insightType) params.insight_type = insightType;
  if (status) params.status = status;
  const res = await client.get<{ job_id: string; insights: ResearchInsight[] }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/insights`,
    { params },
  );
  return res.data.insights || [];
}

export async function updateInsightStatus(
  jobId: string,
  insightId: number,
  status: 'open' | 'addressed' | 'deferred',
): Promise<{ ok: boolean }> {
  const res = await client.post<{ ok: boolean }>(
    `/deep-research/jobs/${encodeURIComponent(jobId)}/insights/${insightId}/status`,
    { status },
  );
  return res.data;
}

export async function extractDeepResearchContextFiles(files: File[]): Promise<Array<{ name: string; content: string }>> {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  const res = await client.post<{ documents: Array<{ name: string; content: string }> }>(
    '/deep-research/context-files',
    fd,
    {
      headers: {
        'Content-Type': null as unknown as string,
      },
      timeout: 120000,
    },
  );
  return res.data.documents || [];
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  const res = await client.get<SessionInfo>(`/sessions/${sessionId}`);
  return res.data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await client.delete(`/sessions/${sessionId}`);
}

export async function listSessions(limit: number = 100): Promise<SessionListItem[]> {
  const res = await client.get<SessionListItem[]>('/sessions', { params: { limit } });
  return res.data;
}
