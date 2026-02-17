import client from './client';
import type { Canvas, Citation, OutlineSection, DraftBlock, Annotation } from '../types';

export async function createCanvas(data: {
  session_id?: string;
  topic?: string;
}): Promise<Canvas> {
  const res = await client.post<Canvas>('/canvas', data);
  return res.data;
}

export async function getCanvas(canvasId: string): Promise<Canvas> {
  const res = await client.get<Canvas>(`/canvas/${canvasId}`);
  return res.data;
}

/**
 * 更新 Canvas 基础字段（不包括 outline 和 drafts）。
 * 若需更新 outline，请使用 upsertOutline()；
 * 若需更新 drafts，请使用 upsertDraft()。
 */
export async function updateCanvas(
  canvasId: string,
  data: Partial<{
    session_id: string;
    topic: string;
    working_title: string;
    abstract: string;
    keywords: string[];
    stage: string;
    refined_markdown: string;
  }>
): Promise<Canvas> {
  const res = await client.patch<Canvas>(`/canvas/${canvasId}`, data);
  return res.data;
}

export async function deleteCanvas(canvasId: string): Promise<void> {
  await client.delete(`/canvas/${canvasId}`);
}

export interface ExportResponse {
  format: string;
  content: string;
  session_id: string;
  canvas_id: string;
}

export async function exportCanvas(
  canvasId: string,
  format: 'json' | 'markdown' = 'json'
): Promise<ExportResponse> {
  const res = await client.get<ExportResponse>(`/canvas/${canvasId}/export`, {
    params: { format },
  });
  return res.data;
}

export type CitationFormat = 'bibtex' | 'text' | 'both';

// 根据 format 参数返回不同结构
export type CitationsResult<T extends CitationFormat> = T extends 'bibtex'
  ? { format: 'bibtex'; content: string }
  : T extends 'text'
    ? { format: 'text'; content: string }
    : { format: 'both'; bibtex: string; reference_list: string; citations: Citation[] };

/**
 * 获取画布的引用列表。
 * @param format - bibtex: 返回 BibTeX 字符串; text: 返回文本参考列表; both: 返回两者及结构化数据
 */
export async function getCanvasCitations<T extends CitationFormat = 'both'>(
  canvasId: string,
  format?: T
): Promise<CitationsResult<T>> {
  const res = await client.get(`/canvas/${canvasId}/citations`, {
    params: { format: format ?? 'both' },
  });
  return res.data;
}

export async function createSnapshot(
  canvasId: string
): Promise<{ version_number: number }> {
  const res = await client.post(`/canvas/${canvasId}/snapshot`);
  return res.data;
}

export async function restoreSnapshot(
  canvasId: string,
  versionNumber: number
): Promise<void> {
  await client.post(`/canvas/${canvasId}/restore/${versionNumber}`);
}

export interface CanvasVersionItem {
  version_number: number;
  created_at: string;
}

export async function listCanvasSnapshots(
  canvasId: string,
  limit = 50
): Promise<CanvasVersionItem[]> {
  const res = await client.get<CanvasVersionItem[]>(`/canvas/${canvasId}/snapshots`, {
    params: { limit },
  });
  return res.data;
}

// ---- Outline & Drafts ----

/**
 * 批量更新/插入大纲章节。
 */
export async function upsertOutline(
  canvasId: string,
  sections: OutlineSection[]
): Promise<Canvas> {
  const res = await client.post<Canvas>(`/canvas/${canvasId}/outline`, { sections });
  return res.data;
}

/**
 * 更新/插入单个草稿块。
 */
export async function upsertDraft(
  canvasId: string,
  block: DraftBlock
): Promise<Canvas> {
  const res = await client.post<Canvas>(`/canvas/${canvasId}/drafts`, { block });
  return res.data;
}

// ---- Citation Management ----

export interface CitationFilterResponse {
  removed_count: number;
  remaining_keys: string[];
}

/**
 * 筛选引用池：保留或删除指定 cite_key。
 * keep_keys 和 remove_keys 二选一。
 */
export async function filterCitations(
  canvasId: string,
  options: { keep_keys?: string[]; remove_keys?: string[] }
): Promise<CitationFilterResponse> {
  const res = await client.post<CitationFilterResponse>(
    `/canvas/${canvasId}/citations/filter`,
    options
  );
  return res.data;
}

/**
 * 删除指定 cite_key 的引用。
 */
export async function deleteCitation(
  canvasId: string,
  citeKey: string
): Promise<{ ok: boolean; canvas_id: string; removed_cite_key: string }> {
  const res = await client.delete(`/canvas/${canvasId}/citations/${citeKey}`);
  return res.data;
}

// ---- AI Edit ----

export interface AIEditRequest {
  section_text: string;
  action: 'rewrite' | 'expand' | 'condense' | 'add_citations' | 'targeted_refine';
  context?: string;
  search_mode?: string;
  directive?: string;
  preserve_citations?: boolean;
}

export interface AIEditResponse {
  edited_text: string;
  citations_added: string[];
  citation_guard_triggered?: boolean;
  citation_guard_message?: string;
}

/**
 * AI 段落级编辑（重写/扩展/精简/添加引用）。
 */
export async function aiEditCanvas(
  canvasId: string,
  data: AIEditRequest
): Promise<AIEditResponse> {
  const res = await client.post<AIEditResponse>(
    `/canvas/${canvasId}/ai-edit`,
    data
  );
  return res.data;
}

export interface CanvasFullRefineRequest {
  content_md?: string;
  directives?: string[];
  save_snapshot_before?: boolean;
  locked_ranges?: Array<{ start: number; end: number; text: string }>;
}

export interface CanvasFullRefineResponse {
  edited_markdown: string;
  snapshot_version?: number | null;
  locked_applied?: number;
  locked_skipped?: number;
  lock_guard_triggered?: boolean;
  lock_guard_message?: string;
}

export async function refineCanvasFull(
  canvasId: string,
  data: CanvasFullRefineRequest
): Promise<CanvasFullRefineResponse> {
  const res = await client.post<CanvasFullRefineResponse>(
    `/canvas/${canvasId}/refine-full`,
    data,
    {
      // Full-document refine can take much longer than default 60s.
      timeout: 300000,
    }
  );
  return res.data;
}

// ---- Annotations ----

/**
 * 批量添加行内批注。
 */
export async function addAnnotations(
  canvasId: string,
  annotations: Annotation[]
): Promise<Canvas> {
  const res = await client.post<Canvas>(
    `/canvas/${canvasId}/annotations`,
    { annotations }
  );
  return res.data;
}

/**
 * 更新 Canvas 阶段跳过控制。
 */
export async function updateCanvasSkipSettings(
  canvasId: string,
  settings: { skip_draft_review?: boolean; skip_refine_review?: boolean }
): Promise<Canvas> {
  const res = await client.patch<Canvas>(`/canvas/${canvasId}`, settings);
  return res.data;
}
