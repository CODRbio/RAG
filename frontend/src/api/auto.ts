import client from './client';

export interface AutoCompleteRequest {
  topic: string;
  session_id?: string;
  canvas_id?: string;
  search_mode?: 'local' | 'web' | 'hybrid';
  max_sections?: number; // 2-6, default 4
}

export interface AutoCompleteResponse {
  session_id: string;
  canvas_id: string;
  markdown: string;
  outline: string[];
  citations: string[];
  total_time_ms: number;
}

/**
 * 自动完成综述：根据主题检索 -> 生成大纲 -> 逐章写作 -> 返回完整 Markdown。
 */
export async function autoComplete(data: AutoCompleteRequest): Promise<AutoCompleteResponse> {
  const res = await client.post<AutoCompleteResponse>('/auto-complete', data);
  return res.data;
}
