import client from './client';

export interface DebugStatus {
  debug: boolean;
  log_dir: string;
}

export async function getDebugStatus(): Promise<DebugStatus> {
  const res = await client.get<DebugStatus>('/debug/status');
  return res.data;
}

export async function toggleDebug(enabled: boolean): Promise<{ ok: boolean; debug: boolean }> {
  const res = await client.post<{ ok: boolean; debug: boolean }>('/debug/toggle', { enabled });
  return res.data;
}
