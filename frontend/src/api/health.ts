import client from './client';

export async function checkHealth(): Promise<{ status: string }> {
  const res = await client.get<{ status: string }>('/health');
  return res.data;
}

export async function getStorageStats(): Promise<Record<string, unknown>> {
  const res = await client.get('/storage/stats');
  return res.data;
}
