import client from './client';

const DEFAULT_DATABASE_URL = 'sqlite:///data/rag.db';

export interface DatabaseConfig {
  url: string;
}

export async function getDatabaseConfig(): Promise<DatabaseConfig> {
  const res = await client.get<DatabaseConfig>('/config/database');
  return res.data;
}

export async function updateDatabaseConfig(url: string): Promise<{ url: string; message?: string }> {
  const res = await client.patch<{ url: string; message?: string }>('/config/database', { url });
  return res.data;
}

export async function pickFolder(): Promise<string | null> {
  try {
    const res = await client.get<{ path: string }>('/config/pick-folder');
    return res.data.path ?? null;
  } catch (e) {
    if ((e as { response?: { status?: number } })?.response?.status === 204) return null;
    throw e;
  }
}

export interface DirEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface DirListing {
  current: string;
  parent: string | null;
  home: string;
  entries: DirEntry[];
}

export async function listDir(path?: string | null): Promise<DirListing> {
  const params = path ? { path } : {};
  const res = await client.get<DirListing>('/config/list-dir', { params });
  return res.data;
}

export { DEFAULT_DATABASE_URL };
