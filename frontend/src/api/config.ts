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

/** No-op: native folder picker removed (headless/server). Use listDir for path selection or rely on server-managed paths. */
export async function pickFolder(): Promise<string | null> {
  return null;
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

export interface CacheClearResult {
  ok: boolean;
  crossref_cache_deleted?: number;
  crossref_cache_by_doi_deleted?: number;
  paper_metadata_deleted?: number;
  older_than_days?: number | string;
}

export async function clearCrossrefCache(olderThanDays = 0): Promise<CacheClearResult> {
  const res = await client.delete<CacheClearResult>('/admin/cache/crossref', {
    params: olderThanDays > 0 ? { older_than_days: olderThanDays } : {},
  });
  return res.data;
}

export async function clearPaperMetadataCache(): Promise<CacheClearResult> {
  const res = await client.delete<CacheClearResult>('/admin/cache/paper-metadata');
  return res.data;
}

export { DEFAULT_DATABASE_URL };
