export type AppTabId = 'chat' | 'ingest' | 'users' | 'graph' | 'compare' | 'scholar';

export const TAB_PATHS: Record<AppTabId, string> = {
  chat: '/chat',
  ingest: '/ingest',
  scholar: '/scholar',
  graph: '/workspace/graph',
  compare: '/analysis',
  users: '/users',
};

export function pathToActiveTab(pathname: string): AppTabId {
  if (pathname.startsWith('/ingest')) return 'ingest';
  if (pathname.startsWith('/scholar')) return 'scholar';
  if (pathname.startsWith('/papers/')) return 'scholar';
  if (pathname.startsWith('/workspace/graph')) return 'graph';
  if (pathname.startsWith('/graph')) return 'graph';
  if (pathname.startsWith('/analysis')) return 'compare';
  if (pathname.startsWith('/compare')) return 'compare';
  if (pathname.startsWith('/users')) return 'users';
  return 'chat';
}

export function buildPaperWorkspacePath(
  paperUid: string,
  opts?: {
    libraryId?: number | null;
    tab?: string | null;
  },
): string {
  const params = new URLSearchParams();
  if (opts?.libraryId != null) params.set('libraryId', String(opts.libraryId));
  if (opts?.tab) params.set('tab', opts.tab);
  const query = params.toString();
  return `/papers/${encodeURIComponent(paperUid)}${query ? `?${query}` : ''}`;
}
