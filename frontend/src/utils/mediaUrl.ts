export function transformMarkdownMediaUrl(uri: string): string {
  if (/^(https?:)?\/\//i.test(uri) || uri.startsWith('data:') || uri.startsWith('blob:')) {
    return uri;
  }

  if (uri.startsWith('/media/') || uri.startsWith('/ga_images/')) {
    const apiBase = import.meta.env.VITE_API_BASE_URL || '';
    if (apiBase) return `${apiBase.replace(/\/api$/, '')}${uri}`;
  }

  return uri;
}
