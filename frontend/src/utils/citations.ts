import type { ChatCitation, CitationAnchor, Source } from '../types';

function normalizeAnchors(
  anchors?: CitationAnchor[] | null,
  fallback?: Pick<ChatCitation, 'chunk_id' | 'page_num' | 'bbox'>,
): CitationAnchor[] {
  const seen = new Set<string>();
  const out: CitationAnchor[] = [];

  for (const anchor of anchors || []) {
    const chunkId = String(anchor?.chunk_id || '').trim();
    if (!chunkId || seen.has(chunkId)) continue;
    seen.add(chunkId);
    out.push({
      chunk_id: chunkId,
      page_num: anchor?.page_num ?? null,
      bbox: Array.isArray(anchor?.bbox) ? anchor.bbox : null,
      snippet: typeof anchor?.snippet === 'string' && anchor.snippet.trim() ? anchor.snippet : null,
    });
  }

  const fallbackChunkId = String(fallback?.chunk_id || '').trim();
  if (out.length === 0 && fallbackChunkId) {
    out.push({
      chunk_id: fallbackChunkId,
      page_num: fallback?.page_num ?? null,
      bbox: Array.isArray(fallback?.bbox) ? fallback.bbox : null,
      snippet: null,
    });
  }

  return out;
}

export function inferSourceType(cite: Pick<ChatCitation, 'provider' | 'doc_id' | 'url' | 'pdf_url'>): 'local' | 'web' {
  if (cite.provider === 'local') return 'local';
  // doc_id takes priority over url: a local doc may carry its original web url
  if (cite.doc_id) return 'local';
  if (cite.url) return 'web';
  if (cite.pdf_url) return 'web';
  return 'web';
}

export function chatCitationToSource(cite: ChatCitation, id: string | number): Source {
  const anchors = normalizeAnchors(cite.anchors, {
    chunk_id: cite.chunk_id,
    page_num: cite.page_num,
    bbox: cite.bbox,
  });
  const primary = anchors[0];

  return {
    id,
    cite_key: cite.cite_key,
    title: cite.title || cite.cite_key,
    authors: cite.authors || [],
    year: cite.year,
    doc_id: cite.doc_id,
    url: cite.url,
    pdf_url: cite.pdf_url,
    doi: cite.doi,
    bbox: primary?.bbox ?? (Array.isArray(cite.bbox) ? cite.bbox : undefined),
    page_num: primary?.page_num ?? cite.page_num,
    chunk_id: primary?.chunk_id ?? cite.chunk_id ?? null,
    anchors,
    type: inferSourceType(cite),
    provider: cite.provider || inferSourceType(cite),
  };
}
