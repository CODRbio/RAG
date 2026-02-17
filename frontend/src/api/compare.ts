import client from './client';

export interface PaperSummary {
  paper_id: string;
  title: string;
  year?: number | null;
  abstract: string;
}

export interface CompareCandidate {
  paper_id: string;
  title: string;
  year?: number | null;
  abstract: string;
  citation_count: number;
  last_cited_turn_index: number;
  is_local_ready: boolean;
}

export interface CandidatesResponse {
  candidates: CompareCandidate[];
  total: number;
}

export interface PapersResponse {
  papers: PaperSummary[];
  total: number;
}

export interface CompareRequest {
  paper_ids: string[];
  aspects?: string[];
  llm_provider?: string;
  model_override?: string;
}

export interface CompareResponse {
  papers: PaperSummary[];
  comparison_matrix: Record<string, Record<string, string>>;
  narrative: string;
}

export async function comparePapers(body: CompareRequest): Promise<CompareResponse> {
  const res = await client.post<CompareResponse>('/compare', body);
  return res.data;
}

export async function listCompareCandidates(
  sessionId: string,
  opts?: { scope?: string; limit?: number; offset?: number }
): Promise<CandidatesResponse> {
  const params: Record<string, string | number> = { session_id: sessionId };
  if (opts?.scope != null) params.scope = opts.scope;
  if (opts?.limit != null) params.limit = opts.limit;
  if (opts?.offset != null) params.offset = opts.offset;
  const res = await client.get<CandidatesResponse>('/compare/candidates', { params });
  return res.data;
}

export async function listAvailablePapers(opts?: {
  limit?: number;
  offset?: number;
  q?: string;
}): Promise<PapersResponse> {
  const params: Record<string, string | number> = {};
  if (opts?.limit != null) params.limit = opts.limit;
  if (opts?.offset != null) params.offset = opts.offset;
  if (opts?.q != null && opts.q !== '') params.q = opts.q;
  const res = await client.get<PapersResponse>('/compare/papers', { params });
  return res.data;
}
