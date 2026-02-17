import client from './client';

export interface GraphStats {
  available: boolean;
  total_nodes: number;
  total_edges: number;
  entity_count: number;
  chunk_count: number;
  entity_types: Record<string, number>;
}

export interface EntityItem {
  name: string;
  type: string;
  mention_count: number;
}

export interface GraphNode {
  id: string;
  type: string;
  paper_id?: string;
  is_center?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

export interface NeighborGraph {
  center: string;
  depth: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ChunkDetail {
  collection: string;
  chunk_id: string;
  paper_id: string;
  content: string;
  section_path?: string;
  page?: number | null;
  content_type?: string;
  chunk_type?: string;
  related_entities?: string[];
}

export async function getGraphStats(): Promise<GraphStats> {
  const res = await client.get<GraphStats>('/graph/stats');
  return res.data;
}

export async function getEntities(params?: {
  entity_type?: string;
  limit?: number;
  offset?: number;
  q?: string;
}): Promise<{ total: number; entities: EntityItem[] }> {
  const res = await client.get('/graph/entities', { params });
  return res.data;
}

export async function getNeighbors(entityName: string, depth = 1): Promise<NeighborGraph> {
  const res = await client.get<NeighborGraph>(
    `/graph/neighbors/${encodeURIComponent(entityName)}`,
    { params: { depth } },
  );
  return res.data;
}

export async function getChunkDetail(params: {
  chunk_id: string;
  collection?: string;
  paper_id?: string;
}): Promise<ChunkDetail> {
  const { chunk_id, collection, paper_id } = params;
  const res = await client.get<ChunkDetail>(`/graph/chunk/${encodeURIComponent(chunk_id)}`, {
    params: { collection, paper_id },
  });
  return res.data;
}
