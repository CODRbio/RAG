import client from './client';
import type {
  GraphScopePayload,
  GraphSnapshotItem,
  GraphType,
  TypedGraphStats,
  TypedGraphSubgraph,
  TypedGraphSummary,
} from '../types';

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
  bbox?: number[] | number[][];
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

export async function getTypedGraphStats(
  graphType: Exclude<GraphType, 'entity'>,
  scope: GraphScopePayload,
): Promise<TypedGraphStats> {
  const res = await client.get<TypedGraphStats>(`/graph/${encodeURIComponent(graphType)}/stats`, {
    params: {
      scope_type: scope.scope_type,
      scope_key: scope.scope_key,
    },
  });
  return res.data;
}

export async function queryTypedGraphSubgraph(
  graphType: Exclude<GraphType, 'entity'>,
  body: {
    scope: GraphScopePayload;
    seed_node_ids?: string[];
    paper_uids?: string[];
    depth?: number;
    limit?: number;
    snapshot_version?: number | null;
  },
): Promise<TypedGraphSubgraph> {
  const res = await client.post<TypedGraphSubgraph>(`/graph/${encodeURIComponent(graphType)}/subgraph`, body);
  return res.data;
}

export async function summarizeTypedGraph(
  graphType: Exclude<GraphType, 'entity'>,
  body: {
    subgraph_request: {
      scope: GraphScopePayload;
      seed_node_ids?: string[];
      paper_uids?: string[];
      depth?: number;
      limit?: number;
      snapshot_version?: number | null;
    };
    question?: string;
    max_items?: number;
    format?: 'markdown';
  },
): Promise<TypedGraphSummary> {
  const res = await client.post<TypedGraphSummary>(`/graph/${encodeURIComponent(graphType)}/summary`, body);
  return res.data;
}

export async function listTypedGraphSnapshots(
  graphType: Exclude<GraphType, 'entity'>,
  scope: GraphScopePayload,
): Promise<{ items: GraphSnapshotItem[] }> {
  const res = await client.get<{ items: GraphSnapshotItem[] }>(`/graph/${encodeURIComponent(graphType)}/snapshots`, {
    params: {
      scope_type: scope.scope_type,
      scope_key: scope.scope_key,
    },
  });
  return res.data;
}

export async function rebuildTypedGraphSnapshot(
  graphType: Exclude<GraphType, 'entity'>,
  scope: GraphScopePayload,
): Promise<GraphSnapshotItem> {
  const res = await client.post<GraphSnapshotItem>(`/graph/${encodeURIComponent(graphType)}/snapshots/rebuild`, scope);
  return res.data;
}
