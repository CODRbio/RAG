import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import {
  AlertCircle,
  ArrowRight,
  Loader2,
  Maximize2,
  RefreshCw,
  Search,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import {
  getChunkDetail,
  getEntities,
  getGraphStats,
  getNeighbors,
  getTypedGraphStats,
  queryTypedGraphSubgraph,
  rebuildTypedGraphSnapshot,
  summarizeTypedGraph,
  type ChunkDetail,
  type EntityItem,
  type GraphStats,
  type NeighborGraph,
} from '../../api/graph';
import { useAnalysisPoolStore, useConfigStore } from '../../stores';
import type {
  GraphScopePayload,
  TypedGraphMetrics,
  GraphType,
  TypedGraphNode,
  TypedGraphStats,
  TypedGraphSubgraph,
} from '../../types';
import { Modal } from '../ui/Modal';

const ENTITY_TYPE_COLORS: Record<string, string> = {
  SPECIES: '#3b82f6',
  LOCATION: '#10b981',
  PHENOMENON: '#f59e0b',
  METHOD: '#8b5cf6',
  SUBSTANCE: '#ef4444',
  CHUNK: '#94a3b8',
  ENTITY: '#6b7280',
};

const ACADEMIC_TYPE_COLORS: Record<string, string> = {
  paper: '#2563eb',
  author: '#059669',
  institution: '#d97706',
  topic: '#7c3aed',
  unknown: '#64748b',
};

type TypedGraphMode = Exclude<GraphType, 'entity'>;

interface ForceNode {
  id: string;
  type: string;
  label?: string;
  paper_id?: string;
  is_center?: boolean;
  is_seed?: boolean;
  val: number;
  pagerank?: number;
  degree?: number;
  x?: number;
  y?: number;
}

interface ForceLink {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

interface GraphData {
  nodes: ForceNode[];
  links: ForceLink[];
}

const GRAPH_TYPES: GraphType[] = ['entity', 'citation', 'author', 'institution'];

function parseSeedList(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function toGraphData(subgraph: TypedGraphSubgraph): GraphData {
  const nodes = (subgraph.nodes || []).map((node: TypedGraphNode) => ({
    id: node.id,
    type: String(node.type || 'unknown').toLowerCase(),
    label: typeof node.label === 'string' ? node.label : node.id,
    is_seed: Boolean(node.is_seed),
    pagerank: typeof node.pagerank === 'number' ? node.pagerank : undefined,
    degree: typeof node.degree === 'number' ? node.degree : undefined,
    val: node.is_seed ? 7 : node.type === 'paper' ? 5 : 4,
  }));
  const links = (subgraph.edges || []).map((edge) => ({
    source: edge.source,
    target: edge.target,
    relation: String(edge.relation || ''),
    weight: typeof edge.weight === 'number' ? edge.weight : 1,
  }));
  return { nodes, links };
}

export function GraphExplorer() {
  const fgRef = useRef<any>(null);
  const currentCollection = useConfigStore((s) => s.currentCollection);
  const analysisPoolItems = useAnalysisPoolStore((s) => s.items);

  const [graphType, setGraphType] = useState<GraphType>('entity');
  const [entityStats, setEntityStats] = useState<GraphStats | null>(null);
  const [typedStats, setTypedStats] = useState<TypedGraphStats | null>(null);
  const [typedMetrics, setTypedMetrics] = useState<TypedGraphMetrics | null>(null);
  const [entities, setEntities] = useState<EntityItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEntity, setSelectedEntity] = useState('');
  const [depth, setDepth] = useState(1);
  const [limit, setLimit] = useState(60);
  const [scopeType, setScopeType] = useState<GraphScopePayload['scope_type']>('collection');
  const [scopeKey, setScopeKey] = useState(currentCollection || 'global');
  const [seedNodeIdsInput, setSeedNodeIdsInput] = useState('');
  const [paperUidsInput, setPaperUidsInput] = useState('');
  const [typedSummary, setTypedSummary] = useState('');
  const [typedSnapshotVersion, setTypedSnapshotVersion] = useState<number | null>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [error, setError] = useState('');
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<ForceNode | null>(null);
  const [chunkModalOpen, setChunkModalOpen] = useState(false);
  const [chunkDetailLoading, setChunkDetailLoading] = useState(false);
  const [chunkDetailError, setChunkDetailError] = useState('');
  const [selectedChunk, setSelectedChunk] = useState<ChunkDetail | null>(null);

  const currentStats = graphType === 'entity' ? entityStats : typedStats;
  const typeColors = graphType === 'entity' ? ENTITY_TYPE_COLORS : ACADEMIC_TYPE_COLORS;
  const analysisPoolPaperUids = useMemo(
    () => analysisPoolItems.map((item) => item.paper_uid).filter(Boolean),
    [analysisPoolItems],
  );
  const typedSeedNodeIds = useMemo(() => parseSeedList(seedNodeIdsInput), [seedNodeIdsInput]);
  const typedPaperUids = useMemo(() => parseSeedList(paperUidsInput), [paperUidsInput]);
  const canRunTypedQuery =
    graphType !== 'entity' &&
    scopeKey.trim().length > 0 &&
    (typedSeedNodeIds.length > 0 || typedPaperUids.length > 0);

  const scopePayload = useMemo<GraphScopePayload>(
    () => ({
      scope_type: scopeType,
      scope_key: scopeKey.trim() || 'global',
    }),
    [scopeKey, scopeType],
  );

  useEffect(() => {
    if (scopeType === 'collection') {
      setScopeKey(currentCollection || 'global');
    }
  }, [currentCollection, scopeType]);

  const loadTypedStats = useCallback(async () => {
    if (graphType === 'entity') return;
    try {
      const data = await getTypedGraphStats(graphType as TypedGraphMode, scopePayload);
      setTypedStats(data);
    } catch {
      setTypedStats(null);
    }
  }, [graphType, scopePayload]);

  useEffect(() => {
    if (graphType === 'entity') {
      getGraphStats().then(setEntityStats).catch(() => setEntityStats(null));
    } else {
      void loadTypedStats();
    }
  }, [graphType, loadTypedStats]);

  useEffect(() => {
    if (graphType !== 'entity') {
      setEntities([]);
      return;
    }
    const timer = setTimeout(() => {
      getEntities({ q: searchQuery || undefined, limit: 50 })
        .then((res) => setEntities(res.entities))
        .catch(() => setEntities([]));
    }, 300);
    return () => clearTimeout(timer);
  }, [graphType, searchQuery]);

  const loadNeighbors = useCallback(async (name: string, d?: number) => {
    setLoading(true);
    setError('');
    try {
      const graph: NeighborGraph = await getNeighbors(name, d ?? depth);
      const nodes = graph.nodes.map((node) => ({
        id: node.id,
        type: node.type,
        paper_id: node.paper_id,
        is_center: node.is_center,
        val: node.is_center ? 8 : node.type === 'CHUNK' ? 2 : 4,
      }));
      const links = graph.edges.map((edge) => ({
        source: edge.source,
        target: edge.target,
        relation: edge.relation,
        weight: edge.weight,
      }));
      setGraphData({ nodes, links });
      setSelectedEntity(name);
      setTypedSummary('');
      setTimeout(() => fgRef.current?.zoomToFit(400, 40), 200);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || '加载失败');
    } finally {
      setLoading(false);
    }
  }, [depth]);

  const runTypedSubgraph = useCallback(async (overrideSeedNodeIds?: string[]) => {
    if (graphType === 'entity') return;
    setLoading(true);
    setError('');
    try {
      const subgraph = await queryTypedGraphSubgraph(graphType as TypedGraphMode, {
        scope: scopePayload,
        seed_node_ids: overrideSeedNodeIds || typedSeedNodeIds,
        paper_uids: typedPaperUids,
        depth,
        limit,
      });
      setGraphData(toGraphData(subgraph));
      setTypedMetrics(subgraph.metrics || null);
      setTypedSnapshotVersion(subgraph.snapshot_version ?? null);
      setSelectedNode(null);
      setTimeout(() => fgRef.current?.zoomToFit(400, 40), 200);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || '图查询失败');
    } finally {
      setLoading(false);
    }
  }, [depth, graphType, limit, scopePayload, typedPaperUids, typedSeedNodeIds]);

  const runTypedSummary = useCallback(async () => {
    if (graphType === 'entity') return;
    setSummaryLoading(true);
    setError('');
    try {
      const summary = await summarizeTypedGraph(graphType as TypedGraphMode, {
        subgraph_request: {
          scope: scopePayload,
          seed_node_ids: typedSeedNodeIds,
          paper_uids: typedPaperUids,
          depth,
          limit,
          snapshot_version: typedSnapshotVersion,
        },
        format: 'markdown',
      });
      setTypedSummary(summary.summary);
      setTypedSnapshotVersion(summary.snapshot_version ?? null);
      setGraphData(toGraphData(summary.subgraph));
      setTypedMetrics(summary.subgraph.metrics || null);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || '图摘要失败');
    } finally {
      setSummaryLoading(false);
    }
  }, [depth, graphType, limit, scopePayload, typedPaperUids, typedSeedNodeIds, typedSnapshotVersion]);

  const rebuildSnapshot = useCallback(async () => {
    if (graphType === 'entity') return;
    setRebuilding(true);
    setError('');
    try {
      const snapshot = await rebuildTypedGraphSnapshot(graphType as TypedGraphMode, scopePayload);
      setTypedSnapshotVersion(snapshot.snapshot_version ?? null);
      await loadTypedStats();
      await runTypedSubgraph();
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || '快照重建失败');
    } finally {
      setRebuilding(false);
    }
  }, [graphType, loadTypedStats, runTypedSubgraph, scopePayload]);

  const handleNodeClick = useCallback((node: ForceNode) => {
    setSelectedNode(node);
    if (graphType === 'entity') {
      if (node.type === 'CHUNK') {
        setChunkModalOpen(true);
        setChunkDetailLoading(true);
        setChunkDetailError('');
        setSelectedChunk(null);
        getChunkDetail({
          chunk_id: String(node.id),
          collection: currentCollection || undefined,
          paper_id: node.paper_id ? String(node.paper_id) : undefined,
        })
          .then((detail) => setSelectedChunk(detail))
          .catch((e: unknown) => {
            const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
            setChunkDetailError(detail || '加载 chunk 详情失败');
          })
          .finally(() => setChunkDetailLoading(false));
        return;
      }
      void loadNeighbors(node.id);
      return;
    }
    setSeedNodeIdsInput(node.id);
    void runTypedSubgraph([node.id]);
  }, [currentCollection, graphType, loadNeighbors, runTypedSubgraph]);

  if (graphType === 'entity' && entityStats && !entityStats.available) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-gray-400">
        <AlertCircle size={40} />
        <p className="text-sm">知识图谱尚未构建，请先通过 Ingest 导入论文并构建图谱</p>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      <div className="w-80 border-r border-gray-200 flex flex-col bg-white">
        <div className="border-b border-gray-100 px-3 py-3">
          <div className="mb-3 flex flex-wrap gap-2">
            {GRAPH_TYPES.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => {
                  setGraphType(item);
                  setGraphData({ nodes: [], links: [] });
                  setTypedSummary('');
                  setTypedMetrics(null);
                  setSelectedNode(null);
                  setError('');
                }}
                className={`rounded-full px-3 py-1 text-xs ${
                  graphType === item ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {item}
              </button>
            ))}
          </div>
          {currentStats && (
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-500">
              <span>节点 {currentStats.total_nodes}</span>
              <span>边 {currentStats.total_edges}</span>
              {'entity_count' in currentStats && <span>实体 {currentStats.entity_count}</span>}
              {'fact_count' in currentStats && <span>事实 {currentStats.fact_count}</span>}
              {'snapshot_status' in currentStats && currentStats.snapshot_status && (
                <span>快照 {currentStats.snapshot_status}</span>
              )}
            </div>
          )}
        </div>

        {graphType === 'entity' ? (
          <>
            <div className="px-3 py-2 border-b border-gray-100">
              <div className="relative">
                <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="搜索实体..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-7 pr-2 py-1.5 text-sm border border-gray-200 rounded-md focus:outline-none focus:border-blue-400"
                />
              </div>
            </div>
            <div className="px-3 py-2 border-b border-gray-100 flex items-center gap-2 text-xs">
              <span className="text-gray-500">深度</span>
              {[1, 2, 3].map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => {
                    setDepth(d);
                    if (selectedEntity) {
                      void loadNeighbors(selectedEntity, d);
                    }
                  }}
                  className={`px-2 py-0.5 rounded ${
                    depth === d ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto">
              {entities.map((entity) => (
                <button
                  key={entity.name}
                  type="button"
                  onClick={() => {
                    void loadNeighbors(entity.name);
                  }}
                  className={`w-full text-left px-3 py-1.5 text-sm border-b border-gray-50 hover:bg-gray-50 flex items-center gap-2 ${
                    selectedEntity === entity.name ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: ENTITY_TYPE_COLORS[entity.type] || '#6b7280' }}
                  />
                  <span className="truncate flex-1">{entity.name}</span>
                  <span className="text-[10px] text-gray-400">{entity.mention_count}</span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3 text-sm">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Scope</label>
              <div className="flex gap-2">
                <select
                  value={scopeType}
                  onChange={(e) => setScopeType(e.target.value as GraphScopePayload['scope_type'])}
                  className="rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
                >
                  <option value="global">global</option>
                  <option value="collection">collection</option>
                  <option value="library">library</option>
                </select>
                <input
                  value={scopeKey}
                  onChange={(e) => setScopeKey(e.target.value)}
                  placeholder={scopeType === 'collection' ? 'collection name' : scopeType === 'library' ? 'library id' : 'global'}
                  className="flex-1 rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">Seed Node IDs</label>
              <textarea
                value={seedNodeIdsInput}
                onChange={(e) => setSeedNodeIdsInput(e.target.value)}
                placeholder="author:... or institution:... or paper:..."
                className="h-20 w-full rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-500 mb-1">Paper UIDs</label>
              <textarea
                value={paperUidsInput}
                onChange={(e) => setPaperUidsInput(e.target.value)}
                placeholder="paper uid, one per line"
                className="h-20 w-full rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
              />
              <div className="mt-2 flex items-center justify-between gap-2">
                <span className="text-[11px] text-gray-500">
                  Analysis Pool {analysisPoolPaperUids.length > 0 ? `(${analysisPoolPaperUids.length})` : ''}
                </span>
                <button
                  type="button"
                  disabled={analysisPoolPaperUids.length === 0}
                  onClick={() => setPaperUidsInput(analysisPoolPaperUids.join('\n'))}
                  className="inline-flex items-center gap-1 rounded-md border border-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                >
                  <ArrowRight size={12} />
                  使用分析池
                </button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Depth</label>
                <select
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  className="rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Limit</label>
                <input
                  type="number"
                  min={10}
                  max={300}
                  value={limit}
                  onChange={(e) => setLimit(Math.max(10, Math.min(300, Number(e.target.value) || 60)))}
                  className="w-24 rounded-md border border-gray-200 px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-2">
              <button
                type="button"
                onClick={() => void runTypedSubgraph()}
                disabled={!canRunTypedQuery}
                className="rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500"
              >
                查询子图
              </button>
              <button
                type="button"
                onClick={() => void runTypedSummary()}
                disabled={summaryLoading || !canRunTypedQuery}
                className="rounded-md border border-gray-200 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                {summaryLoading ? '生成摘要中...' : '图摘要'}
              </button>
              <button
                type="button"
                onClick={() => void rebuildSnapshot()}
                disabled={rebuilding}
                className="inline-flex items-center justify-center gap-2 rounded-md border border-gray-200 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                {rebuilding ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                重建快照
              </button>
            </div>

            {!canRunTypedQuery && (
              <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-3 text-xs text-gray-500">
                先选择 scope，并至少填写一组 seeds（paper uid 或 node id）。
              </div>
            )}
          </div>
        )}

        <div className="px-3 py-2 border-t border-gray-100 text-[10px] text-gray-500">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {Object.entries(typeColors)
              .filter(([key]) => key !== 'unknown')
              .map(([type, color]) => (
                <span key={type} className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  {type}
                </span>
              ))}
          </div>
        </div>
      </div>

      <div className="flex-1 relative bg-gray-50">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/60 z-10">
            <Loader2 size={24} className="animate-spin text-blue-500" />
          </div>
        )}

        {error && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-red-50 text-red-600 text-sm px-4 py-2 rounded-lg shadow">
            {error}
          </div>
        )}

        {graphData.nodes.length === 0 && !loading ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            {graphType === 'entity' ? '选择左侧实体以查看知识图谱' : '输入 scope 与种子后查询子图'}
          </div>
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel={(node: ForceNode) => `${node.label || node.id} (${node.type})`}
            nodeColor={(node: ForceNode) => typeColors[node.type] || typeColors.unknown || '#6b7280'}
            nodeVal={(node: ForceNode) => node.val}
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(node: ForceNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
              if (globalScale < 1.5 && !node.is_center && !node.is_seed && node.id !== hoveredNode) return;
              const rawLabel = node.label || node.id;
              const label = rawLabel.length > 22 ? `${rawLabel.slice(0, 20)}…` : rawLabel;
              const fontSize = Math.max(10 / globalScale, 2);
              ctx.font = `${node.is_center || node.is_seed ? 'bold ' : ''}${fontSize}px sans-serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = node.is_center || node.is_seed ? '#1e40af' : '#374151';
              ctx.fillText(label, node.x || 0, (node.y || 0) + 8 / globalScale);
            }}
            linkColor={() => '#d1d5db'}
            linkWidth={(link: ForceLink) => Math.min(link.weight || 1, 4)}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkLabel={(link: ForceLink) => link.relation}
            onNodeClick={(node) => handleNodeClick(node as ForceNode)}
            onNodeHover={(node) => setHoveredNode((node as ForceNode | null)?.id || null)}
            cooldownTicks={80}
            warmupTicks={30}
          />
        )}

        <div className="absolute bottom-4 right-4 flex flex-col gap-1">
          <button
            type="button"
            onClick={() => fgRef.current?.zoom(fgRef.current.zoom() * 1.3, 300)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <ZoomIn size={16} />
          </button>
          <button
            type="button"
            onClick={() => fgRef.current?.zoom(fgRef.current.zoom() / 1.3, 300)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <ZoomOut size={16} />
          </button>
          <button
            type="button"
            onClick={() => fgRef.current?.zoomToFit(400, 40)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>

      <div className="w-[340px] border-l border-gray-200 bg-white flex flex-col">
        <div className="border-b border-gray-100 px-4 py-3">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-gray-500">
            Graph Details
          </div>
          <div className="mt-2 text-sm font-semibold text-gray-900">
            {graphType === 'entity' ? 'Entity Graph' : `${graphType} graph`}
          </div>
          {typedSnapshotVersion != null && graphType !== 'entity' && (
            <div className="mt-1 text-xs text-gray-500">
              Snapshot v{typedSnapshotVersion}
              {typedStats?.snapshot_status ? ` · ${typedStats.snapshot_status}` : ''}
            </div>
          )}
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
          {graphType !== 'entity' && typedSummary && (
            <section className="rounded-xl border border-gray-200 bg-gray-50 p-3">
              <div className="mb-2 text-xs font-medium text-gray-600">Summary</div>
              <pre className="whitespace-pre-wrap text-xs leading-5 text-gray-700">{typedSummary}</pre>
            </section>
          )}

          {graphType !== 'entity' && typedMetrics && (
            <section className="rounded-xl border border-gray-200 bg-gray-50 p-3">
              <div className="mb-2 text-xs font-medium text-gray-600">Metrics</div>
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                <div>nodes: {typedMetrics.node_count ?? graphData.nodes.length}</div>
                <div>edges: {typedMetrics.edge_count ?? graphData.links.length}</div>
              </div>
              {Array.isArray(typedMetrics.top_nodes) && typedMetrics.top_nodes.length > 0 && (
                <div className="mt-3">
                  <div className="text-[11px] font-medium text-gray-500">Top Nodes</div>
                  <div className="mt-2 space-y-1">
                    {typedMetrics.top_nodes.slice(0, 5).map((item) => (
                      <div key={item.id} className="flex items-center justify-between gap-2 rounded-lg border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700">
                        <span className="truncate">{item.id}</span>
                        <span>{typeof item.score === 'number' ? item.score.toFixed(3) : '—'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {Array.isArray(typedMetrics.bridge_nodes) && typedMetrics.bridge_nodes.length > 0 && (
                <div className="mt-3">
                  <div className="text-[11px] font-medium text-gray-500">Bridge Nodes</div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {typedMetrics.bridge_nodes.slice(0, 8).map((item) => (
                      <span key={item} className="rounded-full bg-gray-200 px-2 py-0.5 text-[11px] text-gray-700">
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          <section className="rounded-xl border border-gray-200 bg-gray-50 p-3">
            <div className="mb-2 text-xs font-medium text-gray-600">Selected Node</div>
            {selectedNode ? (
              <div className="space-y-2 text-xs text-gray-700">
                <div className="font-medium text-gray-900 break-all">{selectedNode.label || selectedNode.id}</div>
                <div>id: {selectedNode.id}</div>
                <div>type: {selectedNode.type}</div>
                {typeof selectedNode.pagerank === 'number' && <div>pagerank: {selectedNode.pagerank.toFixed(4)}</div>}
                {typeof selectedNode.degree === 'number' && <div>degree: {selectedNode.degree}</div>}
                {selectedNode.paper_id && <div>paper_id: {selectedNode.paper_id}</div>}
              </div>
            ) : (
              <div className="text-xs text-gray-500">
                点击图节点后，这里会显示当前对象详情。
              </div>
            )}
          </section>

          <section className="rounded-xl border border-gray-200 bg-gray-50 p-3">
            <div className="mb-2 text-xs font-medium text-gray-600">Legend</div>
            <div className="flex flex-wrap gap-x-3 gap-y-2 text-xs text-gray-600">
              {Object.entries(typeColors)
                .filter(([key]) => key !== 'unknown')
                .map(([type, color]) => (
                  <span key={type} className="flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                    {type}
                  </span>
                ))}
            </div>
          </section>
        </div>
      </div>

      <Modal
        open={chunkModalOpen}
        onClose={() => {
          setChunkModalOpen(false);
          setSelectedChunk(null);
          setChunkDetailError('');
        }}
        title={selectedChunk ? `Chunk ${selectedChunk.chunk_id}` : 'Chunk Detail'}
        maxWidth="max-w-4xl"
      >
        {chunkDetailLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 size={24} className="animate-spin text-blue-500" />
          </div>
        ) : chunkDetailError ? (
          <div className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600">{chunkDetailError}</div>
        ) : selectedChunk ? (
          <div className="space-y-3 text-sm text-slate-700">
            <div className="grid grid-cols-2 gap-2 text-xs text-slate-500">
              <div>collection: {selectedChunk.collection}</div>
              <div>paper: {selectedChunk.paper_id}</div>
              {selectedChunk.section_path && <div>section: {selectedChunk.section_path}</div>}
              {selectedChunk.page != null && <div>page: {selectedChunk.page}</div>}
            </div>
            <div className="whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-slate-800">
              {selectedChunk.content}
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
