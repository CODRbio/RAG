import { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Search, ZoomIn, ZoomOut, Maximize2, Loader2, AlertCircle } from 'lucide-react';
import {
  getGraphStats,
  getEntities,
  getNeighbors,
  getChunkDetail,
  type GraphStats,
  type EntityItem,
  type NeighborGraph,
  type ChunkDetail,
} from '../../api/graph';
import { useConfigStore } from '../../stores';
import { Modal } from '../ui/Modal';

// ── 颜色映射 ──
const TYPE_COLORS: Record<string, string> = {
  SPECIES: '#3b82f6',
  LOCATION: '#10b981',
  PHENOMENON: '#f59e0b',
  METHOD: '#8b5cf6',
  SUBSTANCE: '#ef4444',
  CHUNK: '#94a3b8',
  ENTITY: '#6b7280',
};

interface GraphData {
  nodes: { id: string; type: string; paper_id?: string; is_center?: boolean; val: number }[];
  links: { source: string; target: string; relation: string; weight: number }[];
}

export function GraphExplorer() {
  const fgRef = useRef<any>(null);
  const currentCollection = useConfigStore((s) => s.currentCollection);

  const [stats, setStats] = useState<GraphStats | null>(null);
  const [entities, setEntities] = useState<EntityItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedEntity, setSelectedEntity] = useState('');
  const [depth, setDepth] = useState(1);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [chunkModalOpen, setChunkModalOpen] = useState(false);
  const [chunkDetailLoading, setChunkDetailLoading] = useState(false);
  const [chunkDetailError, setChunkDetailError] = useState('');
  const [selectedChunk, setSelectedChunk] = useState<ChunkDetail | null>(null);

  // 加载统计
  useEffect(() => {
    getGraphStats().then(setStats).catch(() => setStats(null));
  }, []);

  // 搜索实体
  useEffect(() => {
    const timer = setTimeout(() => {
      getEntities({ q: searchQuery || undefined, limit: 50 })
        .then((res) => setEntities(res.entities))
        .catch(() => setEntities([]));
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // 加载子图
  const loadNeighbors = useCallback(async (name: string, d?: number) => {
    setLoading(true);
    setError('');
    try {
      const graph: NeighborGraph = await getNeighbors(name, d ?? depth);
      const nodes = graph.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        paper_id: n.paper_id,
        is_center: n.is_center,
        val: n.is_center ? 8 : n.type === 'CHUNK' ? 2 : 4,
      }));
      const links = graph.edges.map((e) => ({
        source: e.source,
        target: e.target,
        relation: e.relation,
        weight: e.weight,
      }));
      setGraphData({ nodes, links });
      setSelectedEntity(name);
      setTimeout(() => fgRef.current?.zoomToFit(400, 40), 200);
    } catch (e: any) {
      setError(e?.response?.data?.detail || '加载失败');
    } finally {
      setLoading(false);
    }
  }, [depth]);

  const handleEntityClick = (name: string) => {
    loadNeighbors(name);
  };

  const handleNodeClick = useCallback((node: any) => {
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
        .catch((e: any) => setChunkDetailError(e?.response?.data?.detail || '加载 chunk 详情失败'))
        .finally(() => setChunkDetailLoading(false));
      return;
    }
    loadNeighbors(node.id);
  }, [currentCollection, loadNeighbors]);

  // ── 不可用 ──
  if (stats && !stats.available) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
        <AlertCircle size={40} />
        <p className="text-sm">知识图谱尚未构建，请先通过 Ingest 导入论文并构建图谱</p>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── 左侧面板 ── */}
      <div className="w-64 border-r border-gray-200 flex flex-col bg-white">
        {/* 统计 */}
        {stats && (
          <div className="px-3 py-2 border-b border-gray-100 text-xs text-gray-500 flex flex-wrap gap-x-3">
            <span>节点 {stats.total_nodes}</span>
            <span>边 {stats.total_edges}</span>
            <span>实体 {stats.entity_count}</span>
          </div>
        )}

        {/* 搜索 */}
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

        {/* 深度选择 */}
        <div className="px-3 py-2 border-b border-gray-100 flex items-center gap-2 text-xs">
          <span className="text-gray-500">深度</span>
          {[1, 2, 3].map((d) => (
            <button
              key={d}
              onClick={() => {
                setDepth(d);
                if (selectedEntity) loadNeighbors(selectedEntity, d);
              }}
              className={`px-2 py-0.5 rounded ${
                depth === d ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* 实体列表 */}
        <div className="flex-1 overflow-y-auto">
          {entities.map((e) => (
            <button
              key={e.name}
              onClick={() => handleEntityClick(e.name)}
              className={`w-full text-left px-3 py-1.5 text-sm border-b border-gray-50 hover:bg-gray-50 flex items-center gap-2 ${
                selectedEntity === e.name ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
              }`}
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: TYPE_COLORS[e.type] || '#6b7280' }}
              />
              <span className="truncate flex-1">{e.name}</span>
              <span className="text-[10px] text-gray-400">{e.mention_count}</span>
            </button>
          ))}
        </div>

        {/* 图例 */}
        <div className="px-3 py-2 border-t border-gray-100 text-[10px] text-gray-500">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {Object.entries(TYPE_COLORS).filter(([k]) => k !== 'ENTITY').map(([type, color]) => (
              <span key={type} className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                {type}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── 右侧图谱 ── */}
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
            选择左侧实体以查看知识图谱
          </div>
        ) : (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel={(node: any) => `${node.id} (${node.type})`}
            nodeColor={(node: any) => TYPE_COLORS[node.type] || '#6b7280'}
            nodeVal={(node: any) => node.val}
            nodeCanvasObjectMode={() => 'after'}
            nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
              if (globalScale < 1.5 && !node.is_center && node.id !== hoveredNode) return;
              const label = node.id.length > 20 ? node.id.slice(0, 18) + '…' : node.id;
              const fontSize = Math.max(10 / globalScale, 2);
              ctx.font = `${node.is_center ? 'bold ' : ''}${fontSize}px sans-serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = node.is_center ? '#1e40af' : '#374151';
              ctx.fillText(label, node.x, node.y + 8 / globalScale);
            }}
            linkColor={() => '#d1d5db'}
            linkWidth={(link: any) => Math.min(link.weight || 1, 4)}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkLabel={(link: any) => link.relation}
            onNodeClick={handleNodeClick}
            onNodeHover={(node: any) => setHoveredNode(node?.id || null)}
            cooldownTicks={80}
            warmupTicks={30}
          />
        )}

        {/* 缩放控制 */}
        <div className="absolute bottom-4 right-4 flex flex-col gap-1">
          <button
            onClick={() => fgRef.current?.zoom(fgRef.current.zoom() * 1.3, 300)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => fgRef.current?.zoom(fgRef.current.zoom() / 1.3, 300)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <ZoomOut size={16} />
          </button>
          <button
            onClick={() => fgRef.current?.zoomToFit(400, 40)}
            className="p-1.5 bg-white border border-gray-200 rounded shadow-sm hover:bg-gray-50"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      <Modal
        open={chunkModalOpen}
        onClose={() => setChunkModalOpen(false)}
        title="Chunk 详情"
        maxWidth="max-w-3xl"
      >
        {chunkDetailLoading ? (
          <div className="py-10 text-center text-gray-500">
            <Loader2 size={20} className="animate-spin mx-auto mb-2" />
            正在加载内容...
          </div>
        ) : chunkDetailError ? (
          <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg p-3">
            {chunkDetailError}
          </div>
        ) : selectedChunk ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-600">
              <div><span className="text-gray-400">Collection:</span> {selectedChunk.collection || '-'}</div>
              <div><span className="text-gray-400">Paper ID:</span> {selectedChunk.paper_id || '-'}</div>
              <div><span className="text-gray-400">Page:</span> {selectedChunk.page ?? '-'}</div>
              <div><span className="text-gray-400">Type:</span> {selectedChunk.content_type || selectedChunk.chunk_type || '-'}</div>
              <div className="md:col-span-2 break-all">
                <span className="text-gray-400">Chunk ID:</span> {selectedChunk.chunk_id}
              </div>
              {selectedChunk.section_path && (
                <div className="md:col-span-2 break-all">
                  <span className="text-gray-400">Section:</span> {selectedChunk.section_path}
                </div>
              )}
            </div>
            {selectedChunk.related_entities && selectedChunk.related_entities.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-1">关联实体（Top {selectedChunk.related_entities.length}）</div>
                <div className="flex flex-wrap gap-1.5">
                  {selectedChunk.related_entities.map((name) => (
                    <button
                      key={name}
                      onClick={() => {
                        setChunkModalOpen(false);
                        loadNeighbors(name);
                      }}
                      className="px-2 py-1 text-xs rounded-full bg-blue-50 text-blue-700 hover:bg-blue-100"
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div>
              <div className="text-xs text-gray-400 mb-1">内容</div>
              <div className="max-h-[45vh] overflow-y-auto bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                {selectedChunk.content || '(空内容)'}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-500">未找到 chunk 内容</div>
        )}
      </Modal>
    </div>
  );
}
