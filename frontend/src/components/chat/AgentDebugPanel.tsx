import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronRight, Bot, Wrench, CheckCircle2, XCircle, Clock, Zap, AlertTriangle } from 'lucide-react';
import type { AgentDebugData, ToolTraceItem } from '../../types';

interface Props {
  data: AgentDebugData;
}

const TOOL_ICONS: Record<string, string> = {
  search_local: '📚',
  search_web: '🌐',
  search_scholar: '🎓',
  search_ncbi: '🧬',
  explore_graph: '🕸️',
  canvas: '📝',
  get_citations: '📖',
  compare_papers: '⚖️',
  run_code: '🖥️',
};

function LatencyBar({ ms, maxMs }: { ms: number; maxMs: number }) {
  const pct = Math.min((ms / Math.max(maxMs, 1)) * 100, 100);
  const color = ms > 5000 ? 'bg-red-400' : ms > 2000 ? 'bg-amber-400' : 'bg-emerald-400';
  return (
    <div className="flex items-center gap-1.5 min-w-[80px]">
      <div className="flex-1 bg-gray-200 rounded-full h-1.5 relative">
        <div className={`absolute top-0 left-0 h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-gray-500 tabular-nums w-12 text-right">{ms}ms</span>
    </div>
  );
}

function ToolCallCard({ item, maxMs }: { item: ToolTraceItem; maxMs: number }) {
  const [showDetail, setShowDetail] = useState(false);
  return (
    <div className={`rounded-md border text-xs ${item.is_error ? 'bg-red-50 border-red-200' : 'bg-white border-gray-200'}`}>
      <button
        onClick={() => setShowDetail(!showDetail)}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 cursor-pointer hover:bg-gray-50 transition-colors"
      >
        <span>{TOOL_ICONS[item.tool] || '🔧'}</span>
        <span className="font-medium text-gray-700">{item.tool}</span>
        {item.is_error && <AlertTriangle size={10} className="text-red-500" />}
        <span className="ml-auto" />
        {item.tool_latency_ms != null && <LatencyBar ms={item.tool_latency_ms} maxMs={maxMs} />}
      </button>
      {showDetail && (
        <div className="px-2.5 pb-2 space-y-1 border-t border-gray-100">
          <div className="text-gray-500 pt-1">
            <span className="text-gray-400">args: </span>
            <code className="text-[10px] bg-gray-100 px-1 py-0.5 rounded break-all">
              {JSON.stringify(item.arguments).slice(0, 300)}
            </code>
          </div>
          <div className="text-gray-600">
            <span className="text-gray-400">result: </span>
            <span className="break-all">{item.result.slice(0, 400)}{item.result.length > 400 ? '...' : ''}</span>
          </div>
          {item.llm_latency_ms != null && (
            <div className="text-gray-400 text-[10px]">
              LLM thinking: {item.llm_latency_ms}ms | Tool exec: {item.tool_latency_ms ?? '?'}ms
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function AgentDebugPanel({ data }: Props) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const stats = data.agent_stats;
  const maxToolMs = Math.max(...data.tool_trace.map((t) => t.tool_latency_ms ?? 0), 1);

  const contributed = data.tools_contributed;

  return (
    <div className="border rounded-lg bg-white shadow-sm overflow-hidden mb-3">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between bg-gradient-to-r from-indigo-50 to-purple-50 hover:from-indigo-100 hover:to-purple-100 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2 text-xs">
          <Bot size={14} className="text-indigo-600" />
          <span className="font-medium text-gray-700">{t('agentDebug.title', 'Agent Debug')}</span>
          <span className="text-gray-500">
            {stats.total_iterations} {t('agentDebug.iterations', 'iter')}
          </span>
          <span className="text-gray-400">·</span>
          <span className="text-gray-500 flex items-center gap-0.5">
            <Wrench size={10} />
            {stats.total_tool_calls} {t('agentDebug.toolCalls', 'calls')}
          </span>
          <span className="text-gray-400">·</span>
          <span className="text-gray-500 flex items-center gap-0.5">
            <Clock size={10} />
            {stats.total_agent_time_ms}ms
          </span>
          <span className="text-gray-400">·</span>
          {contributed ? (
            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-[10px] font-medium">
              <CheckCircle2 size={10} />
              {t('agentDebug.contributed', 'Contributed')}
            </span>
          ) : (
            <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-gray-200 text-gray-500 text-[10px] font-medium">
              <XCircle size={10} />
              {t('agentDebug.notContributed', 'No contribution')}
            </span>
          )}
        </div>
        {expanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-2 space-y-3">
          {/* Time split */}
          <div className="flex items-center gap-3 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <Zap size={10} className="text-amber-500" />
              LLM: {stats.total_llm_time_ms}ms
            </span>
            <span className="flex items-center gap-1">
              <Wrench size={10} className="text-blue-500" />
              Tools: {stats.total_tool_time_ms}ms
            </span>
            {stats.error_count > 0 && (
              <span className="flex items-center gap-1 text-red-500">
                <AlertTriangle size={10} />
                {stats.error_count} error{stats.error_count > 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* Tools used tags */}
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(stats.tools_used_summary).map(([tool, count]) => (
              <span
                key={tool}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 text-[10px]"
              >
                {TOOL_ICONS[tool] || '🔧'} {tool} x{count}
              </span>
            ))}
          </div>

          {/* Chunk contribution stats */}
          {(data.agent_added_chunks > 0 || data.pre_retrieval_chunks > 0) && (
            <div className="text-[10px] text-gray-500 bg-gray-50 rounded p-2 space-y-0.5">
              <div>{t('agentDebug.preChunks', 'Pre-retrieval chunks')}: {data.pre_retrieval_chunks}</div>
              <div>{t('agentDebug.agentChunks', 'Agent added chunks')}: {data.agent_added_chunks}</div>
              <div>{t('agentDebug.citedFromAgent', 'Cited from agent')}: {data.cited_from_agent}</div>
            </div>
          )}

          {/* Timeline */}
          <div className="space-y-1.5">
            {data.tool_trace.map((item, idx) => (
              <ToolCallCard key={idx} item={item} maxMs={maxToolMs} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
