import { useState } from 'react';
import { ChevronDown, ChevronRight, Database, Globe, Clock, BarChart3, Search } from 'lucide-react';
import type { EvidenceSummary } from '../../types';

interface Props {
  summary: EvidenceSummary;
}

const EVIDENCE_TYPE_LABELS: Record<string, string> = {
  finding: '实验发现',
  method: '方法描述',
  interpretation: '讨论解读',
  background: '背景信息',
  summary: '摘要概述',
};

export function RetrievalDebugPanel({ summary }: Props) {
  const [expanded, setExpanded] = useState(false);
  const diag = summary.diagnostics;

  const totalMs = Math.round(summary.retrieval_time_ms);

  return (
    <div className="border border-gray-200 rounded-lg bg-gray-50 text-xs mb-4 overflow-hidden">
      {/* Header - always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-100 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2 text-gray-600">
          <Search size={12} className="text-blue-500" />
          <span className="font-medium">检索诊断</span>
          <span className="text-gray-400">
            {summary.total_chunks} 条结果 · {totalMs}ms
          </span>
          {summary.sources_used.length > 0 && (
            <span className="text-gray-400">
              · {summary.sources_used.join('+')}
            </span>
          )}
        </div>
        {expanded ? <ChevronDown size={14} className="text-gray-400" /> : <ChevronRight size={14} className="text-gray-400" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-200 space-y-3">
          {/* Pipeline stages */}
          {diag?.stages && (
            <div>
              <div className="font-medium text-gray-500 flex items-center gap-1 mb-1.5">
                <BarChart3 size={11} /> 检索流水线
              </div>
              <div className="space-y-1">
                {Object.entries(diag.stages).map(([name, info]) => {
                  const maxMs = Math.max(
                    ...Object.values(diag.stages!).map((s) => s.time_ms),
                    1
                  );
                  const pct = Math.min((info.time_ms / maxMs) * 100, 100);
                  return (
                    <div key={name} className="flex items-center gap-2">
                      <span className="w-24 text-gray-500 truncate">{name}</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-1.5 relative">
                        <div
                          className="absolute top-0 left-0 h-full bg-blue-400 rounded-full transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-16 text-right text-gray-500">
                        {info.count}条 {Math.round(info.time_ms)}ms
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Source breakdown */}
          {summary.source_breakdown && (
            <div>
              <div className="font-medium text-gray-500 flex items-center gap-1 mb-1.5">
                <Database size={11} /> 来源分布
              </div>
              <div className="flex gap-3">
                {Object.entries(summary.source_breakdown).map(([src, count]) => (
                  <span
                    key={src}
                    className={`px-2 py-0.5 rounded-full ${
                      src === 'local'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {src === 'local' ? '本地' : '网络'} {count}
                  </span>
                ))}
                {summary.cross_validated_count ? (
                  <span className="px-2 py-0.5 rounded-full bg-purple-100 text-purple-700">
                    交叉验证 {summary.cross_validated_count}
                  </span>
                ) : null}
              </div>
            </div>
          )}

          {/* Web providers */}
          {diag?.web_providers && Object.keys(diag.web_providers).length > 0 && (
            <div>
              <div className="font-medium text-gray-500 flex items-center gap-1 mb-1.5">
                <Globe size={11} /> Web 来源
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(diag.web_providers).map(([prov, info]) => (
                  <span key={prov} className="px-2 py-0.5 rounded bg-gray-200 text-gray-600">
                    {prov}: {info.count}条 ({Math.round(info.time_ms)}ms)
                  </span>
                ))}
              </div>
              {diag.content_fetcher && (
                <div className="mt-1 text-gray-400">
                  全文抓取: {diag.content_fetcher.enriched}/{diag.content_fetcher.total}
                </div>
              )}
            </div>
          )}

          {/* Evidence types */}
          {summary.evidence_type_breakdown && (
            <div>
              <div className="font-medium text-gray-500 flex items-center gap-1 mb-1.5">
                <BarChart3 size={11} /> 证据类型
              </div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(summary.evidence_type_breakdown).map(([etype, count]) => (
                  <span key={etype} className="px-2 py-0.5 rounded bg-gray-200 text-gray-600">
                    {EVIDENCE_TYPE_LABELS[etype] || etype} {count}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Time range + documents */}
          {(summary.year_range || summary.total_documents) && (
            <div className="flex items-center gap-3 text-gray-400">
              {summary.year_range && summary.year_range[0] && (
                <span className="flex items-center gap-1">
                  <Clock size={10} />
                  {summary.year_range[0] === summary.year_range[1]
                    ? summary.year_range[0]
                    : `${summary.year_range[0]}–${summary.year_range[1]}`}
                </span>
              )}
              {summary.total_documents ? (
                <span>{summary.total_documents} 篇文献</span>
              ) : null}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
