/**
 * Agent 工具调用轨迹面板。
 * 展示 ReAct 循环中 LLM 调用的每个工具、参数和结果。
 */

import { useState } from 'react';
import type { ToolTraceItem } from '../../types';

interface Props {
  trace: ToolTraceItem[];
}

const toolIcons: Record<string, string> = {
  search_local: '📚',
  search_web: '🌐',
  search_scholar: '🎓',
  explore_graph: '🕸️',
  canvas: '📝',
  get_citations: '📖',
  compare_papers: '⚖️',
  run_code: '🖥️',
};

export function ToolTracePanel({ trace }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!trace || trace.length === 0) return null;

  return (
    <div className="border rounded-lg bg-white shadow-sm overflow-hidden mb-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2.5 flex items-center justify-between bg-gradient-to-r from-purple-50 to-indigo-50 hover:from-purple-100 hover:to-indigo-100 transition-colors"
      >
        <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
          <span>🔧</span>
          <span>Agent 工具调用</span>
          <span className="text-xs bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded-full">
            {trace.length} 次
          </span>
        </div>
        <span className="text-gray-400 text-xs">{expanded ? '收起' : '展开'}</span>
      </button>

      {expanded && (
        <div className="px-4 py-2 space-y-2 max-h-80 overflow-y-auto">
          {trace.map((item, idx) => (
            <div
              key={idx}
              className={`rounded-md p-2.5 text-xs border ${
                item.is_error
                  ? 'bg-red-50 border-red-200'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span>{toolIcons[item.tool] || '🔧'}</span>
                <span className="font-semibold text-gray-700">{item.tool}</span>
                <span className="text-gray-400">#{item.iteration}</span>
                {item.is_error && (
                  <span className="text-red-500 text-[10px] bg-red-100 px-1 rounded">错误</span>
                )}
              </div>
              <div className="text-gray-500 mb-1">
                <span className="text-gray-400">参数: </span>
                <code className="text-[10px] bg-white px-1 py-0.5 rounded border">
                  {JSON.stringify(item.arguments).slice(0, 120)}
                  {JSON.stringify(item.arguments).length > 120 && '...'}
                </code>
              </div>
              <div className="text-gray-600">
                <span className="text-gray-400">结果: </span>
                {item.result.slice(0, 200)}
                {item.result.length > 200 && '...'}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
