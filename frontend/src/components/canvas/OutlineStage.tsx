import { GripVertical, ChevronRight, AlertTriangle, BookOpen } from 'lucide-react';
import type { Canvas } from '../../types';

interface OutlineStageProps {
  canvas: Canvas;
}

/**
 * Outline 阶段：可视化的章节树 + Gap 标注
 * 展示大纲结构、每个章节的状态和关联信息
 */
export function OutlineStage({ canvas }: OutlineStageProps) {
  const { outline, drafts, identified_gaps } = canvas;

  if (!outline || outline.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-[var(--text-tertiary)] text-sm px-6">
        <ChevronRight size={36} className="mb-3 opacity-30" />
        <p className="font-medium mb-1">大纲尚未生成</p>
        <p className="text-xs text-center">
          在 Explore 阶段确认研究计划后，<br />
          大纲结构将显示在此处。
        </p>
      </div>
    );
  }

  const statusConfig: Record<string, { icon: string; label: string; className: string }> = {
    todo: { icon: '⬜', label: '待开始', className: 'text-gray-400' },
    drafting: { icon: '✍️', label: '撰写中', className: 'text-orange-500' },
    done: { icon: '✅', label: '已完成', className: 'text-emerald-600' },
  };

  // 按 level 分组，构建层级视图
  const sortedOutline = [...outline].sort((a, b) => a.order - b.order);

  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      {/* Working Title */}
      {canvas.working_title && (
        <div className="bg-[var(--bg-surface)] rounded-lg p-3 border border-[var(--border-subtle)]">
          <span className="text-xs text-[var(--text-tertiary)]">Working Title</span>
          <h3 className="text-sm font-semibold text-[var(--text-primary)] mt-0.5">
            {canvas.working_title}
          </h3>
        </div>
      )}

      {/* Outline Tree */}
      <div className="space-y-1.5">
        {sortedOutline.map((section) => {
          const draft = drafts[section.id];
          const status = statusConfig[section.status] || statusConfig.todo;
          const hasGaps = section.guidance?.includes('gap') || section.guidance?.includes('Gap');
          const citationCount = draft?.used_citation_ids?.length || 0;
          const indent = (section.level - 1) * 16;

          return (
            <div
              key={section.id}
              className="group bg-[var(--bg-panel)] rounded-lg border border-[var(--border-subtle)] hover:border-[var(--border-highlight)] hover:shadow-sm transition-all"
              style={{ marginLeft: indent }}
            >
              <div className="flex items-start gap-2 p-3">
                {/* Drag Handle (视觉占位，未来可启用拖拽) */}
                <div className="pt-0.5 text-[var(--text-tertiary)] opacity-0 group-hover:opacity-50 transition-opacity cursor-grab">
                  <GripVertical size={14} />
                </div>

                {/* Status Icon */}
                <span className="pt-0.5 text-sm shrink-0" title={status.label}>
                  {status.icon}
                </span>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium text-[var(--text-primary)] truncate">
                      {section.title}
                    </h4>
                    {section.level > 1 && (
                      <span className="text-[10px] text-[var(--text-tertiary)] bg-[var(--bg-muted)] px-1.5 py-0.5 rounded">
                        L{section.level}
                      </span>
                    )}
                  </div>

                  {/* Guidance */}
                  {section.guidance && (
                    <p className="text-xs text-[var(--text-tertiary)] mt-0.5 line-clamp-2">
                      {section.guidance}
                    </p>
                  )}

                  {/* Meta */}
                  <div className="flex items-center gap-3 mt-1.5">
                    {citationCount > 0 && (
                      <span className="flex items-center gap-1 text-[10px] text-teal-600">
                        <BookOpen size={10} />
                        {citationCount} citations
                      </span>
                    )}
                    {hasGaps && (
                      <span className="flex items-center gap-1 text-[10px] text-amber-600">
                        <AlertTriangle size={10} />
                        有信息缺口
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Global Gaps */}
      {identified_gaps.length > 0 && (
        <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-3">
          <h4 className="text-xs font-semibold text-amber-200 mb-2 flex items-center gap-1.5">
            <AlertTriangle size={12} />
            全局信息缺口 ({identified_gaps.length})
          </h4>
          <ul className="space-y-1">
            {identified_gaps.map((gap, i) => (
              <li key={i} className="text-xs text-amber-300 flex items-start gap-1.5">
                <span className="shrink-0 mt-0.5">❗</span>
                <span>{gap}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
