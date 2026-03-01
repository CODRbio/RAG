import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { GripVertical, ChevronRight, AlertTriangle, BookOpen, RotateCcw, Pencil, X, Plus } from 'lucide-react';
import type { Canvas } from '../../types';
import { restartDeepResearchPhase, restartDeepResearchWithOutline } from '../../api/chat';
import { useChatStore, useToastStore, useUIStore } from '../../stores';
import {
  DEEP_RESEARCH_JOB_KEY,
  DEEP_RESEARCH_ARCHIVED_JOBS_KEY,
} from '../workflow/deep-research/types';

interface OutlineStageProps {
  canvas: Canvas;
}

/**
 * Outline 阶段：可视化的章节树 + Gap 标注
 * 展示大纲结构、每个章节的状态和关联信息
 */
export function OutlineStage({ canvas }: OutlineStageProps) {
  const { t } = useTranslation();
  const addToast = useToastStore((s) => s.addToast);
  const { sessionId, setDeepResearchTopic, setShowDeepResearchDialog, setDeepResearchActive, setSessionId, setCanvasId } = useChatStore();
  const { requestSessionListRefresh } = useUIStore();
  const [restarting, setRestarting] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [localTitles, setLocalTitles] = useState<string[]>([]);
  const { outline, drafts, identified_gaps } = canvas;

  const sortedOutline = [...(outline || [])].sort((a, b) => a.order - b.order);

  useEffect(() => {
    if (!isEditMode && sortedOutline.length > 0) {
      setLocalTitles(sortedOutline.map((s) => s.title));
    }
  }, [isEditMode, sortedOutline.length]);

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

  const resolveSourceJobId = (): string | null => {
    const active = localStorage.getItem(DEEP_RESEARCH_JOB_KEY);
    if (active && active.trim()) return active.trim();
    try {
      const raw = localStorage.getItem(DEEP_RESEARCH_ARCHIVED_JOBS_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'string') {
        return parsed[0];
      }
    } catch {
      // noop
    }
    return null;
  };

  const afterRestartSubmitted = (jobId: string, sessionId?: string, canvasIdFromResp?: string) => {
    if (sessionId) setSessionId(sessionId);
    if (canvasIdFromResp) setCanvasId(canvasIdFromResp);
    localStorage.setItem(DEEP_RESEARCH_JOB_KEY, jobId);
    setDeepResearchTopic(canvas.topic || canvas.working_title || '');
    setDeepResearchActive(true);
    setShowDeepResearchDialog(true);
    requestSessionListRefresh();
  };

  const handleRestartPhase = async (
    phase: 'plan' | 'research' | 'generate_claims' | 'write' | 'verify' | 'review_gate' | 'synthesize',
  ) => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('未找到可重启任务，请先完成一次 Deep Research', 'warning');
      return;
    }
    setRestarting(true);
    try {
      const resp = await restartDeepResearchPhase(sourceJobId, { phase });
      afterRestartSubmitted(resp.job_id, resp.session_id, resp.canvas_id);
      addToast(`已提交阶段重启：${phase}`, 'success');
    } catch (err) {
      console.error('[OutlineStage] restart phase failed:', err);
      addToast('阶段重启失败，请重试', 'error');
    } finally {
      setRestarting(false);
    }
  };

  const handleRestartWithOutline = async () => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('未找到可重启任务，请先完成一次 Deep Research', 'warning');
      return;
    }
    const titles = localTitles.map((t) => t.trim()).filter(Boolean);
    if (titles.length === 0) {
      addToast('请至少保留一个章节标题', 'error');
      return;
    }
    setRestarting(true);
    try {
      const resp = await restartDeepResearchWithOutline(
        sourceJobId,
        { new_outline: titles, action: 'research' },
        sessionId ?? undefined,
      );
      afterRestartSubmitted(resp.job_id, resp.session_id, resp.canvas_id);
      addToast('已提交：以此大纲继续研究', 'success');
      setIsEditMode(false);
    } catch (err) {
      console.error('[OutlineStage] restart with outline failed:', err);
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      addToast(msg && typeof msg === 'string' ? msg : '提交失败，请从侧边栏「后台调研」进入该画布后再试', 'error');
    } finally {
      setRestarting(false);
    }
  };

  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      {/* Restart / Edit controls */}
      <div className="bg-[var(--bg-surface)] rounded-lg p-3 border border-[var(--border-subtle)]">
        <div className="flex items-center gap-2 mb-2">
          <RotateCcw size={13} className="text-indigo-500" />
          <span className="text-xs text-[var(--text-tertiary)] font-medium">
            {isEditMode ? t('research.editOutlineTitle', '编辑大纲') : t('research.restartStageTitle', '重启执行')}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {!isEditMode && (
            <>
              <button
                disabled={restarting}
                onClick={() => void handleRestartPhase('plan')}
                className="px-2 py-1 text-[10px] border rounded text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
              >
                {t('research.restartOutlineStage', '重新规划大纲')}
              </button>
              <button
                type="button"
                onClick={() => {
                  setLocalTitles(sortedOutline.map((s) => s.title));
                  setIsEditMode(true);
                }}
                className="px-2 py-1 text-[10px] border rounded text-gray-700 hover:bg-gray-100 flex items-center gap-1"
              >
                <Pencil size={10} />
                {t('research.editOutline', '编辑大纲')}
              </button>
            </>
          )}
          {isEditMode && (
            <>
              <button
                type="button"
                onClick={() => {
                  setLocalTitles(sortedOutline.map((s) => s.title));
                  setIsEditMode(false);
                }}
                className="px-2 py-1 text-[10px] border rounded text-gray-600 hover:bg-gray-100"
              >
                {t('common.cancel', '取消')}
              </button>
              <button
                disabled={restarting || localTitles.filter((t) => t.trim()).length === 0}
                onClick={() => void handleRestartWithOutline()}
                className="px-2 py-1 text-[10px] border rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {t('research.continueWithOutline', '以此大纲继续研究')}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Working Title */}
      {canvas.working_title && (
        <div className="bg-[var(--bg-surface)] rounded-lg p-3 border border-[var(--border-subtle)]">
          <span className="text-xs text-[var(--text-tertiary)]">Working Title</span>
          <h3 className="text-sm font-semibold text-[var(--text-primary)] mt-0.5">
            {canvas.working_title}
          </h3>
        </div>
      )}

      {/* Outline Tree: edit mode (inputs + add/remove) or read-only */}
      <div className="space-y-1.5">
        {isEditMode ? (
          <>
            {localTitles.map((title, idx) => (
              <div
                key={`edit-${idx}`}
                className="flex items-center gap-2 bg-[var(--bg-panel)] rounded-lg border border-[var(--border-subtle)] p-2"
              >
                <div className="pt-0.5 text-[var(--text-tertiary)]">
                  <GripVertical size={14} />
                </div>
                <input
                  type="text"
                  value={title}
                  onChange={(e) =>
                    setLocalTitles((prev) => prev.map((t, i) => (i === idx ? e.target.value : t)))
                  }
                  className="flex-1 min-w-0 text-sm font-medium text-[var(--text-primary)] bg-transparent border-0 focus:ring-1 focus:ring-indigo-500 rounded px-2 py-1"
                  placeholder="章节标题"
                />
                <button
                  type="button"
                  disabled={localTitles.length <= 1}
                  onClick={() => setLocalTitles((prev) => prev.filter((_, i) => i !== idx))}
                  className="p-1 text-red-500 hover:bg-red-500/10 rounded disabled:opacity-40 disabled:cursor-not-allowed"
                  title={t('research.removeSection', '删除章节')}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={() => setLocalTitles((prev) => [...prev, ''])}
              className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-indigo-600 hover:bg-indigo-50 rounded-lg border border-dashed border-indigo-300"
            >
              <Plus size={14} />
              {t('research.addSection', '添加章节')}
            </button>
          </>
        ) : (
          sortedOutline.map((section) => {
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
                  <div className="pt-0.5 text-[var(--text-tertiary)] opacity-0 group-hover:opacity-50 transition-opacity cursor-grab">
                    <GripVertical size={14} />
                  </div>
                  <span className="pt-0.5 text-sm shrink-0" title={status.label}>
                    {status.icon}
                  </span>
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
                    {section.guidance && (
                      <p className="text-xs text-[var(--text-tertiary)] mt-0.5 line-clamp-2">
                        {section.guidance}
                      </p>
                    )}
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
                      <div className="ml-auto" />
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
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
