import { useRef, useState, useCallback, useEffect } from 'react';
import {
  Eye,
  Pencil,
  Undo2,
  Redo2,
  MessageSquarePlus,
  Send,
  Trash2,
  Plus,
  X,
  CheckCircle2,
  XCircle,
  Lightbulb,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useCanvasStore, useToastStore } from '../../stores';
import { useChatStore } from '../../stores';
import {
  aiEditCanvas,
  createSnapshot,
  exportCanvas,
  getCanvas,
  listCanvasSnapshots,
  refineCanvasFull,
  restoreSnapshot,
  updateCanvas,
} from '../../api/canvas';
import { listInsights, updateInsightStatus } from '../../api/chat';
import { restartDeepResearchPhase } from '../../api/chat';
import type { Canvas, Annotation, ResearchInsight, InsightType, InsightStatus } from '../../types';
import {
  DEEP_RESEARCH_JOB_KEY,
  DEEP_RESEARCH_ARCHIVED_JOBS_KEY,
} from '../workflow/deep-research/types';

interface RefineStageProps {
  canvas: Canvas;
}

interface LockedRange {
  id: string;
  start: number;
  end: number;
  text: string;
}

/**
 * Refine 阶段：全文预览 + 行内批注 + 全局指令 + AI 编辑
 */
export function RefineStage({ canvas }: RefineStageProps) {
  const {
    setCanvas,
    canvasContent,
    editMode,
    isAIEditing,
    versionHistory,
    currentVersionIndex,
    pendingAnnotations,
    setCanvasContent,
    setEditMode,
    pushVersion,
    undo,
    redo,
    setIsAIEditing,
    addAnnotation,
    removeAnnotation,
    addDirective,
    removeDirective,
  } = useCanvasStore();
  const addToast = useToastStore((s) => s.addToast);
  const setShowDeepResearchDialog = useChatStore((s) => s.setShowDeepResearchDialog);
  const setDeepResearchTopic = useChatStore((s) => s.setDeepResearchTopic);
  const setDeepResearchActive = useChatStore((s) => s.setDeepResearchActive);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [selection, setSelection] = useState<{
    text: string;
    start: number;
    end: number;
  } | null>(null);

  // 批注输入
  const [showAnnotationInput, setShowAnnotationInput] = useState(false);
  const [annotationText, setAnnotationText] = useState('');
  const [annotationMode, setAnnotationMode] = useState<'annotation' | 'targeted_refine'>('annotation');

  // 全局指令输入
  const [newDirective, setNewDirective] = useState('');
  const [showDirectives, setShowDirectives] = useState(true);
  const [fullRefineDirective, setFullRefineDirective] = useState('');
  const [isFullRefining, setIsFullRefining] = useState(false);
  const [snapshotVersions, setSnapshotVersions] = useState<Array<{ version_number: number; created_at: string }>>([]);
  const [isLoadingSnapshots, setIsLoadingSnapshots] = useState(false);
  const [lockedRanges, setLockedRanges] = useState<LockedRange[]>([]);
  const [activeLockedId, setActiveLockedId] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);

  // Research Insights
  const [showInsights, setShowInsights] = useState(false);
  const [insights, setInsights] = useState<ResearchInsight[]>([]);
  const activeJobId = typeof window !== 'undefined' ? localStorage.getItem('deep_research_active_job_id') || '' : '';

  useEffect(() => {
    if (!activeJobId) {
      setInsights([]);
      return;
    }
    let cancelled = false;
    const fetchInsights = async () => {
      try {
        const data = await listInsights(activeJobId);
        if (!cancelled) setInsights(data);
      } catch {
        // ignore
      }
    };
    fetchInsights();
    const timer = window.setInterval(fetchInsights, 15000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activeJobId]);

  const handleDeferInsight = async (insightId: number) => {
    if (!activeJobId) return;
    try {
      await updateInsightStatus(activeJobId, insightId, 'deferred');
      setInsights((prev) => prev.map((i) => i.id === insightId ? { ...i, status: 'deferred' as InsightStatus } : i));
      addToast('已标记为暂缓处理', 'success');
    } catch {
      addToast('操作失败', 'error');
    }
  };

  const insightTypeLabel = (t: InsightType): string => {
    switch (t) {
      case 'gap': return '信息缺口';
      case 'conflict': return '矛盾冲突';
      case 'limitation': return '不足/限制';
      case 'future_direction': return '未来方向';
      default: return t;
    }
  };

  const insightTypeColor = (t: InsightType): string => {
    switch (t) {
      case 'gap': return 'text-red-700 bg-red-50 border-red-200';
      case 'conflict': return 'text-amber-700 bg-amber-50 border-amber-200';
      case 'limitation': return 'text-purple-700 bg-purple-50 border-purple-200';
      case 'future_direction': return 'text-blue-700 bg-blue-50 border-blue-200';
      default: return 'text-slate-200 bg-slate-800/80 border-slate-600/60';
    }
  };

  const insightStatusLabel = (s: InsightStatus): string => {
    switch (s) {
      case 'open': return '待处理';
      case 'addressed': return '已处理';
      case 'deferred': return '暂缓';
      default: return s;
    }
  };

  const insightStatusColor = (s: InsightStatus): string => {
    switch (s) {
      case 'open': return 'text-amber-700 bg-amber-50';
      case 'addressed': return 'text-emerald-700 bg-emerald-50';
      case 'deferred': return 'text-slate-300 bg-slate-700/70';
      default: return '';
    }
  };

  // Group insights by type
  const groupedInsights = insights.reduce<Record<InsightType, ResearchInsight[]>>((acc, ins) => {
    const key = ins.insight_type as InsightType;
    if (!acc[key]) acc[key] = [];
    acc[key].push(ins);
    return acc;
  }, {} as Record<InsightType, ResearchInsight[]>);

  const openInsightsCount = insights.filter((i) => i.status === 'open').length;

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

  const handleRestartSynthesize = async () => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('未找到可重启任务，请先完成一次 Deep Research', 'warning');
      return;
    }
    setRestarting(true);
    try {
      const resp = await restartDeepResearchPhase(sourceJobId, { phase: 'synthesize' });
      localStorage.setItem(DEEP_RESEARCH_JOB_KEY, resp.job_id);
      setDeepResearchTopic(canvas.topic || canvas.working_title || 'Deep Research');
      setDeepResearchActive(true);
      setShowDeepResearchDialog(true);
      addToast('已提交综合重启', 'success');
    } catch (err) {
      console.error('[RefineStage] restart synthesize failed:', err);
      addToast('综合重启失败，请重试', 'error');
    } finally {
      setRestarting(false);
    }
  };

  const loadSnapshots = useCallback(async () => {
    if (!canvas?.id) return;
    setIsLoadingSnapshots(true);
    try {
      const rows = await listCanvasSnapshots(canvas.id, 30);
      setSnapshotVersions(rows);
    } catch {
      // ignore
    } finally {
      setIsLoadingSnapshots(false);
    }
  }, [canvas?.id]);

  useEffect(() => {
    loadSnapshots();
  }, [loadSnapshots]);

  // 检测选中文本
  const handleSelect = useCallback(() => {
    if (!editMode) return;
    const ta = textareaRef.current;
    if (!ta) return;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    if (start === end) {
      setSelection(null);
      setShowAnnotationInput(false);
      return;
    }
    const selectedText = canvasContent.slice(start, end);
    setSelection({ text: selectedText, start, end });
    if (showAnnotationInput) setShowAnnotationInput(false);
  }, [editMode, canvasContent, showAnnotationInput]);

  // 点击非选区隐藏工具栏
  useEffect(() => {
    const handleClickOutside = () => {
      setTimeout(() => {
        const ta = textareaRef.current;
        if (!ta || ta.selectionStart === ta.selectionEnd) {
          setSelection(null);
        }
      }, 200);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggleEdit = () => {
    if (!editMode) pushVersion();
    setEditMode(!editMode);
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCanvasContent(e.target.value);
  };

  // 提交行内批注
  const handleSubmitAnnotation = () => {
    if (!annotationText.trim() || !selection) return;
    const annotation: Annotation = {
      id: Date.now().toString(36),
      section_id: '',
      target_text: selection.text.slice(0, 200),
      directive: annotationText.trim(),
      status: 'pending',
      created_at: new Date().toISOString(),
    };
    addAnnotation(annotation);
    setAnnotationText('');
    setShowAnnotationInput(false);
    setSelection(null);
    addToast('批注已添加', 'success');
  };

  const handleTargetedRefine = async () => {
    if (!annotationText.trim() || !selection || !canvas) return;
    setIsAIEditing(true);
    pushVersion();
    try {
      const result = await aiEditCanvas(canvas.id, {
        section_text: selection.text,
        action: 'targeted_refine',
        directive: annotationText.trim(),
        context: canvasContent.slice(
          Math.max(0, selection.start - 200),
          Math.min(canvasContent.length, selection.end + 200)
        ),
        search_mode: 'none',
        preserve_citations: true,
      });
      const newContent =
        canvasContent.slice(0, selection.start) +
        result.edited_text +
        canvasContent.slice(selection.end);
      const updatedStart = selection.start;
      const updatedEnd = selection.start + result.edited_text.length;
      setCanvasContent(newContent);
      if (result.citation_guard_triggered) {
        addToast(result.citation_guard_message || '检测到引用丢失，已自动回退该次局部精炼', 'warning');
      } else {
        addToast('已完成局部定向精炼（仅修改选区）', 'success');
      }
      setAnnotationText('');
      setShowAnnotationInput(false);
      setSelection({
        text: result.edited_text,
        start: updatedStart,
        end: updatedEnd,
      });
      setAnnotationMode('annotation');
      // Keep the modified region selected so users can see the changed scope.
      setTimeout(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        ta.focus();
        ta.setSelectionRange(updatedStart, updatedEnd);
      }, 0);
    } catch (err: unknown) {
      const msg = (err as any)?.response?.data?.detail
        || (err as any)?.message
        || (err instanceof Error ? err.message : String(err));
      addToast(`定向精炼失败: ${msg}`, 'error');
    } finally {
      setIsAIEditing(false);
    }
  };

  // 提交全局指令
  const handleAddDirective = () => {
    if (!newDirective.trim()) return;
    addDirective(newDirective.trim());
    setNewDirective('');
    addToast('修改指令已添加', 'success');
  };

  const canUndo = versionHistory.length > 0 && currentVersionIndex > 0;
  const canRedo = currentVersionIndex >= 0 && currentVersionIndex < versionHistory.length - 1;

  const handleLockSelection = () => {
    if (!selection) return;
    if (!selection.text.trim()) {
      addToast('空白选区无法锁定', 'warning');
      return;
    }
    const duplicate = lockedRanges.some((r) => r.start === selection.start && r.end === selection.end);
    if (duplicate) {
      addToast('该选区已锁定', 'info');
      return;
    }
    const lockId = `${selection.start}-${selection.end}-${Date.now()}`;
    setLockedRanges((prev) => [
      ...prev,
      {
        id: lockId,
        start: selection.start,
        end: selection.end,
        text: selection.text,
      },
    ]);
    setActiveLockedId(lockId);
    addToast('已锁定当前选区（全文重整时不会改动）', 'success');
    setSelection(null);
    setShowAnnotationInput(false);
  };

  const handleUnlockRange = (id: string) => {
    setLockedRanges((prev) => prev.filter((r) => r.id !== id));
    if (activeLockedId === id) {
      setActiveLockedId(null);
    }
  };

  const handleClearLocks = () => {
    setLockedRanges([]);
    setActiveLockedId(null);
  };

  const resolveLockedRange = useCallback((lock: LockedRange) => {
    if (
      lock.start >= 0 &&
      lock.end > lock.start &&
      lock.end <= canvasContent.length &&
      canvasContent.slice(lock.start, lock.end) === lock.text
    ) {
      return { start: lock.start, end: lock.end, text: lock.text };
    }
    const idx = canvasContent.indexOf(lock.text);
    if (idx >= 0) {
      return { start: idx, end: idx + lock.text.length, text: lock.text };
    }
    return null;
  }, [canvasContent]);

  const handleFocusLockedRange = (lock: LockedRange) => {
    const located = resolveLockedRange(lock);
    if (!located) {
      addToast('未能在当前文本中定位该锁定片段（可能已被改动）', 'warning');
      return;
    }
    // If the content changed, refresh stored range so next full-refine can still apply lock.
    if (located.start !== lock.start || located.end !== lock.end) {
      setLockedRanges((prev) => prev.map((r) => (
        r.id === lock.id ? { ...r, start: located.start, end: located.end } : r
      )));
    }
    setActiveLockedId(lock.id);
    if (!editMode) setEditMode(true);
    setShowAnnotationInput(false);
    setTimeout(() => {
      const ta = textareaRef.current;
      if (!ta) return;
      ta.focus();
      ta.setSelectionRange(located.start, located.end);
      const line = canvasContent.slice(0, located.start).split('\n').length - 1;
      const lineHeight = 20;
      ta.scrollTop = Math.max(0, line * lineHeight - ta.clientHeight * 0.3);
      setSelection({
        text: located.text,
        start: located.start,
        end: located.end,
      });
    }, 0);
  };

  const handleRefineFull = async () => {
    if (!canvas?.id || !canvasContent.trim()) return;
    setIsFullRefining(true);
    setIsAIEditing(true);
    pushVersion();
    try {
      const extraDirectives = [
        ...(canvas.user_directives || []),
        ...pendingAnnotations
          .filter((a) => a.status === 'pending')
          .map((a) => a.directive.trim())
          .filter(Boolean),
      ];
      if (fullRefineDirective.trim()) extraDirectives.push(fullRefineDirective.trim());

      const validLocks: Array<{ start: number; end: number; text: string }> = [];
      let skippedLocks = 0;
      for (const lock of lockedRanges) {
        if (lock.start < 0 || lock.end <= lock.start || lock.end > canvasContent.length) {
          skippedLocks += 1;
          continue;
        }
        if (canvasContent.slice(lock.start, lock.end) !== lock.text) {
          skippedLocks += 1;
          continue;
        }
        validLocks.push({ start: lock.start, end: lock.end, text: lock.text });
      }

      const result = await refineCanvasFull(canvas.id, {
        content_md: canvasContent,
        directives: extraDirectives,
        save_snapshot_before: true,
        locked_ranges: validLocks,
      });
      setCanvasContent(result.edited_markdown || canvasContent);
      if (result.lock_guard_triggered) {
        addToast(result.lock_guard_message || '锁定保护触发，已回退到重整前文本', 'warning');
      } else if (result.snapshot_version) {
        addToast(`已完成重新精炼（可回退到快照 v${result.snapshot_version}）`, 'success');
      } else {
        addToast('已完成重新精炼', 'success');
      }
      const skippedTotal = skippedLocks + (result.locked_skipped || 0);
      if ((result.locked_applied || 0) > 0) {
        addToast(`锁定保护生效：${result.locked_applied} 个片段`, 'info');
      }
      if (skippedTotal > 0) {
        addToast(`有 ${skippedTotal} 个锁定片段失效并被跳过（通常因文本位置变化）`, 'warning');
      }
      setFullRefineDirective('');
      try {
        const fresh = await getCanvas(canvas.id);
        setCanvas(fresh);
      } catch {
        // ignore
      }
      await loadSnapshots();
    } catch (err: unknown) {
      const msg = (err as any)?.response?.data?.detail
        || (err as any)?.message
        || (err instanceof Error ? err.message : String(err));
      addToast(`重新精炼失败: ${msg}`, 'error');
    } finally {
      setIsFullRefining(false);
      setIsAIEditing(false);
    }
  };

  const handleSaveSnapshot = async () => {
    if (!canvas?.id) return;
    try {
      await updateCanvas(canvas.id, {
        refined_markdown: canvasContent,
        stage: 'refine',
      });
      const ret = await createSnapshot(canvas.id);
      addToast(`已保存快照 v${ret.version_number}`, 'success');
      await loadSnapshots();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`保存快照失败: ${msg}`, 'error');
    }
  };

  const handleRestoreFromSnapshot = async (versionNumber: number) => {
    if (!canvas?.id) return;
    setIsAIEditing(true);
    try {
      await restoreSnapshot(canvas.id, versionNumber);
      const [freshCanvas, md] = await Promise.all([
        getCanvas(canvas.id),
        exportCanvas(canvas.id, 'markdown'),
      ]);
      setCanvas(freshCanvas);
      setCanvasContent(md.content || freshCanvas.refined_markdown || '');
      setLockedRanges([]);
      pushVersion();
      addToast(`已回退到快照 v${versionNumber}`, 'success');
      await loadSnapshots();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`回退失败: ${msg}`, 'error');
    } finally {
      setIsAIEditing(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border-subtle)] bg-[var(--bg-surface)]">
        <div className="flex items-center gap-1">
          <button
            onClick={handleToggleEdit}
            className={`p-1.5 rounded text-xs flex items-center gap-1 transition-colors cursor-pointer ${
              editMode
                ? 'bg-blue-100 text-blue-600'
                : 'hover:bg-[var(--bg-muted)] text-[var(--text-tertiary)]'
            }`}
            title={editMode ? '切换预览' : '切换编辑'}
          >
            {editMode ? <Eye size={14} /> : <Pencil size={14} />}
            <span>{editMode ? '预览' : '编辑'}</span>
          </button>

          {editMode && (
            <>
              <button
                onClick={undo}
                disabled={!canUndo}
                className="p-1.5 hover:bg-[var(--bg-muted)] rounded text-[var(--text-tertiary)] disabled:opacity-30 cursor-pointer"
                title="撤销"
              >
                <Undo2 size={14} />
              </button>
              <button
                onClick={redo}
                disabled={!canRedo}
                className="p-1.5 hover:bg-[var(--bg-muted)] rounded text-[var(--text-tertiary)] disabled:opacity-30 cursor-pointer"
                title="重做"
              >
                <Redo2 size={14} />
              </button>
            </>
          )}
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleRestartSynthesize}
            disabled={restarting}
            className="px-2 py-1.5 rounded text-xs border border-indigo-300 text-indigo-700 hover:bg-indigo-50 disabled:opacity-50 cursor-pointer"
            title="重新执行最终综合（synthesize）"
          >
            {restarting ? '提交中...' : '重新综合'}
          </button>
          <button
            onClick={handleRefineFull}
            disabled={isFullRefining || isAIEditing || !canvasContent.trim()}
            className="px-2 py-1.5 rounded text-xs bg-violet-100 text-violet-700 hover:bg-violet-200 disabled:opacity-50 cursor-pointer"
            title="基于当前全文进行全局重整（慎用，可能改动范围较大）"
          >
            {isFullRefining ? '重整中...' : '全文重整(慎用)'}
          </button>
          <button
            onClick={handleLockSelection}
            disabled={!editMode || !selection}
            className="px-2 py-1.5 rounded text-xs bg-emerald-100 text-emerald-700 hover:bg-emerald-200 disabled:opacity-50 cursor-pointer"
            title="锁定当前选区，全文重整时保持不变"
          >
            锁定选区
          </button>
          <button
            onClick={() => {
              if (!editMode || !selection) return;
              setAnnotationMode('targeted_refine');
              setShowAnnotationInput(true);
            }}
            disabled={!editMode || !selection}
            className="px-2 py-1.5 rounded text-xs bg-sky-100 text-sky-700 hover:bg-sky-200 disabled:opacity-50 cursor-pointer"
            title="对当前选区执行定向精炼"
          >
            定向精炼选区
          </button>
          <button
            onClick={handleSaveSnapshot}
            disabled={!canvasContent.trim()}
            className="px-2 py-1.5 rounded text-xs bg-emerald-100 text-emerald-700 hover:bg-emerald-200 disabled:opacity-50 cursor-pointer"
            title="保存当前全文快照，便于多次回退"
          >
            保存快照
          </button>
          <button
            onClick={() => setShowDirectives((v) => !v)}
            className={`p-1.5 rounded text-xs flex items-center gap-1 cursor-pointer transition-colors ${
              showDirectives
                ? 'bg-indigo-100 text-indigo-600'
                : 'hover:bg-[var(--bg-muted)] text-[var(--text-tertiary)]'
            }`}
            title="修改指令面板"
          >
            <MessageSquarePlus size={14} />
            <span>指令</span>
            {(canvas.user_directives.length + pendingAnnotations.length) > 0 && (
              <span className="bg-indigo-500 text-white text-[9px] px-1 rounded-full">
                {canvas.user_directives.length + pendingAnnotations.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {editMode && activeLockedId && (
        <div className="px-3 py-1.5 text-[10px] text-emerald-300 bg-emerald-900/20 border-b border-emerald-700/30">
          已高亮一个锁定片段。你可以直接执行“定向精炼”，或先解锁后再改写。
        </div>
      )}
      {editMode && !selection && (
        <div className="px-3 py-1 text-[10px] text-[var(--text-tertiary)] bg-[var(--bg-muted)] border-b border-[var(--border-subtle)]">
          提示：先在正文里选中文字，再点击上方“锁定选区”或“定向精炼选区”。
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto bg-[var(--bg-panel)]">
        {canvasContent ? (
          editMode ? (
            <textarea
              ref={textareaRef}
              value={canvasContent}
              onChange={handleContentChange}
              onSelect={handleSelect}
              onMouseUp={handleSelect}
              onKeyUp={handleSelect}
                className="refine-editor w-full h-full p-6 resize-none font-mono text-sm text-[var(--text-primary)] leading-relaxed focus:outline-none bg-[var(--bg-panel)]"
              placeholder="在此编辑 Markdown..."
              spellCheck={false}
            />
          ) : (
            <div className="p-6">
              <div className="prose prose-sm prose-invert max-w-none prose-headings:font-bold prose-h1:text-xl prose-h2:text-lg prose-p:text-[var(--text-secondary)]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {canvasContent}
                </ReactMarkdown>
              </div>
            </div>
          )
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-[var(--text-tertiary)] text-xs">
            <Pencil size={32} className="mb-2 opacity-20" />
            <p>撰写完成后，全文将显示在此处供精炼</p>
          </div>
        )}
      </div>

      {/* Annotation Input (弹出在选中文本时) */}
      {editMode && showAnnotationInput && selection && (
        <div className="border-t border-[var(--border-subtle)] bg-amber-50 p-3">
          <div className="text-xs text-amber-700 mb-1.5 truncate">
            选中: "{selection.text.slice(0, 80)}{selection.text.length > 80 ? '...' : ''}"
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={annotationText}
              onChange={(e) => setAnnotationText(e.target.value)}
              placeholder={annotationMode === 'targeted_refine' ? '输入选区定向精炼指令...' : '输入批注意见...'}
              className="flex-1 border border-amber-500/30 rounded-md px-2.5 py-1.5 text-sm focus:ring-2 focus:ring-amber-400 outline-none bg-slate-900/80 text-slate-100"
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return;
                if (annotationMode === 'targeted_refine') {
                  handleTargetedRefine();
                } else {
                  handleSubmitAnnotation();
                }
              }}
              autoFocus
            />
            <button
              onClick={annotationMode === 'targeted_refine' ? handleTargetedRefine : handleSubmitAnnotation}
              disabled={!annotationText.trim()}
              className="px-3 py-1.5 bg-amber-500 text-white rounded-md text-xs font-medium hover:bg-amber-600 disabled:opacity-50 flex items-center gap-1"
            >
              <span>{annotationMode === 'targeted_refine' ? '执行定向精炼' : '添加批注'}</span>
              <Send size={12} />
            </button>
            <button
              onClick={() => { setShowAnnotationInput(false); setAnnotationText(''); setAnnotationMode('annotation'); }}
              className="p-1.5 text-amber-500 hover:bg-amber-100 rounded"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Directives & Annotations Panel */}
      {showDirectives && (
        <div className="border-t border-[var(--border-subtle)] bg-[var(--bg-surface)] max-h-64 overflow-y-auto">
          {/* Pending Annotations */}
          {pendingAnnotations.length > 0 && (
            <div className="p-3 border-b border-[var(--border-subtle)]">
              <h5 className="text-xs font-semibold text-[var(--text-primary)] mb-2 flex items-center gap-1.5">
                <MessageSquarePlus size={12} />
                行内批注 ({pendingAnnotations.length})
              </h5>
              <div className="space-y-1.5">
                {pendingAnnotations.map((ann) => (
                  <div
                    key={ann.id}
                    className={`flex items-start gap-2 text-xs p-2 rounded-md border ${
                      ann.status === 'applied'
                        ? 'bg-emerald-50 border-emerald-200'
                        : ann.status === 'rejected'
                          ? 'bg-red-50 border-red-200 opacity-60'
                          : 'bg-[var(--bg-muted)] border-[var(--border-subtle)]'
                    }`}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-[var(--text-tertiary)] truncate mb-0.5">
                        "{ann.target_text.slice(0, 60)}{ann.target_text.length > 60 ? '...' : ''}"
                      </div>
                      <div className="text-[var(--text-secondary)]">{ann.directive}</div>
                    </div>
                    <div className="flex items-center gap-0.5 shrink-0">
                      {ann.status === 'pending' && (
                        <>
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400" title="待处理" />
                          <button
                            onClick={() => removeAnnotation(ann.id)}
                            className="p-0.5 text-[var(--text-tertiary)] hover:text-red-500"
                          >
                            <Trash2 size={11} />
                          </button>
                        </>
                      )}
                      {ann.status === 'applied' && <CheckCircle2 size={12} className="text-emerald-500" />}
                      {ann.status === 'rejected' && <XCircle size={12} className="text-red-400" />}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Global Directives */}
          <div className="p-3">
            <h5 className="text-xs font-semibold text-[var(--text-primary)] mb-2">
              全局修改指令
            </h5>
            <div className="text-[10px] text-[var(--text-tertiary)] mb-2">
              建议优先使用“选区定向精炼”；全文重整可能影响未选中内容。
            </div>
            <div className="mb-2 p-2 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-muted)]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-semibold text-[var(--text-primary)]">锁定片段</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--text-tertiary)]">{lockedRanges.length} 个</span>
                  <button
                    onClick={handleClearLocks}
                    disabled={lockedRanges.length === 0}
                    className="px-1.5 py-0.5 text-[10px] text-slate-300 bg-slate-700/70 rounded disabled:opacity-50 cursor-pointer"
                  >
                    清空
                  </button>
                </div>
              </div>
              {lockedRanges.length === 0 ? (
                <div className="text-[10px] text-[var(--text-tertiary)]">在编辑区选中文字后，点击顶部“锁定选区”。</div>
              ) : (
                <div className="space-y-1 max-h-20 overflow-y-auto">
                  {lockedRanges.map((r) => (
                    <div
                      key={r.id}
                      className={`flex items-center gap-2 text-[10px] rounded px-1 py-0.5 ${
                        activeLockedId === r.id ? 'bg-emerald-900/25 ring-1 ring-emerald-700/40' : ''
                      }`}
                    >
                      <span className="flex-1 truncate text-[var(--text-secondary)]">
                        {r.text.replace(/\s+/g, ' ').slice(0, 60)}{r.text.length > 60 ? '...' : ''}
                      </span>
                      {activeLockedId === r.id && (
                        <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-800/60 text-emerald-200">高亮中</span>
                      )}
                      <button
                        onClick={() => handleFocusLockedRange(r)}
                        className="px-1.5 py-0.5 text-[10px] text-emerald-200 bg-emerald-900/40 rounded cursor-pointer"
                      >
                        定位
                      </button>
                      <button
                        onClick={() => handleUnlockRange(r.id)}
                        className="px-1.5 py-0.5 text-[10px] text-red-300 bg-red-900/30 rounded cursor-pointer"
                      >
                        解锁
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={fullRefineDirective}
                onChange={(e) => setFullRefineDirective(e.target.value)}
                placeholder="本轮重新精炼附加指令（可选）..."
                className="flex-1 border border-[var(--border-subtle)] rounded-md px-2.5 py-1.5 text-sm focus:ring-2 focus:ring-[var(--primary)] outline-none"
              />
              <button
                onClick={handleRefineFull}
                disabled={isFullRefining || isAIEditing || !canvasContent.trim()}
                className="px-2.5 py-1.5 bg-violet-500 text-white rounded-md text-xs hover:bg-violet-600 disabled:opacity-50"
              >
                全文重整
              </button>
            </div>
            {canvas.user_directives.length > 0 && (
              <div className="space-y-1 mb-2">
                {canvas.user_directives.map((d, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs bg-[var(--bg-muted)] border border-[var(--border-subtle)] rounded-md px-2.5 py-1.5">
                    <span className="text-indigo-500 shrink-0">{i + 1}.</span>
                    <span className="flex-1 text-[var(--text-secondary)]">{d}</span>
                    <button
                      onClick={() => removeDirective(i)}
                      className="p-0.5 text-[var(--text-tertiary)] hover:text-red-500"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={newDirective}
                onChange={(e) => setNewDirective(e.target.value)}
                placeholder="添加全局修改指令..."
                className="flex-1 border border-[var(--border-subtle)] rounded-md px-2.5 py-1.5 text-sm focus:ring-2 focus:ring-[var(--primary)] outline-none"
                onKeyDown={(e) => e.key === 'Enter' && handleAddDirective()}
              />
              <button
                onClick={handleAddDirective}
                disabled={!newDirective.trim()}
                className="px-2.5 py-1.5 bg-indigo-500 text-white rounded-md text-xs hover:bg-indigo-600 disabled:opacity-50 flex items-center gap-1"
              >
                <Plus size={12} />
                添加
              </button>
            </div>
            <div className="mt-3 border-t border-[var(--border-subtle)] pt-2">
              <h6 className="text-[11px] font-semibold text-[var(--text-primary)] mb-1.5">回退快照</h6>
              {isLoadingSnapshots ? (
                <div className="text-xs text-[var(--text-tertiary)]">加载中...</div>
              ) : snapshotVersions.length === 0 ? (
                <div className="text-xs text-[var(--text-tertiary)]">暂无快照，先点击“保存快照”或执行“重新精炼”。</div>
              ) : (
                <div className="space-y-1.5 max-h-28 overflow-y-auto">
                  {snapshotVersions.map((v) => (
                    <div key={`snap-${v.version_number}`} className="flex items-center justify-between text-xs bg-[var(--bg-muted)] border border-[var(--border-subtle)] rounded-md px-2 py-1.5">
                      <span className="text-[var(--text-secondary)]">
                        v{v.version_number} · {v.created_at.replace('T', ' ').slice(0, 19)}
                      </span>
                      <button
                        onClick={() => handleRestoreFromSnapshot(v.version_number)}
                        className="px-1.5 py-0.5 text-[10px] text-emerald-700 bg-emerald-100 rounded hover:bg-emerald-200 cursor-pointer"
                      >
                        回退
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Version Indicator */}
      {editMode && versionHistory.length > 0 && (
        <div className="h-6 border-t border-[var(--border-subtle)] bg-[var(--bg-muted)] flex items-center justify-center text-[10px] text-[var(--text-tertiary)]">
          版本 {currentVersionIndex >= 0 ? currentVersionIndex + 1 : versionHistory.length}/{versionHistory.length}
        </div>
      )}

      {/* Research Insights Ledger */}
      {(insights.length > 0 || (canvas.research_insights?.length ?? 0) > 0) && (
        <div className="border-t border-[var(--border-subtle)]">
          <button
            onClick={() => setShowInsights((v) => !v)}
            className="w-full flex items-center justify-between px-3 py-2 bg-[var(--bg-surface)] hover:bg-[var(--bg-muted)] transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[var(--text-primary)]">
              <Lightbulb size={13} className="text-amber-500" />
              研究洞察
              {openInsightsCount > 0 && (
                <span className="bg-amber-500 text-white text-[9px] px-1.5 rounded-full">
                  {openInsightsCount}
                </span>
              )}
            </div>
            {showInsights ? <ChevronUp size={14} className="text-[var(--text-tertiary)]" /> : <ChevronDown size={14} className="text-[var(--text-tertiary)]" />}
          </button>
          {showInsights && (
            <div className="max-h-64 overflow-y-auto px-3 py-2 space-y-3 bg-[var(--bg-surface)]">
              {(Object.entries(groupedInsights) as [InsightType, ResearchInsight[]][]).map(([type, items]) => (
                <div key={type}>
                  <div className="text-[10px] font-semibold text-[var(--text-tertiary)] uppercase tracking-wider mb-1">
                    {insightTypeLabel(type)} ({items.length})
                  </div>
                  <div className="space-y-1">
                    {items.map((ins) => (
                      <div
                        key={ins.id}
                        className={`flex items-start gap-2 text-xs p-2 rounded-md border ${insightTypeColor(type)} ${ins.status === 'deferred' ? 'opacity-50' : ''}`}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-[var(--text-secondary)] leading-relaxed">{ins.text}</div>
                          {ins.section_id && (
                            <div className="text-[10px] text-[var(--text-tertiary)] mt-0.5">
                              章节: {ins.section_id}
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <span className={`text-[9px] px-1.5 py-0.5 rounded ${insightStatusColor(ins.status)}`}>
                            {insightStatusLabel(ins.status)}
                          </span>
                          {ins.status === 'open' && (
                            <button
                              onClick={() => handleDeferInsight(ins.id)}
                              className="text-[10px] text-slate-300 hover:text-slate-100 px-1 py-0.5 rounded hover:bg-slate-700/60 cursor-pointer"
                              title="标记为暂缓处理"
                            >
                              暂缓
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {/* Also show canvas-level research_insights if any */}
              {canvas.research_insights && canvas.research_insights.length > 0 && !insights.length && (
                <div>
                  <div className="text-[10px] font-semibold text-[var(--text-tertiary)] mb-1">来自画布</div>
                  {canvas.research_insights.map((text, i) => (
                    <div key={i} className="text-xs text-[var(--text-secondary)] p-1.5 border border-[var(--border-subtle)] rounded-md mb-1">
                      {text}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
