import { FileText, BookOpen, AlertTriangle, ChevronDown, ChevronUp, Send, CheckCircle2, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { listSectionReviews, submitSectionReview, submitGapSupplement, listGapSupplements } from '../../api/chat';
import { useToastStore } from '../../stores';
import { useChatStore } from '../../stores';
import type { Canvas, GapSupplement } from '../../types';

interface DraftingStageProps {
  canvas: Canvas;
}

export function DraftingStage({ canvas }: DraftingStageProps) {
  const { outline, drafts, citation_pool, identified_gaps, skip_draft_review } = canvas;
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  const [reviewingSectionId, setReviewingSectionId] = useState<string | null>(null);
  const [reviewFeedback, setReviewFeedback] = useState<string>('');
  const [submittingFor, setSubmittingFor] = useState<string | null>(null);
  const [reviewBySection, setReviewBySection] = useState<Record<string, { action: string; feedback?: string; created_at?: number }>>({});
  const [gapSupplements, setGapSupplements] = useState<GapSupplement[]>([]);
  const [bulkApproving, setBulkApproving] = useState(false);
  const [showSupplementModal, setShowSupplementModal] = useState(false);
  const [supplementSectionTitle, setSupplementSectionTitle] = useState('');
  const [supplementGapText, setSupplementGapText] = useState('');
  const [supplementType, setSupplementType] = useState<'material' | 'direct_info'>('direct_info');
  const [supplementText, setSupplementText] = useState('');
  const [supplementSubmitting, setSupplementSubmitting] = useState(false);
  const addToast = useToastStore((s) => s.addToast);
  const researchDashboard = useChatStore((s) => s.researchDashboard);
  const setShowDeepResearchDialog = useChatStore((s) => s.setShowDeepResearchDialog);
  const setDeepResearchTopic = useChatStore((s) => s.setDeepResearchTopic);

  const sortedOutline = [...(outline || [])].sort((a, b) => a.order - b.order);
  const activeJobId = localStorage.getItem('deep_research_active_job_id') || '';

  const toggleSection = (id: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  if (!sortedOutline.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-[var(--text-tertiary)] text-sm px-6">
        <FileText size={36} className="mb-3 opacity-30" />
        <p className="font-medium mb-1">尚未开始撰写</p>
        <p className="text-xs text-center">
          确认大纲后开始撰写，<br />
          每个章节的草稿将显示在此处。
        </p>
      </div>
    );
  }

  const citationMap = new Map((citation_pool || []).map((c) => [c.cite_key || c.id, c]));
  const draftedCount = Object.keys(drafts || {}).length;
  const approvedCount = useMemo(
    () => sortedOutline.filter((s) => (reviewBySection[s.title]?.action || '').toLowerCase() === 'approve').length,
    [sortedOutline, reviewBySection],
  );

  // Poll reviews
  useEffect(() => {
    if (skip_draft_review || !activeJobId) {
      setReviewBySection({});
      return;
    }
    let cancelled = false;
    const syncReviews = async () => {
      try {
        const reviews = await listSectionReviews(activeJobId);
        if (cancelled) return;
        const next: Record<string, { action: string; feedback?: string; created_at?: number }> = {};
        reviews.forEach((r) => {
          next[r.section_id] = { action: r.action, feedback: r.feedback, created_at: r.created_at };
        });
        setReviewBySection(next);
      } catch {
        // ignore transient errors
      }
    };
    syncReviews();
    const timer = window.setInterval(syncReviews, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [skip_draft_review, activeJobId]);

  // Poll gap supplements
  useEffect(() => {
    if (!activeJobId) return;
    let cancelled = false;
    const syncSupplements = async () => {
      try {
        const sups = await listGapSupplements(activeJobId);
        if (cancelled) return;
        setGapSupplements(sups);
      } catch {
        // ignore
      }
    };
    syncSupplements();
    const timer = window.setInterval(syncSupplements, 8000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobId]);

  const handleSubmitReview = async (sectionId: string, action: 'approve' | 'revise', feedback: string = '') => {
    const jobId = localStorage.getItem('deep_research_active_job_id') || '';
    if (!jobId) {
      addToast('未检测到活跃任务，无法提交审核', 'error');
      return;
    }
    try {
      setSubmittingFor(sectionId);
      await submitSectionReview(jobId, { section_id: sectionId, action, feedback });
      setReviewBySection((prev) => ({
        ...prev,
        [sectionId]: { action, feedback, created_at: Date.now() / 1000 },
      }));
      if (action === 'approve') {
        const next = {
          ...reviewBySection,
          [sectionId]: { action, feedback, created_at: Date.now() / 1000 },
        };
        const nextApproved = sortedOutline.filter(
          (s) => (next[s.title]?.action || '').toLowerCase() === 'approve',
        ).length;
        if (nextApproved >= sortedOutline.length) {
          addToast('已全部通过，正在触发最终整合（synthesize）...', 'info');
        }
      }
      addToast(
        action === 'approve'
          ? '已通过，任务继续执行'
          : '已提交修改意见，已加入优先重写队列',
        'success',
      );
      if (action === 'revise') {
        setReviewFeedback('');
      }
      setReviewingSectionId(null);
    } catch (e) {
      addToast('提交审核失败，请重试', 'error');
    } finally {
      setSubmittingFor(null);
    }
  };

  const openSupplementModal = (
    sectionTitle: string,
    gapText: string = '',
    type: 'material' | 'direct_info' = 'direct_info',
  ) => {
    setSupplementSectionTitle(sectionTitle);
    setSupplementGapText(gapText);
    setSupplementType(type);
    setSupplementText('');
    setShowSupplementModal(true);
  };

  const closeSupplementModal = () => {
    setShowSupplementModal(false);
    setSupplementSectionTitle('');
    setSupplementGapText('');
    setSupplementText('');
    setSupplementType('direct_info');
    setSupplementSubmitting(false);
  };

  const handleSubmitSupplement = async () => {
    const text = supplementText.trim();
    if (!text) {
      addToast('请输入要补充的资料或观点', 'error');
      return;
    }
    const jobId = localStorage.getItem('deep_research_active_job_id') || '';
    const effectiveGap = (supplementGapText || 'Section-level supplement').trim();
    if (!jobId) {
      const payload = [
        `Section: ${supplementSectionTitle || 'N/A'}`,
        `Gap: ${effectiveGap}`,
        `Supplement type: ${supplementType}`,
        'User supplemental input:',
        text,
      ].join('\n');
      localStorage.setItem('deep_research_pending_user_context', payload);
      setDeepResearchTopic(canvas.topic || supplementSectionTitle || 'Deep Research');
      setShowDeepResearchDialog(true);
      addToast('未检测到活跃任务：已将补充内容带入对话框继续研究', 'info');
      closeSupplementModal();
      return;
    }

    try {
      setSupplementSubmitting(true);
      await submitGapSupplement(jobId, {
        section_id: supplementSectionTitle,
        gap_text: effectiveGap,
        supplement_type: supplementType,
        content: { text },
      });
      const sups = await listGapSupplements(jobId);
      setGapSupplements(sups);
      addToast('已提交补充信息，Agent 将在重写时采纳', 'success');
      closeSupplementModal();
    } catch {
      addToast('补充提交失败，请重试', 'error');
      setSupplementSubmitting(false);
    }
  };

  const handleSupplementGapWithMaterial = (sectionTitle: string, gap: string) => {
    openSupplementModal(sectionTitle, gap, 'material');
  };

  const handleSupplementGapWithDirectInfo = (sectionTitle: string, gap: string) => {
    openSupplementModal(sectionTitle, gap, 'direct_info');
  };

  const getGapStatus = (sectionTitle: string, gap: string): 'unhandled' | 'submitted' | 'consumed' => {
    const matching = gapSupplements.filter(
      (s) => s.section_id === sectionTitle && s.gap_text === gap,
    );
    if (!matching.length) return 'unhandled';
    if (matching.some((s) => s.status === 'consumed')) return 'consumed';
    return 'submitted';
  };

  const gapStatusLabel = (status: 'unhandled' | 'submitted' | 'consumed'): string => {
    switch (status) {
      case 'consumed': return '已采纳进本章重写';
      case 'submitted': return '已补充，待采纳';
      default: return '未处理';
    }
  };

  const gapStatusClass = (status: 'unhandled' | 'submitted' | 'consumed'): string => {
    switch (status) {
      case 'consumed': return 'text-emerald-200 bg-emerald-900/30 border-emerald-500/30';
      case 'submitted': return 'text-sky-200 bg-sky-900/30 border-sky-500/30';
      default: return 'text-slate-300 bg-slate-800/80 border-slate-600/60';
    }
  };

  const handleApproveAllSections = async () => {
    const jobId = localStorage.getItem('deep_research_active_job_id') || '';
    if (!jobId) {
      addToast('未检测到活跃任务，无法批量通过', 'error');
      return;
    }
    const pendingTitles = sortedOutline
      .map((s) => s.title)
      .filter((title) => (reviewBySection[title]?.action || '').toLowerCase() !== 'approve');
    if (!pendingTitles.length) {
      addToast('所有章节已是通过状态', 'info');
      return;
    }
    setBulkApproving(true);
    const success: string[] = [];
    const failed: string[] = [];
    for (const title of pendingTitles) {
      try {
        await submitSectionReview(jobId, { section_id: title, action: 'approve' });
        success.push(title);
      } catch {
        failed.push(title);
      }
    }

    if (success.length) {
      const ts = Date.now() / 1000;
      const merged = { ...reviewBySection };
      success.forEach((title) => {
        merged[title] = { action: 'approve', feedback: '', created_at: ts };
      });
      setReviewBySection(merged);
      const nextApproved = sortedOutline.filter(
        (s) => (merged[s.title]?.action || '').toLowerCase() === 'approve',
      ).length;
      addToast(`已批量通过 ${success.length} 个章节`, 'success');
      if (nextApproved >= sortedOutline.length) {
        addToast('已全部通过，正在触发最终整合（synthesize）...', 'info');
      }
    }
    if (failed.length) {
      addToast(`有 ${failed.length} 个章节提交失败，请重试`, 'error');
    }
    setBulkApproving(false);
  };

  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      <div className="bg-[var(--bg-surface)] rounded-lg p-3 border border-[var(--border-subtle)]">
        <div className="flex items-center justify-between">
          <span className="text-xs text-[var(--text-tertiary)]">章节进度</span>
          <span className="text-xs font-medium text-[var(--text-primary)]">
            {draftedCount} / {sortedOutline.length}
          </span>
        </div>
        {!skip_draft_review && draftedCount > 0 && (
          <div className="mt-2 space-y-0.5">
            <div className="text-[10px] text-amber-200 font-medium">
              审核进度：已通过 {approvedCount} / {sortedOutline.length}
            </div>
            <div className="text-[10px] text-amber-300">规则：章节可继续生成，但需全部通过后才会进入最终整合。</div>
            <div className="pt-1">
              <button
                onClick={handleApproveAllSections}
                disabled={bulkApproving || approvedCount >= sortedOutline.length}
                className="px-2.5 py-1 text-[11px] rounded-md border border-emerald-500/40 text-emerald-200 hover:bg-emerald-900/30 disabled:opacity-50 cursor-pointer"
              >
                {bulkApproving ? '批量提交中...' : '全部通过并触发整合'}
              </button>
            </div>
          </div>
        )}
      </div>

      {sortedOutline.map((section) => {
        const draft = drafts[section.id];
        const isExpanded = expandedSections.has(section.id);
        const hasDraft = !!draft?.content_md;
        const usedCitations = (draft?.used_citation_ids || [])
          .map((key) => citationMap.get(key))
          .filter(Boolean);
        const sectionGaps = (researchDashboard?.sections || [])
          .find((s) => (s.title || '').trim().toLowerCase() === (section.title || '').trim().toLowerCase())
          ?.gaps || [];
        const reviewAction = (reviewBySection[section.title]?.action || '').toLowerCase();
        const reviewStatusText =
          reviewAction === 'approve' ? '已通过' : reviewAction === 'revise' ? '待重写' : '待审核';
        const reviewStatusClass =
          reviewAction === 'approve'
            ? 'text-emerald-200 bg-emerald-900/30 border-emerald-500/30'
            : reviewAction === 'revise'
              ? 'text-amber-200 bg-amber-900/30 border-amber-500/30'
              : 'text-slate-300 bg-slate-800/80 border-slate-600/60';

        return (
          <div
            key={section.id}
            className={`bg-[var(--bg-panel)] rounded-lg border overflow-hidden transition-all ${
              hasDraft ? 'border-emerald-500/30 shadow-sm' : 'border-[var(--border-subtle)]'
            }`}
          >
            <button
              onClick={() => hasDraft && toggleSection(section.id)}
              className={`w-full flex items-center justify-between p-3 text-left ${
                hasDraft ? 'cursor-pointer hover:bg-[var(--bg-surface-hover)]' : 'cursor-default'
              }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-sm shrink-0">
                  {hasDraft ? '✅' : section.status === 'drafting' ? '✍️' : '⬜'}
                </span>
                <div className="min-w-0">
                  <h4 className="text-sm font-medium text-[var(--text-primary)] truncate">
                    {section.title}
                  </h4>
                  {hasDraft && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-[var(--text-tertiary)]">
                        v{draft.version} · {usedCitations.length} citations
                      </span>
                      {!skip_draft_review && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded border ${reviewStatusClass}`}>
                          {reviewStatusText}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
              {hasDraft && (
                <span className="text-[var(--text-tertiary)]">
                  {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </span>
              )}
            </button>

            {hasDraft && isExpanded && (
              <div className="border-t border-[var(--border-subtle)]">
                <div className="px-3 py-2">
                  <pre className="text-xs text-[var(--text-primary)] whitespace-pre-wrap font-sans leading-relaxed">
                    {draft.content_md}
                  </pre>
                </div>

                {usedCitations.length > 0 && (
                  <div className="px-3 py-2 border-t border-[var(--border-subtle)] bg-[var(--bg-muted)]">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <BookOpen size={10} className="text-teal-600" />
                      <span className="text-[10px] font-medium text-teal-300">References</span>
                    </div>
                    <div className="space-y-0.5">
                      {usedCitations.slice(0, 5).map((cite) => (
                        <div key={cite!.cite_key} className="text-[10px] text-[var(--text-tertiary)] truncate">
                          [{cite!.cite_key}] {cite!.title}
                          {cite!.year ? ` (${cite!.year})` : ''}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {!skip_draft_review && (
                <div className="px-3 py-2 border-t border-amber-500/30 bg-amber-900/20">
                    {reviewingSectionId === section.id ? (
                      <div className="space-y-2">
                        <textarea
                          value={reviewFeedback}
                          onChange={(e) => setReviewFeedback(e.target.value)}
                          className="w-full min-h-20 border border-amber-500/30 rounded-md px-2 py-1.5 text-xs bg-slate-900/80 text-slate-100"
                          placeholder="请输入这章的修改意见（例如：补充方法对比、补引文等）"
                        />
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleSubmitReview(section.title, 'revise', reviewFeedback)}
                            disabled={!reviewFeedback.trim() || submittingFor === section.id}
                            className="px-2.5 py-1 text-xs rounded-md bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-50 cursor-pointer flex items-center gap-1"
                          >
                            <Send size={12} />
                            提交修改意见
                          </button>
                          <button
                            onClick={() => {
                              setReviewingSectionId(null);
                              setReviewFeedback('');
                            }}
                            className="px-2.5 py-1 text-xs rounded-md border border-amber-500/30 text-amber-200 hover:bg-amber-900/30 cursor-pointer"
                          >
                            取消
                          </button>
                          <button
                            onClick={() => openSupplementModal(section.title, '', 'direct_info')}
                            className="px-2.5 py-1 text-xs rounded-md border border-indigo-500/40 text-indigo-200 hover:bg-indigo-900/30 cursor-pointer"
                          >
                            补充资料/观点
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        {reviewAction === 'approve' ? (
                          <>
                            <span className="px-2.5 py-1 text-xs rounded-md bg-emerald-900/30 text-emerald-200 border border-emerald-500/30 flex items-center gap-1">
                              <CheckCircle2 size={12} />
                              已通过
                            </span>
                            <button
                              onClick={() => handleSubmitReview(section.title, 'approve')}
                              disabled={submittingFor === section.id || bulkApproving}
                              className="px-2.5 py-1 text-xs rounded-md border border-emerald-500/40 text-emerald-200 hover:bg-emerald-900/30 disabled:opacity-50 cursor-pointer"
                            >
                              重新确认
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => handleSubmitReview(section.title, 'approve')}
                            disabled={submittingFor === section.id || bulkApproving}
                            className="px-2.5 py-1 text-xs rounded-md bg-emerald-500 text-white hover:bg-emerald-600 disabled:opacity-50 cursor-pointer flex items-center gap-1"
                          >
                            <CheckCircle2 size={12} />
                            通过此章
                          </button>
                        )}
                        <button
                          onClick={() => setReviewingSectionId(section.id)}
                          className="px-2.5 py-1 text-xs rounded-md border border-amber-500/40 text-amber-200 hover:bg-amber-900/30 cursor-pointer flex items-center gap-1"
                        >
                          <RefreshCw size={12} />
                          需要修改
                        </button>
                        <button
                          onClick={() => openSupplementModal(section.title, '', 'direct_info')}
                          className="px-2.5 py-1 text-xs rounded-md border border-indigo-500/40 text-indigo-200 hover:bg-indigo-900/30 cursor-pointer"
                        >
                          补充资料/观点
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* 章节级信息缺口（重点：每个大纲下都能看到自己的 gaps） */}
                {sectionGaps.length > 0 && (
                  <div className="px-3 py-2 border-t border-red-500/30 bg-red-900/20">
                    <div className="text-xs font-semibold text-red-200 mb-1.5">本章信息缺口</div>
                    <div className="space-y-1.5">
                      {sectionGaps.map((gap, i) => (
                        <div key={`${section.id}-gap-${i}`} className="rounded-md border border-red-500/30 bg-slate-900/70 p-2">
                          <div className="text-xs text-red-200 leading-relaxed">❗ {gap}</div>
                          {(() => {
                            const st = getGapStatus(section.title, gap);
                            return (
                              <span className={`inline-block mt-1 text-[10px] px-1.5 py-0.5 rounded border ${gapStatusClass(st)}`}>
                                {gapStatusLabel(st)}
                              </span>
                            );
                          })()}
                          <div className="mt-2 flex items-center gap-2">
                            <button
                              onClick={() => handleSupplementGapWithMaterial(section.title, gap)}
                              className="px-2.5 py-1 text-[11px] rounded-md bg-amber-500 text-white hover:bg-amber-600 cursor-pointer"
                            >
                              补材料并继续
                            </button>
                            <button
                              onClick={() => handleSupplementGapWithDirectInfo(section.title, gap)}
                              className="px-2.5 py-1 text-[11px] rounded-md border border-indigo-500/40 text-indigo-200 hover:bg-indigo-900/30 cursor-pointer"
                            >
                              直接补充信息
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}

      {showSupplementModal && (
        <div className="fixed inset-0 bg-black/40 z-[70] flex items-center justify-center p-4">
          <div className="w-full max-w-xl bg-slate-900 rounded-xl shadow-xl border border-slate-700 p-4 space-y-3">
            <div className="text-sm font-semibold text-slate-100">补充资料 / 观点</div>
            <div className="text-xs text-slate-300">
              章节：<span className="font-medium text-slate-100">{supplementSectionTitle || '-'}</span>
              {supplementGapText ? <> · 缺口：<span className="text-slate-100">{supplementGapText}</span></> : null}
            </div>
            <div className="flex items-center gap-3 text-xs">
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  checked={supplementType === 'direct_info'}
                  onChange={() => setSupplementType('direct_info')}
                />
                作为直接观点
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  checked={supplementType === 'material'}
                  onChange={() => setSupplementType('material')}
                />
                作为材料线索
              </label>
            </div>
            <textarea
              value={supplementText}
              onChange={(e) => setSupplementText(e.target.value)}
              className="w-full min-h-32 border border-slate-600 rounded-md px-3 py-2 text-sm bg-slate-950 text-slate-100"
              placeholder={supplementType === 'material'
                ? '粘贴文献线索、URL、关键词、数据库来源、实验线索等...'
                : '补充你对该章节的观点、约束、反例或解释...'}
            />
            <div className="flex items-center justify-between">
              <button
                onClick={() => {
                  const seed = [
                    `Section: ${supplementSectionTitle || 'N/A'}`,
                    `Gap: ${(supplementGapText || 'Section-level supplement').trim()}`,
                    `Supplement type: ${supplementType}`,
                    'User supplemental input:',
                    supplementText.trim(),
                  ].join('\n');
                  localStorage.setItem('deep_research_pending_user_context', seed);
                  setDeepResearchTopic(canvas.topic || supplementSectionTitle || 'Deep Research');
                  setShowDeepResearchDialog(true);
                  addToast('已带入补充内容，可在对话框上传附件后继续研究', 'info');
                  closeSupplementModal();
                }}
                className="px-2.5 py-1.5 text-xs rounded-md border border-amber-500/40 text-amber-200 hover:bg-amber-900/30 cursor-pointer"
              >
                打开对话框上传附件
              </button>
              <div className="flex items-center gap-2">
                <button
                  onClick={closeSupplementModal}
                className="px-3 py-1.5 text-xs rounded-md border border-slate-600 text-slate-200 hover:bg-slate-800 cursor-pointer"
                >
                  取消
                </button>
                <button
                  onClick={handleSubmitSupplement}
                  disabled={supplementSubmitting || !supplementText.trim()}
                  className="px-3 py-1.5 text-xs rounded-md bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 cursor-pointer"
                >
                  {supplementSubmitting ? '提交中...' : '提交补充'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {identified_gaps.length > 0 && (
        <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-3">
          <h4 className="text-xs font-semibold text-amber-200 mb-1.5 flex items-center gap-1.5">
            <AlertTriangle size={12} />
            信息缺口
          </h4>
          <ul className="space-y-0.5">
            {identified_gaps.map((gap, i) => (
              <li key={i} className="text-[11px] text-amber-300 flex items-start gap-1">
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
