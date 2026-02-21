import { useState, useEffect, useMemo } from 'react';
import { Telescope, Loader2, X, ChevronRight, Square } from 'lucide-react';
import { useChatStore, useConfigStore, useToastStore } from '../../stores';
import { useDeepResearchTask } from './deep-research/useDeepResearchTask';
import { ClarifyPhase } from './deep-research/ClarifyPhase';
import { ConfirmPhase } from './deep-research/ConfirmPhase';
import { ProgressMonitor } from './deep-research/ProgressMonitor';
import type { EfficiencyRow } from './deep-research/types';
import { DEEP_RESEARCH_PENDING_CONTEXT_KEY } from './deep-research/types';

/**
 * Deep Research 澄清对话框
 * 在用户触发 Deep Research 时弹出，显示 LLM 生成的澄清问题，
 * 用户回答后发送 Deep Research 请求。
 *
 * Orchestrates three phase sub-components (ClarifyPhase, ConfirmPhase, ProgressMonitor)
 * and delegates all API/polling logic to useDeepResearchTask.
 */
export function DeepResearchDialog() {
  const {
    showDeepResearchDialog,
    setShowDeepResearchDialog,
    deepResearchTopic,
    setDeepResearchTopic,
    clarificationQuestions,
    isStreaming,
    setDeepResearchActive,
  } = useChatStore();
  const { selectedProvider, selectedModel } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);

  const task = useDeepResearchTask();

  // UI-only form state shared across generatePlan and confirmAndRun
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [outputLanguage, setOutputLanguage] = useState<'auto' | 'en' | 'zh'>('auto');
  const [stepModelStrict, setStepModelStrict] = useState(false);
  const [stepModels, setStepModels] = useState<Record<string, string>>({
    scope: 'sonar::sonar-pro',
    plan: '',
    research: '',
    evaluate: '',
    write: '',
    verify: '',
    synthesize: '',
  });
  const [depth, setDepth] = useState<'lite' | 'comprehensive'>('comprehensive');
  const [skipDraftReview, setSkipDraftReview] = useState(false);
  const [skipRefineReview, setSkipRefineReview] = useState(false);
  const [skipClaimGeneration, setSkipClaimGeneration] = useState(false);
  const [keepPreviousJobId, setKeepPreviousJobId] = useState(true);
  const [userContext, setUserContext] = useState('');
  const [userContextMode, setUserContextMode] = useState<'supporting' | 'direct_injection'>('supporting');
  const [tempDocuments, setTempDocuments] = useState<Array<{ name: string; content: string }>>([]);

  // Initialise default answers when clarification questions arrive
  useEffect(() => {
    if (clarificationQuestions.length > 0) {
      const defaults: Record<string, string> = {};
      clarificationQuestions.forEach((q) => {
        defaults[q.id] = q.default || '';
      });
      setAnswers(defaults);
    }
  }, [clarificationQuestions]);

  // Reset local UI state when dialog opens (unless resuming a running job)
  useEffect(() => {
    if (!showDeepResearchDialog) return;
    if (task.activeJobId) {
      task.setPhase('running');
      return;
    }
    task.setPhase('clarify');
    task.setOutlineDraft([]);
    task.setOptimizationPromptDraft('');

    const pendingUserContext = localStorage.getItem(DEEP_RESEARCH_PENDING_CONTEXT_KEY) || '';
    if (pendingUserContext.trim()) {
      setUserContext(pendingUserContext.trim());
      setUserContextMode('direct_injection');
      localStorage.removeItem(DEEP_RESEARCH_PENDING_CONTEXT_KEY);
    } else {
      setUserContext('');
      setUserContextMode('supporting');
    }
    setTempDocuments([]);
    setAnswers({});

    const defaults = useConfigStore.getState().deepResearchDefaults;
    setDepth(defaults.depth);
    setOutputLanguage(defaults.outputLanguage);
    setStepModelStrict(defaults.stepModelStrict);
    setSkipClaimGeneration(defaults.skipClaimGeneration);
    setStepModels({ ...defaults.stepModels });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showDeepResearchDialog]);

  const modelOptions = useMemo(() => {
    const current = selectedModel ? `${selectedProvider}::${selectedModel}` : '';
    return [
      { value: '', label: 'Default (current global model)' },
      ...(current ? [{ value: current, label: `Current: ${current}` }] : []),
      { value: 'sonar::sonar', label: 'sonar (search)' },
      { value: 'sonar::sonar-pro', label: 'sonar-pro (search)' },
      { value: 'sonar::sonar-reasoning-pro', label: 'sonar-reasoning-pro' },
    ];
  }, [selectedProvider, selectedModel]);

  const buildPlanParams = () => ({
    topic: deepResearchTopic,
    answers,
    outputLanguage,
    stepModels,
    stepModelStrict,
  });

  const handleGeneratePlan = () => task.generatePlan(buildPlanParams());

  const handleSkipClarificationAndGenerate = () => {
    setAnswers({});
    task.generatePlan({ ...buildPlanParams(), answers: {} });
  };

  const handleConfirmAndRun = () => task.confirmAndRun({
    ...buildPlanParams(),
    outlineDraft: task.outlineDraft,
    briefDraft: task.briefDraft,
    userContext,
    userContextMode,
    tempDocuments,
    depth,
    skipDraftReview,
    skipRefineReview,
    skipClaimGeneration,
    keepPreviousJobId,
  });

  const handleClose = () => {
    setShowDeepResearchDialog(false);
    if (!task.activeJobId) {
      setDeepResearchActive(false);
    }
  };

  const handleGenerateOptimizationPrompt = (
    lowRows: EfficiencyRow[],
    allRows: EfficiencyRow[],
    targetCoverage: number,
  ) => {
    const targets = lowRows.length > 0 ? lowRows : allRows.slice(0, 3);
    if (!targets.length) {
      addToast('当前样本不足，请至少完成两轮 section evaluate', 'info');
      return;
    }
    const lines: string[] = [];
    lines.push(`# Deep Research Section Optimization Template`);
    lines.push(`Topic: ${deepResearchTopic || '(fill topic)'}`);
    lines.push(`Target coverage: ${targetCoverage.toFixed(2)}`);
    lines.push('');
    lines.push(`Usage: paste selected blocks into "Intervention" before next run.`);
    lines.push('');
    targets.forEach((row, idx) => {
      lines.push(`## ${idx + 1}. ${row.section}`);
      lines.push(`Current signal:`);
      lines.push(`- Coverage: ${row.lastCoverage.toFixed(2)} (target ${targetCoverage.toFixed(2)})`);
      lines.push(`- Avg gain/round: ${row.avgDelta.toFixed(3)}`);
      if (row.per10Steps !== null) {
        lines.push(`- Gain per 10 steps: ${row.per10Steps.toFixed(3)}`);
      }
      lines.push(`Optimization prompt skeleton:`);
      lines.push(`- Scope constraints:`);
      lines.push(`  - Focus only on "${row.section}"`);
      lines.push(`  - Exclude adjacent sections and generic narrative`);
      lines.push(`- Retrieval directives:`);
      lines.push(`  - Expand terminology variants, abbreviations, and mechanism synonyms`);
      lines.push(`  - Prioritize primary studies and data-bearing sources`);
      lines.push(`- Evidence directives:`);
      lines.push(`  - Provide explicit support for each major claim with citation tags`);
      lines.push(`  - Flag weak claims as evidence-limited instead of asserting`);
      lines.push(`- My supplemental evidence:`);
      lines.push(`  - [Paste your materials, notes, or constraints here]`);
      lines.push('');
    });
    task.setOptimizationPromptDraft(lines.join('\n'));
    addToast('已生成章节优化提示词模板，可复制后用于 Intervention', 'success');
  };

  const handleCopyOptimizationPrompt = async () => {
    if (!task.optimizationPromptDraft.trim()) return;
    try {
      await navigator.clipboard.writeText(task.optimizationPromptDraft);
      addToast('已复制优化提示词模板', 'success');
    } catch (err) {
      console.error('[DeepResearch] copy optimization prompt failed:', err);
      addToast('复制失败，请手动复制文本', 'warning');
    }
  };

  const handleInsertOptimizationPrompt = () => {
    task.handleInsertOptimizationPrompt(task.optimizationPromptDraft.trim(), setUserContext);
  };

  if (!showDeepResearchDialog) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col animate-in slide-in-from-bottom-4 duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Telescope size={20} className="text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Deep Research</h2>
              <p className="text-xs text-gray-500">多步深度研究 - 可确认大纲并跟踪进度</p>
            </div>
          </div>
          <button onClick={handleClose} className="p-1 hover:bg-gray-100 rounded-lg transition-colors">
            <X size={18} className="text-gray-400" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Phase indicator */}
          <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2">
            <div className="text-xs text-gray-600">
              <span className={task.phase === 'clarify' ? 'font-semibold text-indigo-600' : ''}>1. 澄清问题</span>
              <span className="mx-2 text-gray-300">→</span>
              <span className={task.phase === 'confirm' ? 'font-semibold text-indigo-600' : ''}>2. 确认大纲</span>
              <span className="mx-2 text-gray-300">→</span>
              <span className={task.phase === 'running' ? 'font-semibold text-indigo-600' : ''}>3. 执行研究</span>
            </div>
          </div>

          {/* Topic input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">研究主题</label>
            <input
              type="text"
              value={deepResearchTopic}
              onChange={(e) => setDeepResearchTopic(e.target.value)}
              className="w-full border border-gray-300 bg-white text-gray-900 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
              placeholder="输入综述主题..."
            />
          </div>

          {task.phase === 'clarify' && (
            <ClarifyPhase
              questions={clarificationQuestions}
              answers={answers}
              onAnswerChange={(id, value) => setAnswers((prev) => ({ ...prev, [id]: value }))}
              isClarifying={task.isClarifying}
              onRegenerate={() => task.runClarify(deepResearchTopic)}
              depth={depth}
              scopeModel={stepModels.scope}
              outputLanguage={outputLanguage}
              topic={deepResearchTopic}
            />
          )}

          {task.phase === 'confirm' && (
            <ConfirmPhase
              outlineDraft={task.outlineDraft}
              onOutlineChange={(idx, value) =>
                task.setOutlineDraft((prev) => prev.map((item, i) => (i === idx ? value : item)))
              }
              onOutlineAdd={() => task.setOutlineDraft((prev) => [...prev, ''])}
              onOutlineRemove={(idx) => task.setOutlineDraft((prev) => prev.filter((_, i) => i !== idx))}
              onOutlineMove={(from, to) => {
                task.setOutlineDraft((prev) => {
                  if (from === to || from < 0 || to < 0 || from >= prev.length || to >= prev.length) return prev;
                  const next = [...prev];
                  const [moved] = next.splice(from, 1);
                  next.splice(to, 0, moved);
                  return next;
                });
              }}
              briefDraft={task.briefDraft}
              initialStats={task.initialStats}
              outputLanguage={outputLanguage}
              onOutputLanguageChange={setOutputLanguage}
              stepModels={stepModels}
              onStepModelChange={(step, value) => setStepModels((prev) => ({ ...prev, [step]: value }))}
              stepModelStrict={stepModelStrict}
              onStepModelStrictChange={setStepModelStrict}
              modelOptions={modelOptions}
              depth={depth}
              onDepthChange={setDepth}
              skipDraftReview={skipDraftReview}
              onSkipDraftReviewChange={setSkipDraftReview}
              skipRefineReview={skipRefineReview}
              onSkipRefineReviewChange={setSkipRefineReview}
              skipClaimGeneration={skipClaimGeneration}
              onSkipClaimGenerationChange={setSkipClaimGeneration}
              keepPreviousJobId={keepPreviousJobId}
              onKeepPreviousJobIdChange={setKeepPreviousJobId}
              userContext={userContext}
              onUserContextChange={setUserContext}
              userContextMode={userContextMode}
              onUserContextModeChange={setUserContextMode}
              tempDocuments={tempDocuments}
              onTempDocumentsChange={setTempDocuments}
            />
          )}

          {task.phase === 'running' && (
            <ProgressMonitor
              researchMonitor={task.researchMonitor}
              progressLogs={task.progressLogs}
              depth={depth}
              optimizationPromptDraft={task.optimizationPromptDraft}
              onGenerateOptimizationPrompt={handleGenerateOptimizationPrompt}
              onInsertOptimizationPrompt={handleInsertOptimizationPrompt}
              onCopyOptimizationPrompt={handleCopyOptimizationPrompt}
            />
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-gray-50 rounded-b-2xl">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
            disabled={task.phase === 'running'}
          >
            取消
          </button>
          <div className="flex items-center gap-2">
            {task.phase === 'clarify' && clarificationQuestions.length > 0 && (
              <button
                onClick={handleSkipClarificationAndGenerate}
                disabled={isStreaming || !deepResearchTopic.trim()}
                className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
              >
                跳过澄清
              </button>
            )}
            {task.phase === 'clarify' && (
              <button
                onClick={handleGeneratePlan}
                disabled={isStreaming || !deepResearchTopic.trim()}
                className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isStreaming ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
                生成大纲
              </button>
            )}
            {task.phase === 'confirm' && (
              <>
                <button
                  onClick={() => task.setPhase('clarify')}
                  disabled={isStreaming}
                  className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
                >
                  返回澄清
                </button>
                <button
                  onClick={handleConfirmAndRun}
                  disabled={isStreaming || !deepResearchTopic.trim()}
                  className="flex items-center gap-2 px-5 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isStreaming ? <Loader2 size={16} className="animate-spin" /> : <ChevronRight size={16} />}
                  确认并开始研究
                </button>
              </>
            )}
            {task.phase === 'running' && (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 text-sm font-medium rounded-lg">
                  <Loader2 size={16} className="animate-spin" />
                  后台研究中
                </div>
                <button
                  onClick={() => task.stopJob()}
                  disabled={task.isStopping || !task.activeJobId}
                  className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 text-sm font-medium rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Square size={14} />
                  {task.isStopping ? '停止中...' : '停止任务'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
