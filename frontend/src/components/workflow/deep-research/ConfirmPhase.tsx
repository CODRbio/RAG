import { useState, useRef } from 'react';
import { Plus, Trash2, GripVertical, Loader2, Paperclip } from 'lucide-react';
import { useToastStore } from '../../../stores';
import { extractDeepResearchContextFiles } from '../../../api/chat';
import type { InitialStats, BriefDraft } from './types';

interface ModelOption {
  value: string;
  label: string;
}

interface ConfirmPhaseProps {
  outlineDraft: string[];
  onOutlineChange: (index: number, value: string) => void;
  onOutlineAdd: () => void;
  onOutlineRemove: (index: number) => void;
  onOutlineMove: (from: number, to: number) => void;
  briefDraft: BriefDraft | null;
  initialStats: InitialStats | null;
  outputLanguage: 'auto' | 'en' | 'zh';
  onOutputLanguageChange: (v: 'auto' | 'en' | 'zh') => void;
  stepModels: Record<string, string>;
  onStepModelChange: (step: string, value: string) => void;
  stepModelStrict: boolean;
  onStepModelStrictChange: (v: boolean) => void;
  modelOptions: ModelOption[];
  depth: 'lite' | 'comprehensive';
  onDepthChange: (v: 'lite' | 'comprehensive') => void;
  skipDraftReview: boolean;
  onSkipDraftReviewChange: (v: boolean) => void;
  skipRefineReview: boolean;
  onSkipRefineReviewChange: (v: boolean) => void;
  skipClaimGeneration: boolean;
  onSkipClaimGenerationChange: (v: boolean) => void;
  keepPreviousJobId: boolean;
  onKeepPreviousJobIdChange: (v: boolean) => void;
  userContext: string;
  onUserContextChange: (v: string) => void;
  userContextMode: 'supporting' | 'direct_injection';
  onUserContextModeChange: (v: 'supporting' | 'direct_injection') => void;
  tempDocuments: Array<{ name: string; content: string }>;
  onTempDocumentsChange: (docs: Array<{ name: string; content: string }>) => void;
}

export function ConfirmPhase({
  outlineDraft,
  onOutlineChange,
  onOutlineAdd,
  onOutlineRemove,
  onOutlineMove,
  initialStats,
  outputLanguage,
  onOutputLanguageChange,
  stepModels,
  onStepModelChange,
  stepModelStrict,
  onStepModelStrictChange,
  modelOptions,
  depth,
  onDepthChange,
  skipDraftReview,
  onSkipDraftReviewChange,
  skipRefineReview,
  onSkipRefineReviewChange,
  skipClaimGeneration,
  onSkipClaimGenerationChange,
  keepPreviousJobId,
  onKeepPreviousJobIdChange,
  userContext,
  onUserContextChange,
  userContextMode,
  onUserContextModeChange,
  tempDocuments,
  onTempDocumentsChange,
}: ConfirmPhaseProps) {
  const addToast = useToastStore((s) => s.addToast);

  // DnD state is local — purely visual, only relevant in this phase
  const [draggingOutlineIndex, setDraggingOutlineIndex] = useState<number | null>(null);
  const [dragOverOutlineIndex, setDragOverOutlineIndex] = useState<number | null>(null);
  const [showAdvancedModels, setShowAdvancedModels] = useState(false);
  const [isExtractingContextFiles, setIsExtractingContextFiles] = useState(false);
  const contextFileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSelectContextFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);
    setIsExtractingContextFiles(true);
    try {
      const docs = await extractDeepResearchContextFiles(files);
      if (!docs.length) {
        addToast('未从文件提取到有效文本（支持 pdf/md/txt）', 'info');
        return;
      }
      const merged = [...tempDocuments];
      docs.forEach((d) => {
        const exists = merged.some((x) => x.name === d.name && x.content === d.content);
        if (!exists) merged.push(d);
      });
      onTempDocumentsChange(merged.slice(0, 10));
      addToast(`已添加 ${docs.length} 份临时材料`, 'success');
    } catch (err) {
      console.error('[DeepResearch] context extract failed:', err);
      addToast('临时材料提取失败，请重试', 'error');
    } finally {
      setIsExtractingContextFiles(false);
      if (contextFileInputRef.current) contextFileInputRef.current.value = '';
    }
  };

  return (
    <>
      {/* Output Language */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Output Language</label>
        <select
          value={outputLanguage}
          onChange={(e) => onOutputLanguageChange(e.target.value as 'auto' | 'en' | 'zh')}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
        >
          <option value="auto">Auto (follow topic language)</option>
          <option value="en">English</option>
          <option value="zh">中文</option>
        </select>
      </div>

      {/* Per-step model overrides */}
      <div className="space-y-2">
        <button
          onClick={() => setShowAdvancedModels((v) => !v)}
          className="text-xs text-indigo-600 hover:text-indigo-700"
        >
          {showAdvancedModels ? '\u25BE Hide' : '\u25B8 Override'} Per-step Models
        </button>
        {showAdvancedModels && (
          <div className="space-y-2 border border-gray-200 rounded-lg p-3">
            <div className="text-[10px] text-gray-400 pb-1.5 border-b border-gray-100">
              Loaded from &#9881; defaults. Changes here apply to this run only.
            </div>
            <label className="flex items-center justify-between text-xs text-gray-600">
              <span className="flex items-center gap-1">
                Strict step model resolution
                <span className="text-gray-400 cursor-help" title="OFF: model failure falls back to default silently. ON: model failure aborts the research immediately.">?</span>
              </span>
              <input
                type="checkbox"
                checked={stepModelStrict}
                onChange={(e) => onStepModelStrictChange(e.target.checked)}
                className="accent-indigo-500"
              />
            </label>
            {['scope', 'plan', 'research', 'evaluate', 'write', 'verify', 'synthesize'].map((step) => (
              <div key={step} className="grid grid-cols-3 items-center gap-2">
                <div className="text-xs font-medium text-gray-600 uppercase">{step}</div>
                <select
                  value={stepModels[step] || ''}
                  onChange={(e) => onStepModelChange(step, e.target.value)}
                  className="col-span-2 border border-gray-200 rounded-md px-2 py-1 text-xs"
                >
                  {modelOptions.map((opt) => (
                    <option key={`${step}-${opt.value || 'default'}`} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Initial stats + outline */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3 text-xs text-indigo-800">
        <div>Initial sources: {initialStats?.total_sources ?? 0}</div>
        <div>Iterations: {initialStats?.total_iterations ?? 0}</div>
        <div>Tip: edit and reorder outline before execution.</div>
      </div>

      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-700">Outline Confirmation</div>
        {outlineDraft.map((item, idx) => (
          <div key={`outline-${idx}`} className="space-y-1">
            {dragOverOutlineIndex === idx && draggingOutlineIndex !== null && (
              <div className="h-0.5 bg-indigo-500 rounded-full" />
            )}
            <div
              className={`flex gap-2 ${draggingOutlineIndex === idx ? 'opacity-60' : ''}`}
              draggable
              onDragStart={() => setDraggingOutlineIndex(idx)}
              onDragEnd={() => {
                setDraggingOutlineIndex(null);
                setDragOverOutlineIndex(null);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOverOutlineIndex(idx);
              }}
              onDrop={() => {
                if (draggingOutlineIndex === null) return;
                onOutlineMove(draggingOutlineIndex, idx);
                setDraggingOutlineIndex(null);
                setDragOverOutlineIndex(null);
              }}
            >
              <button
                type="button"
                className="px-2 py-1 border rounded-md text-gray-400 hover:bg-gray-50 cursor-grab active:cursor-grabbing"
                title="拖拽排序"
              >
                <GripVertical size={14} />
              </button>
              <input
                type="text"
                value={item}
                onChange={(e) => onOutlineChange(idx, e.target.value)}
                placeholder="New section title..."
                className="flex-1 border border-gray-200 rounded-md px-2.5 py-1.5 text-sm"
              />
              <button
                onClick={() => onOutlineRemove(idx)}
                className="px-2 py-1 border rounded-md text-gray-500 hover:bg-gray-50"
              >
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
        <button
          onClick={onOutlineAdd}
          className="inline-flex items-center gap-1 px-2.5 py-1 border rounded-md text-xs text-gray-600 hover:bg-gray-50"
        >
          <Plus size={12} /> Add section
        </button>
        <div className="text-xs text-gray-500">可拖拽左侧图标调整章节顺序。</div>
      </div>

      {/* Research Depth */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2">
        <div className="text-sm font-medium text-gray-700">Research Depth (研究深度)</div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => onDepthChange('lite')}
            className={`flex flex-col items-start p-2.5 rounded-lg border text-left transition-all ${
              depth === 'lite'
                ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300'
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <span className="text-xs font-semibold text-gray-800">Lite</span>
            <span className="text-[10px] text-gray-500 leading-tight mt-0.5">
              Quick but academically usable, ~5-15 min. 4 queries/section (recall+precision), tiered top_k 18/10/10, coverage &ge; 60%.
            </span>
          </button>
          <button
            type="button"
            onClick={() => onDepthChange('comprehensive')}
            className={`flex flex-col items-start p-2.5 rounded-lg border text-left transition-all ${
              depth === 'comprehensive'
                ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300'
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <span className="text-xs font-semibold text-gray-800">Comprehensive</span>
            <span className="text-[10px] text-gray-500 leading-tight mt-0.5">
              Thorough academic review, ~20-60 min. 8 queries/section (recall+precision), tiered top_k 30/15/12, coverage &ge; 80%.
            </span>
          </button>
        </div>
      </div>

      {/* Stage Intervention */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2.5">
        <div className="text-sm font-medium text-gray-700">Stage Intervention (阶段介入)</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-not-allowed opacity-70">
            <input type="checkbox" checked disabled className="accent-indigo-500" />
            <span>澄清意图 (Clarify)</span>
            <span className="ml-auto text-[10px] text-indigo-500 font-medium">必须</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-not-allowed opacity-70">
            <input type="checkbox" checked disabled className="accent-indigo-500" />
            <span>确认大纲 (Confirm Outline)</span>
            <span className="ml-auto text-[10px] text-indigo-500 font-medium">必须</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={!skipDraftReview}
              onChange={(e) => onSkipDraftReviewChange(!e.target.checked)}
              className="accent-indigo-500"
            />
            <span>逐章审阅 (Review Each Section)</span>
            <span className="ml-auto text-[10px] text-gray-400">{skipDraftReview ? '已跳过' : '可选'}</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={!skipRefineReview}
              onChange={(e) => onSkipRefineReviewChange(!e.target.checked)}
              className="accent-indigo-500"
            />
            <span>精炼修改 (Refine with Directives)</span>
            <span className="ml-auto text-[10px] text-gray-400">{skipRefineReview ? '已跳过' : '可选'}</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={skipClaimGeneration}
              onChange={(e) => onSkipClaimGenerationChange(e.target.checked)}
              className="accent-indigo-500"
            />
            <span>跳过前置论点提炼 (Skip Claim Generation)</span>
            <span className="ml-auto text-[10px] text-gray-400">{skipClaimGeneration ? '已跳过' : '可选'}</span>
          </label>
        </div>
        <label className="flex items-center gap-2 text-xs text-gray-500 pt-1 border-t border-gray-200 cursor-pointer">
          <input
            type="checkbox"
            checked={skipDraftReview && skipRefineReview}
            onChange={(e) => {
              onSkipDraftReviewChange(e.target.checked);
              onSkipRefineReviewChange(e.target.checked);
              onSkipClaimGenerationChange(e.target.checked);
            }}
            className="accent-gray-400"
          />
          <span>最小化人工介入（仅保留必须步骤）</span>
        </label>
      </div>

      {/* Job ID strategy */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-1.5">
        <div className="text-sm font-medium text-gray-700">任务 ID 策略</div>
        <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={keepPreviousJobId}
            onChange={(e) => onKeepPreviousJobIdChange(e.target.checked)}
            className="accent-indigo-500"
          />
          <span>开始新任务时保留旧任务 ID（便于后续恢复）</span>
        </label>
        <div className="text-[11px] text-gray-500">
          当前任务将继续在后台运行；新任务会使用新的 job id。
        </div>
      </div>

      {/* Intervention / user context */}
      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-700">Intervention (补充上下文，可选)</div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">文本介入模式</label>
          <select
            value={userContextMode}
            onChange={(e) => onUserContextModeChange(e.target.value as 'supporting' | 'direct_injection')}
            className="w-full border border-gray-200 rounded-md px-2.5 py-1.5 text-xs"
          >
            <option value="supporting">作为补充上下文（默认）</option>
            <option value="direct_injection">作为强提示直接注入（我对内容非常自信）</option>
          </select>
        </div>
        <textarea
          value={userContext}
          onChange={(e) => onUserContextChange(e.target.value)}
          placeholder={userContextMode === 'direct_injection'
            ? '输入高置信观点/约束，系统会作为高优先级提示并要求显式验证...'
            : '可补充新观点、反例、约束条件、重点文献线索...'}
          className="w-full min-h-20 border border-gray-200 rounded-md px-2.5 py-2 text-sm"
        />
        <input
          ref={contextFileInputRef}
          type="file"
          accept=".pdf,.md,.txt"
          multiple
          className="hidden"
          onChange={(e) => handleSelectContextFiles(e.target.files)}
        />
        <button
          onClick={() => contextFileInputRef.current?.click()}
          disabled={isExtractingContextFiles}
          className="inline-flex items-center gap-1 px-2.5 py-1 border rounded-md text-xs text-gray-600 hover:bg-gray-50 disabled:opacity-50"
        >
          {isExtractingContextFiles ? <Loader2 size={12} className="animate-spin" /> : <Paperclip size={12} />}
          上传临时材料 (pdf/md/txt)
        </button>
        {tempDocuments.length > 0 && (
          <div className="space-y-1">
            {tempDocuments.map((doc, idx) => (
              <div key={`${doc.name}-${idx}`} className="flex items-center justify-between text-xs bg-gray-50 border border-gray-200 rounded px-2 py-1.5">
                <span className="truncate pr-2">{doc.name}</span>
                <button
                  onClick={() => onTempDocumentsChange(tempDocuments.filter((_, i) => i !== idx))}
                  className="text-gray-500 hover:text-red-500"
                >
                  移除
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="text-xs text-gray-500">这些材料仅用于本次任务，不写入持久本地知识库。</div>
      </div>
    </>
  );
}
