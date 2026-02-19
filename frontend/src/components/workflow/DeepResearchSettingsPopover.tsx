import { useState, useEffect, useRef } from 'react';
import { Settings, X, HelpCircle } from 'lucide-react';
import { useConfigStore } from '../../stores';

/** Hover tooltip with delay, matching Sidebar HelpTooltip pattern */
function Tip({ content, children }: { content: string; children: React.ReactNode }) {
  const [visible, setVisible] = useState(false);
  const showRef = useRef<number | null>(null);
  const hideRef = useRef<number | null>(null);

  const handleEnter = () => {
    if (hideRef.current) { window.clearTimeout(hideRef.current); hideRef.current = null; }
    showRef.current = window.setTimeout(() => setVisible(true), 600);
  };
  const handleLeave = () => {
    if (showRef.current) { window.clearTimeout(showRef.current); showRef.current = null; }
    hideRef.current = window.setTimeout(() => setVisible(false), 150);
  };

  return (
    <span className="relative inline-flex items-center text-gray-400 cursor-help shrink-0" onMouseEnter={handleEnter} onMouseLeave={handleLeave}>
      {children}
      {visible && (
        <span className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 px-2.5 py-2 text-[10px] leading-relaxed text-gray-100 bg-gray-800/95 rounded-lg shadow-lg max-w-[240px] whitespace-normal z-[100] border border-gray-600/80 pointer-events-none" role="tooltip">
          {content}
        </span>
      )}
    </span>
  );
}

/**
 * Per-step model options for Deep Research.
 * value = "provider::model" or "" for default.
 */
const STEP_MODEL_OPTIONS = [
  { value: '', label: 'Default (global model)' },
  // Sonar (search-optimized)
  { value: 'sonar::sonar', label: 'sonar (search)' },
  { value: 'sonar::sonar-pro', label: 'sonar-pro (search)' },
  { value: 'sonar::sonar-reasoning-pro', label: 'sonar-reasoning-pro' },
  // DeepSeek
  { value: 'deepseek::deepseek-chat', label: 'deepseek-chat' },
  { value: 'deepseek-thinking::deepseek-reasoner', label: 'deepseek-reasoner (thinking)' },
  // Claude
  { value: 'claude::claude-sonnet-4-5', label: 'claude-sonnet-4.5' },
  { value: 'claude::claude-haiku-4-5', label: 'claude-haiku-4.5' },
  { value: 'claude-thinking::claude-sonnet-4-5', label: 'claude-sonnet-4.5 (thinking)' },
  // Gemini
  { value: 'gemini::gemini-pro-latest', label: 'gemini-pro' },
  { value: 'gemini::gemini-flash-latest', label: 'gemini-flash' },
  // Kimi
  { value: 'kimi::kimi-k2.5', label: 'kimi-k2.5' },
];

const RESEARCH_STEPS = ['scope', 'plan', 'research', 'evaluate', 'write', 'verify', 'synthesize'] as const;

interface Props {
  open: boolean;
  onClose: () => void;
}

export function DeepResearchSettingsPopover({ open, onClose }: Props) {
  const { deepResearchDefaults, updateDeepResearchDefaults, setDeepResearchStepModel } = useConfigStore();
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    // Use setTimeout to avoid the opening click triggering close
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handler);
    }, 0);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handler);
    };
  }, [open, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={popoverRef}
      className="absolute bottom-full mb-2 left-0 w-80 bg-white rounded-xl shadow-2xl border border-gray-200 z-50"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <Settings size={13} className="text-indigo-500" />
          <span className="text-xs font-semibold text-gray-800">Deep Research 默认设置</span>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-md transition-colors">
          <X size={12} className="text-gray-400" />
        </button>
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-3.5 max-h-[60vh] overflow-y-auto">
        {/* Depth */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1.5">
            Research Depth
            <Tip content="Lite: faster, fewer iterations, lower coverage threshold (~60%). Comprehensive: thorough academic review, more queries per section, higher coverage (~80%).">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <div className="grid grid-cols-2 gap-2">
            {(['lite', 'comprehensive'] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => updateDeepResearchDefaults({ depth: d })}
                className={`px-2.5 py-1.5 rounded-lg border text-left transition-all text-[11px] ${
                  deepResearchDefaults.depth === d
                    ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-200 font-semibold text-indigo-700'
                    : 'border-gray-200 bg-white hover:border-gray-300 text-gray-600'
                }`}
              >
                <div className="font-medium">{d === 'lite' ? 'Lite' : 'Comprehensive'}</div>
                <div className="text-[9px] text-gray-400 leading-tight mt-0.5">
                  {d === 'lite' ? '~5-15 min, coverage >= 60%' : '~20-60 min, coverage >= 80%'}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Output Language */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1">
            Output Language
            <Tip content="Auto: follow the topic's language. You can also force English or Chinese output regardless of the input language.">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <select
            value={deepResearchDefaults.outputLanguage}
            onChange={(e) => updateDeepResearchDefaults({ outputLanguage: e.target.value as 'auto' | 'en' | 'zh' })}
            className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] focus:ring-2 focus:ring-indigo-500 outline-none"
          >
            <option value="auto">Auto (follow topic language)</option>
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
        </div>

        {/* Year Window */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1">
            Year Window (Hard Filter)
            <Tip content="Apply strict publication-year filtering during retrieval. Empty means no limit.">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-[10px] text-gray-500 mb-1">起始年份</div>
              <input
                type="number"
                min={1900}
                max={2100}
                value={deepResearchDefaults.yearStart ?? ''}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  if (raw === '') {
                    updateDeepResearchDefaults({ yearStart: null });
                    return;
                  }
                  const n = Number(raw);
                  updateDeepResearchDefaults({
                    yearStart: Number.isFinite(n) ? Math.max(1900, Math.min(2100, Math.trunc(n))) : null,
                  });
                }}
                className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] focus:ring-2 focus:ring-indigo-500 outline-none"
                placeholder="e.g. 2020"
              />
            </div>
            <div>
              <div className="text-[10px] text-gray-500 mb-1">结束年份</div>
              <input
                type="number"
                min={1900}
                max={2100}
                value={deepResearchDefaults.yearEnd ?? ''}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  if (raw === '') {
                    updateDeepResearchDefaults({ yearEnd: null });
                    return;
                  }
                  const n = Number(raw);
                  updateDeepResearchDefaults({
                    yearEnd: Number.isFinite(n) ? Math.max(1900, Math.min(2100, Math.trunc(n))) : null,
                  });
                }}
                className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] focus:ring-2 focus:ring-indigo-500 outline-none"
                placeholder="e.g. 2025"
              />
            </div>
          </div>
        </div>

        {/* Per-step Models */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1.5">
            Per-step Models
            <Tip content="Assign different LLMs to each research step. E.g. use sonar-pro for scope/research (web search), claude for write/verify (quality). 'Default' uses the global model selected in the header.">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <div className="space-y-1 border border-gray-100 rounded-lg p-2.5 bg-gray-50">
            {RESEARCH_STEPS.map((step) => (
              <div key={step} className="grid grid-cols-[72px_1fr] items-center gap-1.5">
                <span className="text-[10px] font-medium text-gray-500 uppercase">{step}</span>
                <select
                  value={deepResearchDefaults.stepModels[step] || ''}
                  onChange={(e) => setDeepResearchStepModel(step, e.target.value)}
                  className="border border-gray-200 rounded-md px-1.5 py-1 text-[10px] bg-white focus:ring-1 focus:ring-indigo-400 outline-none"
                >
                  {STEP_MODEL_OPTIONS.map((opt) => (
                    <option key={`${step}-${opt.value}`} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>

        {/* Strict Mode */}
        <label className="flex items-center justify-between text-[11px] text-gray-600 cursor-pointer px-0.5">
          <span className="flex items-center gap-1">
            Strict step model resolution
            <Tip content="OFF (default): if a step's designated model fails to load (API key missing, provider unavailable, etc.), the system silently falls back to the global default model and continues the research. ON: if any step's model fails, the entire research is aborted immediately. Use this when you need a specific model for correctness (e.g. sonar for web search) and a fallback would produce meaningless results.">
              <HelpCircle size={11} />
            </Tip>
          </span>
          <input
            type="checkbox"
            checked={deepResearchDefaults.stepModelStrict}
            onChange={(e) => updateDeepResearchDefaults({ stepModelStrict: e.target.checked })}
            className="accent-indigo-500"
          />
        </label>

        {/* Skip Claim Generation */}
        <label className="flex items-center justify-between text-[11px] text-gray-600 cursor-pointer px-0.5">
          <span className="flex items-center gap-1">
            跳过前置论点提炼 (Skip Claim Generation)
            <Tip content="When OFF (default): before writing each section, the system first extracts 3–5 core claims from evidence with [ref_hash] citations, then expands them into prose. When ON: skip claim extraction and write directly from evidence. Use ON for faster runs or when you prefer free-form section writing.">
              <HelpCircle size={11} />
            </Tip>
          </span>
          <input
            type="checkbox"
            checked={deepResearchDefaults.skipClaimGeneration}
            onChange={(e) => updateDeepResearchDefaults({ skipClaimGeneration: e.target.checked })}
            className="accent-indigo-500"
          />
        </label>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-100 bg-gray-50 rounded-b-xl">
        <p className="text-[9px] text-gray-400">设置自动保存，跨会话持久化。在 Deep Research 对话内可临时覆盖。</p>
      </div>
    </div>
  );
}
