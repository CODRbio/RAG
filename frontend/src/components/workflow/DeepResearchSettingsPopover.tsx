import { useState, useEffect, useRef, useMemo } from 'react';
import { Settings, X, HelpCircle } from 'lucide-react';
import { useConfigStore } from '../../stores';
import { listLLMProviders, listAllLiveModels, type LLMProviderInfo } from '../../api/ingest';

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

interface StepModelOption { value: string; label: string; }

function providerSuffix(id: string): string {
  const parts = id.split('-').slice(1);
  return parts.length ? ` (${parts.join('-')})` : '';
}

function buildStepModelOptions(providers: LLMProviderInfo[]): StepModelOption[] {
  const opts: StepModelOption[] = [{ value: '', label: 'Default (global model)' }];
  for (const p of providers) {
    const suffix = providerSuffix(p.id);
    for (const modelKey of p.models) {
      opts.push({ value: `${p.id}::${modelKey}`, label: `${modelKey}${suffix}` });
    }
  }
  return opts;
}

/** Only Perplexity — for 初步认知 (preliminary knowledge). */
function buildPreliminaryModelOptions(providers: LLMProviderInfo[]): StepModelOption[] {
  const allow = (id: string) => id === 'perplexity' || id.startsWith('perplexity-') || id === 'sonar' || id.startsWith('sonar-');
  const opts: StepModelOption[] = [];
  for (const p of providers) {
    if (!allow(p.id)) continue;
    const suffix = providerSuffix(p.id);
    for (const modelKey of p.models) {
      opts.push({ value: `${p.id}::${modelKey}`, label: `${modelKey}${suffix}` });
    }
  }
  return opts;
}

const RESEARCH_STEPS = ['plan', 'research', 'evaluate', 'write', 'verify', 'synthesize'] as const;

/** Display label for Per-step Models. */
function stepDisplayLabel(step: string): string {
  return step;
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export function DeepResearchSettingsPopover({ open, onClose }: Props) {
  const { deepResearchDefaults, updateDeepResearchDefaults, setDeepResearchStepModel } = useConfigStore();
  const popoverRef = useRef<HTMLDivElement>(null);
  const [llmProviders, setLlmProviders] = useState<LLMProviderInfo[]>([]);

  useEffect(() => {
    listLLMProviders()
      .then((data) => {
        if (!data.providers?.length) return;
        setLlmProviders(data.providers);
        // Async enrich with live model IDs
        listAllLiveModels().then((live) => {
          const platformModels = live.platforms ?? {};
          setLlmProviders((prev) =>
            prev.map((p) => {
              const platform = p.platform ?? p.id.split('-')[0];
              const liveEntry = platformModels[platform];
              if (liveEntry?.models?.length) return { ...p, models: liveEntry.models };
              return p;
            }),
          );
        }).catch(() => {});
      })
      .catch(() => {});
  }, []);

  const stepModelOptions = useMemo(() => buildStepModelOptions(llmProviders), [llmProviders]);
  const preliminaryModelOptions = useMemo(() => buildPreliminaryModelOptions(llmProviders), [llmProviders]);

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
            className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
          >
            <option value="auto">Auto (follow topic language)</option>
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
        </div>

        {/* Gap Query Intent */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1">
            Gap Query Intent
            <Tip content="Controls review preference for Round 2+ gap queries: Broad (default) avoids strict review bias; Prefer review lightly adds review intent; Reviews only applies strict review constraints where supported.">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <select
            value={deepResearchDefaults.gapQueryIntent}
            onChange={(e) => updateDeepResearchDefaults({ gapQueryIntent: e.target.value as 'broad' | 'review_pref' | 'reviews_only' })}
            className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
          >
            <option value="broad">Broad (default)</option>
            <option value="review_pref">Prefer review (soft)</option>
            <option value="reviews_only">Reviews only (strict)</option>
          </select>
        </div>

        {/* 初步研究 模型：仅 Perplexity，用于获取主题的初步认知（联网） */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1">
            初步研究 模型
            <Tip content="仅从 Perplexity/Sonar 中选择，用于获取主题的初步认知（联网），再据此生成澄清问题。">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <select
            value={deepResearchDefaults.preliminaryModel}
            onChange={(e) => updateDeepResearchDefaults({ preliminaryModel: e.target.value })}
            className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none mb-2"
          >
            {preliminaryModelOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* 提问 模型：用于生成澄清问题 */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1">
            提问 模型
            <Tip content="用于根据初步认知和聊天历史，生成后续的澄清问题。Default 使用全局模型。">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <select
            value={deepResearchDefaults.questionModel ?? ''}
            onChange={(e) => updateDeepResearchDefaults({ questionModel: e.target.value })}
            className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-[11px] bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none mb-2"
          >
            {stepModelOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Per-step Models */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1.5">
            Per-step Models
            <Tip content="各研究步骤使用的模型。plan/research/write 等为后续步骤。'Default' 使用页头全局模型。">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <div className="space-y-1 border border-gray-100 rounded-lg p-2.5 bg-gray-50">
            {RESEARCH_STEPS.map((step) => (
              <div key={step} className="grid grid-cols-[72px_1fr] items-center gap-1.5">
                <span className="text-[10px] font-medium text-gray-500 uppercase">{stepDisplayLabel(step)}</span>
                <select
                  value={deepResearchDefaults.stepModels[step] || ''}
                  onChange={(e) => setDeepResearchStepModel(step, e.target.value)}
                  className="border border-gray-200 rounded-md px-1.5 py-1 text-[10px] bg-white text-gray-900 focus:ring-1 focus:ring-indigo-400 outline-none"
                >
                  {stepModelOptions.map((opt) => (
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
            <Tip content="When OFF (default): before writing each section, the system first extracts 3–5 core claims from evidence with [ref:xxxx] citations, then expands them into prose. When ON: skip claim extraction and write directly from evidence. Use ON for faster runs or when you prefer free-form section writing.">
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

        {/* Max Sections */}
        <div>
          <label className="flex items-center gap-1 text-[11px] font-medium text-gray-600 mb-1.5">
            Max Sections (大纲章节数)
            <Tip content="控制生成大纲的最大章节数量。建议 2-6 节，默认 4 节。较少的章节适合聚焦性研究，较多章节适合全面综述。">
              <HelpCircle size={11} />
            </Tip>
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={2}
              max={6}
              step={1}
              value={deepResearchDefaults.maxSections}
              onChange={(e) => updateDeepResearchDefaults({ maxSections: parseInt(e.target.value, 10) })}
              className="flex-1 accent-indigo-500"
            />
            <span className="text-[11px] font-medium text-indigo-600 w-6 text-center">
              {deepResearchDefaults.maxSections}
            </span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-100 bg-gray-50 rounded-b-xl">
        <p className="text-[9px] text-gray-400">设置自动保存，跨会话持久化。在 Deep Research 对话内可临时覆盖。</p>
      </div>
    </div>
  );
}
