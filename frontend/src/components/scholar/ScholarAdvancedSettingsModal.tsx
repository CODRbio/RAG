import { useEffect, useMemo, useState } from 'react';
import { GripVertical, RefreshCw, Settings, Sparkles } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { listLLMProviders, type LLMProviderInfo } from '../../api/ingest';
import type { ScholarDownloaderDefaults } from '../../types';
import {
  DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS,
  useConfigStore,
} from '../../stores/useConfigStore';
import { useScholarStore } from '../../stores/useScholarStore';
import { useToastStore } from '../../stores/useToastStore';
import { Modal } from '../ui/Modal';

interface Props {
  open: boolean;
  onClose: () => void;
}

function scholarLlmGroupLabel(id: string): string {
  const base = id.split('-')[0].toLowerCase();
  const labels: Record<string, string> = {
    openai: 'OpenAI',
    deepseek: 'DeepSeek',
    claude: 'Claude',
    gemini: 'Gemini',
    kimi: 'Kimi',
    qwen: 'Qwen',
    perplexity: 'Perplexity',
    sonar: 'Perplexity',
  };
  return labels[base] || id;
}

function scholarLlmProviderOptionLabel(id: string): string {
  const base = scholarLlmGroupLabel(id);
  const suffix = id.split('-').slice(1).join('-');
  return suffix ? `${base} · ${suffix}` : base;
}

function moveItem<T>(items: T[], fromIndex: number, toIndex: number): T[] {
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
}

export function ScholarAdvancedSettingsModal({ open, onClose }: Props) {
  const { t } = useTranslation();
  const scholarDownloaderDefaults = useConfigStore((s) => s.scholarDownloaderDefaults);
  const updateScholarDownloaderDefaults = useConfigStore((s) => s.updateScholarDownloaderDefaults);
  const resetScholarDownloaderDefaults = useConfigStore((s) => s.resetScholarDownloaderDefaults);
  const applyScholarDownloaderDefaults = useScholarStore((s) => s.applyScholarDownloaderDefaults);
  const scholarHealth = useScholarStore((s) => s.scholarHealth);
  const addToast = useToastStore((s) => s.addToast);

  const [draft, setDraft] = useState<ScholarDownloaderDefaults>(scholarDownloaderDefaults);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);
  const [llmProviders, setLlmProviders] = useState<LLMProviderInfo[]>([]);
  const [isRefreshingLlm, setIsRefreshingLlm] = useState(false);

  useEffect(() => {
    if (!open) return;
    setDraft({
      ...scholarDownloaderDefaults,
      strategyOrder: [...scholarDownloaderDefaults.strategyOrder],
    });
  }, [open, scholarDownloaderDefaults]);

  const loadLlmProviders = async () => {
    setIsRefreshingLlm(true);
    try {
      const data = await listLLMProviders();
      setLlmProviders(data.providers || []);
    } catch {
      addToast(t('scholar.advancedSettingsLoadModelsError'), 'error');
    } finally {
      setIsRefreshingLlm(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    loadLlmProviders();
  }, [open]);

  const groupedProviders = useMemo(
    () =>
      Array.from(
        llmProviders.reduce((acc, provider) => {
          const group = scholarLlmGroupLabel(provider.id);
          if (!acc.has(group)) acc.set(group, []);
          acc.get(group)!.push(provider);
          return acc;
        }, new Map<string, LLMProviderInfo[]>()),
      ),
    [llmProviders],
  );

  const strategyItems = useMemo(
    () =>
      draft.strategyOrder.map((id) => ({
        id,
        label: t(`scholar.strategy.${id}.label`),
        description: t(`scholar.strategy.${id}.description`),
      })),
    [draft.strategyOrder, t],
  );

  const handleProviderChange = (providerId: string) => {
    const provider = llmProviders.find((item) => item.id === providerId);
    setDraft((prev) => ({
      ...prev,
      llmProvider: providerId,
      llmModel: providerId ? provider?.default_model ?? '' : '',
    }));
  };

  const handleSave = () => {
    updateScholarDownloaderDefaults(draft);
    applyScholarDownloaderDefaults(draft);
    addToast(t('scholar.advancedSettingsSaved'), 'success');
    onClose();
  };

  const handleReset = () => {
    const configDefaultStrategyOrder = scholarHealth?.default_strategy_order?.length
      ? [...scholarHealth.default_strategy_order]
      : [...DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS.strategyOrder];
    const defaults = {
      ...DEFAULT_SCHOLAR_DOWNLOADER_DEFAULTS,
      strategyOrder: configDefaultStrategyOrder,
    };
    resetScholarDownloaderDefaults({ strategyOrder: configDefaultStrategyOrder });
    applyScholarDownloaderDefaults(defaults);
    setDraft(defaults);
    addToast(t('scholar.advancedSettingsReset'), 'success');
    onClose();
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t('scholar.advancedSettingsTitle')}
      icon={<Settings size={18} className="text-sky-300" />}
      maxWidth="max-w-2xl"
      variant="dark"
    >
      <div className="space-y-5 text-slate-200">
        <p className="text-sm text-slate-400">
          {t('scholar.advancedSettingsDescription')}
        </p>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-100">{t('scholar.includeAcademiaShort')}</div>
                <div className="text-xs text-slate-400 mt-1">{t('scholar.includeAcademia')}</div>
              </div>
              <button
                type="button"
                onClick={() => setDraft((prev) => ({ ...prev, includeAcademia: !prev.includeAcademia }))}
                className={`relative h-6 w-11 rounded-full transition-colors ${
                  draft.includeAcademia ? 'bg-sky-500' : 'bg-slate-600'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                    draft.includeAcademia ? 'translate-x-5' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>

            <div className="space-y-3 rounded-lg border border-slate-700/80 bg-slate-900/60 p-3">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                    <Sparkles size={14} className="text-rose-300" />
                    {t('scholar.assistLlmToggle')}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">{t('scholar.assistLlmHint')}</div>
                </div>
                <button
                  type="button"
                  onClick={() => setDraft((prev) => ({ ...prev, assistLlmEnabled: !prev.assistLlmEnabled }))}
                  className={`relative h-6 w-11 rounded-full transition-colors ${
                    draft.assistLlmEnabled ? 'bg-rose-500' : 'bg-slate-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                      draft.assistLlmEnabled ? 'translate-x-5' : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>

              <div className={draft.assistLlmEnabled ? 'space-y-3' : 'space-y-3 opacity-50'}>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    {t('scholar.llmForDownloadLabel')}
                  </label>
                  <div className="flex items-center gap-2">
                    <select
                      value={draft.llmProvider}
                      onChange={(e) => handleProviderChange(e.target.value)}
                      disabled={!draft.assistLlmEnabled}
                      className="flex-1 rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:cursor-not-allowed"
                    >
                      <option value="">{t('scholar.llmDefault')}</option>
                      {groupedProviders.map(([group, providers]) => (
                        <optgroup key={group} label={group}>
                          {providers.map((provider) => (
                            <option key={provider.id} value={provider.id}>
                              {scholarLlmProviderOptionLabel(provider.id)}
                            </option>
                          ))}
                        </optgroup>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => loadLlmProviders()}
                      disabled={isRefreshingLlm}
                      className="rounded-lg border border-slate-600 bg-slate-800 p-2 text-slate-300 hover:text-sky-300 hover:border-slate-500 disabled:opacity-60"
                      title={t('scholar.llmRefresh')}
                    >
                      <RefreshCw size={14} className={isRefreshingLlm ? 'animate-spin' : ''} />
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    {t('scholar.llmModelLabel')}
                  </label>
                  <select
                    value={draft.llmModel}
                    onChange={(e) => setDraft((prev) => ({ ...prev, llmModel: e.target.value }))}
                    disabled={!draft.assistLlmEnabled || !draft.llmProvider}
                    className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:cursor-not-allowed"
                  >
                    <option value="">{t('scholar.llmModelDefault')}</option>
                    {(llmProviders.find((provider) => provider.id === draft.llmProvider)?.models ?? []).map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                {t('scholar.browserModeLabel')}
              </label>
              <select
                value={draft.browserMode}
                onChange={(e) => setDraft((prev) => ({ ...prev, browserMode: e.target.value as 'headed' | 'headless' }))}
                className="w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-1 focus:ring-sky-500"
              >
                <option value="headed">{t('scholar.browserModeHeaded')}</option>
                <option value="headless">{t('scholar.browserModeHeadless')}</option>
              </select>
            </div>
          </div>

          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4">
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <div className="text-sm font-semibold text-slate-100">
                  {t('scholar.strategyOrderTitle')}
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  {t('scholar.strategyOrderHint')}
                </div>
              </div>
              <span className="text-[11px] uppercase tracking-wide text-slate-500">
                {t('scholar.dragToReorder')}
              </span>
            </div>

            <div className="space-y-2">
              {strategyItems.map((item, idx) => (
                <div key={item.id} className="space-y-2">
                  {dragOverIndex === idx && draggingIndex !== null && (
                    <div className="h-0.5 rounded-full bg-sky-500" />
                  )}
                  <div
                    draggable
                    onDragStart={() => setDraggingIndex(idx)}
                    onDragEnd={() => {
                      setDraggingIndex(null);
                      setDragOverIndex(null);
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOverIndex(idx);
                    }}
                    onDrop={() => {
                      if (draggingIndex === null || draggingIndex === idx) return;
                      setDraft((prev) => ({
                        ...prev,
                        strategyOrder: moveItem(prev.strategyOrder, draggingIndex, idx),
                      }));
                      setDraggingIndex(null);
                      setDragOverIndex(null);
                    }}
                    className={`flex items-center gap-3 rounded-xl border px-3 py-3 transition-colors ${
                      draggingIndex === idx
                        ? 'border-sky-500/60 bg-sky-500/10 opacity-70'
                        : 'border-slate-700 bg-slate-900/60'
                    }`}
                  >
                    <button
                      type="button"
                      className="rounded-md border border-slate-700 p-2 text-slate-400 cursor-grab active:cursor-grabbing"
                      title={t('scholar.dragToReorder')}
                    >
                      <GripVertical size={14} />
                    </button>
                    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-700 text-xs font-semibold text-slate-200">
                      {idx + 1}
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-slate-100">{item.label}</div>
                      <div className="text-xs text-slate-400 mt-1">{item.description}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3 pt-2">
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-medium text-slate-200 hover:border-slate-500 hover:bg-slate-800"
          >
            {t('scholar.resetAdvancedSettings')}
          </button>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-medium text-slate-200 hover:border-slate-500 hover:bg-slate-800"
            >
              {t('common.cancel')}
            </button>
            <button
              type="button"
              onClick={handleSave}
              className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500"
            >
              {t('common.save')}
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
