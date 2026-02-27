import { Settings, Brain, FileText, Cpu, ChevronDown, ChevronRight, Zap } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal } from '../ui/Modal';
import { useUIStore, useConfigStore, useToastStore } from '../../stores';
import { listUltraLiteProviders, type UltraLiteProviderOption } from '../../api/ingest';

type CiteKeyFormat = 'author_date' | 'numeric' | 'hash';
type MergeLevel = 'document' | 'chunk';
type RerankerMode = 'cascade' | 'bge_only' | 'colbert_only';

/**
 * 高级配置 Modal（暗色主题）：
 * - 引文格式
 * - 重排序策略（含开关 + 模式选择）
 * - 知识图谱
 */
export function SettingsModal() {
  const { t } = useTranslation();
  const { showSettingsModal, setShowSettingsModal } = useUIStore();
  const { ragConfig, updateRagConfig, deepResearchDefaults, updateDeepResearchDefaults } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);
  const [ultraLiteOptions, setUltraLiteOptions] = useState<UltraLiteProviderOption[]>([]);
  const [ultraLiteDefault, setUltraLiteDefault] = useState<string | null>(null);

  const [citeKeyFormat, setCiteKeyFormat] = useState<CiteKeyFormat>(
    () => (localStorage.getItem('adv_cite_key_format') as CiteKeyFormat) || 'author_date'
  );
  const [mergeLevel, setMergeLevel] = useState<MergeLevel>(
    () => (localStorage.getItem('adv_merge_level') as MergeLevel) || 'document'
  );
  const [rerankerMode, setRerankerMode] = useState<RerankerMode>(
    () => (localStorage.getItem('adv_reranker_mode') as RerankerMode) || 'cascade'
  );

  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['citation', 'reranker', 'graph', 'ultralite'])
  );

  useEffect(() => {
    if (!showSettingsModal) return;
    listUltraLiteProviders(false)
      .then((data) => {
        setUltraLiteOptions(data.providers || []);
        setUltraLiteDefault(data.default ?? null);
      })
      .catch(() => {});
  }, [showSettingsModal]);

  const toggleSection = (id: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSave = () => {
    localStorage.setItem('adv_cite_key_format', citeKeyFormat);
    localStorage.setItem('adv_merge_level', mergeLevel);
    localStorage.setItem('adv_reranker_mode', rerankerMode);
    addToast(t('settings.saved'), 'success');
    setShowSettingsModal(false);
  };

  const SectionHeader = ({ id, icon, title }: { id: string; icon: React.ReactNode; title: string }) => (
    <button
      type="button"
      className="w-full flex items-center gap-2 py-2 text-sm font-semibold text-slate-100 hover:text-white"
      onClick={() => toggleSection(id)}
    >
      {expandedSections.has(id)
        ? <ChevronDown size={14} className="text-slate-400" />
        : <ChevronRight size={14} className="text-slate-400" />}
      {icon}
      {title}
    </button>
  );

  /* ── reusable option button ── */
  const OptionBtn = ({
    active,
    onClick,
    label,
    desc,
    activeClass,
  }: {
    active: boolean;
    onClick: () => void;
    label: string;
    desc: string;
    activeClass: string;
  }) => (
    <button
      type="button"
      onClick={onClick}
      className={`px-2 py-1.5 rounded text-[11px] border transition-all text-left ${
        active
          ? activeClass
          : 'bg-slate-800 border-slate-600 text-slate-200 hover:border-slate-500 hover:bg-slate-700'
      }`}
    >
      <div className="font-medium">{label}</div>
      <div className="text-[9px] mt-0.5 opacity-75">{desc}</div>
    </button>
  );

  /* ── toggle button ── */
  const Toggle = ({ checked, onChange, color = 'bg-sky-500' }: { checked: boolean; onChange: () => void; color?: string }) => (
    <button
      type="button"
      onClick={onChange}
      className={`relative w-9 h-5 rounded-full transition-colors ${checked ? color : 'bg-slate-600'}`}
    >
      <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-4' : 'translate-x-0.5'}`} />
    </button>
  );

  return (
    <Modal
      open={showSettingsModal}
      onClose={() => setShowSettingsModal(false)}
      title={t('settings.advancedConfig')}
      icon={<Settings size={20} className="text-slate-300" />}
      maxWidth="max-w-md"
      variant="dark"
    >
      <div className="space-y-3 max-h-[65vh] overflow-y-auto pr-1">

        {/* ── 引文格式 ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="citation"
            icon={<FileText size={14} className="text-amber-400" />}
            title={t('settings.citationFormat')}
          />
          {expandedSections.has('citation') && (
            <div className="space-y-3 pt-2 pl-5">
              <div>
                <label className="block text-xs text-slate-300 mb-1.5">{t('settings.citeKeyFormat')}</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {([
                    { value: 'author_date', label: 'Author-Date', desc: 'Smith 2023' },
                    { value: 'numeric',     label: 'Numeric',     desc: '[1], [2]'  },
                    { value: 'hash',        label: 'Hash',        desc: 'a3f7b2...' },
                  ] as const).map((opt) => (
                    <OptionBtn
                      key={opt.value}
                      active={citeKeyFormat === opt.value}
                      onClick={() => setCiteKeyFormat(opt.value)}
                      label={opt.label}
                      desc={opt.desc}
                      activeClass="bg-amber-900/40 border-amber-500/60 text-amber-200 font-medium"
                    />
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1.5">{t('settings.mergeLevel')}</label>
                <div className="grid grid-cols-2 gap-2">
                  {([
                    { value: 'document', label: t('settings.docLevel'),   desc: t('settings.docLevelDesc')   },
                    { value: 'chunk',    label: t('settings.chunkLevel'),  desc: t('settings.chunkLevelDesc') },
                  ] as const).map((opt) => (
                    <OptionBtn
                      key={opt.value}
                      active={mergeLevel === opt.value}
                      onClick={() => setMergeLevel(opt.value)}
                      label={opt.label}
                      desc={opt.desc}
                      activeClass="bg-amber-900/40 border-amber-500/60 text-amber-200 font-medium"
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── 重排序策略 ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="reranker"
            icon={<Cpu size={14} className="text-sky-400" />}
            title={t('settings.rerankerStrategy')}
          />
          {expandedSections.has('reranker') && (
            <div className="space-y-3 pt-2 pl-5">
              {/* Enable toggle */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-200">{t('settings.enableReranker')}</span>
                <Toggle
                  checked={ragConfig.enableReranker}
                  onChange={() => updateRagConfig({ enableReranker: !ragConfig.enableReranker })}
                  color="bg-sky-500"
                />
              </div>

              {/* Mode selector — only relevant when enabled */}
              <div className={ragConfig.enableReranker ? '' : 'opacity-40 pointer-events-none'}>
                <label className="block text-xs text-slate-300 mb-1.5">{t('settings.rerankerMode')}</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {([
                    { value: 'cascade',      label: 'Cascade',      desc: 'BGE → ColBERT' },
                    { value: 'bge_only',     label: 'BGE Only',     desc: 'BGE cross-encoder' },
                    { value: 'colbert_only', label: 'ColBERT Only', desc: '89 langs' },
                  ] as const).map((opt) => (
                    <OptionBtn
                      key={opt.value}
                      active={rerankerMode === opt.value}
                      onClick={() => setRerankerMode(opt.value)}
                      label={opt.label}
                      desc={opt.desc}
                      activeClass="bg-sky-900/50 border-sky-500/60 text-sky-200 font-medium"
                    />
                  ))}
                </div>
              </div>

              <p className="text-[10px] text-slate-400 bg-slate-900/60 rounded px-2 py-1.5 border border-slate-700 leading-relaxed">
                {t('settings.rerankerScope')}
              </p>
            </div>
          )}
        </div>

        {/* ── Ultra Lite（长文本压缩） ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="ultralite"
            icon={<Zap size={14} className="text-amber-400" />}
            title={t('settings.ultraLite')}
          />
          {expandedSections.has('ultralite') && (
            <div className="space-y-2 pt-2 pl-5">
              <label className="block text-xs text-slate-300 mb-1.5">{t('settings.ultraLiteProvider')}</label>
              <select
                value={deepResearchDefaults.ultra_lite_provider ?? ''}
                onChange={(e) => updateDeepResearchDefaults({ ultra_lite_provider: e.target.value || null })}
                className="w-full rounded border border-slate-600 bg-slate-800 text-slate-200 text-sm px-2.5 py-1.5 focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none"
              >
                <option value="">{t('settings.ultraLiteUseDefault')}</option>
                {ultraLiteOptions.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
              <p className="text-[10px] text-slate-400">{t('settings.ultraLiteDesc')}</p>
            </div>
          )}
        </div>

        {/* ── 知识图谱 ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="graph"
            icon={<Brain size={14} className="text-purple-400" />}
            title={t('settings.knowledgeGraph')}
          />
          {expandedSections.has('graph') && (
            <div className="pt-2 pl-5">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm text-slate-200">{t('settings.enableHippoRAG')}</span>
                  <p className="text-[10px] text-slate-400 mt-0.5">{t('settings.hippoRAGDesc')}</p>
                </div>
                <Toggle
                  checked={ragConfig.enableHippoRAG}
                  onChange={() => updateRagConfig({ enableHippoRAG: !ragConfig.enableHippoRAG })}
                  color="bg-purple-500"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 底部按钮 */}
      <div className="flex justify-end gap-2 mt-5 pt-4 border-t border-slate-700">
        <button
          type="button"
          onClick={() => setShowSettingsModal(false)}
          className="px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 rounded-lg transition-colors"
        >
          {t('common.cancel')}
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="px-4 py-2 text-sm bg-sky-600 text-white rounded-lg hover:bg-sky-500 transition-colors"
        >
          {t('common.save')}
        </button>
      </div>
    </Modal>
  );
}
