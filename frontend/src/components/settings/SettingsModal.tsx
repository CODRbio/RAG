import { Settings, Brain, FileText, Cpu, ChevronDown, ChevronRight, Zap, Database, Trash2, HardDrive } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal } from '../ui/Modal';
import { useUIStore, useConfigStore, useToastStore } from '../../stores';
import { listLLMProviders, listUltraLiteProviders, type LLMProviderInfo, type UltraLiteProviderOption } from '../../api/ingest';
import { getDatabaseConfig, updateDatabaseConfig, DEFAULT_DATABASE_URL, clearCrossrefCache, clearPaperMetadataCache } from '../../api/config';

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
  const [intentProviderOptions, setIntentProviderOptions] = useState<LLMProviderInfo[]>([]);

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
    new Set(['citation', 'reranker', 'graph', 'ultralite', 'database'])
  );
  const [databaseUrl, setDatabaseUrl] = useState<string>(DEFAULT_DATABASE_URL);

  const [crossrefDays, setCrossrefDays] = useState<number>(0);
  const [clearingCrossref, setClearingCrossref] = useState(false);
  const [clearingPaperMeta, setClearingPaperMeta] = useState(false);

  useEffect(() => {
    if (!showSettingsModal) return;
    getDatabaseConfig()
      .then((c) => setDatabaseUrl(c.url || DEFAULT_DATABASE_URL))
      .catch(() => setDatabaseUrl(DEFAULT_DATABASE_URL));
    listLLMProviders()
      .then((data) => {
        setIntentProviderOptions((data.providers || []).filter((p) => !p.id.endsWith('-vision')));
      })
      .catch(() => {});
    listUltraLiteProviders(false)
      .then((data) => {
        setUltraLiteOptions(data.providers || []);
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

  const handleSave = async () => {
    localStorage.setItem('adv_cite_key_format', citeKeyFormat);
    localStorage.setItem('adv_merge_level', mergeLevel);
    localStorage.setItem('adv_reranker_mode', rerankerMode);
    const urlToSave = databaseUrl.trim() || DEFAULT_DATABASE_URL;
    try {
      await updateDatabaseConfig(urlToSave);
    } catch {
      addToast(t('settings.databaseSaveError'), 'error');
      return;
    }
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

  const formatIntentProviderLabel = (provider: LLMProviderInfo): string => {
    const platform = (provider.platform || '').trim();
    if (provider.label && provider.label !== provider.id) return `${provider.label} (${provider.id})`;
    if (platform) return `${provider.id} (${platform})`;
    return provider.id;
  };

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

        {/* ── Intent / Ultra Lite ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="ultralite"
            icon={<Zap size={14} className="text-amber-400" />}
            title={t('settings.intentAndUltraLite')}
          />
          {expandedSections.has('ultralite') && (
            <div className="space-y-2 pt-2 pl-5">
              <label className="block text-xs text-slate-300 mb-1.5">{t('settings.intentProvider')}</label>
              <select
                value={deepResearchDefaults.intent_provider ?? ''}
                onChange={(e) => updateDeepResearchDefaults({ intent_provider: e.target.value || null })}
                className="w-full rounded border border-slate-600 bg-slate-800 text-slate-200 text-sm px-2.5 py-1.5 focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none"
              >
                <option value="">{t('settings.intentProviderUseDefault')}</option>
                {intentProviderOptions.map((p) => (
                  <option key={p.id} value={p.id}>
                    {formatIntentProviderLabel(p)}
                  </option>
                ))}
              </select>
              <p className="text-[10px] text-slate-400">{t('settings.intentProviderDesc')}</p>

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

        {/* ── Graphic Abstract ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="graphic_abstract"
            icon={<Brain size={14} className="text-pink-400" />}
            title="Graphic Abstract"
          />
          {expandedSections.has('graphic_abstract') && (
            <div className="space-y-3 pt-2 pl-5">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm text-slate-200">生成图文摘要</span>
                  <p className="text-[10px] text-slate-400 mt-0.5">在回答或研究报告末尾自动生成总结图像海报</p>
                </div>
                <Toggle
                  checked={ragConfig.enableGraphicAbstract ?? false}
                  onChange={() => updateRagConfig({ enableGraphicAbstract: !ragConfig.enableGraphicAbstract })}
                  color="bg-pink-500"
                />
              </div>
              {ragConfig.enableGraphicAbstract && (
                <div>
                  <label className="block text-xs text-slate-300 mb-1.5">图像模型</label>
                  <select
                    value={ragConfig.graphicAbstractModel ?? 'nanobanana 2'}
                    onChange={(e) => updateRagConfig({ graphicAbstractModel: e.target.value })}
                    className="w-full rounded border border-slate-600 bg-slate-800 text-slate-200 text-sm px-2.5 py-1.5 focus:ring-2 focus:ring-pink-500 focus:border-pink-500 outline-none"
                  >
                    <option value="nanobanana 2">nanobanana 2 (Gemini 2.5 Flash Image)</option>
                    <option value="nanobanana pro">nanobanana pro (Gemini 3 Pro Image)</option>
                    <option value="gpt-image-1.5">gpt-image-1.5 (GPT)</option>
                    <option value="qwen-image-2.0">qwen-image-2.0 (Qwen)</option>
                  </select>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── 数据库 ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="database"
            icon={<Database size={14} className="text-emerald-400" />}
            title={t('settings.database')}
          />
          {expandedSections.has('database') && (
            <div className="space-y-2 pt-2 pl-5">
              <label className="block text-xs text-slate-300 mb-1.5">{t('settings.databaseUrl')}</label>
              <input
                type="text"
                value={databaseUrl}
                onChange={(e) => setDatabaseUrl(e.target.value)}
                placeholder={t('settings.databaseUrlPlaceholder')}
                className="w-full rounded border border-slate-600 bg-slate-800 text-slate-200 text-sm px-2.5 py-1.5 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
              />
              <p className="text-[10px] text-slate-400">{t('settings.databaseUrlRestart')}</p>
            </div>
          )}
        </div>

        {/* ── 存储与缓存 ── */}
        <div className="border border-slate-700 rounded-lg p-3 bg-slate-800/50">
          <SectionHeader
            id="cache"
            icon={<HardDrive size={14} className="text-rose-400" />}
            title="存储与缓存清理"
          />
          {expandedSections.has('cache') && (
            <div className="space-y-4 pt-2 pl-5">
              <p className="text-[10px] text-slate-400 leading-relaxed bg-slate-900/60 rounded px-2 py-1.5 border border-slate-700">
                以下缓存为有价值的持久缓存，正常使用时无需清理。仅在数据异常或需要释放空间时手动执行。
              </p>

              {/* Crossref 缓存 */}
              <div>
                <div className="text-xs text-slate-300 mb-2 font-medium">Crossref / DOI 元数据缓存</div>
                <p className="text-[10px] text-slate-400 mb-2">
                  存储从 Crossref API 拉取的文献标题、作者、期刊等信息，删除后将在下次访问时重新拉取。
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1.5 flex-1">
                    <span className="text-[11px] text-slate-400 whitespace-nowrap">仅删除超过</span>
                    <input
                      type="number"
                      min={0}
                      value={crossrefDays}
                      onChange={(e) => setCrossrefDays(Math.max(0, parseInt(e.target.value) || 0))}
                      className="w-14 text-xs bg-slate-900 border border-slate-600 text-slate-200 rounded px-2 py-1 focus:border-rose-500 focus:outline-none"
                    />
                    <span className="text-[11px] text-slate-400 whitespace-nowrap">天的条目（0 = 清空全部）</span>
                  </div>
                  <button
                    type="button"
                    disabled={clearingCrossref}
                    onClick={async () => {
                      setClearingCrossref(true);
                      try {
                        const r = await clearCrossrefCache(crossrefDays);
                        const total = (r.crossref_cache_deleted ?? 0) + (r.crossref_cache_by_doi_deleted ?? 0);
                        addToast(`已清理 ${total} 条 Crossref 缓存`, 'success');
                      } catch {
                        addToast('Crossref 缓存清理失败', 'error');
                      } finally {
                        setClearingCrossref(false);
                      }
                    }}
                    className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] bg-rose-900/40 border border-rose-600/50 text-rose-300 hover:bg-rose-800/50 disabled:opacity-50 whitespace-nowrap"
                  >
                    <Trash2 size={11} />
                    {clearingCrossref ? '清理中…' : '清理'}
                  </button>
                </div>
              </div>

              {/* 分割线 */}
              <div className="border-t border-slate-700/60" />

              {/* Paper Metadata */}
              <div>
                <div className="text-xs text-slate-300 mb-2 font-medium">论文元数据缓存（paper_metadata）</div>
                <p className="text-[10px] text-slate-400 mb-2">
                  存储入库文献的 DOI、标题、作者等富化信息，用于文献列表显示。清空后已入库文献仍可检索，但元数据需重新入库或手动补全。
                </p>
                <button
                  type="button"
                  disabled={clearingPaperMeta}
                  onClick={async () => {
                    if (!window.confirm('确认清空全部 paper_metadata？此操作不可恢复。')) return;
                    setClearingPaperMeta(true);
                    try {
                      const r = await clearPaperMetadataCache();
                      addToast(`已清空 ${r.paper_metadata_deleted ?? 0} 条论文元数据`, 'success');
                    } catch {
                      addToast('元数据清理失败', 'error');
                    } finally {
                      setClearingPaperMeta(false);
                    }
                  }}
                  className="flex items-center gap-1 px-2.5 py-1 rounded text-[11px] bg-rose-900/40 border border-rose-600/50 text-rose-300 hover:bg-rose-800/50 disabled:opacity-50"
                >
                  <Trash2 size={11} />
                  {clearingPaperMeta ? '清理中…' : '清空全部元数据'}
                </button>
              </div>
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
              {ragConfig.enableHippoRAG && (
                <div className="mt-3 flex items-center justify-between gap-2">
                  <div>
                    <span className="text-sm text-slate-200">{t('settings.graphTopK')}</span>
                    <p className="text-[10px] text-slate-400 mt-0.5">{t('settings.graphTopKDesc')}</p>
                  </div>
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={ragConfig.graphTopK}
                    onChange={(e) =>
                      updateRagConfig({ graphTopK: Math.max(1, Math.min(200, Number(e.target.value) || 1)) })
                    }
                    className="w-16 text-[10px] bg-slate-950 border border-slate-700 text-slate-300 rounded px-2 py-1 focus:border-purple-500 focus:outline-none"
                  />
                </div>
              )}
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
