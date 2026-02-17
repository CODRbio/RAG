import { Settings, Brain, FileText, Cpu, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal } from '../ui/Modal';
import { useUIStore, useConfigStore, useToastStore } from '../../stores';

type CiteKeyFormat = 'author_date' | 'numeric' | 'hash';
type MergeLevel = 'document' | 'chunk';
type RerankerMode = 'cascade' | 'bge_only' | 'colbert_only';

/**
 * 高级配置 Modal：
 * - 引文格式
 * - 重排序策略
 * - 知识图谱
 */
export function SettingsModal() {
  const { t } = useTranslation();
  const { showSettingsModal, setShowSettingsModal } = useUIStore();
  const { ragConfig, updateRagConfig } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);

  // 高级设置状态（持久化到 localStorage）
  const [citeKeyFormat, setCiteKeyFormat] = useState<CiteKeyFormat>(
    () => (localStorage.getItem('adv_cite_key_format') as CiteKeyFormat) || 'author_date'
  );
  const [mergeLevel, setMergeLevel] = useState<MergeLevel>(
    () => (localStorage.getItem('adv_merge_level') as MergeLevel) || 'document'
  );
  const [rerankerMode, setRerankerMode] = useState<RerankerMode>(
    () => (localStorage.getItem('adv_reranker_mode') as RerankerMode) || 'cascade'
  );

  // 折叠区域
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['citation', 'reranker', 'graph'])
  );

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

  const SectionHeader = ({
    id,
    icon,
    title,
  }: {
    id: string;
    icon: React.ReactNode;
    title: string;
  }) => (
    <button
      type="button"
      className="w-full flex items-center gap-2 py-2 text-sm font-semibold text-slate-300 hover:text-slate-100"
      onClick={() => toggleSection(id)}
    >
      {expandedSections.has(id) ? (
        <ChevronDown size={14} className="text-slate-500" />
      ) : (
        <ChevronRight size={14} className="text-slate-500" />
      )}
      {icon}
      {title}
    </button>
  );

  return (
    <Modal
      open={showSettingsModal}
      onClose={() => setShowSettingsModal(false)}
      title={t('settings.advancedConfig')}
      icon={<Settings size={20} className="text-gray-700" />}
      maxWidth="max-w-md"
    >
      <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1">
        {/* ── 引文格式 ── */}
        <div className="border border-slate-700/50 rounded-lg p-3 bg-slate-800/30">
          <SectionHeader
            id="citation"
            icon={<FileText size={14} className="text-amber-600" />}
            title={t('settings.citationFormat')}
          />
          {expandedSections.has('citation') && (
            <div className="space-y-3 pt-2 pl-6">
              {/* cite_key 格式 */}
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">
                  {t('settings.citeKeyFormat')}
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(
                    [
                      { value: 'author_date', label: 'Author-Date', desc: 'Smith2023' },
                      { value: 'numeric', label: 'Numeric', desc: '[1], [2]' },
                      { value: 'hash', label: 'Hash', desc: 'a3f7b2...' },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setCiteKeyFormat(opt.value)}
                      className={`px-2 py-1.5 rounded text-[11px] border transition-all ${
                        citeKeyFormat === opt.value
                          ? 'bg-amber-900/20 border-amber-500/40 text-amber-300 font-medium'
                          : 'bg-slate-900/50 border-slate-700 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      <div>{opt.label}</div>
                      <div className="text-[9px] opacity-60 mt-0.5">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* 合并级别 */}
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">
                  {t('settings.mergeLevel')}
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {(
                    [
                      {
                        value: 'document',
                        label: t('settings.docLevel'),
                        desc: t('settings.docLevelDesc'),
                      },
                      {
                        value: 'chunk',
                        label: t('settings.chunkLevel'),
                        desc: t('settings.chunkLevelDesc'),
                      },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setMergeLevel(opt.value)}
                      className={`px-2 py-2 rounded text-[11px] border transition-all text-left ${
                        mergeLevel === opt.value
                          ? 'bg-amber-900/20 border-amber-500/40 text-amber-300 font-medium'
                          : 'bg-slate-900/50 border-slate-700 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      <div>{opt.label}</div>
                      <div className="text-[9px] opacity-60 mt-0.5">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── 重排序策略 ── */}
        <div className="border border-slate-700/50 rounded-lg p-3 bg-slate-800/30">
          <SectionHeader
            id="reranker"
            icon={<Cpu size={14} className="text-blue-600" />}
            title={t('settings.rerankerStrategy')}
          />
          {expandedSections.has('reranker') && (
            <div className="space-y-3 pt-2 pl-6">
              <div>
                <label className="block text-xs text-slate-400 mb-1.5">
                  {t('settings.rerankerMode')}
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(
                    [
                      {
                        value: 'cascade',
                        label: 'Cascade',
                        desc: 'BGE 粗排 + ColBERT 精排',
                      },
                      {
                        value: 'bge_only',
                        label: 'BGE Only',
                        desc: '仅 BGE-Reranker',
                      },
                      {
                        value: 'colbert_only',
                        label: 'ColBERT Only',
                        desc: '仅 ColBERT (多语言)',
                      },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setRerankerMode(opt.value)}
                      className={`px-2 py-1.5 rounded text-[11px] border transition-all ${
                        rerankerMode === opt.value
                          ? 'bg-sky-900/20 border-sky-500/40 text-sky-300 font-medium'
                          : 'bg-slate-900/50 border-slate-700 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      <div>{opt.label}</div>
                      <div className="text-[9px] opacity-60 mt-0.5">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-300">{t('settings.enableReranker')}</span>
                <button
                  type="button"
                  onClick={() =>
                    updateRagConfig({ enableReranker: !ragConfig.enableReranker })
                  }
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    ragConfig.enableReranker ? 'bg-sky-500' : 'bg-slate-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                      ragConfig.enableReranker
                        ? 'translate-x-4'
                        : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>
              <div className="text-[10px] text-slate-400 bg-slate-900/50 rounded px-2 py-1.5 border border-slate-700">
                ColBERT: <span className="font-mono">jinaai/jina-colbert-v2</span>{' '}
                (89 语言)
              </div>
            </div>
          )}
        </div>

        {/* ── 知识图谱 ── */}
        <div className="border border-slate-700/50 rounded-lg p-3 bg-slate-800/30">
          <SectionHeader
            id="graph"
            icon={<Brain size={14} className="text-purple-600" />}
            title={t('settings.knowledgeGraph')}
          />
          {expandedSections.has('graph') && (
            <div className="space-y-3 pt-2 pl-6">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-300">{t('settings.enableHippoRAG')}</span>
                  <p className="text-[9px] text-slate-500 mt-0.5">
                    {t('settings.hippoRAGDesc')}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() =>
                    updateRagConfig({ enableHippoRAG: !ragConfig.enableHippoRAG })
                  }
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    ragConfig.enableHippoRAG ? 'bg-purple-500' : 'bg-slate-600'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                      ragConfig.enableHippoRAG
                        ? 'translate-x-4'
                        : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 底部按钮 */}
      <div className="flex justify-end gap-2 mt-5 pt-4 border-t border-slate-700/50">
        <button
          type="button"
          onClick={() => setShowSettingsModal(false)}
          className="px-4 py-2 text-sm text-slate-400 hover:bg-slate-800 rounded-lg"
        >
          {t('common.cancel')}
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="px-4 py-2 text-sm bg-sky-600 text-white rounded-lg hover:bg-sky-500"
        >
          {t('common.save')}
        </button>
      </div>
    </Modal>
  );
}
