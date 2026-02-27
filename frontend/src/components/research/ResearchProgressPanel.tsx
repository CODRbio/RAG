/**
 * Deep Research 实时进度面板。
 *
 * 显示 ReCAP Dashboard 的实时状态：
 * - 当前研究阶段 + 各章节进度
 * - 来源数量和置信度
 * - 信息缺口和冲突
 * - 整体进度条
 */

import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import type { ResearchDashboardData, ResearchSectionStatus } from '../../types';
import { restartDeepResearchSection, optimizeSectionEvidence } from '../../api/chat';
import { useChatStore, useToastStore } from '../../stores';
import {
  DEEP_RESEARCH_JOB_KEY,
  DEEP_RESEARCH_ARCHIVED_JOBS_KEY,
} from '../workflow/deep-research/types';

interface Props {
  dashboard: ResearchDashboardData | null;
  isActive: boolean;
}

const statusConfig: Record<string, { icon: string; label: string; color: string }> = {
  pending: { icon: '⬜', label: 'research.pending', color: 'text-gray-400' },
  researching: { icon: '🔍', label: 'research.researching', color: 'text-blue-500' },
  writing: { icon: '✍️', label: 'research.writing', color: 'text-orange-500' },
  reviewing: { icon: '🔄', label: 'research.reviewing', color: 'text-purple-500' },
  done: { icon: '✅', label: 'research.done', color: 'text-green-500' },
};

const confidenceConfig: Record<string, { label: string; color: string; bg: string }> = {
  low: { label: 'research.confidenceLow', color: 'text-red-600', bg: 'bg-red-100' },
  medium: { label: 'research.confidenceMedium', color: 'text-yellow-600', bg: 'bg-yellow-100' },
  high: { label: 'research.confidenceHigh', color: 'text-green-600', bg: 'bg-green-100' },
};

function SectionRow({
  section,
  canRestart,
  onRestart,
  onOptimize,
}: {
  section: ResearchSectionStatus;
  canRestart: boolean;
  onRestart: (sectionTitle: string, action: 'research' | 'write') => void;
  onOptimize: (sectionTitle: string) => void;
}) {
  const { t } = useTranslation();
  const cfg = statusConfig[section.status] || statusConfig.pending;
  const coveragePct = Math.round(section.coverage_score * 100);
  const canOptimize = canRestart && (section.status === 'done' || section.status === 'writing' || section.evidence_scarce);

  return (
    <div className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-gray-50 text-sm">
      <span className="w-5 text-center">{cfg.icon}</span>
      <span className="flex-1 truncate font-medium text-gray-700">
        {section.title}
        {section.evidence_scarce && (
          <span className="ml-1 text-[9px] text-red-400 font-normal">({t('research.evidenceScarce')})</span>
        )}
      </span>
      <span className={`text-xs ${cfg.color}`}>{t(cfg.label)}</span>
      {section.status !== 'pending' && (
        <span className="text-xs text-gray-400 w-10 text-right">{coveragePct}%</span>
      )}
      {section.source_count > 0 && (
        <span className="text-xs text-gray-400 w-12 text-right">{t('research.sourcesCount', { count: section.source_count })}</span>
      )}
      {canRestart && (
        <div className="flex items-center gap-1">
          <button
            onClick={() => onRestart(section.title, 'research')}
            className="px-1.5 py-0.5 text-[10px] border rounded text-blue-700 hover:bg-blue-50"
          >
            Re-search
          </button>
          <button
            onClick={() => onRestart(section.title, 'write')}
            className="px-1.5 py-0.5 text-[10px] border rounded text-amber-700 hover:bg-amber-50"
          >
            Re-write
          </button>
          {canOptimize && (
            <button
              onClick={() => onOptimize(section.title)}
              title={t('research.optimizeEvidenceHint')}
              className="px-1.5 py-0.5 text-[10px] border rounded text-purple-700 hover:bg-purple-50"
            >
              {t('research.optimizeEvidence')}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function ResearchProgressPanel({ dashboard, isActive }: Props) {
  const { t } = useTranslation();
  const addToast = useToastStore((s) => s.addToast);
  const { setShowDeepResearchDialog, setDeepResearchActive, setDeepResearchTopic } = useChatStore();
  const [restarting, setRestarting] = useState(false);
  const [optimizing, setOptimizing] = useState<string | null>(null);

  if (!dashboard) {
    if (!isActive) return null;
    return (
      <div className="border rounded-lg bg-white p-4 text-sm text-gray-400 text-center">
        {t('research.waiting')}
      </div>
    );
  }

  const progressPct = Math.round(dashboard.progress * 100);
  const coveragePct = Math.round(dashboard.coverage * 100);
  const confCfg = confidenceConfig[dashboard.confidence] || confidenceConfig.low;
  const canRestart = !restarting && !optimizing;

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

  const handleOptimizeSection = async (sectionTitle: string) => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('No Deep Research job found for optimization', 'warning');
      return;
    }
    setOptimizing(sectionTitle);
    try {
      const result = await optimizeSectionEvidence(sourceJobId, { section_title: sectionTitle });
      addToast(
        t('research.optimizeEvidenceDone', { count: result.new_chunks, section: sectionTitle }),
        'success',
      );
    } catch (err) {
      console.error('[DeepResearch] evidence optimization failed:', err);
      addToast(t('research.optimizeEvidenceFailed'), 'error');
    } finally {
      setOptimizing(null);
    }
  };

  const handleRestartSection = async (sectionTitle: string, action: 'research' | 'write') => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('No Deep Research job found for restart', 'warning');
      return;
    }
    setRestarting(true);
    try {
      const resp = await restartDeepResearchSection(sourceJobId, {
        section_title: sectionTitle,
        action,
      });
      localStorage.setItem(DEEP_RESEARCH_JOB_KEY, resp.job_id);
      setDeepResearchTopic(dashboard.topic || sectionTitle);
      setDeepResearchActive(true);
      setShowDeepResearchDialog(true);
      addToast(`Restart submitted for ${sectionTitle}`, 'success');
    } catch (err) {
      console.error('[DeepResearch] section restart failed:', err);
      addToast('Section restart failed', 'error');
    } finally {
      setRestarting(false);
    }
  };

  return (
    <div className="border rounded-lg bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-indigo-50 to-blue-50 border-b">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">
            {t('research.progressTitle')}
          </h3>
          <span className={`text-xs px-2 py-0.5 rounded-full ${confCfg.bg} ${confCfg.color} font-medium`}>
            {t('research.confidence')}: {t(confCfg.label)}
          </span>
        </div>
        {dashboard.topic && (
          <p className="text-xs text-gray-500 mt-0.5 truncate">{dashboard.topic}</p>
        )}
      </div>

      {/* Progress Bar */}
      <div className="px-4 pt-3">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>{t('research.overallProgress')}</span>
          <span>{progressPct}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-indigo-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Stats */}
      <div className="px-4 py-2 flex gap-4 text-xs text-gray-500">
        <span>{t('research.sources')}: <strong className="text-gray-700">{dashboard.total_sources}</strong></span>
        <span>{t('research.coverage')}: <strong className="text-gray-700">{coveragePct}%</strong></span>
        <span>{t('research.iterations')}: <strong className="text-gray-700">{dashboard.total_iterations}</strong></span>
      </div>

      {/* Sections */}
      {(dashboard.sections?.length ?? 0) > 0 && (
        <div className="px-2 pb-2">
          <div className="text-xs text-gray-400 px-2 mb-1 font-medium">{t('research.sections')}</div>
          {(dashboard.sections ?? []).map((s, i) => (
            <SectionRow
              key={i}
              section={s}
              canRestart={canRestart}
              onRestart={handleRestartSection}
              onOptimize={handleOptimizeSection}
            />
          ))}
        </div>
      )}

      {/* Coverage Gaps */}
      {(dashboard.coverage_gaps?.length ?? 0) > 0 && (
        <div className="px-4 py-2 border-t">
          <div className="text-xs text-gray-400 mb-1 font-medium">{t('research.coverageGaps')}</div>
          {(dashboard.coverage_gaps ?? []).slice(0, 5).map((g, i) => (
            <div key={i} className="text-xs text-red-500 flex items-start gap-1 py-0.5">
              <span>❗</span>
              <span>{g}</span>
            </div>
          ))}
        </div>
      )}

      {/* Conflicts */}
      {(dashboard.conflict_notes?.length ?? 0) > 0 && (
        <div className="px-4 py-2 border-t">
          <div className="text-xs text-gray-400 mb-1 font-medium">{t('research.conflicts')}</div>
          {(dashboard.conflict_notes ?? []).slice(0, 3).map((c, i) => (
            <div key={i} className="text-xs text-amber-600 flex items-start gap-1 py-0.5">
              <span>⚠️</span>
              <span>{c}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
