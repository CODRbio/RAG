import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Target, FlaskConical, HelpCircle, Ban, BookOpen, Rocket, Clock, Edit3, RotateCcw } from 'lucide-react';
import type { Canvas } from '../../types';
import { restartDeepResearchPhase } from '../../api/chat';
import { useChatStore, useToastStore, useUIStore } from '../../stores';
import {
  DEEP_RESEARCH_JOB_KEY,
  DEEP_RESEARCH_ARCHIVED_JOBS_KEY,
} from '../workflow/deep-research/types';

interface ExploreStageProps {
  canvas: Canvas;
}

/**
 * Explore 阶段：Survey Canvas 结构化卡片（类似 Business Model Canvas）
 * 展示研究简报的各个板块：Goal / Hypothesis / Questions / Exclusions / Sources / Action Plan
 */
export function ExploreStage({ canvas }: ExploreStageProps) {
  const { t } = useTranslation();
  const addToast = useToastStore((s) => s.addToast);
  const { setDeepResearchTopic, setShowDeepResearchDialog, setDeepResearchActive, setSessionId, setCanvasId } = useChatStore();
  const { requestSessionListRefresh } = useUIStore();
  const [restarting, setRestarting] = useState(false);
  const brief = canvas.research_brief;

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

  const handleRestart = async (
    phase: 'plan' | 'research' | 'generate_claims' | 'write' | 'verify' | 'review_gate' | 'synthesize',
  ) => {
    const sourceJobId = resolveSourceJobId();
    if (!sourceJobId) {
      addToast('未找到可重启任务，请先完成一次 Deep Research', 'warning');
      return;
    }
    setRestarting(true);
    try {
      const resp = await restartDeepResearchPhase(sourceJobId, { phase });
      if (resp.session_id) setSessionId(resp.session_id);
      if (resp.canvas_id) setCanvasId(resp.canvas_id);
      localStorage.setItem(DEEP_RESEARCH_JOB_KEY, resp.job_id);
      setDeepResearchTopic(canvas.topic || canvas.working_title || '');
      setDeepResearchActive(true);
      setShowDeepResearchDialog(true);
      requestSessionListRefresh();
      addToast(`已提交阶段重启：${phase}`, 'success');
    } catch (err) {
      console.error('[ExploreStage] restart phase failed:', err);
      addToast('阶段重启失败，请重试', 'error');
    } finally {
      setRestarting(false);
    }
  };

  if (!brief) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-[var(--text-tertiary)] text-sm px-6">
        <Target size={36} className="mb-3 opacity-30" />
        <p className="font-medium mb-1">研究简报尚未生成</p>
        <p className="text-xs text-center">
          通过 Deep Research 的澄清阶段生成研究简报后，<br />
          结构化的研究规划将显示在此处。
        </p>
      </div>
    );
  }

  const cards = [
    {
      icon: <Target size={16} />,
      title: '研究目标 (Goal)',
      color: 'blue',
      content: (
        <div className="space-y-1.5">
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
            {brief.scope || canvas.topic || '—'}
          </p>
          {canvas.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {canvas.keywords.map((kw, i) => (
                <span key={i} className="px-2 py-0.5 text-[10px] bg-blue-50 text-blue-700 rounded-full">
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      icon: <FlaskConical size={16} />,
      title: '假设与标准 (Hypothesis)',
      color: 'purple',
      content: (
        <ul className="space-y-1">
          {brief.success_criteria.length > 0 ? (
            brief.success_criteria.map((c, i) => (
              <li key={i} className="text-sm text-[var(--text-secondary)] flex items-start gap-1.5">
                <span className="text-purple-500 mt-0.5 shrink-0">•</span>
                <span>{c}</span>
              </li>
            ))
          ) : (
            <li className="text-xs text-[var(--text-tertiary)]">尚未定义</li>
          )}
        </ul>
      ),
    },
    {
      icon: <HelpCircle size={16} />,
      title: '核心问题 (Key Questions)',
      color: 'amber',
      content: (
        <ul className="space-y-1">
          {brief.key_questions.length > 0 ? (
            brief.key_questions.map((q, i) => (
              <li key={i} className="text-sm text-[var(--text-secondary)] flex items-start gap-1.5">
                <span className="text-amber-500 mt-0.5 shrink-0 font-mono text-xs">Q{i + 1}</span>
                <span>{q}</span>
              </li>
            ))
          ) : (
            <li className="text-xs text-[var(--text-tertiary)]">尚未定义</li>
          )}
        </ul>
      ),
    },
    {
      icon: <Ban size={16} />,
      title: '排除范围 (Exclusions)',
      color: 'red',
      content: (
        <ul className="space-y-1">
          {brief.exclusions.length > 0 ? (
            brief.exclusions.map((e, i) => (
              <li key={i} className="text-sm text-[var(--text-secondary)] flex items-start gap-1.5">
                <span className="text-red-400 mt-0.5 shrink-0">✗</span>
                <span>{e}</span>
              </li>
            ))
          ) : (
            <li className="text-xs text-[var(--text-tertiary)]">无排除项</li>
          )}
        </ul>
      ),
    },
    {
      icon: <BookOpen size={16} />,
      title: '文献来源 (Source Plan)',
      color: 'teal',
      content: (
        <div className="space-y-1.5">
          {brief.time_range && (
            <div className="flex items-center gap-1.5 text-sm text-[var(--text-secondary)]">
              <Clock size={12} className="text-teal-500" />
              <span>时间范围: {brief.time_range}</span>
            </div>
          )}
          {brief.source_priority.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {brief.source_priority.map((s, i) => (
                <span key={i} className="px-2 py-0.5 text-[10px] bg-teal-50 text-teal-700 rounded-full">
                  {s}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[var(--text-tertiary)]">使用默认检索策略</p>
          )}
        </div>
      ),
    },
    {
      icon: <Rocket size={16} />,
      title: '行动计划 (Action Plan)',
      color: 'indigo',
      content: (
        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
          {brief.action_plan || '根据研究结果生成综述文档，覆盖所有大纲章节。'}
        </p>
      ),
    },
  ];

  const colorMap: Record<string, string> = {
    blue: 'border-l-blue-400',
    purple: 'border-l-purple-400',
    amber: 'border-l-amber-400',
    red: 'border-l-red-400',
    teal: 'border-l-teal-400',
    indigo: 'border-l-indigo-400',
  };

  const iconColorMap: Record<string, string> = {
    blue: 'text-blue-500',
    purple: 'text-purple-500',
    amber: 'text-amber-500',
    red: 'text-red-500',
    teal: 'text-teal-500',
    indigo: 'text-indigo-500',
  };

  return (
    <div className="p-4 space-y-3 overflow-y-auto h-full">
      <div className="bg-[var(--bg-surface)] rounded-lg p-3 border border-[var(--border-subtle)]">
        <div className="flex items-center gap-2 mb-2">
          <RotateCcw size={13} className="text-indigo-500" />
          <span className="text-xs text-[var(--text-tertiary)] font-medium">{t('research.restartStageTitle', '重启执行')}</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button
            disabled={restarting}
            onClick={() => void handleRestart('plan')}
            className="px-2 py-1 text-[10px] border rounded text-indigo-700 hover:bg-indigo-50 disabled:opacity-50"
          >
            {t('research.restartExploreStage', '重新探索（重建简报与大纲）')}
          </button>
        </div>
      </div>

      {/* Topic Header */}
      <div className="bg-[var(--bg-surface)] rounded-lg p-4 border border-[var(--border-subtle)]">
        <div className="flex items-center gap-2 mb-2">
          <Edit3 size={14} className="text-[var(--primary)]" />
          <span className="text-xs font-medium text-[var(--text-tertiary)] uppercase tracking-wider">Research Topic</span>
        </div>
        <h3 className="text-base font-semibold text-[var(--text-primary)] leading-snug">
          {canvas.working_title || canvas.topic || '未命名研究'}
        </h3>
      </div>

      {/* 6-Card Grid */}
      <div className="grid grid-cols-1 gap-3">
        {cards.map((card) => (
          <div
            key={card.title}
            className={`bg-[var(--bg-panel)] rounded-lg border border-[var(--border-subtle)] border-l-4 ${colorMap[card.color]} p-3.5 shadow-sm hover:shadow transition-shadow`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className={iconColorMap[card.color]}>{card.icon}</span>
              <h4 className="text-xs font-semibold text-[var(--text-primary)] uppercase tracking-wide">
                {card.title}
              </h4>
            </div>
            <div>{card.content}</div>
          </div>
        ))}
      </div>

      {/* Identified Gaps */}
      {canvas.identified_gaps.length > 0 && (
        <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-3.5">
          <h4 className="text-xs font-semibold text-amber-200 mb-2 uppercase tracking-wide">
            Information Gaps
          </h4>
          <ul className="space-y-1">
            {canvas.identified_gaps.map((gap, i) => (
              <li key={i} className="text-sm text-amber-300 flex items-start gap-1.5">
                <span className="shrink-0">⚠️</span>
                <span>{gap}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
