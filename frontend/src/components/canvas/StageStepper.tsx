import { Search, ListTree, PenTool, Sparkles, Check } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { CanvasStage } from '../../types';

interface StageStepperProps {
  currentStage: CanvasStage;
  activeStage: CanvasStage;
  onStageClick: (stage: CanvasStage) => void;
}

const STAGES: {
  id: CanvasStage;
  labelKey: string;
  icon: React.ReactNode;
}[] = [
  { id: 'explore', labelKey: 'workflow.explore', icon: <Search size={14} /> },
  { id: 'outline', labelKey: 'workflow.outline', icon: <ListTree size={14} /> },
  { id: 'drafting', labelKey: 'workflow.drafting', icon: <PenTool size={14} /> },
  { id: 'refine', labelKey: 'workflow.refine', icon: <Sparkles size={14} /> },
];

const STAGE_ORDER: CanvasStage[] = ['explore', 'outline', 'drafting', 'refine'];

function getStageIndex(stage: CanvasStage): number {
  return STAGE_ORDER.indexOf(stage);
}

export function StageStepper({ currentStage, activeStage, onStageClick }: StageStepperProps) {
  const { t } = useTranslation();
  const currentIdx = getStageIndex(currentStage);
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [compact, setCompact] = useState(false);

  useEffect(() => {
    const measure = () => {
      const container = containerRef.current;
      const content = contentRef.current;
      if (!container || !content) return;

      const shouldCompact = content.scrollWidth > container.clientWidth + 4;
      setCompact((prev) => (prev === shouldCompact ? prev : shouldCompact));
    };

    measure();
    const ro = new ResizeObserver(measure);
    if (containerRef.current) ro.observe(containerRef.current);
    window.addEventListener('resize', measure);

    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  return (
    <div ref={containerRef} className="px-3 py-2 bg-slate-800/50 border-b border-slate-700/50">
      <div className="overflow-x-auto scrollbar-hide">
        <div ref={contentRef} className="flex items-center gap-1 min-w-max">
          {STAGES.map((stage, idx) => {
            const stageIdx = getStageIndex(stage.id);
            const isCompleted = stageIdx < currentIdx;
            const isCurrent = stage.id === currentStage;
            const isActive = stage.id === activeStage;
            const isClickable = stageIdx <= currentIdx;

            return (
              <div key={stage.id} className="flex items-center">
                {idx > 0 && (
                  <div
                    className={`${compact ? 'w-3' : 'w-6'} h-px mx-0.5 ${
                      stageIdx <= currentIdx ? 'bg-sky-500' : 'bg-slate-700'
                    }`}
                  />
                )}
                <button
                  onClick={() => isClickable && onStageClick(stage.id)}
                  disabled={!isClickable}
                  className={`
                    flex items-center ${compact ? 'justify-center gap-0 px-2 py-1.5' : 'gap-1.5 px-2.5 py-1.5'} rounded-md text-xs font-medium
                    transition-all duration-150 cursor-pointer
                    ${isActive
                      ? 'bg-sky-600 text-white shadow-sm shadow-sky-500/30'
                      : isCompleted
                        ? 'bg-emerald-900/20 text-emerald-400 hover:bg-emerald-900/30'
                        : isCurrent
                          ? 'bg-sky-900/30 text-sky-400 border border-sky-500/40'
                          : 'bg-slate-800/50 text-slate-500'
                    }
                    ${!isClickable ? 'opacity-50 cursor-not-allowed' : ''}
                  `}
                  title={t(stage.labelKey)}
                >
                  {isCompleted && !isActive ? (
                    <Check size={12} className="text-emerald-400" />
                  ) : (
                    stage.icon
                  )}
                  {!compact && <span>{t(stage.labelKey)}</span>}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
