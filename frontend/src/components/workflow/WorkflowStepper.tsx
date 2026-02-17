import { useTranslation } from 'react-i18next';
import { useChatStore } from '../../stores';
import { WORKFLOW_STAGES } from '../../types';

export function WorkflowStepper() {
  const { t } = useTranslation();
  const { workflowStep } = useChatStore();
  
  // idle 状态不显示
  if (workflowStep === 'idle') {
    return null;
  }

  const currentIndex = WORKFLOW_STAGES.findIndex(s => s.id === workflowStep);

  return (
    <div className="flex items-center justify-center gap-2 py-3 px-4 bg-slate-900/60 backdrop-blur-sm border-b border-slate-700/50">
      {WORKFLOW_STAGES.map((stage, index) => {
        const isActive = stage.id === workflowStep;
        const isCompleted = index < currentIndex;

        return (
          <div key={stage.id} className="flex items-center">
            {/* 连接线 */}
            {index > 0 && (
              <div
                className={`w-8 h-0.5 mx-1 transition-colors ${
                  isCompleted ? 'bg-emerald-500' : 'bg-slate-700'
                }`}
              />
            )}
            
            {/* 阶段节点 */}
            <div
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all
                ${isActive 
                  ? 'bg-sky-900/40 text-sky-400 ring-2 ring-sky-500/30 ring-offset-1 ring-offset-slate-900' 
                  : isCompleted 
                    ? 'bg-emerald-900/20 text-emerald-400' 
                    : 'bg-slate-800/50 text-slate-500'
                }
              `}
              title={t(stage.description)}
            >
              <span className="text-base">{stage.icon}</span>
              <span className="hidden sm:inline">{t(stage.label)}</span>
              {isActive && (
                <span className="w-1.5 h-1.5 bg-sky-400 rounded-full animate-pulse shadow-[0_0_6px_rgba(56,189,248,0.8)]" />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
