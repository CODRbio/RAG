import { RefreshCw, Maximize2, Minimize2, BookOpen, Loader2 } from 'lucide-react';

export type AIAction = 'rewrite' | 'expand' | 'condense' | 'add_citations' | 'targeted_refine';

interface Props {
  position: { top: number; left: number };
  isLoading: boolean;
  onAction: (action: AIAction) => void;
}

const ACTIONS: { action: AIAction; label: string; icon: React.ReactNode }[] = [
  { action: 'rewrite', label: '重写', icon: <RefreshCw size={12} /> },
  { action: 'expand', label: '扩展', icon: <Maximize2 size={12} /> },
  { action: 'condense', label: '精简', icon: <Minimize2 size={12} /> },
  { action: 'add_citations', label: '添加引用', icon: <BookOpen size={12} /> },
  { action: 'targeted_refine', label: '定向精炼', icon: <RefreshCw size={12} /> },
];

export function FloatingToolbar({ position, isLoading, onAction }: Props) {
  return (
    <div
      className="fixed z-[100] flex items-center gap-1 bg-slate-900 border border-slate-700 rounded-lg shadow-lg px-1.5 py-1 animate-in fade-in slide-in-from-bottom-1 duration-150"
      style={{ top: position.top, left: position.left }}
    >
      {isLoading ? (
        <div className="flex items-center gap-1.5 px-2 py-1 text-xs text-slate-300">
          <Loader2 size={12} className="animate-spin" />
          AI 编辑中...
        </div>
      ) : (
        ACTIONS.map(({ action, label, icon }) => (
          <button
            key={action}
            onClick={() => onAction(action)}
            className="flex items-center gap-1 px-2 py-1 text-xs text-slate-300 hover:text-sky-200 hover:bg-sky-900/40 rounded transition-colors cursor-pointer whitespace-nowrap"
            title={label}
          >
            {icon}
            {label}
          </button>
        ))
      )}
    </div>
  );
}
