import { Network } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { GraphExplorer } from '../components/graph/GraphExplorer';

export function GraphWorkspacePage() {
  const { t } = useTranslation();

  return (
    <div className="flex h-full min-h-0 flex-col bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.12),transparent_34%),linear-gradient(180deg,rgba(15,23,42,0.94),rgba(2,6,23,0.99))]">
      <div className="border-b border-slate-800/80 px-5 py-4">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-emerald-300/80">
          <Network size={14} />
          Graph Workspace
        </div>
        <h1 className="mt-2 text-xl font-semibold text-slate-100">
          {t('graphWorkspace.title', 'Scope-first Graph Exploration')}
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          {t('graphWorkspace.subtitle', 'Choose graph type, scope, and seeds first. The graph canvas and detail panel update from that explicit object context.')}
        </p>
      </div>
      <div className="min-h-0 flex-1 p-4">
        <div className="h-full overflow-hidden rounded-3xl border border-slate-800 bg-white/95 shadow-[0_24px_80px_rgba(15,23,42,0.35)]">
          <GraphExplorer />
        </div>
      </div>
    </div>
  );
}
