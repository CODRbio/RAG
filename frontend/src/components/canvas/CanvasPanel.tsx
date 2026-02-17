import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FileEdit,
  FileDown,
  FileType,
  X,
  Loader2,
  Plus,
  Telescope,
} from 'lucide-react';
import { useCanvasStore, useChatStore, useUIStore, useToastStore } from '../../stores';
import { createCanvas } from '../../api/canvas';
import { StageStepper } from './StageStepper';
import { ExploreStage } from './ExploreStage';
import { OutlineStage } from './OutlineStage';
import { DraftingStage } from './DraftingStage';
import { RefineStage } from './RefineStage';
import type { CanvasStage } from '../../types';

interface CanvasPanelProps {
  onStartResize: () => void;
}

export function CanvasPanel({ onStartResize }: CanvasPanelProps) {
  const { t } = useTranslation();
  const {
    canvas,
    canvasContent,
    isLoading,
    isAIEditing,
    activeStage,
    setCanvas,
    setActiveStage,
  } = useCanvasStore();
  const { workflowStep, sessionId, setCanvasId, setShowDeepResearchDialog } = useChatStore();
  const { canvasWidth, setCanvasOpen } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);
  const [isCreating, setIsCreating] = useState(false);

  const handleExport = (format: 'md' | 'pdf') => {
    if (!canvasContent) {
      addToast(t('canvas.emptyCannotExport'), 'error');
      return;
    }
    if (format === 'md') {
      const blob = new Blob([canvasContent], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'research_draft.md';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      addToast(t('canvas.exportedMd'), 'success');
    } else {
      addToast(t('canvas.pdfInDev'), 'info');
    }
  };

  // 创建空 Canvas（无需 Deep Research）
  const handleCreateEmptyCanvas = async () => {
    setIsCreating(true);
    try {
      const newCanvas = await createCanvas({
        session_id: sessionId || undefined,
        topic: '',
      });
      if (newCanvas) {
        setCanvas(newCanvas);
        setCanvasId(newCanvas.id);
        addToast(t('canvas.canvasCreated'), 'success');
      }
    } catch (err) {
      console.error('[CanvasPanel] Create canvas failed:', err);
      addToast(t('canvas.createFailed'), 'error');
    } finally {
      setIsCreating(false);
    }
  };

  // 当前 canvas 的真实阶段（用于 stepper 的"已完成"判断）
  const currentStage: CanvasStage = (canvas?.stage as CanvasStage) || 'explore';

  // 渲染对应阶段的内容
  const renderStageContent = () => {
    // 如果没有 canvas 数据，显示可操作的空状态
    if (!canvas) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-slate-500 px-6">
          <div className="p-4 bg-slate-800/50 rounded-full mb-4 shadow-[0_0_20px_rgba(56,189,248,0.1)] border border-slate-700/50 animate-pulse-glow">
            <FileEdit size={36} className="text-sky-500/50" />
          </div>
          <p className="text-sm font-medium mb-1 text-slate-300">{t('canvas.emptyTitle')}</p>
          <p className="text-xs text-center mb-6 leading-relaxed text-slate-500 max-w-[240px]">
            {t('canvas.emptyDesc')}
          </p>
          <div className="flex flex-col gap-3 w-full max-w-48">
            <button
              onClick={() => setShowDeepResearchDialog(true)}
              className="flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600/90 text-white text-xs font-medium rounded-lg hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-500/20 border border-indigo-400/30"
            >
              <Telescope size={14} />
              {t('canvas.startDeepResearch')}
            </button>
            <button
              onClick={handleCreateEmptyCanvas}
              disabled={isCreating}
              className="flex items-center justify-center gap-2 px-4 py-2.5 border border-slate-700 bg-slate-800/40 text-slate-400 text-xs font-medium rounded-lg hover:bg-slate-700 hover:text-slate-200 transition-colors disabled:opacity-50"
            >
              {isCreating ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
              {t('canvas.createBlankCanvas')}
            </button>
          </div>
        </div>
      );
    }

    switch (activeStage) {
      case 'explore':
        return <ExploreStage canvas={canvas} />;
      case 'outline':
        return <OutlineStage canvas={canvas} />;
      case 'drafting':
        return <DraftingStage canvas={canvas} />;
      case 'refine':
        return <RefineStage canvas={canvas} />;
      default:
        return <ExploreStage canvas={canvas} />;
    }
  };

  return (
    <div
      className="bg-slate-900/95 backdrop-blur-md border-l border-slate-700/50 flex flex-col relative flex-shrink-0 z-40 shadow-[-5px_0_30px_rgba(0,0,0,0.3)] transition-[width] duration-100 ease-linear"
      style={{ width: canvasWidth }}
    >
      {/* Header */}
      <div className="h-12 border-b border-slate-700/50 flex items-center justify-between px-4 bg-slate-900/80 shadow-sm">
        <div className="flex items-center gap-2 font-bold text-sm text-slate-200 min-w-0">
          <FileEdit size={16} className="text-sky-500" />
          <span className="truncate">{t('canvas.researchCanvas')}</span>
          {(workflowStep === 'drafting' || isLoading || isAIEditing) && (
            <Loader2 size={12} className="animate-spin text-sky-400" />
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={() => handleExport('md')}
            className="p-1.5 hover:bg-slate-800 rounded text-slate-400 hover:text-sky-400 transition-colors"
            title="Export Markdown"
          >
            <FileDown size={14} />
          </button>
          <button
            onClick={() => handleExport('pdf')}
            className="p-1.5 hover:bg-slate-800 rounded text-slate-400 hover:text-sky-400 transition-colors"
            title="Export PDF"
          >
            <FileType size={14} />
          </button>
          <button
            onClick={() => setCanvasOpen(false)}
            className="p-1.5 hover:bg-slate-800 rounded text-slate-400 hover:text-red-400 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Stage Stepper — 只在有 canvas 时显示 */}
      {canvas && (
        <StageStepper
          currentStage={currentStage}
          activeStage={activeStage}
          onStageClick={(stage) => setActiveStage(stage)}
        />
      )}

      {/* Stage Content */}
      <div className="flex-1 overflow-hidden bg-slate-900/50">
        {renderStageContent()}
      </div>

      {/* Drag Handle */}
      <div
        onMouseDown={(e) => {
          e.preventDefault();
          onStartResize();
        }}
        className="absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-sky-500/50 z-50 group transition-colors"
      >
        <div className="absolute top-1/2 -left-3 w-6 h-8 bg-slate-800 border border-slate-600 rounded shadow-sm flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <div className="text-slate-500 text-[10px]">&#x22ee;</div>
        </div>
      </div>
    </div>
  );
}
