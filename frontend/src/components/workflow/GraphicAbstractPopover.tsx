import { useRef, useEffect } from 'react';
import { Image as ImageIcon, HelpCircle } from 'lucide-react';
import { useConfigStore } from '../../stores';

/** Hover tooltip with delay */
function Tip({ content, children }: { content: string; children: React.ReactNode }) {
  return (
    <span className="relative inline-flex items-center text-gray-400 cursor-help shrink-0 group">
      {children}
      <span className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 px-2.5 py-2 text-[10px] leading-relaxed text-gray-100 bg-gray-800/95 rounded-lg shadow-lg max-w-[240px] whitespace-normal z-[100] border border-gray-600/80 pointer-events-none hidden group-hover:block" role="tooltip">
        {content}
      </span>
    </span>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export function GraphicAbstractPopover({ open, onClose }: Props) {
  const { ragConfig, updateRagConfig } = useConfigStore();
  const popoverRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const timer = setTimeout(() => {
      document.addEventListener('mousedown', handler);
    }, 0);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('mousedown', handler);
    };
  }, [open, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={popoverRef}
      className="absolute bottom-full mb-2 left-0 w-64 bg-white rounded-xl shadow-2xl border border-gray-200 z-50 overflow-hidden"
    >
      <div className="flex items-center justify-between px-3 py-2 bg-pink-50/50 border-b border-pink-100">
        <div className="flex items-center gap-1.5">
          <ImageIcon size={14} className="text-pink-500" />
          <span className="text-xs font-semibold text-gray-800">Graphic Abstract</span>
        </div>
        <label className="flex items-center cursor-pointer">
          <div className={`relative w-7 h-4 rounded-full transition-colors ${ragConfig.enableGraphicAbstract ? 'bg-pink-500' : 'bg-gray-300'}`}>
            <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform ${ragConfig.enableGraphicAbstract ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
          </div>
          <input
            type="checkbox"
            className="hidden"
            checked={ragConfig.enableGraphicAbstract ?? false}
            onChange={(e) => updateRagConfig({ enableGraphicAbstract: e.target.checked })}
          />
        </label>
      </div>

      <div className="p-3">
        <div className="flex items-center justify-between gap-2 mb-2">
          <label className="text-[11px] font-medium text-gray-600 flex items-center gap-1">
            画图模型
            <Tip content="自动抽取回复或研究报告要点，并生成最终 Graphic Abstract 图像">
              <HelpCircle size={10} />
            </Tip>
          </label>
        </div>
        <select
          disabled={!ragConfig.enableGraphicAbstract}
          value={ragConfig.graphicAbstractModel ?? 'nanobanana 2'}
          onChange={(e) => updateRagConfig({ graphicAbstractModel: e.target.value })}
          className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-[11px] bg-white text-gray-900 focus:ring-1 focus:ring-pink-400 outline-none disabled:bg-gray-50 disabled:text-gray-400"
        >
          <option value="nanobanana 2">nanobanana 2 (Gemini 2.5 Flash Image)</option>
          <option value="nanobanana pro">nanobanana pro (Gemini Pro)</option>
          <option value="gpt-image-1.5">gpt-image-1.5 (GPT)</option>
          <option value="qwen-image-2.0">qwen-image-2.0 (Qwen)</option>
        </select>
      </div>
    </div>
  );
}
