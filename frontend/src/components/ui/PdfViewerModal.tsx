import { X } from 'lucide-react';
import { PdfDocumentView } from './PdfDocumentView';

interface PdfViewerModalProps {
  open: boolean;
  onClose: () => void;
  /** PDF 文件 URL（通常为 /api/graph/pdf/{paper_id}） */
  pdfUrl: string;
  /** 高亮所在页码（1-based） */
  pageNumber?: number;
  /** Docling bbox 坐标 [x0, y0, x1, y1]（PDF 物理坐标） */
  bbox?: number[];
  /** 标题（用于显示） */
  title?: string;
}

export function PdfViewerModal({
  open,
  onClose,
  pdfUrl,
  pageNumber = 1,
  bbox,
  title,
}: PdfViewerModalProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/60 z-[80] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl animate-in zoom-in-95 duration-200 border border-slate-700/50"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700/50 shrink-0">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-slate-200 truncate">
              {title || 'PDF 溯源'}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded-lg transition-colors"
            title="关闭"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex-1 min-h-0 p-4">
          <PdfDocumentView
            pdfUrl={pdfUrl}
            pageNumber={pageNumber}
            bbox={bbox}
            title={title}
            className="h-full border-0 bg-transparent"
          />
        </div>
      </div>
    </div>
  );
}
