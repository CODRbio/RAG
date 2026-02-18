import { useState, useCallback, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Loader2 } from 'lucide-react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// PDF.js worker — 使用 CDN 加载以兼容 Vite
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

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
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(pageNumber);
  const [scale, setScale] = useState<number>(1.2);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pageRef = useRef<HTMLDivElement>(null);

  // PDF 原始页面尺寸（由 onRenderSuccess 获取）
  const [pageSize, setPageSize] = useState<{ width: number; height: number } | null>(null);

  const onDocumentLoadSuccess = useCallback(({ numPages: total }: { numPages: number }) => {
    setNumPages(total);
    setCurrentPage(Math.min(pageNumber, total));
    setLoading(false);
    setError(null);
  }, [pageNumber]);

  const onDocumentLoadError = useCallback(() => {
    setLoading(false);
    setError('PDF 加载失败，请确认文件存在。');
  }, []);

  const onPageRenderSuccess = useCallback((page: { originalWidth: number; originalHeight: number }) => {
    setPageSize({ width: page.originalWidth, height: page.originalHeight });
  }, []);

  const goPrev = () => setCurrentPage((p) => Math.max(1, p - 1));
  const goNext = () => setCurrentPage((p) => Math.min(numPages, p + 1));
  const zoomIn = () => setScale((s) => Math.min(3, s + 0.2));
  const zoomOut = () => setScale((s) => Math.max(0.5, s - 0.2));

  if (!open) return null;

  // 计算高亮覆盖层：将 bbox 物理坐标按比例映射到渲染尺寸
  const renderHighlights = () => {
    if (!bbox || bbox.length < 4 || !pageSize) return null;
    // 仅在目标页显示高亮
    if (currentPage !== (pageNumber || 1)) return null;

    const renderedWidth = pageSize.width * scale;
    const renderedHeight = pageSize.height * scale;

    {
      const [x0, y0, x1, y1] = bbox;

      // bbox 是 PDF 物理坐标（单位：pt），页面原始尺寸也是 pt
      const left = (x0 / pageSize.width) * renderedWidth;
      const top = (y0 / pageSize.height) * renderedHeight;
      const width = ((x1 - x0) / pageSize.width) * renderedWidth;
      const height = ((y1 - y0) / pageSize.height) * renderedHeight;

      return (
        <div
          className="absolute bg-yellow-400/40 border border-yellow-500/60 rounded-sm pointer-events-none"
          style={{ left, top, width, height }}
        />
      );
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 z-[80] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl animate-in zoom-in-95 duration-200 border border-slate-700/50"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700/50 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-lg">📄</span>
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-slate-200 truncate">
                {title || 'PDF 溯源'}
              </h3>
              {numPages > 0 && (
                <p className="text-xs text-slate-500">
                  第 {currentPage} / {numPages} 页
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            {/* Zoom controls */}
            <button
              onClick={zoomOut}
              className="p-1.5 text-slate-400 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors"
              title="缩小"
            >
              <ZoomOut size={16} />
            </button>
            <span className="text-xs text-slate-500 w-12 text-center">{Math.round(scale * 100)}%</span>
            <button
              onClick={zoomIn}
              className="p-1.5 text-slate-400 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors"
              title="放大"
            >
              <ZoomIn size={16} />
            </button>
            {/* Page navigation */}
            <div className="w-px h-5 bg-slate-700 mx-1" />
            <button
              onClick={goPrev}
              disabled={currentPage <= 1}
              className="p-1.5 text-slate-400 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-30"
              title="上一页"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={goNext}
              disabled={currentPage >= numPages}
              className="p-1.5 text-slate-400 hover:text-sky-400 hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-30"
              title="下一页"
            >
              <ChevronRight size={16} />
            </button>
            <div className="w-px h-5 bg-slate-700 mx-1" />
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded-lg transition-colors"
              title="关闭"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* PDF Content */}
        <div className="flex-1 overflow-auto flex items-start justify-center p-4 bg-slate-950/50">
          {error ? (
            <div className="text-red-400 text-sm mt-12">{error}</div>
          ) : (
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={
                <div className="flex items-center gap-2 text-slate-400 mt-12">
                  <Loader2 size={18} className="animate-spin" />
                  <span className="text-sm">加载 PDF 中...</span>
                </div>
              }
            >
              {!loading && (
                <div ref={pageRef} className="relative inline-block">
                  <Page
                    pageNumber={currentPage}
                    scale={scale}
                    onRenderSuccess={(page: unknown) => {
                      const p = page as { originalWidth: number; originalHeight: number };
                      onPageRenderSuccess(p);
                    }}
                    loading={
                      <div className="flex items-center gap-2 text-slate-400">
                        <Loader2 size={16} className="animate-spin" />
                        <span className="text-sm">渲染中...</span>
                      </div>
                    }
                  />
                  {/* Bbox highlight overlays */}
                  {renderHighlights()}
                </div>
              )}
            </Document>
          )}
        </div>
      </div>
    </div>
  );
}
