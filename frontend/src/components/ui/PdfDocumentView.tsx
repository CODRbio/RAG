import { useState, useCallback, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, Loader2, Maximize2, ZoomIn, ZoomOut } from 'lucide-react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PdfDocumentViewProps {
  pdfUrl: string;
  pageNumber?: number;
  bbox?: number[];
  title?: string;
  showHeader?: boolean;
  className?: string;
}

export function PdfDocumentView({
  pdfUrl,
  pageNumber = 1,
  bbox,
  title,
  showHeader = true,
  className = '',
}: PdfDocumentViewProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(pageNumber);
  const [scale, setScale] = useState<number>(1.1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pageRef = useRef<HTMLDivElement>(null);
  const [pageSize, setPageSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    setCurrentPage(Math.max(1, pageNumber || 1));
  }, [pageNumber]);

  const onDocumentLoadSuccess = useCallback(
    ({ numPages: total }: { numPages: number }) => {
      setNumPages(total);
      setCurrentPage(Math.min(pageNumber, total));
      setLoading(false);
      setError(null);
    },
    [pageNumber],
  );

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
  const resetZoom = () => setScale(1.1);

  const renderHighlights = () => {
    if (!bbox || bbox.length < 4 || !pageSize) return null;
    if (currentPage !== (pageNumber || 1)) return null;
    const renderedWidth = pageSize.width * scale;
    const renderedHeight = pageSize.height * scale;
    const [x0, y0, x1, y1] = bbox;
    const left = (x0 / pageSize.width) * renderedWidth;
    const top = (y0 / pageSize.height) * renderedHeight;
    const width = ((x1 - x0) / pageSize.width) * renderedWidth;
    const height = ((y1 - y0) / pageSize.height) * renderedHeight;

    return (
      <div
        className="absolute rounded-sm border border-yellow-500/60 bg-yellow-400/35 pointer-events-none"
        style={{ left, top, width, height }}
      />
    );
  };

  return (
    <div className={`flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-slate-700/50 bg-slate-950/60 ${className}`}>
      {showHeader && (
        <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-slate-100">{title || 'Paper PDF'}</div>
            {numPages > 0 && (
              <div className="text-xs text-slate-500">
                第 {currentPage} / {numPages} 页
              </div>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={zoomOut}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-sky-400"
              title="缩小"
            >
              <ZoomOut size={16} />
            </button>
            <span className="w-12 text-center text-xs text-slate-500">{Math.round(scale * 100)}%</span>
            <button
              type="button"
              onClick={zoomIn}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-sky-400"
              title="放大"
            >
              <ZoomIn size={16} />
            </button>
            <button
              type="button"
              onClick={resetZoom}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-sky-400"
              title="重置缩放"
            >
              <Maximize2 size={16} />
            </button>
            <div className="mx-1 h-5 w-px bg-slate-700" />
            <button
              type="button"
              onClick={goPrev}
              disabled={currentPage <= 1}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-sky-400 disabled:opacity-30"
              title="上一页"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              type="button"
              onClick={goNext}
              disabled={currentPage >= numPages}
              className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-sky-400 disabled:opacity-30"
              title="下一页"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto bg-slate-950/30 p-4">
        {error ? (
          <div className="mt-12 text-center text-sm text-red-400">{error}</div>
        ) : (
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <div className="mt-12 flex items-center justify-center gap-2 text-slate-400">
                <Loader2 size={18} className="animate-spin" />
                <span className="text-sm">加载 PDF 中...</span>
              </div>
            }
          >
            {!loading && (
              <div ref={pageRef} className="mx-auto inline-block relative">
                <Page
                  pageNumber={currentPage}
                  scale={scale}
                  onRenderSuccess={(page: unknown) => {
                    const next = page as { originalWidth: number; originalHeight: number };
                    onPageRenderSuccess(next);
                  }}
                  loading={
                    <div className="flex items-center justify-center gap-2 text-slate-400">
                      <Loader2 size={16} className="animate-spin" />
                      <span className="text-sm">渲染中...</span>
                    </div>
                  }
                />
                {renderHighlights()}
              </div>
            )}
          </Document>
        )}
      </div>
    </div>
  );
}
