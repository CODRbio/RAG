import { useState, useEffect, useCallback } from 'react';
import { ChevronRight, Folder, FolderOpen, Home, X, Check, Loader2, AlertCircle, ArrowLeft } from 'lucide-react';
import { listDir } from '../../api/config';
import type { DirListing } from '../../api/config';
import { useTranslation } from 'react-i18next';

interface FolderBrowserModalProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string | null;
}

export function FolderBrowserModal({ open, onClose, onSelect, initialPath }: FolderBrowserModalProps) {
  const { t } = useTranslation();
  const [listing, setListing] = useState<DirListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manualPath, setManualPath] = useState('');

  const navigate = useCallback(async (path?: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const data = await listDir(path);
      setListing(data);
      setManualPath(data.current);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      navigate(initialPath ?? null);
    } else {
      setListing(null);
      setError(null);
    }
  }, [open, initialPath, navigate]);

  if (!open) return null;

  const dirs = listing?.entries.filter((e) => e.is_dir) ?? [];

  const handleManualNavigate = () => {
    const p = manualPath.trim();
    if (p) navigate(p);
  };

  return (
    <div
      className="fixed inset-0 bg-black/70 z-[80] flex items-center justify-center p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="rounded-2xl w-full max-w-lg bg-slate-900 border border-slate-700 shadow-2xl flex flex-col"
        style={{ maxHeight: '80vh' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700 flex-shrink-0">
          <div className="flex items-center gap-2 text-slate-100 font-semibold text-base">
            <FolderOpen size={18} className="text-sky-400" />
            {t('folderBrowser.title')}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          >
            <X size={18} />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-700/60 bg-slate-800/50 flex-shrink-0">
          <button
            onClick={() => listing && navigate(listing.parent)}
            disabled={!listing?.parent || loading}
            className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-30 disabled:cursor-not-allowed flex-shrink-0"
            title={t('folderBrowser.goUp')}
          >
            <ArrowLeft size={16} />
          </button>
          <button
            onClick={() => listing && navigate(listing.home)}
            disabled={loading}
            className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-30 flex-shrink-0"
            title={t('folderBrowser.home')}
          >
            <Home size={16} />
          </button>
          <input
            type="text"
            value={manualPath}
            onChange={(e) => setManualPath(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleManualNavigate()}
            className="flex-1 min-w-0 rounded-lg border border-slate-600 bg-slate-900 px-3 py-1.5 text-slate-200 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-sky-500/50"
            placeholder={t('folderBrowser.pathPlaceholder')}
            spellCheck={false}
          />
          <button
            onClick={handleManualNavigate}
            disabled={loading}
            className="flex-shrink-0 px-3 py-1.5 rounded-lg text-sm bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:opacity-40"
          >
            {t('folderBrowser.go')}
          </button>
        </div>

        {/* Directory listing */}
        <div className="flex-1 overflow-y-auto min-h-0 py-1">
          {loading && (
            <div className="flex items-center justify-center gap-2 py-10 text-slate-400">
              <Loader2 size={18} className="animate-spin" />
              <span className="text-sm">{t('common.loading')}</span>
            </div>
          )}
          {!loading && error && (
            <div className="flex items-start gap-2 m-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              {error}
            </div>
          )}
          {!loading && !error && dirs.length === 0 && (
            <div className="py-10 text-center text-slate-500 text-sm">
              {t('folderBrowser.noSubdirs')}
            </div>
          )}
          {!loading && !error && dirs.map((entry) => (
            <button
              key={entry.path}
              onClick={() => navigate(entry.path)}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-slate-300 hover:bg-slate-800 hover:text-sky-300 transition-colors group"
            >
              <Folder size={16} className="flex-shrink-0 text-sky-500/70 group-hover:text-sky-400" />
              <span className="flex-1 truncate text-sm font-mono">{entry.name}</span>
              <ChevronRight size={14} className="flex-shrink-0 text-slate-600 group-hover:text-slate-400" />
            </button>
          ))}
        </div>

        {/* Footer: current path + confirm */}
        <div className="flex items-center gap-3 px-4 py-3 border-t border-slate-700 bg-slate-800/40 flex-shrink-0">
          <div className="flex-1 min-w-0">
            <div className="text-[10px] text-slate-500 mb-0.5">{t('folderBrowser.selected')}</div>
            <div className="text-xs text-sky-300 font-mono truncate" title={listing?.current ?? ''}>
              {listing?.current ?? '—'}
            </div>
          </div>
          <button
            onClick={() => {
              if (listing?.current) {
                onSelect(listing.current);
                onClose();
              }
            }}
            disabled={!listing?.current}
            className="flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white text-sm font-medium"
          >
            <Check size={15} />
            {t('folderBrowser.select')}
          </button>
        </div>
      </div>
    </div>
  );
}
