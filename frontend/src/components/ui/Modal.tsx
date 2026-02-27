import { X } from 'lucide-react';
import type { ReactNode } from 'react';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  icon?: ReactNode;
  children: ReactNode;
  maxWidth?: string;
  variant?: 'light' | 'dark';
}

export function Modal({
  open,
  onClose,
  title,
  icon,
  children,
  maxWidth = 'max-w-sm',
  variant = 'light',
}: ModalProps) {
  if (!open) return null;

  const isDark = variant === 'dark';

  return (
    <div
      className="fixed inset-0 bg-black/60 z-[70] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className={`rounded-2xl w-full ${maxWidth} p-6 shadow-2xl animate-in zoom-in-95 duration-200 ${
          isDark
            ? 'bg-slate-900 border border-slate-700'
            : 'bg-white border border-gray-100'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div className={`flex justify-between items-center mb-6 pb-4 border-b ${isDark ? 'border-slate-700' : 'border-gray-200'}`}>
            <h3 className={`text-lg font-bold flex items-center gap-2 ${isDark ? 'text-slate-100' : 'text-gray-900'}`}>
              {icon}
              {title}
            </h3>
            <button
              onClick={onClose}
              className={`p-2 rounded-full ${isDark ? 'text-slate-400 hover:bg-slate-800 hover:text-slate-200' : 'hover:bg-gray-100'}`}
            >
              <X size={20} />
            </button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
