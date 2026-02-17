import { useToastStore } from '../../stores';

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`px-4 py-3 rounded-lg shadow-lg text-sm font-medium animate-in slide-in-from-right fade-in duration-300 pointer-events-auto ${
            t.type === 'success'
              ? 'bg-green-600 text-white'
              : t.type === 'error'
              ? 'bg-red-600 text-white'
              : t.type === 'warning'
              ? 'bg-amber-600 text-white'
              : 'bg-gray-800 text-white'
          }`}
        >
          {t.msg}
        </div>
      ))}
    </div>
  );
}
