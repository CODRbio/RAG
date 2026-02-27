import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Clock, X, ListOrdered } from 'lucide-react';
import { useChatStore } from '../../stores';
import { getTaskQueue, cancelTask } from '../../api/chat';
import type { TaskQueueResponse } from '../../types';

interface TaskCenterProps {
  open: boolean;
  onClose: () => void;
  anchor?: React.ReactNode;
}

export function TaskCenter({ open, onClose, anchor }: TaskCenterProps) {
  const { t } = useTranslation();
  const [snap, setSnap] = useState<TaskQueueResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [cancellingId, setCancellingId] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getTaskQueue();
      setSnap(data);
    } catch (e) {
      console.error('[TaskCenter] getTaskQueue failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) refresh();
  }, [open, refresh]);

  useEffect(() => {
    if (!open || !snap || (snap.active_count === 0 && snap.queued.length === 0)) return;
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [open, snap?.active_count, snap?.queued?.length, refresh]);

  const handleCancel = async (taskId: string) => {
    setCancellingId(taskId);
    try {
      await cancelTask(taskId);
      await refresh();
      useChatStore.getState().clearStreamingTask(taskId);
    } catch (e) {
      console.error('[TaskCenter] cancel failed:', e);
    } finally {
      setCancellingId(null);
    }
  };

  if (!open) return anchor ? <>{anchor}</> : null;

  return (
    <div className="absolute right-0 top-full z-50 mt-1 w-80 rounded-xl border border-slate-700 bg-slate-900 shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-700 px-3 py-2">
        <span className="flex items-center gap-2 text-sm font-medium text-slate-200">
          <ListOrdered size={16} />
          {t('taskCenter.title', '任务队列')}
        </span>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
          aria-label="Close"
        >
          <X size={16} />
        </button>
      </div>
      <div className="max-h-96 overflow-y-auto p-2">
        {loading && !snap ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 size={20} className="animate-spin text-slate-400" />
          </div>
        ) : snap ? (
          <>
            <div className="mb-2 text-xs text-slate-500">
              {snap.active_count}/{snap.max_slots} {t('taskCenter.slots', '槽位使用中')}
            </div>
            {snap.active.length > 0 && (
              <div className="mb-3">
                <div className="mb-1 text-xs font-medium uppercase text-slate-400">
                  {t('taskCenter.active', '执行中')}
                </div>
                <ul className="space-y-1">
                  {snap.active.map((a) => (
                    <li
                      key={a.task_id}
                      className="flex items-center gap-2 rounded-lg bg-slate-800/80 px-2 py-1.5 text-sm"
                    >
                      <Loader2 size={14} className="shrink-0 animate-spin text-sky-400" />
                      <span className="truncate text-slate-200">
                        {a.kind === 'dr' ? t('taskCenter.dr', 'Deep Research') : t('taskCenter.chat', 'Chat')}
                        {a.session_id ? ` · ${a.session_id.slice(0, 8)}` : ''}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {snap.queued.length > 0 && (
              <div>
                <div className="mb-1 text-xs font-medium uppercase text-slate-400">
                  {t('taskCenter.queued', '排队中')}
                </div>
                <ul className="space-y-1">
                  {snap.queued.map((q) => (
                    <li
                      key={q.task_id}
                      className="flex items-center justify-between gap-2 rounded-lg bg-slate-800/50 px-2 py-1.5 text-sm"
                    >
                      <span className="flex items-center gap-2 truncate text-slate-300">
                        <Clock size={14} className="shrink-0 text-amber-500" />
                        #{q.queue_position} · {q.kind === 'dr' ? t('taskCenter.dr', 'DR') : t('taskCenter.chat', 'Chat')}
                        {q.session_id ? ` · ${q.session_id.slice(0, 8)}` : ''}
                      </span>
                      <button
                        type="button"
                        disabled={cancellingId === q.task_id}
                        onClick={() => handleCancel(q.task_id)}
                        className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-600 hover:text-red-400 disabled:opacity-50"
                        title={t('taskCenter.cancel', '取消')}
                      >
                        {cancellingId === q.task_id ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <X size={14} />
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {snap.active_count === 0 && snap.queued.length === 0 && (
              <div className="py-4 text-center text-sm text-slate-500">
                {t('taskCenter.empty', '暂无任务')}
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
