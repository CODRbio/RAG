import { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Database, Globe, X } from 'lucide-react';
import { useChatStore } from '../../stores';

const TIMEOUT_MS = 60_000;

/**
 * 当查询与本地知识库范围不符时，弹出此全屏对话框让用户选择：
 * - 本会话暂不使用本地库（仅网络检索）
 * - 仍使用当前库继续检索
 *
 * 60 秒内未选择时，自动执行「本会话暂不使用本地库」。
 */
export function LocalDbChoiceDialog() {
  const { t } = useTranslation();

  const pendingLocalDbChoice = useChatStore((s) => s.pendingLocalDbChoice);
  const startedAt = useChatStore((s) => s.pendingLocalDbChoiceStartedAt);
  const originalMessage = useChatStore((s) => s.pendingLocalDbChoiceOriginalMessage);
  const handler = useChatStore((s) => s.localDbChoiceHandler);

  const [remaining, setRemaining] = useState(TIMEOUT_MS);
  const autoTriggeredRef = useRef(false);

  // 重置状态
  useEffect(() => {
    if (!pendingLocalDbChoice || !startedAt) return;
    autoTriggeredRef.current = false;
    const tick = () => {
      const elapsed = Date.now() - startedAt;
      const left = Math.max(0, TIMEOUT_MS - elapsed);
      setRemaining(left);
      if (left === 0 && !autoTriggeredRef.current) {
        autoTriggeredRef.current = true;
        handler?.('no_local');
      }
    };
    tick();
    const id = setInterval(tick, 200);
    return () => clearInterval(id);
  }, [pendingLocalDbChoice, startedAt, handler]);

  if (!pendingLocalDbChoice) return null;

  const secs = Math.ceil(remaining / 1000);
  const pct = (remaining / TIMEOUT_MS) * 100;

  const handleChoice = (choice: 'no_local' | 'use') => {
    console.log('[LocalDbChoiceDialog] handleChoice:', choice,
      '| autoTriggered:', autoTriggeredRef.current,
      '| handler available:', !!handler);
    if (!autoTriggeredRef.current) {
      autoTriggeredRef.current = true;
      if (handler) {
        handler(choice);
      } else {
        // handler 尚未注入，直接关闭弹窗（最坏情况兜底）
        console.warn('[LocalDbChoiceDialog] handler is null, clearing pending choice');
        useChatStore.getState().clearPendingLocalDbChoice();
      }
    }
  };

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-md rounded-2xl border border-amber-500/30 bg-slate-900 shadow-[0_0_60px_rgba(245,158,11,0.2)] animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 顶部警告条 */}
        <div className="flex items-center gap-3 px-6 pt-5 pb-4 border-b border-slate-700/60">
          <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
            <AlertTriangle size={18} className="text-amber-400" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-slate-100">
              {t('chat.localDbMismatchTitle', '知识库范围提示')}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {t('chat.localDbMismatchSubtitle', '当前问题与本地知识库主题可能不符')}
            </p>
          </div>
        </div>

        {/* 内容 */}
        <div className="px-6 py-4 space-y-4">
          {/* 问题预览 */}
          {originalMessage && (
            <div className="rounded-lg bg-slate-800/60 border border-slate-700/50 px-3 py-2">
              <p className="text-xs text-slate-400 mb-1">
                {t('chat.localDbMismatchQuery', '你的问题')}
              </p>
              <p className="text-sm text-slate-200 line-clamp-2">{originalMessage}</p>
            </div>
          )}

          <p className="text-sm text-slate-300 leading-relaxed">
            {t(
              'chat.localDbMismatchDesc',
              '此问题可能超出当前知识库的覆盖范围，请选择继续方式：',
            )}
          </p>

          {/* 两个选项 */}
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => handleChoice('no_local')}
              className="group flex flex-col items-center gap-2 p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 hover:border-amber-400/50 transition-all text-center"
            >
              <Globe size={20} className="text-amber-400 group-hover:scale-110 transition-transform" />
              <span className="text-xs font-medium text-amber-200 leading-snug">
                {t('chat.localDbChoiceNoLocal', '本会话暂不使用本地库')}
              </span>
              <span className="text-[10px] text-amber-400/60">
                {t('chat.localDbChoiceNoLocalHint', '仅使用网络检索')}
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleChoice('use')}
              className="group flex flex-col items-center gap-2 p-4 rounded-xl border border-sky-500/30 bg-sky-500/10 hover:bg-sky-500/20 hover:border-sky-400/50 transition-all text-center"
            >
              <Database size={20} className="text-sky-400 group-hover:scale-110 transition-transform" />
              <span className="text-xs font-medium text-sky-200 leading-snug">
                {t('chat.localDbChoiceUse', '仍使用当前库')}
              </span>
              <span className="text-[10px] text-sky-400/60">
                {t('chat.localDbChoiceUseHint', '忽略范围警告继续检索')}
              </span>
            </button>
          </div>

          {/* 倒计时 + 进度条 */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>{t('chat.localDbChoiceAutoLabel', '自动选择「不使用本地库」')}</span>
              <span className={secs <= 10 ? 'text-red-400 font-semibold' : 'text-slate-400'}>
                {secs}s
              </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-200 ${
                  secs <= 10 ? 'bg-red-500' : 'bg-amber-500'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        </div>

        {/* 底部跳过（"跳过"即选 no_local，意思是不再询问直接走默认） */}
        <div className="px-6 pb-5 flex justify-end">
          <button
            type="button"
            onClick={() => handleChoice('no_local')}
            className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            <X size={12} />
            {t('chat.localDbChoiceSkip', '跳过（不使用本地库）')}
          </button>
        </div>
      </div>
    </div>
  );
}
