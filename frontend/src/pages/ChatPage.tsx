import { useTranslation } from 'react-i18next';
import { Database, Server } from 'lucide-react';
import { useConfigStore, useToastStore } from '../stores';
import { ChatWindow } from '../components/chat/ChatWindow';
import { ChatInput } from '../components/chat/ChatInput';
import { WorkflowStepper } from '../components/workflow';
import { checkHealth } from '../api/health';

export function ChatPage() {
  const { t } = useTranslation();
  const { dbStatus, setDbStatus, dbAddress } = useConfigStore();
  const addToast = useToastStore((s) => s.addToast);

  const handleConnect = async () => {
    setDbStatus('connecting');
    addToast(t('chatPage.connectingTo', { address: dbAddress }), 'info');
    try {
      await checkHealth();
      setDbStatus('connected');
      addToast(t('chatPage.connected'), 'success');
    } catch {
      setDbStatus('disconnected');
      addToast(t('chatPage.connectFailed'), 'error');
    }
  };

  // 未连接状态
  if (dbStatus === 'disconnected') {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center p-8 animate-in fade-in zoom-in duration-300 z-10 bg-slate-950/90 backdrop-blur-sm">
        <div className="bg-slate-900/80 backdrop-blur-md p-12 rounded-3xl shadow-2xl max-w-2xl w-full text-center border border-slate-700/50">
          <div className="w-20 h-20 bg-sky-900/30 text-sky-400 rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
            <Database size={40} />
          </div>
          <h2 className="text-3xl font-bold text-slate-100 mb-3">{t('chatPage.connectTitle')}</h2>
          <p className="text-slate-400 mb-10 max-w-md mx-auto">
            {t('chatPage.connectDesc')}
          </p>
          <div className="max-w-xs mx-auto">
            <button
              onClick={handleConnect}
              className="w-full group p-4 rounded-xl border-2 border-sky-500 bg-sky-600 text-white hover:bg-sky-500 transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-3 shadow-lg shadow-sky-500/20"
            >
              <Server size={20} />
              <span className="font-bold">{t('chatPage.connectBtn')}</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* 工作流阶段指示器 */}
      <WorkflowStepper />
      
      {/* 聊天窗口 */}
      <main className="flex-1 overflow-y-auto bg-transparent p-8 scrollbar-thin scrollbar-thumb-gray-200">
        <ChatWindow />
      </main>
      
      {/* 输入框 */}
      <ChatInput />
    </div>
  );
}
