import { useState, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { Layers, Users, Lock, AlertCircle, Loader2 } from 'lucide-react';
import { useAuthStore } from '../stores';

export function LoginPage() {
  const { t } = useTranslation();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const { login, isLoading, error, clearError } = useAuthStore();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    clearError();
    try {
      await login(username, password);
    } catch {
      // error 已在 store 中处理
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* 背景装饰 */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-blue-600/20 rounded-full blur-[120px]"></div>
        <div className="absolute top-[40%] -right-[10%] w-[40%] h-[60%] bg-purple-600/20 rounded-full blur-[120px]"></div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg border border-white/20 p-8 rounded-3xl w-full max-w-md shadow-2xl z-10 animate-in fade-in zoom-in duration-500">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-900/50">
            <Layers size={32} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            {t('login.title')}
          </h1>
          <p className="text-gray-400 text-sm mt-2">
            {t('login.subtitle')}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-medium text-gray-300 uppercase mb-2 ml-1">
              {t('login.account')}
            </label>
            <div className="relative">
              <Users
                size={18}
                className="absolute left-3 top-3.5 text-gray-400"
              />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-600 text-white rounded-xl py-3 pl-10 pr-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all placeholder-gray-500"
                placeholder={t('login.usernamePlaceholder')}
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 uppercase mb-2 ml-1">
              {t('login.password')}
            </label>
            <div className="relative">
              <Lock
                size={18}
                className="absolute left-3 top-3.5 text-gray-400"
              />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-gray-800/50 border border-gray-600 text-white rounded-xl py-3 pl-10 pr-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all placeholder-gray-500"
                placeholder="••••••••"
              />
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-400 text-xs bg-red-900/20 p-3 rounded-lg border border-red-900/30">
              <AlertCircle size={14} />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3.5 rounded-xl transition-all shadow-lg shadow-blue-900/40 active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-60"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              t('login.signIn')
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
