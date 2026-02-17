import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User } from '../types';
import { login as apiLogin } from '../api/auth';

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;

  login: (userId: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isLoading: false,
      error: null,

      login: async (userId: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const res = await apiLogin({ user_id: userId, password });
          const user: User = {
            user_id: res.user_id,
            username: res.user_id,
            role: res.role as 'user' | 'admin',
            avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${res.user_id}`,
          };
          localStorage.setItem('token', res.token);
          set({ user, token: res.token, isLoading: false });
        } catch (err: unknown) {
          const message =
            (err as { response?: { data?: { detail?: string } } })?.response
              ?.data?.detail || '登录失败';
          set({ error: message, isLoading: false });
          throw err;
        }
      },

      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, token: null });
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ user: state.user, token: state.token }),
    }
  )
);
