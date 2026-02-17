import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type ActiveTab = 'chat' | 'ingest' | 'users' | 'graph' | 'compare';

interface UIState {
  // 布局尺寸
  sidebarWidth: number;
  canvasWidth: number;

  // 面板开关
  isSidebarOpen: boolean;
  isCanvasOpen: boolean;
  isHistoryOpen: boolean;

  // 当前 Tab
  activeTab: ActiveTab;

  // Modals
  showSettingsModal: boolean;
  showCreateCollectionModal: boolean;
  showUserModal: boolean;

  // Actions
  setSidebarWidth: (width: number) => void;
  setCanvasWidth: (width: number) => void;
  toggleSidebar: () => void;
  toggleCanvas: () => void;
  setCanvasOpen: (open: boolean) => void;
  toggleHistory: () => void;
  setActiveTab: (tab: ActiveTab) => void;
  setShowSettingsModal: (show: boolean) => void;
  setShowCreateCollectionModal: (show: boolean) => void;
  setShowUserModal: (show: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarWidth: 340,
      canvasWidth: 500,
      isSidebarOpen: true,
      isCanvasOpen: false,
      isHistoryOpen: false,
      activeTab: 'chat',
      showSettingsModal: false,
      showCreateCollectionModal: false,
      showUserModal: false,

      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      setCanvasWidth: (width) => set({ canvasWidth: width }),
      toggleSidebar: () => set((s) => ({ isSidebarOpen: !s.isSidebarOpen })),
      toggleCanvas: () => set((s) => ({ isCanvasOpen: !s.isCanvasOpen })),
      setCanvasOpen: (open) => set({ isCanvasOpen: open }),
      toggleHistory: () => set((s) => ({ isHistoryOpen: !s.isHistoryOpen })),
      setActiveTab: (tab) => set({ activeTab: tab }),
      setShowSettingsModal: (show) => set({ showSettingsModal: show }),
      setShowCreateCollectionModal: (show) => set({ showCreateCollectionModal: show }),
      setShowUserModal: (show) => set({ showUserModal: show }),
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        sidebarWidth: state.sidebarWidth,
        canvasWidth: state.canvasWidth,
        isSidebarOpen: state.isSidebarOpen,
      }),
    }
  )
);
