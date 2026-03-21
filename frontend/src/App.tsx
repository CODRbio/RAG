import { useEffect, useCallback, useRef } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { useAuthStore, useUIStore } from './stores';
import { LoginPage } from './pages/LoginPage';
import { ChatPage } from './pages/ChatPage';
import { IngestPage } from './pages/IngestPage';
import { AdminPage } from './pages/AdminPage';
import { CompareView } from './components/compare/CompareView';
import { ScholarPage } from './pages/ScholarPage';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { CanvasPanel } from './components/canvas/CanvasPanel';
import { SettingsModal } from './components/settings/SettingsModal';
import { DeepResearchDialog } from './components/workflow/DeepResearchDialog';
import { LocalDbChoiceDialog } from './components/chat/LocalDbChoiceDialog';
import { ToastContainer } from './components/ui/Toast';
import { PaperWorkspacePage } from './pages/PaperWorkspacePage';
import { AnalysisWorkspacePage } from './pages/AnalysisWorkspacePage';
import { GraphWorkspacePage } from './pages/GraphWorkspacePage';
import { pathToActiveTab } from './utils/appRoutes';

function App() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const location = useLocation();

  const {
    isCanvasOpen,
    setSidebarWidth,
    setCanvasWidth,
    activeTab,
    setActiveTab,
  } = useUIStore();

  // 监听全局登出事件
  useEffect(() => {
    const handleLogout = () => logout();
    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, [logout]);

  // 拖拽调整面板宽度
  const isResizingRef = useRef<false | 'sidebar' | 'canvas'>(false);

  const stopResizing = useCallback(() => {
    isResizingRef.current = false;
  }, []);

  const resize = useCallback(
    (e: MouseEvent) => {
      if (isResizingRef.current === 'sidebar') {
        const newWidth = e.clientX;
        if (newWidth > 200 && newWidth < 600) {
          setSidebarWidth(newWidth);
        }
      } else if (isResizingRef.current === 'canvas') {
        const newWidth = window.innerWidth - e.clientX;
        if (newWidth > 350 && newWidth < 1200) {
          setCanvasWidth(newWidth);
        }
      }
    },
    [setSidebarWidth, setCanvasWidth]
  );

  const handleStartResize = (type: 'sidebar' | 'canvas') => {
    isResizingRef.current = type;
  };

  useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  useEffect(() => {
    const nextTab = pathToActiveTab(location.pathname);
    if (activeTab !== nextTab) {
      setActiveTab(nextTab);
    }
  }, [activeTab, location.pathname, setActiveTab]);

  // 未登录显示登录页
  if (!user) {
    return (
      <>
        <LoginPage />
        <ToastContainer />
      </>
    );
  }

  return (
    <div className="flex h-screen bg-[var(--bg-app)] text-[var(--text-primary)] font-sans overflow-hidden bg-cover bg-fixed">
      {/* Toast */}
      <ToastContainer />

      {/* Sidebar */}
      <Sidebar onStartResize={() => handleStartResize('sidebar')} />

      {/* 主内容区 */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative bg-transparent">
        <Header />
        <div className="flex-1 flex min-h-0 relative">
          <div className="flex-1 flex flex-col min-h-0 overflow-y-auto scrollbar-thin">
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/ingest" element={<IngestPage />} />
              <Route path="/scholar" element={<ScholarPage />} />
              <Route path="/papers/:paperUid" element={<PaperWorkspacePage />} />
              <Route path="/analysis" element={<AnalysisWorkspacePage />} />
              <Route path="/workspace/graph" element={<GraphWorkspacePage />} />
              <Route path="/graph" element={<Navigate to="/workspace/graph" replace />} />
              <Route path="/compare" element={<CompareView />} />
              <Route
                path="/users"
                element={user.role === 'admin' ? <AdminPage /> : <Navigate to="/chat" replace />}
              />
              <Route path="*" element={<Navigate to="/chat" replace />} />
            </Routes>
          </div>
        </div>
      </div>

      {/* Canvas Panel */}
      {isCanvasOpen && (
        <CanvasPanel onStartResize={() => handleStartResize('canvas')} />
      )}

      {/* 高级配置 Modal */}
      <SettingsModal />

      {/* Deep Research 澄清对话框 */}
      <DeepResearchDialog />

      {/* 查询与本地库范围不符时的选择弹窗 */}
      <LocalDbChoiceDialog />
    </div>
  );
}

export default App;
