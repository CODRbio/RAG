import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Settings, Database, MessageSquare, UploadCloud, FileText, Search, ExternalLink, 
  ChevronLeft, ChevronRight, Cpu, Layers, CheckCircle, FileSearch, ArrowRight, 
  BarChart3, AlertCircle, Plus, Server, X, RefreshCw, Trash2, Loader2, Save, 
  Info, Image, Table, Activity, HardDrive, Users, LogOut, Lock, Shield, UserPlus, 
  Globe, Link, Sliders, Filter, MessageSquarePlus, Download, Copy, History, 
  Clock, MoreHorizontal, FileEdit, Network, GitBranch, GripVertical, PanelRightClose, 
  PanelRightOpen, PlugZap, Archive, ArchiveRestore, FileDown, FileType, Pin
} from 'lucide-react';

// --- 登录组件 ---
const LoginScreen = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    // 模拟登录请求
    setTimeout(() => {
      if (username === 'admin' && password === 'admin') {
        onLogin({ username: 'Administrator', role: 'admin', avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=admin' });
      } else if (username === 'user' && password === 'user') {
        onLogin({ username: 'Research User', role: 'user', avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=user' });
      } else {
        setError('用户名或密码错误 (提示: admin/admin 或 user/user)');
        setIsLoading(false);
      }
    }, 800);
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
          <h1 className="text-2xl font-bold text-white tracking-tight">深海科研 RAG 系统</h1>
          <p className="text-gray-400 text-sm mt-2">Remote Research & Collaboration Platform</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-xs font-medium text-gray-300 uppercase mb-2 ml-1">Account</label>
            <div className="relative">
              <Users size={18} className="absolute left-3 top-3.5 text-gray-400" />
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} className="w-full bg-gray-800/50 border border-gray-600 text-white rounded-xl py-3 pl-10 pr-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all placeholder-gray-500" placeholder="Username" />
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-300 uppercase mb-2 ml-1">Password</label>
            <div className="relative">
              <Lock size={18} className="absolute left-3 top-3.5 text-gray-400" />
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full bg-gray-800/50 border border-gray-600 text-white rounded-xl py-3 pl-10 pr-4 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all placeholder-gray-500" placeholder="••••••••" />
            </div>
          </div>
          {error && <div className="flex items-center gap-2 text-red-400 text-xs bg-red-900/20 p-3 rounded-lg border border-red-900/30"><AlertCircle size={14} />{error}</div>}
          <button type="submit" disabled={isLoading} className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3.5 rounded-xl transition-all shadow-lg shadow-blue-900/40 active:scale-[0.98] flex items-center justify-center gap-2">{isLoading ? <Loader2 size={20} className="animate-spin" /> : 'Sign In'}</button>
        </form>
      </div>
    </div>
  );
};

// --- 主应用组件 ---
const App = () => {
  // 用户鉴权状态
  const [currentUser, setCurrentUser] = useState(null); 
  
  const [activeTab, setActiveTab] = useState('chat');
  const [dbStatus, setDbStatus] = useState('disconnected'); 
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  
  // 布局与尺寸状态 (Resizable)
  const [sidebarWidth, setSidebarWidth] = useState(340); // 稍微加宽以容纳详细配置
  const [canvasWidth, setCanvasWidth] = useState(500);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCanvasOpen, setIsCanvasOpen] = useState(false); 
  const [isHistoryOpen, setIsHistoryOpen] = useState(false); 

  // 拖拽逻辑
  const isResizingRef = useRef(false);
  const startResizingSidebar = useCallback(() => { isResizingRef.current = 'sidebar'; }, []);
  const startResizingCanvas = useCallback(() => { isResizingRef.current = 'canvas'; }, []);
  const stopResizing = useCallback(() => { isResizingRef.current = false; }, []);
  
  const resize = useCallback((e) => {
    if (isResizingRef.current === 'sidebar') {
      const newWidth = e.clientX;
      if (newWidth > 280 && newWidth < 500) setSidebarWidth(newWidth);
    } else if (isResizingRef.current === 'canvas') {
      const newWidth = window.innerWidth - e.clientX;
      if (newWidth > 350 && newWidth < 900) setCanvasWidth(newWidth);
    }
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  // Local RAG Settings
  const [localTopK, setLocalTopK] = useState(5);
  const [enableHippoRAG, setEnableHippoRAG] = useState(false);
  const [enableReranker, setEnableReranker] = useState(true);
  
  // --- Web 检索状态 (升级版: 每源独立配置) ---
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [webSources, setWebSources] = useState([
    { id: 'tavily', name: 'Tavily API', enabled: true, topK: 5, threshold: 0.5 },
    { id: 'google', name: 'Google Search', enabled: false, topK: 5, threshold: 0.4 },
    { id: 'scholar', name: 'Google Scholar', enabled: false, topK: 3, threshold: 0.6 },
    { id: 'semantic', name: 'Semantic Scholar', enabled: false, topK: 3, threshold: 0.7 },
  ]);

  // LangGraph Workflow State
  const [workflowStep, setWorkflowStep] = useState('idle'); // idle, explore, outline, drafting, refine
  const [canvasContent, setCanvasContent] = useState(''); // 画布内容

  // Modals
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showCreateCollectionModal, setShowCreateCollectionModal] = useState(false);
  const [showUserModal, setShowUserModal] = useState(false); 

  // Data
  const [files, setFiles] = useState([
    { id: 1, name: 'DeepSea_Mining_Env_Impact.pdf', status: 'Success', time: '10 min ago' },
    { id: 2, name: 'AUV_Navigation_Survey_2024.pdf', status: 'Success', time: '1 hr ago' }
  ]);
  
  // 历史记录 (增加 isArchived 字段)
  const [chatHistory, setChatHistory] = useState([
    { id: 'h1', title: '深海采矿环境影响综述', date: '2023-10-25 14:30', preview: '根据最新文献生成的环境评估...', isArchived: true },
    { id: 'h2', title: 'AUV 导航算法调研', date: '2023-10-24 09:15', preview: 'SLAM 与 USBL 融合方案...', isArchived: false },
  ]);
  
  const [userList, setUserList] = useState([
    { id: 1, username: 'Administrator', role: 'admin', status: 'Active', created: '2023-01-01' },
    { id: 2, username: 'Research User', role: 'user', status: 'Active', created: '2023-06-15' },
  ]);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dbAddress, setDbAddress] = useState('localhost:19530');
  const [currentCollection, setCurrentCollection] = useState('deepsea_research_v1');
  const [collections, setCollections] = useState(['deepsea_research_v1', 'general_ocean_v2']);
  
  const [toasts, setToasts] = useState([]);
  const addToast = (msg, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
  };

  // Auth Handlers
  const handleLogin = (user) => { setCurrentUser(user); addToast(`欢迎回来, ${user.username}`, 'success'); };
  const handleLogout = () => { setCurrentUser(null); setDbStatus('disconnected'); setActiveTab('chat'); addToast('您已安全注销', 'info'); };
  
  // Web 检索配置更新函数
  const toggleWebSource = (id) => {
    setWebSources(prev => prev.map(s => s.id === id ? { ...s, enabled: !s.enabled } : s));
  };
  const updateWebSourceParam = (id, field, value) => {
    setWebSources(prev => prev.map(s => s.id === id ? { ...s, [field]: value } : s));
  };

  // 历史记录归档
  const toggleArchiveHistory = (e, id) => {
    e.stopPropagation();
    setChatHistory(prev => prev.map(h => {
      if (h.id === id) {
        const newState = !h.isArchived;
        addToast(newState ? '项目已归档 (跳出清理周期)' : '项目已取消归档', 'success');
        return { ...h, isArchived: newState };
      }
      return h;
    }));
  };

  // Handlers
  const handleConnect = () => {
    setDbStatus('connecting');
    addToast(`正在连接 Milvus 服务 (${dbAddress})...`, 'info');
    setTimeout(() => {
      setDbStatus('connected');
      addToast('Docker 容器连接成功！', 'success');
      setMessages([{ role: 'assistant', content: '系统已就绪。我是您的深海科研助手，支持多源检索与协作式综述生成。' }]);
    }, 1500);
  };

  const handleSend = () => {
    if (!inputValue.trim()) return;
    const newMsg = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, newMsg]);
    setInputValue('');
    
    // 模拟 LangGraph 工作流触发
    setWorkflowStep('explore');
    
    setTimeout(() => {
      // 步骤1：检索
      setWorkflowStep('outline');
      const activeWebSources = webSources.filter(s => s.enabled);
      const isWeb = webSearchEnabled && activeWebSources.length > 0;
      
      const sources = [
        { id: 1, title: 'DeepSea_Mining_Env_Impact.pdf', score: 0.98, snippet: 'Sediment plumes generated by mining vehicles...', path: '/local/docs/paper1.pdf', type: 'local' }
      ];

      // 模拟多源 Web 检索结果
      if (isWeb) {
        activeWebSources.forEach((src, idx) => {
           sources.push({ 
            id: 100 + idx, 
            title: `[Web] ${src.name} Result`, 
            score: (src.threshold + 0.1).toFixed(2), // 模拟分数
            snippet: `Results retrieved from ${src.name} (Top-${src.topK}). Content matched with threshold > ${src.threshold}...`, 
            path: `https://${src.id}.com/search`, 
            type: 'web' 
          });
        });
      }

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `已完成检索。本地命中 1 条，Web 检索启用 ${activeWebSources.length} 个源。正在构建综述大纲...`,
        sources: sources
      }]);

      // 步骤2：生成 Draft (更新 Canvas)
      setTimeout(() => {
        setWorkflowStep('drafting');
        setIsCanvasOpen(true); // 自动打开 Canvas
        setCanvasContent(`# ${newMsg.content} - Research Draft\n\n## 1. Introduction\nDeep sea mining presents significant environmental challenges...\n\n## 2. Sediment Plumes\nAccording to [1], the impact radius...\n\n## 3. Web Insights\nRecent studies from Google Scholar indicate...\n\n## 4. Policy Implications\n...`);
        
        setTimeout(() => {
          setWorkflowStep('refine');
          setMessages(prev => [...prev, { 
            role: 'assistant', 
            content: `初稿已生成到右侧 Canvas。您可以进一步提问以润色特定章节，或点击右上角导出文档。`
          }]);
          setTimeout(() => setWorkflowStep('idle'), 1000);
        }, 1500);
      }, 1500);

    }, 1000);
  };

  const handleNewChat = () => {
    if (messages.length > 0) {
      const newHistoryItem = {
        id: `h-${Date.now()}`,
        title: messages[0].content.substring(0, 15) + '...',
        date: new Date().toLocaleString(),
        preview: messages[messages.length - 1].content.substring(0, 20) + '...',
        isArchived: false
      };
      setChatHistory(prev => [newHistoryItem, ...prev]);
    }
    setMessages([]); 
    setCanvasContent('');
    setIsCanvasOpen(false);
    addToast('已开启新对话上下文', 'success');
  };

  const handleCreateUser = (userData) => {
    setShowUserModal(false);
    setUserList(prev => [...prev, { ...userData, id: Date.now(), status: 'Active', created: 'Just now' }]);
    addToast(`用户 ${userData.username} 创建成功`, 'success');
  };

  const handleUpload = () => {
    setIsUploading(true);
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          setFiles(prev => [{ id: Date.now(), name: 'New_Uploaded_File.pdf', status: 'Success', time: '刚刚' }, ...prev]);
          addToast(`文件已写入集合: ${currentCollection}`, 'success');
          return 0;
        }
        return prev + 10;
      });
    }, 200);
  };

  const handleDeleteFile = (id) => {
    setFiles(prev => prev.filter(f => f.id !== id));
    addToast('文档已从索引中移除', 'info');
  };

  const handleTabChange = (tab) => {
    // 修复：允许在未连接状态下切换回 'chat'，因为 Chat 页面包含连接按钮
    if (dbStatus !== 'connected' && tab !== 'users' && tab !== 'chat') { 
      addToast('请先连接向量数据库以访问此功能', 'error');
      return;
    }
    setActiveTab(tab);
  };

  const handleCreateCollection = (name) => {
    setShowCreateCollectionModal(false);
    addToast(`正在初始化集合: ${name}...`, 'info');
    setTimeout(() => {
      setCollections(prev => [...prev, name]);
      setCurrentCollection(name);
      addToast(`集合 ${name} 创建成功`, 'success');
    }, 1000);
  };

  const handleRefresh = () => {
    if (dbStatus !== 'connected') return;
    addToast('正在同步 Docker 容器状态...', 'info');
    setTimeout(() => addToast('状态同步正常: Latency 2ms', 'success'), 1000);
  };

  const handleDeleteUser = (id) => {
    if (id === 1) {
      addToast('无法删除超级管理员账号', 'error');
      return;
    }
    setUserList(prev => prev.filter(u => u.id !== id));
    addToast('用户已删除', 'success');
  };

  const handleLoadHistory = (historyId) => {
    addToast(`正在恢复会话: ${historyId}`, 'info');
    const item = chatHistory.find(h => h.id === historyId);
    if (item) {
      setMessages([
        { role: 'user', content: `(恢复的对话) 关于 ${item.title} 的问题...` },
        { role: 'assistant', content: `这是历史会话 "${item.title}" 的上下文记录。您可以继续提问。` }
      ]);
    }
  };

  const handleDeleteHistory = (e, historyId) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(h => h.id !== historyId));
    addToast('历史会话已删除', 'success');
  };

  const handleExportCanvas = (format) => {
    if (!canvasContent) {
      addToast('画布为空，无法导出', 'error');
      return;
    }
    const blob = new Blob([canvasContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_draft.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addToast(`文档已导出为 ${format.toUpperCase()}`, 'success');
  };

  const handleExportMessage = (content) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_snippet.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addToast('消息片段已导出', 'success');
  };

  // 渲染如果未登录
  if (!currentUser) return <LoginScreen onLogin={handleLogin} />;

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans overflow-hidden">
      {/* Toast Container */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id} className={`px-4 py-3 rounded-lg shadow-lg text-sm font-medium animate-in slide-in-from-right fade-in duration-300 pointer-events-auto ${t.type === 'success' ? 'bg-green-600 text-white' : 'bg-gray-800 text-white'}`}>{t.msg}</div>
        ))}
      </div>

      {/* --- Modals --- */}
      {showUserModal && (
        <div className="fixed inset-0 bg-black/50 z-[70] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-2xl w-full max-w-sm p-6 shadow-2xl animate-in zoom-in-95 duration-200" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2"><UserPlus size={20}/> 新建用户</h3>
            <form onSubmit={(e) => {
              e.preventDefault();
              const formData = new FormData(e.target);
              handleCreateUser({ username: formData.get('username'), role: formData.get('role') });
            }} className="space-y-4">
              <div><label className="text-xs font-medium text-gray-500 uppercase">用户名</label><input name="username" required type="text" className="w-full mt-1 border rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"/></div>
              <div><label className="text-xs font-medium text-gray-500 uppercase">初始密码</label><input name="password" required type="password" className="w-full mt-1 border rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"/></div>
              <div><label className="text-xs font-medium text-gray-500 uppercase">角色权限</label><select name="role" className="w-full mt-1 border rounded-md p-2 text-sm bg-white"><option value="user">普通用户 (Research User)</option><option value="admin">系统管理员 (Admin)</option></select></div>
              <div className="mt-6 flex justify-end gap-2"><button type="button" onClick={() => setShowUserModal(false)} className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm">取消</button><button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm">创建账户</button></div>
            </form>
          </div>
        </div>
      )}
      {showCreateCollectionModal && (
        <div className="fixed inset-0 bg-black/50 z-[70] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-2xl w-full max-w-sm p-6 shadow-2xl animate-in zoom-in-95 duration-200" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-bold mb-4">新建向量集合</h3>
            <div className="space-y-4"><div><label className="text-xs font-medium text-gray-500 uppercase">集合名称</label><input id="new-col-name" type="text" className="w-full mt-1 border rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"/></div><div><label className="text-xs font-medium text-gray-500 uppercase">向量维度</label><select className="w-full mt-1 border rounded-md p-2 text-sm bg-gray-50"><option>1536 (OpenAI)</option><option>1024 (BGE-Large)</option></select></div></div>
            <div className="mt-6 flex justify-end gap-2"><button onClick={() => setShowCreateCollectionModal(false)} className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm">取消</button><button onClick={() => handleCreateCollection(document.getElementById('new-col-name').value || 'new_collection')} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm">创建</button></div>
          </div>
        </div>
      )}
      {showSettingsModal && (
        <div className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-2xl w-full max-w-lg p-6 shadow-2xl animate-in zoom-in-95 duration-200 border border-gray-100" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-6 pb-4 border-b"><h3 className="text-lg font-bold text-gray-900 flex items-center gap-2"><Shield size={20} className="text-blue-600"/> 系统高级配置</h3><button onClick={() => setShowSettingsModal(false)} className="p-2 hover:bg-gray-100 rounded-full"><X size={20}/></button></div>
            <div className="space-y-5"><div className="grid grid-cols-2 gap-4"><div><label className="text-sm font-medium text-gray-700 block mb-1">API Timeout (ms)</label><input type="number" defaultValue={30000} className="w-full bg-gray-50 border border-gray-200 rounded-lg p-2.5 text-sm"/></div><div><label className="text-sm font-medium text-gray-700 block mb-1">Max Tokens</label><input type="number" defaultValue={4096} className="w-full bg-gray-50 border border-gray-200 rounded-lg p-2.5 text-sm"/></div></div><div className="bg-yellow-50 p-3 rounded-lg flex gap-3 text-sm text-yellow-800 border border-yellow-100"><AlertCircle size={18} className="flex-shrink-0 mt-0.5" /><p>警告：修改这些参数会影响所有用户的请求行为。请谨慎操作。</p></div></div>
            <div className="mt-8 flex justify-end gap-3 pt-4 border-t"><button onClick={() => setShowSettingsModal(false)} className="px-5 py-2.5 text-gray-600 hover:bg-gray-100 rounded-xl text-sm font-medium">取消</button><button onClick={() => { setShowSettingsModal(false); addToast('系统全局配置已更新', 'success'); }} className="px-5 py-2.5 bg-gray-900 text-white rounded-xl hover:bg-black text-sm font-medium">保存更改</button></div>
          </div>
        </div>
      )}

      {/* --- 左侧边栏 (Resizable) --- */}
      <div 
        className="bg-white border-r flex flex-col relative flex-shrink-0 z-40 transition-none"
        style={{ width: isSidebarOpen ? sidebarWidth : 80 }}
      >
        {/* 拖拽把手 (Right Handle) */}
        {isSidebarOpen && (
          <div 
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-blue-400 z-50 group transition-colors"
            onMouseDown={startResizingSidebar}
          >
            <div className="absolute top-1/2 -right-3 w-6 h-8 bg-white border rounded shadow flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
              <GripVertical size={12} className="text-gray-400"/>
            </div>
          </div>
        )}

        <div className="p-6 flex items-center gap-3 border-bottom h-20 overflow-hidden">
          <div className="bg-blue-600 p-2 rounded-lg text-white flex-shrink-0 shadow-lg shadow-blue-200">
            <Layers size={24} />
          </div>
          {isSidebarOpen && <span className="font-bold text-xl tracking-tight whitespace-nowrap">RAG Lab</span>}
        </div>

        {/* 侧边栏内容区 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-8 scrollbar-hide">
          {/* 用户信息 */}
          <div className={`flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100 ${!isSidebarOpen && 'justify-center'}`}>
            <img src={currentUser.avatar} alt="Avatar" className="w-10 h-10 rounded-full bg-white border" />
            {isSidebarOpen && (
              <div className="flex-1 min-w-0">
                <div className="font-bold text-sm truncate text-gray-900">{currentUser.username}</div>
                <div className="text-xs text-gray-500 flex items-center gap-1"><Shield size={10} className="text-blue-600"/><span className="capitalize">{currentUser.role}</span></div>
              </div>
            )}
          </div>

          {/* 引擎配置 */}
          <section>
            <div className="flex items-center gap-2 mb-4 text-gray-500 font-semibold text-xs uppercase tracking-wider whitespace-nowrap">
              <Cpu size={14} /> {isSidebarOpen && '检索策略配置'}
            </div>
            {isSidebarOpen && (
              <div className="space-y-3 animate-in fade-in duration-300">
                
                {/* 核心检索参数 */}
                <div className="bg-gray-50 rounded-lg p-3 border border-gray-200 space-y-3">
                    {/* Top-K Slider */}
                    <div>
                      <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                        <span>Local RAG Top-K</span>
                        <span className="font-mono bg-white px-1 rounded border">{localTopK}</span>
                      </div>
                      <input type="range" min="1" max="50" step="1" value={localTopK} onChange={(e) => setLocalTopK(e.target.value)} className="w-full accent-blue-600 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"/>
                    </div>

                    {/* HippoRAG Toggle (New) */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Network size={14} className="text-purple-500"/>
                        <span className="text-sm text-gray-700">HippoRAG 图检索</span>
                      </div>
                      <input type="checkbox" checked={enableHippoRAG} onChange={(e) => setEnableHippoRAG(e.target.checked)} className="accent-purple-600 w-4 h-4 cursor-pointer" />
                    </div>

                    {/* Reranker Toggle (New) */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Filter size={14} className="text-orange-500"/>
                        <span className="text-sm text-gray-700">ColBERT 重排序</span>
                      </div>
                      <input type="checkbox" checked={enableReranker} onChange={(e) => setEnableReranker(e.target.checked)} className="accent-orange-600 w-4 h-4 cursor-pointer" />
                    </div>
                </div>
              </div>
            )}
          </section>

          {/* Web Search Config (Enhanced) */}
          <section>
            <div className="flex items-center gap-2 mb-4 text-gray-500 font-semibold text-xs uppercase tracking-wider whitespace-nowrap">
              <Globe size={14} /> {isSidebarOpen && 'Web 增强检索'}
            </div>
            {isSidebarOpen && (
              <div className="space-y-3 animate-in fade-in duration-300">
                {/* 总开关 */}
                <div className="border border-gray-200 rounded-xl p-3 hover:border-blue-200 hover:bg-blue-50/50 transition-all cursor-pointer">
                  <label className="flex items-center justify-between cursor-pointer w-full">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg shadow-sm transition-colors ${webSearchEnabled ? 'bg-indigo-50 text-indigo-600' : 'bg-gray-100 text-gray-400'}`}>
                        <Globe size={16} />
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-700">联网搜索</div>
                        <div className="text-[10px] text-gray-400">实时获取互联网信息</div>
                      </div>
                    </div>
                    <input 
                      type="checkbox" 
                      checked={webSearchEnabled}
                      onChange={(e) => {
                        setWebSearchEnabled(e.target.checked);
                        addToast(e.target.checked ? 'Web 检索已启用' : 'Web 检索已关闭', 'info');
                      }}
                      className="accent-blue-600 w-4 h-4 cursor-pointer" 
                    />
                  </label>
                </div>

                {webSearchEnabled && (
                  <div className="bg-white border border-gray-200 rounded-xl overflow-hidden animate-in slide-in-from-top-2 duration-200">
                    <div className="bg-gray-50 px-3 py-2 border-b text-[10px] font-bold text-gray-500 uppercase">Search Sources</div>
                    <div className="divide-y">
                      {webSources.map(source => (
                        <div key={source.id} className="p-3 hover:bg-gray-50 transition-colors">
                          <div className="flex items-center justify-between mb-2">
                            <span className={`text-sm font-medium ${source.enabled ? 'text-gray-800' : 'text-gray-400'}`}>{source.name}</span>
                            <input 
                              type="checkbox" 
                              checked={source.enabled} 
                              onChange={() => toggleWebSource(source.id)} 
                              className="accent-blue-600 w-4 h-4 cursor-pointer" 
                            />
                          </div>
                          {source.enabled && (
                            <div className="space-y-2 animate-in slide-in-from-top-1 duration-200">
                              <div className="flex items-center justify-between text-[10px] text-gray-500">
                                <span>Top-K: <span className="font-mono text-gray-900">{source.topK}</span></span>
                                <span>Threshold: <span className="font-mono text-gray-900">{source.threshold}</span></span>
                              </div>
                              <div className="grid grid-cols-2 gap-2">
                                <input 
                                  type="range" min="1" max="20" step="1" 
                                  value={source.topK} 
                                  onChange={(e) => updateWebSourceParam(source.id, 'topK', e.target.value)} 
                                  className="w-full accent-blue-600 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                />
                                <input 
                                  type="range" min="0" max="1" step="0.1" 
                                  value={source.threshold} 
                                  onChange={(e) => updateWebSourceParam(source.id, 'threshold', e.target.value)} 
                                  className="w-full accent-green-600 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Service Connection (Fixed: Always visible in sidebar) */}
          <section>
            <div className="flex items-center gap-2 mb-4 text-gray-500 font-semibold text-xs uppercase tracking-wider whitespace-nowrap">
              <Database size={14} /> {isSidebarOpen && '服务连接'}
            </div>
            {isSidebarOpen && (
              <div className="space-y-3 animate-in fade-in duration-300">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Service Address</label>
                  <div className="relative">
                    <input 
                      type="text" 
                      disabled={currentUser.role !== 'admin'}
                      className={`w-full bg-gray-50 border rounded-md p-2 text-sm pl-8 ${dbStatus === 'connected' ? 'text-green-700 border-green-200 bg-green-50' : ''} disabled:opacity-60 disabled:cursor-not-allowed`}
                      value={dbAddress}
                      onChange={(e) => setDbAddress(e.target.value)}
                    />
                    <Server size={14} className="absolute left-2.5 top-2.5 text-gray-400"/>
                  </div>
                </div>
                {dbStatus === 'connected' ? (
                  <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                    <div className="flex justify-between items-center text-xs text-gray-500">
                      <span>Status:</span>
                      <span className="text-green-600 flex items-center gap-1"><div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div> Online</span>
                    </div>
                  </div>
                ) : (
                  <button 
                    onClick={handleConnect} 
                    disabled={dbStatus === 'connecting'} 
                    className="w-full text-xs text-blue-600 font-medium py-2 border border-blue-100 rounded-md hover:bg-blue-50 flex justify-center items-center gap-2 transition-colors cursor-pointer"
                  >
                    {dbStatus === 'connecting' ? <Loader2 size={14} className="animate-spin"/> : <PlugZap size={14}/>}
                    {dbStatus === 'connecting' ? '连接中...' : '连接服务'}
                  </button>
                )}
              </div>
            )}
          </section>

          {/* Projects / History (Archived Support) */}
          <section>
             <div className="flex items-center justify-between mb-4 text-gray-500 font-semibold text-xs uppercase tracking-wider whitespace-nowrap">
              <div className="flex items-center gap-2"><History size={14} /> {isSidebarOpen && '历史项目'}</div>
            </div>
            {isSidebarOpen && (
              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {chatHistory.map(h => (
                  <div 
                    key={h.id} 
                    onClick={() => handleLoadHistory(h.id)}
                    className="group p-2 rounded-lg hover:bg-gray-100 cursor-pointer text-sm relative"
                  >
                    <div className="flex justify-between items-start">
                      <span className="font-medium text-gray-800 truncate flex-1">{h.title}</span>
                      {h.isArchived && <Pin size={12} className="text-yellow-500 fill-yellow-500 flex-shrink-0 ml-1"/>}
                    </div>
                    <div className="text-[10px] text-gray-400 flex items-center gap-1 mt-1"><Clock size={10}/> {h.date}</div>
                    
                    {/* Hover Actions */}
                    <div className="absolute right-1 top-1 hidden group-hover:flex gap-1 bg-white shadow-sm p-1 rounded border">
                       <button onClick={(e) => toggleArchiveHistory(e, h.id)} className="p-1 hover:text-yellow-600 text-gray-400" title={h.isArchived ? "取消归档" : "永久保存"}><Archive size={12}/></button>
                       <button onClick={(e) => handleDeleteHistory(e, h.id)} className="p-1 hover:text-red-600 text-gray-400" title="删除"><Trash2 size={12}/></button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
        
        {/* Footer Buttons */}
        <div className="p-4 border-t bg-gray-50/50 space-y-2">
          {currentUser.role === 'admin' && (
            <button onClick={() => setShowSettingsModal(true)} className={`w-full flex items-center justify-center gap-2 bg-gray-900 text-white py-3 rounded-xl text-sm font-medium hover:bg-black transition-all ${!isSidebarOpen && 'px-0'}`}><Settings size={18} />{isSidebarOpen && '高级配置'}</button>
          )}
          <button onClick={handleLogout} className={`w-full flex items-center justify-center gap-2 text-red-600 hover:bg-red-50 py-3 rounded-xl text-sm font-medium transition-all ${!isSidebarOpen && 'px-0'}`}><LogOut size={18} />{isSidebarOpen && '注销'}</button>
        </div>
      </div>

      {/* --- 中间主内容区 (Chat) --- */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-white relative">
        {/* 顶部导航 */}
        <header className="bg-white border-b px-6 h-16 flex items-center justify-between flex-shrink-0 z-30">
          <div className="flex gap-6">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="text-gray-400 hover:text-gray-600 p-1"><MoreHorizontal size={20}/></button>
            <div className="h-8 w-[1px] bg-gray-200"></div>
            <button onClick={() => handleTabChange('chat')} className={`h-16 flex items-center gap-2 px-1 border-b-2 font-medium text-sm transition-all focus:outline-none ${activeTab === 'chat' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500'}`}><MessageSquare size={18} /> 智能问答</button>
            <button onClick={() => handleTabChange('ingest')} className={`h-16 flex items-center gap-2 px-1 border-b-2 font-medium text-sm transition-all focus:outline-none ${activeTab === 'ingest' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500'}`}><UploadCloud size={18} /> 数据入库</button>
            {currentUser.role === 'admin' && (<button onClick={() => handleTabChange('users')} className={`h-16 flex items-center gap-2 px-1 border-b-2 font-medium text-sm transition-all focus:outline-none ${activeTab === 'users' ? 'border-purple-600 text-purple-600' : 'border-transparent text-gray-400 hover:text-gray-600'}`}><Users size={18} />用户管理</button>)}
          </div>
          
          {/* Workflow Status Indicator */}
          {workflowStep !== 'idle' && (
            <div className="flex items-center gap-2 px-4 py-1.5 bg-blue-50 text-blue-700 rounded-full text-xs font-bold border border-blue-100 animate-pulse">
              <GitBranch size={14} />
              <span className="uppercase tracking-wide">Workflow: {workflowStep}</span>
            </div>
          )}

          <div className="flex items-center gap-3">
             {activeTab === 'chat' && dbStatus === 'connected' && (
               <>
                 <button onClick={handleNewChat} className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-xs font-medium transition-colors border border-gray-200"><MessageSquarePlus size={14} /> 新对话</button>
                 <button onClick={() => setIsHistoryOpen(!isHistoryOpen)} className={`p-2 rounded-lg transition-colors cursor-pointer ${isHistoryOpen ? 'bg-blue-50 text-blue-600' : 'hover:bg-gray-100 text-gray-500'}`} title="切换历史记录"><History size={18} /></button>
               </>
             )}
             
             {/* 修复：顶部状态栏现在是一个可点击的按钮 */}
             {dbStatus === 'connected' ? (
                <button onClick={handleRefresh} className="flex items-center gap-2 px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs font-medium border border-green-100 shadow-sm cursor-pointer hover:bg-green-100 transition-colors">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  System: Online
                </button>
              ) : (
                <button 
                  onClick={handleConnect}
                  className="flex items-center gap-2 px-3 py-1 bg-red-50 text-red-700 hover:bg-red-100 rounded-full text-xs font-medium border border-red-100 shadow-sm transition-colors cursor-pointer animate-pulse"
                  title="点击连接数据库"
                >
                  <PlugZap size={14} />
                  Disconnected (Click to Connect)
                </button>
              )}

             <button 
                onClick={() => setIsCanvasOpen(!isCanvasOpen)} 
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${isCanvasOpen ? 'bg-blue-100 border-blue-200 text-blue-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}`}
                title="Toggle Canvas"
             >
               {isCanvasOpen ? <PanelRightClose size={14}/> : <PanelRightOpen size={14}/>} Canvas
             </button>
          </div>
        </header>

        {/* 聊天内容滚动区 */}
        <main className="flex-1 overflow-y-auto bg-gray-50/50 p-8 scrollbar-thin scrollbar-thumb-gray-200">
           {/* 未连接状态 (非用户管理Tab) */}
           {dbStatus === 'disconnected' && activeTab !== 'users' && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 animate-in fade-in zoom-in duration-300 z-10 bg-gray-50/90 backdrop-blur-sm">
                <div className="bg-white p-12 rounded-3xl shadow-xl max-w-2xl w-full text-center border border-gray-100"><div className="w-20 h-20 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm"><Database size={40} /></div><h2 className="text-3xl font-bold text-gray-900 mb-3">连接远程服务</h2><p className="text-gray-500 mb-10 max-w-md mx-auto">请连接至 Docker 容器以访问向量数据。</p><div className="max-w-xs mx-auto"><button onClick={handleConnect} className="w-full group p-4 rounded-xl border-2 border-blue-600 bg-blue-600 text-white hover:bg-blue-700 transition-all active:scale-95 cursor-pointer flex items-center justify-center gap-3 shadow-lg shadow-blue-200"><Server size={20} /><span className="font-bold">连接服务节点</span></button></div></div>
              </div>
            )}

           {activeTab === 'chat' ? (
             <div className="max-w-3xl mx-auto space-y-6 pb-24">
                {messages.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                    <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mb-4"><MessageSquare size={32} className="opacity-40"/></div>
                    <p>开启新的科研对话...</p>
                  </div>
                )}
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2 duration-300`}>
                    <div className={`max-w-[90%] rounded-2xl p-5 shadow-sm group ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border text-gray-800'}`}>
                      <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                      
                      {/* 助手消息底部工具栏 */}
                      {msg.role === 'assistant' && (
                        <div className="mt-3 pt-2 border-t border-gray-100 flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                           <button onClick={() => { navigator.clipboard.writeText(msg.content); addToast('内容已复制', 'info'); }} className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors cursor-pointer" title="复制内容"><Copy size={14} /></button>
                           <button onClick={() => handleExportMessage(msg.content)} className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors cursor-pointer" title="导出片段"><Download size={14} /></button>
                        </div>
                      )}

                      {msg.sources && (
                        <div className="mt-4 pt-3 border-t border-gray-100/50 space-y-2">
                          <div className="text-[10px] font-bold uppercase tracking-wider opacity-70 flex items-center gap-1"><FileSearch size={10}/> References</div>
                          {msg.sources.map(src => (
                            <div key={src.id} className="flex items-center justify-between bg-black/5 p-2 rounded-lg text-xs hover:bg-black/10 transition-colors cursor-pointer">
                              <span className="truncate flex-1 font-medium">{src.title}</span>
                              <span className="ml-2 text-[10px] opacity-70 border px-1 rounded">{src.type === 'web' ? `Thresh>${src.score}` : `Score:${src.score}`}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
             </div>
           ) : activeTab === 'ingest' ? (
             <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in">
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="col-span-1 bg-gray-900 text-white rounded-2xl p-6 shadow-lg flex flex-col justify-between relative overflow-hidden"><div className="absolute top-0 right-0 p-4 opacity-10"><Database size={100} /></div><div><div className="flex items-center gap-2 text-gray-400 text-xs font-bold uppercase mb-2"><HardDrive size={14} /> Node Status</div><div className="text-2xl font-bold font-mono tracking-tight">{dbAddress}</div></div><div className="mt-6 space-y-2 text-sm text-gray-300"><div className="flex justify-between"><span>RAM:</span> <span className="text-white">4.2 GB / 16 GB</span></div><div className="flex justify-between"><span>Role:</span> <span className="text-green-400">Read/Write</span></div></div></div>
                  <div className="col-span-2 bg-white border rounded-2xl p-6 shadow-sm flex flex-col justify-center"><div className="flex justify-between items-center mb-6"><div><h3 className="text-lg font-bold text-gray-900">选择集合 (Collection)</h3><p className="text-xs text-gray-500">将文档上传至指定的知识库分区</p></div>{currentUser.role === 'admin' && (<button onClick={() => setShowCreateCollectionModal(true)} className="flex items-center gap-2 px-4 py-2 bg-blue-50 text-blue-600 rounded-lg text-sm font-bold hover:bg-blue-100 transition-colors"><Plus size={16} /> 新建集合</button>)}</div><div className="flex gap-3"><select value={currentCollection} onChange={(e) => setCurrentCollection(e.target.value)} className="flex-1 bg-gray-50 border border-gray-200 rounded-xl p-3 font-medium text-gray-700 outline-none focus:ring-2 focus:ring-blue-500">{collections.map(c => <option key={c} value={c}>{c}</option>)}</select></div></div>
                </div>
               {/* Ingest UI */}
               <div className="bg-white p-8 rounded-2xl border shadow-sm text-center">
                 <UploadCloud size={48} className="mx-auto text-blue-500 mb-4"/>
                 <h2 className="text-xl font-bold">Upload Documents</h2>
                 <p className="text-gray-500 mt-2 mb-6">Drag and drop PDFs to add to {currentCollection}</p>
                 {isUploading ? (<div className="w-full max-w-md mx-auto mt-6"><div className="flex justify-between text-xs mb-1 text-gray-500"><span>Indexing...</span><span>{uploadProgress}%</span></div><div className="h-2 bg-gray-100 rounded-full overflow-hidden"><div className="h-full bg-blue-600 transition-all duration-300" style={{ width: `${uploadProgress}%` }}></div></div></div>) : (<button onClick={handleUpload} className="bg-gray-900 text-white px-6 py-2 rounded-lg text-sm font-medium">Select Files</button>)}
               </div>
               
               {/* History Table */}
              <div className="bg-white border rounded-2xl overflow-hidden shadow-sm">
                 <div className="px-6 py-4 border-b bg-gray-50 flex justify-between items-center"><span className="font-bold text-sm">集合数据概览</span></div>
                <table className="w-full text-sm">
                  <thead><tr className="text-left text-gray-400 border-b"><th className="px-6 py-3 font-medium">文件名</th><th className="px-6 py-3 font-medium">状态</th><th className="px-6 py-3 font-medium text-right">操作</th></tr></thead>
                  <tbody className="divide-y">
                    {files.map((item) => (
                      <tr key={item.id} className="hover:bg-gray-50 transition-colors"><td className="px-6 py-4 font-medium flex items-center gap-2"><FileText size={16} className="text-gray-400" /> {item.name}</td><td className="px-6 py-4"><span className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase ${item.status === 'Success' ? 'bg-green-50 text-green-600' : 'bg-blue-50 text-blue-600'}`}>{item.status}</span></td><td className="px-6 py-4 text-right"><button onClick={() => handleDeleteFile(item.id)} className="text-gray-300 hover:text-red-500"><Trash2 size={16}/></button></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
             </div>
           ) : (
             <div className="max-w-5xl mx-auto space-y-8 p-8 animate-in fade-in duration-300">
               <div className="flex justify-between items-end">
                <div><h2 className="text-2xl font-bold text-gray-900">用户权限管理</h2><p className="text-gray-500 mt-1">管理系统访问权限与角色分配</p></div>
                <button onClick={() => setShowUserModal(true)} className="flex items-center gap-2 bg-blue-600 text-white px-5 py-2.5 rounded-xl font-medium shadow-lg shadow-blue-200 hover:bg-blue-700 transition-all"><UserPlus size={18} /> 新建用户</button>
               </div>
               <div className="bg-white border rounded-2xl overflow-hidden shadow-sm"><table className="w-full text-sm"><thead className="bg-gray-50"><tr className="text-left text-gray-500 border-b"><th className="px-6 py-4 font-medium">用户名 / 角色</th><th className="px-6 py-4 font-medium">状态</th><th className="px-6 py-4 font-medium">创建时间</th><th className="px-6 py-4 font-medium text-right">管理操作</th></tr></thead><tbody className="divide-y">{userList.map((user) => (<tr key={user.id} className="hover:bg-gray-50 transition-colors"><td className="px-6 py-4"><div className="flex items-center gap-3"><div className={`w-8 h-8 rounded-full flex items-center justify-center ${user.role === 'admin' ? 'bg-purple-100 text-purple-600' : 'bg-blue-100 text-blue-600'}`}>{user.role === 'admin' ? <Shield size={14}/> : <Users size={14}/>}</div><div><div className="font-bold text-gray-900">{user.username}</div><div className="text-xs text-gray-500 capitalize">{user.role}</div></div></div></td><td className="px-6 py-4"><span className={`px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide ${user.status === 'Active' ? 'bg-green-50 text-green-700 border border-green-100' : 'bg-gray-100 text-gray-500 border border-gray-200'}`}>{user.status}</span></td><td className="px-6 py-4 text-gray-500 font-mono text-xs">{user.created}</td><td className="px-6 py-4 text-right"><button onClick={() => handleDeleteUser(user.id)} className="text-gray-400 hover:text-red-600 hover:bg-red-50 p-2 rounded-lg transition-colors" title="删除用户"><Trash2 size={16} /></button></td></tr>))}</tbody></table></div>
             </div>
           )}
        </main>

        {/* 底部输入框 */}
        {activeTab === 'chat' && (
          <div className="p-6 bg-white border-t z-30">
            <div className="max-w-3xl mx-auto relative">
              <input type="text" value={inputValue} onChange={(e) => setInputValue(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} placeholder="输入研究问题，例如：'生成关于深海采矿羽流扩散的综述大纲'..." className="w-full bg-gray-50 border border-gray-300 rounded-xl py-4 pl-5 pr-14 shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all" />
              <button onClick={handleSend} className="absolute right-2 top-2 bottom-2 aspect-square bg-blue-600 text-white rounded-lg flex items-center justify-center hover:bg-blue-700 transition-colors"><ArrowRight size={20}/></button>
            </div>
          </div>
        )}
      </div>

      {/* --- 右侧 Canvas 面板 (Resizable) --- */}
      {isCanvasOpen && (
        <div 
          className="bg-white border-l flex flex-col relative flex-shrink-0 z-40 shadow-xl"
          style={{ width: canvasWidth }}
        >
          {/* 拖拽把手 (Left Handle) */}
          <div 
            className="absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-blue-400 z-50 group transition-colors"
            onMouseDown={startResizingCanvas}
          >
             <div className="absolute top-1/2 -left-3 w-6 h-8 bg-white border rounded shadow flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
              <GripVertical size={12} className="text-gray-400"/>
            </div>
          </div>

          <div className="h-14 border-b flex items-center justify-between px-4 bg-gray-50">
            <div className="flex items-center gap-2 font-bold text-sm text-gray-700">
              <FileEdit size={16} className="text-blue-600"/> 
              Research Canvas
              {workflowStep === 'drafting' && <Loader2 size={12} className="animate-spin text-gray-400"/>}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => handleExportCanvas('md')} className="p-1.5 hover:bg-white rounded text-gray-500" title="Export Markdown"><FileDown size={14}/></button>
              <button onClick={() => handleExportCanvas('pdf')} className="p-1.5 hover:bg-white rounded text-gray-500" title="Export PDF"><FileType size={14}/></button>
              <button onClick={() => setIsCanvasOpen(false)} className="p-1.5 hover:bg-white rounded text-gray-500"><X size={14}/></button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-white">
            {canvasContent ? (
              <div className="prose prose-sm max-w-none prose-headings:font-bold prose-h1:text-xl prose-h2:text-lg prose-p:text-gray-600">
                <pre className="whitespace-pre-wrap font-sans text-sm">{canvasContent}</pre>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-400 text-xs">
                <FileEdit size={32} className="mb-2 opacity-20"/>
                <p>Generated content will appear here</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* --- 右侧历史记录边栏 (Float) --- */}
      {activeTab === 'chat' && isHistoryOpen && dbStatus === 'connected' && (
        <div className={`absolute top-16 bottom-0 right-0 w-80 bg-white border-l flex flex-col animate-in slide-in-from-right duration-300 shadow-2xl z-50`}>
          <div className="p-4 border-b flex items-center justify-between bg-gray-50"><span className="font-bold text-sm text-gray-700 flex items-center gap-2"><History size={16} /> 历史对话</span><button onClick={() => setIsHistoryOpen(false)} className="text-gray-400 hover:text-gray-600"><X size={16} /></button></div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {chatHistory.length === 0 ? (<div className="text-center text-gray-400 py-10 text-xs">暂无历史记录</div>) : (chatHistory.map(history => (
                <div key={history.id} onClick={() => handleLoadHistory(history.id)} className="group p-3 rounded-xl border border-transparent hover:border-gray-200 hover:bg-gray-50 cursor-pointer transition-all relative">
                  <div className="flex justify-between items-start">
                    <h4 className="text-sm font-medium text-gray-800 line-clamp-1 pr-2">{history.title}</h4>
                    {history.isArchived && <Archive size={12} className="text-yellow-500 fill-yellow-500 flex-shrink-0"/>}
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-gray-400 mt-1"><Clock size={10} /> {history.date}</div>
                  <p className="text-xs text-gray-500 mt-2 line-clamp-2">{history.preview}</p>
                  <div className="absolute top-8 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-all">
                    <button onClick={(e) => toggleArchiveHistory(e, history.id)} className="p-1.5 text-gray-300 hover:text-yellow-600 hover:bg-white rounded-full" title={history.isArchived ? "取消永久保存" : "永久保存 (跳出清理)"}>{history.isArchived ? <ArchiveRestore size={12}/> : <Archive size={12}/>}</button>
                    <button onClick={(e) => handleDeleteHistory(e, history.id)} className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-white rounded-full" title="删除"><Trash2 size={12} /></button>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="p-4 border-t bg-gray-50"><button onClick={() => {setChatHistory([]); addToast('历史记录已清空', 'info');}} className="w-full text-xs text-red-500 hover:text-red-600 flex items-center justify-center gap-1 py-2 hover:bg-red-50 rounded-lg transition-colors"><Trash2 size={12} /> 清空所有记录</button></div>
        </div>
      )}
    </div>
  );
};

export default App;