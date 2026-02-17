import { create } from 'zustand';
import type { Canvas, CanvasStage, Annotation } from '../types';

interface CanvasState {
  canvas: Canvas | null;
  canvasContent: string; // Markdown 内容（用于 Refine 阶段预览和编辑）
  isLoading: boolean;

  // 阶段控制
  activeStage: CanvasStage;  // 当前用户正在查看的阶段（可独立于 canvas.stage 导航）

  // P1 编辑模式
  editMode: boolean;
  versionHistory: string[];      // 快照列表（Markdown 内容）
  currentVersionIndex: number;   // 当前版本指针（-1 = 当前编辑态）
  isAIEditing: boolean;

  // 批注（Refine 阶段）
  pendingAnnotations: Annotation[];

  setCanvas: (canvas: Canvas | null) => void;
  setCanvasContent: (content: string) => void;
  appendCanvasContent: (delta: string) => void;
  clearCanvas: () => void;
  setIsLoading: (loading: boolean) => void;

  // 阶段导航
  setActiveStage: (stage: CanvasStage) => void;

  // P1 编辑方法
  setEditMode: (mode: boolean) => void;
  pushVersion: () => void;        // 保存当前内容为快照
  undo: () => void;
  redo: () => void;
  setIsAIEditing: (editing: boolean) => void;

  // 批注管理
  addAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (id: string) => void;
  updateAnnotationStatus: (id: string, status: Annotation['status']) => void;
  clearAnnotations: () => void;

  // 全局指令管理
  addDirective: (directive: string) => void;
  removeDirective: (index: number) => void;
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  canvas: null,
  canvasContent: '',
  isLoading: false,

  activeStage: 'explore',

  editMode: false,
  versionHistory: [],
  currentVersionIndex: -1,
  isAIEditing: false,

  pendingAnnotations: [],

  setCanvas: (canvas) => set({
    canvas,
    activeStage: canvas?.stage || 'explore',
  }),
  setCanvasContent: (content) => set({ canvasContent: content }),
  appendCanvasContent: (delta) =>
    set((state) => ({ canvasContent: state.canvasContent + delta })),
  clearCanvas: () =>
    set({
      canvas: null,
      canvasContent: '',
      activeStage: 'explore',
      editMode: false,
      versionHistory: [],
      currentVersionIndex: -1,
      isAIEditing: false,
      pendingAnnotations: [],
    }),
  setIsLoading: (loading) => set({ isLoading: loading }),

  setActiveStage: (stage) => set({ activeStage: stage }),

  setEditMode: (mode) => set({ editMode: mode }),

  pushVersion: () => {
    const { canvasContent, versionHistory, currentVersionIndex } = get();
    // 如果不在最新版本，截断后续历史
    const base =
      currentVersionIndex >= 0
        ? versionHistory.slice(0, currentVersionIndex + 1)
        : [...versionHistory];
    const next = [...base, canvasContent];
    // 限制历史深度（最多 50 个快照）
    if (next.length > 50) next.shift();
    set({ versionHistory: next, currentVersionIndex: next.length - 1 });
  },

  undo: () => {
    const { versionHistory, currentVersionIndex } = get();
    if (versionHistory.length === 0) return;
    const idx =
      currentVersionIndex < 0
        ? versionHistory.length - 1
        : currentVersionIndex - 1;
    if (idx < 0) return;
    set({
      canvasContent: versionHistory[idx],
      currentVersionIndex: idx,
    });
  },

  redo: () => {
    const { versionHistory, currentVersionIndex } = get();
    if (currentVersionIndex < 0 || currentVersionIndex >= versionHistory.length - 1) return;
    const idx = currentVersionIndex + 1;
    set({
      canvasContent: versionHistory[idx],
      currentVersionIndex: idx,
    });
  },

  setIsAIEditing: (editing) => set({ isAIEditing: editing }),

  // 批注管理
  addAnnotation: (annotation) => set((state) => ({
    pendingAnnotations: [...state.pendingAnnotations, annotation],
  })),
  removeAnnotation: (id) => set((state) => ({
    pendingAnnotations: state.pendingAnnotations.filter((a) => a.id !== id),
  })),
  updateAnnotationStatus: (id, status) => set((state) => ({
    pendingAnnotations: state.pendingAnnotations.map((a) =>
      a.id === id ? { ...a, status } : a
    ),
  })),
  clearAnnotations: () => set({ pendingAnnotations: [] }),

  // 全局指令管理
  addDirective: (directive) => set((state) => {
    if (!state.canvas) return {};
    return {
      canvas: {
        ...state.canvas,
        user_directives: [...state.canvas.user_directives, directive],
      },
    };
  }),
  removeDirective: (index) => set((state) => {
    if (!state.canvas) return {};
    return {
      canvas: {
        ...state.canvas,
        user_directives: state.canvas.user_directives.filter((_, i) => i !== index),
      },
    };
  }),
}));
