import { create } from 'zustand';
import {
  askPaper,
  compareAssistantPapers,
  getAcademicAssistantTask,
  listAnnotations,
  startDiscovery,
  startMediaAnalysis,
  streamAcademicAssistantTask,
  summarizePaper,
  upsertAnnotation,
} from '../api/academicAssistant';
import type {
  AcademicAssistantTaskInfo,
  AcademicAssistantTaskState,
  AssistantScope,
  DiscoveryMode,
  PaperCompareResult,
  PaperLocator,
  PaperQaResult,
  PaperSummaryResult,
  ResourceAnnotation,
  ResourceAnnotationUpsert,
} from '../types';

const ACTIVE_TASKS_STORAGE_KEY = 'academic_assistant_active_tasks_v1';

function locatorKey(locator: PaperLocator): string {
  return locator.paper_uid || `${locator.paper_id || ''}:${locator.collection || ''}`;
}

function compareKey(paperUids: string[]): string {
  return [...paperUids].sort().join('|');
}

function readPersistedTaskIds(): string[] {
  try {
    const raw = localStorage.getItem(ACTIVE_TASKS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item) => String(item)).filter(Boolean);
  } catch {
    return [];
  }
}

function writePersistedTaskIds(taskIds: string[]): void {
  try {
    localStorage.setItem(ACTIVE_TASKS_STORAGE_KEY, JSON.stringify([...new Set(taskIds)].slice(-20)));
  } catch {
    // ignore
  }
}

function mergeTaskInfo(
  previous: AcademicAssistantTaskState | undefined,
  info: AcademicAssistantTaskInfo,
): AcademicAssistantTaskState {
  const payload = (info.payload || {}) as Record<string, unknown>;
  const rawStatus = String(info.status || '').toLowerCase();
  const status =
    rawStatus === 'completed' || rawStatus === 'error' || rawStatus === 'cancelled' || rawStatus === 'timeout'
      ? rawStatus
      : rawStatus === 'submitted'
        ? 'submitted'
        : 'running';
  return {
    task_id: info.task_id,
    status,
    task_type: String(payload.type || previous?.task_type || 'task'),
    mode: payload.mode ? String(payload.mode) : previous?.mode || null,
    message: previous?.message || null,
    result: (payload.result as Record<string, unknown> | undefined) || previous?.result || null,
    error_message: info.error_message || previous?.error_message || null,
    updated_at: Date.now(),
  };
}

interface AcademicAssistantStoreState {
  summaries: Record<string, PaperSummaryResult>;
  answers: Record<string, PaperQaResult>;
  comparisons: Record<string, PaperCompareResult>;
  annotations: Record<string, ResourceAnnotation[]>;
  tasks: Record<string, AcademicAssistantTaskState>;
  loadingKeys: Record<string, boolean>;
  summarizePaper: (params: {
    locator: PaperLocator;
    scope?: AssistantScope;
    question?: string;
    llm_provider?: string;
    model_override?: string;
  }) => Promise<PaperSummaryResult>;
  askPaper: (params: {
    locator: PaperLocator;
    question: string;
    scope?: AssistantScope;
    llm_provider?: string;
    model_override?: string;
  }) => Promise<PaperQaResult>;
  comparePapers: (params: {
    paper_uids: string[];
    aspects?: string[];
    scope?: AssistantScope;
    llm_provider?: string;
    model_override?: string;
  }) => Promise<PaperCompareResult>;
  startDiscoveryTask: (params: {
    mode: DiscoveryMode;
    paper_uids?: string[];
    node_ids?: string[];
    scope?: AssistantScope;
    question?: string;
    limit?: number;
  }) => Promise<AcademicAssistantTaskState>;
  startMediaAnalysisTask: (params: {
    paper_uids: string[];
    scope?: AssistantScope;
    force_reparse?: boolean;
    upsert_vectors?: boolean;
  }) => Promise<AcademicAssistantTaskState>;
  streamTask: (taskId: string, signal?: AbortSignal) => Promise<void>;
  loadTask: (taskId: string) => Promise<AcademicAssistantTaskState>;
  restoreActiveTasks: () => Promise<string[]>;
  listAnnotations: (params: {
    paper_uid?: string;
    resource_type?: string;
    resource_id?: string;
    target_kind?: string;
    status?: string;
  }) => Promise<ResourceAnnotation[]>;
  saveAnnotation: (body: ResourceAnnotationUpsert) => Promise<ResourceAnnotation>;
  clearTask: (taskId: string) => void;
}

export const useAcademicAssistantStore = create<AcademicAssistantStoreState>((set, get) => ({
  summaries: {},
  answers: {},
  comparisons: {},
  annotations: {},
  tasks: {},
  loadingKeys: {},

  summarizePaper: async (params) => {
    const key = `summary:${locatorKey(params.locator)}`;
    set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: true } }));
    try {
      const result = await summarizePaper(params);
      set((state) => ({
        summaries: { ...state.summaries, [locatorKey(params.locator)]: result },
        loadingKeys: { ...state.loadingKeys, [key]: false },
      }));
      return result;
    } catch (error) {
      set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: false } }));
      throw error;
    }
  },

  askPaper: async (params) => {
    const key = `qa:${locatorKey(params.locator)}:${params.question}`;
    set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: true } }));
    try {
      const result = await askPaper(params);
      set((state) => ({
        answers: { ...state.answers, [key]: result },
        loadingKeys: { ...state.loadingKeys, [key]: false },
      }));
      return result;
    } catch (error) {
      set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: false } }));
      throw error;
    }
  },

  comparePapers: async (params) => {
    const key = compareKey(params.paper_uids);
    set((state) => ({ loadingKeys: { ...state.loadingKeys, [`compare:${key}`]: true } }));
    try {
      const result = await compareAssistantPapers(params);
      set((state) => ({
        comparisons: { ...state.comparisons, [key]: result },
        loadingKeys: { ...state.loadingKeys, [`compare:${key}`]: false },
      }));
      return result;
    } catch (error) {
      set((state) => ({ loadingKeys: { ...state.loadingKeys, [`compare:${key}`]: false } }));
      throw error;
    }
  },

  startDiscoveryTask: async (params) => {
    const started = await startDiscovery(params);
    const nextTask: AcademicAssistantTaskState = {
      task_id: started.task_id,
      status: 'submitted',
      task_type: 'discovery',
      mode: params.mode,
      message: started.message,
      result: null,
      error_message: null,
      updated_at: Date.now(),
    };
    set((state) => ({
      tasks: { ...state.tasks, [started.task_id]: nextTask },
    }));
    writePersistedTaskIds([...readPersistedTaskIds(), started.task_id]);
    return nextTask;
  },

  startMediaAnalysisTask: async (params) => {
    const started = await startMediaAnalysis(params);
    const nextTask: AcademicAssistantTaskState = {
      task_id: started.task_id,
      status: 'submitted',
      task_type: 'media-analysis',
      mode: null,
      message: started.message,
      result: null,
      error_message: null,
      updated_at: Date.now(),
    };
    set((state) => ({
      tasks: { ...state.tasks, [started.task_id]: nextTask },
    }));
    writePersistedTaskIds([...readPersistedTaskIds(), started.task_id]);
    return nextTask;
  },

  streamTask: async (taskId, signal) => {
    try {
      for await (const evt of streamAcademicAssistantTask(taskId, signal)) {
        set((state) => {
          const prev = state.tasks[taskId];
          if (evt.event === 'progress') {
            return {
              tasks: {
                ...state.tasks,
                [taskId]: {
                  task_id: taskId,
                  status: 'running',
                  task_type: prev?.task_type || String((evt.data.type as string | undefined) || 'task'),
                  mode: prev?.mode || (evt.data.mode ? String(evt.data.mode) : null),
                  message: evt.data.stage ? String(evt.data.stage) : prev?.message || null,
                  result: prev?.result || null,
                  error_message: null,
                  updated_at: Date.now(),
                },
              },
            };
          }
          if (evt.event === 'done') {
            writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
            return {
              tasks: {
                ...state.tasks,
                [taskId]: {
                  ...(prev || {
                    task_id: taskId,
                    status: 'completed',
                    task_type: 'task',
                    updated_at: Date.now(),
                  }),
                  status: 'completed',
                  result: evt.data,
                  error_message: null,
                  updated_at: Date.now(),
                },
              },
            };
          }
          if (evt.event === 'error' || evt.event === 'cancelled' || evt.event === 'timeout') {
            writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
            const terminalStatus =
              evt.event === 'cancelled' ? 'cancelled' : evt.event === 'timeout' ? 'timeout' : 'error';
            return {
              tasks: {
                ...state.tasks,
                [taskId]: {
                  ...(prev || {
                    task_id: taskId,
                    task_type: 'task',
                    updated_at: Date.now(),
                  }),
                  status: terminalStatus,
                  error_message: evt.data.message ? String(evt.data.message) : prev?.error_message || null,
                  updated_at: Date.now(),
                },
              },
            };
          }
          return state;
        });
      }
    } catch (error) {
      // Try to recover the real backend status before deciding whether to evict the
      // persisted task id. A transient SSE drop should NOT permanently lose the task.
      let finalStatus: AcademicAssistantTaskState['status'] = 'error';
      try {
        const recovered = await get().loadTask(taskId);
        finalStatus = recovered.status;
      } catch {
        // loadTask failed (e.g. network down) — keep local error status
      }
      const isTerminal =
        finalStatus === 'completed' ||
        finalStatus === 'error' ||
        finalStatus === 'cancelled' ||
        finalStatus === 'timeout';
      if (isTerminal) {
        writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
      }
      set((state) => ({
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...(state.tasks[taskId] || {
              task_id: taskId,
              task_type: 'task',
              updated_at: Date.now(),
            }),
            status: finalStatus === 'completed' || finalStatus === 'cancelled' || finalStatus === 'timeout'
              ? finalStatus
              : 'error',
            error_message: isTerminal && finalStatus === 'error'
              ? (error instanceof Error ? error.message : String(error))
              : state.tasks[taskId]?.error_message || null,
            updated_at: Date.now(),
          },
        },
      }));
      throw error;
    }
  },

  loadTask: async (taskId) => {
    const info = await getAcademicAssistantTask(taskId);
    const next = mergeTaskInfo(get().tasks[taskId], info);
    set((state) => ({
      tasks: { ...state.tasks, [taskId]: next },
    }));
    if (next.status === 'completed' || next.status === 'error' || next.status === 'cancelled' || next.status === 'timeout') {
      writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
    }
    return next;
  },

  restoreActiveTasks: async () => {
    const restored: string[] = [];
    const persisted = readPersistedTaskIds();
    for (const taskId of persisted) {
      try {
        const task = await get().loadTask(taskId);
        if (task.status === 'submitted' || task.status === 'running') {
          restored.push(taskId);
        }
      } catch {
        writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
      }
    }
    return restored;
  },

  listAnnotations: async (params) => {
    const query = {
      paper_uid: params.paper_uid,
      resource_type: params.resource_type,
      resource_id: params.resource_id,
      target_kind: params.target_kind,
      status: params.status,
    };
    const data = await listAnnotations(query);
    const key = params.paper_uid || `${params.resource_type || ''}:${params.resource_id || ''}`;
    set((state) => ({
      annotations: { ...state.annotations, [key]: data.items || [] },
    }));
    return data.items || [];
  },

  saveAnnotation: async (body) => {
    const item = await upsertAnnotation(body);
    const key = item.paper_uid || `${item.resource_type}:${item.resource_id}`;
    set((state) => {
      const existing = state.annotations[key] || [];
      const next = [...existing.filter((row) => row.id !== item.id), item].sort((a, b) => b.id - a.id);
      return {
        annotations: { ...state.annotations, [key]: next },
      };
    });
    return item;
  },

  clearTask: (taskId) => {
    set((state) => {
      const nextTasks = { ...state.tasks };
      delete nextTasks[taskId];
      return { tasks: nextTasks };
    });
    writePersistedTaskIds(readPersistedTaskIds().filter((id) => id !== taskId));
  },
}));
