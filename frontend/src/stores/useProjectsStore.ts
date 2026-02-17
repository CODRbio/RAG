import { create } from 'zustand';
import type { Project } from '../types';
import {
  listProjects as apiListProjects,
  archiveProject as apiArchiveProject,
  unarchiveProject as apiUnarchiveProject,
  deleteProject as apiDeleteProject,
} from '../api/projects';

interface ProjectsState {
  projects: Project[];
  isLoading: boolean;

  fetchProjects: (includeArchived?: boolean) => Promise<void>;
  toggleArchive: (canvasId: string, isArchived: boolean) => Promise<void>;
  deleteProject: (canvasId: string) => Promise<void>;
  clearProjects: () => void;
}

export const useProjectsStore = create<ProjectsState>((set, get) => ({
  projects: [],
  isLoading: false,

  fetchProjects: async (includeArchived = true) => {
    set({ isLoading: true });
    try {
      const projects = await apiListProjects(includeArchived);
      set({ projects, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },

  toggleArchive: async (canvasId, isArchived) => {
    try {
      if (isArchived) {
        await apiUnarchiveProject(canvasId);
      } else {
        await apiArchiveProject(canvasId);
      }
      // 更新本地状态
      set((state) => ({
        projects: state.projects.map((p) =>
          p.id === canvasId ? { ...p, archived: !isArchived } : p
        ),
      }));
    } catch {
      // 重新获取
      get().fetchProjects();
    }
  },

  deleteProject: async (canvasId) => {
    try {
      await apiDeleteProject(canvasId);
      set((state) => ({
        projects: state.projects.filter((p) => p.id !== canvasId),
      }));
    } catch {
      get().fetchProjects();
    }
  },

  clearProjects: () => set({ projects: [] }),
}));
