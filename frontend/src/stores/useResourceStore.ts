import { create } from 'zustand';
import {
  createResourceNote,
  createResourceTag,
  deleteResourceNote,
  deleteResourceTag,
  getResourceState,
  listResourceNotes,
  listResourceTags,
  patchResourceState,
  updateResourceNote,
} from '../api/resources';
import type {
  ResourceNote,
  ResourceNoteCreate,
  ResourceNoteUpdate,
  ResourceRef,
  ResourceTag,
  ResourceTagUpsert,
  ResourceUserState,
  ResourceUserStateUpsert,
} from '../types';

export function resourceKey(ref: ResourceRef): string {
  return `${ref.resource_type}:${ref.resource_id}`;
}

interface ResourceStoreState {
  states: Record<string, ResourceUserState>;
  tags: Record<string, ResourceTag[]>;
  notes: Record<string, ResourceNote[]>;
  loadingKeys: Record<string, boolean>;
  loadState: (ref: ResourceRef) => Promise<ResourceUserState>;
  upsertState: (body: ResourceUserStateUpsert) => Promise<ResourceUserState>;
  loadTags: (ref: ResourceRef) => Promise<ResourceTag[]>;
  addTag: (body: ResourceTagUpsert) => Promise<ResourceTag>;
  removeTag: (body: ResourceTagUpsert) => Promise<boolean>;
  loadNotes: (ref: ResourceRef) => Promise<ResourceNote[]>;
  createNote: (body: ResourceNoteCreate) => Promise<ResourceNote>;
  updateNote: (ref: ResourceRef, noteId: number, body: ResourceNoteUpdate) => Promise<ResourceNote>;
  removeNote: (ref: ResourceRef, noteId: number) => Promise<boolean>;
  hydrateResource: (ref: ResourceRef) => Promise<void>;
  clearResource: (ref: ResourceRef) => void;
}

function sortNotes(items: ResourceNote[]): ResourceNote[] {
  return [...items].sort((a, b) => {
    const left = new Date(a.updated_at || a.created_at || 0).getTime();
    const right = new Date(b.updated_at || b.created_at || 0).getTime();
    return right - left;
  });
}

function sortTags(items: ResourceTag[]): ResourceTag[] {
  return [...items].sort((a, b) => a.tag.localeCompare(b.tag));
}

export const useResourceStore = create<ResourceStoreState>((set, get) => ({
  states: {},
  tags: {},
  notes: {},
  loadingKeys: {},

  loadState: async (ref) => {
    const key = resourceKey(ref);
    set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: true } }));
    try {
      const data = await getResourceState(ref);
      set((state) => ({
        states: { ...state.states, [key]: data },
        loadingKeys: { ...state.loadingKeys, [key]: false },
      }));
      return data;
    } catch (error) {
      set((state) => ({ loadingKeys: { ...state.loadingKeys, [key]: false } }));
      throw error;
    }
  },

  upsertState: async (body) => {
    const key = resourceKey(body);
    const data = await patchResourceState(body);
    set((state) => ({
      states: { ...state.states, [key]: data },
    }));
    return data;
  },

  loadTags: async (ref) => {
    const key = resourceKey(ref);
    const data = await listResourceTags(ref);
    const items = sortTags(data.items || []);
    set((state) => ({
      tags: { ...state.tags, [key]: items },
    }));
    return items;
  },

  addTag: async (body) => {
    const key = resourceKey(body);
    const item = await createResourceTag(body);
    set((state) => ({
      tags: {
        ...state.tags,
        [key]: sortTags([...(state.tags[key] || []).filter((existing) => existing.id !== item.id), item]),
      },
    }));
    return item;
  },

  removeTag: async (body) => {
    const key = resourceKey(body);
    const res = await deleteResourceTag(body);
    if (res.deleted) {
      set((state) => ({
        tags: {
          ...state.tags,
          [key]: (state.tags[key] || []).filter((item) => item.normalized_tag !== body.tag.trim().replace(/\s+/g, ' ').toLowerCase()),
        },
      }));
    }
    return res.deleted;
  },

  loadNotes: async (ref) => {
    const key = resourceKey(ref);
    const data = await listResourceNotes(ref);
    const items = sortNotes(data.items || []);
    set((state) => ({
      notes: { ...state.notes, [key]: items },
    }));
    return items;
  },

  createNote: async (body) => {
    const key = resourceKey(body);
    const item = await createResourceNote(body);
    set((state) => ({
      notes: {
        ...state.notes,
        [key]: sortNotes([item, ...(state.notes[key] || [])]),
      },
    }));
    return item;
  },

  updateNote: async (ref, noteId, body) => {
    const key = resourceKey(ref);
    const item = await updateResourceNote(noteId, body);
    set((state) => ({
      notes: {
        ...state.notes,
        [key]: sortNotes((state.notes[key] || []).map((existing) => (existing.id === noteId ? item : existing))),
      },
    }));
    return item;
  },

  removeNote: async (ref, noteId) => {
    const key = resourceKey(ref);
    const res = await deleteResourceNote(noteId);
    if (res.deleted) {
      set((state) => ({
        notes: {
          ...state.notes,
          [key]: (state.notes[key] || []).filter((item) => item.id !== noteId),
        },
      }));
    }
    return res.deleted;
  },

  hydrateResource: async (ref) => {
    await Promise.all([
      get().loadState(ref),
      get().loadTags(ref),
      get().loadNotes(ref),
    ]);
  },

  clearResource: (ref) => {
    const key = resourceKey(ref);
    set((state) => {
      const nextStates = { ...state.states };
      const nextTags = { ...state.tags };
      const nextNotes = { ...state.notes };
      const nextLoading = { ...state.loadingKeys };
      delete nextStates[key];
      delete nextTags[key];
      delete nextNotes[key];
      delete nextLoading[key];
      return {
        states: nextStates,
        tags: nextTags,
        notes: nextNotes,
        loadingKeys: nextLoading,
      };
    });
  },
}));
