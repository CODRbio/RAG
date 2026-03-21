import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ResourceRef } from '../types';

export interface AnalysisPoolItem {
  paper_uid: string;
  title: string;
  subtitle?: string | null;
  paper_id?: string | null;
  collection?: string | null;
  library_id?: number | null;
  resource_ref?: ResourceRef | null;
}

interface AnalysisPoolState {
  items: AnalysisPoolItem[];
  addItems: (items: AnalysisPoolItem | AnalysisPoolItem[]) => void;
  removeItem: (paperUid: string) => void;
  clear: () => void;
  has: (paperUid: string) => boolean;
}

function normalizeItem(item: AnalysisPoolItem): AnalysisPoolItem | null {
  const paperUid = String(item.paper_uid || '').trim();
  if (!paperUid) return null;
  return {
    paper_uid: paperUid,
    title: String(item.title || paperUid).trim() || paperUid,
    subtitle: item.subtitle || null,
    paper_id: item.paper_id || null,
    collection: item.collection || null,
    library_id: item.library_id ?? null,
    resource_ref: item.resource_ref || null,
  };
}

export const useAnalysisPoolStore = create<AnalysisPoolState>()(
  persist(
    (set, get) => ({
      items: [],

      addItems: (value) =>
        set((state) => {
          const incoming = (Array.isArray(value) ? value : [value])
            .map(normalizeItem)
            .filter((item): item is AnalysisPoolItem => item != null);
          if (incoming.length === 0) return {};
          const next = new Map(state.items.map((item) => [item.paper_uid, item] as const));
          incoming.forEach((item) => {
            next.set(item.paper_uid, { ...(next.get(item.paper_uid) || {}), ...item });
          });
          return { items: Array.from(next.values()) };
        }),

      removeItem: (paperUid) =>
        set((state) => ({
          items: state.items.filter((item) => item.paper_uid !== paperUid),
        })),

      clear: () => set({ items: [] }),

      has: (paperUid) => get().items.some((item) => item.paper_uid === paperUid),
    }),
    {
      name: 'analysis-pool-storage',
      partialize: (state) => ({ items: state.items }),
    },
  ),
);
