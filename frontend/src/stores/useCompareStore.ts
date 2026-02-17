import { create } from 'zustand';

interface CompareState {
  comparePreselectedPaperIds: string[];
  addComparePreselected: (id: string | string[]) => void;
  removeComparePreselected: (id: string) => void;
  clearComparePreselected: () => void;
}

export const useCompareStore = create<CompareState>((set) => ({
  comparePreselectedPaperIds: [],
  addComparePreselected: (id) =>
    set((state) => {
      const ids = Array.isArray(id) ? id : [id];
      const next = new Set([...state.comparePreselectedPaperIds, ...ids]);
      return { comparePreselectedPaperIds: [...next] };
    }),
  removeComparePreselected: (id) =>
    set((state) => ({
      comparePreselectedPaperIds: state.comparePreselectedPaperIds.filter((x) => x !== id),
    })),
  clearComparePreselected: () => set({ comparePreselectedPaperIds: [] }),
}));
