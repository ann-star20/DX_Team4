import { create } from 'zustand';

export type AppState = 'idle' | 'vod_browse' | 'conversation' | 'recommendation';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface VODItem {
  id: string;
  title: string;
  genre: string;
  description?: string;
  popularity?: number;
  featured?: boolean;
  rating?: string;
  year?: number;
}

interface AppStore {
  state: AppState;
  setState: (s: AppState) => void;

  chatHistory: ChatMessage[];
  addChatMessage: (msg: ChatMessage) => void;
  clearChatHistory: () => void;

  watchHistory: string[];
  addWatchHistory: (vodId: string) => void;

  lastVodId: string | null;
  setLastVodId: (id: string | null) => void;

  currentEmotion: string | null;
  setCurrentEmotion: (e: string | null) => void;

  recommendedItems: VODItem[];
  setRecommendedItems: (items: VODItem[]) => void;
  clearRecommendedItems: () => void;

  vodCatalog: VODItem[];
  setVodCatalog: (items: VODItem[]) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  state: 'vod_browse',
  setState: (state) => set({ state }),

  chatHistory: [],
  addChatMessage: (msg) =>
    set((s) => ({ chatHistory: [...s.chatHistory, msg] })),
  clearChatHistory: () => set({ chatHistory: [] }),

  watchHistory: [],
  addWatchHistory: (vodId) =>
    set((s) => ({
      watchHistory: s.watchHistory.includes(vodId)
        ? s.watchHistory
        : [...s.watchHistory, vodId],
    })),

  lastVodId: null,
  setLastVodId: (lastVodId) => set({ lastVodId }),

  currentEmotion: null,
  setCurrentEmotion: (currentEmotion) => set({ currentEmotion }),

  recommendedItems: [],
  setRecommendedItems: (recommendedItems) => set({ recommendedItems }),
  clearRecommendedItems: () => set({ recommendedItems: [] }),

  vodCatalog: [],
  setVodCatalog: (vodCatalog) => set({ vodCatalog }),
}));
