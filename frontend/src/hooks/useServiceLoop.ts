import { useCallback } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';

/**
 * 서비스 루프: 대화 → 감정 → 추천 → 시청 → VOD 기반 대화 → 재추천
 * AI API만 호출, 프론트엔드는 AI 직접 의존 없음
 */
export function useServiceLoop() {
  const {
    addChatMessage,
    addWatchHistory,
    setLastVodId,
    setCurrentEmotion,
    setRecommendedItems,
    setState,
    watchHistory,
    lastVodId,
    currentEmotion,
    chatHistory,
  } = useAppStore();

  const sendMessage = useCallback(
    async (text: string) => {
      addChatMessage({ role: 'user', content: text });
      const historyWithNew = [
        ...chatHistory,
        { role: 'user' as const, content: text },
      ];

      const context =
        lastVodId && chatHistory.length === 0
          ? { lastVodTitle: '방금 본 콘텐츠' }
          : undefined;

      const { reply } = await api.chat({
        message: text,
        history: historyWithNew,
        context,
      });
      addChatMessage({ role: 'assistant', content: reply });
      return reply;
    },
    [addChatMessage, chatHistory, lastVodId]
  );

  const analyzeEmotion = useCallback(async (text: string) => {
    const { emotion } = await api.emotion({ text });
    setCurrentEmotion(emotion);
    return emotion;
  }, [setCurrentEmotion]);

  const fetchRecommend = useCallback(async () => {
    const { items, message } = await api.recommend({
      emotion: currentEmotion || undefined,
      watchHistory,
      lastVodId: lastVodId || undefined,
      limit: 6,
    });
    setRecommendedItems(items);
    setState('recommendation');
    return { items, message };
  }, [
    currentEmotion,
    watchHistory,
    lastVodId,
    setRecommendedItems,
    setState,
  ]);

  const onVodEnd = useCallback(
    async (vodId: string, vodTitle: string, genre: string) => {
      addWatchHistory(vodId);
      setLastVodId(vodId);

      const { question } = await api.vodAnalysis({ vodId, vodTitle, genre });
      addChatMessage({ role: 'assistant', content: question });
      setState('conversation');
    },
    [addWatchHistory, setLastVodId, addChatMessage, setState]
  );

  const requestRecommend = useCallback(async () => {
    setState('recommendation');
    return fetchRecommend();
  }, [setState, fetchRecommend]);

  return {
    sendMessage,
    analyzeEmotion,
    fetchRecommend,
    onVodEnd,
    requestRecommend,
  };
}
