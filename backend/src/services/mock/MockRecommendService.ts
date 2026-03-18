import type {
  IAIRecommendService,
  RecommendRequest,
  RecommendResponse,
  VODItem,
} from '../interfaces.js';
import { VOD_CATALOG } from '../../data/vodCatalog.js';

export class MockRecommendService implements IAIRecommendService {
  async recommend(request: RecommendRequest): Promise<RecommendResponse> {
    const { emotion, watchHistory = [], lastVodId, limit = 6 } = request;

    let pool = [...VOD_CATALOG];

    // 시청 기록 제외
    if (watchHistory.length > 0) {
      pool = pool.filter((v) => !watchHistory.includes(v.id));
    }

    // 직전 VOD와 같은 장르 우선 (간단 룰)
    if (lastVodId) {
      const last = VOD_CATALOG.find((v) => v.id === lastVodId);
      if (last) {
        const sameGenre = pool.filter((v) => v.genre === last.genre);
        if (sameGenre.length >= limit) {
          pool = sameGenre;
        }
      }
    }

    // 감정 기반 간단 매핑 (룰 기반)
    if (emotion) {
      const emotionGenreMap: Record<string, string[]> = {
        joy: ['로맨스', '코미디'],
        sadness: ['멜로', '드라마'],
        tired: ['다큐', '예능'],
        calm: ['다큐', '영화'],
        excited: ['액션', '스릴러'],
        neutral: ['드라마', '영화'],
      };
      const preferred = emotionGenreMap[emotion] || ['드라마'];
      pool.sort((a, b) => {
        const aMatch = preferred.includes(a.genre) ? 1 : 0;
        const bMatch = preferred.includes(b.genre) ? 1 : 0;
        return bMatch - aMatch;
      });
    }

    // 셔플 후 limit개
    const shuffled = pool.sort(() => Math.random() - 0.5);
    const items: VODItem[] = shuffled.slice(0, limit);

    const messages = [
      '이런 분위기의 드라마 어떠세요?',
      '지금 기분에 맞는 콘텐츠예요.',
      '추천해 드릴게요!',
    ];

    return {
      items,
      message: messages[Math.floor(Math.random() * messages.length)],
    };
  }
}
