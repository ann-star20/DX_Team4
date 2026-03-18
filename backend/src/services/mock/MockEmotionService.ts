import type {
  IAIEmotionService,
  EmotionRequest,
  EmotionResponse,
  EmotionType,
} from '../interfaces.js';

const EMOTION_KEYWORDS: Record<EmotionType, string[]> = {
  joy: ['기쁘', '좋아', '행복', '신나', '재미'],
  sadness: ['슬프', '우울', '힘들', '아쉬'],
  tired: ['피곤', '지쳐', '힘들', '졸려'],
  neutral: ['그냥', '보통', '괜찮'],
  excited: ['신나', '설레', '기대'],
  calm: ['편하', '차분', '여유'],
};

export class MockEmotionService implements IAIEmotionService {
  async analyze(request: EmotionRequest): Promise<EmotionResponse> {
    const { text } = request;
    const lower = text.toLowerCase();

    for (const [emotion, keywords] of Object.entries(EMOTION_KEYWORDS)) {
      if (keywords.some((kw) => lower.includes(kw))) {
        return {
          emotion: emotion as EmotionType,
          confidence: 0.8,
        };
      }
    }

    return {
      emotion: 'neutral',
      confidence: 0.5,
    };
  }
}
