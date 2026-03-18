import type {
  IAIChatService,
  IAIEmotionService,
  IAIRecommendService,
  IAIVODAnalysisService,
} from './interfaces.js';
import { MockChatService } from './mock/MockChatService.js';
import { MockEmotionService } from './mock/MockEmotionService.js';
import { MockRecommendService } from './mock/MockRecommendService.js';
import { MockVODAnalysisService } from './mock/MockVODAnalysisService.js';

const useMock = process.env.USE_MOCK_AI !== 'false';

let chatInstance: IAIChatService | null = null;
let emotionInstance: IAIEmotionService | null = null;
let recommendInstance: IAIRecommendService | null = null;
let vodAnalysisInstance: IAIVODAnalysisService | null = null;

export function getChatService(): IAIChatService {
  if (!chatInstance) {
    chatInstance = useMock ? new MockChatService() : new MockChatService(); // TODO: RealChatService
  }
  return chatInstance;
}

export function getEmotionService(): IAIEmotionService {
  if (!emotionInstance) {
    emotionInstance = useMock ? new MockEmotionService() : new MockEmotionService(); // TODO: RealEmotionService
  }
  return emotionInstance;
}

export function getRecommendService(): IAIRecommendService {
  if (!recommendInstance) {
    recommendInstance = useMock ? new MockRecommendService() : new MockRecommendService(); // TODO: RealRecommendService
  }
  return recommendInstance;
}

export function getVODAnalysisService(): IAIVODAnalysisService {
  if (!vodAnalysisInstance) {
    vodAnalysisInstance = useMock ? new MockVODAnalysisService() : new MockVODAnalysisService(); // TODO: RealVODAnalysisService
  }
  return vodAnalysisInstance;
}
