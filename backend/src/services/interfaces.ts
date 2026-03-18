/**
 * AI 서비스 인터페이스 - 모든 AI 기능은 이 레이어 뒤로 숨김
 * 팀원이 Real AI 제공 시 이 인터페이스 구현체만 교체하면 됨
 */

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  history?: ChatMessage[];
  context?: {
    emotion?: string;
    lastVodTitle?: string;
  };
}

export interface ChatResponse {
  reply: string;
}

export interface IAIChatService {
  chat(request: ChatRequest): Promise<ChatResponse>;
}

// ---

export type EmotionType = 'joy' | 'sadness' | 'tired' | 'neutral' | 'excited' | 'calm';

export interface EmotionRequest {
  text: string;
  voiceTone?: string; // 추후 음성 톤 분석용
}

export interface EmotionResponse {
  emotion: EmotionType;
  confidence: number;
}

export interface IAIEmotionService {
  analyze(request: EmotionRequest): Promise<EmotionResponse>;
}

// ---

export interface VODItem {
  id: string;
  title: string;
  genre: string;
  thumbnail?: string;
  description?: string;
  popularity?: number;
  featured?: boolean;
  rating?: string;
  year?: number;
}

export interface RecommendRequest {
  emotion?: EmotionType;
  watchHistory?: string[];
  lastVodId?: string;
  limit?: number;
}

export interface RecommendResponse {
  items: VODItem[];
  message?: string;
}

export interface IAIRecommendService {
  recommend(request: RecommendRequest): Promise<RecommendResponse>;
}

// ---

export interface VODAnalysisRequest {
  vodId: string;
  vodTitle: string;
  genre: string;
}

export interface VODAnalysisResponse {
  question: string;
  keywords: string[];
}

export interface IAIVODAnalysisService {
  analyze(request: VODAnalysisRequest): Promise<VODAnalysisResponse>;
}
