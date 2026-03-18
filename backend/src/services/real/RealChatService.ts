/**
 * Real AI Chat Service - 팀원이 AI 모델 제공 시 구현
 * IAIChatService 인터페이스 구현
 */
import type { IAIChatService, ChatRequest, ChatResponse } from '../interfaces.js';

export class RealChatService implements IAIChatService {
  async chat(_request: ChatRequest): Promise<ChatResponse> {
    // TODO: 실제 LLM API 연동
    throw new Error('RealChatService not implemented. Use USE_MOCK_AI=true');
  }
}
