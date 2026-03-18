import type {
  IAIVODAnalysisService,
  VODAnalysisRequest,
  VODAnalysisResponse,
} from '../interfaces.js';

const FOLLOWUP_QUESTIONS = [
  '방금 장면 인상 깊지 않았어요?',
  '어땠어요? 다음에 비슷한 거 추천해 드릴게요.',
  '재미있게 보셨나요?',
  '다음에 이런 분위기 영화 더 추천해 드릴까요?',
];

export class MockVODAnalysisService implements IAIVODAnalysisService {
  async analyze(request: VODAnalysisRequest): Promise<VODAnalysisResponse> {
    return {
      question:
        FOLLOWUP_QUESTIONS[
          Math.floor(Math.random() * FOLLOWUP_QUESTIONS.length)
        ],
      keywords: [request.genre, request.vodTitle].filter(Boolean),
    };
  }
}
