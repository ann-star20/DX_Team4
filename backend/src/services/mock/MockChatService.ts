import type {
  IAIChatService,
  ChatRequest,
  ChatResponse,
} from '../interfaces.js';

const GREETINGS = [
  '안녕하세요! 오늘 하루 어땠어요?',
  '반가워요! 기분이 어떠세요?',
  '안녕하세요! 편하게 이야기해 주세요.',
];

const RECOMMEND_PROMPTS = [
  '이런 분위기의 드라마 어떠세요?',
  '지금 기분에 딱 맞는 콘텐츠를 골라봤어요.',
  '이거 한번 보시는 건 어때요?',
];

const VOD_FOLLOWUP = [
  '방금 장면 인상 깊지 않았어요?',
  '어땠어요? 다음에 비슷한 거 추천해 드릴게요.',
  '재미있게 보셨나요?',
];

export class MockChatService implements IAIChatService {
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const { message, context } = request;
    const lower = message.toLowerCase().trim();

    // 추천 요청
    if (
      lower.includes('추천') ||
      lower.includes('볼거') ||
      lower.includes('뭐볼까')
    ) {
      return {
        reply: RECOMMEND_PROMPTS[Math.floor(Math.random() * RECOMMEND_PROMPTS.length)],
      };
    }

    // VOD 종료 후 대화
    if (context?.lastVodTitle) {
      return {
        reply: VOD_FOLLOWUP[Math.floor(Math.random() * VOD_FOLLOWUP.length)],
      };
    }

    // 인사
    if (
      lower.includes('안녕') ||
      lower.includes('하이') ||
      lower.includes('hello')
    ) {
      return {
        reply: GREETINGS[Math.floor(Math.random() * GREETINGS.length)],
      };
    }

    // 감정 키워드 기반 공감
    if (lower.includes('피곤') || lower.includes('힘들')) {
      return { reply: '오늘 많이 힘드셨나 봐요. 편하게 쉬실 수 있는 영화 추천해 드릴까요?' };
    }
    if (lower.includes('기쁘') || lower.includes('좋아')) {
      return { reply: '기분이 좋으시다니 저도 기뻐요! 이런 분위기 콘텐츠 어떠세요?' };
    }
    if (lower.includes('슬프') || lower.includes('우울')) {
      return { reply: '마음이 무겁으시군요. 위로가 되는 영화 같이 찾아볼까요?' };
    }

    // 기본 응답
    const defaults = [
      '그렇군요. 더 이야기해 주세요.',
      '이해했어요. 볼만한 거 추천해 드릴까요?',
      '편하게 말씀해 주세요.',
    ];
    return {
      reply: defaults[Math.floor(Math.random() * defaults.length)],
    };
  }
}
