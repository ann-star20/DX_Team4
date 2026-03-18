# 마음이음 (Maumieum)

감정 기반 대화 + VOD 추천 음성 인터랙티브 TV 서비스 프로토타입

## 구조

- **frontend**: React + Vite + Zustand
- **backend**: Express + TypeScript
- **AI**: Mock 기반 (실제 AI 모델 교체 가능)

## 실행

```bash
# 루트에서
npm install
npm run dev
```

- 프론트엔드: http://localhost:5173
- 백엔드 API: http://localhost:3001

## API

| 엔드포인트 | 설명 |
|-----------|------|
| POST /api/chat | 대화 |
| POST /api/emotion | 감정 분석 |
| POST /api/recommend | VOD 추천 |
| GET /api/vod/catalog | VOD 목록 |
| POST /api/vod/analysis | VOD 종료 후 대화 |

## 환경변수 (backend/.env)

- `PORT`: API 서버 포트 (기본 3001)
- `USE_MOCK_AI`: `true` = Mock AI, `false` = Real AI (추후 연동)
