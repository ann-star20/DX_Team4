# 마음이음 - 웹 패키징 개발 작업 순서

> **핵심 원칙**: AI 의존성 분리 | AI 없이 동작 | API 레이어 뒤로 AI 숨김

---

## Phase 1: 프로젝트 기반 구축

### 1.1 프로젝트 구조 설계
- [x] 모노레포 또는 분리 구조 결정 (frontend / backend / shared)
- [x] 폴더 구조 생성
- [x] TypeScript, ESLint, Prettier 설정

### 1.2 프론트엔드 초기 설정
- [x] React + Vite 프로젝트 생성
- [x] Zustand 상태관리 설정
- [x] 라우팅 구조 (VOD 탐색 / 대화 모드 / 추천 상태)

### 1.3 백엔드 초기 설정
- [x] NestJS 또는 Express + TypeScript 프로젝트 생성
- [x] API 서버 기본 구조
- [x] 환경변수 설정 (.env)

---

## Phase 2: API 레이어 설계 (AI 분리 핵심)

### 2.1 AI 서비스 인터페이스 정의
- [x] `IAIChatService` - LLM 대화 인터페이스
- [x] `IAIEmotionService` - 감정 분석 인터페이스
- [x] `IAIRecommendService` - 추천 모델 인터페이스
- [x] `IAIVODAnalysisService` - VOD 분석 인터페이스 (선택)

### 2.2 Mock 구현체 작성
- [x] `MockChatService` - 고정/랜덤 응답 대화
- [x] `MockEmotionService` - 미리 정의된 감정 반환
- [x] `MockRecommendService` - 룰 기반/랜덤 VOD 추천
- [x] `MockVODAnalysisService` - VOD 종료 후 대화용 (선택)

### 2.3 API 엔드포인트 설계
- [x] `POST /api/chat` - 대화 요청 (내부에서 Mock/Real AI 호출)
- [x] `POST /api/emotion` - 감정 분석 (내부에서 Mock/Real 호출)
- [x] `POST /api/recommend` - 추천 요청 (내부에서 Mock/Real 호출)
- [x] `POST /api/vod/analysis` - VOD 종료 후 대화 (선택)
- [x] 환경변수로 Mock/Real 전환 가능하게 (`USE_MOCK_AI=true`)

---

## Phase 3: 핵심 비즈니스 로직 (AI 무관)

### 3.1 상태 관리 설계
- [ ] Zustand 스토어: `useAppStore` (전역 상태)
- [ ] 상태: `idle` | `vod_browse` | `conversation` | `recommendation`
- [ ] 사용자 시청 기록 저장 구조 (로컬/메모리)
- [ ] 대화 맥락(history) 저장 구조

### 3.2 VOD 데이터 구조
- [ ] Mock VOD 목록 데이터 (제목, 장르, 메타데이터)
- [ ] 시청 기록 추적 로직
- [ ] 추천 결과 포맷 정의

### 3.3 서비스 루프 오케스트레이션
- [ ] `대화 → 감정 → 추천 → 시청 → VOD 기반 대화 → 재추천` 흐름 구현
- [ ] API 레이어만 호출, AI 직접 의존 없음

---

## Phase 4: UI 구현

### 4.1 기본 화면 (VOD 탐색)
- [ ] 넷플릭스 스타일 VOD 리스트 UI
- [ ] VOD 카드 컴포넌트
- [ ] 장르/카테고리 필터 (선택)

### 4.2 대화 모드 UI
- [ ] 트리거 시 화면 dim
- [ ] 우측 하단 캐릭터/아바타 영역
- [ ] 대화 버블 또는 채팅 UI
- [ ] 상태 전환 애니메이션

### 4.3 추천 상태 UI
- [ ] 추천 VOD 리스트 표시
- [ ] "이런 분위기의 드라마 어떠세요?" 등 문구 연동

### 4.4 VOD 시청 화면
- [ ] 간단한 플레이어 (또는 시뮬레이션)
- [ ] 종료 시 → VOD 기반 대화 트리거

---

## Phase 5: 음성 인터페이스 (Mock 가능)

### 5.1 STT 연동
- [ ] Web Speech API 또는 외부 STT 서비스 연동
- [ ] STT 실패/미지원 시 **텍스트 입력 폴백** 구현 (AI 없이 테스트용)

### 5.2 TTS 연동
- [ ] Web Speech API TTS 또는 외부 TTS
- [ ] TTS 실패 시 **텍스트만 표시** 폴백

### 5.3 Wake Word "마음이음"
- [ ] 초기: 버튼 클릭으로 대화 모드 진입 (Wake word 대체)
- [ ] 추후: Porcupine 등 Wake word 엔진 연동 (선택)

---

## Phase 6: 셋톱박스 환경 시뮬레이션

### 6.1 리모컨/키보드 네비게이션
- [ ] 방향키로 VOD 탐색
- [ ] Enter로 선택, Back으로 이전
- [ ] TV 리모컨 UX 가정한 키 매핑

### 6.2 해상도/레이아웃
- [ ] 1920x1080 또는 TV 해상도 기준 레이아웃
- [ ] 대형 폰트, 터치/포인터 친화적

---

## Phase 7: AI 교체 준비 (추후 팀원 연동용)

### 7.1 DI/Factory 패턴
- [ ] `AIServiceFactory` - 환경에 따라 Mock/Real 인스턴스 반환
- [ ] `RealChatService`, `RealEmotionService` 등 빈 껍데기 클래스 준비

### 7.2 API 스펙 문서화
- [ ] 각 AI API 입력/출력 스펙 문서 작성
- [ ] 팀원이 Real AI 구현 시 참고할 수 있도록

### 7.3 환경 전환
- [ ] `USE_MOCK_AI=false` 시 Real AI 서비스 주입
- [ ] Real AI 서버 URL 등 설정 분리

---

## 체크리스트 요약

| Phase | 내용 | AI 의존 |
|-------|------|----------|
| 1 | 프로젝트 기반 | 없음 |
| 2 | API 레이어 + Mock | Mock만 사용 |
| 3 | 비즈니스 로직 | API 호출만 |
| 4 | UI | 없음 |
| 5 | 음성 (텍스트 폴백) | 없음 |
| 6 | 셋톱박스 UX | 없음 |
| 7 | AI 교체 준비 | 교체 시점 |

---

## 권장 작업 순서 (실제 코딩)

1. **Phase 1** → 프로젝트 뼈대
2. **Phase 2** → API + Mock (AI 없이 전체 플로우 검증 가능하게)
3. **Phase 3** → 상태/루프 (API 호출 연결)
4. **Phase 4** → UI (화면부터 보이게)
5. **Phase 5** → 음성 (텍스트 폴백 먼저)
6. **Phase 6** → 셋톱박스 UX
7. **Phase 7** → AI 교체용 정리

> 💡 **Tip**: Phase 2 Mock을 먼저 완성하면, 이후 모든 개발을 AI 없이 진행할 수 있다.
