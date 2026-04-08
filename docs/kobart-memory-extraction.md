# KoBART 기반 장기기억 추출 실험 가이드

기존 `/extract-memory`(Gemini 기반)는 그대로 두고, KoBART 실험용으로 `/extract-memory-kobart`를 추가했습니다.

## 1) 의존성 설치

```bash
cd /Users/user/Documents/maum
.venv/bin/pip install -r requirements.txt
```

## 2) 실행

```bash
.venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 3) KoBART 추출 API 호출 예시

```bash
curl -X POST "http://127.0.0.1:8000/extract-memory-kobart" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "senior-demo",
    "limit": 50,
    "save_to_db": true,
    "mark_extracted_after_save": true,
    "store_comparison_stats": true,
    "compare_with_llm": true,
    "model_id": "gogamza/kobart-summarization"
  }'
```

응답 포맷은 기존 LLM 추출과 동일합니다.

```json
{
  "memory": [
    {
      "memory_date": "YYYY-MM-DD",
      "content": "string",
      "category": "Family|Health|Routine|Preference|Event|Emotion",
      "emotion": "string",
      "importance_score": 1,
      "entity": ["string"]
    }
  ]
}
```

## 4) metadata 저장 구조

`long_term_memory.metadata`에 아래 키가 들어갑니다.

- `source`: `kobart`
- `category`, `emotion`, `importance_score`, `entity`
- `intent`: 규칙 기반 의도 레이블
- `keywords`: 대화 키워드
- `sentence_importance`: 문장별 중요도
- `comparison.keyword_overlap_jaccard`: LLM 추출 결과와 키워드 중첩도

## 5) LLM 비교 지표 해석

- `avg_keyword_overlap_jaccard`: KoBART 요약과 LLM 추출 키워드의 평균 겹침 비율
- `avg_importance_gap`: KoBART 중요도와 LLM 평균 중요도 차이

값이 높다고 무조건 성능 우수는 아니므로, 실제 운영 판단은 수동 샘플 검토와 함께 진행하세요.

## 6) daily_sentiment_stats 저장

`store_comparison_stats=true`이면 사용자별로 당일(`report_date=today`) 레코드를 갱신합니다.

- `emotion_distribution`:
  - `source: kobart_vs_llm`
  - `kobart_memory_count`, `llm_memory_count`
  - `avg_keyword_overlap_jaccard`, `avg_importance_gap`
  - `model_id`
- `daily_avg_incongruence`: `avg_importance_gap`
- `total_utterance`: 이번 배치 대화 수
