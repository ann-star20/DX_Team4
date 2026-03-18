# AI API 스펙 (팀원 연동용)

실제 AI 모델 연동 시 아래 스펙에 맞춰 구현하면 됩니다.

## POST /api/chat

**Request**
```json
{
  "message": "string",
  "history": [{ "role": "user"|"assistant", "content": "string" }],
  "context": { "emotion": "string", "lastVodTitle": "string" }
}
```

**Response**
```json
{
  "reply": "string"
}
```

---

## POST /api/emotion

**Request**
```json
{
  "text": "string",
  "voiceTone": "string (optional)"
}
```

**Response**
```json
{
  "emotion": "joy"|"sadness"|"tired"|"neutral"|"excited"|"calm",
  "confidence": 0.0-1.0
}
```

---

## POST /api/recommend

**Request**
```json
{
  "emotion": "string (optional)",
  "watchHistory": ["vodId1", "vodId2"],
  "lastVodId": "string (optional)",
  "limit": 6
}
```

**Response**
```json
{
  "items": [{ "id": "string", "title": "string", "genre": "string" }],
  "message": "string (optional)"
}
```

---

## POST /api/vod/analysis

**Request**
```json
{
  "vodId": "string",
  "vodTitle": "string",
  "genre": "string"
}
```

**Response**
```json
{
  "question": "string",
  "keywords": ["string"]
}
```
