"""
KoBART 기반 장기기억 추출 파이프라인 (기존 LLM 파이프라인과 병행 운영용).

목표:
- short_term_history 미추출 대화에서 로컬/오픈모델 기반으로 요약, 의도, 키워드, 중요도 추출
- long_term_memory.metadata에 구조화 결과 적재
- 필요 시 기존 LLM 추출과 품질 비교 지표 생성
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.database import (
    check_duplicate_memory,
    get_extracted_conversations,
    get_unextracted_conversations,
    mark_as_extracted,
    save_long_term_memory,
    upsert_daily_sentiment_stat,
)
from app.memory_agent import EXTRACTION_SYSTEM_PROMPT
from app.schemas import ExtractedMemoryItem, ExtractionResult

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "오늘", "어제", "내일", "정말", "진짜", "그냥", "너무", "조금", "많이", "이제", "지금",
    "저는", "나는", "제가", "합니다", "했어요", "그래서", "그리고", "하지만", "근데",
    "어르신", "말씀", "있어요", "없어요", "같아요", "것", "수", "때", "좀", "잘", "안",
}

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("건강상태_공유", ("아프", "병원", "약", "혈압", "당뇨", "통증", "수술", "치료")),
    ("가족_회상", ("아들", "딸", "손주", "남편", "아내", "가족", "형제", "자매")),
    ("일상_루틴", ("아침", "점심", "저녁", "산책", "운동", "잠", "수면", "식사", "복용")),
    ("선호_표현", ("좋아", "싫어", "선호", "즐겨", "자주", "취미")),
    ("감정_표현", ("외롭", "슬프", "불안", "기쁘", "행복", "화나", "무섭", "걱정")),
    ("요청_질문", ("해줘", "도와", "알려", "어떻게", "왜", "뭐", "?", "문의")),
]


@dataclass
class KobartMemory:
    memory_date: str
    content: str
    category: str
    emotion: str
    importance_score: int
    entity: list[str]
    intent: str
    keywords: list[str]
    sentence_scores: list[dict[str, Any]]
    source: str = "kobart"

    def to_extracted_item(self) -> ExtractedMemoryItem:
        return ExtractedMemoryItem(
            memory_date=self.memory_date,
            content=self.content,
            category=self.category,
            emotion=self.emotion,
            importance_score=self.importance_score,
            entity=self.entity,
        )


def _build_conversation_lines(messages: list[dict]) -> list[str]:
    lines: list[str] = []
    for msg in messages:
        ts = msg.get("chat_time", "")
        if isinstance(ts, str) and "T" in ts:
            ts = ts[:16].replace("T", " ")
        lines.append(f"[{ts}] {msg.get('role', 'user')}: {msg.get('content', '')}")
    return lines


def _chunks_from_lines(lines: list[str], max_chars: int = 900) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    current = 0
    for ln in lines:
        if current + len(ln) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = [ln]
            current = len(ln)
        else:
            buf.append(ln)
            current += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _extract_keywords(text: str, top_k: int = 8) -> list[str]:
    tokens = re.findall(r"[가-힣A-Za-z]{2,}", text)
    kept = [t for t in tokens if t not in _STOPWORDS]
    freq = Counter(kept)
    return [w for w, _ in freq.most_common(top_k)]


def _infer_intent(text: str) -> str:
    t = text.lower()
    for intent, markers in _INTENT_RULES:
        if any(m in t for m in markers):
            return intent
    return "일반_대화"


def _infer_emotion(text: str) -> str:
    t = text
    if any(k in t for k in ("외롭", "슬프", "우울", "눈물", "힘들")):
        return "슬픔"
    if any(k in t for k in ("걱정", "불안", "무섭")):
        return "공포"
    if any(k in t for k in ("화나", "짜증", "분노")):
        return "분노"
    if any(k in t for k in ("기쁘", "행복", "즐겁", "좋았")):
        return "기쁨"
    return "중립"


def _infer_category(intent: str) -> str:
    if intent == "가족_회상":
        return "Family"
    if intent == "건강상태_공유":
        return "Health"
    if intent == "일상_루틴":
        return "Routine"
    if intent == "선호_표현":
        return "Preference"
    if intent == "감정_표현":
        return "Emotion"
    return "Event"


def _sentence_scores(text: str) -> list[dict[str, Any]]:
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    out: list[dict[str, Any]] = []
    for s in sentences:
        score = 3
        if any(k in s for k in ("병원", "수술", "통증", "약", "응급")):
            score += 3
        if any(k in s for k in ("아들", "딸", "손주", "남편", "아내", "가족")):
            score += 2
        if any(k in s for k in ("외롭", "슬프", "불안", "행복", "기쁘", "화나")):
            score += 2
        if len(s) >= 24:
            score += 1
        out.append({"sentence": s, "importance": max(1, min(10, score))})
    return out


def _importance_from_scores(scored: list[dict[str, Any]]) -> int:
    if not scored:
        return 3
    peak = max(int(x.get("importance", 3)) for x in scored)
    avg = sum(int(x.get("importance", 3)) for x in scored) / len(scored)
    blended = round((peak * 0.6) + (avg * 0.4))
    return max(1, min(10, int(blended)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_text_tokens(text: str) -> set[str]:
    return set(_extract_keywords(text, top_k=30))


class KobartSummarizer:
    """KoBART 요약기. 모델 로드는 최초 1회만 수행."""

    def __init__(self, model_id: str = "gogamza/kobart-summarization") -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        # 샌드박스/권한 이슈 회피: 프로젝트 내부 캐시 디렉터리 사용
        hf_home = Path(__file__).resolve().parent.parent / ".hf_cache"
        hf_home.mkdir(parents=True, exist_ok=True)
        (hf_home / "transformers").mkdir(parents=True, exist_ok=True)
        (hf_home / "hub").mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "KoBART 요약을 위해 transformers가 필요합니다. "
                "pip install transformers sentencepiece"
            ) from e
        logger.info("KoBART 요약 모델 로드: %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(hf_home),
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            cache_dir=str(hf_home),
        )

    def summarize(self, text: str, max_length: int = 96, min_length: int = 20) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        output_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            num_beams=4,
            do_sample=False,
            min_length=min_length,
            max_length=max_length,
        )
        summaries = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if not summaries:
            return text[:220]
        return summaries[0].strip() or text[:220]


async def _llm_preview_extract(messages: list[dict]) -> list[ExtractedMemoryItem]:
    """기존 LLM 프롬프트로 저장 없이 비교용 미리보기 추출."""
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.1,
        max_output_tokens=2048,
        thinking_budget=0,
    )
    conv_text = "\n".join(_build_conversation_lines(messages))
    prompt_text = f"{EXTRACTION_SYSTEM_PROMPT}\n\n# Conversation Logs\n{conv_text}"
    try:
        response = await llm.ainvoke(prompt_text)
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = ExtractionResult.model_validate(json.loads(raw))
        return parsed.memory
    except Exception:
        logger.exception("LLM 비교 추출 실패")
        return []


async def extract_memories_kobart(
    user_id: str | None = None,
    *,
    limit: int = 50,
    save_to_db: bool = True,
    mark_extracted_after_save: bool = True,
    compare_with_llm: bool = True,
    store_comparison_stats: bool = True,
    source_mode: str = "unextracted",
    model_id: str = "gogamza/kobart-summarization",
) -> dict:
    """
    KoBART 기반 장기기억 추출.

    Returns:
      {
        "extracted_count": int,
        "user_id": str | None,
        "memories": [...],
        "comparison": {...}
      }
    """
    if source_mode == "extracted":
        conversations = get_extracted_conversations(user_id=user_id, limit=limit)
    else:
        conversations = get_unextracted_conversations(user_id=user_id, limit=limit)
    if not conversations:
        return {"extracted_count": 0, "user_id": user_id, "memories": [], "comparison": {}}

    grouped: dict[str, list[dict]] = {}
    for m in conversations:
        grouped.setdefault(m["user_id"], []).append(m)

    summarizer = KobartSummarizer(model_id=model_id)
    settings = get_settings()
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )

    strict_memories: list[ExtractedMemoryItem] = []
    comparison: dict[str, Any] = {"users": {}}

    for uid, msgs in grouped.items():
        lines = _build_conversation_lines(msgs)
        chunks = _chunks_from_lines(lines, max_chars=900)
        user_items: list[KobartMemory] = []

        for chunk in chunks:
            summary = await asyncio.to_thread(summarizer.summarize, chunk)
            intent = _infer_intent(chunk)
            kws = _extract_keywords(chunk, top_k=8)
            sentence_sc = _sentence_scores(chunk)
            importance = _importance_from_scores(sentence_sc)
            emotion = _infer_emotion(chunk)
            category = _infer_category(intent)
            when = msgs[-1].get("chat_time")
            memory_date = datetime.now().strftime("%Y-%m-%d")
            if isinstance(when, str) and len(when) >= 10:
                memory_date = when[:10]

            item = KobartMemory(
                memory_date=memory_date,
                content=summary,
                category=category,
                emotion=emotion,
                importance_score=importance,
                entity=kws[:5],
                intent=intent,
                keywords=kws,
                sentence_scores=sentence_sc,
            )
            user_items.append(item)

        llm_items: list[ExtractedMemoryItem] = []
        if compare_with_llm:
            llm_items = await _llm_preview_extract(msgs)

        llm_tokens = set()
        if llm_items:
            llm_text = " ".join(x.content for x in llm_items)
            llm_tokens = _normalize_text_tokens(llm_text)

        overlap_scores: list[float] = []
        imp_gap: list[float] = []

        for item in user_items:
            if check_duplicate_memory(uid, item.content):
                continue

            kobart_tokens = _normalize_text_tokens(item.content)
            overlap = _jaccard(kobart_tokens, llm_tokens)
            overlap_scores.append(overlap)
            if llm_items:
                llm_imp_avg = sum(int(x.importance_score) for x in llm_items) / len(llm_items)
                imp_gap.append(abs(item.importance_score - llm_imp_avg))

            metadata = {
                "source": item.source,
                "category": item.category,
                "emotion": item.emotion,
                "importance_score": item.importance_score,
                "entity": item.entity,
                "intent": item.intent,
                "keywords": item.keywords,
                "sentence_importance": item.sentence_scores,
                "comparison": {
                    "llm_reference_count": len(llm_items),
                    "keyword_overlap_jaccard": round(overlap, 4),
                },
            }

            if save_to_db:
                emb = embeddings_model.embed_query(item.content)
                save_long_term_memory(
                    user_id=uid,
                    memory_date=item.memory_date,
                    content=item.content,
                    embedding=emb,
                    metadata=metadata,
                )

            strict_memories.append(item.to_extracted_item())

        if source_mode != "extracted" and mark_extracted_after_save and save_to_db:
            chat_ids = [m["chat_id"] for m in msgs]
            mark_as_extracted(chat_ids)

        comparison["users"][uid] = {
            "kobart_memory_count": len(user_items),
            "llm_memory_count": len(llm_items),
            "avg_keyword_overlap_jaccard": round(sum(overlap_scores) / len(overlap_scores), 4)
            if overlap_scores else None,
            "avg_importance_gap": round(sum(imp_gap) / len(imp_gap), 4) if imp_gap else None,
        }

        if store_comparison_stats:
            try:
                report_date = datetime.now().strftime("%Y-%m-%d")
                cmp_info = comparison["users"][uid]
                distribution = {
                    "source": "kobart_vs_llm",
                    "kobart_memory_count": cmp_info["kobart_memory_count"],
                    "llm_memory_count": cmp_info["llm_memory_count"],
                    "avg_keyword_overlap_jaccard": cmp_info["avg_keyword_overlap_jaccard"],
                    "avg_importance_gap": cmp_info["avg_importance_gap"],
                    "model_id": model_id,
                }
                upsert_daily_sentiment_stat(
                    uid,
                    report_date,
                    {
                        "emotion_distribution": distribution,
                        "daily_avg_incongruence": cmp_info["avg_importance_gap"] or 0,
                        "total_utterance": len(msgs),
                    },
                )
            except Exception:
                logger.exception("daily_sentiment_stats 저장 실패 user=%s", uid)

    # 기존 LLM 추출과 동일한 엄격 JSON 포맷 보장
    strict = ExtractionResult(memory=strict_memories)
    return strict.model_dump()

