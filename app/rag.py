"""
Pipeline 2: Empathetic RAG Chat
사용자 발화 → 하이브리드 검색(벡터 + GIN 메타데이터 via Supabase RPC) → 공감형 답변 생성.
"""

import logging
import re
from difflib import SequenceMatcher

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.database import HybridSearchResult, hybrid_search

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """당신은 60대 이상 어르신들의 말벗이 되어주는 친절하고 따뜻한 AI 손주입니다.
아래 [기억(Context)]은 오직 이 대화 상대(동일 사용자)에게서만 수집된 기억입니다. 다른 분의 이야기로 착각하지 마세요.

{user_context}

- 규칙 1: 반드시 존댓말(해요체)을 사용하고, 살갑고 다정하게 말하세요.
- 규칙 2: [기억]에 없는 내용은 지어내지 말고 솔직하게 되물어보세요.
- 규칙 3: 문장을 너무 길지 않게, 간결하고 듣기 편하게 작성하세요.
- 규칙 4 (안전): 어르신이 죽음/극심한 우울을 언급하면 섣부른 위로를 피하고 `[ALERT: DEPRESSION]` 플래그를 맨 앞에 달아주세요. 화를 내시면 `[ALERT: ANGER]` 플래그를 다세요.
- 규칙 5 (감정 분류): 어르신 발화에서 감지한 주요 감정 1개를 답변 맨 끝에 `[EMO:감정]` 형식으로 태그하세요. 감정은 반드시 다음 중 하나: 기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 중립

[기억(Context)]:
{context}
"""

ALERT_PATTERN = re.compile(r"^\[ALERT:\s*(\w+)\]\s*")
EMO_PATTERN = re.compile(r"\[EMO:\s*(\S+?)\]\s*$")
FALLBACK_ANSWER = "어르신, 제가 잘 알아듣지 못했어요. 다시 한번 말씀해 주시겠어요?"

_AVATAR_PERSONA = {
    "boy": "화면의 말벗은 다정한 소년 손주 캐릭터입니다. 말투는 밝고 순수하게 유지하세요.",
    "girl": "화면의 말벗은 다정한 소녀 손주 캐릭터입니다. 말투는 부드럽고 따뜻하게 유지하세요.",
    "dog": "화면의 말벗은 귀여운 강아지 캐릭터입니다. 짧고 활기 있는 문장으로 반응하되 존댓말은 유지하세요.",
    "cat": "화면의 말벗은 차분한 고양이 캐릭터입니다. 여유 있고 담백한 톤으로 말하되 존댓말은 유지하세요.",
    "robot": "화면의 말벗은 친절한 로봇 캐릭터입니다. 정중하고 명확한 말투로 답하되 딱딱하지 않게 하세요.",
}


def _format_user_context(
    display_name: str | None,
    gender: str | None,
    age: int | None,
    avatar_id: str | None,
) -> str:
    """프론트에서 넘긴 프로필을 시스템 프롬프트에 삽입할 블록으로 만든다."""
    lines: list[str] = []
    if display_name and display_name.strip():
        name = display_name.strip()
        lines.append(f"- 어르신 이름(호칭): '{name} 님'이라고 부르세요.")
    if gender and gender.strip():
        lines.append(
            f"- 성별 정보: {gender.strip()}. 호칭은 상황에 맞게 할머니/할아버지/어르신 등 자연스럽게 선택하세요."
        )
    if age is not None and 1 <= age <= 120:
        lines.append(f"- 나이: {age}세.")
    if avatar_id and avatar_id in _AVATAR_PERSONA:
        lines.append(f"- {_AVATAR_PERSONA[avatar_id]}")
    if not lines:
        return ""
    return "[사용자 프로필 — 답변에 자연스럽게 반영]\n" + "\n".join(lines)


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )


_KEYWORD_PROMPT = (
    "사용자 발화에서 검색에 쓸 핵심 명사·키워드(장소, 인물, 사물, 활동, 꽃 이름 등)를 "
    "쉼표로 구분해 최대 5개 추출하세요. 동의어·유의어도 포함하세요.\n"
    "발화: {q}\n키워드:"
)


async def _extract_query_entities(message: str) -> list[str]:
    """사용자 발화에서 검색용 키워드를 빠르게 LLM 추출한다."""
    settings = get_settings()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            project=settings.gcp_project_id,
            location=settings.gcp_location,
            temperature=0,
            max_output_tokens=60,
            thinking_budget=0,
        )
        resp = await llm.ainvoke(_KEYWORD_PROMPT.format(q=message))
        raw = (resp.content or "").strip()
        keywords = [k.strip() for k in raw.split(",") if k.strip()]
        logger.info("쿼리 키워드 추출: %s → %s", message[:40], keywords)
        return keywords[:5]
    except Exception:
        logger.warning("쿼리 키워드 추출 실패 — entity 없이 검색")
        return []


def _deduplicate(results: list[HybridSearchResult], similarity_threshold: float = 0.85) -> list[HybridSearchResult]:
    """텍스트가 거의 동일한 기억만 제거한다 (threshold=0.85 → 내용이 다른 기억은 보존)."""
    kept: list[HybridSearchResult] = []
    for r in results:
        is_dup = False
        for k in kept:
            ratio = SequenceMatcher(None, r.content, k.content).ratio()
            if ratio >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


def _format_memories(results: list[HybridSearchResult]) -> str:
    if not results:
        return "(아직 참고할 기억이 없습니다. 어르신의 말씀에 자연스럽게 반응해 주세요.)"
    lines: list[str] = []
    for r in results:
        meta = r.metadata or {}
        imp = meta.get("importance_score", 5)
        tag_parts = [meta.get("category", ""), f"감정:{meta.get('emotion', '')}"]
        if meta.get("entity"):
            tag_parts.append(f"관련:{','.join(meta['entity'])}")
        if int(imp) >= 7:
            tag_parts.append(f"★중요도:{imp}")
        tag = " | ".join(p for p in tag_parts if p)
        lines.append(f"- [{tag}] {r.content}")
    return "\n".join(lines)


def _parse_alert(text: str) -> tuple[str, str | None]:
    """답변 텍스트에서 [ALERT: ...] 플래그를 분리한다."""
    match = ALERT_PATTERN.match(text)
    if match:
        alert_type = match.group(1)
        clean_text = text[match.end():].strip()
        return clean_text, alert_type
    return text, None


def _parse_emotion_tag(text: str) -> tuple[str, str]:
    """답변 끝의 [EMO:감정] 태그를 분리한다. 없으면 '중립' 반환."""
    match = EMO_PATTERN.search(text)
    if match:
        emotion = match.group(1)
        clean_text = text[:match.start()].strip()
        return clean_text, emotion
    return text, "중립"


async def empathetic_chat(
    user_id: str,
    message: str,
    *,
    category: str | None = None,
    emotion: str | None = None,
    display_name: str | None = None,
    gender: str | None = None,
    age: int | None = None,
    avatar_id: str | None = None,
    fetch_k: int = 15,
    final_k: int = 5,
    vector_weight: float = 0.65,
    metadata_weight: float = 0.10,
    recency_weight: float = 0.15,
    importance_weight: float = 0.10,
) -> dict:
    """하이브리드 RAG 기반 공감형 챗봇 응답을 생성한다.

    장기기억 검색·컨텍스트는 항상 ``user_id``로 격리된다 (Supabase ``hybrid_search``의
    ``WHERE user_id = p_user_id``). 다른 사용자의 벡터/메타데이터는 후보에 포함되지 않는다.
    """
    settings = get_settings()

    emb_model = _get_embeddings_model()
    query_embedding = emb_model.embed_query(message)

    query_entities = await _extract_query_entities(message)

    raw_results = hybrid_search(
        user_id=user_id,
        query_embedding=query_embedding,
        category=category,
        emotion=emotion,
        entities=query_entities or None,
        top_k=fetch_k,
        vector_weight=vector_weight,
        metadata_weight=metadata_weight,
        recency_weight=recency_weight,
        importance_weight=importance_weight,
    )

    results = _deduplicate(raw_results)[:final_k]

    context_str = _format_memories(results)
    logger.info(
        "하이브리드 검색 완료 (user_id 격리): user=%s, 원본=%d건→중복제거=%d건, category=%s, emotion=%s",
        user_id, len(raw_results), len(results), category, emotion,
    )

    user_context = _format_user_context(display_name, gender, age, avatar_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.7,
        max_output_tokens=256,
        thinking_budget=0,
    )

    formatted = prompt.invoke(
        {"context": context_str, "question": message, "user_context": user_context}
    )
    resp = llm.invoke(formatted.messages)
    raw_answer = resp.content or ""

    if not raw_answer.strip():
        logger.warning("LLM 빈 응답 — fallback 사용")
        raw_answer = FALLBACK_ANSWER

    answer, alert = _parse_alert(raw_answer)
    answer, detected_emotion = _parse_emotion_tag(answer)

    return {
        "answer": answer,
        "alert": alert,
        "detected_emotion": detected_emotion,
        "retrieved_memories": [r.content for r in results],
    }
