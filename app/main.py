"""
FastAPI 엔트리포인트: /chat, /extract-memory 엔드포인트 + 정적 파일 서빙.
FastAPI 엔트리포인트 — 비동기 섀도우 파이프라인
[아키텍처]
  /stt  → Google STT만 await → transcript + request_id 즉시 반환
         → BackgroundTask: HuBERT 음성 감정 분석 → 캐시 저장
  /chat → LLM 답변 우선 생성 → 즉시 반환
         → BackgroundTask: shadow_settlement_process
           (텍스트 감정 + 캐시 폴링 + Soft Voting + 세션/DB 갱신)
"""

import asyncio
import base64
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from google.api_core import exceptions as gcp_exceptions
from google.cloud import texttospeech
from pydantic import BaseModel

from app.database import (
    save_chat_message,
    save_chat_message_with_emotion,
    get_senior_profile,
    get_weekly_reports,
    get_daily_reports_since,
    get_user_messages_since,
    list_recent_memories,
    list_daily_sentiment_stats,
    log_intervention,
    update_intervention_response,
    get_recent_interventions,
)

from app.config import VOICE_EMOTION_CACHE, get_settings
from app.sentiment_inference import analyze_audio_sentiment, analyze_text_sentiment, load_soft_voting_params

from app.report_monthly_keywords_llm import extract_monthly_keywords_with_llm
from app.report_caregiver_data import build_mood_weather_calendar_14d, extract_monthly_keywords, normalize_report_date_key, month_start_iso_for_chat_filter
from app.report_reminiscence_llm import filter_positive_memory_candidates, generate_reminiscence_lines
from app.emotion import update_user_emotion, update_action_fatigue
from app.memory_agent import extract_memories
from app.rag import empathetic_chat
from app.stt import transcribe_bytes
from app.reminiscence import build_morning_brief, generate_reminiscence_welcome
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ExtractedMemoryItem,
    ExtractMemoryRequest,
    ExtractMemoryResponse,
    ExtractionResult,
    ParseScheduleRequest,
    ParseScheduleResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="마음 - 시니어 감정 맞춤형 RAG 챗봇",
    version="0.1.0",
    description="60대 이상 독거 어르신을 위한 음성 기반 감정 맞춤형 말벗 챗봇 백엔드",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /chat ───────────────────────────────────────────────────────

_extract_lock = asyncio.Lock()

_VOICE_CACHE_LOCK = asyncio.Lock()

# request_id → STT·채팅 메타 (섀도우 정산 전까지 누적)
_SHADOW_TURN_LOCK = asyncio.Lock()
SHADOW_TURN_BUFFER: dict[str, dict] = {}

EMOTION_SCORE_MAP = {
    "공포": 6.0,
    "슬픔": 4.0,
    "분노": 3.0,
    "혐오": 2.0,
    "놀람": 0.0,
    "중립": 0.0,
    "기쁨": -6.0,
}


def _generate_request_id() -> str:
    return f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}"


def _expected_value(prob_map: dict | None) -> float:
    if not isinstance(prob_map, dict):
        return 0.0
    total = 0.0
    for emotion, prob in prob_map.items():
        total += float(prob) * float(EMOTION_SCORE_MAP.get(str(emotion), 0.0))
    return total


def _apply_soft_voting(text_emotion: dict, voice_emotion: dict | None) -> tuple[str, float, float, dict]:
    params = load_soft_voting_params()
    w_text = float(params["w_text"])
    w_audio = float(params["w_audio"])
    neutral_margin = float(params["neutral_margin"])

    text_probs = text_emotion.get("probs") or {}
    voice_probs = (voice_emotion or {}).get("probs") or {}
    labels = list(EMOTION_SCORE_MAP.keys())

    combined: dict[str, float] = {}
    for label in labels:
        combined[label] = (w_text * float(text_probs.get(label, 0.0))) + (w_audio * float(voice_probs.get(label, 0.0)))

    top_label = max(combined, key=combined.get)
    non_neutral = {k: v for k, v in combined.items() if k != "중립"}
    best_non_neutral = max(non_neutral, key=non_neutral.get)

    if top_label == "중립":
        if float(combined["중립"] - non_neutral[best_non_neutral]) < neutral_margin:
            top_label = best_non_neutral

    top_score = float(combined[top_label])
    incongruence_score = abs(_expected_value(text_probs) - _expected_value(voice_probs))

    return top_label, top_score, incongruence_score, {
        "w_text": w_text,
        "w_audio": w_audio,
        "neutral_margin": neutral_margin,
        "text_probs": text_probs,
        "voice_probs": voice_probs,
        "combined_probs": combined,
        "final_emotion": top_label,
        "final_score": top_score,
    }


async def _cache_voice_emotion(request_id: str, voice_emotion: dict) -> None:
    now_ts = time.time()
    async with _VOICE_CACHE_LOCK:
        VOICE_EMOTION_CACHE[request_id] = {"voice_emotion": voice_emotion, "stored_at": now_ts}


def _sweep_stale_voice_cache_unlocked(now_ts: float, ttl: float) -> None:
    stale_keys = [
        key
        for key, item in VOICE_EMOTION_CACHE.items()
        if (now_ts - float(item.get("stored_at", 0.0))) > ttl
    ]
    for key in stale_keys:
        VOICE_EMOTION_CACHE.pop(key, None)


async def _pop_voice_emotion(request_id: str) -> dict | None:
    """캐시에 이미 있으면 즉시 꺼냄(대기 없음). 섀도우 정산에는 `_wait_and_pop_voice_emotion` 사용."""
    settings = get_settings()
    ttl = float(settings.voice_emotion_cache_ttl_seconds)
    now_ts = time.time()
    async with _VOICE_CACHE_LOCK:
        _sweep_stale_voice_cache_unlocked(now_ts, ttl)
        row = VOICE_EMOTION_CACHE.pop(request_id, None)
    if not row:
        return None
    return row.get("voice_emotion")


async def _wait_and_pop_voice_emotion(request_id: str) -> dict | None:
    """`/stt` 백그라운드 HuBERT가 끝나 캐시에 올라올 때까지 대기 후 pop.

    `/chat` 응답은 빠르게 나가고, 섀도우 태스크가 여기서 잠깐 기다리면
    `voice_emotion`이 None이 되는 레이스를 줄인다.
    """
    settings = get_settings()
    ttl = float(settings.voice_emotion_cache_ttl_seconds)
    wait_max = float(settings.voice_emotion_shadow_wait_seconds)
    poll = 0.15
    deadline = time.time() + wait_max

    while time.time() < deadline:
        async with _VOICE_CACHE_LOCK:
            now_ts = time.time()
            _sweep_stale_voice_cache_unlocked(now_ts, ttl)
            if request_id in VOICE_EMOTION_CACHE:
                row = VOICE_EMOTION_CACHE.pop(request_id, None)
                if row:
                    ve = row.get("voice_emotion")
                    logger.info("음성 감정 캐시 수신: request_id=%s", request_id)
                    return ve
        await asyncio.sleep(poll)

    logger.warning(
        "음성 감정 캐시 없음(분석 지연·실패 또는 타임아웃 %.1fs): request_id=%s",
        wait_max,
        request_id,
    )
    return None


async def _background_audio_analysis(request_id: str, audio_bytes: bytes) -> None:
    try:
        voice_emotion = await asyncio.to_thread(analyze_audio_sentiment, audio_bytes)
        await _cache_voice_emotion(request_id, voice_emotion)
        logger.info("음성 감정 캐시 저장 완료: request_id=%s emotion=%s", request_id, voice_emotion.get("top_emotion"))
    except Exception:
        logger.exception("백그라운드 음성 감정 분석 실패: request_id=%s", request_id)


async def _register_stt_turn(request_id: str, user_id: str, stt_text: str) -> None:
    now = time.time()
    async with _SHADOW_TURN_LOCK:
        row = SHADOW_TURN_BUFFER.setdefault(request_id, {})
        row["user_id"] = user_id
        row["stt_text"] = stt_text
        row["stt_at"] = now


async def _merge_chat_into_shadow(request_id: str, req: ChatRequest, llm_result: dict) -> None:
    now = time.time()
    async with _SHADOW_TURN_LOCK:
        row = SHADOW_TURN_BUFFER.setdefault(request_id, {})
        row["user_id"] = req.user_id
        row["message"] = req.message
        row["request_dump"] = req.model_dump()
        row["llm_result"] = llm_result
        row["chat_at"] = now


async def _prune_stale_shadow_rows(max_age_seconds: float = 3600.0) -> None:
    now = time.time()
    async with _SHADOW_TURN_LOCK:
        stale = [
            k
            for k, v in SHADOW_TURN_BUFFER.items()
            if now - float(v.get("stt_at", v.get("chat_at", now))) > max_age_seconds
        ]
        for k in stale:
            SHADOW_TURN_BUFFER.pop(k, None)


async def _shadow_persist_turn(request_id: str) -> None:
    """텍스트·음성 감정 + 소프트 보팅 후 사용자/어시스턴트 메시지를 DB에 기록."""
    async with _SHADOW_TURN_LOCK:
        row = SHADOW_TURN_BUFFER.pop(request_id, None)
    if not row or "llm_result" not in row:
        logger.warning("섀도우 버퍼에 정산할 턴 없음: request_id=%s", request_id)
        return
    message = str(row.get("message") or "")
    user_id = str(row["user_id"])
    llm_result = row["llm_result"]
    answer = str(llm_result.get("answer") or "")
    try:
        text_emotion, voice_emotion = await asyncio.gather(
            asyncio.to_thread(analyze_text_sentiment, message),
            _wait_and_pop_voice_emotion(request_id),
        )
        emotion_label, emotion_score, incongruence_score, voting_log = _apply_soft_voting(text_emotion, voice_emotion)

        def _save_user() -> None:
            save_chat_message_with_emotion(
                user_id=user_id,
                role="user",
                content=message,
                emotion_label=emotion_label,
                emotion_score=emotion_score,
                incongruence_score=incongruence_score,
                text_emotion=text_emotion,
                voice_emotion=voice_emotion,
                soft_voting_log=voting_log,
            )

        await asyncio.to_thread(_save_user)
        await asyncio.to_thread(save_chat_message, user_id, "assistant", answer)
        logger.info("섀도우 DB 저장 완료: request_id=%s emotion=%s", request_id, emotion_label)
    except Exception:
        logger.exception("섀도우 정산/저장 실패: request_id=%s", request_id)


async def _background_simple_turn(req: ChatRequest, answer: str) -> None:
    """request_id 없음: 감정 필드 없이 단기 이력만 저장."""
    try:
        await asyncio.to_thread(save_chat_message, req.user_id, "user", req.message)
        await asyncio.to_thread(save_chat_message, req.user_id, "assistant", answer)
    except Exception:
        logger.exception("단순 턴 저장 실패: user_id=%s", req.user_id)

async def _background_extract(user_id: str) -> None:
    """응답 반환 후 백그라운드로 장기기억 추출."""
    if _extract_lock.locked():
        return
    async with _extract_lock:
        try:
            extracted = await extract_memories(user_id=user_id, limit=50)
            if extracted:
                logger.info("백그라운드 기억 추출 완료: %d건", len(extracted))
        except Exception:
            logger.exception("백그라운드 기억 추출 실패")


async def _shadow_and_extract(request_id: str, user_id: str) -> None:
    await _shadow_persist_turn(request_id)
    await _background_extract(user_id)


async def _simple_save_and_extract(req: ChatRequest, answer: str) -> None:
    await _background_simple_turn(req, answer)
    await _background_extract(req.user_id)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, bg: BackgroundTasks):
    """Critical path: RAG(임베딩→검색→답변 LLM)만 await. 선택 시 같은 응답에 TTS(MP3) 포함.

    request_id가 있으면(권장: /stt 직후) 텍스트·음성 감정·소프트 보팅과 DB 저장은 섀도우에서 수행한다.
    """
    try:
        result = await empathetic_chat(
            user_id=req.user_id,
            message=req.message,
            category=req.category,
            emotion=req.emotion,
            display_name=req.display_name,
            gender=req.gender,
            age=req.age,
            avatar_id=req.avatar_id,
        )
    except Exception:
        logger.exception("RAG 챗봇 실패")
        raise HTTPException(status_code=502, detail="답변 생성에 실패했어요. 잠시 후 다시 시도해 주세요.")

    audio_b64: str | None = None
    if req.include_tts and result.get("answer"):
        try:
            mp3 = await asyncio.to_thread(_synthesize_mp3_bytes, result["answer"])
            audio_b64 = base64.b64encode(mp3).decode("ascii")
        except Exception:
            logger.exception("응답용 TTS 생성 실패 (텍스트만 반환)")

    if req.request_id:
        await _merge_chat_into_shadow(req.request_id, req, result)
        await _prune_stale_shadow_rows()
        bg.add_task(_shadow_and_extract, req.request_id, req.user_id)
    else:
        bg.add_task(_simple_save_and_extract, req, result["answer"])

    return ChatResponse(
        **result,
        request_id=req.request_id,
        audio_base64=audio_b64,
    )


# ── POST /extract-memory ─────────────────────────────────────────────

@app.post("/extract-memory", response_model=ExtractMemoryResponse)
async def extract_memory_endpoint(req: ExtractMemoryRequest):
    """단기 대화에서 장기 기억을 추출하여 벡터 DB에 저장한다."""
    try:
        extracted = await extract_memories(user_id=req.user_id, limit=req.limit)
    except Exception:
        logger.exception("기억 추출 파이프라인 실패")
        raise HTTPException(status_code=502, detail="기억 추출에 실패했어요. 잠시 후 다시 시도해 주세요.")

    memories = [ExtractedMemoryItem(**m) for m in extracted]
    return ExtractMemoryResponse(
        extracted_count=len(memories),
        user_id=req.user_id,
        memories=memories,
    )
class KobartExtractRequest(BaseModel):
    user_id: str | None = None
    limit: int = 50
    save_to_db: bool = True
    mark_extracted_after_save: bool = True
    compare_with_llm: bool = True
    store_comparison_stats: bool = True
    source_mode: str = "unextracted"  # unextracted | extracted
    model_id: str = "gogamza/kobart-summarization"


# ── POST /tts ─────────────────────────────────────────────────────────

class TtsRequest(BaseModel):
    text: str


@app.post("/extract-memory-kobart", response_model=ExtractionResult)
async def extract_memory_kobart_endpoint(_req: KobartExtractRequest):
    """
    KoBART 기반 장기기억 추출 실험 엔드포인트.
    source_mode=extracted 로 주면 short_term_history(is_extracted=true)를 기준으로 비교한다.
    """
    raise HTTPException(
        status_code=501,
        detail="KoBART 추출 모듈이 이 빌드에 포함되어 있지 않습니다.",
    )


_tts_client: texttospeech.TextToSpeechClient | None = None

def _get_tts_client() -> texttospeech.TextToSpeechClient:
    global _tts_client
    if _tts_client is None:
        _tts_client = texttospeech.TextToSpeechClient()
    return _tts_client

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining enclosing keycap
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000231A-\U0000231B"
    "\U00002934-\U00002935"
    "\U000025AA-\U000025FE"
    "\U00002B05-\U00002B55"
    "\U00003030\U0000303D"
    "\U00003297\U00003299"
    "]+",
    flags=re.UNICODE,
)

def _clean_text_for_tts(text: str) -> str:
    """TTS 전에 이모지, 특수기호 등 음성으로 읽히면 어색한 문자를 제거한다."""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"[*#_~`\[\](){}|]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _synthesize_mp3_bytes(text: str) -> bytes:
    client = _get_tts_client()
    clean_text = _clean_text_for_tts(text)
    if not clean_text:
        raise ValueError("empty TTS text")
    synthesis_input = texttospeech.SynthesisInput(text=clean_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Neural2-A",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.95,
        pitch=1.0,
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content


@app.post("/tts")
async def tts_endpoint(req: TtsRequest):
    """텍스트를 자연스러운 한국어 음성(MP3)으로 변환한다."""
    try:
        client = _get_tts_client()
        clean_text = _clean_text_for_tts(req.text)
        if not clean_text:
            raise HTTPException(status_code=400, detail="읽을 텍스트가 없습니다.")
        synthesis_input = texttospeech.SynthesisInput(text=clean_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Neural2-A",
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
            pitch=1.0,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return Response(content=response.audio_content, media_type="audio/mpeg")
    except Exception:
        logger.exception("TTS 변환 실패")
        raise HTTPException(status_code=502, detail="음성 변환에 실패했어요.")


# ── POST /stt ─────────────────────────────────────────────────────────

@app.post("/stt")
async def stt_endpoint(
    bg: BackgroundTasks,
    audio: UploadFile = File(..., description="녹음 파일 (WebM/Opus 또는 WAV LINEAR16; 프론트는 링버퍼+발화를 WAV로 합쳐 전송 가능)"),
    user_id: str = Form("anonymous"),
    language_code: str = Form("ko-KR"),
    sample_rate_hertz: int = Form(48000),
):
    """Critical path: STT 후 request_id 즉시 반환, 음성감정은 shadow 처리."""
    """Google Speech-to-Text — WebM/Opus 또는 WAV(실제 샘플레이트는 sample_rate_hertz와 일치)."""
    body = await audio.read()
    if not body:
        raise HTTPException(status_code=400, detail="녹음 데이터가 비어 있습니다.")
    try:
        text = await asyncio.to_thread(
            transcribe_bytes,
            body,
            content_type=audio.content_type,
            language_code=language_code,
            sample_rate_hertz=sample_rate_hertz,
        )
    except gcp_exceptions.GoogleAPIError as e:
        logger.exception("Speech-to-Text API 오류")
        msg = getattr(e, "message", None) or str(e)
        raise HTTPException(
            status_code=502,
            detail=f"음성 인식 서비스 오류입니다. GCP에서 Speech-to-Text API 사용 설정과 결제·권한을 확인해 주세요. ({msg})",
        )
    except Exception:
        logger.exception("음성 인식 실패")
        raise HTTPException(
            status_code=502,
            detail="음성 인식에 실패했어요. 잠시 후 다시 시도해 주세요.",
        )

    request_id = _generate_request_id()
    await _register_stt_turn(request_id, user_id, text)
    await _prune_stale_shadow_rows()
    bg.add_task(_background_audio_analysis, request_id, body)
    return {
        "text": text,
        "request_id": request_id,
        "user_id": user_id,
        "language_code": language_code,
    }

# ── GET /profile/{user_id} ────────────────────────────────────────────

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """사용자 프로필 + 정서 상태 원스톱 조회."""
    profile = get_senior_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없어요.")
    return profile


@app.get("/morning-brief/{user_id}")
async def morning_brief(
    user_id: str,
    min_importance: int = 7,
    memory_limit: int = 250,
    max_memories: int = 8,
):
    """
    아침 TV/안내용: effective_distress가 WARNING(41~79)이면 행복·고중요도 회상 후보,
    DANGER(80+)이면 caregiver_alert + 오늘 일간 리포트 requires_check.
    별도 테이블 없이 long_term_memory를 조건 필터한다.
    """
    data = build_morning_brief(
        user_id,
        min_importance=min_importance,
        memory_fetch_limit=memory_limit,
        max_memories=max_memories,
    )
    if data.get("error") == "no_profile":
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없어요.")
    return data


# ── GET /welcome/{user_id} ───────────────────────────────────────────

@app.get("/welcome/{user_id}")
async def welcome_message(user_id: str, display_name: str | None = None):
    """
    채팅 화면 첫 인사 말풍선용 발화문.
    distress_score >= 35(WARNING 진입)이면 장기기억(emotion 행복계열, importance >= 6)을
    활용한 긍정 회상 발화문을 LLM으로 생성한다.
    그 외에는 기본 인사 문구를 반환한다.
    """
    from app.database import get_senior_profile as _gsp
    profile = _gsp(user_id)
    distress = float((profile or {}).get("distress_score") or 0)
    db_name = (profile or {}).get("name") or None
    name = display_name or db_name or None
    name_str = f"{name} 님" if name else "어르신"

    if distress >= 35:
        msg = await generate_reminiscence_welcome(user_id, name)
        return {"message": msg, "reminiscence_mode": True, "distress_score": distress}

    return {
        "message": f"{name_str}, 안녕하세요! 오늘 하루는 어떠셨어요? 편하게 말씀해 주세요.",
        "reminiscence_mode": False,
        "distress_score": distress,
    }


# ── POST /parse-schedule ────────────────────────────────────────────

@app.post("/parse-schedule", response_model=ParseScheduleResponse)
async def parse_schedule(body: ParseScheduleRequest):
    """
    사용자 발화에서 일정/루틴 추가 의도와 시간·설명을 LLM으로 추출한다.
    프론트엔드 채팅창에서 일정 추가 요청 감지 및 내용 요약에 사용한다.
    """
    import json as _json
    import re as _re
    from langchain_google_genai import ChatGoogleGenerativeAI

    text = body.text.strip()
    if not text:
        return ParseScheduleResponse(is_schedule=False)
    settings = get_settings()

    prompt = f"""사용자 발화를 분석하여 일정/루틴/할 일 추가 요청인지 판단하고, 맞다면 시간·설명·반복 여부를 추출하세요.

사용자 발화: "{text}"

출력 규칙:
- is_schedule: 일정/루틴/할 일 추가 요청이면 true, 아니면 false
- time: "HH:MM" 24시간제. 시간 정보가 없으면 null
- desc: 핵심 내용을 간결하게 요약 (조사·동사·키워드 제거, 명사 위주 5~12자 이내)
  좋은 예: "병원 진료", "혈압약 복용", "손녀에게 전화", "산책", "점심 식사"
  나쁜 예: "병원에 가는", "약 먹는 것을 잊지 않", "루틴을 등록"
- recurring: "매일", "날마다", "항상", "매주", "every day" 등 반복/주기 표현이 있으면 true, 없으면 false
- 오직 JSON 한 덩어리만 출력. 마크다운 코드펜스 없이.

예시:
입력: "오후 3시에 병원 예약 있는데 일정 추가해줘"
출력: {{"is_schedule": true, "time": "15:00", "desc": "병원 예약", "recurring": false}}

입력: "매일 아침 8시에 혈압약 먹는 루틴 등록해줘"
출력: {{"is_schedule": true, "time": "08:00", "desc": "혈압약 복용", "recurring": true}}

입력: "내일 오전 10시 반에 복지관 노래 교실 일정 넣어줘"
출력: {{"is_schedule": true, "time": "10:30", "desc": "복지관 노래 교실", "recurring": false}}

입력: "날마다 저녁 7시에 손녀한테 전화하는 루틴 만들어줘"
출력: {{"is_schedule": true, "time": "19:00", "desc": "손녀 전화", "recurring": true}}

입력: "내 할 일에 매일 아침 7시 스트레칭 넣어줘"
출력: {{"is_schedule": true, "time": "07:00", "desc": "아침 스트레칭", "recurring": true}}

입력: "오늘 날씨 어때?"
출력: {{"is_schedule": false, "time": null, "desc": null, "recurring": false}}"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.1,
        max_output_tokens=80,
        thinking_budget=0,
    )
    try:
        resp = await llm.ainvoke(prompt)
        raw = (resp.content or "").strip()
        raw = _re.sub(r"```[a-z]*\s*|\s*```", "", raw).strip()
        data = _json.loads(raw)
        return ParseScheduleResponse(
            is_schedule=bool(data.get("is_schedule")),
            time=data.get("time") or None,
            desc=data.get("desc") or None,
            recurring=bool(data.get("recurring", False)),
        )
    except Exception:
        logger.exception("일정 파싱 LLM 실패")
        return ParseScheduleResponse(is_schedule=False)


# ── GET /reports/{user_id} ───────────────────────────────────────────

@app.get("/reports/{user_id}")
async def get_reports(user_id: str, days: int = 7):
    """최근 N일간 일간 케어 리포트 목록 조회."""
    reports = get_weekly_reports(user_id, limit=days)
    return {"user_id": user_id, "reports": reports}


@app.get("/memory-benchmark/{user_id}")
async def memory_benchmark(user_id: str, days: int = 7):
    """
    KoBART vs LLM 비교 요약 조회.
    - daily_sentiment_stats의 kobart_vs_llm 지표 시계열
    - long_term_memory 최근 메모리 source별 건수
    """
    stats = list_daily_sentiment_stats(user_id, limit=max(1, min(days, 60)))
    trend: list[dict] = []
    for s in stats:
        dist = s.get("emotion_distribution") or {}
        if dist.get("source") != "kobart_vs_llm":
            continue
        trend.append(
            {
                "report_date": s.get("report_date"),
                "kobart_memory_count": dist.get("kobart_memory_count"),
                "llm_memory_count": dist.get("llm_memory_count"),
                "avg_keyword_overlap_jaccard": dist.get("avg_keyword_overlap_jaccard"),
                "avg_importance_gap": dist.get("avg_importance_gap"),
                "model_id": dist.get("model_id"),
                "total_utterance": s.get("total_utterance"),
            }
        )

    memories = list_recent_memories(user_id, limit=300)

    source_counts: dict[str, int] = {}
    for m in memories:
        meta = m.get("metadata") or {}
        src = meta.get("source") or "legacy"
        source_counts[src] = source_counts.get(src, 0) + 1

    latest = trend[0] if trend else None

    return {
        "user_id": user_id,
        "days": days,
        "latest": latest,
        "trend": trend,
        "long_term_memory_source_counts": source_counts,
        "notes": {
            "avg_keyword_overlap_jaccard": "KoBART 요약 vs LLM 추출 키워드 자카드 평균",
            "avg_importance_gap": "KoBART 중요도와 LLM 평균 중요도 차이(낮을수록 유사)",
        },
    }


# ── GET /report-summary/{user_id} ────────────────────────────────────

@app.get("/report-summary/{user_id}")
async def get_report_summary(user_id: str):
    """리포트 UI에 필요한 가공 데이터 반환."""
    KST = timezone(timedelta(hours=9))
    today = datetime.now(KST).date()
    day_names = ['월', '화', '수', '목', '금', '토', '일']

    reports = get_weekly_reports(user_id, limit=7)

    date_map: dict[str, dict] = {}
    for r in reports:
        dk = normalize_report_date_key(r.get("report_date"))
        if dk:
            date_map[dk] = r

    weekly_scores = []
    weekly_labels = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        ds = d.isoformat()
        r = date_map.get(ds)
        score = round(float(r["mood_avg_score"]), 1) if r and r.get("mood_avg_score") is not None else None
        weekly_scores.append(score)
        label = '오늘' if i == 0 else day_names[d.weekday()]
        weekly_labels.append(label)

    today_str = today.isoformat()
    today_report = date_map.get(today_str)
    today_score = round(float(today_report["mood_avg_score"]), 1) if today_report and today_report.get("mood_avg_score") is not None else None

    past_scores = [s for s in weekly_scores[:-1] if s is not None]
    avg_7d = round(sum(past_scores) / len(past_scores), 1) if past_scores else None

    diff = None
    diff_label = ""
    if today_score is not None and avg_7d is not None:
        diff = round(today_score - avg_7d, 1)
        if diff > 0:
            diff_label = f"평소보다 {abs(diff)}점 높습니다."
        elif diff < 0:
            diff_label = f"평소보다 {abs(diff)}점 낮습니다."
        else:
            diff_label = "평소와 비슷합니다."

    care_level = "NORMAL"
    if today_report and today_report.get("today_care_level"):
        care_level = today_report["today_care_level"]

    total_utterance = today_report.get("total_utterance", 0) if today_report else 0

    # ── 회상 요법: 긍정 기억 + LLM 멘트 (실패 시 템플릿) ──
    reminiscence: list[dict] = []
    try:
        memories = list_recent_memories(user_id, limit=50)
        candidates = filter_positive_memory_candidates(memories, max_items=3)
        messages = await generate_reminiscence_lines(candidates)
        reminiscence = [
            {
                "date": c.get("memory_date"),
                "content": c.get("content", ""),
                "emotion": c.get("emotion", ""),
                "message": msg,
            }
            for c, msg in zip(candidates, messages, strict=True)
        ]
    except Exception:
        logger.exception("회상 기억 조회/생성 실패")

    # ── 보호자 리포트: 최근 14일 마음 날씨 달력(mood_avg_score) + 월간 키워드 ──
    care_calendar_month: dict = {
        "kind": "14d",
        "start_date": (today - timedelta(days=13)).isoformat(),
        "end_date": today.isoformat(),
        "days": [],
    }
    monthly_keywords: list[dict] = []
    try:
        since_14d = (today - timedelta(days=13)).isoformat()
        cal_rows = get_daily_reports_since(user_id, since_14d, limit=20)
        care_calendar_month = build_mood_weather_calendar_14d(today, cal_rows)
    except Exception:
        logger.exception("마음 날씨 달력 데이터 조회 실패")

    try:
        since_chat = month_start_iso_for_chat_filter(today.year, today.month)
        chat_texts = get_user_messages_since(user_id, since_chat, limit=500)
        monthly_keywords = await extract_monthly_keywords_with_llm(
            chat_texts, top_n=5
        )
        if not monthly_keywords:
            monthly_keywords = extract_monthly_keywords(chat_texts, top_n=5)
    except Exception:
        logger.exception("월간 키워드 조회 실패")

    return {
        "user_id": user_id,
        "care_level": care_level,
        "weekly_scores": weekly_scores,
        "weekly_labels": weekly_labels,
        "today_score": today_score,
        "avg_7d": avg_7d,
        "diff": diff,
        "diff_label": diff_label,
        "total_utterance": total_utterance,
        "reminiscence": reminiscence,
        "care_calendar_month": care_calendar_month,
        "monthly_keywords": monthly_keywords,
    }


# ── POST /intervention ───────────────────────────────────────────────

class InterventionRequest(BaseModel):
    user_id: str
    action_type: str
    trigger_emotion: str | None = None
    suggested_content: str | None = None

class InterventionFeedback(BaseModel):
    user_id: str
    action_id: int
    user_response: str

@app.post("/intervention")
async def create_intervention(req: InterventionRequest):
    """행동 개입을 기록한다 (VOD_추천, 음악_재생, 스트레칭_제안 등)."""
    try:
        record = log_intervention(
            user_id=req.user_id,
            action_type=req.action_type,
            trigger_emotion=req.trigger_emotion,
            suggested_content=req.suggested_content,
        )
        return {"status": "logged", "action": record}
    except Exception:
        logger.exception("행동 개입 기록 실패")
        raise HTTPException(status_code=502, detail="행동 개입 기록 실패")


@app.post("/intervention/feedback")
async def intervention_feedback(req: InterventionFeedback):
    """행동 개입에 대한 사용자 반응을 기록하고 추천 피로도를 갱신한다."""
    try:
        result = update_intervention_response(req.action_id, req.user_response)
        fatigue = update_action_fatigue(req.user_id, req.user_response)
        return {"status": "updated", "action": result, "action_fatigue": fatigue}
    except Exception:
        logger.exception("행동 개입 피드백 실패")
        raise HTTPException(status_code=502, detail="피드백 기록 실패")


# ── GET /interventions/{user_id} ─────────────────────────────────────

@app.get("/interventions/{user_id}")
async def list_interventions(user_id: str, limit: int = 10):
    """최근 행동 개입 이력 조회."""
    records = get_recent_interventions(user_id, limit=limit)
    return {"user_id": user_id, "interventions": records}


# ── GET /weather ──────────────────────────────────────────────────────

KMA_AUTH_KEY = "a_c5KbU5TUq3OSm1OU1Kjg"
KMA_BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
KST = timezone(timedelta(hours=9))
SEOUL_STN = 108

@app.get("/weather")
async def weather_endpoint():
    """서울 현재 날씨 (기상청 API) — 기온, 습도, 강수량."""
    now_kst = datetime.now(KST)
    errors = []

    def _safe_float(v: str) -> float | None:
        try:
            f = float(v)
            return None if f <= -9 else f
        except Exception:
            return None

    def _parse_weather_text(text: str, station: int) -> dict | None:
        """KMA 원문 텍스트에서 해당 관측소 1건을 파싱한다."""
        station_str = str(station)
        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 정상 데이터는 대체로 최소 16개 컬럼을 가진다.
            if len(parts) < 16:
                continue
            # [YYMMDDHHMI, STN, ...] 형태를 가정
            stn = parts[1]
            if stn != station_str:
                continue
            return {
                "temperature": _safe_float(parts[11]),
                "humidity": _safe_float(parts[13]),
                "precipitation": _safe_float(parts[15]),
            }
        return None

    for hours_back in range(0, 4):
        t = now_kst - timedelta(hours=hours_back)
        tm = t.strftime("%Y%m%d%H00")
        url = f"{KMA_BASE_URL}?tm={tm}&stn={SEOUL_STN}&help=0&authKey={KMA_AUTH_KEY}"

        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                text = resp.text

            parsed = _parse_weather_text(text, SEOUL_STN)
            if parsed is not None:
                return {
                    "station": "서울",
                    "time": tm,
                    "temperature": parsed["temperature"],
                    "humidity": parsed["humidity"],
                    "precipitation": parsed["precipitation"],
                }
            # 예외가 없어도 데이터 포맷이 달라 파싱이 안 되는 경우를 로깅에 남긴다.
            preview = " ".join(text.strip().splitlines()[:2])[:160]
            errors.append(f"{tm}: parse_miss preview='{preview}'")
        except Exception as e:
            errors.append(f"{tm}: {e}")
            continue

    logger.warning("날씨 데이터 조회 실패: %s", errors)
    return {
        "station": "서울",
        "time": now_kst.strftime("%Y%m%d%H00"),
        "temperature": None,
        "humidity": None,
        "precipitation": None,
    }


# ── Health Check ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Frontend ─────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
