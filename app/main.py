"""
FastAPI 엔트리포인트 — 비동기 섀도우 파이프라인 (shadow.md).

[아키텍처]
  /stt  → Google STT만 await → transcript + request_id 즉시 반환
         → BackgroundTask: HuBERT 음성 감정 분석 → 캐시 저장
  /chat → empathetic_chat(LLM) 최우선 → 즉시 반환
         → BackgroundTask: 메시지 저장 + shadow_settlement_process
           (텍스트 감정 + 캐시 폴링 + Soft Voting + 프로필 갱신, emotion_session 없음)
"""

import asyncio
import importlib
import importlib.util
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    peek_senior_profile_cache,
    ensure_senior_profile,
    sync_all_cached_profiles_from_db,
    update_chat_emotion_fields,
    get_weekly_reports,
    list_daily_sentiment_stats,
    list_recent_memories,
    log_intervention,
    update_intervention_response,
    get_recent_interventions,
)
from app.emotion import update_action_fatigue, update_user_emotion
from app.kobart_memory_agent import extract_memories_kobart
from app.memory_agent import extract_memories
from app.rag import empathetic_chat
from app.stt import transcribe_bytes
from app.reminiscence import build_morning_brief
from app.scheduler import start_scheduler, stop_scheduler
from app.session_manager import (
    run_sentiment_pipeline,
    generate_request_id,
    store_audio_sentiment,
    pop_audio_sentiment,
    cleanup_stale_cache,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ExtractedMemoryItem,
    ExtractionResult,
    ExtractMemoryRequest,
    ExtractMemoryResponse,
)

# ── [전역 타임존 룰] 모든 시간은 KST 기준 ──
KST = ZoneInfo("Asia/Seoul")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


async def _profile_cache_periodic_sync_loop() -> None:
    """캐시된 프로필을 주기적으로 DB와 동기화 (이벤트 루프 블로킹 방지: to_thread)."""
    while True:
        try:
            await asyncio.sleep(120)
            await asyncio.to_thread(sync_all_cached_profiles_from_db)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("주기적 프로필 캐시 동기화 실패")


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    start_scheduler()
    _cache_task = asyncio.create_task(_profile_cache_periodic_sync_loop())
    try:
        yield
    finally:
        _cache_task.cancel()
        try:
            await _cache_task
        except asyncio.CancelledError:
            pass
        stop_scheduler()


app = FastAPI(
    title="마음 - 시니어 감정 맞춤형 RAG 챗봇",
    version="0.1.0",
    description="60대 이상 독거 어르신을 위한 음성 기반 감정 맞춤형 말벗 챗봇 백엔드",
    lifespan=_app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /chat ───────────────────────────────────────────────────────

_extract_semaphore = asyncio.Semaphore(1)
_EXTRACT_STAGGER_SEC = 8.0


async def _background_extract_low_priority(user_id: str) -> None:
    """
    장기기억 추출: /chat 응답·TTS와 경쟁하지 않도록 지연 + 세마포 1로 리소스 점유 최소화.
    """
    await asyncio.sleep(_EXTRACT_STAGGER_SEC)
    async with _extract_semaphore:
        await asyncio.sleep(0.2)
        try:
            extracted = await extract_memories(user_id=user_id, limit=50)
            if extracted:
                logger.info("백그라운드 기억 추출 완료: %d건", len(extracted))
        except Exception:
            logger.exception("백그라운드 기억 추출 실패")


_sentiment_mod = None

def _sentiment_fns():
    """
    감정 분석 모듈(sentiment_inference.py)의 함수들을 lazy import한다.
    lru_cache 대신 모듈-전역 변수로 캐시 — 예외 시에는 다음 호출에서 재시도.
    """
    global _sentiment_mod
    if _sentiment_mod is not None:
        return _sentiment_mod

    mod_path = Path(__file__).resolve().parent / "sentiment_inference.py"
    src_path = Path(__file__).resolve().parent.parent / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if not mod_path.is_file():
        raise FileNotFoundError(f"sentiment inference module not found: {mod_path}")
    spec = importlib.util.spec_from_file_location("app_sentiment_inference", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec from: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fns = (
        mod.analyze_audio_sentiment,
        mod.analyze_text_sentiment,
        mod.apply_soft_voting,
        mod.apply_soft_voting_text_only,
    )
    _sentiment_mod = fns
    return fns


# ── 백그라운드: HuBERT 음성 감정 분석 → 캐시 저장 ────────────────────

async def _background_audio_analysis(request_id: str, audio_bytes: bytes) -> None:
    """HuBERT 추론은 asyncio.to_thread로 격리하여 이벤트 루프 차단 방지."""
    try:
        analyze_audio, _, _, _ = _sentiment_fns()
        result = await asyncio.to_thread(analyze_audio, audio_bytes)
        store_audio_sentiment(request_id, result)
    except Exception:
        logger.exception("Request ID [%s] - 백그라운드 음성 감정 분석 실패", request_id)


# ── 백그라운드: 섀도우 정산 파이프라인 ────────────────────────────────

async def shadow_settlement_process(
    user_id: str,
    text: str,
    request_id: str | None,
    user_chat_id: str | None,
) -> None:
    """
    BackgroundTask (async): LLM 응답 후 비동기 감정 정산.

    a) 캐시에서 음성 감정 결과 폴링 (0.5s × 4회 = 최대 2초)
    b) 텍스트 감정 분석 + Soft Voting
    c) 프로필 감정/정서 갱신 (emotion_session 미사용)
    d) 사용자 발화 레코드에 감정 메타 후행 업데이트
    e) 캐시 TTL 정리
    """
    try:
        t0 = time.monotonic()
        now_kst = datetime.now(KST)

        # ── (a) 캐시 폴링: 음성 감정 결과 ──
        audio_result = None
        if request_id:
            for attempt in range(4):
                audio_result = pop_audio_sentiment(request_id)
                if audio_result is not None:
                    logger.info(
                        "Request ID [%s] - 캐시 히트 (attempt %d, top=%s)",
                        request_id, attempt + 1, audio_result.get("top_emotion"),
                    )
                    break
                await asyncio.sleep(0.5)
            if audio_result is None:
                logger.info(
                    "Request ID [%s] - 음성 분석 결과 없음, 텍스트 전용 폴백", request_id,
                )

        # ── (b) 텍스트 감정 분석 + Soft Voting (CPU-bound → thread) ──
        sentiment = None
        try:
            _, analyze_text, _, apply_sv_text_only = _sentiment_fns()
            sentiment = await asyncio.to_thread(
                run_sentiment_pipeline, text, audio_result, analyze_text, apply_sv_text_only,
            )
        except Exception:
            logger.exception("Request ID [%s] - 감정 분석 파이프라인 실패", request_id)

        # ── (c) 프로필 감정·케어 지표 (emotion_session 없음) ──
        if sentiment:
            try:
                update_user_emotion(user_id, sentiment["final_emotion"])
            except Exception:
                logger.exception("Request ID [%s] - 프로필 감정 갱신 실패", request_id)

        # ── (d) 사용자 발화 레코드에 감정 메타 후행 업데이트 ──
        if sentiment and user_chat_id:
            try:
                update_chat_emotion_fields(
                    chat_id=user_chat_id,
                    emotion_label=sentiment["final_emotion"],
                    emotion_score=sentiment["emotion_score"],
                    incongruence_score=sentiment["incongruence_score"],
                    text_emotion=sentiment.get("text_emotion"),
                    voice_emotion=sentiment.get("voice_emotion"),
                    soft_voting_log=sentiment.get("soft_voting_log"),
                )
            except Exception:
                logger.exception("Request ID [%s] - 발화 감정 메타 업데이트 실패", request_id)

        # ── (e) 캐시 TTL 정리 ──
        cleanup_stale_cache()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Shadow 정산 완료: user=%s, request_id=%s, emotion=%s, has_audio=%s, elapsed=%dms",
            user_id, request_id,
            sentiment["final_emotion"] if sentiment else "N/A",
            audio_result is not None,
            elapsed_ms,
        )
    except Exception:
        logger.exception(
            "Shadow 정산 실패 (메인 응답은 정상 전달됨): user=%s, request_id=%s",
            user_id, request_id,
        )


async def _persist_chat_and_shadow(req: ChatRequest, assistant_answer: str) -> None:
    """
    응답 반환 후: 사용자/어시스턴트 메시지 저장 → 섀도우 정산.
    DB 오류는 로그만 남기며, 이미 응답·TTS는 완료된 상태다.
    """
    try:
        ensure_senior_profile(req.user_id)
    except Exception:
        logger.exception("프로필 보장 실패 (백그라운드): user=%s", req.user_id)

    user_chat_id: str | None = None
    try:
        user_msg = save_chat_message(req.user_id, role="user", content=req.message)
        user_chat_id = str(user_msg.get("chat_id", ""))
    except Exception:
        logger.exception("사용자 메시지 저장 실패 (백그라운드): user=%s", req.user_id)

    try:
        save_chat_message(req.user_id, role="assistant", content=assistant_answer)
    except Exception:
        logger.exception("어시스턴트 메시지 저장 실패 (백그라운드): user=%s", req.user_id)

    await shadow_settlement_process(
        req.user_id, req.message, req.request_id, user_chat_id,
    )

    # 장기기억 추출: /chat 응답 후 비동기·저우선 (별도 태스크로 블로킹 없음)
    asyncio.create_task(_background_extract_low_priority(req.user_id))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, bg: BackgroundTasks):
    """
    [섀도우 파이프라인] empathetic_chat(LLM)만 크리티컬 경로.

    1. LLM 답변 생성
    2. care_level 등은 peek_senior_profile_cache(메모리)만 사용 — DB 대기 없음
    3. BackgroundTask: 저장 + 섀도우 정산 + (저우선) 장기기억 추출

    [선택] TTS 스트리밍: Google Cloud TTS는 일반적으로 전체 합성 후 MP3 반환.
    실시간 부분 TTS를 원하면 (a) 문장 단위로 LLM 스트리밍 + 문장별 /tts 호출,
    또는 (b) Web Speech API로 클라이언트 즉시 재생, 또는 (c) SSE로 청크 텍스트를
    먼저 보내고 TTS를 큐잉하는 구조를 별도 검토.
    """
    t0 = time.monotonic()

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
        logger.exception("RAG 챗봇 응답 생성 실패")
        raise HTTPException(status_code=502, detail="답변 생성에 실패했어요. 잠시 후 다시 시도해 주세요.")

    # 메모리 캐시만 — DB 라운드트립 없음 (미스 시 기본값)
    cached = peek_senior_profile_cache(req.user_id)
    if cached:
        result.setdefault("care_level", cached.get("care_level", "NORMAL"))
        result.setdefault("distress_score", cached.get("distress_score", 0))
        result.setdefault("distress_score_daily", cached.get("distress_score_daily", 50))
    else:
        result.setdefault("care_level", "NORMAL")
        result.setdefault("distress_score", 0)
        result.setdefault("distress_score_daily", 50)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "/chat 즉시 응답: user=%s, request_id=%s, elapsed=%dms",
        req.user_id, req.request_id, elapsed_ms,
    )

    bg.add_task(_persist_chat_and_shadow, req, result.get("answer", ""))

    return ChatResponse(**result)


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


class TtsRequest(BaseModel):
    text: str


@app.post("/extract-memory-kobart", response_model=ExtractionResult)
async def extract_memory_kobart_endpoint(req: KobartExtractRequest):
    """
    KoBART 기반 장기기억 추출 실험 엔드포인트.
    source_mode=extracted 로 주면 short_term_history(is_extracted=true)를 기준으로 비교한다.
    """
    try:
        result = await extract_memories_kobart(
            user_id=req.user_id,
            limit=req.limit,
            save_to_db=req.save_to_db,
            mark_extracted_after_save=req.mark_extracted_after_save,
            compare_with_llm=req.compare_with_llm,
            store_comparison_stats=req.store_comparison_stats,
            source_mode=req.source_mode,
            model_id=req.model_id,
        )
        return ExtractionResult.model_validate(result)
    except Exception:
        logger.exception("KoBART 기억 추출 파이프라인 실패")
        raise HTTPException(
            status_code=502,
            detail="KoBART 기억 추출에 실패했어요. 모델/의존성/환경을 확인해 주세요.",
        )


# ── POST /tts ─────────────────────────────────────────────────────────

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


# ── POST /stt (섀도우 파이프라인: STT만 await → 즉시 반환) ────────────


@app.post("/stt")
async def stt_endpoint(
    bg: BackgroundTasks,
    audio: UploadFile = File(..., description="녹음 파일 (권장: WebM/Opus 또는 WAV PCM)"),
    user_id: str = Form("default_user"),
    language_code: str = Form("ko-KR"),
    sample_rate_hertz: int = Form(48000),
):
    """
    [섀도우 파이프라인] Google STT → transcript + request_id 즉시 반환.
    HuBERT 음성 감정 분석은 BackgroundTask로 분리.
    """
    t0 = time.monotonic()
    body = await audio.read()
    logger.info("/stt 요청 수신: user=%s, bytes=%d, content_type=%s",
                user_id, len(body), audio.content_type)

    if not body:
        raise HTTPException(status_code=400, detail="녹음 데이터가 비어 있습니다.")

    # ── [1] 메모리 전용 request_id (emotion_session DB 없음) ──
    try:
        request_id = generate_request_id(user_id)
    except Exception:
        logger.exception("Request ID 생성 실패 — 폴백 ID 사용")
        request_id = f"{user_id}:fallback:{int(time.time())}"

    # ── [2] Google STT (await — 이것만 기다린다) ──
    try:
        text = await asyncio.to_thread(
            transcribe_bytes,
            body,
            content_type=audio.content_type,
            language_code=language_code,
            sample_rate_hertz=sample_rate_hertz,
        )
    except gcp_exceptions.GoogleAPIError as e:
        msg = getattr(e, "message", None) or str(e)
        logger.error("Speech-to-Text API 오류: %s", msg)
        raise HTTPException(status_code=502, detail=f"음성 인식 서비스 오류입니다. ({msg})")
    except Exception:
        logger.exception("음성 인식 실패")
        raise HTTPException(status_code=502, detail="음성 인식에 실패했어요. 잠시 후 다시 시도해 주세요.")

    # ── [3] 즉시 응답 (감정 분석 기다리지 않음) ──
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "/stt 즉시 응답: user=%s, request_id=%s, text_len=%d, elapsed=%dms",
        user_id, request_id, len(text or ""), elapsed_ms,
    )

    # ── [4] 백그라운드: HuBERT 음성 감정 분석 → 캐시 저장 ──
    bg.add_task(_background_audio_analysis, request_id, body)

    return {
        "text": text,
        "request_id": request_id,
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
_KST_OFFSET = timezone(timedelta(hours=9))
SEOUL_STN = 108

@app.get("/weather")
async def weather_endpoint():
    """서울 현재 날씨 (기상청 API) — 기온, 습도, 강수량."""
    now_kst = datetime.now(_KST_OFFSET)
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


@app.get("/test.html")
async def test_page():
    return FileResponse(STATIC_DIR / "test.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
