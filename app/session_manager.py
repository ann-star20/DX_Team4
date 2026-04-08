"""
세션 관리 및 감정 통합 모듈 — 비동기 섀도우 파이프라인 (shadow.md).

책임:
  1. request_id 기반 음성 감정 캐시 (audio_sentiment_cache)
  2. emotion_session 테이블 기반 5분 세션 관리 (handle_emotion_session)
  3. 텍스트 + 음성 감정 Soft Voting 통합 (run_sentiment_pipeline)
  4. 정서 피로도(senior_profile.distress_score) 갱신 호출

흐름:
  /stt → STT 즉시 반환 + BackgroundTask(HuBERT → 캐시)
  /chat → LLM 즉시 반환 + BackgroundTask(shadow 정산: 캐시 폴링 → 텍스트 감정 → Soft Voting → 세션/DB)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse

from app.database import (
    get_open_session,
    create_emotion_session,
    close_emotion_session,
    increment_session_fields,
    get_last_assistant_chat_time,
    update_distress_and_care_level,
    update_profile_incongruence_ema,
    save_chat_message_with_emotion,
    ensure_senior_profile,
    get_senior_profile,
    upsert_senior_profile,
    update_profile_realtime_fields,
)
from app.emotion import get_emotion_score

logger = logging.getLogger(__name__)

# ── 전역 타임존 룰: 모든 시간은 KST 기준 ──
KST = ZoneInfo("Asia/Seoul")

# ── [명세서 §3] 세션 타임아웃: 5분 ──
SESSION_TIMEOUT = timedelta(minutes=5)

# ── [명세서 §3] 응답 속도 계산 시 시스템 발화 간주 시간: 5초(5000ms) ──
SYSTEM_UTTERANCE_MS = 5000

# ── [불일치 지수 EMA] 세션 누적용(최신 발화 가중치 높게) ──
SESSION_INCONGRUENCE_EMA_ALPHA = 0.6

# ── [프로필 불일치 EMA] 세션 종료 요약 반영용 ──
PROFILE_INCONGRUENCE_EMA_ALPHA = 0.2

# ── [불일치 레벨] 위험/주의/정상 ──
INCONGRUENCE_RISK_THRESHOLD = 4.0
INCONGRUENCE_WARN_THRESHOLD = 2.0

DISTRESS_MIN = -100.0
DISTRESS_MAX = 100.0
DAILY_MIN = 0.0
DAILY_MAX = 100.0

# ============================================================
# 음성 감정 캐시 (request_id 기반, 동시 접속 격리)
# ============================================================

_audio_cache: dict[str, dict[str, Any]] = {}
_audio_cache_lock = threading.Lock()
CACHE_TTL_MINUTES = 10


def generate_request_id(user_id: str) -> str:
    """
    /stt에서 호출. DB 없이 순수 메모리 — 타임스탬프(ms) + UUID 조각으로 유일성 보장.
    형식: {user_id}:{ts_ms}:{uuid12}
    """
    ts_ms = int(time.time() * 1000)
    rid = f"{user_id}:{ts_ms}:{uuid.uuid4().hex[:12]}"
    logger.info("Request ID 생성(메모리): %s", rid)
    return rid


def store_audio_sentiment(request_id: str, result: dict) -> None:
    """HuBERT 분석 완료 후 request_id를 키로 캐시에 저장한다."""
    with _audio_cache_lock:
        _audio_cache[request_id] = {
            "result": result,
            "stored_at": datetime.now(KST),
        }
    logger.info(
        "Request ID [%s] - 음성 분석 완료 및 캐시 저장됨 (top=%s)",
        request_id, result.get("top_emotion"),
    )


def pop_audio_sentiment(request_id: str) -> dict | None:
    """캐시에서 request_id의 결과를 .pop()으로 꺼낸다. 없으면 None."""
    with _audio_cache_lock:
        entry = _audio_cache.pop(request_id, None)
    return entry["result"] if entry else None


def cleanup_stale_cache() -> None:
    """10분 이상 된 캐시 항목을 정리한다 (메모리 누수 방지)."""
    now = datetime.now(KST)
    cutoff = timedelta(minutes=CACHE_TTL_MINUTES)
    with _audio_cache_lock:
        stale = [k for k, v in _audio_cache.items() if (now - v["stored_at"]) > cutoff]
        for k in stale:
            del _audio_cache[k]
    if stale:
        logger.info("캐시 TTL 정리: %d건 삭제", len(stale))


# ============================================================
# 세션 관리 핵심 알고리즘
# ============================================================

def handle_emotion_session(
    user_id: str,
    emotion_label: str,
    emotion_score: float,
    incongruence: float,
) -> dict:
    """
    [명세서 §3] emotion_session 테이블을 관리한다.

    Args:
        user_id: 사용자 ID
        emotion_label: Soft Voting 후 최종 감정 라벨
        emotion_score: 감정 → 점수 변환값 (emotion.EMOTION_SCORES 기반)
        incongruence: 이번 턴의 정서 불일치 지수

    Returns:
        {"session_id": str, "is_new": bool, "raw_score_sum": float,
         "response_time_ms": int, "distress_update": dict}
    """
    now_kst = datetime.now(KST)
    mood_state = emotion_label

    # ── [상태 확인] is_closed=False인 세션 조회 ──
    session = get_open_session(user_id)

    if session is None:
        # ── [케이스 A-1: 세션이 아예 없음] 새 세션 생성 ──
        new_session = create_emotion_session(
            user_id=user_id,
            now_kst=now_kst,
            raw_score=emotion_score,
            incongruence=incongruence,
        )
        session_id = new_session.get("session_id", "")
        logger.info("새 세션 생성 (첫 세션): user=%s, session=%s", user_id, session_id)

        # Step 1: 발화 시 실시간 필드만 갱신 (점수/케어레벨 누적 금지)
        distress_update = update_profile_realtime_fields(
            user_id=user_id,
            now_kst=now_kst,
            current_mood_state=mood_state,
        )
        return {
            "session_id": session_id,
            "is_new": True,
            "raw_score_sum": max(-10, min(10, emotion_score)),
            "response_time_ms": 0,
            "distress_update": distress_update,
        }

    # ── 세션의 last_activity_at을 KST로 변환하여 5분 초과 여부 판정 ──
    last_activity_raw = session.get("last_activity_at")
    if last_activity_raw:
        # [타임존 룰] DB에서 꺼낸 시간을 KST로 변환
        # Python 3.10의 fromisoformat은 소수점 자릿수가 6이 아닌 경우 실패하므로 dateutil 사용
        last_activity = isoparse(last_activity_raw).astimezone(KST)
    else:
        last_activity = now_kst - SESSION_TIMEOUT - timedelta(seconds=1)

    elapsed = now_kst - last_activity

    if elapsed > SESSION_TIMEOUT:
        # ── [케이스 A-2: 5분 초과 → 기존 세션 닫고 새 세션 생성] ──
        old_session_id = session["session_id"]
        # 세션 종료 시점: 세션 요약 불일치 지수를 senior_profile EMA로 반영
        old_incongruence = float(session.get("total_incongruence_sum") or 0.0)
        update_profile_incongruence_ema(
            user_id=user_id,
            session_incongruence=old_incongruence,
            now_kst=now_kst,
            alpha=PROFILE_INCONGRUENCE_EMA_ALPHA,
        )
        close_emotion_session(old_session_id)
        logger.info(
            "세션 종료 (5분 초과): user=%s, session=%s, elapsed=%s",
            user_id, old_session_id, elapsed,
        )

        new_session = create_emotion_session(
            user_id=user_id,
            now_kst=now_kst,
            raw_score=emotion_score,
            incongruence=incongruence,
        )
        session_id = new_session.get("session_id", "")
        logger.info("새 세션 생성 (타임아웃): user=%s, session=%s", user_id, session_id)

        # Step 2: 닫힌 세션은 프로필로 정산
        settle_session_to_profile(session, now_kst)

        # Step 1: 발화 시 실시간 필드만 갱신
        distress_update = update_profile_realtime_fields(
            user_id=user_id,
            now_kst=now_kst,
            current_mood_state=mood_state,
        )
        return {
            "session_id": session_id,
            "is_new": True,
            "raw_score_sum": max(-10, min(10, emotion_score)),
            "response_time_ms": 0,
            "distress_update": distress_update,
        }

    # ── [케이스 B: 기존 세션 유지 (5분 이내)] ──
    session_id = session["session_id"]

    # [명세서 §3 응답 속도 계산]
    # 가장 최근 assistant 발화 시간과 현재 시간의 차이에서 5초를 뺀다
    response_time_ms = 0
    last_assistant_time = get_last_assistant_chat_time(user_id)
    if last_assistant_time is not None:
        delta_ms = int((now_kst - last_assistant_time).total_seconds() * 1000)
        # [명세서] 5초(시스템 발화 간주)를 빼고, 음수면 0 처리
        response_time_ms = max(0, delta_ms - SYSTEM_UTTERANCE_MS)

    # [명세서 §3] 기존 세션 필드 누적 업데이트
    prev_raw = float(session.get("raw_score_sum") or 0)
    prev_incong = session.get("total_incongruence_sum")
    prev_resp_ms = int(session.get("total_response_time_ms") or 0)
    prev_count = int(session.get("utterance_count") or 0)

    # [Clamp] raw_score_sum 누적 후 -10 ~ 10으로 제한
    new_raw = max(-10, min(10, prev_raw + emotion_score))
    # [EMA] total_incongruence_sum은 단순 누적합 대신 지수이동평균으로 유지
    if prev_incong is None or prev_count <= 0:
        new_incong = float(incongruence)
    else:
        prev_incong_val = float(prev_incong or 0.0)
        new_incong = (
            SESSION_INCONGRUENCE_EMA_ALPHA * float(incongruence)
            + (1.0 - SESSION_INCONGRUENCE_EMA_ALPHA) * prev_incong_val
        )
    new_resp_ms = prev_resp_ms + response_time_ms
    new_count = prev_count + 1

    increment_session_fields(
        session_id=session_id,
        now_kst=now_kst,
        new_raw_score_sum=new_raw,
        new_incongruence_sum=new_incong,
        new_response_time_ms=new_resp_ms,
        new_utterance_count=new_count,
    )
    logger.info(
        "세션 업데이트: user=%s, session=%s, raw_sum=%.1f, incong=%.2f, resp_ms=%d, count=%d",
        user_id, session_id, new_raw, new_incong, new_resp_ms, new_count,
    )

    # Step 1: 발화 시 실시간 필드만 갱신
    distress_update = update_profile_realtime_fields(
        user_id=user_id,
        now_kst=now_kst,
        current_mood_state=mood_state,
    )
    return {
        "session_id": session_id,
        "is_new": False,
        "raw_score_sum": new_raw,
        "response_time_ms": response_time_ms,
        "distress_update": distress_update,
    }


def settle_session_to_profile(session: dict, now_kst: datetime | None = None) -> dict:
    """
    Step 2: 세션 종료 정산.
    1) raw_score_sum을 distress_score, distress_score_daily에 각각 합산
    2) incongruence_score = (old * 0.5) + session.total_incongruence_sum
    3) distress_score 기준으로 care_level 재판정
    """
    now = now_kst or datetime.now(KST)
    user_id = str(session.get("user_id") or "")
    if not user_id:
        return {}

    ensure_senior_profile(user_id)
    profile = get_senior_profile(user_id) or {}

    raw_sum = float(session.get("raw_score_sum") or 0.0)
    session_incong = float(session.get("total_incongruence_sum") or 0.0)

    old_distress = float(profile.get("distress_score") or 0.0)
    old_daily = float(profile.get("distress_score_daily") or 0.0)
    old_incong = float(profile.get("incongruence_score") or 0.0)

    new_distress = max(DISTRESS_MIN, min(DISTRESS_MAX, old_distress + raw_sum))
    new_daily = max(DAILY_MIN, min(DAILY_MAX, old_daily + raw_sum))
    new_incong = (old_incong * 0.5) + session_incong

    if new_distress >= 80:
        care_level = "DANGER"
    elif new_distress >= 41:
        care_level = "WARNING"
    else:
        care_level = "NORMAL"

    payload = {
        "distress_score": round(new_distress, 4),
        "distress_score_daily": round(new_daily, 4),
        "incongruence_score": round(new_incong, 4),
        "care_level": care_level,
        "updated_at": now.isoformat(),
    }
    upsert_senior_profile(user_id, payload)
    logger.info(
        "세션 정산 완료: user=%s, session=%s, raw=%.2f, incong=%.2f, distress=%.2f, daily=%.2f, care=%s",
        user_id,
        session.get("session_id"),
        raw_sum,
        session_incong,
        new_distress,
        new_daily,
        care_level,
    )
    return payload


# ============================================================
# Step 3 지원: 감정 분석 + Soft Voting 통합 파이프라인
# ============================================================

def run_sentiment_pipeline(
    text: str,
    audio_emotion_cached: dict | None,
    analyze_text_fn,
    apply_soft_voting_text_only_fn,
) -> dict:
    """
    텍스트 감정 분석 + 캐시된 음성 감정과 Soft Voting 수행.

    - 음성 캐시가 있으면: 텍스트 결과(analyze_text_fn)와 음성 결과를 조합하여
      최종 감정 및 불일치 지수를 산출한다.
    - 음성 캐시가 없으면: apply_soft_voting_text_only_fn으로 텍스트 전용 결과를 사용한다.

    Returns:
        {
            "final_emotion": str,
            "emotion_score": float,   # emotion.EMOTION_SCORES 기반
            "incongruence_score": float,
            "text_emotion": dict,
            "voice_emotion": dict | None,
            "soft_voting_log": dict | None,
        }
    """
    def _calculate_expected_score(prob_map: dict[str, float] | None) -> float:
        """감정확률 분포의 기대 감정점수 E[X]를 계산한다."""
        if not prob_map:
            return 0.0
        expected = 0.0
        for emotion, prob in prob_map.items():
            expected += float(prob) * float(get_emotion_score(str(emotion)))
        return float(expected)

    def _incongruence_level(score: float) -> str:
        """불일치 지수를 3단계(위험/주의/정상)로 분류한다."""
        if score >= INCONGRUENCE_RISK_THRESHOLD:
            return "위험"
        if score >= INCONGRUENCE_WARN_THRESHOLD:
            return "주의"
        return "정상"

    # [텍스트 감정 분석] — 항상 실행
    text_result = analyze_text_fn(text)

    if audio_emotion_cached is not None:
        # [Soft Voting: 음성+텍스트]
        # audio_emotion_cached는 /stt에서 analyze_audio_sentiment가 반환한 dict
        # (top_emotion, top_prob 등)
        text_top = text_result.get("top_emotion", "중립")
        audio_top = audio_emotion_cached.get("top_emotion", "중립")

        # [불일치 지수] 확률 기반 기댓값 차이(Expected Value Distance)
        text_probs = text_result.get("probs")
        audio_probs = audio_emotion_cached.get("probs")
        text_expected = _calculate_expected_score(text_probs if isinstance(text_probs, dict) else None)
        audio_expected = _calculate_expected_score(audio_probs if isinstance(audio_probs, dict) else None)
        incongruence = abs(text_expected - audio_expected)
        incongruence_level = _incongruence_level(incongruence)

        # [최종 감정 결정] 음성 분석의 top_prob이 더 높으면 음성 결과 채택, 아니면 텍스트
        text_prob = float(text_result.get("top_prob", 0))
        audio_prob = float(audio_emotion_cached.get("top_prob", 0))
        # 가중치 반영: 음성 결과에 가중치 0.45, 텍스트에 0.55
        if audio_prob * 0.45 >= text_prob * 0.55:
            final_emotion = audio_top
        else:
            final_emotion = text_top

        soft_voting_log = {
            "method": "soft_voting",
            "text_top": text_top,
            "text_prob": text_prob,
            "text_expected_score": text_expected,
            "audio_top": audio_top,
            "audio_prob": audio_prob,
            "audio_expected_score": audio_expected,
            "final": final_emotion,
            "incongruence_score": incongruence,
            "incongruence_level": incongruence_level,
        }
        logger.info(
            "Soft Voting(EVD): text=%s(%.2f, E=%.2f) + audio=%s(%.2f, E=%.2f) → final=%s, incong=%.2f(%s)",
            text_top, text_prob, text_expected,
            audio_top, audio_prob, audio_expected,
            final_emotion, incongruence, incongruence_level,
        )
    else:
        # [텍스트 전용] 음성 결과 없음 → text-only soft voting
        text_only_result = apply_soft_voting_text_only_fn(text)
        final_emotion = text_only_result.get("top_emotion", text_result.get("top_emotion", "중립"))
        incongruence = float(text_only_result.get("incongruence_score", 0))
        incongruence_level = _incongruence_level(incongruence)

        soft_voting_log = {
            "method": "text_only",
            "text_top": text_result.get("top_emotion"),
            "text_prob": text_result.get("top_prob"),
            "final": final_emotion,
            "incongruence_score": incongruence,
            "incongruence_level": incongruence_level,
        }
        logger.info(
            "텍스트 전용 감정 분석: text=%s(%.2f) → final=%s",
            text_result.get("top_emotion"), text_result.get("top_prob", 0),
            final_emotion,
        )

    # [감정 → 점수 변환] emotion.py의 EMOTION_SCORES 매핑 사용
    emotion_score = float(get_emotion_score(final_emotion))

    return {
        "final_emotion": final_emotion,
        "emotion_score": emotion_score,
        "incongruence_score": incongruence,
        "incongruence_level": incongruence_level,
        "text_emotion": text_result,
        "voice_emotion": audio_emotion_cached,
        "soft_voting_log": soft_voting_log,
    }
