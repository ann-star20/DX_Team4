"""
Supabase REST API (PostgREST) 기반 DB 클라이언트.
IPv6 전용 DB에서도 문제없이 동작한다.
"""

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse

from supabase import Client, create_client

from app.config import get_settings

# ── 전역 타임존 룰: 모든 시간은 KST 기준 ──
KST = ZoneInfo("Asia/Seoul")

logger = logging.getLogger(__name__)

_client: Client | None = None


def _disable_proxy_env_for_backend() -> None:
    """
    로컬 프록시(HTTP_PROXY/HTTPS_PROXY)가 켜져 있으면
    Supabase PostgREST 요청이 403 ProxyError로 실패할 수 있어 비활성화한다.
    """
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)


def get_supabase() -> Client:
    global _client
    if _client is None:
        _disable_proxy_env_for_backend()
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


# ── Short-term History CRUD ──────────────────────────────────────────

def save_chat_message(user_id: str, role: str, content: str) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "role": role,
        "content": content,
    }
    result = sb.table("short_term_history").insert(row).execute()
    return result.data[0] if result.data else row


def get_unextracted_conversations(
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    sb = get_supabase()
    query = (
        sb.table("short_term_history")
        .select("*")
        .eq("is_extracted", False)
        .order("chat_time", desc=False)
        .limit(limit)
    )
    if user_id:
        query = query.eq("user_id", user_id)
    result = query.execute()
    return result.data or []


def get_extracted_conversations(
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    sb = get_supabase()
    query = (
        sb.table("short_term_history")
        .select("*")
        .eq("is_extracted", True)
        .order("chat_time", desc=False)
        .limit(limit)
    )
    if user_id:
        query = query.eq("user_id", user_id)
    result = query.execute()
    return result.data or []


def mark_as_extracted(chat_ids: list[str]) -> None:
    sb = get_supabase()
    sb.table("short_term_history").update(
        {"is_extracted": True}
    ).in_("chat_id", chat_ids).execute()


# ── Long-term Memory CRUD ────────────────────────────────────────────

def save_long_term_memory(
    user_id: str,
    memory_date: str,
    content: str,
    embedding: list[float],
    metadata: dict,
) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "memory_date": memory_date,
        "content": content,
        "embedding": embedding,
        "metadata": metadata,
    }
    result = sb.table("long_term_memory").insert(row).execute()
    return result.data[0] if result.data else row


def list_recent_memories(user_id: str, limit: int = 300) -> list[dict]:
    """회상 후보 선별 등: 최근 장기기억을 memory_date 내림차순으로 가져온다."""
    sb = get_supabase()
    result = (
        sb.table("long_term_memory")
        .select("id, content, memory_date, metadata")
        .eq("user_id", user_id)
        .order("memory_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Senior Profile CRUD ──────────────────────────────────────────────
# 메모리 캐시: get_senior_profile은 캐시 우선 → 미스 시 DB. /chat 크리티컬 경로는 peek만 사용.

_user_profile_cache: dict[str, dict[str, Any]] = {}
_profile_cache_lock = threading.RLock()


def _fetch_senior_profile_from_db(user_id: str) -> dict | None:
    sb = get_supabase()
    result = sb.table("senior_profile").select("*").eq("user_id", user_id).execute()
    row = result.data[0] if result.data else None
    with _profile_cache_lock:
        if row:
            _user_profile_cache[user_id] = dict(row)
        else:
            _user_profile_cache.pop(user_id, None)
    return row


def get_senior_profile(user_id: str) -> dict | None:
    """캐시 히트 시 DB 호출 없음. 미스 시 DB 조회 후 캐시 적재."""
    with _profile_cache_lock:
        if user_id in _user_profile_cache:
            return dict(_user_profile_cache[user_id])
    return _fetch_senior_profile_from_db(user_id)


def peek_senior_profile_cache(user_id: str) -> dict | None:
    """DB 호출 없이 메모리 캐시만 조회. /chat 즉시 응답용."""
    with _profile_cache_lock:
        if user_id in _user_profile_cache:
            return dict(_user_profile_cache[user_id])
    return None


def refresh_senior_profile_cache(user_id: str) -> dict | None:
    """DB에서 다시 읽어 캐시 갱신 (섀도우/백그라운드 동기화용)."""
    return _fetch_senior_profile_from_db(user_id)


def sync_all_cached_profiles_from_db() -> None:
    """캐시에 등록된 user_id 전원 DB와 동기화. asyncio.to_thread로 호출 권장."""
    with _profile_cache_lock:
        ids = list(_user_profile_cache.keys())
    for uid in ids:
        try:
            _fetch_senior_profile_from_db(uid)
        except Exception:
            logger.exception("프로필 캐시 동기화 실패: user=%s", uid)


def upsert_senior_profile(user_id: str, data: dict) -> dict:
    sb = get_supabase()
    row = {"user_id": user_id, **data}
    result = sb.table("senior_profile").upsert(row).execute()
    out = result.data[0] if result.data else row
    with _profile_cache_lock:
        merged = dict(_user_profile_cache.get(user_id, {}))
        merged.update(out)
        _user_profile_cache[user_id] = merged
    return out


def ensure_senior_profile(user_id: str) -> dict:
    """senior_profile에 해당 user_id 행이 없으면 기본값으로 생성한다 (FK 위반 방지)."""
    profile = get_senior_profile(user_id)
    if profile is not None:
        return profile
    return upsert_senior_profile(user_id, {
        "distress_score": 0,
        "distress_score_daily": 50,
        "care_level": "NORMAL",
        "current_session_score": 0,
    })


def update_profile_realtime_fields(
    user_id: str,
    *,
    now_kst: datetime,
    current_mood_state: str | None = None,
) -> dict:
    """
    발화 시 실시간 필드만 갱신한다.
    - distress_score / distress_score_daily / care_level / incongruence_score는 갱신하지 않는다.
    """
    payload = {
        "last_message_at": now_kst.isoformat(),
        "updated_at": now_kst.isoformat(),
    }
    if current_mood_state:
        payload["current_mood_state"] = current_mood_state
    try:
        return upsert_senior_profile(user_id, payload)
    except Exception:
        # 스키마가 mood_state를 쓰는 환경 호환
        payload.pop("current_mood_state", None)
        if current_mood_state:
            payload["mood_state"] = current_mood_state
        return upsert_senior_profile(user_id, payload)


# ── Daily Care Report CRUD ───────────────────────────────────────────

def get_daily_report(user_id: str, report_date: str) -> dict | None:
    sb = get_supabase()
    result = (
        sb.table("daily_care_report")
        .select("*")
        .eq("user_id", user_id)
        .eq("report_date", report_date)
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_daily_report(user_id: str, report_date: str, data: dict) -> dict:
    sb = get_supabase()
    existing = get_daily_report(user_id, report_date)
    if existing:
        result = (
            sb.table("daily_care_report")
            .update(data)
            .eq("report_id", existing["report_id"])
            .execute()
        )
    else:
        row = {"user_id": user_id, "report_date": report_date, **data}
        result = sb.table("daily_care_report").insert(row).execute()
    return result.data[0] if result.data else data


def get_weekly_reports(user_id: str, limit: int = 7) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("daily_care_report")
        .select("*")
        .eq("user_id", user_id)
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Daily Sentiment Stats CRUD ───────────────────────────────────────

def get_daily_sentiment_stat(user_id: str, report_date: str) -> dict | None:
    sb = get_supabase()
    result = (
        sb.table("daily_sentiment_stats")
        .select("*")
        .eq("user_id", user_id)
        .eq("report_date", report_date)
        .order("stat_id", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_daily_sentiment_stat(user_id: str, report_date: str, data: dict) -> dict:
    sb = get_supabase()
    existing = get_daily_sentiment_stat(user_id, report_date)
    if existing:
        result = (
            sb.table("daily_sentiment_stats")
            .update(data)
            .eq("stat_id", existing["stat_id"])
            .execute()
        )
    else:
        row = {"user_id": user_id, "report_date": report_date, **data}
        result = sb.table("daily_sentiment_stats").insert(row).execute()
    return result.data[0] if result.data else data


def list_daily_sentiment_stats(user_id: str, limit: int = 14) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("daily_sentiment_stats")
        .select("*")
        .eq("user_id", user_id)
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Intervention Action Log CRUD ─────────────────────────────────────

def log_intervention(
    user_id: str,
    action_type: str,
    trigger_emotion: str | None = None,
    suggested_content: str | None = None,
) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "action_type": action_type,
        "trigger_emotion": trigger_emotion,
        "suggested_content": suggested_content,
    }
    result = sb.table("intervention_action_log").insert(row).execute()
    return result.data[0] if result.data else row


def update_intervention_response(action_id: int, user_response: str) -> dict:
    sb = get_supabase()
    result = (
        sb.table("intervention_action_log")
        .update({"user_response": user_response})
        .eq("action_id", action_id)
        .execute()
    )
    return result.data[0] if result.data else {}


def get_recent_interventions(user_id: str, limit: int = 10) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("intervention_action_log")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Emotion Session CRUD ─────────────────────────────────────────────

def get_open_session(user_id: str) -> dict | None:
    """[세션 조회] user_id가 일치하고 is_closed=False인 가장 최근 세션을 반환한다."""
    sb = get_supabase()
    result = (
        sb.table("emotion_session")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_closed", False)
        .order("start_time", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def create_emotion_session(
    user_id: str,
    now_kst: datetime,
    raw_score: float,
    incongruence: float = 0.0,
    utterance_count: int = 1,
) -> dict:
    """[세션 생성] 새로운 emotion_session 행을 만든다. 시간은 KST ISO 문자열로 저장."""
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "start_time": now_kst.isoformat(),
        "last_activity_at": now_kst.isoformat(),
        "is_closed": False,
        "raw_score_sum": float(max(-10, min(10, raw_score))),
        "total_incongruence_sum": float(incongruence),
        "total_response_time_ms": 0,
        "utterance_count": utterance_count,
    }
    result = sb.table("emotion_session").insert(row).execute()
    return result.data[0] if result.data else row


def close_emotion_session(session_id: str) -> None:
    """[세션 종료] 기존 세션의 is_closed를 True로 업데이트한다."""
    sb = get_supabase()
    sb.table("emotion_session").update(
        {"is_closed": True}
    ).eq("session_id", session_id).execute()


def update_emotion_session(
    session_id: str,
    now_kst: datetime,
    raw_score_sum: float,
    incongruence_delta: float,
    response_time_ms: int,
) -> dict:
    """[세션 업데이트] 기존 세션에 이번 턴의 수치를 누적한다."""
    sb = get_supabase()
    result = (
        sb.table("emotion_session")
        .update({
            "last_activity_at": now_kst.isoformat(),
            # [Clamp] raw_score_sum은 -10 ~ 10 범위로 제한
            "raw_score_sum": float(max(-10, min(10, raw_score_sum))),
            "total_incongruence_sum": float(incongruence_delta),
            "total_response_time_ms": int(response_time_ms),
            "utterance_count": None,  # placeholder, 실제로는 RPC/직접 +1
        })
        .eq("session_id", session_id)
        .execute()
    )
    return result.data[0] if result.data else {}


def increment_session_fields(
    session_id: str,
    now_kst: datetime,
    new_raw_score_sum: float,
    new_incongruence_sum: float,
    new_response_time_ms: int,
    new_utterance_count: int,
) -> dict:
    """[세션 필드 전체 덮어쓰기] 호출자가 계산한 누적값을 그대로 저장한다."""
    sb = get_supabase()
    result = (
        sb.table("emotion_session")
        .update({
            "last_activity_at": now_kst.isoformat(),
            "raw_score_sum": float(max(-10, min(10, new_raw_score_sum))),
            "total_incongruence_sum": float(new_incongruence_sum),
            "total_response_time_ms": int(new_response_time_ms),
            "utterance_count": int(new_utterance_count),
        })
        .eq("session_id", session_id)
        .execute()
    )
    return result.data[0] if result.data else {}


# ── Short-term History 확장 (감정 메타 저장) ──────────────────────────

def save_chat_message_with_emotion(
    user_id: str,
    role: str,
    content: str,
    *,
    session_id: str | None = None,
    emotion_label: str | None = None,
    emotion_score: float | None = None,
    incongruence_score: float | None = None,
    text_emotion: dict | None = None,
    voice_emotion: dict | None = None,
    soft_voting_log: dict | None = None,
) -> dict:
    """[발화 저장 + 감정 메타] short_term_history에 감정 분석 결과까지 한 번에 저장한다."""
    sb = get_supabase()
    row: dict[str, Any] = {
        "user_id": user_id,
        "role": role,
        "content": content,
    }
    if session_id is not None:
        row["session_id"] = session_id
    if emotion_label is not None:
        row["emotion_label"] = emotion_label
    if emotion_score is not None:
        row["emotion_score"] = float(emotion_score)
    if incongruence_score is not None:
        row["incongruence_score"] = float(incongruence_score)
    if text_emotion is not None:
        row["text_emotion"] = text_emotion
    if voice_emotion is not None:
        row["voice_emotion"] = voice_emotion
    if soft_voting_log is not None:
        row["soft_voting_log"] = soft_voting_log
    result = sb.table("short_term_history").insert(row).execute()
    return result.data[0] if result.data else row


def update_chat_emotion_fields(
    chat_id: str,
    *,
    session_id: str | None = None,
    emotion_label: str | None = None,
    emotion_score: float | None = None,
    incongruence_score: float | None = None,
    text_emotion: dict | None = None,
    voice_emotion: dict | None = None,
    soft_voting_log: dict | None = None,
) -> dict:
    """[Shadow 후처리] 이미 저장된 사용자 발화 레코드에 감정 분석 결과를 후행 업데이트한다."""
    sb = get_supabase()
    updates: dict[str, Any] = {}
    if session_id is not None:
        updates["session_id"] = session_id
    if emotion_label is not None:
        updates["emotion_label"] = emotion_label
    if emotion_score is not None:
        updates["emotion_score"] = float(emotion_score)
    if incongruence_score is not None:
        updates["incongruence_score"] = float(incongruence_score)
    if text_emotion is not None:
        updates["text_emotion"] = text_emotion
    if voice_emotion is not None:
        updates["voice_emotion"] = voice_emotion
    if soft_voting_log is not None:
        updates["soft_voting_log"] = soft_voting_log
    if not updates:
        return {}
    result = (
        sb.table("short_term_history")
        .update(updates)
        .eq("chat_id", chat_id)
        .execute()
    )
    return result.data[0] if result.data else {}


def get_last_assistant_chat_time(user_id: str) -> datetime | None:
    """[응답 속도 계산용] 가장 최근 assistant 발화의 chat_time을 KST로 반환한다."""
    sb = get_supabase()
    result = (
        sb.table("short_term_history")
        .select("chat_time")
        .eq("user_id", user_id)
        .eq("role", "assistant")
        .order("chat_time", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    raw = result.data[0].get("chat_time")
    if raw is None:
        return None
    # [타임존 룰] DB에서 가져온 시간을 KST로 변환 (Python 3.10 호환 파서)
    dt = isoparse(raw)
    return dt.astimezone(KST)


# ── Senior Profile 정서 피로도 갱신 ──────────────────────────────────

def update_distress_and_care_level(
    user_id: str,
    emotion_delta: float,
    now_kst: datetime,
) -> dict:
    """[정서 피로도 갱신] emotion_delta를 distress_score에 더하고 care_level을 재판별한다."""
    profile = get_senior_profile(user_id)
    if profile is None:
        profile = {
            "distress_score": 0,
            "care_level": "NORMAL",
        }

    distress = float(profile.get("distress_score") or 0)
    # [명세서 §4] 세션에서 도출된 감정 점수를 distress_score에 합산
    distress = max(-100, min(100, distress + emotion_delta))

    # [Care Level 판정] 80 이상 DANGER, 41 이상 WARNING, 나머지 NORMAL
    if distress >= 80:
        care_level = "DANGER"
    elif distress >= 41:
        care_level = "WARNING"
    else:
        care_level = "NORMAL"

    update_data = {
        "distress_score": round(distress, 2),
        "care_level": care_level,
        "last_message_at": now_kst.isoformat(),
        "updated_at": now_kst.isoformat(),
    }
    upsert_senior_profile(user_id, update_data)
    return {**update_data, "user_id": user_id}


def update_profile_incongruence_ema(
    user_id: str,
    session_incongruence: float,
    now_kst: datetime,
    *,
    alpha: float = 0.2,
) -> dict:
    """
    senior_profile.incongruence_score를 EMA로 갱신한다.
    - 세션 종료 시점 요약 지표 업데이트 용도
    - new = old*(1-alpha) + current*alpha
    """
    profile = get_senior_profile(user_id) or {}
    old_value = profile.get("incongruence_score")
    if old_value is None:
        new_value = float(session_incongruence)
    else:
        old = float(old_value or 0.0)
        new_value = ((1.0 - alpha) * old) + (alpha * float(session_incongruence))

    update_data = {
        "incongruence_score": round(float(new_value), 4),
        "updated_at": now_kst.isoformat(),
    }
    upsert_senior_profile(user_id, update_data)
    return {**update_data, "user_id": user_id}


# ── Hybrid Search (Vector + GIN Metadata via RPC) ────────────────────

@dataclass
class HybridSearchResult:
    id: str
    user_id: str
    memory_date: str | None
    content: str
    metadata: dict
    vector_score: float
    metadata_score: float
    recency_score: float
    importance_score: float
    final_score: float


def hybrid_search(
    user_id: str,
    query_embedding: list[float],
    *,
    category: str | None = None,
    emotion: str | None = None,
    entities: list[str] | None = None,
    top_k: int = 15,
    vector_weight: float = 0.65,
    metadata_weight: float = 0.10,
    recency_weight: float = 0.15,
    importance_weight: float = 0.10,
) -> list[HybridSearchResult]:
    """Supabase RPC로 하이브리드 검색을 실행한다. ``long_term_memory``는 ``user_id``로만 필터된다."""
    sb = get_supabase()

    params: dict[str, Any] = {
        "p_user_id": user_id,
        "p_query_embedding": query_embedding,
        "p_top_k": top_k,
        "p_vector_weight": vector_weight,
        "p_metadata_weight": metadata_weight,
        "p_recency_weight": recency_weight,
        "p_importance_weight": importance_weight,
    }
    if category:
        params["p_category"] = category
    if emotion:
        params["p_emotion"] = emotion
    if entities:
        params["p_entities"] = entities

    result = sb.rpc("hybrid_search", params).execute()

    return [
        HybridSearchResult(
            id=r["id"],
            user_id=r["user_id"],
            memory_date=r.get("memory_date"),
            content=r["content"],
            metadata=r.get("metadata", {}),
            vector_score=float(r.get("vector_score", 0)),
            metadata_score=float(r.get("metadata_score", 0)),
            recency_score=float(r.get("recency_score", 0)),
            importance_score=float(r.get("importance_score", 0)),
            final_score=float(r.get("final_score", 0)),
        )
        for r in (result.data or [])
    ]


def check_duplicate_memory(
    user_id: str,
    content: str,
    query_embedding: list[float] | None = None,
    threshold: float = 0.80,
) -> bool:
    """새 기억이 기존 기억과 텍스트 수준에서 중복인지 확인한다.

    text-embedding-004 한국어 임베딩은 짧은 문장끼리 코사인 유사도가
    0.91~0.93 밴드에 몰리기 때문에 벡터 기반 중복 탐지가 부정확하다.
    텍스트 SequenceMatcher ratio >= threshold(0.80)이면 중복으로 판정.
    """
    from difflib import SequenceMatcher

    sb = get_supabase()
    result = (
        sb.table("long_term_memory")
        .select("id, content")
        .eq("user_id", user_id)
        .execute()
    )

    for r in (result.data or []):
        existing = r.get("content", "")
        ratio = SequenceMatcher(None, content, existing).ratio()
        if ratio >= threshold:
            logger.info(
                "중복 기억 차단 (text_ratio=%.3f): 기존='%s' vs 신규='%s'",
                ratio, existing[:60], content[:60],
            )
            return True
    return False
