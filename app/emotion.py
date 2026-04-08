"""
정서피로도 관리 엔진.
- 감정 → 점수 매핑 (러셀 감정 원형 모델 + 응급 트리아지 기반)
- 세션 분리 (10분 침묵 기준)
- 일간 30% 감쇄 (Decay)
- care_level 판정 (NORMAL / WARNING / DANGER)
- Daily Care Report 자동 갱신
- 행동 개입 피로도 관리
"""

import logging
from datetime import date, datetime, timedelta, timezone

from dateutil.parser import isoparse

from app.database import (
    get_senior_profile,
    upsert_senior_profile,
    get_daily_report,
    upsert_daily_report,
)

logger = logging.getLogger(__name__)

SESSION_GAP = timedelta(minutes=10)
DECAY_RATE = 0.7

DISTRESS_MIN = -30
DISTRESS_MAX = 100
SESSION_SCORE_MIN = -10
SESSION_SCORE_MAX = 10
DAILY_DEFAULT = 50
DAILY_MIN = 0
DAILY_MAX = 100

FATIGUE_DECAY_PER_HOUR = 0.05
FATIGUE_ACCEPT_BUMP = -0.1
FATIGUE_REJECT_BUMP = 0.3
FATIGUE_IGNORE_BUMP = 0.2

EMOTION_SCORES: dict[str, int] = {
    "공포": 6, "fear": 6, "불안": 6,
    "슬픔": 4, "sadness": 4, "우울": 4,
    "분노": 3, "anger": 3, "angry": 3, "화남": 3,
    "혐오": 2, "disgust": 2,
    "놀람": -1, "surprise": -1,
    "중립": 0, "neutral": 0,
    "기쁨": -6, "happiness": -6, "행복": -6, "즐거움": -6,
}

DANGER_EMOTIONS = {"공포", "슬픔", "우울", "불안", "fear", "sadness"}

CARE_THRESHOLDS = [
    (80, "DANGER"),
    (41, "WARNING"),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_care_level(score: float) -> str:
    for threshold, level in CARE_THRESHOLDS:
        if score >= threshold:
            return level
    return "NORMAL"


def compute_effective_distress_from_profile(profile: dict) -> tuple[float, str]:
    """
    저장된 senior_profile로 effective 점수와 care_level을 재계산한다.
    (update_user_emotion 시점의 공식과 동일: distress + clamp(session))
    """
    distress = float(profile.get("distress_score") or 0)
    session = float(profile.get("current_session_score") or 0)
    s = _clamp(session, SESSION_SCORE_MIN, SESSION_SCORE_MAX)
    effective = _clamp(distress + s, DISTRESS_MIN, DISTRESS_MAX)
    return effective, _compute_care_level(effective)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_emotion_score(emotion: str) -> int:
    return EMOTION_SCORES.get(emotion.lower().strip(), 0)


def is_danger_emotion(emotion: str) -> bool:
    return emotion.lower().strip() in DANGER_EMOTIONS


def update_user_emotion(user_id: str, emotion: str) -> dict:
    """
    사용자 발화의 감정을 받아 Senior_Profile + Daily_Care_Report를 업데이트한다.

    Returns:
        distress_score, distress_score_daily, care_level, mood_state 등 포함 dict.
    """
    now = _now_utc()
    today = now.date()
    today_str = today.isoformat()
    emotion_delta = get_emotion_score(emotion)

    profile = get_senior_profile(user_id)

    if profile is None:
        profile = {
            "distress_score": 0,
            "distress_score_daily": DAILY_DEFAULT,
            "care_level": "NORMAL",
            "current_session_score": 0,
            "last_message_at": now.isoformat(),
            "last_decay_date": today_str,
            "mood_state": "중립",
            "recent_action_fatigue": 0.0,
        }

    distress = float(profile.get("distress_score") or 0)
    daily = float(profile.get("distress_score_daily") or DAILY_DEFAULT)
    session_score = float(profile.get("current_session_score") or 0)
    fatigue = float(profile.get("recent_action_fatigue") or 0.0)

    last_decay_str = profile.get("last_decay_date")
    last_decay = date.fromisoformat(last_decay_str) if last_decay_str else today

    last_msg_str = profile.get("last_message_at")
    last_msg = (
        isoparse(last_msg_str)
        if last_msg_str
        else now - SESSION_GAP - timedelta(seconds=1)
    )
    if last_msg.tzinfo is None:
        last_msg = last_msg.replace(tzinfo=timezone.utc)

    # ── 1. 일간 감쇄 (Decay) ──────────────────────────────────────
    days_since_decay = (today - last_decay).days
    if days_since_decay >= 1:
        for _ in range(days_since_decay):
            distress *= DECAY_RATE
        daily = DAILY_DEFAULT
        fatigue = _clamp(fatigue - (FATIGUE_DECAY_PER_HOUR * 24 * days_since_decay), 0, 1)
        last_decay = today
        logger.info("일간 감쇄: user=%s, days=%d, distress=%.1f", user_id, days_since_decay, distress)

    # ── 2. 세션 경계 판정 (10분 침묵) ─────────────────────────────
    if (now - last_msg) >= SESSION_GAP:
        clamped = _clamp(session_score, SESSION_SCORE_MIN, SESSION_SCORE_MAX)
        distress = _clamp(distress + clamped, DISTRESS_MIN, DISTRESS_MAX)
        session_score = 0
        hours_idle = (now - last_msg).total_seconds() / 3600
        fatigue = _clamp(fatigue - (FATIGUE_DECAY_PER_HOUR * hours_idle), 0, 1)
        logger.info("세션 종료 정산: user=%s, delta=%.0f, distress=%.1f", user_id, clamped, distress)

    # ── 3. 현재 발화 감정 점수 반영 ──────────────────────────────
    session_score += emotion_delta
    daily = _clamp(daily + emotion_delta, DAILY_MIN, DAILY_MAX)

    # ── 4. care_level 판정 ────────────────────────────────────────
    effective = distress + _clamp(session_score, SESSION_SCORE_MIN, SESSION_SCORE_MAX)
    effective = _clamp(effective, DISTRESS_MIN, DISTRESS_MAX)
    care_level = _compute_care_level(effective)

    # ── 5. mood_state 결정 ────────────────────────────────────────
    if daily <= 30:
        mood_state = "우울"
    elif daily <= 45:
        mood_state = "적적함"
    elif daily <= 55:
        mood_state = "보통"
    elif daily <= 70:
        mood_state = "편안함"
    else:
        mood_state = "기분좋음"

    # ── 6. Senior Profile 저장 ────────────────────────────────────
    profile_update = {
        "distress_score": round(distress, 2),
        "distress_score_daily": round(daily, 2),
        "care_level": care_level,
        "current_session_score": round(session_score, 2),
        "last_message_at": now.isoformat(),
        "last_decay_date": last_decay.isoformat(),
        "mood_state": mood_state,
        "recent_action_fatigue": round(fatigue, 3),
        "updated_at": now.isoformat(),
    }
    upsert_senior_profile(user_id, profile_update)

    # ── 7. Daily Care Report 갱신 ─────────────────────────────────
    _update_daily_report(user_id, today_str, emotion, daily, care_level)

    logger.info(
        "감정 업데이트: user=%s, emo=%s(%+d), session=%.0f, "
        "distress=%.1f, daily=%.0f, care=%s, mood=%s",
        user_id, emotion, emotion_delta, session_score,
        distress, daily, care_level, mood_state,
    )
    return profile_update


def _update_daily_report(
    user_id: str,
    report_date: str,
    emotion: str,
    daily_score: float,
    care_level: str,
) -> None:
    """일간 리포트를 해당 발화 감정으로 갱신한다."""
    try:
        existing = get_daily_report(user_id, report_date)

        total = (existing.get("total_utterance") or 0) + 1 if existing else 1
        danger = (existing.get("danger_count") or 0) if existing else 0
        if is_danger_emotion(emotion):
            danger += 1

        prev_avg = float(existing.get("mood_avg_score") or DAILY_DEFAULT) if existing else DAILY_DEFAULT
        new_avg = prev_avg + (daily_score - prev_avg) / total

        requires_check = care_level == "DANGER" or danger >= 3

        report_data = {
            "dominant_emotion": emotion,
            "mood_avg_score": round(new_avg, 2),
            "today_care_level": care_level,
            "total_utterance": total,
            "danger_count": danger,
            "requires_check": requires_check,
        }
        upsert_daily_report(user_id, report_date, report_data)
    except Exception:
        logger.exception("Daily Care Report 갱신 실패")


def update_action_fatigue(user_id: str, response: str) -> float:
    """행동 개입 후 사용자 반응에 따라 추천 피로도를 갱신한다."""
    profile = get_senior_profile(user_id)
    fatigue = float((profile or {}).get("recent_action_fatigue") or 0.0)

    if response == "ACCEPT":
        fatigue = _clamp(fatigue + FATIGUE_ACCEPT_BUMP, 0, 1)
    elif response == "REJECT":
        fatigue = _clamp(fatigue + FATIGUE_REJECT_BUMP, 0, 1)
    elif response == "IGNORE":
        fatigue = _clamp(fatigue + FATIGUE_IGNORE_BUMP, 0, 1)

    upsert_senior_profile(user_id, {
        "recent_action_fatigue": round(fatigue, 3),
        "last_accepted_action" if response == "ACCEPT" else "last_rejected_action":
            response,
        "updated_at": _now_utc().isoformat(),
    })
    return fatigue
