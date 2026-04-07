"""
회상 기억 요법(reminiscence)용: 행복·고중요도 장기기억 선별 + 아침 브리프.
별도 테이블 없이 long_term_memory.metadata(importance_score, emotion)로 필터한다.
"""

from __future__ import annotations

import logging
from datetime import date

from app.database import (
    get_daily_report,
    get_senior_profile,
    list_recent_memories,
    upsert_daily_report,
)
from app.emotion import compute_effective_distress_from_profile

logger = logging.getLogger(__name__)

# metadata.emotion 문자열에 포함되면 ‘긍정/행복’ 계열로 본다 (부분 일치)
HAPPY_EMOTION_MARKERS: tuple[str, ...] = (
    "기쁨",
    "행복",
    "즐거움",
    "즐거",
    "편안",
    "만족",
    "좋았",
    "좋아",
    "신나",
    "행복해",
    "기쁘",
    "happy",
    "joy",
)


def _importance_from_metadata(meta: dict) -> int:
    raw = meta.get("importance_score")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _is_happy_emotion(emotion: str | None) -> bool:
    if not emotion:
        return False
    e = emotion.strip()
    el = e.lower()
    for m in HAPPY_EMOTION_MARKERS:
        if m in e or m in el:
            return True
    return False


def pick_happy_reminiscence_memories(
    rows: list[dict],
    *,
    min_importance: int = 7,
    max_items: int = 8,
) -> list[dict]:
    """importance_score >= min 이고 감정이 행복 계열인 기억만 고른다."""
    picked: list[dict] = []
    for r in rows:
        meta = r.get("metadata") or {}
        if _importance_from_metadata(meta) < min_importance:
            continue
        emo = meta.get("emotion")
        if isinstance(emo, str) and _is_happy_emotion(emo):
            picked.append(
                {
                    "id": r.get("id"),
                    "content": r.get("content"),
                    "memory_date": r.get("memory_date"),
                    "metadata": meta,
                }
            )
        if len(picked) >= max_items:
            break
    return picked


def build_morning_brief(
    user_id: str,
    *,
    min_importance: int = 7,
    memory_fetch_limit: int = 250,
    max_memories: int = 8,
) -> dict:
    """
    TV/아침 루틴용: effective_distress 구간에 따라 회상 후보 또는 보호자 알림.

    - 41 <= effective < 80 (WARNING): 행복·고중요도 기억 목록
    - effective >= 80 (DANGER): caregiver_alert, 오늘 일간 리포트 requires_check 갱신
    """
    profile = get_senior_profile(user_id)
    if not profile:
        return {
            "user_id": user_id,
            "error": "no_profile",
            "effective_distress": None,
            "care_level": None,
            "reminiscence_mode": False,
            "caregiver_alert": False,
            "happy_memories": [],
        }

    effective, care_level = compute_effective_distress_from_profile(profile)

    out: dict = {
        "user_id": user_id,
        "distress_score": profile.get("distress_score"),
        "current_session_score": profile.get("current_session_score"),
        "effective_distress": round(effective, 2),
        "care_level": care_level,
        "distress_score_daily": profile.get("distress_score_daily"),
        "mood_state": profile.get("mood_state"),
        "reminiscence_mode": False,
        "caregiver_alert": False,
        "happy_memories": [],
        "suggested_opening": None,
    }

    if effective >= 80:
        out["caregiver_alert"] = True
        out["reminiscence_mode"] = False
        today_str = date.today().isoformat()
        try:
            existing = get_daily_report(user_id, today_str)
            if existing:
                report_data = {
                    "requires_check": True,
                    "today_care_level": "DANGER",
                    "dominant_emotion": existing.get("dominant_emotion"),
                    "mood_avg_score": existing.get("mood_avg_score"),
                    "total_utterance": existing.get("total_utterance") or 0,
                    "danger_count": existing.get("danger_count") or 0,
                }
            else:
                report_data = {
                    "dominant_emotion": "중립",
                    "mood_avg_score": 50.0,
                    "today_care_level": "DANGER",
                    "total_utterance": 0,
                    "danger_count": 0,
                    "requires_check": True,
                }
            upsert_daily_report(user_id, today_str, report_data)
        except Exception:
            logger.exception("보호자 알림용 일간 리포트 갱신 실패 user=%s", user_id)
        logger.warning(
            "케어 DANGER(effective>=%s): 보호자 알림 플래그 user=%s eff=%.1f",
            80,
            user_id,
            effective,
        )
        return out

    if 41 <= effective < 80:
        out["reminiscence_mode"] = True
        rows = list_recent_memories(user_id, limit=memory_fetch_limit)
        memories = pick_happy_reminiscence_memories(
            rows,
            min_importance=min_importance,
            max_items=max_memories,
        )
        out["happy_memories"] = memories
        if memories:
            first = memories[0]["content"]
            out["suggested_opening"] = (
                f"어르신, 예전에 이런 좋은 날도 있었잖아요. {first[:120]}"
                + ("…" if len(first) > 120 else "")
            )
        else:
            out["suggested_opening"] = (
                "어르신, 오늘 아침 기분은 어떠세요? 좋았던 일이 떠오르시면 말씀해 주세요."
            )

    return out
