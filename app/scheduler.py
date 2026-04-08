from __future__ import annotations

import ast
import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings
from app.database import get_supabase, upsert_senior_profile
from app.session_manager import settle_session_to_profile

logger = logging.getLogger(__name__)

DISTRESS_MIN = -100.0
DISTRESS_MAX = 100.0
SESSION_TIMEOUT_MIN = 5
AGING_FACTOR = 0.7
KST = ZoneInfo("Asia/Seoul")

_scheduler: AsyncIOScheduler | None = None


def _now_kst() -> datetime:
    return datetime.now(KST)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_care_level(score: float) -> str:
    if score >= 80:
        return "DANGER"
    if score >= 41:
        return "WARNING"
    return "NORMAL"


def _parse_trigger(spec: str):
    """
    .env 기반 트리거 파싱:
    - interval:seconds=30
    - interval:minutes=5
    - cron:minute='*/1'
    - cron:hour=0,minute=0
    """
    if not spec or ":" not in spec:
        raise ValueError(f"Invalid cron spec: {spec}")
    kind, args_str = spec.split(":", 1)
    kind = kind.strip().lower()
    kwargs = {}
    if args_str.strip():
        for token in args_str.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid token in cron spec: {token}")
            k, v = token.split("=", 1)
            key = k.strip()
            raw = v.strip()
            try:
                val = ast.literal_eval(raw)
            except Exception:
                val = raw
            kwargs[key] = val

    if kind == "interval":
        return IntervalTrigger(timezone="Asia/Seoul", **kwargs)
    if kind == "cron":
        return CronTrigger(timezone="Asia/Seoul", **kwargs)
    raise ValueError(f"Unsupported trigger kind: {kind}")


def close_inactive_emotion_sessions() -> None:
    """5분 무활동 session 종료 + senior_profile 정산."""
    sb = get_supabase()
    now_utc = _now_utc()
    now_kst = now_utc.astimezone(KST)
    cutoff_iso = (now_utc - timedelta(minutes=SESSION_TIMEOUT_MIN)).isoformat()

    result = (
        sb.table("emotion_session")
        .select("*")
        .eq("is_closed", False)
        .lte("last_activity_at", cutoff_iso)
        .execute()
    )
    sessions = result.data or []
    if not sessions:
        return

    closed_count = 0
    for s in sessions:
        user_id = str(s.get("user_id") or "")
        if not user_id:
            continue
        # 세션 종료 시 정산: 명시적으로 KST 시각 전달
        settle_session_to_profile(s, now_kst)

        sb.table("emotion_session").update(
            {"is_closed": True, "last_activity_at": now_utc.isoformat()}
        ).eq("session_id", s["session_id"]).execute()
        closed_count += 1

    logger.info("5분 세션 클로징 완료: %d건", closed_count)


def run_midnight_aging() -> None:
    """자정 리셋: daily=0, distress/incongruence 70% 감쇄, care_level 재판정."""
    sb = get_supabase()
    today = date.today().isoformat()
    now_iso = _now_kst().isoformat()

    result = sb.table("senior_profile").select("*").execute()
    profiles = result.data or []
    if not profiles:
        return

    updated_count = 0
    for p in profiles:
        user_id = str(p.get("user_id") or "")
        if not user_id:
            continue
        prev_distress = float(p.get("distress_score") or 0.0)
        prev_incong = float(p.get("incongruence_score") or 0.0)
        decayed_distress = _clamp(prev_distress * AGING_FACTOR, DISTRESS_MIN, DISTRESS_MAX)
        decayed_incong = max(0.0, prev_incong * AGING_FACTOR)

        upsert_senior_profile(
            user_id,
            {
                "distress_score": round(decayed_distress, 4),
                "distress_score_daily": 0.0,
                "incongruence_score": round(decayed_incong, 4),
                "care_level": _compute_care_level(decayed_distress),
                "last_decay_date": today,
                "updated_at": now_iso,
            },
        )
        updated_count += 1

    logger.info("자정 에이징 완료: %d명", updated_count)


def _safe_job(job_name: str, fn) -> None:
    try:
        fn()
    except Exception:
        logger.exception("스케줄러 작업 실패: %s", job_name)


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return

    settings = get_settings()
    session_trigger = _parse_trigger(settings.cron_session_check)
    midnight_trigger = _parse_trigger(settings.cron_midnight_reset)

    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    scheduler.add_job(
        lambda: _safe_job("close_inactive_emotion_sessions", close_inactive_emotion_sessions),
        trigger=session_trigger,
        id="close_inactive_emotion_sessions",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        lambda: _safe_job("run_midnight_aging", run_midnight_aging),
        trigger=midnight_trigger,
        id="run_midnight_aging",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "APScheduler started (session=%s, midnight=%s)",
        settings.cron_session_check,
        settings.cron_midnight_reset,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info("APScheduler stopped")

