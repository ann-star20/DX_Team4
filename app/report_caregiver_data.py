"""보호자 리포트: 이번 달 감정 달력·월간 키워드 가공."""

from __future__ import annotations

import calendar
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

KST = timezone(timedelta(hours=9))

_KR_STOP = frozenset(
    """
    은 는 이 가 을 를 에 의 와 과 도 만 보 다 때 거 등 좀 잘 게 음 응 그 이거 그거 저거 뭐
    예요 아요 해요 네요 습니다 요 죠 지 만요 은요 는요 이요 가요 을요 를요
    안녕 하세요 감사합니다 그리고 그런데 그래서 정말 너무 매우 아주 조금 오늘 같아 내가 때문에 기분이 있는
    """.split()
)


def build_care_calendar_month(
    today: date,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    rows: [{report_date, today_care_level}, ...]
    이번 달 1일~오늘까지 셀 + JS getDay()용 첫 요일.
    """
    y, m = today.year, today.month
    level_by_date: dict[str, str | None] = {}
    for r in rows:
        ds = r.get("report_date")
        if not ds:
            continue
        ds = str(ds)
        lv = r.get("today_care_level")
        if isinstance(lv, str) and lv in ("NORMAL", "WARNING", "DANGER"):
            level_by_date[ds] = lv
        else:
            level_by_date[ds] = None

    _, last_day = calendar.monthrange(y, m)
    first = date(y, m, 1)
    first_js_weekday = (first.weekday() + 1) % 7

    days: list[dict[str, Any]] = []
    for d in range(1, last_day + 1):
        dd = date(y, m, d)
        if dd > today:
            break
        ds = dd.isoformat()
        days.append({"day": d, "date": ds, "level": level_by_date.get(ds)})

    return {
        "year": y,
        "month": m,
        "first_weekday_js": first_js_weekday,
        "days": days,
    }


def extract_monthly_keywords(texts: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    """간단 한국어 연속 음절 추출 + 빈도."""
    counter: Counter[str] = Counter()
    for t in texts:
        if not t:
            continue
        for w in re.findall(r"[가-힣]{2,}", t):
            if w in _KR_STOP or len(w) < 2:
                continue
            counter[w] += 1
    return [{"word": w, "count": n} for w, n in counter.most_common(top_n)]


def month_start_iso_for_chat_filter(year: int, month: int) -> str:
    """Supabase chat_time 비교용 (월초 KST)."""
    dt = datetime(year, month, 1, 0, 0, 0, tzinfo=KST)
    return dt.isoformat()
