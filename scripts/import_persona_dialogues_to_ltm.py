#!/usr/bin/env python3
"""Persona JSONL(한 줄에 JSON 하나)을 파싱해 long_term_memory에 적재한다.

- user_id: profile.persona-id
- content: HS* → user, SS* → assistant, 턴 번호 순
- metadata: talk_id, source만 (감정·페르소나 세부 등 미포함)
- embedding: NULL (스키마에서 nullable)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import get_supabase


def content_from_talk(talk_content: dict) -> str:
    items: list[tuple[int, int, str, str]] = []
    for key, val in talk_content.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        m = re.match(r"^(HS|SS)(\d+)$", key)
        if not m:
            continue
        turn = int(m.group(2))
        prefix = m.group(1)
        role_order = 0 if prefix == "HS" else 1
        role = "user" if prefix == "HS" else "assistant"
        items.append((turn, role_order, role, val))
    items.sort(key=lambda x: (x[0], x[1]))
    return "\n".join(f"{r}: {t}" for _, _, r, t in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="+", help="JSONL 파일(여러 개면 순서대로 이어서 처리)")
    ap.add_argument("--memory-date", default=None, help="YYYY-MM-DD (기본: 오늘)")
    args = ap.parse_args()
    memory_date = args.memory_date or date.today().isoformat()
    sb = get_supabase()
    n = 0
    for path in args.jsonl:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.exception(f"JSON 파싱 실패: {line[:200]!r}")
                raise
            user_id = obj["profile"]["persona-id"]
            talk_id = obj["talk"]["id"]["talk-id"]
            body = content_from_talk(obj["talk"]["content"])
            row = {
                "user_id": user_id,
                "memory_date": memory_date,
                "content": body,
                "embedding": None,
                "metadata": {
                    "source": "persona_dialogue_import",
                    "talk_id": talk_id,
                },
            }
            try:
                sb.table("long_term_memory").insert(row).execute()
            except Exception:
                logger.exception(
                    f"long_term_memory insert 실패: user_id={user_id} talk_id={talk_id}"
                )
                raise
            n += 1
            logger.debug(f"{n} {user_id} {talk_id}")
    logger.info(f"적재 완료: {n} rows")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()
