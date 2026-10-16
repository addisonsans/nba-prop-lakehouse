"""Snapshot completed NBA games (full ESPN game summaries) into the raw zone.

ESPN's public site API is used because stats.nba.com / cdn.nba.com block
cloud IPs. Each summary holds both teams' player box scores plus the injury
report, play-by-play and the ESPN consensus line, stored untouched.

Lambda event (all optional):
  {"date": "2026-04-15"}          # ingest one ET game date
  {"lookback_days": 3}            # default: the last N ET dates (yesterday first)

S3 layout:
  raw/espn/scoreboard/game_date=YYYY-MM-DD/scoreboard.json.gz
  raw/espn/summary/game_date=YYYY-MM-DD/event=<espn_event_id>.json.gz
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

from . import http
from .storage import Store, envelope

log = logging.getLogger()
log.setLevel(logging.INFO)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def ingest_date(store: Store, game_date: date, *, pause: float = 0.5) -> dict:
    """Fetch the scoreboard for one ET date and every completed game's summary.

    Idempotent: summaries already in the store are skipped, so re-runs and
    overlapping lookback windows cost only one scoreboard request.
    """
    params = {"dates": game_date.strftime("%Y%m%d")}
    scoreboard, _ = http.get_json(f"{ESPN_BASE}/scoreboard", params)
    store.put_json(
        f"raw/espn/scoreboard/game_date={game_date}/scoreboard.json.gz",
        envelope(scoreboard, source=f"{ESPN_BASE}/scoreboard", params=params),
    )

    stats = {"date": str(game_date), "games": 0, "written": 0, "skipped_existing": 0, "not_final": 0}
    for event in scoreboard.get("events", []):
        stats["games"] += 1
        if not event.get("status", {}).get("type", {}).get("completed"):
            stats["not_final"] += 1
            continue
        key = f"raw/espn/summary/game_date={game_date}/event={event['id']}.json.gz"
        if store.exists(key):
            stats["skipped_existing"] += 1
            continue
        summary_params = {"event": event["id"]}
        summary, _ = http.get_json(f"{ESPN_BASE}/summary", summary_params)
        store.put_json(key, envelope(summary, source=f"{ESPN_BASE}/summary", params=summary_params))
        stats["written"] += 1
        time.sleep(pause)
    return stats


def target_dates(event: dict, today_utc: date) -> list[date]:
    """Dates to ingest. The schedule fires in the early ET morning, when the UTC
    date equals the ET date, so 'yesterday' is today_utc - 1."""
    if event.get("date"):
        return [date.fromisoformat(event["date"])]
    lookback = int(event.get("lookback_days", os.environ.get("LOOKBACK_DAYS", 3)))
    return [today_utc - timedelta(days=i) for i in range(1, lookback + 1)]


def handler(event, context):
    store = Store(os.environ["OUTPUT_URI"])
    results = [
        ingest_date(store, d) for d in target_dates(event or {}, datetime.now(timezone.utc).date())
    ]
    for r in results:
        log.info("boxscores %s", r)
    return {"results": results}
