"""One-time backfill of historical NBA odds from The Odds API (paid plans only).

For each game day:
  - events list as of 10am ET                          (1 credit)
  - player_points lines for each game, 1h before tip   (10 credits per game, from 2023-05-03)
  - moneylines (h2h) for all games, 1h before tip      (10 credits per snapshot; a second
    snapshot is taken when late games tip 3h+ after the early ones)

Files land in the same layout as the daily odds job, with `data` holding the same
shape the live endpoints return, so downstream code treats both sources alike.
snapshot_ts is the time we requested; the API's actual snapshot time is in _meta.
Existing files are skipped, so re-running never spends credits twice.

Run locally (needs boto3 for S3 + SSM):
  uv run --python 3.12 --with boto3 python -m ingest.backfill \
      --start 2022-10-18 --end 2026-06-30 --out s3://<bucket>
  env: ODDS_API_KEY or ODDS_API_KEY_PARAM; AWS_PROFILE for S3/SSM access.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import http
from .odds import ODDS_BASE, SPORT, CreditsExhausted, _iso, _remaining, api_key
from .storage import Store, envelope

log = logging.getLogger(__name__)

HIST = f"{ODDS_BASE}/historical/sports/{SPORT}"
ET = ZoneInfo("America/New_York")
PROPS_AVAILABLE_FROM = datetime(2023, 5, 3, 5, 30, tzinfo=timezone.utc)
OFFSEASON_MONTHS = {7, 8, 9}
LATE_SLATE_GAP = timedelta(hours=3)
BEFORE_TIP = timedelta(hours=1)


def game_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.month not in OFFSEASON_MONTHS:
            days.append(d)
        d += timedelta(days=1)
    return days


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def h2h_times(tips: list[datetime]) -> list[datetime]:
    """1h before the first tip, plus 1h before the first late game if the slate is split."""
    tips = sorted(tips)
    times = [tips[0] - BEFORE_TIP]
    late = [t for t in tips if t - tips[0] >= LATE_SLATE_GAP]
    if late:
        times.append(late[0] - BEFORE_TIP)
    return times


class Backfill:
    def __init__(self, store: Store, key: str, *, markets: str = "player_points", regions: str = "us",
                 min_credits: int = 5000, pause: float = 0.2):
        self.store, self.key, self.markets, self.regions = store, key, markets, regions
        self.min_credits, self.pause = min_credits, pause
        self.remaining: int | None = None

    def _get(self, url: str, params: dict) -> dict:
        if self.remaining is not None and self.remaining < self.min_credits:
            raise CreditsExhausted(f"Stopping: {self.remaining} credits left (reserve {self.min_credits}).")
        body, headers = http.get_json(url, {"apiKey": self.key, **params})
        if (latest := _remaining(headers)) is not None:
            self.remaining = latest
        time.sleep(self.pause)
        return body

    def _fetch(self, key: str, url: str, params: dict) -> dict | None:
        """Fetch a historical snapshot and store it, unless already stored."""
        if self.store.exists(key):
            return None
        body = self._get(url, params)
        self.store.put_json(key, envelope(body["data"], source=url, params=params, extra={
            "historical": True, "snapshot_timestamp": body.get("timestamp"), "credits_remaining": self.remaining}))
        return body

    def day(self, d: date) -> dict:
        stats = {"date": d.isoformat(), "games": 0, "props": 0, "h2h": 0}
        prefix = f"raw/odds/{SPORT}/snapshot_date={d.isoformat()}"
        as_of = datetime(d.year, d.month, d.day, 10, tzinfo=ET).astimezone(timezone.utc)

        ev_key = f"{prefix}/snapshot_ts={as_of:%Y%m%dT%H%M%SZ}/events.json.gz"
        ev_params = {"date": _iso(as_of)}
        body = self._fetch(ev_key, f"{HIST}/events", ev_params)
        events = body["data"] if body is not None else self._read_events(ev_key)
        todays = [e for e in events if _parse(e["commence_time"]).astimezone(ET).date() == d]
        stats["games"] = len(todays)
        if not todays:
            return stats

        for e in todays:
            tip = _parse(e["commence_time"])
            if tip < PROPS_AVAILABLE_FROM:
                continue
            t = tip - BEFORE_TIP
            params = {"date": _iso(t), "regions": self.regions, "markets": self.markets, "oddsFormat": "american"}
            if self._fetch(f"{prefix}/snapshot_ts={t:%Y%m%dT%H%M%SZ}/event={e['id']}.json.gz",
                           f"{HIST}/events/{e['id']}/odds", params) is not None:
                stats["props"] += 1

        for t in h2h_times([_parse(e["commence_time"]) for e in todays]):
            params = {"date": _iso(t), "regions": self.regions, "markets": "h2h", "oddsFormat": "american"}
            if self._fetch(f"{prefix}/snapshot_ts={t:%Y%m%dT%H%M%SZ}/h2h.json.gz", f"{HIST}/odds", params) is not None:
                stats["h2h"] += 1
        return stats

    def _read_events(self, key: str) -> list:
        return self.store.get_json(key)["data"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", default=os.environ.get("OUTPUT_URI"), required="OUTPUT_URI" not in os.environ)
    p.add_argument("--min-credits", type=int, default=5000, help="stop when remaining credits fall below this")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    os.environ.setdefault("ODDS_API_KEY_PARAM", "/nba-lakehouse/odds-api-key")
    job = Backfill(Store(args.out), api_key(), min_credits=args.min_credits)
    totals = {"days": 0, "games": 0, "props": 0, "h2h": 0}
    for d in game_days(args.start, args.end):
        stats = job.day(d)
        totals["days"] += 1
        for k in ("games", "props", "h2h"):
            totals[k] += stats[k]
        if stats["games"]:
            log.info("%s credits_remaining=%s", stats, job.remaining)
    log.info("done %s credits_remaining=%s", totals, job.remaining)


if __name__ == "__main__":
    main()
