"""Snapshot today's NBA player-prop lines (The Odds API) plus the ESPN injury report.

Free historical player-prop data barely exists, so this job builds our own
history going forward: one snapshot per day, shortly before evening tip-offs.

Credit math (The Odds API free tier ~500 credits/month): each event-odds
call costs (#markets x #regions). With 1 market x 1 region and ~8-15 games a
day, one daily snapshot fits; MIN_CREDITS_REMAINING stops us before running dry.

Env:
  OUTPUT_URI              s3://bucket or local path
  ODDS_API_KEY_PARAM      SSM SecureString parameter name holding the key
  ODDS_API_KEY            (local runs only) the key itself, overrides SSM
  MARKETS                 default "player_points"
  REGIONS                 default "us"
  LOOKAHEAD_HOURS         default 24
  MIN_CREDITS_REMAINING   default 25

S3 layout:
  raw/odds/basketball_nba/snapshot_date=YYYY-MM-DD/snapshot_ts=<ts>/events.json.gz
  raw/odds/basketball_nba/snapshot_date=YYYY-MM-DD/snapshot_ts=<ts>/event=<id>.json.gz
  raw/odds/basketball_nba/snapshot_date=YYYY-MM-DD/snapshot_ts=<ts>/game_lines.json.gz
    (moneyline, spread and total for every game on the slate: one call, 3 credits)
  raw/espn/injuries/snapshot_date=YYYY-MM-DD/snapshot_ts=<ts>.json.gz
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from . import http
from .boxscores import ESPN_BASE
from .storage import Store, envelope

log = logging.getLogger()
log.setLevel(logging.INFO)

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
GAME_MARKETS = "h2h,spreads,totals"


class CreditsExhausted(RuntimeError):
    pass


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def api_key() -> str:
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    import boto3

    resp = boto3.client("ssm").get_parameter(Name=os.environ["ODDS_API_KEY_PARAM"], WithDecryption=True)
    return resp["Parameter"]["Value"]


def snapshot(store: Store, key: str, now: datetime) -> dict:
    markets = os.environ.get("MARKETS", "player_points")
    regions = os.environ.get("REGIONS", "us")
    lookahead = int(os.environ.get("LOOKAHEAD_HOURS", 24))
    min_credits = int(os.environ.get("MIN_CREDITS_REMAINING", 25))
    cost_per_event = len(markets.split(",")) * len(regions.split(","))

    snap_date, snap_ts = now.strftime("%Y-%m-%d"), now.strftime("%Y%m%dT%H%M%SZ")
    prefix = f"raw/odds/{SPORT}/snapshot_date={snap_date}/snapshot_ts={snap_ts}"

    # The events listing is free (does not use credits).
    ev_params = {"apiKey": key, "commenceTimeFrom": _iso(now), "commenceTimeTo": _iso(now + timedelta(hours=lookahead))}
    events, headers = http.get_json(f"{ODDS_BASE}/sports/{SPORT}/events", ev_params)
    store.put_json(f"{prefix}/events.json.gz", envelope(events, source=f"{ODDS_BASE}/sports/{SPORT}/events", params=ev_params))

    remaining = _remaining(headers)
    stats = {"snapshot_ts": snap_ts, "events": len(events), "written": 0, "credits_remaining": remaining}
    for ev in events:
        if remaining is not None and remaining - cost_per_event < min_credits:
            stats["credits_remaining"] = remaining
            raise CreditsExhausted(f"Stopping: {remaining} credits left (reserve {min_credits}). Partial snapshot: {stats}")
        odds_params = {"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american"}
        url = f"{ODDS_BASE}/sports/{SPORT}/events/{ev['id']}/odds"
        body, headers = http.get_json(url, odds_params)
        if (latest := _remaining(headers)) is not None:
            remaining = latest
        store.put_json(
            f"{prefix}/event={ev['id']}.json.gz",
            envelope(body, source=url, params=odds_params, extra={"credits_remaining": remaining, "credits_last": headers.get("x-requests-last")}),
        )
        stats["written"] += 1

    # Game lines for the whole slate in one bulk call, after the props (the priority).
    lines_params = {"apiKey": key, "regions": regions, "markets": GAME_MARKETS, "oddsFormat": "american",
                    "commenceTimeFrom": _iso(now), "commenceTimeTo": _iso(now + timedelta(hours=lookahead))}
    lines_cost = len(GAME_MARKETS.split(",")) * len(regions.split(","))
    if events and (remaining is None or remaining - lines_cost >= min_credits):
        url = f"{ODDS_BASE}/sports/{SPORT}/odds"
        body, headers = http.get_json(url, lines_params)
        remaining = _remaining(headers) or remaining
        store.put_json(f"{prefix}/game_lines.json.gz",
                       envelope(body, source=url, params=lines_params, extra={"credits_remaining": remaining}))
        stats["game_lines"] = len(body)
    stats["credits_remaining"] = remaining
    return stats


def snapshot_injuries(store: Store, now: datetime) -> str:
    body, _ = http.get_json(f"{ESPN_BASE}/injuries")
    return store.put_json(
        f"raw/espn/injuries/snapshot_date={now:%Y-%m-%d}/snapshot_ts={now:%Y%m%dT%H%M%SZ}.json.gz",
        envelope(body, source=f"{ESPN_BASE}/injuries"),
    )


def _remaining(headers: dict) -> int | None:
    value = headers.get("x-requests-remaining")
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def handler(event, context):
    store = Store(os.environ["OUTPUT_URI"])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    # Injuries first: free, and still useful on a day the odds call fails.
    log.info("injuries -> %s", snapshot_injuries(store, now))
    stats = snapshot(store, api_key(), now)
    log.info("odds %s", stats)
    return stats
