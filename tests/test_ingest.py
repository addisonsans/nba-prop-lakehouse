import gzip
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import backfill, boxscores, databricks_sync, odds  # noqa: E402
from ingest.storage import Store  # noqa: E402


def read(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


class FakeHttp:
    """Routes get_json calls by URL suffix and records them."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, params=None, **_):
        self.calls.append((url, params))
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                return response(params) if callable(response) else response
        raise AssertionError(f"unexpected URL {url}")


def test_boxscores_writes_only_final_games_and_is_idempotent(tmp_path, monkeypatch):
    scoreboard = {"events": [
        {"id": "1", "status": {"type": {"completed": True}}},
        {"id": "2", "status": {"type": {"completed": False}}},
    ]}
    fake = FakeHttp({"/scoreboard": (scoreboard, {}), "/summary": ({"boxscore": {}}, {})})
    monkeypatch.setattr(boxscores.http, "get_json", fake)
    store = Store(str(tmp_path))

    first = boxscores.ingest_date(store, date(2026, 4, 15), pause=0)
    second = boxscores.ingest_date(store, date(2026, 4, 15), pause=0)

    assert first == {"date": "2026-04-15", "games": 2, "written": 1, "skipped_existing": 0, "not_final": 1, "missing": 0}
    assert second["written"] == 0 and second["skipped_existing"] == 1
    saved = read(tmp_path / "raw/espn/summary/game_date=2026-04-15/event=1.json.gz")
    assert saved["data"] == {"boxscore": {}} and saved["_meta"]["params"] == {"event": "1"}
    assert fake.calls[0][1] == {"dates": "20260415"}


def test_boxscore_target_dates():
    assert boxscores.target_dates({"date": "2026-04-15"}, date(2026, 10, 16)) == [date(2026, 4, 15)]
    assert boxscores.target_dates({"lookback_days": 2}, date(2026, 10, 16)) == [date(2026, 10, 15), date(2026, 10, 14)]


def _odds_fake(remaining_after_each):
    credits = iter(remaining_after_each)
    events = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    return FakeHttp({
        "/events": (events, {"x-requests-remaining": "100"}),
        "/odds": lambda p: ({"bookmakers": []}, {"x-requests-remaining": str(next(credits)), "x-requests-last": "1"}),
    })


def test_odds_snapshot_writes_each_event_and_strips_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(odds.http, "get_json", _odds_fake([99, 98, 97]))
    now = datetime(2026, 11, 2, 21, 0, tzinfo=timezone.utc)

    stats = odds.snapshot(Store(str(tmp_path)), "SECRET", now)

    assert stats["written"] == 3 and stats["credits_remaining"] == 97
    files = list(tmp_path.rglob("*.json.gz"))
    assert len(files) == 4  # events listing + 3 events
    assert all("SECRET" not in gzip.decompress(f.read_bytes()).decode() for f in files)


def test_odds_snapshot_stops_before_exhausting_credits(tmp_path, monkeypatch):
    monkeypatch.setenv("MIN_CREDITS_REMAINING", "25")
    monkeypatch.setattr(odds.http, "get_json", _odds_fake([26, 25]))
    now = datetime(2026, 11, 2, 21, 0, tzinfo=timezone.utc)
    store = Store(str(tmp_path))
    # After event b only 25 credits remain; spending 1 more would dip below the reserve.
    with pytest.raises(odds.CreditsExhausted):
        odds.snapshot(store, "k", now)
    assert len(list(tmp_path.rglob("event=*.json.gz"))) == 2


def _backfill_fake(tips, remaining=50000):
    events = [{"id": f"g{i}", "commence_time": t} for i, t in enumerate(tips)]
    def odds_route(params):
        return ({"timestamp": params["date"], "data": {"markets": params["markets"]}},
                {"x-requests-remaining": str(remaining), "x-requests-last": "10"})
    return FakeHttp({
        "/events": ({"timestamp": "2024-01-15T15:00:00Z", "data": events}, {"x-requests-remaining": str(remaining)}),
        "/odds": odds_route,
    })


def test_backfill_day_fetches_todays_games_once(tmp_path, monkeypatch):
    # 1:10pm, 7pm and 10pm ET on Jan 15, plus a game on Jan 16 that must be ignored.
    fake = _backfill_fake(["2024-01-15T18:10:00Z", "2024-01-16T00:00:00Z", "2024-01-16T03:00:00Z", "2024-01-16T18:00:00Z"])
    monkeypatch.setattr(backfill.http, "get_json", fake)
    job = backfill.Backfill(Store(str(tmp_path)), "SECRET", pause=0)

    first = job.day(date(2024, 1, 15))
    calls_after_first = len(fake.calls)
    second = job.day(date(2024, 1, 15))

    assert first == {"date": "2024-01-15", "games": 3, "props": 3, "h2h": 2, "missing": 0}
    assert second == {"date": "2024-01-15", "games": 3, "props": 0, "h2h": 0, "missing": 0}
    assert len(fake.calls) == calls_after_first  # re-run spends nothing
    prop = read(tmp_path / "raw/odds/basketball_nba/snapshot_date=2024-01-15/snapshot_ts=20240115T171000Z/event=g0.json.gz")
    assert prop["data"] == {"markets": "player_points"} and prop["_meta"]["historical"] is True
    assert (tmp_path / "raw/odds/basketball_nba/snapshot_date=2024-01-15/snapshot_ts=20240115T230000Z/h2h.json.gz").exists()
    assert all("SECRET" not in gzip.decompress(f.read_bytes()).decode() for f in tmp_path.rglob("*.json.gz"))


def test_backfill_skips_props_before_they_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill.http, "get_json", _backfill_fake(["2023-01-10T00:00:00Z"]))
    stats = backfill.Backfill(Store(str(tmp_path)), "k", pause=0).day(date(2023, 1, 9))
    assert stats == {"date": "2023-01-09", "games": 1, "props": 0, "h2h": 1, "missing": 0}


def test_backfill_stops_at_credit_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill.http, "get_json", _backfill_fake(["2024-01-16T00:00:00Z"], remaining=4999))
    with pytest.raises(odds.CreditsExhausted):
        backfill.Backfill(Store(str(tmp_path)), "k", pause=0, min_credits=5000).day(date(2024, 1, 15))
    assert not list(tmp_path.rglob("event=*.json.gz"))


def test_backfill_skips_games_with_no_props_snapshot(tmp_path, monkeypatch):
    fake = _backfill_fake(["2024-01-16T00:00:00Z", "2024-01-16T01:00:00Z"])
    inner = fake.routes["/odds"]
    def odds_route(params):
        if params["markets"] == "player_points" and params["date"] == "2024-01-15T23:00:00Z":
            raise backfill.http.HTTPStatusError("HTTP 404 from api.the-odds-api.com", 404)
        return inner(params)
    fake.routes["/odds"] = odds_route
    monkeypatch.setattr(backfill.http, "get_json", fake)

    stats = backfill.Backfill(Store(str(tmp_path)), "k", pause=0).day(date(2024, 1, 15))

    assert stats == {"date": "2024-01-15", "games": 2, "props": 1, "h2h": 1, "missing": 1}


def test_backfill_calendar_helpers():
    assert backfill.game_days(date(2024, 6, 30), date(2024, 10, 1)) == [date(2024, 6, 30), date(2024, 10, 1)]
    tip = lambda h: datetime(2024, 1, 15, h, tzinfo=timezone.utc)
    assert backfill.h2h_times([tip(23), tip(23)]) == [tip(22)]
    assert backfill.h2h_times([tip(18), tip(21), tip(23)]) == [tip(17), tip(20)]


def test_databricks_sync_uploads_each_new_object_to_same_path(monkeypatch):
    calls = []
    def fake_call(method, url, *, data=None, headers=None, **_):
        calls.append((method, url, data, headers))
        if url.endswith("/oidc/v1/token"):
            return b'{"access_token": "TOKEN", "expires_in": 3600}'
        return b""
    class FakeS3:
        def get_object(self, Bucket, Key):
            return {"Body": type("B", (), {"read": lambda self: f"{Bucket}/{Key}".encode()})()}
    import boto3
    monkeypatch.setattr(boto3, "client", lambda name: FakeS3())
    monkeypatch.setattr(databricks_sync, "_call", fake_call)
    monkeypatch.setattr(databricks_sync, "_token", {})
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc.example.com/")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client")
    monkeypatch.setenv("DATABRICKS_SECRET", "s3cret")
    monkeypatch.setenv("VOLUME_PATH", "/Volumes/nba/raw/landing")
    record = lambda key: {"s3": {"bucket": {"name": "lake"}, "object": {"key": key}}}

    out = databricks_sync.handler({"Records": [record("raw/odds/snapshot_date%3D2026-10-16/a.json.gz"),
                                               record("historical/nba_stats/teams/teams.parquet")]}, None)

    assert out == {"copied": 2}
    token_calls = [c for c in calls if c[1].endswith("/oidc/v1/token")]
    puts = [c for c in calls if c[0] == "PUT"]
    assert len(token_calls) == 1  # token reused across files
    assert puts[0][1] == "https://dbc.example.com/api/2.0/fs/files/Volumes/nba/raw/landing/raw/odds/snapshot_date%3D2026-10-16/a.json.gz?overwrite=true"
    assert puts[0][2] == b"lake/raw/odds/snapshot_date=2026-10-16/a.json.gz"
    assert puts[1][3]["Authorization"] == "Bearer TOKEN"
