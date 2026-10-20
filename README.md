# NBA Prop Lakehouse

Free historical NBA player-prop data barely exists, so this project collects
its own: a daily AWS pipeline snapshots prop lines, box scores and injury
reports into S3, and Databricks turns them into a lakehouse for modeling
player-points distributions.

```
EventBridge Scheduler (ET)
 ├─ 5:00pm ─> Lambda nba-lakehouse-odds      ─> The Odds API (props + game lines) + ESPN injuries ─┐
 └─ 6:00am ─> Lambda nba-lakehouse-boxscores ─> ESPN scoreboard + game summaries ─────────────┤
                                                                                              v
                                                      S3  raw/  (gzipped JSON, Hive-style partitions)
                                                                                              │
                                     Databricks: bronze -> silver -> gold -> MLflow model  <──┘  (phase 2)
CloudWatch alarm on any Lambda error -> SNS email · AWS Budget alert at $5/month
```

## Raw zone layout

| Path | What | Written by |
|------|------|------------|
| `raw/odds/basketball_nba/snapshot_date=/snapshot_ts=/events.json.gz` | Upcoming games (free call) | odds |
| `raw/odds/basketball_nba/snapshot_date=/snapshot_ts=/event=<id>.json.gz` | Player-points lines per book | odds |
| `raw/espn/injuries/snapshot_date=/snapshot_ts=.json.gz` | League injury report | odds |
| `raw/espn/scoreboard/game_date=/scoreboard.json.gz` | Games for an ET date | boxscores |
| `raw/espn/summary/game_date=/event=<id>.json.gz` | Full game summary: box score, PBP, injuries | boxscores |

Every file is `{"_meta": {source, params, fetched_at, ...}, "data": <untouched API response>}`.
API keys are stripped from `_meta`.

## Design notes

- **ESPN, not stats.nba.com:** stats.nba.com and cdn.nba.com block cloud IPs.
- **No Lambda dependencies:** stdlib + the runtime's boto3, so Terraform zips `src/` directly.
- **Idempotent:** box-score ingestion skips games already stored and re-checks the last 3 days, so late finals and retries are safe.
- **Credit guard:** the odds job stops (and alarms) before The Odds API free quota runs out.
- **Databricks gets data pushed, not pulled:** this AWS project's managed resource control policy denies S3/STS calls from
  principals outside the organization, so Databricks can't read the bucket via a storage credential. Instead every new
  object triggers `nba-lakehouse-databricks-sync`, which copies it to the Unity Catalog volume `/Volumes/nba/raw/landing`
  (same paths as the bucket) through the Files API, signed in as a service principal with access to that volume only.
  `python -m ingest.databricks_sync --bucket <bucket>` re-copies everything.
- **Secret handling:** the API key lives in an SSM SecureString created by hand, so it never appears in Terraform state or git.

## Deploy

Prereqs: AWS CLI signed in with profile `nba-lakehouse` (`aws login --profile nba-lakehouse`), Terraform ≥ 1.6, a free key from <https://the-odds-api.com>.

```bash
# 1. Store the odds key (once)
aws ssm put-parameter --profile nba-lakehouse --region us-east-2 \
  --name /nba-lakehouse/odds-api-key --type SecureString --value "<YOUR_KEY>"

# 2. Provision
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set alert_email
terraform init && terraform apply

# 3. Confirm the SNS subscription email AWS sends you, then smoke-test:
aws lambda invoke --profile nba-lakehouse --region us-east-2 \
  --function-name nba-lakehouse-boxscores --payload '{"date":"2026-10-15"}' \
  --cli-binary-format raw-in-base64-out /dev/stdout
```

## Databricks

`databricks/` is a Databricks Asset Bundle. Job `nba-lakehouse-bronze-silver` (6:30am and 5:30pm ET):

- **bronze:** Auto Loader loads new landing-volume files into `nba.bronze` (one row per file, JSON kept as a string)
  and reloads the historical parquet tables.
- **silver:** rebuilds `nba.silver` from bronze: prop lines (one row per game × book × player × line, over/under side
  by side), moneylines, ESPN games/box scores/injuries, the cleaned historical tables, and crosswalks:
  - Odds API games → NBA.com `game_id`, and sportsbook player names → NBA.com `player_id`. Names are matched only
    against the two teams in that game, strict rules before loose ones (`databricks/notebooks/name_match.py`,
    tested in `tests/test_name_match.py`); 58,355 of 58,372 player-games match, the rest are ambiguous ("J. Green").
    A name shared by two different players (Chris Johnson ×2 in 2012-13) is left unmatched, never guessed.
  - ESPN player ids → NBA.com `player_id`, so a player's history continues past 2025-26 (NBA.com data ends there,
    ESPN feeds the live seasons). In the 2,637 games both sources cover, an identical stat line on the same team
    in the same game identifies the player regardless of spelling; an id takes the match ≥ 90% of its games agree
    on, in both directions (687 players, median 84 games each). Players with no shared games fall back to an
    exact, unambiguous name; rookies stay null.

  - One timeline (`all_games`, `all_player_games`, `all_team_games`): NBA.com through 2025-26, ESPN games after,
    players keyed by `player_key` (NBA.com id, or `espn:<id>` for rookies). Odds games and names are matched
    against it, including tonight's games before they're played.
- **gold:** `props`, one row per player per game with a line: the consensus line (the one most books post),
  the market's no-vig over probability, best prices, game spread/total, features computed only from earlier
  games (recent and season scoring, minutes, shot volume, usage share, rest, team pace, opponent defensive
  rating), and once played the points and result (over/under/push; a player who sits out is a void bet).
  Tonight's props sit in the same table with no result, so the model scores them the way it was backtested.
  54,632 graded props from May 2023 to June 2026.

```bash
cd databricks && databricks bundle deploy && databricks bundle run bronze_ingest   # bronze -> silver -> gold
```

## Historical backfill

`src/ingest/backfill.py` pulls past lines from The Odds API's historical endpoints (paid plans):
player-points props 1h before each tip since 2023-05-03, plus game moneylines since 2022-23.
It writes into the same `raw/odds/` layout (`_meta.historical = true`), skips files already
stored, and stops at a credit reserve. Roughly 11 credits per game day plus 10 per game.

```bash
cd src && AWS_PROFILE=nba-lakehouse uv run --python 3.12 --with boto3 --with 'botocore[crt]' \
  python -m ingest.backfill --start 2022-10-18 --end 2026-06-30 --out s3://<bucket>
```

ESPN box scores for past seasons (free) use the same layout as the daily job:

```bash
cd src && AWS_PROFILE=nba-lakehouse uv run --python 3.12 --with boto3 --with 'botocore[crt]' \
  python -m ingest.boxscores --start 2024-10-22 --end 2026-06-30 --out s3://<bucket>
```

## Test

```bash
uv run --python 3.12 --with pytest pytest -q tests
```
