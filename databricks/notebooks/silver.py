# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: parsed, typed, linked
# MAGIC
# MAGIC Rebuilt in full from `nba.bronze` on every run (the data is small, and a full rebuild means a
# MAGIC parsing fix applies to all history at once).
# MAGIC
# MAGIC | Table | One row per |
# MAGIC |---|---|
# MAGIC | `odds_events` | Odds API game |
# MAGIC | `odds_player_points` | game × snapshot × sportsbook × player × line, over/under prices side by side |
# MAGIC | `odds_moneylines` | game × snapshot × sportsbook |
# MAGIC | `espn_games`, `espn_player_box`, `espn_injuries` | ESPN game / player-game / injury-report entry |
# MAGIC | `games`, `team_games`, `player_games`, `player_seasons`, `teams`, `schedule_2026_27` | historical NBA.com tables, cleaned |
# MAGIC | `odds_game_xref` | Odds API game ↔ NBA.com `game_id` |
# MAGIC | `odds_player_xref` | Odds API player name in a game ↔ NBA.com `player_id` |
# MAGIC | `espn_game_xref` | ESPN game ↔ NBA.com `game_id` |
# MAGIC | `espn_player_xref` | ESPN player ↔ NBA.com `player_id` |
# MAGIC
# MAGIC Team names are mapped to NBA.com abbreviations everywhere, so every table joins on the same codes.

# COMMAND ----------

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = "nba"
B, S = f"{CATALOG}.bronze", f"{CATALOG}.silver"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {S} COMMENT 'Parsed, typed and linked tables built from bronze'")

TEAM_BY_NAME = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
ESPN_ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


# element_at() reads a bare string as a literal key, not a column name, so always pass a Column.
def team_from_name(col):
    col = F.col(col) if isinstance(col, str) else col
    return F.element_at(F.create_map(*[F.lit(x) for kv in TEAM_BY_NAME.items() for x in kv]), col)


def team_from_espn(col):
    col = F.col(col) if isinstance(col, str) else col
    return F.coalesce(F.element_at(F.create_map(*[F.lit(x) for kv in ESPN_ABBR.items() for x in kv]), col), col)


def espn_ts(col):
    """ESPN times omit seconds: 2026-10-15T23:30Z."""
    return F.coalesce(F.try_to_timestamp(col, F.lit("yyyy-MM-dd'T'HH:mmX")), F.try_to_timestamp(col))


def et_date(ts_col):
    return F.to_date(F.from_utc_timestamp(ts_col, "America/New_York"))


def save(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{S}.{name}")
    return spark.table(f"{S}.{name}")

# COMMAND ----------

# MAGIC %md ## Odds API

# COMMAND ----------

META = "struct<historical:boolean, snapshot_timestamp:string, fetched_at:string>"
OUTCOME = "struct<name:string, description:string, price:double, point:double>"
BOOK = f"struct<key:string, title:string, last_update:string, markets:array<struct<key:string, last_update:string, outcomes:array<{OUTCOME}>>>>"
EVENT = f"struct<id:string, commence_time:string, home_team:string, away_team:string, bookmakers:array<{BOOK}>>"

odds = spark.table(f"{B}.odds")


def snapshot_cols(meta):
    return [
        F.to_timestamp("snapshot_ts", "yyyyMMdd'T'HHmmss'Z'").alias("snapshot_requested_at"),
        # Historical files carry the API's snapshot time; live files were fetched in real time.
        F.to_timestamp(F.coalesce(meta["snapshot_timestamp"], meta["fetched_at"])).alias("snapshot_at"),
        F.coalesce(meta["historical"], F.lit(False)).alias("is_historical"),
    ]


def game_cols(ev):
    start = F.to_timestamp(ev["commence_time"])
    return [
        ev["id"].alias("event_id"), start.alias("commence_time"), et_date(start).alias("game_date"),
        team_from_name(ev["home_team"]).alias("home_team"), team_from_name(ev["away_team"]).alias("away_team"),
    ]


# Games: latest listing of each event wins (start times can move).
events = (
    odds.where("file_kind = 'events'")
    .select(F.from_json("raw_json", f"struct<_meta:{META}, data:array<{EVENT}>>").alias("j"), "snapshot_ts")
    .select(F.explode("j.data").alias("ev"), "snapshot_ts")
    .select(*game_cols(F.col("ev")), "snapshot_ts")
)
latest = F.row_number().over(Window.partitionBy("event_id").orderBy(F.desc("snapshot_ts")))
odds_events = save(events.withColumn("rn", latest).where("rn = 1").drop("rn", "snapshot_ts"), "odds_events")

# Player points: one row per book x player x line, over and under side by side.
props = (
    odds.where("file_kind = 'event'")
    .select(F.from_json("raw_json", f"struct<_meta:{META}, data:{EVENT}>").alias("j"), "snapshot_ts")
    .select("j.*", "snapshot_ts")
    .select(*game_cols(F.col("data")), *snapshot_cols(F.col("_meta")), F.explode("data.bookmakers").alias("b"))
    .select("*", F.explode("b.markets").alias("m"))
    .select("*", F.explode("m.outcomes").alias("o"))
)
odds_player_points = save(
    props.groupBy(
        "event_id", "commence_time", "game_date", "home_team", "away_team",
        "snapshot_requested_at", "snapshot_at", "is_historical",
        F.col("b.key").alias("bookmaker"), F.col("m.key").alias("market"),
        F.col("o.description").alias("player_name"), F.col("o.point").alias("line"),
    ).agg(
        F.max(F.when(F.col("o.name") == "Over", F.col("o.price"))).alias("over_price"),
        F.max(F.when(F.col("o.name") == "Under", F.col("o.price"))).alias("under_price"),
        F.max(F.to_timestamp("m.last_update")).alias("book_updated_at"),
    ),
    "odds_player_points",
)

# Moneylines: the h2h files hold every game on the slate.
h2h = (
    odds.where("file_kind = 'h2h'")
    .select(F.from_json("raw_json", f"struct<_meta:{META}, data:array<{EVENT}>>").alias("j"), "snapshot_ts")
    .select(F.explode("j.data").alias("ev"), "j._meta", "snapshot_ts")
    .select(*game_cols(F.col("ev")), *snapshot_cols(F.col("_meta")), F.col("ev.home_team").alias("home_name"),
            F.col("ev.away_team").alias("away_name"), F.explode("ev.bookmakers").alias("b"))
    .select("*", F.explode("b.markets").alias("m"))
    .select("*", F.explode("m.outcomes").alias("o"))
)
odds_moneylines = save(
    h2h.groupBy(
        "event_id", "commence_time", "game_date", "home_team", "away_team",
        "snapshot_requested_at", "snapshot_at", "is_historical", F.col("b.key").alias("bookmaker"),
    ).agg(
        F.max(F.when(F.col("o.name") == F.col("home_name"), F.col("o.price"))).alias("home_price"),
        F.max(F.when(F.col("o.name") == F.col("away_name"), F.col("o.price"))).alias("away_price"),
        F.max(F.to_timestamp("m.last_update")).alias("book_updated_at"),
    )
    # A slate snapshot taken after a game tipped off has in-game prices; keep pre-game only.
    .where("snapshot_at < commence_time"),
    "odds_moneylines",
)

unmapped = odds_events.where("home_team IS NULL OR away_team IS NULL").count()
assert unmapped == 0, f"{unmapped} Odds API games have a team name missing from TEAM_BY_NAME"

# COMMAND ----------

# MAGIC %md ## ESPN

# COMMAND ----------

ATHLETE = ("struct<starter:boolean, didNotPlay:boolean, reason:string, ejected:boolean, stats:array<string>, "
           "athlete:struct<id:string, displayName:string, position:struct<abbreviation:string>>>")
TEAM_BOX = f"struct<team:struct<id:string, abbreviation:string>, statistics:array<struct<names:array<string>, athletes:array<{ATHLETE}>>>>"
COMPETITOR = "struct<homeAway:string, score:string, winner:boolean, team:struct<id:string, abbreviation:string>>"
HEADER = f"struct<id:string, season:struct<year:int, type:int>, competitions:array<struct<date:string, neutralSite:boolean, competitors:array<{COMPETITOR}>>>>"
SEASON_TYPE = F.create_map(F.lit(1), F.lit("preseason"), F.lit(2), F.lit("regular"), F.lit(3), F.lit("playoffs"), F.lit(5), F.lit("playin"))

summary = spark.table(f"{B}.espn_summary").select(
    F.from_json("raw_json", f"struct<data:struct<boxscore:struct<players:array<{TEAM_BOX}>>, header:{HEADER}>>").alias("j"),
    F.col("event").alias("espn_game_id"), F.to_date("game_date").alias("game_date"),
)
header = summary.select(
    "espn_game_id", "game_date", F.col("j.data.header.season.year").alias("season_end"),
    F.element_at(SEASON_TYPE, F.col("j.data.header.season.type")).alias("season_type"),
    F.col("j.data.header.competitions")[0].alias("c"),
)
home = F.filter("c.competitors", lambda x: x["homeAway"] == "home")[0]
away = F.filter("c.competitors", lambda x: x["homeAway"] == "away")[0]
espn_games = save(
    header.select(
        "espn_game_id", "game_date", "season_end", "season_type",
        espn_ts(F.col("c.date")).alias("tipoff_at"), F.col("c.neutralSite").alias("is_neutral"),
        team_from_espn(home["team"]["abbreviation"]).alias("home_team"),
        team_from_espn(away["team"]["abbreviation"]).alias("away_team"),
        home["score"].cast("int").alias("home_pts"), away["score"].cast("int").alias("away_pts"),
    ),
    "espn_games",
)

players = (
    summary.select("espn_game_id", "game_date", F.explode("j.data.boxscore.players").alias("t"))
    .select("espn_game_id", "game_date", F.col("t.team.abbreviation").alias("espn_team"),
            F.col("t.statistics")[0].alias("g"))
    .select("*", F.explode("g.athletes").alias("a"))
)
stat = F.when(F.size("a.stats") > 0, F.map_from_arrays("g.names", "a.stats"))


# Some older ESPN box scores use "--" for an empty stat.
def made(name):
    return F.split(stat[name], "-")[0].try_cast("int")


def attempted(name):
    return F.split(stat[name], "-")[1].try_cast("int")


espn_player_box = save(
    players.select(
        "espn_game_id", "game_date", team_from_espn("espn_team").alias("team"),
        F.col("a.athlete.id").alias("espn_player_id"), F.col("a.athlete.displayName").alias("player"),
        F.col("a.athlete.position.abbreviation").alias("position"),
        F.col("a.starter").alias("starter"), F.col("a.didNotPlay").alias("did_not_play"),
        F.when(F.col("a.didNotPlay"), F.col("a.reason")).alias("dnp_reason"), F.col("a.ejected").alias("ejected"),
        stat["MIN"].try_cast("int").alias("min"), stat["PTS"].try_cast("int").alias("pts"),
        made("FG").alias("fgm"), attempted("FG").alias("fga"),
        made("3PT").alias("fg3m"), attempted("3PT").alias("fg3a"),
        made("FT").alias("ftm"), attempted("FT").alias("fta"),
        stat["OREB"].try_cast("int").alias("oreb"), stat["DREB"].try_cast("int").alias("dreb"), stat["REB"].try_cast("int").alias("reb"),
        stat["AST"].try_cast("int").alias("ast"), stat["STL"].try_cast("int").alias("stl"), stat["BLK"].try_cast("int").alias("blk"),
        stat["TO"].try_cast("int").alias("tov"), stat["PF"].try_cast("int").alias("pf"), stat["+/-"].try_cast("int").alias("plus_minus"),
    )
    # ESPN lists one 2025-26 Bulls player with no athlete record at all (no id, no name).
    .where("espn_player_id IS NOT NULL"),
    "espn_player_box",
)

INJURY = ("struct<status:string, date:string, shortComment:string, type:struct<description:string>, "
          "details:struct<type:string, location:string, side:string, returnDate:string>, "
          "athlete:struct<displayName:string, links:array<struct<href:string>>>>")
espn_injuries = save(
    spark.table(f"{B}.espn_injuries")
    .select(F.to_timestamp("snapshot_ts", "yyyyMMdd'T'HHmmss'Z'").alias("snapshot_at"),
            F.from_json("raw_json", f"struct<data:struct<injuries:array<struct<displayName:string, injuries:array<{INJURY}>>>>>").alias("j"))
    .select("snapshot_at", F.explode("j.data.injuries").alias("t"))
    .select("snapshot_at", team_from_name(F.col("t.displayName")).alias("team"), F.explode("t.injuries").alias("i"))
    .select(
        "snapshot_at", "team",
        F.regexp_extract(F.col("i.athlete.links")[0]["href"], r"/id/(\d+)", 1).alias("espn_player_id"),
        F.col("i.athlete.displayName").alias("player"), F.col("i.status").alias("status"),
        F.col("i.type.description").alias("status_detail"), espn_ts(F.col("i.date")).alias("reported_at"),
        F.col("i.details.type").alias("injury"), F.col("i.details.location").alias("body_part"),
        F.col("i.details.side").alias("side"), F.to_date("i.details.returnDate").alias("expected_return"),
        F.col("i.shortComment").alias("comment"),
    ),
    "espn_injuries",
)

# COMMAND ----------

# MAGIC %md ## Historical NBA.com tables

# COMMAND ----------

for name in ["games", "team_games", "player_games", "player_seasons", "teams", "schedule_2026_27"]:
    save(spark.table(f"{B}.nba_stats_{name}"), name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Crosswalks
# MAGIC Odds API games → NBA.com games by date and teams. Neutral-site games can list home/away
# MAGIC differently between sources, so the team pair is matched in either order.

# COMMAND ----------

nba_games = spark.table(f"{S}.games").select("game_id", "game_date", "home_team", "away_team")
pair = lambda h, a: F.array_sort(F.array(h, a))
same_teams = pair(F.col("o.home_team"), F.col("o.away_team")) == pair(F.col("g.home_team"), F.col("g.away_team"))
cols = ["o.event_id", "g.game_id", "o.game_date", "o.home_team", "o.away_team"]
same_day = odds_events.alias("o").join(nba_games.alias("g"), (F.col("o.game_date") == F.col("g.game_date")) & same_teams)
# Postponed games and playoff/play-in games listed before the schedule was final can be off by a
# day or two; accept those only when exactly one game between the two teams is that close.
near = (
    odds_events.join(same_day.select("o.event_id"), "event_id", "left_anti").alias("o")
    .join(nba_games.alias("g"), (F.abs(F.datediff("o.game_date", "g.game_date")) <= 2) & same_teams)
    .withColumn("n", F.count("*").over(Window.partitionBy("o.event_id"))).where("n = 1")
)
odds_game_xref = save(
    same_day.select(*cols, F.lit("same_day").alias("match_method"))
    .unionByName(near.select(*cols, F.lit("within_2_days").alias("match_method"))),
    "odds_game_xref",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Odds API player names → NBA.com `player_id`. Each name is matched only against players on the
# MAGIC two teams in that game, which keeps same-name players apart. Rules, in order:
# MAGIC 1. **exact** after normalizing (accents, punctuation, Jr./III suffixes)
# MAGIC 2. **nickname**: same last name and one first name starts with the other (Nic/Nicolas, Herb/Herbert)
# MAGIC 3. **all_words**: every word of exactly one NBA.com name appears ("Wembanyama Victor", "Brook Robert Lopez")
# MAGIC 4. **fuzzy**: closest name by similarity ≥ 0.85, only if no other candidate is close
# MAGIC 5. **surname**: exactly one candidate's surname appears and a first-name initial agrees
# MAGIC    ("Alfred Joel Horford Reynoso" → Al Horford)
# MAGIC
# MAGIC A name shared by two different candidates (Chris Johnson ×2 in 2012-13) is left unmatched as
# MAGIC `ambiguous_same_name` rather than guessed.
# MAGIC
# MAGIC Rules 1-2 run against that game's box score, then the teams' season rosters (a player with a
# MAGIC posted line who then sat out has no box-score row). Rules 3-5 run last, against both full
# MAGIC rosters, so a loose rule can never pick a teammate when the right player simply sat out.

# COMMAND ----------


from name_match import AMBIGUOUS, match_player, merge, name_index, norm  # noqa: E402  (notebooks/name_match.py)

wanted = (
    odds_player_points.select("event_id", "player_name").distinct()
    .join(odds_game_xref.select("event_id", "game_id", "home_team", "away_team"), "event_id")
    .toPandas()
)
pg = spark.table(f"{S}.player_games").select("game_id", "season", "team", "player_id", "player")
box = pg.join(odds_game_xref.select("game_id"), "game_id").toPandas()
seasons = pg.join(odds_game_xref.select("game_id"), "game_id").select("season").distinct()
season_roster = pg.join(seasons, "season").select("season", "team", "player_id", "player").distinct().toPandas()
game_season = pg.select("game_id", "season").distinct().join(odds_game_xref.select("game_id"), "game_id").toPandas()

box_by_game = {g: name_index(zip(d.player, d.player_id)) for g, d in box.groupby("game_id")}
roster_by_team = {k: name_index(zip(d.player, d.player_id)) for k, d in season_roster.groupby(["season", "team"])}
season_of = dict(zip(game_season.game_id, game_season.season))

rows = []
for r in wanted.itertuples(index=False):
    season = season_of.get(r.game_id)
    roster = merge(roster_by_team.get((season, r.home_team), {}), roster_by_team.get((season, r.away_team), {}))
    box_players = box_by_game.get(r.game_id, {})
    pid, method = match_player(r.player_name, box_players, roster)
    rows.append((r.event_id, r.game_id, r.player_name, None if pid is None else int(pid), method))

xref = spark.createDataFrame(
    pd.DataFrame(rows, columns=["event_id", "game_id", "player_name", "player_id", "match_method"]),
    "event_id string, game_id string, player_name string, player_id long, match_method string",
)
odds_player_xref = save(xref, "odds_player_xref")

# COMMAND ----------

# MAGIC %md
# MAGIC ESPN → NBA.com. NBA.com history ends with 2025-26; from 2026-27 box scores come from ESPN, so
# MAGIC each ESPN player id needs its NBA.com id to keep one continuous history per player.
# MAGIC
# MAGIC Games where both sources overlap (2024-25 and 2025-26) are matched by date and teams. Within a
# MAGIC game, a player's stat line (points, rebounds, assists, shots made/attempted, ...) is nearly a
# MAGIC fingerprint: the same line on the same team in the same game is the same player, whatever the
# MAGIC name says. Lines shared by two teammates (two 0-0-0 bench cameos) are skipped. An ESPN id takes
# MAGIC the NBA.com id that most of its games agree on, provided ≥ 90% agree, both directions.
# MAGIC ESPN players with no overlapping games (back from a long absence) fall back to an exact,
# MAGIC unambiguous name match; true rookies have no NBA.com id and stay null.

# COMMAND ----------

espn_game_xref = save(
    espn_games.alias("o").join(nba_games.alias("g"), (F.col("o.game_date") == F.col("g.game_date")) & same_teams)
    .select("o.espn_game_id", "g.game_id", "o.game_date", "o.home_team", "o.away_team"),
    "espn_game_xref",
)

LINE = ["pts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
unique_line = lambda df: (
    df.withColumn("n", F.count("*").over(Window.partitionBy("game_id", "team", *LINE))).where("n = 1").drop("n")
)
espn_lines = unique_line(
    espn_player_box.where("NOT did_not_play AND pts IS NOT NULL").join(espn_game_xref.select("espn_game_id", "game_id"), "espn_game_id")
    .select("game_id", "team", "espn_player_id", F.col("player").alias("espn_name"), *LINE)
)
nba_lines = unique_line(
    spark.table(f"{S}.player_games").join(espn_game_xref.select("game_id"), "game_id")
    .select("game_id", "team", "player_id", F.col("player").alias("nba_name"), *LINE)
)
votes = (
    espn_lines.join(nba_lines, ["game_id", "team", *LINE])
    .groupBy("espn_player_id", "player_id").agg(F.count("*").alias("games_matched"))
)
by_espn = Window.partitionBy("espn_player_id")
by_nba = Window.partitionBy("player_id")
stat_matches = (
    votes.withColumn("espn_share", F.col("games_matched") / F.sum("games_matched").over(by_espn))
    .withColumn("nba_share", F.col("games_matched") / F.sum("games_matched").over(by_nba))
    .where("espn_share >= 0.9 AND nba_share >= 0.9")
    .select("espn_player_id", "player_id", "games_matched", F.lit("stat_line").alias("match_method"))
)

espn_names = espn_player_box.groupBy("espn_player_id").agg(
    F.max_by("player", "game_date").alias("espn_name"), F.max("game_date").alias("espn_last_game"))
leftover = espn_names.join(stat_matches, "espn_player_id", "left_anti").toPandas()
nba_index = name_index(spark.table(f"{S}.player_games").groupBy("player_id").agg(F.max_by("player", "game_date").alias("player"))
                       .toPandas()[["player", "player_id"]].itertuples(index=False))
name_rows = []
for r in leftover.itertuples(index=False):
    pid = nba_index.get(norm(r.espn_name)) if r.espn_name else None
    method = "ambiguous_same_name" if pid == AMBIGUOUS else ("name_only" if pid is not None else None)
    name_rows.append((r.espn_player_id, None if pid in (None, AMBIGUOUS) else int(pid), method))
name_matches = spark.createDataFrame(
    pd.DataFrame(name_rows, columns=["espn_player_id", "player_id", "match_method"]),
    "espn_player_id string, player_id long, match_method string",
)

nba_names = spark.table(f"{S}.player_games").groupBy("player_id").agg(F.max_by("player", "game_date").alias("nba_name"))
espn_player_xref = save(
    stat_matches.unionByName(name_matches.withColumn("games_matched", F.lit(None).cast("long")))
    .join(espn_names, "espn_player_id").join(nba_names, "player_id", "left")
    .select("espn_player_id", "espn_name", "player_id", "nba_name", "match_method", "games_matched", "espn_last_game"),
    "espn_player_xref",
)

# COMMAND ----------

# MAGIC %md ## Quality checks

# COMMAND ----------

checks = {
    "odds games": odds_events.count(),
    "odds games matched to NBA.com (games through 2025-26)": odds_game_xref.count(),
    "odds games in NBA.com date range but unmatched": odds_events.join(odds_game_xref, "event_id", "left_anti")
        .where(F.col("game_date") <= nba_games.agg(F.max("game_date")).first()[0]).count(),
    "prop rows": odds_player_points.count(),
    "prop rows with a missing over or under price": odds_player_points.where("over_price IS NULL OR under_price IS NULL").count(),
    "player-game names to match": odds_player_xref.count(),
    "player-game names unmatched": odds_player_xref.where("player_id IS NULL").count(),
    "moneyline rows": odds_moneylines.count(),
    "ESPN player-game rows": espn_player_box.count(),
    "ESPN games matched to NBA.com": espn_game_xref.count(),
    "ESPN players": espn_player_xref.count(),
    "ESPN players matched to NBA.com": espn_player_xref.where("player_id IS NOT NULL").count(),
    "NBA.com ids claimed by 2+ ESPN ids (must be 0)": espn_player_xref.where("player_id IS NOT NULL")
        .groupBy("player_id").count().where("count > 1").count(),
}
display(spark.createDataFrame([(k, int(v)) for k, v in checks.items()], "check string, value long"))
display(odds_player_xref.groupBy("match_method").count().orderBy(F.desc("count")))
display(odds_player_xref.where("player_id IS NULL").groupBy("player_name").count().orderBy(F.desc("count")).limit(50))
display(espn_player_xref.groupBy("match_method").count().orderBy(F.desc("count")))
# Stat-line matches whose names differ: worth a glance, but the stats decide.
display(espn_player_xref.where(F.col("match_method") == "stat_line")
        .where(F.lower(F.regexp_replace("espn_name", "[^A-Za-z]", "")) != F.lower(F.regexp_replace("nba_name", "[^A-Za-z]", ""))))
display(espn_player_xref.where("player_id IS NULL OR match_method != 'stat_line'").orderBy(F.desc("espn_last_game")))
