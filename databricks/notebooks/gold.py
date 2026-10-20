# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: one row per player prop, with pre-game features and the result
# MAGIC
# MAGIC | Table | One row per |
# MAGIC |---|---|
# MAGIC | `prop_lines` | game × player × sportsbook: that book's last line before tip-off |
# MAGIC | `player_features` | player × game (played or upcoming): stats from **earlier** games only |
# MAGIC | `team_features` | team × game (played or upcoming): pace and ratings from earlier games only |
# MAGIC | `props` | game × player: consensus line, market probability, features, and (once played) points and result |
# MAGIC
# MAGIC Every feature window ends at the game *before* the one it describes, so nothing a model sees
# MAGIC could not have been known at tip-off. Upcoming games (tonight's slate) sit in the same tables
# MAGIC with no result yet: the model scores them exactly the way it was backtested.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = "nba"
S, G = f"{CATALOG}.silver", f"{CATALOG}.gold"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {G} COMMENT 'Model-ready tables built from silver'")


def save(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.{name}")
    return spark.table(f"{G}.{name}")


def implied(price):
    """American odds -> implied probability (vig included)."""
    return F.when(price > 0, 100 / (price + 100)).otherwise(-price / (-price + 100))


def season_of(date_col):
    y = F.when(F.month(date_col) >= 8, F.year(date_col)).otherwise(F.year(date_col) - 1)
    return F.concat(y.cast("string"), F.lit("-"), F.lpad(((y + 1) % 100).cast("string"), 2, "0"))


events = spark.table(f"{S}.odds_events")
# An Odds API game with no played game_id is pending only if it's recent; older ones never happened
# as listed (moved, or not in the stats feed, like the 2023 In-Season Tournament final) and are dropped.
pending = events.where(F.col("commence_time") >= F.current_timestamp() - F.expr("INTERVAL 36 HOURS")).select("event_id")
xref = (
    spark.table(f"{S}.odds_player_xref").where("player_key IS NOT NULL")
    .join(pending.withColumn("is_pending", F.lit(True)), "event_id", "left")
    .where("game_id IS NOT NULL OR is_pending").drop("is_pending")
)
# A played game is identified by its game_id; a game not played yet by its Odds API event.
row_id = F.coalesce(F.col("game_id"), F.concat(F.lit("event:"), F.col("event_id")))

# COMMAND ----------

# MAGIC %md ## Lines
# MAGIC Each book's last pre-tip snapshot with both sides posted. A player listed under two spellings
# MAGIC ("Anthony Davis", "Anthony Davis Jr.") collapses to one `player_key`; the latest quote wins.

# COMMAND ----------

quotes = (
    spark.table(f"{S}.odds_player_points")
    .where("market = 'player_points' AND snapshot_at < commence_time AND over_price IS NOT NULL AND under_price IS NOT NULL")
    .join(xref.select("event_id", "player_name", "player_key", "game_id"), ["event_id", "player_name"])
)
last_quote = Window.partitionBy("event_id", "player_key", "bookmaker").orderBy(F.desc("snapshot_at"), F.desc("book_updated_at"))
p_over, p_under = implied(F.col("over_price")), implied(F.col("under_price"))
prop_lines = save(
    quotes.withColumn("rn", F.row_number().over(last_quote)).where("rn = 1")
    .select("event_id", "game_id", "player_key", "player_name", "bookmaker", "commence_time", "game_date",
            "snapshot_at", ((F.unix_timestamp("commence_time") - F.unix_timestamp("snapshot_at")) / 60).cast("int").alias("minutes_before_tip"),
            "line", "over_price", "under_price",
            (p_over / (p_over + p_under)).alias("p_over_novig"), (p_over + p_under - 1).alias("vig")),
    "prop_lines",
)

# Consensus per player-game: the line most books post (ties -> the lower line), the average no-vig
# probability among books at that line, and the best price on each side there.
by_line = prop_lines.groupBy("event_id", "game_id", "player_key", "line").agg(
    F.count("*").alias("books_at_line"), F.avg("p_over_novig").alias("market_p_over"),
    F.max("over_price").alias("best_over_price"), F.max("under_price").alias("best_under_price"),
)
modal = Window.partitionBy("event_id", "player_key").orderBy(F.desc("books_at_line"), "line")
consensus = (
    by_line.withColumn("rn", F.row_number().over(modal)).where("rn = 1").drop("rn")
    .join(prop_lines.groupBy("event_id", "player_key").agg(
        F.count("*").alias("n_books"), F.min("line").alias("line_min"), F.max("line").alias("line_max"),
        F.max("snapshot_at").alias("last_snapshot_at"), F.max_by("player_name", "snapshot_at").alias("player_name"),
    ), ["event_id", "player_key"])
)

# COMMAND ----------

# MAGIC %md ## Player features
# MAGIC The player timeline (every game played) plus one row per upcoming prop game. The upcoming row's
# MAGIC team is the one of the two teams the player last played for.

# COMMAND ----------

played = spark.table(f"{S}.all_player_games").select(
    "player_key", "player", "game_id", F.lit(None).cast("string").alias("event_id"), "season", "season_type", "game_date",
    "team", "opp", "is_home", "min", "pts", "fga", "fta", "fg3a", "fg3m", "tov", "reb", "ast",
)
latest = spark.table(f"{S}.player_latest_team").select("player_key", F.col("player").alias("latest_name"), F.col("team").alias("latest_team"))
upcoming = (
    xref.where("game_id IS NULL").select("event_id", "player_key").distinct()
    .join(events.select("event_id", "game_date", "home_team", "away_team"), "event_id")
    .join(latest, "player_key")
    .where("latest_team IN (home_team, away_team)")
    .select("player_key", F.col("latest_name").alias("player"), F.lit(None).cast("string").alias("game_id"), "event_id",
            season_of(F.col("game_date")).alias("season"), F.lit("regular").alias("season_type"), "game_date",
            F.col("latest_team").alias("team"),
            F.when(F.col("latest_team") == F.col("home_team"), F.col("away_team")).otherwise(F.col("home_team")).alias("opp"),
            (F.col("latest_team") == F.col("home_team")).alias("is_home"),
            *[F.lit(None).cast("long").alias(c) for c in ["min", "pts", "fga", "fta", "fg3a", "fg3m", "tov", "reb", "ast"]])
)
team_shot_volume = spark.table(f"{S}.all_team_games").select("game_id", "team", F.col("fga").alias("team_fga"),
                                                             F.col("fta").alias("team_fta"), F.col("tov").alias("team_tov"))
timeline = (
    played.unionByName(upcoming)
    .join(team_shot_volume, ["game_id", "team"], "left")
    .withColumn("row_id", row_id)
    .withColumn("season_end", F.substring("season", 1, 4).cast("int") + 1)
)

order = ["game_date", "row_id"]
career = Window.partitionBy("player_key").orderBy(*order)
last = lambda n: career.rowsBetween(-n, -1)
this_season = Window.partitionBy("player_key", "season").orderBy(*order).rowsBetween(Window.unboundedPreceding, -1)
usage = F.col("fga") + 0.44 * F.col("fta") + F.col("tov")
team_usage = F.col("team_fga") + 0.44 * F.col("team_fta") + F.col("team_tov")

prev_season = (
    played.groupBy("player_key", "season").agg(F.avg("pts").alias("pts_prev_season"), F.avg("min").alias("min_prev_season"),
                                                F.count("*").alias("games_prev_season"))
    .withColumn("season_end", F.substring("season", 1, 4).cast("int") + 2).drop("season")
)

player_features = save(
    timeline.select(
        "row_id", "player_key", "player", "game_id", "event_id", "season", "season_type", "game_date", "team", "opp", "is_home",
        "pts", "min",
        *[F.avg("pts").over(last(n)).alias(f"pts_l{n}") for n in (5, 10, 20)],
        F.stddev("pts").over(last(10)).alias("pts_sd_l10"),
        *[F.avg("min").over(last(n)).alias(f"min_l{n}") for n in (5, 10)],
        *[F.avg(c).over(last(10)).alias(f"{c}_l10") for c in ("fga", "fta", "fg3a")],
        F.try_divide(F.sum("pts").over(last(10)), F.sum("min").over(last(10))).alias("pts_per_min_l10"),
        F.try_divide(F.sum(usage).over(last(10)), F.sum(team_usage).over(last(10))).alias("usage_share_l10"),
        F.avg("pts").over(this_season).alias("pts_season"),
        F.avg("min").over(this_season).alias("min_season"),
        F.count("pts").over(this_season).alias("games_season"),
        F.count("pts").over(career.rowsBetween(Window.unboundedPreceding, -1)).alias("games_career"),
        F.datediff("game_date", F.lag("game_date").over(career)).alias("days_rest"),
        "season_end",
    )
    .join(prev_season, ["player_key", "season_end"], "left").drop("season_end"),
    "player_features",
)

# COMMAND ----------

# MAGIC %md ## Team features
# MAGIC Possessions are estimated the same way for both sources: FGA + 0.44·FTA − OREB + TOV.

# COMMAND ----------

team_played = spark.table(f"{S}.all_team_games").select(
    "game_id", F.lit(None).cast("string").alias("event_id"), "season", "game_date", "team", "pts", "opp_pts",
    (F.col("fga") + 0.44 * F.col("fta") - F.col("oreb") + F.col("tov")).alias("poss"),
)
upcoming_events = events.join(xref.where("game_id IS NULL").select("event_id").distinct(), "event_id")
team_upcoming = upcoming_events.select(
    F.lit(None).cast("string").alias("game_id"), "event_id", season_of(F.col("game_date")).alias("season"), "game_date",
    F.explode(F.array("home_team", "away_team")).alias("team"),
    *[F.lit(None).cast(t).alias(c) for c, t in (("pts", "long"), ("opp_pts", "long"), ("poss", "double"))],
)
team_timeline = team_played.unionByName(team_upcoming).withColumn("row_id", row_id)
team_last10 = Window.partitionBy("team").orderBy("game_date", "row_id").rowsBetween(-10, -1)
team_features = save(
    team_timeline.select(
        "row_id", "team", "game_date",
        F.avg("poss").over(team_last10).alias("pace_l10"),
        (100 * F.try_divide(F.sum("pts").over(team_last10), F.sum("poss").over(team_last10))).alias("off_rtg_l10"),
        (100 * F.try_divide(F.sum("opp_pts").over(team_last10), F.sum("poss").over(team_last10))).alias("def_rtg_l10"),
    ),
    "team_features",
)

# COMMAND ----------

# MAGIC %md ## Props
# MAGIC Game lines: the NBA.com closing spread/total for history, else the median of the books' last
# MAGIC pre-tip quotes from the live snapshots. The result is null until the game is played; a player
# MAGIC with a line who then sits out gets `did_play = false` (sportsbooks void those bets).

# COMMAND ----------

live_lines = (
    spark.table(f"{S}.odds_game_lines").where("snapshot_at < commence_time")
    .withColumn("rn", F.row_number().over(Window.partitionBy("event_id", "bookmaker").orderBy(F.desc("snapshot_at"))))
    .where("rn = 1")
    .groupBy("event_id").agg(F.percentile("home_spread", 0.5).alias("live_home_spread"), F.percentile("total", 0.5).alias("live_total"))
)
games = spark.table(f"{S}.all_games").select("game_id", "home_spread_close", "total_close")
own = player_features.alias("f")
team_f = team_features.select("row_id", "team", *[F.col(c).alias(f"team_{c}") for c in ("pace_l10", "off_rtg_l10", "def_rtg_l10")])
opp_f = team_features.select("row_id", F.col("team").alias("opp"), *[F.col(c).alias(f"opp_{c}") for c in ("pace_l10", "off_rtg_l10", "def_rtg_l10")])
FEATURES = [c for c in player_features.columns
            if c not in ("row_id", "player_key", "player", "game_id", "event_id", "season", "season_type", "game_date", "team", "opp", "is_home", "pts", "min")]

home_spread = F.coalesce(F.col("home_spread_close"), F.col("live_home_spread"))
team_spread = F.when(F.col("is_home"), home_spread).otherwise(-home_spread)
total = F.coalesce(F.col("total_close"), F.col("live_total"))

props = (
    consensus.withColumn("row_id", row_id)
    .join(events.select("event_id", "commence_time", "game_date", "home_team", "away_team"), "event_id")
    .join(own.select("row_id", "player_key", "season", "season_type", "team", "opp", "is_home",
                     F.col("pts").alias("actual_pts"), F.col("min").alias("actual_min"), *FEATURES),
          ["row_id", "player_key"], "left")
    .join(team_f, ["row_id", "team"], "left")
    .join(opp_f, ["row_id", "opp"], "left")
    .join(games, "game_id", "left")
    .join(live_lines, "event_id", "left")
    .select(
        "event_id", "game_id", "player_key", "player_name", season_of(F.col("game_date")).alias("season"),
        "season_type", "game_date", "commence_time", "home_team", "away_team", "team", "opp", "is_home",
        "line", "market_p_over", "best_over_price", "best_under_price", "n_books", "books_at_line", "line_min", "line_max",
        "last_snapshot_at",
        team_spread.alias("team_spread"), total.alias("game_total"), (total / 2 - team_spread / 2).alias("team_implied_total"),
        *FEATURES,
        "team_pace_l10", "team_off_rtg_l10", "team_def_rtg_l10", "opp_pace_l10", "opp_off_rtg_l10", "opp_def_rtg_l10",
        F.col("game_id").isNull().alias("is_upcoming"),
        F.when(F.col("game_id").isNotNull(), F.col("actual_pts").isNotNull()).alias("did_play"),
        "actual_pts", "actual_min",
        F.when(F.col("actual_pts") > F.col("line"), "over").when(F.col("actual_pts") < F.col("line"), "under")
         .when(F.col("actual_pts") == F.col("line"), "push").alias("result"),
    )
)
props = save(props, "props")

# COMMAND ----------

# MAGIC %md ## Quality checks

# COMMAND ----------

checks = {
    "prop rows": props.count(),
    "prop rows with no features (sat out, or team unknown)": props.where("team IS NULL").count(),
    "played, player sat out (void)": props.where("did_play = false").count(),
    "played with a result": props.where("result IS NOT NULL").count(),
    "upcoming": props.where("is_upcoming").count(),
    "missing game total": props.where("game_total IS NULL").count(),
    "missing pts_l10 (no prior games)": props.where("pts_l10 IS NULL").count(),
    "duplicate (event, player) rows (must be 0)": props.groupBy("event_id", "player_key").count().where("count > 1").count(),
}
display(spark.createDataFrame([(k, int(v)) for k, v in checks.items()], "check string, value long"))
# How sharp is the market? Over rate by season, and the line vs. the player's recent average.
display(props.where("result IN ('over', 'under')").groupBy("season").agg(
    F.count("*").alias("props"), F.avg((F.col("result") == "over").cast("int")).alias("over_rate"),
    F.avg("market_p_over").alias("avg_market_p_over"), F.avg(F.col("line") - F.col("pts_l10")).alias("line_minus_pts_l10"),
).orderBy("season"))
display(props.where("is_upcoming").orderBy("commence_time", F.desc("line")))
