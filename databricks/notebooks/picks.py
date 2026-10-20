# Databricks notebook source
# MAGIC %md
# MAGIC # Daily picks
# MAGIC
# MAGIC Scores tonight's props with the same model the backtest evaluated, then **appends** new picks to
# MAGIC `nba.gold.pick_log`. A pick is written once, before tip-off, and never changed: later runs skip any
# MAGIC player-game already in the log, so a moved line or a better model can't rewrite history.
# MAGIC
# MAGIC | Table | What |
# MAGIC |---|---|
# MAGIC | `predictions` | every upcoming prop with the model's numbers (rebuilt each run) |
# MAGIC | `pick_log` | locked picks, append-only |
# MAGIC | `pick_record` | picks joined to results: won / lost / push / void and profit |

# COMMAND ----------

# MAGIC %run ./model_common

# COMMAND ----------

from datetime import datetime, timezone

from pyspark.sql import Window

PICK_EV = 0.02  # expected value per unit staked at the best available price
MAX_LINE_AGE_HOURS = 3  # only pick from fresh quotes: the backtest used lines from ~1h before tip

props = spark.table(f"{G}.props")
graded = props.where("did_play AND result IS NOT NULL").groupBy("season").count().toPandas()
# The dispersion and blend are fitted on the latest complete season with plenty of props,
# using a model that never saw it (exactly as in the backtest).
calibration_season = graded[graded["count"] >= 5000]["season"].max()
current_season = games_pd["season"].max()
if calibration_season == current_season:  # mid-way through a big season: use the one before
    calibration_season = sorted(graded[graded["count"] >= 5000]["season"])[-2]

early = fit_mu(games_pd[games_pd["season"] < calibration_season])
held = games_pd[games_pd["season"] == calibration_season]
dispersion = fit_dispersion(early.predict(held[FEATURES]), held["pts"].values)
cal = props.where("did_play AND result IS NOT NULL AND result != 'push'").where(F.col("season") == calibration_season)
cal = predict_props(early, dispersion, prep(cal.select(*PROP_COLS).toPandas()))
blend = LogisticRegression(C=1e3).fit(blend_x(cal), (cal["result"] == "over").astype(int))
model = fit_mu(games_pd)  # every game played so far, this season included

# COMMAND ----------

# MAGIC %md ## Score tonight's props

# COMMAND ----------

injury_key = F.coalesce(F.col("player_id").cast("string"), F.concat(F.lit("espn:"), F.col("espn_player_id")))
injuries = spark.table(f"{S}.espn_injuries")
listed_out = (
    injuries.where(F.col("snapshot_at") == injuries.agg(F.max("snapshot_at")).first()[0]).where("status = 'Out'")
    .join(spark.table(f"{S}.espn_player_xref").select("espn_player_id", "player_id"), "espn_player_id", "left")
    .select(injury_key.alias("player_key")).distinct().withColumn("listed_out", F.lit(True))
)
upcoming = (
    props.where("is_upcoming AND commence_time > current_timestamp() AND team IS NOT NULL")
    .where(F.col("last_snapshot_at") >= F.current_timestamp() - F.expr(f"INTERVAL {MAX_LINE_AGE_HOURS} HOURS"))
    .join(listed_out, "player_key", "left").fillna(False, subset=["listed_out"])
)
now = datetime.now(timezone.utc).replace(microsecond=0)

with mlflow.start_run(run_name=f"picks {now:%Y-%m-%d %H:%M}") as run:
    mlflow.log_params({"calibration_season": calibration_season, "pick_ev": PICK_EV, "min_career_games": MIN_CAREER_GAMES,
                       "blend_weight_model": float(blend.coef_[0][0]), "blend_weight_market": float(blend.coef_[0][1])})
    up = upcoming.select(*PROP_COLS, "listed_out").toPandas()
    if up.empty:
        print("No upcoming props to score.")
        preds = up
    else:
        preds = predict_props(model, dispersion, prep(up))
        preds["blend_p_over_cond"] = blend.predict_proba(blend_x(preds))[:, 1]
        no_push = 1 - preds["model_p_push"]
        preds["blend_p_over"], preds["blend_p_under"] = preds["blend_p_over_cond"] * no_push, (1 - preds["blend_p_over_cond"]) * no_push
        d_over, d_under = decimal(preds["best_over_price"].values), decimal(preds["best_under_price"].values)
        preds["ev_over"] = preds["blend_p_over"] * (d_over - 1) - preds["blend_p_under"]
        preds["ev_under"] = preds["blend_p_under"] * (d_under - 1) - preds["blend_p_over"]
        preds["side"] = np.where(preds["ev_over"] >= preds["ev_under"], "over", "under")
        preds["ev"] = preds[["ev_over", "ev_under"]].max(axis=1)
        preds["is_pick"] = (preds["ev"] > PICK_EV) & (preds["games_career"] >= MIN_CAREER_GAMES) & ~preds["listed_out"]
    mlflow.log_metrics({"upcoming_props": len(preds), "picks": int(preds["is_pick"].sum()) if len(preds) else 0})
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md ## Save predictions and lock new picks

# COMMAND ----------

PRED_COLS = ["event_id", "player_key", "player_name", "game_date", "commence_time", "line", "market_p_over",
             "best_over_price", "best_under_price", "mu", "blend_p_over", "blend_p_under", "side", "ev", "is_pick", "listed_out"]
PRED_SCHEMA = ("event_id string, player_key string, player_name string, game_date date, commence_time timestamp, line double, "
               "market_p_over double, best_over_price double, best_under_price double, mu double, blend_p_over double, "
               "blend_p_under double, side string, ev double, is_pick boolean, listed_out boolean")
pred_df = spark.createDataFrame(preds[PRED_COLS] if len(preds) else pd.DataFrame(columns=PRED_COLS), PRED_SCHEMA)
pred_df.withColumn("scored_at", F.lit(now)).withColumn("model_run_id", F.lit(run_id)) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.predictions")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {G}.pick_log (
  picked_at timestamp, model_run_id string, event_id string, player_key string, player_name string,
  game_date date, commence_time timestamp, line double, side string, price double, bookmaker string,
  model_p_win double, ev double
) COMMENT 'Locked picks: one row per player-game, written once before tip-off, never updated'
""")
# Where to get the best price: the book quoting it at the consensus line.
best_book = (
    spark.table(f"{G}.prop_lines").alias("l")
    .join(pred_df.where("is_pick").alias("p"), ["event_id", "player_key", "line"])
    .withColumn("price", F.when(F.col("p.side") == "over", F.col("l.over_price")).otherwise(F.col("l.under_price")))
    .withColumn("rn", F.row_number().over(Window.partitionBy("event_id", "player_key").orderBy(F.desc("price"))))
    .where("rn = 1").select("event_id", "player_key", "price", "bookmaker")
)
new_picks = (
    pred_df.where("is_pick").join(best_book, ["event_id", "player_key"])
    .join(spark.table(f"{G}.pick_log").select("event_id", "player_key"), ["event_id", "player_key"], "left_anti")
    .select(F.lit(now).alias("picked_at"), F.lit(run_id).alias("model_run_id"), "event_id", "player_key", "player_name",
            "game_date", "commence_time", "line", "side", "price", "bookmaker",
            F.when(F.col("side") == "over", F.col("blend_p_over")).otherwise(F.col("blend_p_under")).alias("model_p_win"), "ev")
)
new_picks.write.mode("append").saveAsTable(f"{G}.pick_log")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {G}.pick_record AS
SELECT l.*, p.actual_pts,
  CASE WHEN p.did_play = false THEN 'void'
       WHEN p.result IS NULL THEN 'pending'
       WHEN p.result = 'push' THEN 'push'
       WHEN p.result = l.side THEN 'won' ELSE 'lost' END AS outcome,
  CASE WHEN p.result IS NULL OR p.did_play = false OR p.result = 'push' THEN 0
       WHEN p.result = l.side THEN CASE WHEN l.price > 0 THEN l.price / 100 ELSE 100 / abs(l.price) END
       ELSE -1 END AS profit
FROM {G}.pick_log l
LEFT JOIN {G}.props p ON p.event_id = l.event_id AND p.player_key = l.player_key
""")
display(spark.table(f"{G}.predictions").orderBy(F.desc("ev")))
display(spark.table(f"{G}.pick_record").groupBy("outcome").agg(F.count("*").alias("picks"), F.round(F.sum("profit"), 2).alias("units")))
