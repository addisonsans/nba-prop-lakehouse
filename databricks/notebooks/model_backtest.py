# Databricks notebook source
# MAGIC %md
# MAGIC # Points model: walk-forward backtest against real prop lines
# MAGIC
# MAGIC **Model.** A gradient-boosted regressor (Poisson loss) predicts a player's expected points μ from
# MAGIC the pre-game features in `nba.gold`. Points are spread around μ as a negative binomial whose
# MAGIC dispersion depends on μ (fitted on a held-out season), which gives P(over), P(under) and P(push)
# MAGIC for any line. The line is never a feature: the model forms its own view first.
# MAGIC
# MAGIC **Blend.** A logistic regression combines the model's P(over) with the market's no-vig P(over),
# MAGIC fitted on the previous season's props. If the model adds nothing, it learns to copy the market.
# MAGIC
# MAGIC **Walk-forward.** To test season S: the model that fixes the dispersion and the blend is trained on
# MAGIC seasons ≤ S-2 and fitted on S-1; the model that scores S is trained on seasons ≤ S-1. Nothing from S
# MAGIC is used before it is scored.
# MAGIC
# MAGIC **Scoring.** Log loss of P(over) on over/under outcomes vs. the market's own probability (the bar to
# MAGIC beat), then flat 1-unit bets wherever expected value clears a threshold, at the best price across
# MAGIC books (line shopping) and at the median book's price. A market-only baseline (bet wherever the best
# MAGIC price beats the market's own no-vig probability) shows how much profit is line shopping, not the model.
# MAGIC Players with fewer than 10 career games are never bet: the model knows nothing about them.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

CATALOG = "nba"
S, G = f"{CATALOG}.silver", f"{CATALOG}.gold"
FIRST_TRAIN_SEASON = "2009-10"
TEST_SEASONS = ["2024-25", "2025-26"]  # 2023-24 has too few prior-season props to fit the blend
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
MIN_CAREER_GAMES = 10

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/nba-points-model")

# COMMAND ----------

# MAGIC %md ## Training frame: every player-game, same features as `gold.props`

# COMMAND ----------

pf = spark.table(f"{G}.player_features")
tf = spark.table(f"{G}.team_features")
team_cols = ["pace_l10", "off_rtg_l10", "def_rtg_l10"]
games = spark.table(f"{S}.all_games").select("game_id", "home_spread_close", "total_close")
home_spread = F.col("home_spread_close")
team_spread = F.when(F.col("is_home"), home_spread).otherwise(-home_spread)

frame = (
    pf.where("pts IS NOT NULL AND game_id IS NOT NULL").where(F.col("season") >= FIRST_TRAIN_SEASON)
    .join(tf.select("row_id", "team", *[F.col(c).alias(f"team_{c}") for c in team_cols]), ["row_id", "team"], "left")
    .join(tf.select("row_id", F.col("team").alias("opp"), *[F.col(c).alias(f"opp_{c}") for c in team_cols]), ["row_id", "opp"], "left")
    .join(games, "game_id", "left")
    .withColumn("team_spread", team_spread)
    .withColumn("game_total", F.col("total_close"))
    .withColumn("team_implied_total", F.col("game_total") / 2 - F.col("team_spread") / 2)
)

PLAYER = [c for c in pf.columns if c not in (
    "row_id", "player_key", "player", "game_id", "event_id", "season", "season_type", "game_date", "team", "opp", "is_home", "pts", "min")]
FEATURES = PLAYER + [f"team_{c}" for c in team_cols] + [f"opp_{c}" for c in team_cols] + [
    "team_spread", "game_total", "team_implied_total", "is_home_i", "is_playoffs_i"]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_home_i"] = df["is_home"].astype(float)
    df["is_playoffs_i"] = (df["season_type"] != "regular").astype(float)
    df[FEATURES] = df[FEATURES].astype(float)
    return df


games_pd = prep(frame.select("season", "season_type", "is_home", "pts", *[c for c in FEATURES if not c.endswith("_i")]).toPandas())
props_pd = prep(
    spark.table(f"{G}.props").where("did_play AND result IS NOT NULL")
    .select("event_id", "game_id", "player_key", "player_name", "game_date", "season", "season_type", "is_home", "line",
            "market_p_over", "best_over_price", "best_under_price", "median_over_decimal", "median_under_decimal",
            "n_books", "actual_pts", "result",
            *[c for c in FEATURES if not c.endswith("_i")])
    .toPandas()
)
print(f"{len(games_pd):,} player-games for training, {len(props_pd):,} graded props")

# COMMAND ----------

# MAGIC %md ## Model pieces

# COMMAND ----------

GBM_PARAMS = dict(loss="poisson", learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=0)


def fit_mu(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**GBM_PARAMS).fit(train[FEATURES], train["pts"])


def fit_dispersion(mu: np.ndarray, y: np.ndarray, bins: int = 20):
    """Negative binomial: var = mu + alpha * mu^2, with alpha estimated per bin of mu (held-out data)."""
    edges = np.unique(np.quantile(mu, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(mu, edges[1:-1]), 0, len(edges) - 2)
    centers, alphas = [], []
    for b in range(len(edges) - 1):
        m, r = mu[idx == b], y[idx == b] - mu[idx == b]
        centers.append(m.mean())
        alphas.append(max((np.mean(r ** 2) - m.mean()) / np.mean(m ** 2), 1e-3))
    return np.array(centers), np.array(alphas)


def line_probs(mu, line, dispersion):
    centers, alphas = dispersion
    alpha = np.interp(mu, centers, alphas)
    n = 1 / alpha
    p = n / (n + mu)
    p_under = nbinom.cdf(np.ceil(line) - 1, n, p)
    p_over = 1 - nbinom.cdf(np.floor(line), n, p)
    return p_over, p_under, np.clip(1 - p_over - p_under, 0, 1)


logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def blend_x(df):
    return np.column_stack([logit(df["model_p_over_cond"]), logit(df["market_p_over"])])


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def decimal(price):
    return np.where(price > 0, 1 + price / 100, 1 + 100 / np.abs(price))


def bets(df, p_over, p_under, d_over, d_under, threshold):
    """Flat 1-unit bet on the side with the higher expected value (decimal odds), if it clears the threshold."""
    ev_over, ev_under = p_over * (d_over - 1) - p_under, p_under * (d_under - 1) - p_over
    side = np.where(ev_over >= ev_under, "over", "under")
    ev = np.maximum(ev_over, ev_under)
    take = (ev > threshold) & (df["games_career"].values >= MIN_CAREER_GAMES)
    won = df["result"].values == side
    push = df["result"].values == "push"
    payout = np.where(side == "over", d_over - 1, d_under - 1)
    profit = np.where(push, 0.0, np.where(won, payout, -1.0))
    return take, side, ev, profit


def predict_props(model, dispersion, df):
    out = df.copy()
    out["mu"] = model.predict(out[FEATURES])
    out["model_p_over"], out["model_p_under"], out["model_p_push"] = line_probs(out["mu"].values, out["line"].values, dispersion)
    out["model_p_over_cond"] = out["model_p_over"] / (out["model_p_over"] + out["model_p_under"])
    return out

# COMMAND ----------

# MAGIC %md ## Walk-forward

# COMMAND ----------

season_list = sorted(games_pd["season"].unique())
scored, summary = [], []
for test in TEST_SEASONS:
    prior = season_list[season_list.index(test) - 1]
    with mlflow.start_run(run_name=f"backtest {test}"):
        mlflow.log_params({**GBM_PARAMS, "test_season": test, "calibration_season": prior,
                           "first_train_season": FIRST_TRAIN_SEASON, "n_features": len(FEATURES)})

        # 1. Dispersion and blend, fitted on season S-1 with a model that never saw it.
        early = fit_mu(games_pd[games_pd["season"] < prior])
        held = games_pd[games_pd["season"] == prior]
        dispersion = fit_dispersion(early.predict(held[FEATURES]), held["pts"].values)
        cal = predict_props(early, dispersion, props_pd[(props_pd["season"] == prior)])
        cal = cal[cal["result"] != "push"]
        blend = LogisticRegression(C=1e3).fit(blend_x(cal), (cal["result"] == "over").astype(int))

        # 2. Score season S with a model trained through S-1.
        model = fit_mu(games_pd[games_pd["season"] < test])
        mu_test = games_pd[games_pd["season"] == test]
        test_df = predict_props(model, dispersion, props_pd[props_pd["season"] == test])
        test_df["blend_p_over_cond"] = blend.predict_proba(blend_x(test_df))[:, 1]
        no_push = 1 - test_df["model_p_push"]
        test_df["blend_p_over"] = test_df["blend_p_over_cond"] * no_push
        test_df["blend_p_under"] = (1 - test_df["blend_p_over_cond"]) * no_push

        decided = test_df[test_df["result"] != "push"]
        y = (decided["result"] == "over").astype(int).values
        metrics = {
            "props": len(test_df),
            "mae_points_model": float(np.mean(np.abs(model.predict(mu_test[FEATURES]) - mu_test["pts"]))),
            "logloss_market": log_loss(decided["market_p_over"].values, y),
            "logloss_model": log_loss(decided["model_p_over_cond"].values, y),
            "logloss_blend": log_loss(decided["blend_p_over_cond"].values, y),
            "blend_weight_model": float(blend.coef_[0][0]),
            "blend_weight_market": float(blend.coef_[0][1]),
        }
        best = (decimal(test_df["best_over_price"].values), decimal(test_df["best_under_price"].values))
        median = (test_df["median_over_decimal"].values, test_df["median_under_decimal"].values)
        market = (test_df["market_p_over"].values * no_push, (1 - test_df["market_p_over"].values) * no_push)
        blended = (test_df["blend_p_over"].values, test_df["blend_p_under"].values)
        for pricing, probs, (po, pu) in [("best", blended, best), ("median", blended, median), ("marketonly_best", market, best)]:
            for t in THRESHOLDS:
                take, side, ev, profit = bets(test_df, *probs, po, pu, t)
                n = int(take.sum())
                tag = f"{pricing}_ev{int(t * 100)}"
                metrics[f"bets_{tag}"] = n
                metrics[f"roi_{tag}"] = float(profit[take].mean()) if n else 0.0
                metrics[f"hit_{tag}"] = float((test_df["result"].values[take] == side[take]).mean()) if n else 0.0
                if pricing == "best" and t == 0.02:
                    test_df["pick"] = np.where(take, side, None)
                    test_df["pick_ev"] = ev
                    test_df["pick_profit"] = np.where(take, profit, np.nan)
        mlflow.log_metrics(metrics)

        # Calibration: does a predicted 60% over actually go over 60% of the time?
        decided = decided.assign(bin=pd.cut(decided["blend_p_over_cond"], np.linspace(0, 1, 11)))
        calib = decided.groupby("bin", observed=True).agg(props=("result", "size"), predicted=("blend_p_over_cond", "mean"),
                                                          actual=("result", lambda r: (r == "over").mean())).reset_index()
        calib["bin"] = calib["bin"].astype(str)
        mlflow.log_table(calib, "calibration.json")
        summary.append({"season": test, **metrics})
        scored.append(test_df)

summary_pd = pd.DataFrame(summary)
display(summary_pd[["season", "props", "mae_points_model", "logloss_market", "logloss_model", "logloss_blend",
                    "blend_weight_model", "blend_weight_market"]])
roi_cols = [c for c in summary_pd.columns if c.startswith(("bets_", "roi_", "hit_"))]
display(summary_pd.melt(id_vars="season", value_vars=roi_cols).pivot(index="variable", columns="season", values="value").reset_index())

# COMMAND ----------

# MAGIC %md ## Save the backtest picks
# MAGIC Every test-season prop with the model's numbers, for the record page and later analysis.

# COMMAND ----------

keep = ["event_id", "game_id", "player_key", "player_name", "game_date", "season", "line", "market_p_over",
        "best_over_price", "best_under_price", "n_books", "mu", "model_p_over", "model_p_under", "model_p_push",
        "blend_p_over", "blend_p_under", "pick", "pick_ev", "pick_profit", "actual_pts", "result"]
backtest = pd.concat(scored)[keep]
(spark.createDataFrame(backtest.astype({"pick": "object"}).replace({np.nan: None}))
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.backtest_picks"))
display(spark.table(f"{G}.backtest_picks").where("pick IS NOT NULL").groupBy("season", "pick").agg(
    F.count("*").alias("bets"), F.round(F.avg("pick_profit"), 4).alias("roi"), F.round(F.sum("pick_profit"), 1).alias("units")))
