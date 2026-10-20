# Databricks notebook source
# MAGIC %md
# MAGIC # Points model: shared pieces
# MAGIC Used by `model_backtest` and `picks` via `%run ./model_common`, so the backtest and the live picks
# MAGIC run exactly the same model. Loads the training frame (`games_pd`): every player-game since
# MAGIC {FIRST_TRAIN_SEASON} with the same pre-game features as `gold.props`.

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
MIN_CAREER_GAMES = 10  # never bet a player the model knows almost nothing about

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/nba-points-model")

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
PROP_COLS = ["event_id", "game_id", "player_key", "player_name", "game_date", "commence_time", "season", "season_type",
             "is_home", "line", "market_p_over", "best_over_price", "best_under_price", "median_over_decimal",
             "median_under_decimal", "n_books", "actual_pts", "result", *[c for c in FEATURES if not c.endswith("_i")]]

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
