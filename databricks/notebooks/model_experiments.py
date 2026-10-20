# Databricks notebook source
# MAGIC %md
# MAGIC # Model comparison on the development season
# MAGIC
# MAGIC Every candidate is scored on the same 2024-25 props with the same metrics. 2025-26 stays locked
# MAGIC as the final test, so nothing here is chosen by looking at it.
# MAGIC
# MAGIC | Candidate | Trained on | Predicts |
# MAGIC |---|---|---|
# MAGIC | A. Constant | 2023-24 props | the historical over rate, for every prop |
# MAGIC | B. Market | nothing | the books' no-vig P(over) |
# MAGIC | C. Points regression + blend (current) | player-games ≤ 2023-24; blend on 2023-24 props | points → negative binomial → blended P(over) |
# MAGIC | D. Logistic regression classifier | 2023-24 props | over/under, from features + line + market |
# MAGIC | E. Gradient boosting classifier | 2023-24 props | over/under, from features + line + market |

# COMMAND ----------

# MAGIC %run ./model_common

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEV_TEST, PRIOR = "2024-25", "2023-24"
EV = 0.02

props_pd = prep(spark.table(f"{G}.props").where("did_play AND result IS NOT NULL").select(*PROP_COLS).toPandas())
props_pd["market_logit"] = logit(props_pd["market_p_over"])
train = props_pd[(props_pd["season"] < DEV_TEST) & (props_pd["result"] != "push")]
test = props_pd[props_pd["season"] == DEV_TEST].copy()
y_train = (train["result"] == "over").astype(int)
CLS_FEATURES = FEATURES + ["line", "market_logit"]
print(f"classifier training props: {len(train):,}; test props: {len(test):,}")

# Candidate C (current model), exactly as in the backtest; its push probability is shared by all
# candidates so they are bet and scored on the same terms.
early = fit_mu(games_pd[games_pd["season"] < PRIOR])
held = games_pd[games_pd["season"] == PRIOR]
dispersion = fit_dispersion(early.predict(held[FEATURES]), held["pts"].values)
cal = predict_props(early, dispersion, props_pd[props_pd["season"] == PRIOR])
cal = cal[cal["result"] != "push"]
blend = LogisticRegression(C=1e3).fit(blend_x(cal), (cal["result"] == "over").astype(int))
current = predict_props(fit_mu(games_pd[games_pd["season"] < DEV_TEST]), dispersion, test)
p_push = current["model_p_push"].values

logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
gbc = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=100,
                                     l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=30, random_state=0)
candidates = {
    "A. constant": np.full(len(test), y_train.mean()),
    "B. market": test["market_p_over"].values,
    "C. points regression + blend (current)": blend.predict_proba(blend_x(current))[:, 1],
    "D. logistic regression classifier": logreg.fit(train[CLS_FEATURES], y_train).predict_proba(test[CLS_FEATURES])[:, 1],
    "E. gradient boosting classifier": gbc.fit(train[CLS_FEATURES], y_train).predict_proba(test[CLS_FEATURES])[:, 1],
}

# COMMAND ----------

decided = (test["result"] != "push").values
y = (test["result"].values[decided] == "over").astype(int)
best = (decimal(test["best_over_price"].values), decimal(test["best_under_price"].values))
median = (test["median_over_decimal"].values, test["median_under_decimal"].values)
rows = []
for name, p_cond in candidates.items():
    p_over, p_under = p_cond * (1 - p_push), (1 - p_cond) * (1 - p_push)
    row = {"model": name, "log_loss": log_loss(p_cond[decided], y),
           "brier": float(np.mean((p_cond[decided] - y) ** 2)),
           "avg_p_over": float(p_cond.mean())}
    for pricing, (po, pu) in {"best": best, "median": median}.items():
        take, side, ev, profit = bets(test, p_over, p_under, po, pu, EV)
        row[f"bets_{pricing}"] = int(take.sum())
        row[f"roi_{pricing}"] = float(profit[take].mean()) if take.any() else 0.0
        row[f"units_{pricing}"] = float(profit[take].sum())
    rows.append(row)
    with mlflow.start_run(run_name=f"compare {DEV_TEST}: {name}"):
        mlflow.log_params({"test_season": DEV_TEST, "model": name, "ev_threshold": EV})
        mlflow.log_metrics({k: v for k, v in row.items() if k != "model"})

results = pd.DataFrame(rows)
results["over_rate_actual"] = y.mean()
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance (candidate C)
# MAGIC Shuffle one feature across the test props, re-score, and measure how much worse the blended log
# MAGIC loss and the points error get. Averaged over 5 shuffles. Bigger increase = the model relies on it more.

# COMMAND ----------

model_c = fit_mu(games_pd[games_pd["season"] < DEV_TEST])
test_games = games_pd[games_pd["season"] == DEV_TEST]


def blended_log_loss(df):
    scored = predict_props(model_c, dispersion, df)
    p = blend.predict_proba(blend_x(scored))[:, 1]
    return log_loss(p[decided], y)


base_ll = blended_log_loss(test)
base_mae = float(np.mean(np.abs(model_c.predict(test_games[FEATURES]) - test_games["pts"])))
rng = np.random.default_rng(0)
imp = []
for f in FEATURES:
    d_ll, d_mae = [], []
    for _ in range(5):
        shuffled, shuffled_games = test.copy(), test_games.copy()
        shuffled[f] = rng.permutation(shuffled[f].values)
        shuffled_games[f] = rng.permutation(shuffled_games[f].values)
        d_ll.append(blended_log_loss(shuffled) - base_ll)
        d_mae.append(float(np.mean(np.abs(model_c.predict(shuffled_games[FEATURES]) - shuffled_games["pts"]))) - base_mae)
    imp.append({"feature": f, "log_loss_increase": float(np.mean(d_ll)), "log_loss_sd": float(np.std(d_ll)),
                "mae_increase": float(np.mean(d_mae))})
importance = pd.DataFrame(imp).sort_values("log_loss_increase", ascending=False)
with mlflow.start_run(run_name=f"importance {DEV_TEST}"):
    mlflow.log_metrics({"base_log_loss": base_ll, "base_mae": base_mae})
    mlflow.log_table(importance, "importance.json")
spark.createDataFrame(importance).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.feature_importance_dev")
display(importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection: drop features whose shuffling didn't hurt
# MAGIC Retrain candidate C with only the features that raised log loss when shuffled, and re-score.
# MAGIC Caveat: correlated features (points over the last 5/10/20 games) share credit, so one can look
# MAGIC useless while its siblings carry the signal.

# COMMAND ----------

ALL_FEATURES = list(FEATURES)
kept = importance[importance["log_loss_increase"] > 0]["feature"].tolist()
dropped = [f for f in ALL_FEATURES if f not in kept]


def score_feature_set(features):
    global FEATURES
    FEATURES = features  # fit_mu / predict_props read the shared FEATURES list
    try:
        e = fit_mu(games_pd[games_pd["season"] < PRIOR])
        disp = fit_dispersion(e.predict(held[FEATURES]), held["pts"].values)
        c = predict_props(e, disp, props_pd[props_pd["season"] == PRIOR])
        c = c[c["result"] != "push"]
        b = LogisticRegression(C=1e3).fit(blend_x(c), (c["result"] == "over").astype(int))
        m = fit_mu(games_pd[games_pd["season"] < DEV_TEST])
        scored = predict_props(m, disp, test)
        p_cond = b.predict_proba(blend_x(scored))[:, 1]
        p_over, p_under = p_cond * (1 - scored["model_p_push"].values), (1 - p_cond) * (1 - scored["model_p_push"].values)
        out = {"n_features": len(features), "log_loss": log_loss(p_cond[decided], y),
               "mae": float(np.mean(np.abs(m.predict(test_games[FEATURES]) - test_games["pts"]))),
               "blend_weight_model": float(b.coef_[0][0])}
        for pricing, (po, pu) in {"best": best, "median": median}.items():
            take, side, ev, profit = bets(test, p_over, p_under, po, pu, EV)
            out[f"bets_{pricing}"], out[f"roi_{pricing}"] = int(take.sum()), float(profit[take].mean()) if take.any() else 0.0
        return out
    finally:
        FEATURES = ALL_FEATURES


selection = pd.DataFrame([{"features": "all", **score_feature_set(ALL_FEATURES)},
                          {"features": "kept (importance > 0)", **score_feature_set(kept)}])
with mlflow.start_run(run_name=f"selection {DEV_TEST}"):
    mlflow.log_params({"dropped": ", ".join(dropped)})
    for _, r in selection.iterrows():
        mlflow.log_metrics({f"{'all' if r['features'] == 'all' else 'kept'}_{k}": float(v) for k, v in r.items() if k != "features"})
print("dropped:", dropped)
spark.createDataFrame(selection).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.feature_selection_dev")
display(selection)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter tuning
# MAGIC Settings are chosen on points prediction for the season *before* the development season (train
# MAGIC ≤ 2022-23, validate on 2023-24 games, ~26k rows: much less noisy than prop outcomes). The winner is
# MAGIC then checked once on the 2024-25 props against the current settings.

# COMMAND ----------

from sklearn.metrics import mean_poisson_deviance

grid_rng = np.random.default_rng(1)
space = {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [50, 200, 800],
         "l2_regularization": [0.0, 1.0, 5.0]}
trials = [dict(GBM_PARAMS)]  # the current settings first
while len(trials) < 13:
    t = {**GBM_PARAMS, **{k: grid_rng.choice(v).item() for k, v in space.items()}}
    if t not in trials:
        trials.append(t)

tune_train, tune_valid = games_pd[games_pd["season"] < PRIOR], games_pd[games_pd["season"] == PRIOR]
tuning = []
for i, params in enumerate(trials):
    m = HistGradientBoostingRegressor(**params).fit(tune_train[FEATURES], tune_train["pts"])
    pred = np.clip(m.predict(tune_valid[FEATURES]), 1e-3, None)
    tuning.append({"trial": i, **{k: params[k] for k in space}, "trees_used": m.n_iter_,
                   "poisson_deviance": float(mean_poisson_deviance(tune_valid["pts"], pred)),
                   "mae": float(np.mean(np.abs(pred - tune_valid["pts"])))})
tuning = pd.DataFrame(tuning).sort_values("poisson_deviance")
spark.createDataFrame(tuning).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.tuning_dev")
display(tuning)

# COMMAND ----------

INT_PARAMS = {"max_leaf_nodes", "min_samples_leaf"}
best_params = {**GBM_PARAMS, **{k: int(tuning.iloc[0][k]) if k in INT_PARAMS else float(tuning.iloc[0][k]) for k in space}}
default_params = dict(GBM_PARAMS)
checks = []
for label, params in [("current settings", default_params), ("tuned settings", best_params)]:
    GBM_PARAMS = params  # fit_mu reads the shared GBM_PARAMS
    checks.append({"settings": label, **{k: params[k] for k in space}, **score_feature_set(ALL_FEATURES)})
GBM_PARAMS = default_params
checks = pd.DataFrame(checks)
with mlflow.start_run(run_name=f"tuning {DEV_TEST}"):
    mlflow.log_params({f"best_{k}": best_params[k] for k in space})
    mlflow.log_table(tuning, "tuning.json")
    mlflow.log_table(checks, "dev_check.json")
spark.createDataFrame(checks).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.tuning_check_dev")
display(checks)
