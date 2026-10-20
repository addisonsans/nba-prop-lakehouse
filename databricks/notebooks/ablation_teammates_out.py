# Databricks notebook source
# MAGIC %md
# MAGIC # Ablation: how much do the teammates-out features help?
# MAGIC
# MAGIC Train candidate C twice, with and without `teammates_out_n / _usage / _pts`, everything else
# MAGIC identical, and score both on the 2024-25 props (2025-26 stays locked). Significance comes from a
# MAGIC paired bootstrap: resample props with replacement 2,000 times and recompute the log-loss gap.
# MAGIC
# MAGIC Caveat: for past games "out" means "not in the box score", which includes late scratches the
# MAGIC market may not have priced 1h before tip, so part of any gain may be leakage.

# COMMAND ----------

# MAGIC %run ./model_common

# COMMAND ----------

DEV_TEST, PRIOR = "2024-25", "2023-24"
OUT = ["teammates_out_n", "teammates_out_usage", "teammates_out_pts"]
ALL_FEATURES = list(FEATURES)

props_pd = prep(spark.table(f"{G}.props").where("did_play AND result IS NOT NULL").select(*PROP_COLS).toPandas())
test = props_pd[props_pd["season"] == DEV_TEST]
decided = (test["result"] != "push").values
y = (test["result"].values[decided] == "over").astype(int)
test_games = games_pd[games_pd["season"] == DEV_TEST]
held = games_pd[games_pd["season"] == PRIOR]


def per_prop_log_loss(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(features):
    global FEATURES
    FEATURES = features
    try:
        early = fit_mu(games_pd[games_pd["season"] < PRIOR])
        disp = fit_dispersion(early.predict(held[FEATURES]), held["pts"].values)
        cal = predict_props(early, disp, props_pd[props_pd["season"] == PRIOR])
        cal = cal[cal["result"] != "push"]
        blend = LogisticRegression(C=1e3).fit(blend_x(cal), (cal["result"] == "over").astype(int))
        model = fit_mu(games_pd[games_pd["season"] < DEV_TEST])
        scored = predict_props(model, disp, test)
        return {
            "blend": blend.predict_proba(blend_x(scored))[:, 1][decided],
            "model": scored["model_p_over_cond"].values[decided],
            "mae_all_games": float(np.mean(np.abs(model.predict(test_games[FEATURES]) - test_games["pts"]))),
            "mae_2plus_out": float(np.mean(np.abs(
                model.predict(test_games[test_games["teammates_out_n"] >= 2][FEATURES])
                - test_games[test_games["teammates_out_n"] >= 2]["pts"]))),
            "blend_weight_model": float(blend.coef_[0][0]),
        }
    finally:
        FEATURES = ALL_FEATURES


with_out = run(ALL_FEATURES)
without = run([f for f in ALL_FEATURES if f not in OUT])
market = test["market_p_over"].values[decided]
many_out = test["teammates_out_n"].values[decided] >= 2

# COMMAND ----------

rng = np.random.default_rng(0)
B = 2000


def paired(a, b, mask=None):
    """Mean log-loss of a minus b (negative = a better), 95% bootstrap CI, share of resamples where a is not better."""
    la, lb = per_prop_log_loss(a), per_prop_log_loss(b)
    d = (la - lb) if mask is None else (la - lb)[mask]
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {"n": int(len(d)), "diff_x1e4": d.mean() * 1e4, "ci_low_x1e4": np.percentile(boots, 2.5) * 1e4,
            "ci_high_x1e4": np.percentile(boots, 97.5) * 1e4, "p_not_better": float((boots >= 0).mean())}


rows = []
for subset, mask in [("all props", None), ("2+ teammates out", many_out)]:
    rows.append({"comparison": "model alone: with vs without", "subset": subset, **paired(with_out["model"], without["model"], mask)})
    rows.append({"comparison": "blend: with vs without", "subset": subset, **paired(with_out["blend"], without["blend"], mask)})
    rows.append({"comparison": "blend with vs market", "subset": subset, **paired(with_out["blend"], market, mask)})
    rows.append({"comparison": "blend without vs market", "subset": subset, **paired(without["blend"], market, mask)})
result = pd.DataFrame(rows)
summary = pd.DataFrame([
    {"variant": "with teammates out", **{k: with_out[k] for k in ("mae_all_games", "mae_2plus_out", "blend_weight_model")}},
    {"variant": "without", **{k: without[k] for k in ("mae_all_games", "mae_2plus_out", "blend_weight_model")}},
])
spark.createDataFrame(result).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.ablation_teammates_out")
spark.createDataFrame(summary).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{G}.ablation_teammates_out_summary")
with mlflow.start_run(run_name=f"ablation teammates_out {DEV_TEST}"):
    mlflow.log_table(result, "ablation.json")
    mlflow.log_table(summary, "summary.json")
display(summary)
display(result)
