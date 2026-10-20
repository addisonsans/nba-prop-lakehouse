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

# MAGIC %run ./model_common

# COMMAND ----------

TEST_SEASONS = ["2024-25", "2025-26"]  # 2023-24 has too few prior-season props to fit the blend
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
props_pd = prep(spark.table(f"{G}.props").where("did_play AND result IS NOT NULL").select(*PROP_COLS).toPandas())
print(f"{len(games_pd):,} player-games for training, {len(props_pd):,} graded props")

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
