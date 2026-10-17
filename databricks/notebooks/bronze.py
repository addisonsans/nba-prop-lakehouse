# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: landing files -> Delta tables
# MAGIC
# MAGIC The S3 lake is mirrored into `/Volumes/nba/raw/landing` (see `src/ingest/databricks_sync.py`).
# MAGIC This notebook loads it into `nba.bronze` without changing any content:
# MAGIC
# MAGIC | Table | Source | One row per |
# MAGIC |---|---|---|
# MAGIC | `odds` | `raw/odds/` (live + historical backfill) | file: events list, one game's props, or a day's moneylines |
# MAGIC | `espn_scoreboard` | `raw/espn/scoreboard/` | game date |
# MAGIC | `espn_summary` | `raw/espn/summary/` | game (box score, play-by-play, injuries) |
# MAGIC | `espn_injuries` | `raw/espn/injuries/` | injury-report snapshot |
# MAGIC | `nba_stats_*` | `historical/nba_stats/` | rows of the 2007-2026 parquet tables, as uploaded |
# MAGIC
# MAGIC JSON stays a string (`raw_json`); parsing happens in silver, so a parsing mistake never loses data.
# MAGIC Auto Loader tracks which files it has loaded, so each run only reads new ones.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "nba"
LANDING = f"/Volumes/{CATALOG}/raw/landing"
CHECKPOINTS = f"/Volumes/{CATALOG}/bronze/checkpoints"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze COMMENT 'Raw lake files as Delta tables, content unchanged'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.bronze.checkpoints COMMENT 'Auto Loader progress for bronze tables'")

# COMMAND ----------


def path_value(path, name):
    """Value of a `name=value` folder or file name in a landing path."""
    return F.regexp_extract(path, rf"{name}=([^/]+?)(?:\.json\.gz)?(?:/|$)", 1)


def ingest_json_files(source: str, table: str, path_fields: list[str], extra=None) -> None:
    """Incrementally load every new *.json.gz under LANDING/source into nba.bronze.<table>."""
    files = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("wholetext", "true")  # one row per file, not per line
        .option("pathGlobFilter", "*.json.gz")
        .load(f"{LANDING}/{source}")
    )
    path = F.col("_metadata.file_path")
    columns = [path_value(path, name).alias(name) for name in path_fields]
    columns += [c.alias(n) for n, c in (extra or {}).items()]
    columns += [
        F.col("value").alias("raw_json"),
        path.alias("source_file"),
        F.col("_metadata.file_modification_time").alias("file_modified_at"),
        F.current_timestamp().alias("ingested_at"),
    ]
    (
        files.select(*columns)
        .writeStream.option("checkpointLocation", f"{CHECKPOINTS}/{table}")
        .trigger(availableNow=True)
        .toTable(f"{CATALOG}.bronze.{table}")
        .awaitTermination()
    )


path = F.col("_metadata.file_path")
ingest_json_files(
    "raw/odds/basketball_nba", "odds", ["snapshot_date", "snapshot_ts", "event"],
    # events = game list, event = one game's player props, h2h = moneylines for the slate
    extra={"file_kind": F.regexp_extract(path, r"/(events|event|h2h)(?:=[^/]*)?\.json\.gz$", 1)},
)
ingest_json_files("raw/espn/scoreboard", "espn_scoreboard", ["game_date"])
ingest_json_files("raw/espn/summary", "espn_summary", ["game_date", "event"])
ingest_json_files("raw/espn/injuries", "espn_injuries", ["snapshot_date", "snapshot_ts"])

# COMMAND ----------

# The historical parquet tables are small and static: reload them whole each run.
for name in ["games", "team_games", "player_games", "player_seasons", "schedule_2026_27", "teams"]:
    (
        spark.read.parquet(f"{LANDING}/historical/nba_stats/{name}/")
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.bronze.nba_stats_{name}")
    )

# COMMAND ----------

tables = [r.tableName for r in spark.sql(f"SHOW TABLES IN {CATALOG}.bronze").collect()]
display(spark.createDataFrame(
    [(t, spark.table(f"{CATALOG}.bronze.{t}").count()) for t in sorted(tables)], "table string, rows long"
))
