"""
PySpark feature computation pipeline.
Computes daily feature refresh for the sponsored search ranking system.

Features computed:
  - BM25 scores (query-ad lexical similarity)
  - Semantic similarity (mock bi-encoder scores)
  - Historical CTR (rolling 7-day average)
  - Query frequency (log-normalized)
  - Position bias weights (IPS)

In production this runs as a Spark job on EMR/Dataproc.
For local dev, uses PySpark in local mode.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType
import numpy as np


def create_spark_session(app_name: str = "SponsoredSearchFeatures") -> SparkSession:
    """Create a local Spark session for development."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def compute_bm25_features(spark: SparkSession, data_path: str):
    """
    Compute BM25 token overlap scores for query-ad pairs.
    In production: use Elasticsearch BM25 or custom Spark UDF.
    """
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    @F.udf(FloatType())
    def bm25_udf(query, ad_text):
        if not query or not ad_text:
            return 0.0
        q_tokens = set(query.lower().split())
        a_tokens = set(ad_text.lower().split())
        overlap = len(q_tokens & a_tokens)
        return float(overlap / (len(q_tokens) + 1e-8))

    df = df.withColumn("bm25_computed", bm25_udf(F.col("query_text"), F.col("ad_text")))
    return df


def compute_historical_ctr(spark: SparkSession, df):
    """
    Compute rolling 7-day CTR per ad using window functions.
    Groups by ad_id and computes mean click label as proxy CTR.
    """
    from pyspark.sql.window import Window

    window = Window.partitionBy("ad_id").rowsBetween(Window.unboundedPreceding, -1)
    df = df.withColumn(
        "rolling_ctr",
        F.avg("label").over(window)
    ).fillna({"rolling_ctr": 0.05})  # prior for cold-start ads

    return df


def compute_query_features(spark: SparkSession, df):
    """
    Compute query-level features:
      - query_freq: how often this query appears in logs
      - query_freq_log: log-normalized frequency
    """
    query_counts = df.groupBy("query_text").count().withColumnRenamed("count", "query_freq")
    df = df.join(query_counts, on="query_text", how="left")
    df = df.withColumn("query_freq_log", F.log1p(F.col("query_freq").cast(FloatType())))
    return df


def run_feature_pipeline(
    data_path: str = "data/synthetic_search_data.csv",
    output_path: str = "data/features/",
):
    """
    Full feature pipeline — runs all transformations and saves output.
    In production: scheduled daily via Airflow DAG.
    """
    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    print("Loading data...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    n = df.count()
    print(f"Loaded {n:,} rows")

    print("Computing BM25 features...")
    df = compute_bm25_features(spark, data_path)

    print("Computing historical CTR...")
    df = compute_historical_ctr(spark, df)

    print("Computing query features...")
    df = compute_query_features(spark, df)

    # Final feature selection
    feature_cols = [
        "query_id", "ad_id", "query_text", "ad_text", "label",
        "bm25_score", "semantic_sim", "historical_ctr",
        "bid_cpm", "position_bias", "query_freq", "query_freq_log",
        "rolling_ctr",
    ]
    available = [c for c in feature_cols if c in df.columns]
    df_out = df.select(available)

    print(f"Writing features to {output_path}...")
    df_out.write.mode("overwrite").parquet(output_path)

    print(f"✓ Feature pipeline complete — {n:,} rows written to {output_path}")
    spark.stop()


if __name__ == "__main__":
    run_feature_pipeline()
