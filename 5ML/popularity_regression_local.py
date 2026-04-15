"""
popularity_regression_local.py
===============================
Spotify Big Data Project — AI / Advanced Analytics Component
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Local version — reads master_ml.csv from local disk instead of OCI.

Usage:
    python popularity_regression_local.py --csv master_ml.csv
"""

import argparse
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, LongType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def create_spark_session():
    return SparkSession.builder \
        .appName("SpotifyBigData_MLRegression_Local") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()


def cast_numeric(df):
    return df \
        .withColumn("popularity",           col("popularity").cast(IntegerType())) \
        .withColumn("danceability",         col("danceability").cast(DoubleType())) \
        .withColumn("energy",               col("energy").cast(DoubleType())) \
        .withColumn("loudness",             col("loudness").cast(DoubleType())) \
        .withColumn("speechiness",          col("speechiness").cast(DoubleType())) \
        .withColumn("acousticness",         col("acousticness").cast(DoubleType())) \
        .withColumn("instrumentalness",     col("instrumentalness").cast(DoubleType())) \
        .withColumn("liveness",             col("liveness").cast(DoubleType())) \
        .withColumn("valence",              col("valence").cast(DoubleType())) \
        .withColumn("tempo",                col("tempo").cast(DoubleType())) \
        .withColumn("duration_ms",          col("duration_ms").cast(DoubleType())) \
        .withColumn("key",                  col("key").cast(DoubleType())) \
        .withColumn("mode",                 col("mode").cast(DoubleType())) \
        .withColumn("time_signature",       col("time_signature").cast(DoubleType())) \
        .withColumn("year",                 col("year").cast(DoubleType())) \
        .withColumn("release_month",        col("release_month").cast(DoubleType())) \
        .withColumn("followers_log",        col("followers_log").cast(DoubleType())) \
        .withColumn("artist_popularity",    col("artist_popularity").cast(DoubleType())) \
        .withColumn("album_popularity",     col("album_popularity").cast(DoubleType())) \
        .withColumn("total_tracks",         col("total_tracks").cast(DoubleType())) \
        .withColumn("is_chart_hit",         col("is_chart_hit").cast(DoubleType())) \
        .withColumn("in_spotify_playlists", col("in_spotify_playlists").cast(DoubleType())) \
        .withColumn("artist_count",         col("artist_count").cast(DoubleType()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="master_ml.csv",
                        help="Path to master_ml CSV file")
    parser.add_argument("--trees", type=int, default=50,
                        help="Number of trees (default 50 for speed)")
    args = parser.parse_args()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("  Spotify Big Data — ML Regression (Local)")
    print(f"  Input CSV : {args.csv}")
    print(f"  Trees     : {args.trees}")
    print("=" * 60)

    # ── Load ──
    print("\n[1/7] Loading dataset...")
    df = spark.read.option("header", "true").csv(args.csv)
    df = cast_numeric(df)
    df = df.dropna(subset=["popularity"])
    total = df.count()
    print(f"  Rows: {total:,}")

    # ── Features ──
    print("\n[2/7] Defining features...")
    numeric_features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms", "key", "mode", "time_signature",
        "year", "release_month", "followers_log", "artist_popularity",
        "album_popularity", "total_tracks",
        "is_chart_hit", "in_spotify_playlists", "artist_count",
    ]
    numeric_features = [f for f in numeric_features if f in df.columns]
    print(f"  Features: {len(numeric_features)}")

    # ── Encode genre ──
    print("\n[3/7] Encoding genre...")
    df = df.fillna({"genre": "unknown"})
    genre_indexer = StringIndexer(
        inputCol="genre", outputCol="genre_index", handleInvalid="keep"
    )

    # ── Impute ──
    imputer = Imputer(
        inputCols=numeric_features,
        outputCols=[f + "_imp" for f in numeric_features],
        strategy="mean"
    )
    imputed_features = [f + "_imp" for f in numeric_features] + ["genre_index"]

    # ── Assemble ──
    assembler = VectorAssembler(
        inputCols=imputed_features,
        outputCol="features",
        handleInvalid="skip"
    )

    # ── Model ──
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="popularity",
        numTrees=args.trees,
        maxDepth=8,
        maxBins=128,
        seed=42,
        predictionCol="predicted_popularity"
    )

    pipeline = Pipeline(stages=[genre_indexer, imputer, assembler, rf])

    # ── Split ──
    print("\n[4/7] Splitting 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

    # ── Train ──
    print(f"\n[5/7] Training Random Forest ({args.trees} trees)...")
    print("  Please wait...")
    model = pipeline.fit(train_df)
    print("  Training complete!")

    # ── Evaluate ──
    print("\n[6/7] Evaluating...")
    predictions = model.transform(test_df)

    rmse = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="rmse").evaluate(predictions)
    mae  = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="mae").evaluate(predictions)
    r2   = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="r2").evaluate(predictions)

    print("\n  ╔══════════════════════════════════╗")
    print(f"  ║  RMSE : {rmse:.4f}                   ║")
    print(f"  ║  MAE  : {mae:.4f}                   ║")
    print(f"  ║  R²   : {r2:.4f}                   ║")
    print("  ╚══════════════════════════════════╝")

    # ── Feature importance ──
    rf_model     = model.stages[-1]
    importances  = rf_model.featureImportances.toArray()
    importance_data = sorted(
        zip(imputed_features, importances),
        key=lambda x: x[1], reverse=True
    )

    print("\n  Top 10 most important features:")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print("  " + "-" * 42)
    for fname, imp in importance_data[:10]:
        print(f"  {fname.replace('_imp',''):<30} {imp:>10.4f}")

    # ── Save results locally ──
    print("\n[7/7] Saving results...")

    predictions.select(
        "track_name", "artist_name", "genre",
        "popularity", "predicted_popularity",
        "danceability", "energy", "valence", "followers_log"
    ).toPandas().to_csv("ml_predictions.csv", index=False)
    print("  Saved: ml_predictions.csv")

    import pandas as pd
    pd.DataFrame(
        [(n.replace("_imp",""), float(i), r+1)
         for r, (n, i) in enumerate(importance_data)],
        columns=["feature", "importance", "rank"]
    ).to_csv("ml_feature_importance.csv", index=False)
    print("  Saved: ml_feature_importance.csv")

    pd.DataFrame(
        [("RMSE", rmse), ("MAE", mae), ("R2", r2)],
        columns=["metric", "value"]
    ).to_csv("ml_metrics.csv", index=False)
    print("  Saved: ml_metrics.csv")

    print("\n" + "=" * 60)
    print("  ML Regression complete!")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("  Files: ml_predictions.csv, ml_feature_importance.csv,")
    print("         ml_metrics.csv")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
