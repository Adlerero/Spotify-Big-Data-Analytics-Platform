"""
popularity_regression_audio_only.py
=====================================
Spotify Big Data Project — AI Component (Experiment)
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Same Random Forest Regressor but using ONLY audio features —
    excluding genre, year, artist_popularity, followers_log, album_popularity.
    This experiment isolates the predictive power of the sonic
    characteristics of a track.

Features used (audio only):
    danceability, energy, loudness, speechiness, acousticness,
    instrumentalness, liveness, valence, tempo, duration_ms,
    key, mode, time_signature

Usage:
    python popularity_regression_audio_only.py --csv master_ml.csv --trees 50
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def create_spark_session():
    return SparkSession.builder \
        .appName("SpotifyBigData_AudioOnly") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="master_ml.csv")
    parser.add_argument("--trees", type=int, default=50)
    args = parser.parse_args()

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("  Spotify Big Data — ML Regression (Audio Features Only)")
    print("  Excluded: genre, year, artist_popularity,")
    print("            followers_log, album_popularity")
    print("=" * 60)

    # ── Load ──
    print("\n[1/6] Loading dataset...")
    df = spark.read.option("header", "true").csv(args.csv)
    df = df.withColumn("popularity", col("popularity").cast(IntegerType()))
    df = df.dropna(subset=["popularity"])
    print(f"  Rows: {df.count():,}")

    # ── Audio features ONLY ──
    audio_features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms", "key", "mode", "time_signature",
    ]

    # Cast all to double
    for f in audio_features:
        df = df.withColumn(f, col(f).cast(DoubleType()))

    audio_features = [f for f in audio_features if f in df.columns]
    print(f"\n[2/6] Audio features only: {audio_features}")

    # ── Impute ──
    imputer = Imputer(
        inputCols=audio_features,
        outputCols=[f + "_imp" for f in audio_features],
        strategy="mean"
    )
    imputed = [f + "_imp" for f in audio_features]

    # ── Assemble ──
    assembler = VectorAssembler(
        inputCols=imputed,
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

    pipeline = Pipeline(stages=[imputer, assembler, rf])

    # ── Split ──
    print("\n[3/6] Splitting 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

    # ── Train ──
    print(f"\n[4/6] Training ({args.trees} trees, audio features only)...")
    print("  Please wait...")
    model = pipeline.fit(train_df)
    print("  Training complete!")

    # ── Evaluate ──
    print("\n[5/6] Evaluating...")
    predictions = model.transform(test_df)

    rmse = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="rmse").evaluate(predictions)
    mae  = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="mae").evaluate(predictions)
    r2   = RegressionEvaluator(labelCol="popularity",
        predictionCol="predicted_popularity", metricName="r2").evaluate(predictions)

    print("\n  ╔══════════════════════════════════════════════╗")
    print(  "  ║  AUDIO FEATURES ONLY — Results              ║")
    print(f"  ║  RMSE : {rmse:.4f}                             ║")
    print(f"  ║  MAE  : {mae:.4f}                             ║")
    print(f"  ║  R²   : {r2:.4f}                             ║")
    print(  "  ╚══════════════════════════════════════════════╝")

    # ── Feature importance ──
    rf_model    = model.stages[-1]
    importances = rf_model.featureImportances.toArray()
    importance_data = sorted(
        zip(imputed, importances),
        key=lambda x: x[1], reverse=True
    )

    print("\n  Audio feature importance ranking:")
    print(f"  {'Feature':<25} {'Importance':>10}  {'Bar'}")
    print("  " + "-" * 55)
    for fname, imp in importance_data:
        clean = fname.replace("_imp", "")
        bar   = "█" * int(imp * 200)
        print(f"  {clean:<25} {imp:>10.4f}  {bar}")

    # ── Compare with full model ──
    print("\n  ── Comparison with full model ──")
    print(f"  {'Model':<30} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 48)
    print(f"  {'Full model (all features)':<30} {'9.7424':>8} {'0.6240':>8}")
    print(f"  {'Audio features only':<30} {rmse:>8.4f} {r2:>8.4f}")
    delta_r2 = 0.6240 - r2
    print(f"\n  R² drop when removing context features: {delta_r2:.4f}")
    print(f"  This means genre/year/artist account for")
    print(f"  {delta_r2*100:.1f}% of the explainable variance.")

    # ── Save ──
    print("\n[6/6] Saving results...")
    import pandas as pd
    pd.DataFrame(
        [(n.replace("_imp",""), float(i), r+1)
         for r, (n, i) in enumerate(importance_data)],
        columns=["feature", "importance", "rank"]
    ).to_csv("ml_audio_feature_importance.csv", index=False)

    pd.DataFrame(
        [("RMSE", rmse), ("MAE", mae), ("R2", r2)],
        columns=["metric", "value"]
    ).to_csv("ml_audio_metrics.csv", index=False)

    print("  Saved: ml_audio_feature_importance.csv")
    print("  Saved: ml_audio_metrics.csv")

    print("\n" + "=" * 60)
    print("  Experiment complete!")
    print(f"  Full model R²        : 0.6240")
    print(f"  Audio-only model R²  : {r2:.4f}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
