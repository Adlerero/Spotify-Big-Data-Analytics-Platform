"""
spark_aggregations.py
=====================
Spotify Big Data Project — Batch Processing Layer (Step 3 of 3)
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Reads the joined master table from processed/joined/,
    performs multiple aggregations to generate analytical datasets,
    and writes results to the curated/ layer ready for
    Power BI dashboards and PostgreSQL loading.

Aggregations produced:
    1. agg_genre        — avg audio features + popularity by genre
    2. agg_year         — avg popularity + audio features by year (trend)
    3. agg_label        — top labels by avg popularity and track count
    4. agg_hit_vs_nohit — avg audio features: chart hits vs non-hits
    5. agg_artist       — top artists by avg popularity and followers
    6. master_ml        — full master table with all features for ML model

Input  (OCI Object Storage):
    oci://BUCKET@NAMESPACE/processed/joined/

Output (OCI Object Storage):
    oci://BUCKET@NAMESPACE/curated/agg_genre/
    oci://BUCKET@NAMESPACE/curated/agg_year/
    oci://BUCKET@NAMESPACE/curated/agg_label/
    oci://BUCKET@NAMESPACE/curated/agg_hit_vs_nohit/
    oci://BUCKET@NAMESPACE/curated/agg_artist/
    oci://BUCKET@NAMESPACE/curated/master_ml/

Usage (OCI Data Flow):
    Arguments: <BUCKET> <NAMESPACE>
    Example:   bd-raw-spotify axz6vs6cibbb
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, round as spark_round,
    desc, sum as spark_sum, max as spark_max,
    when, lit
)
from pyspark.sql.types import IntegerType, FloatType, LongType, DoubleType


def create_spark_session():
    return SparkSession.builder \
        .appName("SpotifyBigData_Aggregations") \
        .getOrCreate()


def build_path(bucket, namespace, prefix):
    return f"oci://{bucket}@{namespace}/{prefix}"


def write_curated(df, path, label):
    """Write a curated aggregation to OCI as a single CSV file."""
    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(path)
    print(f"  Written {label} to: {path}")
    df.show(10, truncate=False)


def cast_numeric(df):
    """Cast all audio feature and metric columns to proper numeric types."""
    return df \
        .withColumn("popularity",       col("popularity").cast(IntegerType())) \
        .withColumn("danceability",     col("danceability").cast(FloatType())) \
        .withColumn("energy",           col("energy").cast(FloatType())) \
        .withColumn("loudness",         col("loudness").cast(FloatType())) \
        .withColumn("speechiness",      col("speechiness").cast(FloatType())) \
        .withColumn("acousticness",     col("acousticness").cast(FloatType())) \
        .withColumn("instrumentalness", col("instrumentalness").cast(FloatType())) \
        .withColumn("liveness",         col("liveness").cast(FloatType())) \
        .withColumn("valence",          col("valence").cast(FloatType())) \
        .withColumn("tempo",            col("tempo").cast(FloatType())) \
        .withColumn("duration_ms",      col("duration_ms").cast(LongType())) \
        .withColumn("year",             col("year").cast(IntegerType())) \
        .withColumn("followers",        col("followers").cast(LongType())) \
        .withColumn("followers_log",    col("followers_log").cast(DoubleType())) \
        .withColumn("streams",          col("streams").cast(LongType())) \
        .withColumn("is_chart_hit",     col("is_chart_hit").cast(IntegerType())) \
        .withColumn("artist_popularity",col("artist_popularity").cast(IntegerType())) \
        .withColumn("album_popularity", col("album_popularity").cast(IntegerType())) \
        .withColumn("release_month",    col("release_month").cast(IntegerType())) \
        .withColumn("release_year",     col("release_year").cast(IntegerType()))


def main():
    if len(sys.argv) != 3:
        print("Usage: spark_aggregations.py <BUCKET> <NAMESPACE>")
        sys.exit(1)

    bucket    = sys.argv[1]
    namespace = sys.argv[2]

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("  Spotify Big Data — Spark Aggregations Job")
    print(f"  Bucket   : {bucket}")
    print(f"  Namespace: {namespace}")
    print("=" * 60)

    cur  = lambda f: build_path(bucket, namespace, f"curated/{f}")
    proc = lambda f: build_path(bucket, namespace, f"processed/{f}")

    # ─────────────────────────────────────────────
    # LOAD joined master table
    # ─────────────────────────────────────────────
    print("\n[0/6] Loading joined master table...")

    df = spark.read.option("header", "true").csv(proc("joined"))
    df = cast_numeric(df)

    total = df.count()
    print(f"  Master table rows : {total:,}")
    print(f"  Columns           : {len(df.columns)}")

    # ─────────────────────────────────────────────
    # AGG 1: By genre
    # ─────────────────────────────────────────────
    print("\n[1/6] Aggregation by genre...")

    agg_genre = df.groupBy("genre").agg(
        count("*").alias("track_count"),
        spark_round(avg("popularity"),       2).alias("avg_popularity"),
        spark_round(avg("danceability"),     3).alias("avg_danceability"),
        spark_round(avg("energy"),           3).alias("avg_energy"),
        spark_round(avg("valence"),          3).alias("avg_valence"),
        spark_round(avg("acousticness"),     3).alias("avg_acousticness"),
        spark_round(avg("speechiness"),      3).alias("avg_speechiness"),
        spark_round(avg("tempo"),            1).alias("avg_tempo"),
        spark_round(avg("loudness"),         2).alias("avg_loudness"),
        spark_sum("is_chart_hit").alias("chart_hits"),
    ).orderBy(desc("track_count"))

    print(f"  Genres found: {agg_genre.count()}")
    write_curated(agg_genre, cur("agg_genre"), "agg_genre")

    # ─────────────────────────────────────────────
    # AGG 2: By year (trend over time)
    # ─────────────────────────────────────────────
    print("\n[2/6] Aggregation by year (trend)...")

    agg_year = df.filter(
        col("year").between(2009, 2024)
    ).groupBy("year").agg(
        count("*").alias("track_count"),
        spark_round(avg("popularity"),   2).alias("avg_popularity"),
        spark_round(avg("danceability"), 3).alias("avg_danceability"),
        spark_round(avg("energy"),       3).alias("avg_energy"),
        spark_round(avg("valence"),      3).alias("avg_valence"),
        spark_round(avg("acousticness"), 3).alias("avg_acousticness"),
        spark_round(avg("tempo"),        1).alias("avg_tempo"),
        spark_sum("is_chart_hit").alias("chart_hits"),
    ).orderBy("year")

    write_curated(agg_year, cur("agg_year"), "agg_year")

    # ─────────────────────────────────────────────
    # AGG 3: By label (top 30)
    # ─────────────────────────────────────────────
    print("\n[3/6] Aggregation by label (top 30)...")

    agg_label = df.filter(
        col("label") != "Independent"
    ).groupBy("label").agg(
        count("*").alias("track_count"),
        spark_round(avg("popularity"),      2).alias("avg_popularity"),
        spark_round(avg("album_popularity"),2).alias("avg_album_popularity"),
        spark_sum("is_chart_hit").alias("chart_hits"),
        spark_sum("streams").alias("total_streams"),
    ).orderBy(desc("avg_popularity")) \
     .limit(30)

    write_curated(agg_label, cur("agg_label"), "agg_label")

    # ─────────────────────────────────────────────
    # AGG 4: Hit vs Non-hit comparison
    # ─────────────────────────────────────────────
    print("\n[4/6] Aggregation hit vs non-hit...")

    agg_hit = df.groupBy("is_chart_hit").agg(
        count("*").alias("track_count"),
        spark_round(avg("popularity"),       2).alias("avg_popularity"),
        spark_round(avg("danceability"),     3).alias("avg_danceability"),
        spark_round(avg("energy"),           3).alias("avg_energy"),
        spark_round(avg("valence"),          3).alias("avg_valence"),
        spark_round(avg("acousticness"),     3).alias("avg_acousticness"),
        spark_round(avg("speechiness"),      3).alias("avg_speechiness"),
        spark_round(avg("instrumentalness"), 3).alias("avg_instrumentalness"),
        spark_round(avg("liveness"),         3).alias("avg_liveness"),
        spark_round(avg("tempo"),            1).alias("avg_tempo"),
        spark_round(avg("loudness"),         2).alias("avg_loudness"),
        spark_round(avg("duration_ms"),      0).alias("avg_duration_ms"),
        spark_round(avg("followers_log"),    3).alias("avg_followers_log"),
        spark_round(avg("artist_popularity"),2).alias("avg_artist_popularity"),
    ).orderBy("is_chart_hit")

    write_curated(agg_hit, cur("agg_hit_vs_nohit"), "agg_hit_vs_nohit")

    # ─────────────────────────────────────────────
    # AGG 5: Top artists
    # ─────────────────────────────────────────────
    print("\n[5/6] Aggregation top artists (top 50)...")

    agg_artist = df.groupBy("artist_name").agg(
        count("*").alias("track_count"),
        spark_round(avg("popularity"),      2).alias("avg_popularity"),
        spark_max("followers").alias("followers"),
        spark_round(avg("artist_popularity"),1).alias("artist_popularity"),
        spark_sum("is_chart_hit").alias("chart_hits"),
        spark_sum("streams").alias("total_streams"),
        spark_round(avg("danceability"),    3).alias("avg_danceability"),
        spark_round(avg("energy"),          3).alias("avg_energy"),
        spark_round(avg("valence"),         3).alias("avg_valence"),
    ).orderBy(desc("avg_popularity")) \
     .limit(50)

    write_curated(agg_artist, cur("agg_artist"), "agg_artist")

    # ─────────────────────────────────────────────
    # AGG 6: Master ML dataset
    # Select only the columns needed for the regression model
    # ─────────────────────────────────────────────
    print("\n[6/6] Writing ML-ready master dataset...")

    ml_cols = [
        "track_id", "track_name", "artist_name", "genre",
        "popularity",
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms", "key", "mode", "time_signature",
        "year", "release_month", "release_year",
        "followers", "followers_log", "artist_popularity",
        "album_popularity", "album_type", "total_tracks",
        "label", "genre_0",
        "is_chart_hit", "streams",
        "in_spotify_playlists", "in_spotify_charts",
        "artist_count",
    ]

    # Only keep columns that exist in the dataframe
    ml_cols_existing = [c for c in ml_cols if c in df.columns]
    master_ml = df.select(ml_cols_existing)

    # Drop rows missing the target variable (popularity)
    master_ml = master_ml.dropna(subset=["popularity"])

    ml_count = master_ml.count()
    print(f"  ML dataset rows    : {ml_count:,}")
    print(f"  ML dataset columns : {len(master_ml.columns)}")

    write_curated(master_ml, cur("master_ml"), "master_ml")

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Aggregations complete. Curated layer ready.")
    print("  Files written:")
    print("    curated/agg_genre/")
    print("    curated/agg_year/")
    print("    curated/agg_label/")
    print("    curated/agg_hit_vs_nohit/")
    print("    curated/agg_artist/")
    print("    curated/master_ml/")
    print("  Next step: run popularity_regression.py")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
