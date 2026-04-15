"""
spark_cleaning.py
=================
Spotify Big Data Project — Batch Processing Layer (Step 1 of 3)
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Reads the 4 raw CSV datasets from OCI Object Storage (raw/ layer),
    performs data cleaning and standardization on each one,
    and writes the cleaned results to the processed/ layer.

Cleaning operations per dataset:
    1. spotify_1m_tracks   — remove nulls, remove duplicates, filter invalid rows,
                             cast types, normalize audio feature column names
    2. charts_2023         — remove nulls, normalize percentage columns to 0-1,
                             cast streams to long
    3. artist_data_2023    — remove nulls, cast followers/popularity to int
    4. albums_data_2023    — remove nulls, parse release_date, extract release_month,
                             cast explicit to boolean

Input  (OCI Object Storage):
    oci://BUCKET@NAMESPACE/raw/spotify_1m_tracks.csv
    oci://BUCKET@NAMESPACE/raw/charts_2023.csv
    oci://BUCKET@NAMESPACE/raw/artist_data_2023.csv
    oci://BUCKET@NAMESPACE/raw/albums_data_2023.csv

Output (OCI Object Storage):
    oci://BUCKET@NAMESPACE/processed/tracks/
    oci://BUCKET@NAMESPACE/processed/charts/
    oci://BUCKET@NAMESPACE/processed/artists/
    oci://BUCKET@NAMESPACE/processed/albums/

Usage (OCI Data Flow):
    Set BUCKET and NAMESPACE as arguments when creating the Data Flow application.
    Script location: oci://BUCKET@NAMESPACE/scripts/spark_cleaning.py
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, lower, when, lit, to_timestamp,
    month, year, regexp_replace
)
from pyspark.sql.types import (
    IntegerType, FloatType, LongType, BooleanType
)


def create_spark_session():
    return SparkSession.builder \
        .appName("SpotifyBigData_Cleaning") \
        .getOrCreate()


def build_path(bucket, namespace, prefix):
    return f"oci://{bucket}@{namespace}/{prefix}"


# ─────────────────────────────────────────────
# DATASET 1: spotify_1m_tracks
# ─────────────────────────────────────────────
def clean_tracks(spark, input_path, output_path):
    print("\n[1/4] Cleaning spotify_1m_tracks...")

    df = spark.read.option("header", "true").csv(input_path)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # Drop rows with nulls in critical columns
    critical_cols = ["artist_name", "track_name", "track_id",
                     "popularity", "year", "genre"]
    df = df.dropna(subset=critical_cols)

    # Drop duplicate track_ids — keep first occurrence
    df = df.dropDuplicates(["track_id"])

    # Cast types
    df = df \
        .withColumn("popularity",        col("popularity").cast(IntegerType())) \
        .withColumn("year",              col("year").cast(IntegerType())) \
        .withColumn("danceability",      col("danceability").cast(FloatType())) \
        .withColumn("energy",            col("energy").cast(FloatType())) \
        .withColumn("key",               col("key").cast(IntegerType())) \
        .withColumn("loudness",          col("loudness").cast(FloatType())) \
        .withColumn("mode",              col("mode").cast(IntegerType())) \
        .withColumn("speechiness",       col("speechiness").cast(FloatType())) \
        .withColumn("acousticness",      col("acousticness").cast(FloatType())) \
        .withColumn("instrumentalness",  col("instrumentalness").cast(FloatType())) \
        .withColumn("liveness",          col("liveness").cast(FloatType())) \
        .withColumn("valence",           col("valence").cast(FloatType())) \
        .withColumn("tempo",             col("tempo").cast(FloatType())) \
        .withColumn("duration_ms",       col("duration_ms").cast(LongType())) \
        .withColumn("time_signature",    col("time_signature").cast(IntegerType()))

    # Filter out invalid rows
    # popularity must be 0-100
    df = df.filter(col("popularity").between(0, 100))
    # duration at least 30 seconds
    df = df.filter(col("duration_ms") >= 30000)
    # year must be reasonable
    df = df.filter(col("year").between(1900, 2025))
    # audio features must be in valid range
    for feat in ["danceability", "energy", "speechiness",
                 "acousticness", "instrumentalness", "liveness", "valence"]:
        df = df.filter(col(feat).between(0.0, 1.0))

    # Trim whitespace on string columns
    df = df \
        .withColumn("artist_name", trim(col("artist_name"))) \
        .withColumn("track_name",  trim(col("track_name"))) \
        .withColumn("genre",       trim(lower(col("genre"))))

    # Drop the unnamed index column if present
    if "_c0" in df.columns:
        df = df.drop("_c0")

    clean_count = df.count()
    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed: {raw_count - clean_count:,} rows")

    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(output_path)
    print(f"  Written to: {output_path}")

    df.show(5, truncate=True)
    df.printSchema()
    return df


# ─────────────────────────────────────────────
# DATASET 2: charts_2023
# ─────────────────────────────────────────────
def clean_charts(spark, input_path, output_path):
    print("\n[2/4] Cleaning charts_2023...")

    df = spark.read.option("header", "true").csv(input_path)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # Drop nulls in critical columns
    df = df.dropna(subset=["track_name", "artist(s)_name", "streams"])

    # Rename columns for consistency
    df = df \
        .withColumnRenamed("artist(s)_name", "artist_name") \
        .withColumnRenamed("bpm",            "tempo")

    # Cast streams to long
    df = df.withColumn("streams", col("streams").cast(LongType()))

    # Cast chart/playlist counts to int
    for c in ["in_spotify_playlists", "in_spotify_charts",
              "in_apple_playlists", "in_apple_charts",
              "in_deezer_playlists", "in_deezer_charts",
              "in_shazam_charts", "artist_count",
              "released_year", "released_month", "released_day"]:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(IntegerType()))

    # Normalize percentage audio features to 0-1 (Charts 2023 stores them as 0-100)
    pct_cols = {
        "danceability_%": "danceability",
        "valence_%":      "valence",
        "energy_%":       "energy",
        "acousticness_%": "acousticness",
        "instrumentalness_%": "instrumentalness",
        "liveness_%":     "liveness",
        "speechiness_%":  "speechiness",
    }
    for old_name, new_name in pct_cols.items():
        if old_name in df.columns:
            df = df.withColumn(new_name,
                               (col(old_name).cast(FloatType()) / 100.0)) \
                   .drop(old_name)

    # Cast tempo
    df = df.withColumn("tempo", col("tempo").cast(FloatType()))

    # Trim strings
    df = df \
        .withColumn("track_name",   trim(col("track_name"))) \
        .withColumn("artist_name",  trim(col("artist_name")))

    # Add is_chart_hit flag — all rows in this dataset are chart hits
    df = df.withColumn("is_chart_hit", lit(1))

    clean_count = df.count()
    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed: {raw_count - clean_count:,} rows")

    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(output_path)
    print(f"  Written to: {output_path}")

    df.show(5, truncate=True)
    df.printSchema()
    return df


# ─────────────────────────────────────────────
# DATASET 3: artist_data_2023
# ─────────────────────────────────────────────
def clean_artists(spark, input_path, output_path):
    print("\n[3/4] Cleaning artist_data_2023...")

    df = spark.read.option("header", "true").csv(input_path)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # Rename columns for clarity
    df = df \
        .withColumnRenamed("id",               "artist_id") \
        .withColumnRenamed("name",             "artist_name") \
        .withColumnRenamed("artist_popularity","artist_popularity")

    # Drop nulls in critical columns
    df = df.dropna(subset=["artist_id", "artist_name"])

    # Drop duplicate artist_ids
    df = df.dropDuplicates(["artist_id"])

    # Cast numeric columns
    df = df \
        .withColumn("artist_popularity", col("artist_popularity").cast(IntegerType())) \
        .withColumn("followers",         col("followers").cast(LongType()))

    # Filter invalid popularity scores
    df = df.filter(
        col("artist_popularity").isNull() |
        col("artist_popularity").between(0, 100)
    )

    # Trim artist name
    df = df.withColumn("artist_name", trim(col("artist_name")))

    # Fill null followers with 0
    df = df.fillna({"followers": 0, "artist_popularity": 0})

    clean_count = df.count()
    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed: {raw_count - clean_count:,} rows")

    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(output_path)
    print(f"  Written to: {output_path}")

    df.show(5, truncate=True)
    df.printSchema()
    return df


# ─────────────────────────────────────────────
# DATASET 4: albums_data_2023
# ─────────────────────────────────────────────
def clean_albums(spark, input_path, output_path):
    print("\n[4/4] Cleaning albums_data_2023...")

    df = spark.read.option("header", "true").csv(input_path)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # Keep only the columns we need — drop redundant audio feature columns
    # (we have those in tracks already) and the many artist_N columns
    keep_cols = [
        "track_id", "track_name", "track_number",
        "duration_ms", "album_id", "album_name",
        "album_type", "album_popularity", "release_date",
        "label", "total_tracks",
        "artist_id", "artist_0"
    ]
    # Only keep columns that actually exist in the file
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(keep_cols)

    # Drop nulls in critical columns
    df = df.dropna(subset=["track_id", "track_name"])

    # Drop duplicate track_ids
    df = df.dropDuplicates(["track_id"])

    # Cast types
    df = df \
        .withColumn("album_popularity", col("album_popularity").cast(IntegerType())) \
        .withColumn("total_tracks",     col("total_tracks").cast(IntegerType())) \
        .withColumn("duration_ms",      col("duration_ms").cast(LongType()))

    # Parse release_date — format is '2015-06-09 00:00:00 UTC'
    # Strip the ' UTC' suffix first, then parse as timestamp
    df = df.withColumn("release_date_clean",
                       regexp_replace(col("release_date"), " UTC$", ""))
    df = df.withColumn("release_date",
                       to_timestamp(col("release_date_clean"), "yyyy-MM-dd HH:mm:ss")) \
           .drop("release_date_clean")
    df = df.withColumn("release_month", month(col("release_date")))
    df = df.withColumn("release_year",  year(col("release_date")))

    # Trim string columns
    df = df \
        .withColumn("track_name",  trim(col("track_name"))) \
        .withColumn("album_name",  trim(col("album_name"))) \
        .withColumn("label",       trim(col("label"))) \
        .withColumn("album_type",  trim(lower(col("album_type"))))

    # Fill nulls
    df = df.fillna({
        "label":           "Independent",
        "album_popularity": 0,
        "total_tracks":     1,
        "release_month":    0,
        "release_year":     0,
    })

    clean_count = df.count()
    print(f"  Cleaned rows: {clean_count:,}")
    print(f"  Removed: {raw_count - clean_count:,} rows")

    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(output_path)
    print(f"  Written to: {output_path}")

    df.show(5, truncate=True)
    df.printSchema()
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("Usage: spark_cleaning.py <BUCKET> <NAMESPACE>")
        print("Example: spark_cleaning.py bd-raw-spotify axz6vs6cibbb")
        sys.exit(1)

    bucket    = sys.argv[1]
    namespace = sys.argv[2]

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("  Spotify Big Data — Spark Cleaning Job")
    print(f"  Bucket   : {bucket}")
    print(f"  Namespace: {namespace}")
    print("=" * 60)

    raw  = lambda f: build_path(bucket, namespace, f"raw/{f}")
    proc = lambda f: build_path(bucket, namespace, f"processed/{f}")

    clean_tracks(spark,  raw("spotify_1m_tracks.csv"), proc("tracks"))
    clean_charts(spark,  raw("charts_2023.csv"),       proc("charts"))
    clean_artists(spark, raw("artist_data_2023.csv"),  proc("artists"))
    clean_albums(spark,  raw("albums_data_2023.csv"),  proc("albums"))

    print("\n" + "=" * 60)
    print("  Cleaning complete. All datasets written to processed/")
    print("  Next step: run spark_joins.py")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
