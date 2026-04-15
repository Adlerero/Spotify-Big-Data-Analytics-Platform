"""
spark_joins.py
==============
Spotify Big Data Project — Batch Processing Layer (Step 2 of 3)
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Reads the 4 cleaned datasets from OCI Object Storage (processed/ layer),
    performs three LEFT JOINs to build a single enriched master table,
    and writes the result to processed/joined/.

Fixes applied:
    - Albums deduplicated before join (keep most popular album per track_id)
    - All numeric columns explicitly cast after join
    - Join 1 uses track_id instead of artist_name to avoid fan-out

Input  (OCI Object Storage):
    oci://BUCKET@NAMESPACE/processed/tracks/
    oci://BUCKET@NAMESPACE/processed/artists/
    oci://BUCKET@NAMESPACE/processed/albums/
    oci://BUCKET@NAMESPACE/processed/charts/

Output (OCI Object Storage):
    oci://BUCKET@NAMESPACE/processed/joined/

Usage (OCI Data Flow):
    Arguments: <BUCKET> <NAMESPACE>
    Example:   bd-raw-spotify axz6vs6cibbb
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import (
    col, trim, lower, lit, log1p, coalesce,
    row_number, rank
)
from pyspark.sql.types import (
    IntegerType, FloatType, LongType, DoubleType
)


def create_spark_session():
    return SparkSession.builder \
        .appName("SpotifyBigData_Joins") \
        .getOrCreate()


def build_path(bucket, namespace, prefix):
    return f"oci://{bucket}@{namespace}/{prefix}"


def read_processed(spark, path):
    return spark.read.option("header", "true").csv(path)


def normalize_key(df, col_name, new_col_name):
    return df.withColumn(new_col_name, trim(lower(col(col_name))))


def cast_tracks(df):
    """Cast all numeric columns in tracks that come as strings from CSV."""
    return df \
        .withColumn("popularity",       col("popularity").cast(IntegerType())) \
        .withColumn("year",             col("year").cast(IntegerType())) \
        .withColumn("danceability",     col("danceability").cast(FloatType())) \
        .withColumn("energy",           col("energy").cast(FloatType())) \
        .withColumn("key",              col("key").cast(IntegerType())) \
        .withColumn("loudness",         col("loudness").cast(FloatType())) \
        .withColumn("mode",             col("mode").cast(IntegerType())) \
        .withColumn("speechiness",      col("speechiness").cast(FloatType())) \
        .withColumn("acousticness",     col("acousticness").cast(FloatType())) \
        .withColumn("instrumentalness", col("instrumentalness").cast(FloatType())) \
        .withColumn("liveness",         col("liveness").cast(FloatType())) \
        .withColumn("valence",          col("valence").cast(FloatType())) \
        .withColumn("tempo",            col("tempo").cast(FloatType())) \
        .withColumn("duration_ms",      col("duration_ms").cast(LongType())) \
        .withColumn("time_signature",   col("time_signature").cast(IntegerType()))


def main():
    if len(sys.argv) != 3:
        print("Usage: spark_joins.py <BUCKET> <NAMESPACE>")
        sys.exit(1)

    bucket    = sys.argv[1]
    namespace = sys.argv[2]

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("  Spotify Big Data — Spark Joins Job (v2)")
    print(f"  Bucket   : {bucket}")
    print(f"  Namespace: {namespace}")
    print("=" * 60)

    proc = lambda f: build_path(bucket, namespace, f"processed/{f}")

    # ─────────────────────────────────────────────
    # LOAD all 4 processed datasets
    # ─────────────────────────────────────────────
    print("\n[1/6] Loading processed datasets...")

    tracks  = read_processed(spark, proc("tracks"))
    artists = read_processed(spark, proc("artists"))
    albums  = read_processed(spark, proc("albums"))
    charts  = read_processed(spark, proc("charts"))

    # Cast numeric types on tracks immediately after loading
    tracks = cast_tracks(tracks)

    print(f"  Tracks  : {tracks.count():,} rows")
    print(f"  Artists : {artists.count():,} rows")
    print(f"  Albums  : {albums.count():,} rows")
    print(f"  Charts  : {charts.count():,} rows")

    # ─────────────────────────────────────────────
    # DEDUPLICATE ALBUMS
    # Multiple albums can share the same track_name (covers, compilations).
    # Keep only the most popular album per track_id to avoid row explosion.
    # ─────────────────────────────────────────────
    print("\n[2/6] Deduplicating albums (keep most popular per track_id)...")

    albums_slim = albums.select(
        col("track_id"),
        col("album_id"),
        col("album_name"),
        col("album_type"),
        col("album_popularity").cast(IntegerType()),
        col("label"),
        col("total_tracks").cast(IntegerType()),
        col("release_month").cast(IntegerType()),
        col("release_year").cast(IntegerType()),
    ).dropna(subset=["track_id"])

    # Use window function to keep only the top album per track_id
    window_spec = Window.partitionBy("track_id") \
                        .orderBy(col("album_popularity").desc())
    albums_dedup = albums_slim \
        .withColumn("rn", row_number().over(window_spec)) \
        .filter(col("rn") == 1) \
        .drop("rn")

    print(f"  Albums before dedup : {albums_slim.count():,}")
    print(f"  Albums after dedup  : {albums_dedup.count():,}")

    # ─────────────────────────────────────────────
    # DEDUPLICATE ARTISTS
    # Keep one row per artist_name (most popular)
    # ─────────────────────────────────────────────
    print("\n[3/6] Deduplicating artists (keep most popular per artist_name)...")

    artists_slim = artists.select(
        col("artist_name"),
        col("artist_id"),
        col("artist_popularity").cast(IntegerType()),
        col("followers").cast(LongType()),
        col("genre_0"),
        col("genre_1"),
        col("genre_2"),
    ).dropna(subset=["artist_name"])

    window_art = Window.partitionBy("artist_name") \
                       .orderBy(col("artist_popularity").desc())
    artists_dedup = artists_slim \
        .withColumn("rn", row_number().over(window_art)) \
        .filter(col("rn") == 1) \
        .drop("rn")

    print(f"  Artists after dedup : {artists_dedup.count():,}")

    # ─────────────────────────────────────────────
    # JOIN 1: Tracks + Artists on artist_name
    # ─────────────────────────────────────────────
    print("\n[4/6] JOIN 1 — Tracks + Artists (on artist_name)...")

    tracks  = normalize_key(tracks,        "artist_name", "join_artist")
    artists_dedup = normalize_key(artists_dedup, "artist_name", "join_artist") \
                    .drop("artist_name")

    joined = tracks.join(artists_dedup, on="join_artist", how="left")

    matched_artists = joined.filter(col("artist_id").isNotNull()).count()
    total_after_j1  = joined.count()
    print(f"  Tracks matched to artist : {matched_artists:,}")
    print(f"  Total rows after JOIN 1  : {total_after_j1:,}")

    # ─────────────────────────────────────────────
    # JOIN 2: + Albums on track_id (exact key — no fan-out)
    # ─────────────────────────────────────────────
    print("\n[5/6] JOIN 2 — + Albums (on track_id)...")

    joined = joined.join(albums_dedup, on="track_id", how="left")

    total_after_j2 = joined.count()
    print(f"  Total rows after JOIN 2  : {total_after_j2:,}")

    # ─────────────────────────────────────────────
    # JOIN 3: + Charts on track_name + artist_name
    # ─────────────────────────────────────────────
    print("\n[6/6] JOIN 3 — + Charts 2023 (on track_name + artist_name)...")

    charts_slim = charts.select(
        col("track_name"),
        col("artist_name"),
        col("streams").cast(LongType()),
        col("in_spotify_playlists").cast(IntegerType()),
        col("in_spotify_charts").cast(IntegerType()),
        col("artist_count").cast(IntegerType()),
        col("is_chart_hit").cast(IntegerType()),
    ).dropna(subset=["track_name", "artist_name"])

    charts_slim = normalize_key(charts_slim, "track_name",  "join_track")
    charts_slim = normalize_key(charts_slim, "artist_name", "join_artist_c") \
                  .drop("track_name", "artist_name")

    joined = normalize_key(joined, "track_name", "join_track")
    joined = joined.withColumn("join_artist_c", col("join_artist"))

    joined = joined.join(
        charts_slim,
        on=["join_track", "join_artist_c"],
        how="left"
    )

    chart_hits    = joined.filter(col("is_chart_hit") == 1).count()
    total_after_j3 = joined.count()
    print(f"  Chart hits matched       : {chart_hits:,}")
    print(f"  Total rows after JOIN 3  : {total_after_j3:,}")

    # ─────────────────────────────────────────────
    # POST-JOIN CLEANUP
    # ─────────────────────────────────────────────
    print("\nPost-join cleanup...")

    # Fill nulls from unmatched joins
    joined = joined.fillna({
        "artist_popularity":    0,
        "album_popularity":     0,
        "total_tracks":         1,
        "release_month":        0,
        "release_year":         0,
        "in_spotify_playlists": 0,
        "in_spotify_charts":    0,
        "artist_count":         1,
        "is_chart_hit":         0,
        "label":                "Independent",
        "album_type":           "unknown",
        "genre_0":              "unknown",
    })

    # Fill null followers and streams with 0
    joined = joined \
        .withColumn("followers",
                    coalesce(col("followers").cast(LongType()), lit(0))) \
        .withColumn("streams",
                    coalesce(col("streams").cast(LongType()), lit(0)))

    # Add log-scaled followers for ML model
    joined = joined.withColumn("followers_log",
                               log1p(col("followers").cast(DoubleType())))

    # Drop temporary join key columns
    joined = joined.drop("join_artist", "join_track", "join_artist_c")

    # ─────────────────────────────────────────────
    # FINAL VALIDATION
    # ─────────────────────────────────────────────
    final_count = joined.count()
    print(f"\nFinal master table rows    : {final_count:,}")
    print(f"Final master table columns : {len(joined.columns)}")

    # Sanity check — should be close to original 1.15M
    if final_count > tracks.count() * 1.05:
        print("  WARNING: Row count is more than 5% above base tracks.")
        print("  Check for remaining join duplicates.")
    else:
        print("  Row count looks correct.")

    # ─────────────────────────────────────────────
    # WRITE OUTPUT
    # ─────────────────────────────────────────────
    output_path = proc("joined")
    print(f"\nWriting master table to: {output_path}")

    joined.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(output_path)

    print("\nSample of joined master table:")
    joined.select(
        "track_name", "artist_name", "genre", "popularity",
        "danceability", "energy", "valence",
        "followers", "is_chart_hit", "streams"
    ).show(10, truncate=True)

    joined.printSchema()

    print("\n" + "=" * 60)
    print("  Joins complete. Master written to processed/joined/")
    print("  Next step: run spark_aggregations.py")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
