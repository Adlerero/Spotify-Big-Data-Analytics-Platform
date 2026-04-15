"""
load_to_db.py
=============
Spotify Big Data Project — Analytical Data Model
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Creates a SQLite analytical database (spotify_analytics.db) and loads
    all curated datasets produced by the Spark pipeline.

Tables created:
    dim_tracks        — core track dimension (from master_ml)
    dim_artists       — artist profiles (from agg_artist)
    fact_popularity   — central fact table with all features
    agg_genre         — aggregations by genre
    agg_year          — trend over time
    agg_label         — top labels
    agg_hit_vs_nohit  — chart hits vs non-hits comparison
    ml_metrics        — model evaluation results
    ml_feature_importance — feature importance ranking

Prerequisites:
    pip install pandas

Usage:
    python load_to_db.py

    Make sure these CSV files are in the same folder:
        master_ml.csv
        agg_genre.csv
        agg_year.csv
        agg_label.csv
        agg_hit_vs_nohit.csv
        agg_artist.csv
        ml_metrics.csv
        ml_feature_importance.csv
        ml_audio_metrics.csv
        ml_audio_feature_importance.csv
"""

import sqlite3
import pandas as pd
import os


DB_NAME = "spotify_analytics.db"


def connect():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def load_csv(filename, nrows=None):
    """Load a CSV file into a DataFrame."""
    if not os.path.exists(filename):
        print(f"  WARNING: {filename} not found — skipping")
        return None
    df = pd.read_csv(filename, nrows=nrows, on_bad_lines="skip", engine="python")
    print(f"  Loaded {filename}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def create_tables(conn):
    """Create all analytical tables."""
    cursor = conn.cursor()

    cursor.executescript("""
        -- Core track dimension
        CREATE TABLE IF NOT EXISTS dim_tracks (
            track_id        TEXT PRIMARY KEY,
            track_name      TEXT,
            artist_name     TEXT,
            genre           TEXT,
            year            INTEGER,
            duration_ms     INTEGER,
            is_chart_hit    INTEGER DEFAULT 0,
            streams         INTEGER DEFAULT 0
        );

        -- Artist dimension
        CREATE TABLE IF NOT EXISTS dim_artists (
            artist_name         TEXT PRIMARY KEY,
            track_count         INTEGER,
            avg_popularity      REAL,
            followers           INTEGER,
            artist_popularity   REAL,
            chart_hits          INTEGER,
            total_streams       INTEGER,
            avg_danceability    REAL,
            avg_energy          REAL,
            avg_valence         REAL
        );

        -- Central fact table
        CREATE TABLE IF NOT EXISTS fact_popularity (
            track_id            TEXT PRIMARY KEY,
            track_name          TEXT,
            artist_name         TEXT,
            genre               TEXT,
            popularity          INTEGER,
            danceability        REAL,
            energy              REAL,
            loudness            REAL,
            speechiness         REAL,
            acousticness        REAL,
            instrumentalness    REAL,
            liveness            REAL,
            valence             REAL,
            tempo               REAL,
            duration_ms         INTEGER,
            year                INTEGER,
            release_month       INTEGER,
            followers           INTEGER,
            followers_log       REAL,
            artist_popularity   INTEGER,
            album_popularity    INTEGER,
            label               TEXT,
            album_type          TEXT,
            is_chart_hit        INTEGER,
            streams             INTEGER,
            in_spotify_playlists INTEGER,
            genre_0             TEXT
        );

        -- Genre aggregation
        CREATE TABLE IF NOT EXISTS agg_genre (
            genre               TEXT PRIMARY KEY,
            track_count         INTEGER,
            avg_popularity      REAL,
            avg_danceability    REAL,
            avg_energy          REAL,
            avg_valence         REAL,
            avg_acousticness    REAL,
            avg_speechiness     REAL,
            avg_tempo           REAL,
            avg_loudness        REAL,
            chart_hits          INTEGER
        );

        -- Year trend aggregation
        CREATE TABLE IF NOT EXISTS agg_year (
            year                INTEGER PRIMARY KEY,
            track_count         INTEGER,
            avg_popularity      REAL,
            avg_danceability    REAL,
            avg_energy          REAL,
            avg_valence         REAL,
            avg_acousticness    REAL,
            avg_tempo           REAL,
            chart_hits          INTEGER
        );

        -- Label aggregation
        CREATE TABLE IF NOT EXISTS agg_label (
            label                   TEXT PRIMARY KEY,
            track_count             INTEGER,
            avg_popularity          REAL,
            avg_album_popularity    REAL,
            chart_hits              INTEGER,
            total_streams           INTEGER
        );

        -- Hit vs non-hit comparison
        CREATE TABLE IF NOT EXISTS agg_hit_vs_nohit (
            is_chart_hit            INTEGER PRIMARY KEY,
            track_count             INTEGER,
            avg_popularity          REAL,
            avg_danceability        REAL,
            avg_energy              REAL,
            avg_valence             REAL,
            avg_acousticness        REAL,
            avg_speechiness         REAL,
            avg_instrumentalness    REAL,
            avg_liveness            REAL,
            avg_tempo               REAL,
            avg_loudness            REAL,
            avg_duration_ms         REAL,
            avg_followers_log       REAL,
            avg_artist_popularity   REAL
        );

        -- ML model metrics
        CREATE TABLE IF NOT EXISTS ml_metrics (
            model_name  TEXT,
            metric      TEXT,
            value       REAL,
            PRIMARY KEY (model_name, metric)
        );

        -- ML feature importance
        CREATE TABLE IF NOT EXISTS ml_feature_importance (
            model_name  TEXT,
            feature     TEXT,
            importance  REAL,
            rank        INTEGER,
            PRIMARY KEY (model_name, feature)
        );
    """)

    conn.commit()
    print("  All tables created successfully.")


def load_all(conn):
    """Load all curated CSVs into the database."""

    # ── fact_popularity (from master_ml — sample 200k for performance) ──
    print("\n  Loading fact_popularity (master_ml)...")
    df = load_csv("master_ml.csv")
    if df is not None:
        cols = [
            "track_id", "track_name", "artist_name", "genre",
            "popularity", "danceability", "energy", "loudness",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo", "duration_ms",
            "year", "release_month", "followers", "followers_log",
            "artist_popularity", "album_popularity", "label",
            "album_type", "is_chart_hit", "streams",
            "in_spotify_playlists", "genre_0"
        ]
        cols_existing = [c for c in cols if c in df.columns]
        df[cols_existing].to_sql(
            "fact_popularity", conn,
            if_exists="replace", index=False,
            chunksize=10000
        )
        print(f"    Loaded {len(df):,} rows into fact_popularity")

        # Also load dim_tracks from same source
        dim_cols = ["track_id", "track_name", "artist_name",
                    "genre", "year", "duration_ms", "is_chart_hit", "streams"]
        dim_existing = [c for c in dim_cols if c in df.columns]
        df[dim_existing].drop_duplicates("track_id").to_sql(
            "dim_tracks", conn,
            if_exists="replace", index=False,
            chunksize=10000
        )
        print(f"    Loaded {len(df):,} rows into dim_tracks")

    # ── dim_artists ──
    print("\n  Loading dim_artists...")
    df = load_csv("agg_artist.csv")
    if df is not None:
        df.to_sql("dim_artists", conn, if_exists="replace", index=False)
        print(f"    Loaded {len(df):,} rows into dim_artists")

    # ── agg_genre ──
    print("\n  Loading agg_genre...")
    df = load_csv("agg_genre.csv")
    if df is not None:
        df.to_sql("agg_genre", conn, if_exists="replace", index=False)
        print(f"    Loaded {len(df):,} rows into agg_genre")

    # ── agg_year ──
    print("\n  Loading agg_year...")
    df = load_csv("agg_year.csv")
    if df is not None:
        df.to_sql("agg_year", conn, if_exists="replace", index=False)
        print(f"    Loaded {len(df):,} rows into agg_year")

    # ── agg_label ──
    print("\n  Loading agg_label...")
    df = load_csv("agg_label.csv")
    if df is not None:
        df.to_sql("agg_label", conn, if_exists="replace", index=False)
        print(f"    Loaded {len(df):,} rows into agg_label")

    # ── agg_hit_vs_nohit ──
    print("\n  Loading agg_hit_vs_nohit...")
    df = load_csv("agg_hit_vs_nohit.csv")
    if df is not None:
        df.to_sql("agg_hit_vs_nohit", conn, if_exists="replace", index=False)
        print(f"    Loaded {len(df):,} rows into agg_hit_vs_nohit")

    # ── ml_metrics (full model + audio only) ──
    print("\n  Loading ml_metrics...")
    rows = []
    for fname, model_name in [
        ("ml_metrics.csv",       "Full model"),
        ("ml_audio_metrics.csv", "Audio only model"),
    ]:
        df = load_csv(fname)
        if df is not None:
            df["model_name"] = model_name
            rows.append(df)
    if rows:
        pd.concat(rows).to_sql(
            "ml_metrics", conn, if_exists="replace", index=False
        )

    # ── ml_feature_importance (full model + audio only) ──
    print("\n  Loading ml_feature_importance...")
    rows = []
    for fname, model_name in [
        ("ml_feature_importance.csv",       "Full model"),
        ("ml_audio_feature_importance.csv", "Audio only model"),
    ]:
        df = load_csv(fname)
        if df is not None:
            df["model_name"] = model_name
            rows.append(df)
    if rows:
        pd.concat(rows).to_sql(
            "ml_feature_importance", conn, if_exists="replace", index=False
        )

    conn.commit()


def print_summary(conn):
    """Print row counts for all tables."""
    cursor = conn.cursor()
    tables = [
        "dim_tracks", "dim_artists", "fact_popularity",
        "agg_genre", "agg_year", "agg_label",
        "agg_hit_vs_nohit", "ml_metrics", "ml_feature_importance"
    ]

    print("\n" + "=" * 50)
    print("  DATABASE SUMMARY")
    print("=" * 50)
    print(f"  {'Table':<30} {'Rows':>10}")
    print("  " + "-" * 42)
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:<30} {count:>10,}")
        except Exception:
            print(f"  {table:<30} {'ERROR':>10}")

    print("=" * 50)
    print(f"  Database file: {DB_NAME}")
    print("=" * 50)


def run_sample_queries(conn):
    """Run a few sample queries to verify data."""
    print("\n  Sample queries:")

    print("\n  Top 5 genres by avg popularity:")
    df = pd.read_sql("""
        SELECT genre, avg_popularity, track_count, chart_hits
        FROM agg_genre
        ORDER BY avg_popularity DESC
        LIMIT 5
    """, conn)
    print(df.to_string(index=False))

    print("\n  Popularity trend (last 5 years):")
    df = pd.read_sql("""
        SELECT year, avg_popularity, track_count
        FROM agg_year
        WHERE year >= 2019
        ORDER BY year
    """, conn)
    print(df.to_string(index=False))

    print("\n  Hit vs Non-hit comparison:")
    df = pd.read_sql("""
        SELECT is_chart_hit, track_count,
               avg_popularity, avg_danceability,
               avg_energy, avg_valence
        FROM agg_hit_vs_nohit
        ORDER BY is_chart_hit
    """, conn)
    print(df.to_string(index=False))

    print("\n  ML model comparison:")
    df = pd.read_sql("""
        SELECT model_name, metric, ROUND(value, 4) as value
        FROM ml_metrics
        ORDER BY model_name, metric
    """, conn)
    print(df.to_string(index=False))


def main():
    print("=" * 55)
    print("  Spotify Big Data — SQLite Analytical Database")
    print("=" * 55)

    # Check required files
    required = ["master_ml.csv", "agg_genre.csv", "agg_year.csv",
                "agg_label.csv", "agg_hit_vs_nohit.csv", "agg_artist.csv",
                "ml_metrics.csv", "ml_feature_importance.csv"]

    print("\n[1/4] Checking required files...")
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"  Missing files: {missing}")
        print("  Download them from OCI bucket first.")
    else:
        print("  All required files found.")

    # Connect and create
    print(f"\n[2/4] Creating database: {DB_NAME}")
    conn = connect()
    create_tables(conn)

    # Load data
    print("\n[3/4] Loading data...")
    load_all(conn)

    # Summary
    print("\n[4/4] Verifying data...")
    print_summary(conn)
    run_sample_queries(conn)

    conn.close()
    print(f"\n  Done! Database saved as: {DB_NAME}")
    print("  Open it with DB Browser for SQLite or pgAdmin.")


if __name__ == "__main__":
    main()
