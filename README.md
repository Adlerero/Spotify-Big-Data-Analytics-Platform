# Spotify Big Data Analytics Platform
**What Makes a Song Successful on Spotify?**

**Course:** Big Data — Universidad Panamericana, Campus Aguascalientes  
**Student:** Adler Antonio Calvillo Arellano (ID: 0257691)  
**Professor:** Alfredo Márquez Martínez  
**Platform:** Oracle Cloud Infrastructure (OCI) + Local (Spark MLlib, SQLite, Power BI)

---

## Project Overview

End-to-end Big Data pipeline that processes over 1.15 million Spotify tracks to identify the patterns that determine a song's success. Combines audio features, artist profiles, album metadata, and real chart data across 8 pipeline layers.

**Central question:** What factors determine the success of a song on Spotify — its audio characteristics, the artist's profile, or the distribution strategy behind it?

**Key finding:** Genre (66.3%) and release year (22.0%) together explain 88% of popularity variance. All 13 audio features combined account for less than 5%.

---

## Architecture

```
Data Sources (4 Kaggle CSVs)
        ↓
Ingestion (Python + OCI CLI)
        ↓
OCI Object Storage — Data Lake
    raw/          ← original unmodified CSVs
    processed/    ← cleaned + joined by Spark
    curated/      ← aggregations + ML-ready dataset
        ↓
Batch Processing (OCI Data Flow — PySpark)
    spark_cleaning.py → spark_joins.py → spark_aggregations.py
        ↓
Streaming Simulation (Python local)
    streaming_producer.py → streaming_consumer.py
        ↓
Analytical Model (SQLite — 9 tables, 1.1M+ rows)
        ↓
Visualization (Power BI Desktop — 7 charts)
        ↓
AI Component (Spark MLlib local — 2 Random Forest models)
```

---

## Repository Structure

```
spotify-bigdata/
│
├── README.md
│
├── 1_ingestion/
│   └── ingest_raw.py                        # Verify files + generate ingestion log
│
├── 2_processing/
│   ├── spark_cleaning.py                    # OCI Data Flow — cleaning all 4 datasets
│   ├── spark_joins.py                       # OCI Data Flow — 3 LEFT JOINs
│   └── spark_aggregations.py               # OCI Data Flow — 6 aggregations + master_ml
│
├── 3_streaming/
│   ├── streaming_producer.py               # Generate 1,000 simulated play events
│   └── streaming_consumer.py              # Micro-batch processing (10 batches x 100)
│
├── 4_database/
│   └── load_to_db.py                       # Load curated CSVs into SQLite
│
├── 5_ml/
│   ├── popularity_regression_local.py      # Full model — 22 features, R²=0.624
│   └── popularity_regression_audio_only.py # Audio-only model — 13 features, R²=0.156
```

---

## Datasets

All datasets sourced from Kaggle. Download and place in the same folder before running scripts.

| File | Source | Rows |
|---|---|---|
| `spotify_1m_tracks.csv` | kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks | 1,159,764 |
| `charts_2023.csv` | kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023 | 953 |
| `artist_data_2023.csv` | kaggle.com/datasets/tonygordonjr/spotify-dataset-2023 | 37,012 |
| `albums_data_2023.csv` | kaggle.com/datasets/tonygordonjr/spotify-dataset-2023 | 438,973 |

---

## Prerequisites

### OCI Setup
- OCI account with access to: Object Storage, Data Flow
- OCI CLI installed and configured on Windows (`oci setup config`)
- Bucket created: `bd-raw-spotify` in region `mx-queretaro-1`
- Bucket prefixes created: `raw/`, `processed/`, `curated/`, `scripts/`

### Local Setup
```cmd
pip install pyspark pandas
```

### Verify OCI CLI
```cmd
oci os ns get
```

---

## Step-by-Step Execution

### Step 1 — Upload raw datasets to OCI

```cmd
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name raw/spotify_1m_tracks.csv --file spotify_1m_tracks.csv
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name raw/charts_2023.csv --file charts_2023.csv
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name raw/artist_data_2023.csv --file artist_data_2023.csv
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name raw/albums_data_2023.csv --file albums_data_2023.csv
```

📸 Screenshot: OCI Console showing 4 files in raw/ prefix.

---

### Step 2 — Run ingestion verification

```cmd
python 1_ingestion/ingest_raw.py --bucket bd-raw-spotify --namespace axz6vs6cibbb
```

Expected: `ALL FILES VERIFIED SUCCESSFULLY`

📸 Screenshot: Terminal output.

---

### Step 3 — Upload Spark scripts to OCI

```cmd
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name scripts/spark_cleaning.py --file 2_processing/spark_cleaning.py
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name scripts/spark_joins.py --file 2_processing/spark_joins.py
oci os object put --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name scripts/spark_aggregations.py --file 2_processing/spark_aggregations.py
```

---

### Step 4 — Run spark_cleaning.py (OCI Data Flow)

Create application in OCI Console → Analytics & AI → Data Flow → Applications:

| Field | Value |
|---|---|
| Name | `spotify-spark-cleaning` |
| Language | Python |
| File name | `scripts/spark_cleaning.py` |
| Arguments | `bd-raw-spotify axz6vs6cibbb` |
| Shape | VM.Standard.E4.Flex, 1 OCPU, 8 GB |
| Executors | 1 |

Expected log output:
```
[1/4] Cleaning spotify_1m_tracks... Cleaned rows: 1,156,328
[4/4] Cleaning albums_data_2023...  Cleaned rows: 438,522
```

📸 Screenshot: OCI Data Flow job Succeeded.

---

### Step 5 — Run spark_joins.py (OCI Data Flow)

Same configuration, file: `scripts/spark_joins.py`

Expected log output:
```
Final master table rows    : 1,158,376
Final master table columns : 39
```

📸 Screenshot: OCI Data Flow job Succeeded.

---

### Step 6 — Run spark_aggregations.py (OCI Data Flow)

Same configuration, file: `scripts/spark_aggregations.py`

Expected log output:
```
Genres found: 82
ML dataset rows: 1,158,376
Aggregations complete. Curated layer ready.
```

📸 Screenshot: OCI Data Flow job Succeeded.

---

### Step 7 — Verify folder structure

```cmd
oci os object list --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --prefix raw/
oci os object list --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --prefix processed/
oci os object list --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --prefix curated/
```

📸 Screenshot: All 3 prefixes with files visible.

---

### Step 8 — Download curated CSVs for local steps

List and download each file using the exact part-00000-*.csv filename:
```cmd
oci os object list --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --prefix curated/master_ml/
oci os object get --bucket-name bd-raw-spotify --namespace axz6vs6cibbb --name curated/master_ml/EXACT_NAME.csv --file master_ml.csv
# Repeat for: agg_genre.csv, agg_year.csv, agg_label.csv, agg_hit_vs_nohit.csv, agg_artist.csv
# Also download: processed/tracks/EXACT_NAME.csv → processed_tracks.csv
```

---

### Step 9 — Run streaming simulation

```cmd
python 3_streaming/streaming_producer.py --local --events 1000 --rate 50 --csv processed_tracks.csv
python 3_streaming/streaming_consumer.py --input simulated_events.jsonl --batch-size 100
```

Expected: 10 micro-batches processed, streaming_results.csv generated.

📸 Screenshots: Producer running + Consumer micro-batch output.

---

### Step 10 — Load SQLite database

Place all curated CSVs + ml_metrics.csv + ml_feature_importance.csv + ml_audio_metrics.csv + ml_audio_feature_importance.csv in the same folder.

```cmd
python 4_database/load_to_db.py
```

Expected: `Done! Database saved as: spotify_analytics.db`

📸 Screenshot: DB Browser showing 9 tables.

---

### Step 11 — Train ML models

```cmd
python 5_ml/popularity_regression_local.py --csv master_ml.csv --trees 50
python 5_ml/popularity_regression_audio_only.py --csv master_ml.csv --trees 50
```

Expected metrics:
- Full model: RMSE=9.74, R²=0.624
- Audio-only: RMSE=14.60, R²=0.156

📸 Screenshot: Terminal showing metrics and feature importance.

---

### Step 12 — Build Power BI Dashboard

1. Install: http://www.ch-werner.de/sqliteodbc/sqliteodbc_w64.exe
2. Configure DSN via odbcad32: Name=`SpotifyDB`, path to `spotify_analytics.db`
3. Power BI → Get Data → ODBC → SpotifyDB
4. Load: `agg_genre`, `agg_hit_vs_nohit`, `agg_year`, `dim_artists`, `ml_feature_importance`, `ml_metrics`, `agg_label`
5. Build 7 charts as described in Final Report Section 8

📸 Screenshot: Completed dashboard (2 pages).

---

## Results Summary

| Metric | Value |
|---|---|
| Total tracks processed | 1,158,376 |
| Genres analyzed | 82 |
| Chart hits identified | 373 |
| Full model R² | 0.6240 |
| Audio-only model R² | 0.1562 |
| Top predictor | genre_index (66.3%) |
| SQLite tables | 9 |
| Power BI charts | 7 |

---

## Key Findings

1. **Genre explains 66% of popularity** — genre positioning matters more than how a song sounds.
2. **Audio features alone explain only 15.6%** — R² drops from 0.624 to 0.156 without context features.
3. **Chart hits and non-hits sound almost identical** — the 55-point popularity gap is driven by distribution, not sonic quality.
4. **Pop dominates** with avg popularity 55.70, followed by hip-hop (46.41) and rock (46.27).
5. **Loudness and duration** are the strongest audio predictors — not danceability or energy.

---

## Notes

- Raw CSV files must remain in their original downloaded form — never upload pre-cleaned data.
- OCI Data Flow output files use the naming convention `part-00000-*.csv` — this is expected.
- The SQLite database (`spotify_analytics.db`) is generated locally and not stored in OCI.
- Spark MLlib was run locally due to an OCI Data Flow internal network timeout during the ML job.
