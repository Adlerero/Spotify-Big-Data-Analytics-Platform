# Spotify Big Data Analytics Platform
**Big Data Final Project — Universidad Panamericana**
**Course:** Machine Learning II (Big Data)
**Student:** Adler Antonio Calvillo Arellano
**Platform:** Oracle Cloud Infrastructure (OCI)

---

## Project Overview

End-to-end Big Data pipeline that processes over 1.15 million Spotify tracks to identify the patterns that determine a song's success. Combines audio features, artist profiles, album metadata, and real chart data to answer: *What makes a song popular on Spotify?*

**Datasets used:**
- Spotify 1 Million Tracks — 1,159,765 rows (Kaggle)
- Most Streamed Spotify Songs 2023 — 954 rows (Kaggle)
- Spotify Artist Data 2023 — 37,013 rows (Kaggle)
- Spotify Albums Data 2023 — 438,974 rows (Kaggle)

---

## Repository Structure

```
spotify-bigdata/
├── README.md
├── 1_ingestion/
│   └── download_datasets.py        # Downloads CSVs and uploads to OCI raw/
├── 2_processing/
│   ├── spark_cleaning.py           # Null removal, type casting, normalization
│   ├── spark_joins.py              # Joins: tracks + artists + charts + albums
│   └── spark_aggregations.py      # GroupBy genre, year, label, artist
├── 3_streaming/
│   ├── streaming_producer.py       # Simulates track play events to OCI Streaming
│   └── streaming_consumer.py      # Spark micro-batch consumer
├── 4_database/
│   ├── create_tables.sql           # PostgreSQL DDL
│   └── load_to_db.py               # Loads curated data to PostgreSQL
├── 5_ml/
│   └── popularity_regression.py   # Random Forest Regressor with Spark MLlib
└── screenshots/
    ├── 01_raw_bucket_folder.png
    ├── 02_ingestion_script_run.png
    └── ... (all execution evidence)
```

---

## Prerequisites

### OCI Setup
- OCI account with access to: Object Storage, Data Flow, Streaming, DB System, NoSQL Database
- OCI CLI configured in Cloud Shell
- A bucket created (e.g. `spotify-bigdata-bucket`) with prefixes: `raw/`, `processed/`, `curated/`, `scripts/`

### Python dependencies (Cloud Shell)
```bash
pip install kaggle pandas
```

### Configure Kaggle API
```bash
mkdir -p ~/.kaggle
# Upload your kaggle.json to Cloud Shell, then:
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Step-by-Step Execution

### Step 1 — Ingest raw data

```bash
# In OCI Cloud Shell:
python3 1_ingestion/download_datasets.py
```

This script will:
1. Download all 4 datasets from Kaggle
2. Upload them to `raw/` in OCI Object Storage
3. Write a timestamped `ingestion_log.txt`

**Screenshot:** raw/ folder in OCI Object Storage bucket showing all 4 files.

---

### Step 2 — Verify raw layer

```bash
oci os object list --bucket-name spotify-bigdata-bucket --prefix raw/
```

**Screenshot:** Terminal output listing raw files with sizes.

---

### Step 3 — Run Spark cleaning job (OCI Data Flow)

1. Upload script to OCI:
```bash
oci os object put \
  --bucket-name spotify-bigdata-bucket \
  --file 2_processing/spark_cleaning.py \
  --name scripts/spark_cleaning.py
```

2. Create and run Data Flow application pointing to `scripts/spark_cleaning.py`
3. Input: `oci://spotify-bigdata-bucket/raw/`
4. Output: `oci://spotify-bigdata-bucket/processed/`

**Screenshot:** OCI Data Flow job status = Succeeded.

---

### Step 4 — Run Spark joins job

```bash
oci os object put \
  --bucket-name spotify-bigdata-bucket \
  --file 2_processing/spark_joins.py \
  --name scripts/spark_joins.py
```

Run as Data Flow application. Output → `processed/joined/`

**Screenshot:** Spark `.show()` output of joined DataFrame.

---

### Step 5 — Run Spark aggregations job

```bash
oci os object put \
  --bucket-name spotify-bigdata-bucket \
  --file 2_processing/spark_aggregations.py \
  --name scripts/spark_aggregations.py
```

Output → `curated/`

**Screenshot:** Aggregation results by genre and year.

---

### Step 6 — Run streaming simulation

Terminal 1 (producer):
```bash
python3 3_streaming/streaming_producer.py
```

Terminal 2 (consumer — run as Data Flow or local Spark):
```bash
python3 3_streaming/streaming_consumer.py
```

**Screenshot:** OCI Streaming console showing incoming messages + consumer output.

---

### Step 7 — Create and load PostgreSQL database

```bash
# Connect to OCI DB System and run DDL:
psql -h <db-host> -U admin -d spotifydb -f 4_database/create_tables.sql

# Load data:
python3 4_database/load_to_db.py
```

**Screenshot:** `\dt` command showing tables created + `SELECT COUNT(*)` on each table.

---

### Step 8 — Train ML model

```bash
oci os object put \
  --bucket-name spotify-bigdata-bucket \
  --file 5_ml/popularity_regression.py \
  --name scripts/popularity_regression.py
```

Run as Data Flow application. Input: curated ML-ready dataset.

**Screenshot:** Terminal showing RMSE, R², and sample predictions.

---

### Step 9 — Build Power BI dashboard

1. Connect Power BI Desktop to OCI PostgreSQL
2. Import curated tables
3. Build visualizations (see report Section 8 for details)

**Screenshot:** Completed dashboard with all 4 charts.

---

## Architecture Summary

```
Data Sources (Kaggle CSVs)
        ↓
Ingestion (Python + OCI Cloud Shell)
        ↓
Raw Layer (OCI Object Storage — raw/)
        ↓
Batch Processing (OCI Data Flow — PySpark)
  · Cleaning  · Transforms  · Joins  · Aggregations
        ↓
Processed Layer (OCI Object Storage — processed/)
        ↓
Curated Layer (OCI Object Storage — curated/)
        ↓
Streaming Simulation (OCI Streaming — Kafka-compatible)
        ↓
Analytical Model (PostgreSQL + OCI NoSQL)
        ↓
Visualization (Power BI Dashboard)
        ↓
AI Component (Spark MLlib — Random Forest Regression)
```

---

## Key Analytical Questions Answered

1. What audio features most strongly predict a song's popularity score?
2. Do artists with more followers consistently produce more popular songs?
3. How have danceability, energy, and valence evolved from 2009 to 2024?
4. Which record labels dominate the most popular genres?
5. What distinguishes the 954 chart hits from the remaining 1.15M songs?

---

## Notes

- All raw datasets must be kept in their original unprocessed form in `raw/`
- Do not re-upload pre-cleaned datasets — the pipeline performs all cleaning internally
- OCI Data Flow minimum configuration: 1 OCPU, 8GB memory (VM.Standard.E3.Flex)
- Spark output files follow the `part-00000-*.csv` naming convention — this is expected
