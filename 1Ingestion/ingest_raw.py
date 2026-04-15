"""
ingest_raw.py
=============
Spotify Big Data Project — Ingestion Layer
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Verifies that all raw dataset files are present in OCI Object Storage
    under the raw/ prefix, logs ingestion metadata (file name, size, timestamp),
    and produces a structured ingestion_log.txt as evidence.

Usage:
    Run from OCI Cloud Shell after manually uploading the 4 CSV files:
        python3 ingest_raw.py --bucket YOUR_BUCKET_NAME --namespace YOUR_NAMESPACE

Prerequisites:
    - OCI CLI configured in Cloud Shell (oci setup config)
    - 4 CSV files already uploaded to raw/ prefix in the bucket:
        raw/spotify_1m_tracks.csv
        raw/charts_2023.csv
        raw/artist_data_2023.csv
        raw/albums_data_2023.csv

Output:
    - ingestion_log.txt (uploaded to raw/ingestion_log.txt in the bucket)
    - Console summary of all verified files
"""

import subprocess
import json
import argparse
import sys
from datetime import datetime, timezone


EXPECTED_FILES = [
    {
        "name": "spotify_1m_tracks.csv",
        "oci_path": "raw/spotify_1m_tracks.csv",
        "description": "Spotify 1 Million Tracks — primary dataset (1.15M rows)",
        "min_size_bytes": 50_000_000,   # expect at least 50MB
    },
    {
        "name": "charts_2023.csv",
        "oci_path": "raw/charts_2023.csv",
        "description": "Most Streamed Spotify Songs 2023 — chart hits (954 rows)",
        "min_size_bytes": 10_000,       # small file ~50KB
    },
    {
        "name": "artist_data_2023.csv",
        "oci_path": "raw/artist_data_2023.csv",
        "description": "Spotify Artist Data 2023 — artist profiles (37k rows)",
        "min_size_bytes": 1_000_000,    # expect at least 1MB
    },
    {
        "name": "albums_data_2023.csv",
        "oci_path": "raw/albums_data_2023.csv",
        "description": "Spotify Albums Data 2023 — album/track metadata (439k rows)",
        "min_size_bytes": 50_000_000,   # expect at least 50MB
    },
]


def run_oci_command(cmd):
    """Run an OCI CLI command and return parsed JSON output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip()
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return result.stdout.strip(), None


def get_object_metadata(bucket, namespace, oci_path):
    """Fetch metadata (size) for a single object using oci os object list."""
    cmd = [
        "oci", "os", "object", "list",
        "--bucket-name", bucket,
        "--namespace", namespace,
        "--prefix", oci_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        objects = data.get("data", [])
        for obj in objects:
            if obj.get("name") == oci_path:
                return {
                    "size_bytes": obj.get("size", 0),
                    "last_modified": obj.get("time-modified", "unknown"),
                    "etag": obj.get("etag", ""),
                }
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def format_size(size_bytes):
    """Convert bytes to human-readable string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f} GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.2f} MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.2f} KB"
    return f"{size_bytes} B"


def verify_bucket_exists(bucket, namespace):
    """Check that the target bucket is accessible."""
    cmd = [
        "oci", "os", "bucket", "get",
        "--bucket-name", bucket,
        "--namespace", namespace,
    ]
    data, error = run_oci_command(cmd)
    if error:
        return False, error
    return True, None


def upload_log(bucket, namespace, log_content):
    """Upload the ingestion log to raw/ingestion_log.txt in OCI."""
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "ingestion_log.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    cmd = [
        "oci", "os", "object", "put",
        "--bucket-name", bucket,
        "--namespace", namespace,
        "--name", "raw/ingestion_log.txt",
        "--file", tmp_path,
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Spotify Big Data — Raw Ingestion Verification"
    )
    parser.add_argument("--bucket", required=True, help="OCI Object Storage bucket name")
    parser.add_argument("--namespace", required=True, help="OCI Object Storage namespace")
    args = parser.parse_args()

    bucket = args.bucket
    namespace = args.namespace
    ingestion_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 65)
    print("  Spotify Big Data Project — Raw Ingestion Verification")
    print("=" * 65)
    print(f"  Bucket   : {bucket}")
    print(f"  Namespace: {namespace}")
    print(f"  Timestamp: {ingestion_timestamp}")
    print("=" * 65)

    # Verify bucket access
    print("\n[1/3] Verifying bucket access...")
    ok, err = verify_bucket_exists(bucket, namespace)
    if not ok:
        print(f"  ERROR: Cannot access bucket '{bucket}': {err}")
        print("  Make sure the bucket exists and OCI CLI is configured.")
        sys.exit(1)
    print(f"  OK — Bucket '{bucket}' is accessible.")

    # Verify each expected file
    print("\n[2/3] Verifying raw dataset files...")
    print()

    results = []
    all_ok = True

    for file_info in EXPECTED_FILES:
        oci_path = file_info["oci_path"]
        print(f"  Checking: {oci_path}")
        metadata = get_object_metadata(bucket, namespace, oci_path)

        if metadata is None:
            status = "MISSING"
            size_str = "—"
            note = "File not found in bucket. Upload it before continuing."
            all_ok = False
            print(f"    Status : MISSING")
            print(f"    Action : Upload {file_info['name']} to raw/ prefix")
        else:
            size_bytes = metadata.get("size_bytes", 0)
            size_str = format_size(size_bytes)
            last_mod = metadata.get("last_modified", "unknown")

            if size_bytes < file_info["min_size_bytes"]:
                status = "WARNING — file smaller than expected"
                note = f"Expected at least {format_size(file_info['min_size_bytes'])}. File may be incomplete or wrong version."
                all_ok = False
                print(f"    Status : WARNING (smaller than expected)")
            else:
                status = "OK"
                note = "File verified successfully."
                print(f"    Status : OK")

            print(f"    Size   : {size_str}")
            print(f"    Modified: {last_mod}")

        results.append({
            "file": file_info["name"],
            "oci_path": oci_path,
            "description": file_info["description"],
            "status": status,
            "size": size_str,
            "note": note,
        })
        print()

    # Build log content
    log_lines = []
    log_lines.append("=" * 65)
    log_lines.append("  SPOTIFY BIG DATA PROJECT — INGESTION LOG")
    log_lines.append("  Universidad Panamericana — Big Data Final Project")
    log_lines.append("  Student: Adler Antonio Calvillo Arellano")
    log_lines.append("=" * 65)
    log_lines.append(f"  Ingestion timestamp : {ingestion_timestamp}")
    log_lines.append(f"  OCI Bucket          : {bucket}")
    log_lines.append(f"  OCI Namespace       : {namespace}")
    log_lines.append(f"  Raw prefix          : raw/")
    log_lines.append("=" * 65)
    log_lines.append("")
    log_lines.append("DATASET VERIFICATION RESULTS")
    log_lines.append("-" * 65)

    for r in results:
        log_lines.append(f"File        : {r['file']}")
        log_lines.append(f"OCI path    : {r['oci_path']}")
        log_lines.append(f"Description : {r['description']}")
        log_lines.append(f"Size        : {r['size']}")
        log_lines.append(f"Status      : {r['status']}")
        log_lines.append(f"Note        : {r['note']}")
        log_lines.append("-" * 65)

    overall = "ALL FILES VERIFIED SUCCESSFULLY" if all_ok else "ISSUES FOUND — REVIEW WARNINGS ABOVE"
    log_lines.append("")
    log_lines.append(f"OVERALL STATUS: {overall}")
    log_lines.append("")
    log_lines.append("PIPELINE NEXT STEPS:")
    log_lines.append("  1. Run spark_cleaning.py via OCI Data Flow (raw/ → processed/)")
    log_lines.append("  2. Run spark_joins.py via OCI Data Flow (processed/ → processed/joined/)")
    log_lines.append("  3. Run spark_aggregations.py via OCI Data Flow (processed/joined/ → curated/)")
    log_lines.append("  4. Run streaming_producer.py + streaming_consumer.py")
    log_lines.append("  5. Load curated/ into PostgreSQL with load_to_db.py")
    log_lines.append("  6. Train model with popularity_regression.py via OCI Data Flow")
    log_lines.append("")
    log_lines.append(f"Log generated: {ingestion_timestamp}")

    log_content = "\n".join(log_lines)

    # Upload log to OCI
    print("[3/3] Uploading ingestion_log.txt to raw/...")
    uploaded = upload_log(bucket, namespace, log_content)
    if uploaded:
        print("  OK — raw/ingestion_log.txt uploaded successfully.")
    else:
        print("  WARNING — Could not upload log to OCI. Saving locally as ingestion_log.txt")
        with open("ingestion_log.txt", "w") as f:
            f.write(log_content)

    # Final summary
    print()
    print("=" * 65)
    print(f"  OVERALL STATUS: {overall}")
    print("=" * 65)

    if not all_ok:
        print()
        print("  ACTION REQUIRED:")
        for r in results:
            if r["status"] != "OK":
                print(f"  - {r['file']}: {r['note']}")
        print()
        sys.exit(1)
    else:
        print()
        print("  All datasets are in place. You may proceed to Spark processing.")
        print()


if __name__ == "__main__":
    main()
