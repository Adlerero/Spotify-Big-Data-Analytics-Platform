"""
streaming_producer.py
=====================
Spotify Big Data Project — Streaming Layer
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Simulates real-time track play events by reading tracks from the
    processed dataset and publishing JSON messages to OCI Streaming
    (Kafka-compatible). Each message represents a user playing a song.

Event schema (JSON):
    {
        "event_id":        "uuid",
        "timestamp":       "2026-04-13T22:00:00Z",
        "track_id":        "53QF56cjZA9RTuuMZDrSA6",
        "track_name":      "I Won't Give Up",
        "artist_name":     "Jason Mraz",
        "genre":           "acoustic",
        "popularity":      68,
        "danceability":    0.483,
        "energy":          0.303,
        "valence":         0.139,
        "duration_ms":     240166,
        "play_duration_ms": 180000
    }

Prerequisites:
    pip install oci kafka-python

Usage:
    python streaming_producer.py \
        --bootstrap "cell-1.streaming.mx-queretaro-1.oci.oraclecloud.com:9092" \
        --topic "spotify-play-events" \
        --stream-ocid "ocid1.stream.oc1..." \
        --user "ocid1.user.oc1..." \
        --tenancy "ocid1.tenancy.oc1..." \
        --fingerprint "xx:xx:xx:..." \
        --key-file "~/.oci/oci_api_key.pem" \
        --csv "processed_tracks.csv" \
        --events 1000 \
        --rate 10
"""

import argparse
import csv
import json
import random
import time
import uuid
from datetime import datetime, timezone


def load_tracks(csv_path, max_tracks=5000):
    """Load a sample of tracks from the processed CSV."""
    tracks = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_tracks:
                    break
                # Only keep rows with essential fields
                if row.get("track_name") and row.get("artist_name"):
                    tracks.append(row)
        print(f"  Loaded {len(tracks):,} tracks from {csv_path}")
    except FileNotFoundError:
        print(f"  ERROR: File not found: {csv_path}")
        print("  Download it first with:")
        print("  oci os object get --bucket-name bd-raw-spotify \\")
        print("    --namespace axz6vs6cibbb \\")
        print("    --name processed/tracks/part-00000-*.csv \\")
        print("    --file processed_tracks.csv")
        raise
    return tracks


def build_event(track):
    """Build a simulated play event from a track row."""
    duration_ms = int(float(track.get("duration_ms", 180000) or 180000))
    # Simulate user listening between 30s and full duration
    play_duration = random.randint(
        min(30000, duration_ms),
        duration_ms
    )

    def safe_int(val, default=0):
        try:
            return int(float(val)) if val else default
        except (ValueError, TypeError):
            return default

    def safe_float(val, default=0.0):
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    return {
        "event_id":        str(uuid.uuid4()),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "track_id":        track.get("track_id", ""),
        "track_name":      track.get("track_name", ""),
        "artist_name":     track.get("artist_name", ""),
        "genre":           track.get("genre", "unknown"),
        "popularity":      safe_int(track.get("popularity", 0)),
        "danceability":    safe_float(track.get("danceability", 0)),
        "energy":          safe_float(track.get("energy", 0)),
        "valence":         safe_float(track.get("valence", 0)),
        "duration_ms":     duration_ms,
        "play_duration_ms": play_duration,
    }


def produce_to_oci_streaming(args, tracks):
    """Publish events to OCI Streaming using Kafka-compatible API."""
    try:
        from kafka import KafkaProducer
        from kafka.errors import KafkaError
    except ImportError:
        print("  ERROR: kafka-python not installed.")
        print("  Run: pip install kafka-python")
        raise

    # OCI Streaming uses SASL_SSL with PLAIN mechanism
    # Username format: <tenancy_ocid>/<user_ocid>/<stream_pool_ocid>
    username = f"{args.tenancy}/{args.user}/{args.stream_ocid}"

    # Read private key for password
    key_path = args.key_file.replace("~", str(__import__("pathlib").Path.home()))
    with open(key_path, "r") as f:
        private_key = f.read()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=username,
        sasl_plain_password=private_key,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        retries=3,
    )

    print(f"\n  Connected to OCI Streaming: {args.bootstrap}")
    print(f"  Topic: {args.topic}")
    print(f"  Sending {args.events} events at {args.rate} events/sec...")
    print()

    sent      = 0
    errors    = 0
    start     = time.time()
    interval  = 1.0 / args.rate

    for i in range(args.events):
        track = random.choice(tracks)
        event = build_event(track)

        try:
            producer.send(
                args.topic,
                key=event["track_id"],
                value=event
            )
            sent += 1

            if sent % 100 == 0:
                elapsed = time.time() - start
                rate    = sent / elapsed
                print(f"  Sent: {sent:,} events | "
                      f"Rate: {rate:.1f} evt/s | "
                      f"Elapsed: {elapsed:.1f}s")

        except Exception as e:
            errors += 1
            print(f"  ERROR sending event {i}: {e}")

        time.sleep(interval)

    producer.flush()
    producer.close()

    elapsed = time.time() - start
    print(f"\n  Done! Sent: {sent:,} | Errors: {errors} | "
          f"Total time: {elapsed:.1f}s")


def produce_local_simulation(tracks, num_events, rate):
    """
    Fallback: simulate streaming locally without OCI.
    Prints events to stdout and saves to a local JSONL file.
    Useful for testing or if OCI Streaming is not configured.
    """
    output_file = "simulated_events.jsonl"
    print(f"\n  LOCAL SIMULATION MODE")
    print(f"  Writing {num_events} events to {output_file}")
    print(f"  Rate: {rate} events/sec")
    print()

    interval = 1.0 / rate
    sent     = 0
    start    = time.time()

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_events):
            track = random.choice(tracks)
            event = build_event(track)

            f.write(json.dumps(event) + "\n")
            sent += 1

            if sent % 50 == 0:
                elapsed = time.time() - start
                print(f"  Simulated: {sent:,}/{num_events} events | "
                      f"Elapsed: {elapsed:.1f}s")
                # Show a sample event every 100
                if sent % 100 == 0:
                    print(f"  Sample: {event['track_name']} "
                          f"by {event['artist_name']} "
                          f"(genre: {event['genre']}, "
                          f"popularity: {event['popularity']})")

            time.sleep(interval)

    elapsed = time.time() - start
    print(f"\n  Simulation complete!")
    print(f"  Events written : {sent:,}")
    print(f"  Output file    : {output_file}")
    print(f"  Total time     : {elapsed:.1f}s")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Spotify Big Data — Streaming Producer"
    )
    parser.add_argument("--bootstrap",   help="OCI Streaming bootstrap server")
    parser.add_argument("--topic",       default="spotify-play-events")
    parser.add_argument("--stream-ocid", help="OCI Stream Pool OCID")
    parser.add_argument("--user",        help="OCI User OCID")
    parser.add_argument("--tenancy",     help="OCI Tenancy OCID")
    parser.add_argument("--fingerprint", help="API key fingerprint")
    parser.add_argument("--key-file",    default="~/.oci/oci_api_key.pem")
    parser.add_argument("--csv",         default="processed_tracks.csv",
                        help="Path to processed tracks CSV")
    parser.add_argument("--events",      type=int, default=1000,
                        help="Number of events to produce")
    parser.add_argument("--rate",        type=float, default=5.0,
                        help="Events per second")
    parser.add_argument("--local",       action="store_true",
                        help="Run in local simulation mode (no OCI required)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spotify Big Data — Streaming Producer")
    print("=" * 60)

    # Load tracks
    print(f"\n[1/2] Loading tracks from {args.csv}...")
    tracks = load_tracks(args.csv)

    # Produce events
    print(f"\n[2/2] Producing {args.events} play events...")

    if args.local or not args.bootstrap:
        # Local simulation — no OCI Streaming needed
        produce_local_simulation(tracks, args.events, args.rate)
    else:
        # OCI Streaming (Kafka)
        produce_to_oci_streaming(args, tracks)


if __name__ == "__main__":
    main()
