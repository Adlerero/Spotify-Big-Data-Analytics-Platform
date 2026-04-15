"""
streaming_consumer.py
=====================
Spotify Big Data Project — Streaming Layer
Universidad Panamericana — Big Data Final Project
Student: Adler Antonio Calvillo Arellano

Purpose:
    Reads simulated track play events from a local JSONL file
    (produced by streaming_producer.py in local mode) and processes
    them in micro-batches to simulate Spark Structured Streaming.

    Computes per micro-batch:
        - Top 5 most played tracks
        - Play count per genre
        - Average play duration per genre
        - Total events processed

    Results are printed to console and saved to streaming_results.csv.

Usage:
    # First run the producer:
    python streaming_producer.py --local --events 1000 --rate 50

    # Then run the consumer:
    python streaming_consumer.py --input simulated_events.jsonl --batch-size 100
"""

import argparse
import json
import csv
import time
from collections import defaultdict
from datetime import datetime, timezone


def process_batch(batch, batch_number):
    """
    Process a single micro-batch of events.
    Returns aggregated results for this batch.
    """
    if not batch:
        return None

    # Counters
    track_counts   = defaultdict(int)
    genre_counts   = defaultdict(int)
    genre_duration = defaultdict(list)
    total_plays    = len(batch)

    for event in batch:
        track_key = f"{event.get('track_name', 'Unknown')} — {event.get('artist_name', 'Unknown')}"
        genre     = event.get("genre", "unknown")
        duration  = event.get("play_duration_ms", 0)

        track_counts[track_key] += 1
        genre_counts[genre]     += 1
        genre_duration[genre].append(duration)

    # Top 5 tracks
    top_tracks = sorted(
        track_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Avg duration per genre
    genre_avg_duration = {
        genre: sum(durations) / len(durations)
        for genre, durations in genre_duration.items()
    }

    # Top 5 genres
    top_genres = sorted(
        genre_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "batch_number":      batch_number,
        "batch_size":        total_plays,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "top_tracks":        top_tracks,
        "top_genres":        top_genres,
        "genre_avg_duration": genre_avg_duration,
    }


def print_batch_results(result):
    """Print micro-batch results to console."""
    print(f"\n{'='*55}")
    print(f"  MICRO-BATCH #{result['batch_number']} "
          f"| {result['batch_size']} events "
          f"| {result['timestamp']}")
    print(f"{'='*55}")

    print("\n  Top 5 most played tracks:")
    for i, (track, count) in enumerate(result["top_tracks"], 1):
        print(f"    {i}. {track[:45]:<45} ({count} plays)")

    print("\n  Top 5 genres:")
    for genre, count in result["top_genres"]:
        bar = "█" * min(count, 30)
        print(f"    {genre:<20} {bar} {count}")

    print("\n  Avg play duration by genre (top 5):")
    top_genre_names = [g for g, _ in result["top_genres"]]
    for genre in top_genre_names:
        if genre in result["genre_avg_duration"]:
            avg_sec = result["genre_avg_duration"][genre] / 1000
            print(f"    {genre:<20} {avg_sec:.1f}s avg")


def save_results(all_results, output_file):
    """Save all batch results summary to CSV."""
    if not all_results:
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_number", "batch_size", "timestamp",
            "top_track_1", "top_track_1_plays",
            "top_genre_1", "top_genre_1_plays",
            "top_genre_2", "top_genre_2_plays",
        ])

        for r in all_results:
            top_track  = r["top_tracks"][0]  if r["top_tracks"]  else ("", 0)
            top_genre1 = r["top_genres"][0]  if len(r["top_genres"]) > 0 else ("", 0)
            top_genre2 = r["top_genres"][1]  if len(r["top_genres"]) > 1 else ("", 0)

            writer.writerow([
                r["batch_number"],
                r["batch_size"],
                r["timestamp"],
                top_track[0][:50],
                top_track[1],
                top_genre1[0],
                top_genre1[1],
                top_genre2[0],
                top_genre2[1],
            ])

    print(f"\n  Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Spotify Big Data — Streaming Consumer (micro-batch)"
    )
    parser.add_argument("--input",      default="simulated_events.jsonl",
                        help="Input JSONL file from producer")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Events per micro-batch")
    parser.add_argument("--output",     default="streaming_results.csv",
                        help="Output CSV for batch results")
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds between batches (simulates processing time)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spotify Big Data — Streaming Consumer")
    print("  Mode: Micro-batch simulation")
    print(f"  Input      : {args.input}")
    print(f"  Batch size : {args.batch_size} events")
    print("=" * 60)

    # Load all events from JSONL file
    print(f"\n[1/3] Loading events from {args.input}...")
    events = []
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        print(f"  ERROR: {args.input} not found.")
        print("  Run streaming_producer.py --local first.")
        return

    total_events = len(events)
    print(f"  Total events loaded : {total_events:,}")

    if total_events == 0:
        print("  No events to process.")
        return

    # Process in micro-batches
    print(f"\n[2/3] Processing {total_events:,} events "
          f"in batches of {args.batch_size}...")

    all_results   = []
    batch_number  = 1
    total_batches = (total_events + args.batch_size - 1) // args.batch_size

    for i in range(0, total_events, args.batch_size):
        batch  = events[i:i + args.batch_size]
        result = process_batch(batch, batch_number)

        if result:
            all_results.append(result)
            print_batch_results(result)

        batch_number += 1

        # Simulate processing delay between batches
        if i + args.batch_size < total_events:
            time.sleep(args.delay)

    # Save results
    print(f"\n[3/3] Saving results...")
    save_results(all_results, args.output)

    # Final summary
    total_processed = sum(r["batch_size"] for r in all_results)
    print("\n" + "=" * 60)
    print("  Streaming Consumer — Final Summary")
    print("=" * 60)
    print(f"  Total batches processed : {len(all_results)}")
    print(f"  Total events processed  : {total_processed:,}")
    print(f"  Results file            : {args.output}")
    print("\n  Overall top genres across all batches:")

    # Aggregate across all batches
    global_genre_counts = defaultdict(int)
    for r in all_results:
        for genre, count in r["top_genres"]:
            global_genre_counts[genre] += count

    for genre, count in sorted(
        global_genre_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"    {genre:<25} {count:>6} plays")

    print("\n  Streaming simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
