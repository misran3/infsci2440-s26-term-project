"""Helper script to read batches of curated_labels.csv for review."""

import pandas as pd
import sys
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "curated_labels.csv"

def show_batch(start: int, size: int = 10):
    """Show a batch of reviews for manual verification."""
    df = pd.read_csv(CSV_PATH)
    end = min(start + size, len(df))

    print(f"\n{'='*80}")
    print(f"REVIEWS {start+1} to {end} of {len(df)}")
    print(f"{'='*80}\n")

    for i in range(start, end):
        row = df.iloc[i]
        print(f"[{i}] review_id: {row['review_id']}")
        print(f"    rating: {row['rating']} | auto_topic: {row['auto_topic']} | confidence: {row['confidence_score']}")
        print(f"    matched: {row['matched_keywords']}")
        print(f"    text: {row['text'][:500]}{'...' if len(str(row['text'])) > 500 else ''}")
        print(f"    current verified_topic: {row['verified_topic'] if pd.notna(row['verified_topic']) and row['verified_topic'] else '(empty)'}")
        print("-" * 80)

def update_verified(updates: dict):
    """Update verified_topic for given indices."""
    df = pd.read_csv(CSV_PATH)
    for idx, topic in updates.items():
        df.at[idx, 'verified_topic'] = topic
    df.to_csv(CSV_PATH, index=False)
    print(f"Updated {len(updates)} rows")

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    show_batch(start, size)
