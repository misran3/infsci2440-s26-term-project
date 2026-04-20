"""Clean and preprocess reviews."""

import argparse
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import DATA, PREPROCESS


def preprocess_data(
    input_path: Path | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Clean and preprocess reviews.

    Args:
        input_path: Input CSV path. Defaults to raw_reviews.
        output_path: Output CSV path. Defaults to clean_reviews.
        force: If True, reprocess even if output exists.

    Returns:
        Cleaned DataFrame, or None if skipped.
    """
    input_path = input_path or DATA.raw_reviews
    output_path = output_path or DATA.clean_reviews

    if output_path.exists() and not force:
        print(f"Skipping preprocess: {output_path} already exists")
        print("  Use --force to re-process")
        return None

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    original_count = len(df)

    # Remove duplicates by text
    df = df.drop_duplicates(subset=["text"])
    after_dedup = len(df)
    print(f"  Removed {original_count - after_dedup} duplicates.")

    # Remove empty/null text
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]

    # Normalize whitespace
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x).strip()))

    # Filter by text length
    df = df[df["text"].str.len() >= PREPROCESS.min_text_length]
    df = df[df["text"].str.len() <= PREPROCESS.max_text_length]

    # Filter by rating range
    df = df[df["rating"].between(PREPROCESS.min_rating, PREPROCESS.max_rating)]

    # Reset index
    df = df.reset_index(drop=True)

    cleaned_count = len(df)
    removed = original_count - cleaned_count
    pct_kept = (cleaned_count / original_count) * 100

    print(f"Cleaned: {original_count} -> {cleaned_count} reviews")
    print(f"  Removed {removed} reviews ({pct_kept:.1f}% kept)")

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


def main():
    """Entry point for script."""
    parser = argparse.ArgumentParser(description="Clean and preprocess reviews")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    preprocess_data(force=args.force)


if __name__ == "__main__":
    main()
