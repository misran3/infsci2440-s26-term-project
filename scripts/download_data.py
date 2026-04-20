"""Download Amazon Reviews dataset from HuggingFace."""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download
import pandas as pd

from src.config import DATA, DATASET


def download_data(force: bool = False) -> pd.DataFrame | None:
    """
    Download Amazon Reviews 2023 - Software category from HuggingFace.

    Args:
        force: If True, re-download even if file exists. Default False.

    Returns:
        DataFrame with downloaded reviews, or None if skipped.
    """
    if DATA.raw_reviews.exists() and not force:
        print(f"Skipping download: {DATA.raw_reviews} already exists")
        print("  Use --force to re-download")
        return None

    print(f"Downloading dataset: {DATASET.repo_id} ({DATASET.filename})...")
    print("This may take several minutes on first run.")

    # Download JSONL file from HuggingFace Hub
    file_path = hf_hub_download(
        repo_id=DATASET.repo_id,
        filename=DATASET.filename,
        repo_type="dataset",
    )

    # Load JSONL into DataFrame
    df = pd.read_json(file_path, lines=True)
    print(f"Downloaded {len(df)} reviews.")

    # Select and rename columns
    df = df[["text", "rating", "title", "parent_asin"]].copy()
    df = df.rename(columns={"parent_asin": "product_id"})

    # Add review_id
    df["review_id"] = [f"R{i:06d}" for i in range(len(df))]

    # Reorder columns
    df = df[["review_id", "text", "rating", "title", "product_id"]]

    # Ensure data directory exists
    DATA.raw_reviews.parent.mkdir(parents=True, exist_ok=True)

    # Save raw data
    df.to_csv(DATA.raw_reviews, index=False)
    print(f"Saved to {DATA.raw_reviews}")

    return df


def main():
    """Entry point for script."""
    parser = argparse.ArgumentParser(
        description="Download Amazon Reviews dataset from HuggingFace"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    download_data(force=args.force)


if __name__ == "__main__":
    main()
