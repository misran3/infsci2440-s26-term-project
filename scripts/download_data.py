"""Download Amazon Reviews dataset from HuggingFace."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
import pandas as pd

from src.config import DATA, DATASET


def download_data() -> pd.DataFrame:
    """
    Download Amazon Reviews 2023 - Software category from HuggingFace.

    Returns:
        DataFrame with downloaded reviews.
    """
    print(f"Downloading dataset: {DATASET.name} ({DATASET.subset})...")
    print("This may take several minutes on first run.")

    # Load dataset from HuggingFace
    dataset = load_dataset(
        DATASET.name,
        DATASET.subset,
        split=DATASET.split,
        trust_remote_code=DATASET.trust_remote_code,
    )

    # Convert to DataFrame
    df = dataset.to_pandas()
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
    download_data()


if __name__ == "__main__":
    main()
