# scripts/create_sample.py
"""Create balanced sample dataset for testing."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import DATA, SAMPLE


def create_sample(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Create stratified sample dataset.

    Args:
        input_path: Input CSV path. Defaults to main_corpus.
        output_path: Output CSV path. Defaults to sample_reviews.

    Returns:
        Sample DataFrame.
    """
    input_path = input_path or DATA.main_corpus
    output_path = output_path or DATA.sample_reviews

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Sample n reviews per rating (1-5 stars)
    samples = []
    for rating in range(1, 6):
        rating_reviews = df[df["rating"] == rating]
        n_available = len(rating_reviews)
        n_sample = min(SAMPLE.reviews_per_rating, n_available)

        if n_available < SAMPLE.reviews_per_rating:
            print(f"  Warning: Only {n_available} reviews with rating {rating}")

        sample = rating_reviews.sample(n=n_sample, random_state=SAMPLE.random_state)
        samples.append(sample)

    sample_df = pd.concat(samples, ignore_index=True)

    # Shuffle the combined sample
    sample_df = sample_df.sample(frac=1, random_state=SAMPLE.random_state).reset_index(
        drop=True
    )

    # Save sample
    sample_df.to_csv(output_path, index=False)

    print(f"Created sample with {len(sample_df)} reviews:")
    print(sample_df["rating"].value_counts().sort_index().to_string())
    print(f"Saved to {output_path}")

    return sample_df


def main():
    """Entry point for script."""
    create_sample()


if __name__ == "__main__":
    main()
