"""Merge review batch files into curated_labels.csv."""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def merge_batches(
    batch_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Merge all batch_*.csv files into curated_labels.csv.

    Args:
        batch_dir: Directory containing batch files. Defaults to data/review_batches.
        output_path: Output CSV path. Defaults to data/curated_labels.csv.

    Returns:
        Updated DataFrame.
    """
    project_root = Path(__file__).parent.parent
    batch_dir = batch_dir or project_root / "data" / "review_batches"
    output_path = output_path or project_root / "data" / "curated_labels.csv"

    # Read the main CSV
    main_df = pd.read_csv(output_path)
    print(f"Main CSV: {len(main_df)} rows")

    # Read and merge all batch files
    all_updates = {}
    batch_files = sorted(batch_dir.glob("batch_*.csv"))

    if not batch_files:
        print(f"No batch files found in {batch_dir}")
        return main_df

    for batch_file in batch_files:
        batch_df = pd.read_csv(batch_file)
        for _, row in batch_df.iterrows():
            all_updates[int(row["index"])] = row["verified_topic"]
        print(f"Loaded {batch_file.name}: {len(batch_df)} rows")

    print(f"\nTotal updates: {len(all_updates)}")

    # Apply updates
    for idx, topic in all_updates.items():
        main_df.at[idx, "verified_topic"] = topic

    # Verify
    reviewed = main_df["verified_topic"].notna() & (main_df["verified_topic"] != "")
    print(f"Rows with verified_topic: {reviewed.sum()}")

    # Save
    main_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return main_df


def main():
    """Entry point for script."""
    merge_batches()


if __name__ == "__main__":
    main()
