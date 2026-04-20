# scripts/prepare_all_data.py
"""Run all data preparation scripts in sequence."""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA


def main():
    """Run complete data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Run complete data preparation pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if files exist")
    args = parser.parse_args()

    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    if args.force:
        print("(--force: re-running all steps)")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/6] Downloading from HuggingFace...")
    print("-" * 40)
    from scripts.download_data import download_data

    download_data(force=args.force)

    # Step 2: Preprocess
    print("\n[2/6] Cleaning and preprocessing...")
    print("-" * 40)
    from scripts.preprocess_data import preprocess_data

    preprocess_data(force=args.force)

    # Step 3: Tokenize
    print("\n[3/6] Tokenizing sentences...")
    print("-" * 40)
    from scripts.tokenize_sentences import tokenize_sentences

    tokenize_sentences(force=args.force)

    # Step 4: Sample
    print("\n[4/6] Creating sample dataset...")
    print("-" * 40)
    from scripts.create_sample import create_sample

    create_sample(force=args.force)

    # Step 5: Label
    print("\n[5/6] Auto-labeling reviews...")
    print("-" * 40)
    from scripts.label_training_data import label_training_data

    label_training_data(force=args.force)

    # Step 6: Validate
    print("\n[6/6] Validating outputs...")
    print("-" * 40)
    from scripts.validate_data import validate_data

    success = validate_data()

    # Final status
    print("\n" + "=" * 60)
    if success:
        print("DATA PREPARATION COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print(f"  1. Open {DATA.curated_labels} in a spreadsheet")
        print("  2. Fill in the 'verified_topic' column")
        print("  3. Save the file")
        print("  4. Run model training scripts")
    else:
        print("DATA PREPARATION FAILED")
        print("=" * 60)
        print("\nReview errors above and fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
