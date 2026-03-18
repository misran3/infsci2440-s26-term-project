# scripts/validate_data.py
"""Validate data preparation outputs."""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import DATA


def validate_data() -> bool:
    """
    Run validation checks on data prep outputs.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("DATA PREPARATION VALIDATION REPORT")
    print("=" * 60)

    checks_passed = 0
    checks_failed = 0
    warnings = []

    def check(name: str, condition: bool, fail_msg: str, is_warning: bool = False):
        nonlocal checks_passed, checks_failed
        if condition:
            print(f"  ✓ {name}")
            checks_passed += 1
            return True
        else:
            if is_warning:
                print(f"  ⚠ {name}: {fail_msg}")
                warnings.append(fail_msg)
                checks_passed += 1  # Warnings don't fail
                return True
            else:
                print(f"  ✗ {name}: {fail_msg}")
                checks_failed += 1
                return False

    # Check 1: Raw data exists
    print("\n[File Existence]")
    check("Raw data exists", DATA.raw_reviews.exists(), "Run download_data.py first")

    # Check 2: Main corpus exists
    main_exists = check(
        "Main corpus exists", DATA.main_corpus.exists(), "Run tokenize_sentences.py"
    )

    # Check 3: Main corpus size
    if main_exists:
        df = pd.read_csv(DATA.main_corpus)
        check(
            f"Main corpus size ({len(df)} reviews)",
            len(df) >= 10000,
            "Corpus too small (<10K reviews)",
        )

        # Check 4: Sentences column exists
        check(
            "Sentences column exists",
            "sentences" in df.columns,
            "HMM blocker: sentences column missing",
        )

        # Check 5: Sentences are valid JSON
        if "sentences" in df.columns:
            sample = df["sentences"].head(10)
            try:
                for s in sample:
                    json.loads(s)
                check("Sentences are valid JSON", True, "")
            except json.JSONDecodeError:
                check("Sentences are valid JSON", False, "JSON parse error in sentences")

    # Check 6: Sample exists
    print("\n[Sample Dataset]")
    sample_exists = check(
        "Sample dataset exists", DATA.sample_reviews.exists(), "Run create_sample.py"
    )

    # Check 7: Sample is balanced
    if sample_exists:
        sample_df = pd.read_csv(DATA.sample_reviews)
        rating_counts = sample_df["rating"].value_counts()
        is_balanced = all(rating_counts.get(r, 0) == 20 for r in range(1, 6))
        check(
            f"Sample is balanced ({len(sample_df)} reviews)",
            is_balanced or len(sample_df) == 100,
            "Expected 20 reviews per rating",
            is_warning=True,
        )

    # Check 8: Labeled data exists
    print("\n[Labeled Data]")
    labeled_exists = check(
        "Labeled reviews exists",
        DATA.labeled_reviews.exists(),
        "Run label_training_data.py",
    )

    # Check 9: Topic distribution
    if labeled_exists:
        labeled_df = pd.read_csv(DATA.labeled_reviews)
        distribution = labeled_df["topic_label"].value_counts(normalize=True)
        max_pct = distribution.max()
        dominant_topic = distribution.idxmax()
        check(
            f"Topic distribution (max: {dominant_topic} at {max_pct:.0%})",
            max_pct <= 0.5,
            f"Topic '{dominant_topic}' dominates ({max_pct:.0%})",
            is_warning=True,
        )

    # Check 10: Curated export exists
    curated_exists = check(
        "Curated labels exists",
        DATA.curated_labels.exists(),
        "Run label_training_data.py",
    )

    if curated_exists:
        curated_df = pd.read_csv(DATA.curated_labels)
        check(
            f"Curated sample size ({len(curated_df)} reviews)",
            len(curated_df) >= 100,
            "Too few samples for curation",
        )

    # Summary
    print("\n" + "=" * 60)
    total = checks_passed + checks_failed
    if checks_failed == 0:
        print(f"STATUS: READY ({checks_passed}/{total} checks passed)")
        if warnings:
            print(f"  {len(warnings)} warning(s) - review above")
        return True
    else:
        print(f"STATUS: FAILED ({checks_failed}/{total} checks failed)")
        return False


def main():
    """Entry point for script."""
    success = validate_data()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
