"""Auto-label reviews with topics and export high-confidence samples for curation."""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import DATA, LABELING


def auto_label_review(text: str) -> tuple[str, int, list[str]]:
    """
    Assign topic based on keyword matching.

    Args:
        text: Review text.

    Returns:
        Tuple of (topic, confidence_score, matched_keywords).
    """
    text_lower = text.lower()

    scores = {}
    matched = {}

    for topic, keywords in LABELING.topic_keywords.items():
        matches = [kw for kw in keywords if kw in text_lower]
        scores[topic] = len(matches)
        matched[topic] = matches

    best_topic = max(scores, key=scores.get)
    confidence = scores[best_topic]
    keywords_matched = matched.get(best_topic, [])

    if confidence == 0:
        return "other", 0, []

    return best_topic, confidence, keywords_matched


def label_training_data(
    input_path: Path | None = None,
    output_path: Path | None = None,
    curated_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Auto-label reviews and export high-confidence samples.

    Args:
        input_path: Input CSV path. Defaults to main_corpus.
        output_path: Output CSV path. Defaults to labeled_reviews.
        curated_path: Curated export path. Defaults to curated_labels.

    Returns:
        Tuple of (labeled_df, curated_df).
    """
    input_path = input_path or DATA.main_corpus
    output_path = output_path or DATA.labeled_reviews
    curated_path = curated_path or DATA.curated_labels

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Auto-labeling {len(df)} reviews...")

    # Apply auto-labeling
    results = df["text"].apply(auto_label_review)
    df["topic_label"] = results.apply(lambda x: x[0])
    df["confidence_score"] = results.apply(lambda x: x[1])
    df["matched_keywords"] = results.apply(lambda x: json.dumps(x[2]))

    # Print distribution
    print("\nTopic distribution:")
    distribution = df["topic_label"].value_counts()
    for topic, count in distribution.items():
        pct = (count / len(df)) * 100
        print(f"  {topic}: {count} ({pct:.1f}%)")

    # Save labeled data
    df.to_csv(output_path, index=False)
    print(f"\nSaved labeled reviews to {output_path}")

    # Export high-confidence samples for curation
    print(f"\nExporting high-confidence samples for curation...")

    # Filter high-confidence (2+ keyword matches), exclude "other"
    high_conf = df[
        (df["confidence_score"] >= LABELING.min_confidence_for_curation)
        & (df["topic_label"] != "other")
    ]

    # Stratify by topic (approximately equal per topic)
    topics = [t for t in LABELING.topics if t != "other"]
    samples_per_topic = LABELING.curated_sample_size // len(topics)

    curated_samples = []
    for topic in topics:
        topic_reviews = high_conf[high_conf["topic_label"] == topic]
        n_sample = min(samples_per_topic, len(topic_reviews))
        if n_sample > 0:
            sample = topic_reviews.sample(n=n_sample, random_state=42)
            curated_samples.append(sample)
            print(f"  {topic}: {n_sample} samples")

    if not curated_samples:
        print("  Warning: No high-confidence samples found!")
        curated_df = pd.DataFrame()
    else:
        curated_df = pd.concat(curated_samples, ignore_index=True)

        # Prepare curation columns
        curated_df = curated_df.rename(columns={"topic_label": "auto_topic"})
        curated_df["verified_topic"] = ""  # Human fills this
        curated_df["notes"] = ""  # Optional human notes

        # Select columns for curation file
        curated_df = curated_df[
            [
                "review_id",
                "text",
                "rating",
                "title",
                "product_id",
                "auto_topic",
                "confidence_score",
                "matched_keywords",
                "verified_topic",
                "notes",
            ]
        ]

        curated_df.to_csv(curated_path, index=False)
        print(f"\nExported {len(curated_df)} samples to {curated_path}")
        print("Next: Open in spreadsheet and fill 'verified_topic' column")

    return df, curated_df


def main():
    """Entry point for script."""
    label_training_data()


if __name__ == "__main__":
    main()
