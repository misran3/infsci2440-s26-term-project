"""Pre-tokenize reviews into sentences using NLTK."""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
import pandas as pd

from src.config import DATA


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt_tab", quiet=True)


def tokenize_sentences(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Pre-tokenize reviews into sentences.

    Args:
        input_path: Input CSV path. Defaults to clean_reviews.
        output_path: Output CSV path. Defaults to main_corpus.

    Returns:
        DataFrame with sentences column added.
    """
    input_path = input_path or DATA.clean_reviews
    output_path = output_path or DATA.main_corpus

    ensure_nltk_data()
    from nltk.tokenize import sent_tokenize

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Tokenizing {len(df)} reviews...")

    # Tokenize each review into sentences and store as JSON
    df["sentences"] = df["text"].apply(lambda x: json.dumps(sent_tokenize(str(x))))

    # Add sentence count for analysis
    df["sentence_count"] = df["sentences"].apply(lambda x: len(json.loads(x)))

    # Calculate statistics
    avg_sentences = df["sentence_count"].mean()
    print(f"  Average sentences per review: {avg_sentences:.1f}")

    # Save tokenized data
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


def main():
    """Entry point for script."""
    tokenize_sentences()


if __name__ == "__main__":
    main()
