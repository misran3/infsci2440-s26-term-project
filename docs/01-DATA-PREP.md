# Data Preparation

This document covers data acquisition, preprocessing, and labeling required before any component can be built. **This is a blocker for all team members.**

---

## Dataset Overview

**Amazon Reviews 2023 - Software Category**
- Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Subset: Software product reviews only
- Estimated size: ~50,000-100,000 reviews
- Fields used: `text`, `rating`, `title`, `parent_asin`

---

## Data Pipeline

```
HUGGINGFACE DATASET
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DOWNLOAD                                                     │
│     scripts/download_data.py                                     │
│     Output: data/amazon_reviews_software_raw.csv                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CLEAN & PREPROCESS                                           │
│     - Remove duplicates                                          │
│     - Remove empty/null text                                     │
│     - Normalize whitespace                                       │
│     - Filter to English (optional)                               │
│     Output: data/amazon_reviews_software.csv                     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PRE-TOKENIZE SENTENCES                                       │
│     - Split each review into sentences (NLTK punkt)              │
│     - Store as JSON array in 'sentences' column                  │
│     Output: data/amazon_reviews_software.csv (updated)           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. CREATE SAMPLE DATASET                                        │
│     - Random sample of ~100 reviews for testing                  │
│     - Balanced across ratings (1-5 stars)                        │
│     Output: data/sample_reviews.csv                              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. LABEL TRAINING DATA (for Naive Bayes)                        │
│     - Auto-label using keyword rules                             │
│     - Manual verification on 10% sample                          │
│     Output: data/labeled_reviews.csv                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Download Dataset

```python
# scripts/download_data.py

from datasets import load_dataset
import pandas as pd

def download_amazon_reviews():
    """Download Amazon Reviews 2023 - Software category."""
    print("Downloading dataset from HuggingFace...")

    # Load Software category subset
    # NOTE: trust_remote_code=True executes code from the HuggingFace repo.
    # This is generally safe for well-known datasets but review the repo if concerned.
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Software",
        split="full",
        trust_remote_code=True
    )

    # Convert to DataFrame
    df = dataset.to_pandas()

    # Select relevant columns
    df = df[['text', 'rating', 'title', 'parent_asin']].copy()
    df = df.rename(columns={'parent_asin': 'product_id'})

    # Add review_id
    df['review_id'] = [f"R{i:06d}" for i in range(len(df))]

    # Reorder columns
    df = df[['review_id', 'text', 'rating', 'title', 'product_id']]

    # Save raw data
    output_path = 'data/amazon_reviews_software_raw.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} reviews to {output_path}")

    return df

if __name__ == "__main__":
    download_amazon_reviews()
```

### Running Download

```bash
uv run python scripts/download_data.py
```

**Expected output:** `data/amazon_reviews_software_raw.csv` (~50-100MB)

---

## Step 2: Clean & Preprocess

```python
# scripts/preprocess_data.py

import pandas as pd
import re

def clean_reviews(input_path: str, output_path: str):
    """Clean and preprocess reviews."""
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    original_count = len(df)

    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])

    # Remove empty/null text
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']

    # Normalize whitespace
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()))

    # Remove very short reviews (< 10 chars)
    df = df[df['text'].str.len() >= 10]

    # Remove very long reviews (> 5000 chars) - likely spam
    df = df[df['text'].str.len() <= 5000]

    # Ensure rating is 1-5
    df = df[df['rating'].between(1, 5)]

    cleaned_count = len(df)
    print(f"Cleaned: {original_count} → {cleaned_count} reviews ({original_count - cleaned_count} removed)")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df

if __name__ == "__main__":
    clean_reviews(
        'data/amazon_reviews_software_raw.csv',
        'data/amazon_reviews_software_clean.csv'
    )
```

---

## Step 3: Pre-tokenize Sentences

Pre-tokenizing sentences during data prep avoids the ~50ms NLTK cold-start per review during HMM analysis.

```python
# scripts/tokenize_sentences.py

import pandas as pd
import nltk
import json

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def add_sentence_tokenization(input_path: str, output_path: str):
    """Pre-tokenize reviews into sentences."""
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print("Tokenizing sentences...")
    df['sentences'] = df['text'].apply(lambda x: json.dumps(sent_tokenize(str(x))))

    # Also add sentence count for analysis
    df['sentence_count'] = df['sentences'].apply(lambda x: len(json.loads(x)))

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Stats
    avg_sentences = df['sentence_count'].mean()
    print(f"Average sentences per review: {avg_sentences:.1f}")

    return df

if __name__ == "__main__":
    add_sentence_tokenization(
        'data/amazon_reviews_software_clean.csv',
        'data/amazon_reviews_software.csv'
    )
```

---

## Step 4: Create Sample Dataset

A small sample dataset enables fast iteration during development.

```python
# scripts/create_sample.py

import pandas as pd

def create_sample_dataset(input_path: str, output_path: str, n_per_rating: int = 20):
    """Create balanced sample dataset for testing."""
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Sample n reviews per rating (1-5 stars)
    samples = []
    for rating in range(1, 6):
        rating_reviews = df[df['rating'] == rating]
        sample = rating_reviews.sample(n=min(n_per_rating, len(rating_reviews)), random_state=42)
        samples.append(sample)

    sample_df = pd.concat(samples, ignore_index=True)
    sample_df.to_csv(output_path, index=False)

    print(f"Created sample with {len(sample_df)} reviews:")
    print(sample_df['rating'].value_counts().sort_index())

    return sample_df

if __name__ == "__main__":
    create_sample_dataset(
        'data/amazon_reviews_software.csv',
        'data/sample_reviews.csv',
        n_per_rating=20  # 100 total reviews
    )
```

---

## Step 5: Label Training Data (for Naive Bayes)

The Naive Bayes classifier needs labeled training data. We use a hybrid approach:
1. **Auto-label** using keyword rules (~80% accuracy)
2. **Verify** a random 10% sample manually

### Topic Definitions

```python
TOPICS = {
    "performance": {
        "keywords": ["slow", "fast", "crash", "bug", "freeze", "lag", "speed",
                     "memory", "cpu", "hang", "stuck", "error", "glitch"],
        "description": "Speed, crashes, bugs, stability, errors"
    },
    "usability": {
        "keywords": ["easy", "difficult", "confusing", "intuitive", "ui",
                     "interface", "learn", "simple", "complicated", "user-friendly"],
        "description": "UI, ease of use, learning curve"
    },
    "features": {
        "keywords": ["feature", "function", "capability", "missing", "wish",
                     "need", "want", "add", "include", "option", "setting"],
        "description": "Functionality, capabilities, missing features"
    },
    "pricing": {
        "keywords": ["price", "cost", "expensive", "cheap", "value",
                     "subscription", "free", "pay", "money", "worth", "affordable"],
        "description": "Cost, value, subscription model"
    },
    "support": {
        "keywords": ["support", "help", "customer service", "response",
                     "documentation", "update", "team", "email", "contact"],
        "description": "Customer service, documentation, updates"
    },
    "compatibility": {
        "keywords": ["install", "compatible", "windows", "mac", "linux",
                     "version", "os", "update", "upgrade", "work with"],
        "description": "Installation, OS support, integrations"
    }
}
```

### Auto-labeling Script

```python
# scripts/label_training_data.py

import pandas as pd
import re

TOPICS = {
    "performance": ["slow", "fast", "crash", "bug", "freeze", "lag", "speed",
                    "memory", "cpu", "hang", "stuck", "error", "glitch"],
    "usability": ["easy", "difficult", "confusing", "intuitive", "ui",
                  "interface", "learn", "simple", "complicated"],
    "features": ["feature", "function", "capability", "missing", "wish",
                 "need", "want", "add", "include", "option"],
    "pricing": ["price", "cost", "expensive", "cheap", "value",
                "subscription", "free", "pay", "money", "worth"],
    "support": ["support", "help", "customer service", "response",
                "documentation", "update", "team"],
    "compatibility": ["install", "compatible", "windows", "mac", "linux",
                      "version", "os", "upgrade"],
}

def auto_label_review(text: str) -> str:
    """Assign topic based on keyword matching."""
    text_lower = text.lower()

    scores = {}
    for topic, keywords in TOPICS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return "other"
    return best_topic

def label_dataset(input_path: str, output_path: str):
    """Auto-label reviews with topics."""
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    print("Auto-labeling reviews...")
    df['topic_label'] = df['text'].apply(auto_label_review)

    # Print distribution
    print("\nTopic distribution:")
    print(df['topic_label'].value_counts())

    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Save verification sample
    verification_sample = df.sample(n=min(100, len(df)), random_state=42)
    verification_sample.to_csv('data/verification_sample.csv', index=False)
    print("Saved 100-review verification sample to data/verification_sample.csv")

    return df

if __name__ == "__main__":
    label_dataset(
        'data/amazon_reviews_software.csv',
        'data/labeled_reviews.csv'
    )
```

### Manual Verification

**Owner:** Person 2 (Naive Bayes owner)

After auto-labeling, manually verify `data/verification_sample.csv`:

1. Open the CSV in a spreadsheet
2. For each review, check if `topic_label` is correct
3. Add a `verified` column: `correct` / `wrong` / `ambiguous`
4. Calculate accuracy: `correct / (correct + wrong)`

**Target:** >70% accuracy on auto-labels is acceptable for MVP.

**If accuracy < 70%:** Refine keyword rules in `label_training_data.py` and re-run.

---

## Data Files Summary

| File | Description | Size | Who Needs It |
|------|-------------|------|--------------|
| `amazon_reviews_software_raw.csv` | Raw download | ~50-100MB | Nobody (intermediate) |
| `amazon_reviews_software.csv` | Cleaned + tokenized | ~40-80MB | Everyone |
| `sample_reviews.csv` | 100 review sample | ~50KB | Everyone (testing) |
| `labeled_reviews.csv` | With topic labels | ~40-80MB | Person 2 (NB training) |
| `verification_sample.csv` | For manual verification | ~50KB | Team lead |

---

## Data Loader Implementation

```python
# src/data/loader.py

import pandas as pd
import json
from pathlib import Path
from typing import List, Optional
from src.data.structures import Review

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_reviews(
    path: Optional[str] = None,
    sample: bool = False,
    limit: Optional[int] = None
) -> List[Review]:
    """
    Load reviews from CSV.

    Args:
        path: Path to CSV file. Defaults to main dataset.
        sample: If True, load sample_reviews.csv instead.
        limit: Maximum number of reviews to load.

    Returns:
        List of Review objects.
    """
    if path is None:
        if sample:
            path = DATA_DIR / "sample_reviews.csv"
        else:
            path = DATA_DIR / "amazon_reviews_software.csv"

    df = pd.read_csv(path)

    if limit:
        df = df.head(limit)

    reviews = []
    for _, row in df.iterrows():
        # Parse pre-tokenized sentences if available
        sentences = None
        if 'sentences' in row and pd.notna(row['sentences']):
            try:
                sentences = json.loads(row['sentences'])
            except json.JSONDecodeError:
                pass

        review = Review(
            review_id=str(row['review_id']),
            text=str(row['text']),
            rating=int(row['rating']),
            title=str(row.get('title', '')),
            product_id=str(row.get('product_id', '')),
            sentences=sentences
        )
        reviews.append(review)

    return reviews

def load_labeled_reviews() -> tuple[List[Review], List[str]]:
    """Load reviews with topic labels for Naive Bayes training."""
    path = DATA_DIR / "labeled_reviews.csv"
    df = pd.read_csv(path)

    reviews = load_reviews(path=str(path))
    labels = df['topic_label'].tolist()

    return reviews, labels
```

---

## Running Full Data Pipeline

```bash
# 1. Download (requires internet)
uv run python scripts/download_data.py

# 2. Clean
uv run python scripts/preprocess_data.py

# 3. Tokenize sentences
uv run python scripts/tokenize_sentences.py

# 4. Create sample
uv run python scripts/create_sample.py

# 5. Label for Naive Bayes
uv run python scripts/label_training_data.py

# Or run all at once:
uv run python scripts/prepare_all_data.py
```

---

## Blockers & Dependencies

| Component | Blocked Until | Required Data |
|-----------|---------------|---------------|
| TF-IDF Retriever | Step 3 complete | `amazon_reviews_software.csv` |
| Naive Bayes | Step 5 complete | `labeled_reviews.csv` |
| Bayesian Network | Step 3 complete | `amazon_reviews_software.csv` (with topics from NB) |
| HMM | Step 3 complete | `amazon_reviews_software.csv` (pre-tokenized) |
| All unit tests | Step 4 complete | `sample_reviews.csv` |

---

## Validation Checklist

Before proceeding to component development:

- [ ] `amazon_reviews_software.csv` exists and has >10,000 reviews
- [ ] `sample_reviews.csv` has exactly 100 reviews (20 per rating)
- [ ] `labeled_reviews.csv` has topic_label column
- [ ] Verification sample shows >70% labeling accuracy
- [ ] All CSVs load without errors using `load_reviews()`
- [ ] Pre-tokenized sentences parse correctly as JSON

---

## Future Enhancements (Optional)

- **Explicit train/test split:** If classifier results are inconsistent, create `labeled_train.csv` and `labeled_test.csv` with fixed split for reproducibility.
- **Consolidated script:** Create `prepare_all_data.py` to run all 5 scripts in sequence.
- **Data augmentation:** For underrepresented topics, consider synonym replacement or back-translation.
