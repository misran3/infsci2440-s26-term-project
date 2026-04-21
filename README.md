# Survey Analysis Agent

An Intelligent Agent for Probabilistic and Context-Aware Analysis of Free-Text Survey Responses.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## File Structure

```
.
├── app.py                          # Streamlit web interface
├── src/
│   ├── loaders/                    # Data loading utilities
│   │   ├── loader.py
│   │   └── structures.py
│   ├── utils/                      # Shared utilities
│   │   ├── logger.py
│   │   └── errors.py
│   ├── search/                     # Query expansion and retrieval
│   │   ├── beam_search.py          # WordNet synonym expansion via beam search
│   │   ├── term_filter.py          # LLM-based filtering of irrelevant terms
│   │   └── tfidf_retriever.py
│   ├── classification/             # Topic classification
│   │   └── naive_bayes.py
│   ├── reasoning/                  # Probabilistic reasoning and LLM
│   │   ├── bayesian_network.py
│   │   ├── hmm_sentiment.py
│   │   └── llm_summarizer.py
│   ├── pipeline.py                 # Main pipeline orchestrator
│   └── mocks.py                    # Mock implementations for testing
├── scripts/                        # Data preparation scripts
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── tokenize_sentences.py
│   ├── create_sample.py
│   ├── label_training_data.py
│   └── train_models.py
├── tests/                          # Test suite
├── models/                         # Trained model files (.pkl)
└── data/                           # Dataset files (.csv)
```

## Data Preparation

Download and prepare the Amazon Reviews dataset:

```bash
# Run full pipeline (skips already-completed steps)
uv run prepare-data

# Force re-run all steps
uv run prepare-data --force
```

Individual scripts:
```bash
uv run download-data      # Download from HuggingFace (~5 min)
uv run preprocess-data    # Clean and filter
uv run tokenize-sentences # Split into sentences
uv run create-sample      # Create balanced 100-review sample
uv run label-data         # Auto-label topics
uv run validate-data      # Verify outputs
```

Output files in `data/`:
- `amazon_reviews_software_raw.csv` - Raw download
- `amazon_reviews_software_clean.csv` - Cleaned
- `amazon_reviews_software.csv` - Main corpus with sentences
- `sample_reviews.csv` - 100 balanced samples
- `labeled_reviews.csv` - Auto-labeled corpus
- `curated_labels.csv` - For human curation

## Label Curation

After running `label-data`, review and verify the auto-labeled topics:

```bash
# View a batch of reviews (start_index, batch_size)
python3 scripts/review_batch.py 0 30

# After editing batch files in data/review_batches/, merge them:
python3 scripts/merge_review_batches.py
```

Batch files are stored in `data/review_batches/` (e.g., `batch_0_29.csv`) with format:
```csv
index,verified_topic
0,performance
1,usability
...
```

Valid topics: `performance`, `usability`, `features`, `pricing`, `support`, `compatibility`, `other`

## Query Expansion

The search module expands user queries using WordNet synonyms and filters out irrelevant terms:

```python
from src.search import BeamSearchExpander, TermFilter

expander = BeamSearchExpander(beam_width=3, max_depth=2)
term_filter = TermFilter()

# Expand "bug" using WordNet synonyms
result = expander.expand("bug")
# ['tease', 'badger', 'microbe', 'germ', 'bug', 'pester', ...]

# Filter to software-relevant terms using Claude Haiku
filtered = await term_filter.filter("bug", result.expanded_terms)
# ['bug']  -- irrelevant terms like 'badger', 'germ' removed
```

### How Term Filtering Works

1. **BeamSearchExpander** finds synonyms via WordNet, but many are irrelevant (e.g., "bug" → "badger", "germ")
2. **TermFilter** uses Claude Haiku to identify software-relevant terms
3. Results are cached in `data/term_filter_cache.json` to avoid repeated LLM calls

Cache format (`term -> is_relevant`):
```json
{
  "glitch": true,
  "microbe": false,
  "crash": true,
  "badger": false
}
```

Cached lookups are ~1.6µs vs ~1-2s for LLM calls.

## Topic Classification

The classification module uses Naive Bayes to categorize reviews into topics:

### Training the Classifier

```bash
# Train on sample dataset (5K reviews, fast)
uv run python scripts/train_classifier.py --dataset sample

# Train on full dataset (all labeled reviews)
uv run python scripts/train_classifier.py --dataset full

# Train and evaluate against gold set
uv run python scripts/train_classifier.py --dataset sample --evaluate
```

The trained model is saved to `models/naive_bayes.pkl`.

### Using the Classifier

```python
from src.classification.naive_bayes import TopicClassifier
from src.loaders.structures import Review

# Load trained model
classifier = TopicClassifier.load("models/naive_bayes.pkl")

# Classify reviews
reviews = [Review("1", "The app crashes constantly", 1, "Buggy", "P1")]
results = classifier.predict(reviews)

print(results[0].predicted_topic)  # Topic.PERFORMANCE
print(results[0].confidence)       # 0.85

# Filter reviews by topic
filter_result = classifier.filter_by_topic(reviews, "performance", min_confidence=0.5)
print(filter_result.filtered_reviews)
print(filter_result.topic_distribution)
```

### Available Topics

- `performance` - crashes, bugs, speed, memory issues
- `usability` - UI, ease of use, interface design
- `features` - functionality, missing features, capabilities
- `pricing` - cost, value, subscription
- `support` - customer service, documentation
- `compatibility` - installation, OS support, versions
- `other` - reviews that don't fit other categories

## Running the UI

```bash
# Export AWS credentials for LLM-based term filtering (optional but recommended)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Start the Streamlit web interface
uv run streamlit run app.py
```

If AWS credentials are not configured, the UI will skip LLM term filtering and use all BeamSearch-expanded terms directly.

The UI displays all pipeline stages:
1. Query expansion (BeamSearch + LLM filtering)
2. TF-IDF retrieval
3. Topic classification (Naive Bayes)
4. Probabilistic insights (Bayesian Network)
5. Sentiment sequences (HMM)
6. Summary (LLM)
