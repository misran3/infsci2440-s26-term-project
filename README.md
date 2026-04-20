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
│   │   ├── beam_search.py
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
