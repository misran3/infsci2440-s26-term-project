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
