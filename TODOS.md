# Project TODOs

Comprehensive task list for Survey Analysis Agent project.

---

## Infrastructure & Setup

- [ ] Clone repository and set up project structure (see 00-PROJECT-OVERVIEW.md)
- [ ] Install uv package manager
- [ ] Run `uv sync` to install dependencies
- [ ] Download NLTK data (wordnet, punkt, vader_lexicon)
- [ ] Set up .env with API keys (ANTHROPIC_API_KEY for Bedrock)
- [ ] Create data/ directory structure

---

## Data Preparation (BLOCKER - complete before component work)

- [ ] Implement scripts/download_data.py
- [ ] Run download_data.py to get amazon_reviews_software_raw.csv
- [ ] Implement scripts/preprocess_data.py (clean, normalize, filter)
- [ ] Run preprocess_data.py to get amazon_reviews_software_clean.csv
- [ ] Implement scripts/tokenize_sentences.py (pre-tokenize with NLTK)
- [ ] Run tokenize_sentences.py to get amazon_reviews_software.csv
- [ ] Implement scripts/create_sample.py (100 balanced reviews)
- [ ] Run create_sample.py to get sample_reviews.csv
- [ ] Implement scripts/label_training_data.py (auto-label topics)
- [ ] Run label_training_data.py to get labeled_reviews.csv
- [ ] Person 2: Verify 100-review sample, target >70% accuracy
- [ ] Implement src/loaders/structures.py with all dataclasses
- [ ] Implement src/loaders/loader.py with load_reviews() function
- [ ] Verify all CSVs load without errors

---

## Person 1: Search & Expansion

- [ ] Implement BeamSearchExpander class in search/beam_search.py
- [ ] Implement get_synonyms() using WordNet
- [ ] Implement score_expansion() scoring function
- [ ] Implement expand() with beam search algorithm
- [ ] Write test_expand_returns_original_plus_synonyms
- [ ] Write test_expand_empty_query_returns_empty
- [ ] Implement TFIDFRetriever class in search/tfidf_retriever.py
- [ ] Implement fit() to build TF-IDF matrix
- [ ] Implement retrieve() with cosine similarity
- [ ] Implement get_matching_terms() for UI transparency
- [ ] Write test_retrieve_returns_relevant_reviews
- [ ] Write test_retrieve_empty_corpus
- [ ] Tune beam_width and max_depth parameters
- [ ] Tune TF-IDF max_features and ngram_range

---

## Person 2: Classification & UI

- [ ] Create topic keyword lists for auto-labeling training data
- [ ] Label training data (semi-supervised or manual, 500-1000 reviews)
- [ ] Implement TopicClassifier class in classification/naive_bayes.py
- [ ] Implement fit() with train/test split
- [ ] Implement predict() returning TopicClassification
- [ ] Implement filter_by_topic() with confidence threshold
- [ ] Implement detect_topic_from_query() for query analysis
- [ ] Add fallback: return all candidates if NB filters to 0 results
- [ ] Write test_classifier_predicts_correct_topic
- [ ] Write test_classifier_handles_empty_input
- [ ] Write test_filter_by_topic_no_matches_returns_all (NEW)
- [ ] Create Streamlit app skeleton in app.py (at root)
- [ ] Implement query input with search button
- [ ] Implement optional filters (topic, rating, sentiment)
- [ ] Display Query Expansion section (beam search visualization)
- [ ] Display TF-IDF Retrieval section
- [ ] Display Topic Classification section
- [ ] Display Bayesian Network insights section
- [ ] Display HMM Sentiment section
- [ ] Display LLM Summary section
- [ ] Add error handling for graceful degradation in UI

---

## Person 3: Probabilistic & LLM

- [ ] Implement ReviewBayesianNetwork class in reasoning/bayesian_network.py
- [ ] Define network structure (topic → sentiment → rating)
- [ ] Implement _prepare_data() to convert reviews to DataFrame
- [ ] Implement _infer_sentiment() (rating-based heuristic)
- [ ] Implement fit() with Maximum Likelihood Estimation
- [ ] Implement query() returning BayesianInsights
- [ ] Write test_bayesian_query_returns_probabilities
- [ ] Write test_bayesian_unknown_topic_uses_default
- [ ] Implement SentimentHMM class in reasoning/hmm_sentiment.py
- [ ] Implement _sentence_to_sentiment() using VADER
- [ ] Implement _review_to_sequence() converting review to observation sequence
- [ ] Implement fit() training HMM on sequences
- [ ] Implement analyze() using Viterbi decoding
- [ ] Implement get_transition_matrix()
- [ ] Implement get_common_patterns()
- [ ] Write test_hmm_analyze_multi_sentence
- [ ] Write test_hmm_single_sentence_review
- [ ] Implement LLMSummarizer class in reasoning/llm_summarizer.py
- [ ] Update model default to Claude 4.5 Haiku via Bedrock
- [ ] Implement _get_system_prompt()
- [ ] Implement summarize() with LLM call
- [ ] Implement _build_context() formatting pipeline results
- [ ] Implement _format_reviews() for LLM context
- [ ] Implement _generate_fallback_summary() for API failures
- [ ] Write test_summarize_with_mock
- [ ] Write test_summarize_api_failure_returns_fallback (NEW)

---

## Pipeline & Integration

- [ ] Implement SurveyAnalysisPipeline class in pipeline.py
- [ ] Implement run() orchestrating all components
- [ ] Add pre-filter support (topic, rating, sentiment)
- [ ] Handle empty query with early return
- [ ] Handle no results with graceful degradation
- [ ] Write test_pipeline_happy_path
- [ ] Write test_pipeline_empty_query
- [ ] Write test_pipeline_no_results
- [ ] Implement mocks.py with all mock implementations
- [ ] Integration testing: replace mocks with real components
- [ ] Debug interface issues between components

---

## Model Training (MUST FOLLOW SEQUENCE)

**Step 1: Data Prep** (see Data Preparation section above)

**Step 2: Parallel Training**
- [ ] Implement scripts/train_models.py with save/load functions
- [ ] Train TF-IDF on amazon_reviews_software.csv → models/tfidf_vectorizer.pkl
- [ ] Train Naive Bayes on labeled_reviews.csv → models/naive_bayes.pkl
- [ ] Train HMM on amazon_reviews_software.csv → models/hmm_model.pkl

**Step 3: Sequential Training (after Naive Bayes)**
- [ ] Run Naive Bayes predict() on full corpus to assign topics
- [ ] Train Bayesian Network on topic-classified reviews → models/bayesian_network.pkl

**Retraining Triggers:**
- [ ] TF-IDF: Retrain if corpus changes >10%
- [ ] Naive Bayes: Retrain if labeled data changes or accuracy <70%
- [ ] HMM: Retrain if corpus or tokenization changes
- [ ] Bayesian Net: Retrain if NB model changes

---

## Error Handling

- [ ] Implement UserError and ComponentError in utils/errors.py
- [ ] Implement setup_logger() in utils/logger.py
- [ ] Add Beam Search: auto-download WordNet on LookupError
- [ ] Add Beam Search: return original query on empty input
- [ ] Add TF-IDF: handle MemoryError with UserError
- [ ] Add TF-IDF: auto-fit if not fitted
- [ ] Add Naive Bayes: skip classification if no training data
- [ ] Add Bayesian Net: return default probs on pgmpy errors
- [ ] Add HMM: skip analysis if no valid sequences
- [ ] Add LLM: retry 3x with backoff on rate limit
- [ ] Add LLM: use fallback summary on all API failures

---

## Documentation Updates (from review)

- [ ] Update 01-SHARED-INTERFACES: Add tfidf_score field to Review dataclass
- [ ] Update 01-SHARED-INTERFACES: Update LLMSummarizer default model to Bedrock
- [ ] Update 04-PERSON3: Fix _generate_fallback_summary indentation
- [ ] Update 04-PERSON3: Update LLMSummarizer default model to Bedrock
- [ ] Update 05-ERROR-HANDLING: Add zero-result fallback for Naive Bayes
- [ ] Update 06-TEST-SPECIFICATION: Add test_filter_by_topic_no_matches_returns_all
- [ ] Update 06-TEST-SPECIFICATION: Add test_summarize_api_failure_returns_fallback
- [ ] Create 01-DATA-PREP.md with data preparation guidance

---

## Testing

- [ ] Set up tests/conftest.py with fixtures
- [ ] Run all tests: uv run pytest tests/ -v
- [ ] Achieve passing tests for all components
- [ ] Run coverage report: uv run pytest tests/ --cov=src

---

## Deployment

- [ ] Generate requirements.txt from pyproject.toml
- [ ] Create .streamlit/config.toml
- [ ] Create .streamlit/secrets.toml.example
- [ ] Test Streamlit app locally
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Configure secrets in Streamlit Cloud
- [ ] Verify deployment with sample queries

---

## Demo Preparation

- [ ] Prepare sample queries for demo
- [ ] Test error handling scenarios
- [ ] Create backup plan if LLM API fails
- [ ] Document demo flow

---

## Evaluation Metrics

- [ ] Beam Search: Measure expansion recall improvement
- [ ] Beam Search: Manual check synonym quality (20 samples)
- [ ] TF-IDF: Measure Precision@k (annotate 50 results)
- [ ] TF-IDF: Measure term coverage
- [ ] Naive Bayes: Report accuracy on test set
- [ ] Naive Bayes: Report per-topic F1 scores
- [ ] Bayesian Network: Compute log-likelihood on test data
- [ ] Bayesian Network: Sanity check probabilities
- [ ] HMM: Measure perplexity on held-out sequences
- [ ] HMM: Verify discovered patterns are meaningful
- [ ] LLM: Manual check factual accuracy (10 summaries)
- [ ] LLM: Measure response time (target < 5s)
- [ ] UI: Measure time from search to results (target < 10s)
- [ ] UI: Verify all pipeline steps visible

---

## Post-Integration (Optional)

- [ ] Create shared TF-IDF vectorizer config
- [ ] Add user filters (topic, rating, sentiment dropdowns)
- [ ] Add visualization charts (topic distribution, sentiment flow)
- [ ] Add export results (CSV/PDF)
- [ ] Add caching for TF-IDF vectors and model predictions
