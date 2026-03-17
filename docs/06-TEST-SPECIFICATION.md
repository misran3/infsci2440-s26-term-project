# Test Specification

## Test Strategy
**Core unit tests only** — 1-2 tests per component, ~15 tests total.

## Test List

### Person 1: Search & Expansion

```python
# tests/test_beam_search.py

def test_expand_returns_original_plus_synonyms():
    """Beam search should return original terms plus synonyms."""
    expander = BeamSearchExpander(beam_width=3, max_depth=1)
    result = expander.expand("fast shipping")

    assert "fast" in result.expanded_terms
    assert "shipping" in result.expanded_terms
    assert len(result.expanded_terms) > 2  # Has synonyms

def test_expand_empty_query_returns_empty():
    """Empty query should return empty expansion."""
    expander = BeamSearchExpander()
    result = expander.expand("")

    assert result.original_query == ""
    assert result.expanded_terms == []
```

```python
# tests/test_tfidf.py

def test_retrieve_returns_relevant_reviews():
    """TF-IDF should return reviews matching query terms."""
    corpus = [
        Review("1", "Fast shipping, arrived next day", 5, "Great", "A"),
        Review("2", "Software crashes constantly", 1, "Bad", "B"),
        Review("3", "Delivery was slow but product is good", 3, "OK", "C"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping", "delivery"], top_k=2)

    # Should return shipping/delivery reviews, not the software one
    assert len(results) == 2
    assert all("software" not in r.text.lower() for r in results)

def test_retrieve_empty_corpus():
    """Empty corpus should return empty results."""
    retriever = TFIDFRetriever([])
    retriever.fit()
    results = retriever.retrieve(["anything"])
    assert results == []
```

### Person 2: Classification

```python
# tests/test_naive_bayes.py

def test_classifier_predicts_correct_topic():
    """Classifier should predict topics from training data."""
    train_reviews = [
        Review("1", "App crashes every time", 1, "Bad", "A"),
        Review("2", "Easy to use interface", 5, "Good", "B"),
        Review("3", "Too expensive for what it does", 2, "Meh", "C"),
    ]
    train_labels = ["performance", "usability", "pricing"]

    classifier = TopicClassifier()
    classifier.fit(train_reviews, train_labels)

    test_reviews = [Review("4", "Software freezes constantly", 1, "Terrible", "D")]
    predictions = classifier.predict(test_reviews)

    assert predictions[0].predicted_topic == "performance"

def test_classifier_handles_empty_input():
    """Classifier should handle empty review list."""
    classifier = TopicClassifier()
    # Fit with some data first
    classifier.fit(
        [Review("1", "Test", 3, "T", "A")],
        ["other"]
    )
    results = classifier.predict([])
    assert results == []

def test_filter_by_topic_no_matches_returns_all():
    """When topic filter matches 0 reviews, return all candidates as fallback."""
    reviews = [
        Review("1", "Software crashes constantly", 1, "Bad", "A"),
        Review("2", "App freezes on startup", 2, "Terrible", "B"),
    ]
    labels = ["performance", "performance"]

    classifier = TopicClassifier()
    classifier.fit(reviews, labels)

    # Filter for a topic that doesn't match any reviews
    filtered, _ = classifier.filter_by_topic(reviews, "pricing", min_confidence=0.5)

    # Should return all candidates as fallback, not empty list
    assert len(filtered) == len(reviews)
```

### Person 3: Probabilistic & LLM

```python
# tests/test_bayesian.py

def test_bayesian_query_returns_probabilities():
    """Bayesian network should return valid probabilities."""
    reviews = [
        Review("1", "Crashes", 1, "Bad", "A", topic="performance", sentiment="negative"),
        Review("2", "Works great", 5, "Good", "B", topic="performance", sentiment="positive"),
        Review("3", "Too slow", 2, "Meh", "C", topic="performance", sentiment="negative"),
    ]

    bn = ReviewBayesianNetwork()
    bn.fit(reviews)
    insights = bn.query("performance")

    assert 0 <= insights.p_negative_given_topic <= 1
    assert 0 <= insights.p_positive_given_topic <= 1
    # Should sum to ~1 (with neutral)
    assert insights.p_negative_given_topic + insights.p_positive_given_topic <= 1.0

def test_bayesian_unknown_topic_uses_default():
    """Unknown topic should not crash, return defaults."""
    bn = ReviewBayesianNetwork()
    bn.fit([Review("1", "Test", 3, "T", "A", topic="other")])

    # Query unknown topic
    insights = bn.query("unknown_topic")
    # Should return something, not crash
    assert insights.topic == "unknown_topic"
```

```python
# tests/test_hmm.py

def test_hmm_analyze_multi_sentence():
    """HMM should analyze sentiment per sentence."""
    hmm = SentimentHMM()

    review = Review(
        "1",
        "Great product. But it broke after a week. Customer service helped.",
        3, "Mixed", "A"
    )
    result = hmm.analyze(review)

    assert len(result.sentences) == 3
    assert len(result.sentiment_states) == 3
    assert result.sentiment_states[0] == "positive"  # "Great product"
    assert result.sentiment_states[1] == "negative"  # "broke after a week"

def test_hmm_single_sentence_review():
    """Single sentence reviews should not crash."""
    hmm = SentimentHMM()
    review = Review("1", "Good product", 4, "Good", "A")
    result = hmm.analyze(review)

    assert len(result.sentences) == 1
    assert len(result.sentiment_states) == 1
```

```python
# tests/test_llm.py

def test_summarize_with_mock():
    """LLM summarizer should produce non-empty summary."""
    # Use mock to avoid API calls
    from src.mocks import MockLLMSummarizer

    summarizer = MockLLMSummarizer()
    result = create_mock_pipeline_result()

    summary = summarizer.summarize(result)

    assert len(summary) > 50
    assert result.bayesian_insights.topic in summary

def test_summarize_api_failure_returns_fallback():
    """LLM summarizer should return fallback when API fails."""
    from unittest.mock import patch, MagicMock
    from src.reasoning.llm_summarizer import LLMSummarizer

    summarizer = LLMSummarizer()
    result = create_mock_pipeline_result()

    # Simulate API failure
    with patch.object(summarizer.agent, 'run_sync', side_effect=Exception("API error")):
        summary = summarizer.summarize(result)

    # Should return fallback summary, not crash
    assert "AI-generated summary unavailable" in summary or len(summary) > 0
    assert result.bayesian_insights.topic in summary
```

### Integration Tests

```python
# tests/test_pipeline.py

def test_pipeline_happy_path():
    """Full pipeline should complete without errors."""
    pipeline = create_test_pipeline()  # Uses small test dataset

    result = pipeline.run("shipping problems")

    assert result.expansion is not None
    assert len(result.candidate_reviews) > 0
    assert len(result.filtered_reviews) > 0
    assert result.bayesian_insights is not None
    assert result.llm_summary != ""

def test_pipeline_empty_query():
    """Empty query should return early with error."""
    pipeline = create_test_pipeline()

    result = pipeline.run("")

    assert result.error == "Please enter a query"
    assert result.filtered_reviews == []

def test_pipeline_no_results():
    """Query with no matches should handle gracefully."""
    pipeline = create_test_pipeline()

    result = pipeline.run("quantum_entanglement_xyz_nonexistent")

    assert len(result.candidate_reviews) == 0
    assert result.llm_summary == "" or "No matching reviews" in result.llm_summary
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific component tests
uv run pytest tests/test_beam_search.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## Test Fixtures

```python
# tests/conftest.py

import pytest
from src.data.structures import Review

@pytest.fixture
def sample_reviews():
    """Common test reviews."""
    return [
        Review("1", "Fast shipping, product works great", 5, "Excellent", "A"),
        Review("2", "Software keeps crashing on startup", 1, "Terrible", "B"),
        Review("3", "Good value for the price", 4, "Worth it", "C"),
        Review("4", "Installation was confusing", 2, "Frustrating", "D"),
        Review("5", "Customer support was helpful", 5, "Great service", "E"),
    ]

@pytest.fixture
def mock_pipeline_result():
    """Mock PipelineResult for LLM testing."""
    # ... create mock result
```
