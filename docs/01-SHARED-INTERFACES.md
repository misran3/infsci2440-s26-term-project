# Shared Interfaces

This document defines the contracts between components. Each person can develop independently by mocking these interfaces.

---

## Data Structures

### Review
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Review:
    """Single review from the dataset."""
    review_id: str
    text: str
    rating: int  # 1-5
    title: str
    product_id: str
    # Added by pipeline:
    topic: Optional[str] = None
    sentiment: Optional[str] = None  # "positive", "negative", "neutral"
    sentiment_sequence: Optional[list[str]] = None  # per-sentence sentiments
    tfidf_score: Optional[float] = None  # Added by TFIDFRetriever
    sentences: Optional[list[str]] = None  # Pre-tokenized sentences for HMM
```

### QueryExpansion
```python
@dataclass
class QueryExpansion:
    """Result of beam search query expansion."""
    original_query: str
    expanded_terms: list[str]  # All expanded terms
    beam_paths: list[dict]     # For UI visualization
    # beam_paths example:
    # [{"path": ["shipping", "delivery"], "score": 0.9},
    #  {"path": ["shipping", "package"], "score": 0.85}]
```

### TopicClassification
```python
@dataclass
class TopicClassification:
    """Result of Naive Bayes classification."""
    review_id: str
    predicted_topic: str
    confidence: float
    top_features: list[str]  # Words that contributed most
```

### BayesianInsights
```python
@dataclass
class BayesianInsights:
    """Probabilistic insights from Bayesian Network."""
    topic: str
    p_negative_given_topic: float
    p_positive_given_topic: float
    p_low_rating_given_negative: float  # P(1-2 stars | negative)
    p_high_rating_given_positive: float  # P(4-5 stars | positive)
```

### SentimentSequence
```python
@dataclass
class SentimentSequence:
    """HMM analysis of sentiment flow in a review."""
    review_id: str
    sentences: list[str]
    sentiment_states: list[str]  # ["positive", "negative", "positive"]
    transitions: dict[str, float]  # {"pos_to_neg": 0.25, "neg_to_pos": 0.6}
```

### PipelineResult
```python
@dataclass
class PipelineResult:
    """Final result passed to LLM and UI."""
    query: str
    expansion: QueryExpansion
    candidate_reviews: list[Review]
    filtered_reviews: list[Review]  # After topic filtering
    topic_classifications: list[TopicClassification]
    bayesian_insights: BayesianInsights
    sentiment_sequences: list[SentimentSequence]
    llm_summary: str
```

---

## Component Interfaces

### 1. Beam Search (Person 1)

```python
class BeamSearchExpander:
    """Expands queries using beam search + WordNet synonyms."""

    def __init__(self, beam_width: int = 3, max_depth: int = 2):
        """
        Args:
            beam_width: Number of candidates to keep at each step
            max_depth: Maximum expansion depth
        """
        pass

    def expand(self, query: str) -> QueryExpansion:
        """
        Expand a query using beam search.

        Args:
            query: User's original query string

        Returns:
            QueryExpansion with expanded terms and beam paths
        """
        pass
```

**Mock for other components:**
```python
def mock_beam_search(query: str) -> QueryExpansion:
    return QueryExpansion(
        original_query=query,
        expanded_terms=[query, f"{query} issues", f"{query} problems"],
        beam_paths=[{"path": [query, f"{query} issues"], "score": 0.9}]
    )
```

---

### 2. TF-IDF Retrieval (Person 1)

```python
class TFIDFRetriever:
    """Retrieves reviews using TF-IDF similarity."""

    def __init__(self, corpus: list[Review]):
        """
        Args:
            corpus: All reviews to search over
        """
        pass

    def fit(self) -> None:
        """Fit TF-IDF vectorizer on corpus."""
        pass

    def retrieve(
        self,
        expanded_terms: list[str],
        top_k: int = 500
    ) -> list[Review]:
        """
        Retrieve top-k reviews matching expanded terms.

        Args:
            expanded_terms: Terms from beam search expansion
            top_k: Number of reviews to return

        Returns:
            List of Review objects sorted by relevance
        """
        pass
```

**Mock:**
```python
def mock_tfidf_retrieve(terms: list[str], top_k: int) -> list[Review]:
    return [
        Review(review_id="1", text="Shipping was slow", rating=2, title="Bad", product_id="A"),
        Review(review_id="2", text="Fast delivery!", rating=5, title="Good", product_id="B"),
    ][:top_k]
```

---

### 3. Naive Bayes Classifier (Person 2)

```python
class TopicClassifier:
    """Classifies reviews into topics using Naive Bayes."""

    TOPICS = ["shipping", "quality", "price", "support", "features", "other"]

    def __init__(self):
        pass

    def fit(self, reviews: list[Review], labels: list[str]) -> None:
        """
        Train the classifier.

        Args:
            reviews: Training reviews
            labels: Topic labels for each review
        """
        pass

    def predict(self, reviews: list[Review]) -> list[TopicClassification]:
        """
        Classify reviews into topics.

        Args:
            reviews: Reviews to classify

        Returns:
            TopicClassification for each review
        """
        pass

    def filter_by_topic(
        self,
        reviews: list[Review],
        target_topic: str,
        min_confidence: float = 0.5
    ) -> list[Review]:
        """
        Filter reviews to those matching target topic.

        Args:
            reviews: Candidate reviews
            target_topic: Topic to filter for
            min_confidence: Minimum classification confidence

        Returns:
            Filtered reviews with topic assigned
        """
        pass
```

**Mock:**
```python
def mock_classify(reviews: list[Review]) -> list[TopicClassification]:
    return [
        TopicClassification(
            review_id=r.review_id,
            predicted_topic="shipping",
            confidence=0.85,
            top_features=["delivery", "arrived", "package"]
        )
        for r in reviews
    ]
```

---

### 4. Bayesian Network (Person 3)

```python
class ReviewBayesianNetwork:
    """Bayesian Network for probabilistic reasoning over reviews."""

    def __init__(self):
        pass

    def fit(self, reviews: list[Review]) -> None:
        """
        Learn network structure and CPTs from data.

        Args:
            reviews: Reviews with topic and sentiment assigned
        """
        pass

    def query(self, topic: str) -> BayesianInsights:
        """
        Query the network for probabilistic insights.

        Args:
            topic: Topic to query about

        Returns:
            BayesianInsights with conditional probabilities
        """
        pass
```

**Mock:**
```python
def mock_bayesian_query(topic: str) -> BayesianInsights:
    return BayesianInsights(
        topic=topic,
        p_negative_given_topic=0.68,
        p_positive_given_topic=0.32,
        p_low_rating_given_negative=0.75,
        p_high_rating_given_positive=0.85
    )
```

---

### 5. HMM Sentiment Analyzer (Person 3)

```python
class SentimentHMM:
    """HMM for analyzing sentiment sequences in reviews."""

    STATES = ["positive", "negative", "neutral"]

    def __init__(self):
        pass

    def fit(self, reviews: list[Review]) -> None:
        """
        Train HMM on review sentiment sequences.

        Args:
            reviews: Reviews to learn from
        """
        pass

    def analyze(self, review: Review) -> SentimentSequence:
        """
        Analyze sentiment sequence in a single review.

        Args:
            review: Review to analyze

        Returns:
            SentimentSequence with per-sentence sentiments
        """
        pass

    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """
        Get learned transition probabilities.

        Returns:
            Nested dict: transition_matrix["positive"]["negative"] = P(neg|pos)
        """
        pass
```

**Mock:**
```python
def mock_hmm_analyze(review: Review) -> SentimentSequence:
    return SentimentSequence(
        review_id=review.review_id,
        sentences=["Product arrived fast.", "But packaging was damaged."],
        sentiment_states=["positive", "negative"],
        transitions={"pos_to_neg": 0.25, "neg_to_pos": 0.6}
    )
```

---

### 6. LLM Summarizer (Person 3)

```python
class LLMSummarizer:
    """Generates natural language summaries using LLM."""

    def __init__(self, model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        """
        Args:
            model: LLM model to use
        """
        pass

    def summarize(self, pipeline_result: PipelineResult) -> str:
        """
        Generate summary from pipeline results.

        Args:
            pipeline_result: All outputs from earlier pipeline stages

        Returns:
            Natural language summary string
        """
        pass
```

**Mock:**
```python
def mock_llm_summarize(result: PipelineResult) -> str:
    return f"Found {len(result.filtered_reviews)} reviews about {result.bayesian_insights.topic}. " \
           f"{result.bayesian_insights.p_negative_given_topic*100:.0f}% are negative."
```

---

### 7. Pipeline Orchestrator

```python
class SurveyAnalysisPipeline:
    """Orchestrates all components."""

    def __init__(
        self,
        expander: BeamSearchExpander,
        retriever: TFIDFRetriever,
        classifier: TopicClassifier,
        bayesian_net: ReviewBayesianNetwork,
        hmm: SentimentHMM,
        summarizer: LLMSummarizer
    ):
        pass

    def run(
        self,
        query: str,
        topic_filter: str = None,
        rating_filter: tuple[int, int] = None,
        sentiment_filter: str = None
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            query: User's query
            topic_filter: Optional pre-filter by topic
            rating_filter: Optional pre-filter by rating range (min, max)
            sentiment_filter: Optional pre-filter by sentiment

        Returns:
            Complete PipelineResult
        """
        pass
```

---

## File Structure

See `00-PROJECT-OVERVIEW.md` for the complete project file structure.

---

## Dependencies

```bash
# Install all dependencies with uv
uv sync

# Or install specific groups
uv sync --group dev  # Include dev dependencies (pytest)
```

See `pyproject.toml` for full dependency list.

---

## Testing Strategy

### Unit Tests (Each person writes for their components)
```python
# Example: test_beam_search.py
def test_expand_simple_query():
    expander = BeamSearchExpander(beam_width=3, max_depth=2)
    result = expander.expand("shipping")

    assert result.original_query == "shipping"
    assert len(result.expanded_terms) > 1
    assert "shipping" in result.expanded_terms

def test_expand_with_synonyms():
    expander = BeamSearchExpander()
    result = expander.expand("fast")

    # Should find synonyms like "quick", "rapid"
    assert any(term in result.expanded_terms for term in ["quick", "rapid", "speedy"])
```

### Integration Tests (After Phase 2)
```python
# Example: test_pipeline.py
def test_full_pipeline():
    pipeline = SurveyAnalysisPipeline(...)
    result = pipeline.run("shipping problems")

    assert result.expansion is not None
    assert len(result.filtered_reviews) > 0
    assert result.llm_summary != ""
```
