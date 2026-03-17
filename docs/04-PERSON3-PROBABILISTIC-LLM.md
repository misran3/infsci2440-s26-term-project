# Person 3: Probabilistic & LLM

## Ownership
- **Bayesian Network** (AI technique - Course Units 4-5)
- **HMM Sentiment Sequences** (AI technique - Course Units 4-5)
- **LLM Integration** (Summarization)

---

## Component 1: Bayesian Network

### Training Prerequisites

**Before training the Bayesian Network:**
1. Data Prep must be complete (`amazon_reviews_software.csv` exists)
2. Naive Bayes must be trained and run on the corpus
3. Reviews must have `topic` field populated from NB classification

```
DATA PREP → NAIVE BAYES → BAYESIAN NETWORK
              (topics)       (learns P(sentiment|topic))
```

### What It Does
Models probabilistic relationships between review attributes (Topic, Sentiment, Rating) using a Bayesian Network. Enables queries like:
- "What's the probability a review about performance is negative?"
- "Given a negative review, what's the likely rating?"

### Course Alignment
**Units 4-5: Agents with Uncertain Knowledge**
- Bayesian Networks represent joint probability distributions
- Conditional probability tables (CPTs) encode P(child | parents)
- Can perform probabilistic inference (query given evidence)

### Network Structure

```
          ┌─────────┐
          │  Topic  │
          └────┬────┘
               │
               ▼
          ┌─────────┐
          │Sentiment│
          └────┬────┘
               │
               ▼
          ┌─────────┐
          │ Rating  │
          └─────────┘

Assumptions:
- Topic influences Sentiment (e.g., pricing complaints tend negative)
- Sentiment influences Rating (negative → low stars)
```

### Implementation

```python
# src/reasoning/bayesian_network.py

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class BayesianInsights:
    topic: str
    p_negative_given_topic: float
    p_positive_given_topic: float
    p_low_rating_given_negative: float
    p_high_rating_given_positive: float

class ReviewBayesianNetwork:
    def __init__(self):
        # Define network structure
        self.model = BayesianNetwork([
            ('topic', 'sentiment'),
            ('sentiment', 'rating_category')
        ])
        self.inference = None
        self.is_fitted = False

    def _prepare_data(self, reviews: List['Review']) -> pd.DataFrame:
        """Convert reviews to DataFrame for pgmpy."""
        data = []
        for r in reviews:
            # Categorize rating
            if r.rating <= 2:
                rating_cat = 'low'
            elif r.rating == 3:
                rating_cat = 'medium'
            else:
                rating_cat = 'high'

            # Determine sentiment (simple rule-based for now)
            # In production, this would come from HMM or sentiment classifier
            sentiment = self._infer_sentiment(r.text, r.rating)

            data.append({
                'topic': r.topic or 'other',
                'sentiment': sentiment,
                'rating_category': rating_cat
            })

        return pd.DataFrame(data)

    def _infer_sentiment(self, text: str, rating: int) -> str:
        """Simple sentiment inference from rating."""
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        return 'neutral'

    def fit(self, reviews: List['Review']) -> None:
        """Learn CPTs from data using Maximum Likelihood Estimation."""
        df = self._prepare_data(reviews)

        # Fit CPTs using MLE
        self.model.fit(df, estimator=MaximumLikelihoodEstimator)

        # Set up inference engine
        self.inference = VariableElimination(self.model)
        self.is_fitted = True

    def query(self, topic: str) -> BayesianInsights:
        """Query the network for insights about a topic."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # P(sentiment | topic)
        sentiment_given_topic = self.inference.query(
            variables=['sentiment'],
            evidence={'topic': topic}
        )

        # Extract probabilities
        sentiment_probs = sentiment_given_topic.values
        sentiment_states = sentiment_given_topic.state_names['sentiment']

        p_negative = sentiment_probs[sentiment_states.index('negative')] if 'negative' in sentiment_states else 0
        p_positive = sentiment_probs[sentiment_states.index('positive')] if 'positive' in sentiment_states else 0

        # P(rating | sentiment=negative)
        try:
            rating_given_negative = self.inference.query(
                variables=['rating_category'],
                evidence={'sentiment': 'negative'}
            )
            rating_probs_neg = rating_given_negative.values
            rating_states = rating_given_negative.state_names['rating_category']
            p_low_given_neg = rating_probs_neg[rating_states.index('low')] if 'low' in rating_states else 0
        except:
            p_low_given_neg = 0.5

        # P(rating | sentiment=positive)
        try:
            rating_given_positive = self.inference.query(
                variables=['rating_category'],
                evidence={'sentiment': 'positive'}
            )
            rating_probs_pos = rating_given_positive.values
            p_high_given_pos = rating_probs_pos[rating_states.index('high')] if 'high' in rating_states else 0
        except:
            p_high_given_pos = 0.5

        return BayesianInsights(
            topic=topic,
            p_negative_given_topic=float(p_negative),
            p_positive_given_topic=float(p_positive),
            p_low_rating_given_negative=float(p_low_given_neg),
            p_high_rating_given_positive=float(p_high_given_pos)
        )

    def get_cpds(self) -> dict:
        """Return CPTs for visualization/debugging."""
        return {cpd.variable: cpd for cpd in self.model.get_cpds()}
```

### Evaluation Metrics (1-3)

1. **Log-Likelihood**: How well does the model fit held-out test data?
   ```python
   # Higher is better (less negative)
   log_likelihood = model.score(test_df)
   ```

2. **Inference Sanity Check**: Do probabilities make intuitive sense?
   ```python
   # e.g., P(negative | pricing) should be > P(negative | features)
   # Manual verification on known relationships
   ```

### UI Output Format
```
Probabilistic Insights (Bayesian Network):
  Topic: performance

  ┌─────────────────────────────────────────┐
  │  P(Negative | Topic=performance) = 68%  │
  │  P(Positive | Topic=performance) = 22%  │
  │  P(Neutral | Topic=performance) = 10%   │
  └─────────────────────────────────────────┘

  ┌─────────────────────────────────────────┐
  │  Given Negative sentiment:              │
  │  P(1-2 stars) = 75%                     │
  │  P(3 stars) = 18%                       │
  │  P(4-5 stars) = 7%                      │
  └─────────────────────────────────────────┘
```

---

## Component 2: HMM Sentiment Sequences

### Training: Optional but Recommended

The HMM can operate in two modes:

| Mode | How It Works | Quality |
|------|--------------|---------|
| **Unfitted** | Uses VADER sentiment directly | Basic |
| **Fitted** | Learns transition patterns from data | Better |

**Recommendation:** Train the HMM on your corpus. It learns patterns like "reviews often start positive, turn negative, then recover" — which VADER alone can't capture.

**When to retrain:**
- When corpus changes significantly
- When sentence tokenization approach changes

### What It Does
Models sentiment transitions within individual reviews using a Hidden Markov Model. Reviews are split into sentences, each sentence is classified as positive/negative/neutral, and the HMM learns transition patterns.

### Course Alignment
**Units 4-5: Agents with Uncertain Knowledge**
- HMMs model sequential data with hidden states
- Transition matrix: P(state_t | state_{t-1})
- Emission matrix: P(observation | state)
- Viterbi algorithm finds most likely state sequence

### Why This Design?
The original proposal suggested HMM for "tracking user intent across multi-turn sessions" — but Amazon Reviews data doesn't have conversations. Instead, we use HMM to analyze **sentiment flow within a single review**, treating each sentence as a time step.

### Implementation

```python
# src/reasoning/hmm_sentiment.py

import nltk
from nltk.tokenize import sent_tokenize
from hmmlearn import hmm
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

@dataclass
class SentimentSequence:
    review_id: str
    sentences: List[str]
    sentiment_states: List[str]
    transitions: Dict[str, float]

class SentimentHMM:
    STATES = ['positive', 'negative', 'neutral']
    STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = hmm.CategoricalHMM(
            n_components=n_states,
            n_iter=100,
            random_state=42
        )
        self.sia = SentimentIntensityAnalyzer()
        self.is_fitted = False

    def _sentence_to_sentiment(self, sentence: str) -> int:
        """Classify sentence sentiment using VADER."""
        scores = self.sia.polarity_scores(sentence)
        compound = scores['compound']

        if compound >= 0.05:
            return self.STATE_TO_IDX['positive']
        elif compound <= -0.05:
            return self.STATE_TO_IDX['negative']
        return self.STATE_TO_IDX['neutral']

    def _review_to_sequence(self, review: 'Review') -> tuple:
        """Convert review to sentiment sequence."""
        sentences = sent_tokenize(review.text)
        if len(sentences) < 2:
            return None, None

        observations = [self._sentence_to_sentiment(s) for s in sentences]
        return sentences, np.array(observations).reshape(-1, 1)

    def fit(self, reviews: List['Review']) -> dict:
        """Train HMM on review sentiment sequences."""
        all_sequences = []
        lengths = []

        for review in reviews:
            sentences, obs = self._review_to_sequence(review)
            if obs is not None and len(obs) >= 2:
                all_sequences.append(obs)
                lengths.append(len(obs))

        if not all_sequences:
            raise ValueError("No valid sequences to train on")

        # Concatenate all sequences
        X = np.vstack(all_sequences)

        # Fit HMM
        self.model.fit(X, lengths)
        self.is_fitted = True

        return {
            "n_sequences": len(all_sequences),
            "avg_length": np.mean(lengths),
            "transition_matrix": self.model.transmat_
        }

    def analyze(self, review: 'Review') -> SentimentSequence:
        """Analyze sentiment sequence in a review."""
        sentences, obs = self._review_to_sequence(review)

        if obs is None:
            return SentimentSequence(
                review_id=review.review_id,
                sentences=[review.text],
                sentiment_states=['neutral'],
                transitions={}
            )

        # Use Viterbi to find most likely state sequence
        if self.is_fitted:
            _, state_sequence = self.model.decode(obs)
        else:
            # If not fitted, just use observed sentiments
            state_sequence = obs.flatten()

        sentiment_states = [self.STATES[s] for s in state_sequence]

        # Compute transition counts for this review
        transitions = self._compute_transitions(sentiment_states)

        return SentimentSequence(
            review_id=review.review_id,
            sentences=sentences,
            sentiment_states=sentiment_states,
            transitions=transitions
        )

    def _compute_transitions(self, states: List[str]) -> Dict[str, float]:
        """Compute transition probabilities from state sequence."""
        counts = {
            'pos_to_neg': 0, 'pos_to_pos': 0, 'pos_to_neu': 0,
            'neg_to_pos': 0, 'neg_to_neg': 0, 'neg_to_neu': 0,
            'neu_to_pos': 0, 'neu_to_neg': 0, 'neu_to_neu': 0,
        }
        totals = {'positive': 0, 'negative': 0, 'neutral': 0}

        for i in range(len(states) - 1):
            curr, next_ = states[i], states[i + 1]
            key = f"{curr[:3]}_to_{next_[:3]}"
            counts[key] = counts.get(key, 0) + 1
            totals[curr] += 1

        # Convert to probabilities
        transitions = {}
        for key, count in counts.items():
            parts = key.split('_to_')
            curr = {'pos': 'positive', 'neg': 'negative', 'neu': 'neutral'}[parts[0]]
            if totals[curr] > 0:
                transitions[key] = count / totals[curr]

        return transitions

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get learned transition matrix."""
        if not self.is_fitted:
            return {}

        matrix = {}
        for i, from_state in enumerate(self.STATES):
            matrix[from_state] = {}
            for j, to_state in enumerate(self.STATES):
                matrix[from_state][to_state] = float(self.model.transmat_[i, j])

        return matrix

    def get_common_patterns(self, reviews: List['Review'], top_k: int = 5) -> List[tuple]:
        """Find most common sentiment transition patterns."""
        pattern_counts = {}

        for review in reviews:
            seq = self.analyze(review)
            pattern = tuple(seq.sentiment_states)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:top_k]
```

### Evaluation Metrics (1-3)

1. **Perplexity**: How well does the model predict held-out sequences?
   ```python
   perplexity = model.score(test_sequences) / -len(test_sequences)
   ```

2. **Pattern Discovery**: Does the model find meaningful patterns?
   ```python
   # Show top-5 common patterns and verify they make sense
   common_patterns = hmm.get_common_patterns(reviews)
   ```

### UI Output Format
```
Sentiment Sequences (HMM):
  Analyzed 142 reviews

  Learned Transition Probabilities:
  ┌──────────┬──────────┬──────────┬──────────┐
  │ From\To  │ Positive │ Negative │ Neutral  │
  ├──────────┼──────────┼──────────┼──────────┤
  │ Positive │   65%    │   25%    │   10%    │
  │ Negative │   40%    │   45%    │   15%    │
  │ Neutral  │   35%    │   30%    │   35%    │
  └──────────┴──────────┴──────────┴──────────┘

  Common Patterns:
  1. Positive → Positive (45 reviews) - "Consistently happy"
  2. Positive → Negative → Positive (32 reviews) - "Complaint sandwich"
  3. Negative → Negative (28 reviews) - "Consistently unhappy"
```

---

## Component 3: LLM Summarization

### What It Does
Takes the filtered reviews and probabilistic insights from the pipeline, then generates a natural language summary using Claude or another LLM.

### Why LLM Here?
- Classical AI does the heavy lifting (filtering, classification, reasoning)
- LLM makes output human-readable
- Shows how classical AI + LLM can complement each other

### Implementation (using Pydantic AI)

Pydantic AI provides a **provider-agnostic** interface to LLMs. You can switch between OpenAI, Anthropic, Gemini, etc. by changing the model string.

```python
# src/reasoning/llm_summarizer.py

from pydantic_ai import Agent
from pydantic import BaseModel
from src.data.structures import PipelineResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SummaryOutput(BaseModel):
    """Structured output from LLM."""
    summary: str
    key_themes: list[str]
    representative_quotes: list[str]

class LLMSummarizer:
    def __init__(self, model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        """
        Initialize with a Pydantic AI model via Amazon Bedrock.

        Default: Claude 4.5 Haiku via Bedrock (fast, capable)

        Alternative models:
        - "openai:gpt-4o-mini" (OpenAI)
        - "anthropic:claude-3-5-sonnet-20241022" (Anthropic direct)
        - "gemini-1.5-flash" (Google)
        """
        self.agent = Agent(
            model,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        return """You are an AI assistant that summarizes product review analysis results.
Given pipeline analysis data and sample reviews, write a concise summary that:
1. States the key finding (topic, number of reviews, sentiment distribution)
2. Highlights 2-3 main themes or patterns
3. Includes 2-3 representative quotes from the actual reviews

Be concise and actionable. Do not invent quotes - only use text from the provided reviews."""

    def summarize(self, result: PipelineResult) -> str:
        """Generate summary from pipeline results."""
        context = self._build_context(result)
        reviews_text = self._format_reviews(result.filtered_reviews[:10])

        prompt = f"""Summarize these product review analysis results:

**Query:** {result.query}

**Pipeline Analysis:**
{context}

**Sample Reviews (top 10):**
{reviews_text}"""

        try:
            response = self.agent.run_sync(prompt)
            return response.data
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._generate_fallback_summary(result)

    def _build_context(self, result: PipelineResult) -> str:
        """Build context string from pipeline results."""
        lines = []

        # Query expansion
        lines.append(f"- Query expanded to: {', '.join(result.expansion.expanded_terms[:10])}")

        # Retrieval
        lines.append(f"- Retrieved {len(result.candidate_reviews)} candidates, filtered to {len(result.filtered_reviews)}")

        # Bayesian insights
        bi = result.bayesian_insights
        lines.append(f"- Topic: {bi.topic}")
        lines.append(f"- P(Negative | {bi.topic}) = {bi.p_negative_given_topic:.0%}")
        lines.append(f"- P(Positive | {bi.topic}) = {bi.p_positive_given_topic:.0%}")

        # HMM insights
        if result.sentiment_sequences:
            sample_seq = result.sentiment_sequences[0]
            lines.append(f"- Common sentiment flow: {' → '.join(sample_seq.sentiment_states[:5])}")

        return '\n'.join(lines)

    def _format_reviews(self, reviews: list) -> str:
        """Format reviews for LLM context."""
        lines = []
        for r in reviews:
            rating_stars = '★' * r.rating + '☆' * (5 - r.rating)
            lines.append(f"[{rating_stars}] {r.text[:300]}...")
        return '\n\n'.join(lines)

    def _generate_fallback_summary(self, result: PipelineResult) -> str:
        """Generate basic summary when LLM fails."""
        bi = result.bayesian_insights
        n = len(result.filtered_reviews)
        return (
            f"Found {n} reviews about **{bi.topic}**. "
            f"{bi.p_negative_given_topic:.0%} are negative, "
            f"{bi.p_positive_given_topic:.0%} are positive. "
            f"*(AI-generated summary unavailable)*"
        )
```

### Environment Setup

```bash
# For Amazon Bedrock (default - Claude 4.5 Haiku)
# Configure AWS credentials via AWS CLI or environment variables
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Alternative: For OpenAI models
export OPENAI_API_KEY="sk-..."

# Alternative: For Anthropic direct
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Mock for Testing
```python
def mock_llm_summarize(result: PipelineResult) -> str:
    """Mock LLM for testing without API calls."""
    bi = result.bayesian_insights
    n = len(result.filtered_reviews)

    return f"""Based on {n} reviews about **{bi.topic}**:

**Key Finding:** {bi.p_negative_given_topic:.0%} of reviews about {bi.topic} are negative, with most giving 1-2 stars.

**Common Themes:**
- Users frequently mention issues with reliability
- Despite complaints, some note recent improvements

**Representative Quotes:**
- "{result.filtered_reviews[0].text[:100]}..."
- "{result.filtered_reviews[1].text[:100]}..." (if available)
"""
```

### Evaluation Metrics (1-3)

1. **Factual Accuracy**: Does the summary correctly reflect the pipeline outputs?
   ```python
   # Manual check: 10 summaries verified against pipeline data
   ```

2. **Response Time**: How long does LLM call take?
   ```python
   # Target: < 5 seconds for summary generation
   ```

---

## Development Checklist

### Week 1-2 (Parallel Development)
- [ ] Set up development environment
- [ ] Implement `ReviewBayesianNetwork` with pgmpy
- [ ] Write unit tests for Bayesian Network
- [ ] Implement `SentimentHMM` with hmmlearn
- [ ] Write unit tests for HMM
- [ ] Set up LLM API access (OpenAI, Anthropic, or Gemini)
- [ ] Implement `LLMSummarizer`
- [ ] Test with mock pipeline data

### Week 3 (Integration)
- [ ] Train Bayesian Network on classified reviews (needs Person 2's NB)
- [ ] Train HMM on real reviews
- [ ] Integrate LLM with full pipeline
- [ ] Test end-to-end flow
- [ ] Tune model parameters

### Week 4 (Polish)
- [ ] Compute evaluation metrics
- [ ] Optimize LLM prompts
- [ ] Ensure UI outputs are formatted correctly
- [ ] Demo preparation
- [ ] Documentation

---

## Dependencies

All dependencies are managed in `pyproject.toml`. Install with:
```bash
uv sync
```

Key dependencies for Person 3:
- `pgmpy>=0.1.24` (Bayesian Network)
- `hmmlearn>=0.3` (Hidden Markov Model)
- `pydantic-ai>=0.0.10` (Provider-agnostic LLM)
- `nltk>=3.8` (Sentence tokenization, VADER sentiment)

---

## Testing

```python
# tests/test_bayesian.py

def test_fit_and_query():
    reviews = [
        Review("1", "Crashes constantly", 1, "Bad", "A", topic="performance"),
        Review("2", "Works great", 5, "Good", "B", topic="performance"),
        Review("3", "Too expensive", 2, "Meh", "C", topic="pricing"),
    ]

    bn = ReviewBayesianNetwork()
    bn.fit(reviews)

    insights = bn.query("performance")

    assert insights.topic == "performance"
    assert 0 <= insights.p_negative_given_topic <= 1
    assert 0 <= insights.p_positive_given_topic <= 1


# tests/test_hmm.py

def test_analyze_review():
    hmm = SentimentHMM()

    # Don't need to fit for basic analysis (uses VADER)
    review = Review("1", "Great product. But battery dies fast. Still recommend.", 4, "OK", "A")
    result = hmm.analyze(review)

    assert len(result.sentences) == 3
    assert result.sentiment_states[0] == 'positive'  # "Great product"
    assert result.sentiment_states[1] == 'negative'  # "battery dies fast"


# tests/test_llm.py

def test_summarize_mock():
    # Use mock to avoid API calls in tests
    from src.mocks import mock_llm_summarize

    result = PipelineResult(...)  # Create mock result
    summary = mock_llm_summarize(result)

    assert len(summary) > 100
    assert result.bayesian_insights.topic in summary
```
