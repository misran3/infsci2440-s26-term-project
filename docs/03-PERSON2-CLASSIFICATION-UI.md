# Person 2: Classification & UI

## Ownership
- **Naive Bayes Topic Classification** (AI technique - Course Unit 7)
- **Streamlit UI** (Integration)

---

## Component 1: Naive Bayes Topic Classifier

### What It Does
Classifies product reviews into predefined topics using Multinomial Naive Bayes. This allows filtering reviews to only those relevant to the user's query topic.

### Course Alignment
**Unit 7: Agents with Learning Capabilities - Statistical Methods**
- Naive Bayes is a probabilistic classifier
- Uses Bayes' theorem: P(topic|words) ∝ P(words|topic) × P(topic)
- Assumes word independence (naive assumption)
- Fast training and inference

### Topics for Software Reviews
```python
TOPICS = [
    "performance",    # Speed, crashes, bugs, stability
    "usability",      # UI, ease of use, learning curve
    "features",       # Functionality, capabilities, missing features
    "pricing",        # Cost, value, subscription
    "support",        # Customer service, documentation, updates
    "compatibility",  # Installation, OS support, integrations
    "other"           # Catch-all
]
```

### Implementation

```python
# src/classification/naive_bayes.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class TopicClassification:
    review_id: str
    predicted_topic: str
    confidence: float
    top_features: List[str]

class TopicClassifier:
    TOPICS = ["performance", "usability", "features", "pricing",
              "support", "compatibility", "other"]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = MultinomialNB(alpha=1.0)  # Laplace smoothing
        self.is_fitted = False

    def fit(self, reviews: List['Review'], labels: List[str]) -> dict:
        """
        Train the classifier.

        Returns:
            dict with training metrics
        """
        texts = [r.text for r in reviews]

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train
        self.classifier.fit(X_train_vec, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = (y_pred == y_test).mean()

        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred)
        }

    def predict(self, reviews: List['Review']) -> List[TopicClassification]:
        """Classify reviews into topics."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        texts = [r.text for r in reviews]
        X = self.vectorizer.transform(texts)

        # Get predictions and probabilities
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)

        results = []
        feature_names = self.vectorizer.get_feature_names_out()

        for i, review in enumerate(reviews):
            topic = predictions[i]
            confidence = probabilities[i].max()

            # Get top contributing features for this prediction
            topic_idx = list(self.classifier.classes_).index(topic)
            feature_scores = X[i].toarray().flatten() * self.classifier.feature_log_prob_[topic_idx]
            top_feature_indices = np.argsort(feature_scores)[-5:][::-1]
            top_features = [feature_names[idx] for idx in top_feature_indices if feature_scores[idx] > 0]

            results.append(TopicClassification(
                review_id=review.review_id,
                predicted_topic=topic,
                confidence=confidence,
                top_features=top_features[:3]
            ))

        return results

    def filter_by_topic(
        self,
        reviews: List['Review'],
        target_topic: str,
        min_confidence: float = 0.5
    ) -> Tuple[List['Review'], List[TopicClassification]]:
        """Filter reviews to those matching target topic."""
        classifications = self.predict(reviews)

        filtered = []
        filtered_classifications = []

        for review, classification in zip(reviews, classifications):
            if (classification.predicted_topic == target_topic and
                classification.confidence >= min_confidence):
                review.topic = classification.predicted_topic
                filtered.append(review)
                filtered_classifications.append(classification)

        return filtered, filtered_classifications

    def detect_topic_from_query(self, query: str) -> str:
        """Infer topic from user's query."""
        # Simple keyword matching for topic detection
        query_lower = query.lower()
        topic_keywords = {
            "performance": ["slow", "fast", "crash", "bug", "freeze", "lag", "speed"],
            "usability": ["easy", "difficult", "confusing", "intuitive", "ui", "interface"],
            "features": ["feature", "function", "capability", "missing", "wish", "need"],
            "pricing": ["price", "cost", "expensive", "cheap", "value", "subscription", "free"],
            "support": ["support", "help", "customer service", "response", "documentation"],
            "compatibility": ["install", "compatible", "windows", "mac", "linux", "version"],
        }

        scores = {topic: 0 for topic in self.TOPICS}
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[topic] += 1

        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0 else "other"
```

### Training Data
For training, you'll need labeled examples. Options:

1. **Semi-supervised labeling**: Use keyword rules to auto-label, then manually verify a sample
2. **Use review metadata**: Some datasets have category tags
3. **Manual labeling**: Label 500-1000 reviews (can split among team)

```python
# Example: Keyword-based auto-labeling for initial training
def auto_label_review(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["crash", "bug", "slow", "freeze"]):
        return "performance"
    elif any(w in text_lower for w in ["easy", "confusing", "interface"]):
        return "usability"
    # ... more rules
    return "other"
```

### Model Persistence

```python
import joblib

# Save trained classifier
def save_model(classifier: TopicClassifier, path: str = "models/naive_bayes.pkl"):
    joblib.dump({
        'vectorizer': classifier.vectorizer,
        'classifier': classifier.classifier,
        'is_fitted': classifier.is_fitted
    }, path)

# Load trained classifier
def load_model(path: str = "models/naive_bayes.pkl") -> TopicClassifier:
    data = joblib.load(path)
    classifier = TopicClassifier()
    classifier.vectorizer = data['vectorizer']
    classifier.classifier = data['classifier']
    classifier.is_fitted = data['is_fitted']
    return classifier
```

**When to retrain:**
- When labeled training data changes (new labels, corrections)
- When accuracy on verification sample drops below 70%
- When new topics are added to TOPICS list

### Evaluation Metrics (1-3)

1. **Accuracy**: Overall classification accuracy on test set
   ```python
   accuracy = correct_predictions / total_predictions
   ```

2. **Per-topic F1**: F1 score for each topic (handles class imbalance)
   ```python
   from sklearn.metrics import f1_score
   f1_per_topic = f1_score(y_true, y_pred, average=None)
   ```

### UI Output Format
```
Topic Classification (Naive Bayes):
  Detected topic: "performance"
  Filtered: 142 reviews (from 487 candidates)
  Confidence threshold: 0.5

  Classification breakdown:
    performance: 142 (29%)
    usability: 98 (20%)
    features: 87 (18%)
    ...

  Top features for "performance": crash, slow, freeze
```

---

## Component 2: Streamlit UI

### What It Does
Provides a web interface for users to:
1. Enter queries with optional filters
2. See each AI pipeline step's output
3. View the final LLM summary

### Design

```
+------------------------------------------------------------------+
|  Survey Analysis Agent                                            |
+------------------------------------------------------------------+
|                                                                   |
|  Query: [_________________________________] [Search]              |
|                                                                   |
|  Optional Filters:                                                |
|  Topic: [All v]  Rating: [All v]  Sentiment: [All v]             |
|                                                                   |
+------------------------------------------------------------------+
|  PIPELINE RESULTS                                                 |
+------------------------------------------------------------------+
|  1. Query Expansion (Beam Search)                                 |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ Original: "shipping problems"                                │ |
|  │ Expanded: shipping, delivery, transport, problems, issues    │ |
|  │ [Show expansion tree]                                        │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
|  2. TF-IDF Retrieval                                              |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ Found 487 candidate reviews                                  │ |
|  │ Top terms: shipping (312), delivery (98), issues (77)        │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
|  3. Topic Classification (Naive Bayes)                            |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ Detected topic: performance                                  │ |
|  │ Filtered to 142 relevant reviews                             │ |
|  │ Confidence: 0.85                                             │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
|  4. Probabilistic Insights (Bayesian Network)                     |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ P(Negative | Topic=performance) = 0.68                       │ |
|  │ P(1-2 stars | Negative) = 0.75                               │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
|  5. Sentiment Analysis (HMM)                                      |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ Common pattern: Positive → Negative → Positive               │ |
|  │ P(Neg→Pos) = 0.60 (customers often end positively)           │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
|  6. Summary (LLM)                                                 |
|  ┌─────────────────────────────────────────────────────────────┐ |
|  │ Based on 142 reviews about performance issues:               │ |
|  │                                                              │ |
|  │ **Key Findings:**                                            │ |
|  │ - 68% of performance reviews are negative                    │ |
|  │ - Common complaints: crashes, slow loading, memory leaks     │ |
|  │ - Despite frustrations, many reviewers note recent updates   │ |
|  │   have improved stability                                    │ |
|  │                                                              │ |
|  │ **Representative Quotes:**                                   │ |
|  │ - "Software crashes every time I try to export..."          │ |
|  │ - "Latest update finally fixed the memory issues"            │ |
|  └─────────────────────────────────────────────────────────────┘ |
|                                                                   |
+------------------------------------------------------------------+
```

### Implementation

```python
# app.py

import streamlit as st
from src.pipeline import SurveyAnalysisPipeline
from src.loaders.loader import load_reviews

# Page config
st.set_page_config(
    page_title="Survey Analysis Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("Survey Analysis Agent")
st.markdown("*Analyze product reviews using AI techniques from INFSCI2440*")

# Initialize pipeline (cached)
@st.cache_resource
def get_pipeline():
    reviews = load_reviews()
    return SurveyAnalysisPipeline.from_reviews(reviews)

pipeline = get_pipeline()

# Query input
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter your query:", placeholder="e.g., shipping problems")
with col2:
    search_button = st.button("Search", type="primary")

# Optional filters
with st.expander("Optional Filters"):
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        topic_filter = st.selectbox("Topic", ["All"] + pipeline.classifier.TOPICS)
    with filter_col2:
        rating_filter = st.selectbox("Rating", ["All", "1-2 stars", "3 stars", "4-5 stars"])
    with filter_col3:
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Negative"])

# Run pipeline on search
if search_button and query:
    with st.spinner("Analyzing..."):
        result = pipeline.run(
            query=query,
            topic_filter=None if topic_filter == "All" else topic_filter,
            rating_filter=parse_rating_filter(rating_filter),
            sentiment_filter=None if sentiment_filter == "All" else sentiment_filter.lower()
        )

    # Display results
    st.markdown("---")
    st.header("Pipeline Results")

    # 1. Query Expansion
    with st.container():
        st.subheader("1. Query Expansion (Beam Search)")
        st.markdown(f"**Original:** {result.expansion.original_query}")
        st.markdown(f"**Expanded:** {', '.join(result.expansion.expanded_terms)}")
        with st.expander("Show expansion tree"):
            for path in result.expansion.beam_paths[:5]:
                st.text(f"  {' → '.join(path['path'])} (score: {path['score']:.2f})")

    # 2. TF-IDF
    with st.container():
        st.subheader("2. TF-IDF Retrieval")
        st.markdown(f"Found **{len(result.candidate_reviews)}** candidate reviews")

    # 3. Topic Classification
    with st.container():
        st.subheader("3. Topic Classification (Naive Bayes)")
        st.markdown(f"Detected topic: **{result.bayesian_insights.topic}**")
        st.markdown(f"Filtered to **{len(result.filtered_reviews)}** relevant reviews")

    # 4. Bayesian Network
    with st.container():
        st.subheader("4. Probabilistic Insights (Bayesian Network)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "P(Negative | Topic)",
                f"{result.bayesian_insights.p_negative_given_topic:.0%}"
            )
        with col2:
            st.metric(
                "P(Low Rating | Negative)",
                f"{result.bayesian_insights.p_low_rating_given_negative:.0%}"
            )

    # 5. HMM
    with st.container():
        st.subheader("5. Sentiment Sequences (HMM)")
        if result.sentiment_sequences:
            # Show common patterns
            transitions = result.sentiment_sequences[0].transitions
            st.markdown(f"P(Negative → Positive) = {transitions.get('neg_to_pos', 0):.0%}")

    # 6. LLM Summary
    with st.container():
        st.subheader("6. Summary")
        st.markdown(result.llm_summary)

def parse_rating_filter(filter_str):
    if filter_str == "1-2 stars":
        return (1, 2)
    elif filter_str == "3 stars":
        return (3, 3)
    elif filter_str == "4-5 stars":
        return (4, 5)
    return None
```

### Running Locally
```bash
cd project
streamlit run app.py
```

### Evaluation Metrics (1-3)

1. **UI Responsiveness**: Time from search click to results displayed
   ```python
   # Target: < 10 seconds for full pipeline
   ```

2. **All Steps Visible**: Checklist that each pipeline step's output is shown

---

## Development Checklist

### Week 1-2 (Parallel Development)
- [ ] Set up development environment
- [ ] Create topic keyword lists for auto-labeling
- [ ] Auto-label training data (or manually label sample)
- [ ] Implement `TopicClassifier` class
- [ ] Write unit tests for Naive Bayes
- [ ] Create basic Streamlit app skeleton
- [ ] Test with mock pipeline data

### Week 3 (Integration)
- [ ] Train classifier on real Amazon Reviews
- [ ] Tune TF-IDF and NB parameters
- [ ] Integrate all pipeline components into UI
- [ ] Test end-to-end flow
- [ ] Fix any interface issues

### Week 4 (Polish)
- [ ] Compute evaluation metrics
- [ ] Improve UI styling
- [ ] Add error handling
- [ ] Demo preparation
- [ ] Documentation

---

## Dependencies

All dependencies are managed in `pyproject.toml`. Install with:
```bash
uv sync
```

Key dependencies for Person 2:
- `scikit-learn>=1.3` (Naive Bayes, TF-IDF)
- `streamlit>=1.30` (Web UI)
- `pandas>=2.0` (Data handling)

---

## Testing

```python
# tests/test_naive_bayes.py

def test_fit_and_predict():
    reviews = [
        Review("1", "Software crashes constantly", 1, "Bad", "A"),
        Review("2", "Easy to use interface", 5, "Good", "B"),
        Review("3", "Too expensive for features", 2, "Meh", "C"),
    ]
    labels = ["performance", "usability", "pricing"]

    classifier = TopicClassifier()
    classifier.fit(reviews, labels)

    # Test prediction
    test_reviews = [Review("4", "The app freezes all the time", 1, "Bad", "D")]
    results = classifier.predict(test_reviews)

    assert results[0].predicted_topic == "performance"
    assert results[0].confidence > 0.5

def test_filter_by_topic():
    # ... test filtering logic
```
