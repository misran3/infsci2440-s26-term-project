# Person 1: Search & Expansion

## Ownership
- **Beam Search Query Expansion** (AI technique - Course Unit 2)
- **TF-IDF Retrieval** (ML component)

---

## Component 1: Beam Search Query Expansion

### What It Does
Expands user queries by finding related terms using beam search over WordNet synonyms. This makes the query expansion process **transparent** — users can see exactly how their query was expanded.

### Course Alignment
**Unit 2: Agents for Problem Solving: Classical and Beyond**
- Beam search is a heuristic search algorithm
- Explores a graph of word relationships (synonyms)
- Keeps top-k candidates at each step (beam width)

### Algorithm

```
BEAM SEARCH QUERY EXPANSION

Input: query = "shipping problems"
       beam_width = 3
       max_depth = 2

Step 0: Initial beam
  candidates = ["shipping problems"]

Step 1: Expand each candidate with synonyms
  "shipping" synonyms: ["transport", "delivery", "freight"]
  "problems" synonyms: ["issues", "difficulties", "troubles"]

  Generate combinations:
    "shipping issues"      score: 0.92
    "shipping difficulties" score: 0.88
    "delivery problems"    score: 0.85
    "transport issues"     score: 0.80
    ...

  Keep top-3 (beam_width):
    beam = ["shipping issues", "shipping difficulties", "delivery problems"]

Step 2: Expand again (max_depth)
  ... continue expansion ...

Output:
  expanded_terms = ["shipping", "problems", "issues", "delivery",
                   "difficulties", "transport", "troubles"]
  beam_paths = visualization data
```

### Implementation

```python
# src/search/beam_search.py

from nltk.corpus import wordnet
from dataclasses import dataclass
from typing import List, Dict
import nltk

# Download WordNet if not present
nltk.download('wordnet', quiet=True)

@dataclass
class QueryExpansion:
    original_query: str
    expanded_terms: List[str]
    beam_paths: List[Dict]

class BeamSearchExpander:
    def __init__(self, beam_width: int = 3, max_depth: int = 2):
        self.beam_width = beam_width
        self.max_depth = max_depth

    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name().lower())
        return list(synonyms)[:5]  # Limit to top 5

    def score_expansion(self, original: str, expanded: str) -> float:
        """Score an expansion based on semantic similarity."""
        # Simple scoring: prefer shorter edits, common synonyms
        # Could be enhanced with word embeddings
        original_words = set(original.lower().split())
        expanded_words = set(expanded.lower().split())
        overlap = len(original_words & expanded_words)
        return overlap / max(len(expanded_words), 1) + 0.1

    def expand(self, query: str) -> QueryExpansion:
        """Expand query using beam search."""
        words = query.lower().split()
        all_terms = set(words)
        beam_paths = []

        # Current beam of (expansion_string, score, path)
        beam = [(query, 1.0, [query])]

        for depth in range(self.max_depth):
            candidates = []

            for current, score, path in beam:
                current_words = current.split()

                for i, word in enumerate(current_words):
                    synonyms = self.get_synonyms(word)

                    for syn in synonyms:
                        # Create new expansion by replacing word
                        new_words = current_words.copy()
                        new_words[i] = syn
                        new_expansion = ' '.join(new_words)

                        new_score = self.score_expansion(query, new_expansion)
                        new_path = path + [new_expansion]

                        candidates.append((new_expansion, new_score, new_path))
                        all_terms.add(syn)

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.beam_width]

            # Record paths for visualization
            for exp, score, path in beam:
                beam_paths.append({"path": path, "score": score})

        return QueryExpansion(
            original_query=query,
            expanded_terms=list(all_terms),
            beam_paths=beam_paths[:10]  # Top 10 paths for UI
        )
```

### Evaluation Metrics (1-3)

1. **Expansion Recall**: Do expanded terms retrieve more relevant reviews?
   ```python
   # Compare: reviews found with original query vs expanded query
   recall_improvement = len(expanded_results) / len(original_results)
   ```

2. **Synonym Quality**: Are expanded terms semantically related?
   ```python
   # Manual check on sample of 20 expansions
   # Count: relevant synonyms / total synonyms generated
   ```

### UI Output Format
```
Query Expansion (Beam Search):
  Original: "shipping problems"
  Expanded: shipping, delivery, transport, problems, issues, difficulties

  Expansion Tree:
  ├── "shipping problems" (1.0)
  │   ├── "shipping issues" (0.92)
  │   ├── "delivery problems" (0.85)
  │   └── "transport problems" (0.80)
```

---

## Component 2: TF-IDF Retrieval

### What It Does
Retrieves the most relevant reviews from the corpus based on expanded query terms using TF-IDF (Term Frequency-Inverse Document Frequency).

### Why TF-IDF (vs. embeddings)?
- **Interpretable**: Can show which terms matched
- **Fast**: No neural network inference needed
- **Course-aligned**: Statistical NLP method
- **Differentiates from existing project**: They use vector embeddings

### Implementation

```python
# src/search/tfidf_retriever.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from src.data.structures import Review

class TFIDFRetriever:
    def __init__(self, corpus: List[Review]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        self.tfidf_matrix = None

    def fit(self) -> None:
        """Fit TF-IDF vectorizer on corpus."""
        texts = [r.text for r in self.corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(
        self,
        expanded_terms: List[str],
        top_k: int = 500
    ) -> List[Review]:
        """Retrieve top-k reviews matching expanded terms."""
        # Join expanded terms into query
        query = ' '.join(expanded_terms)
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return reviews with similarity scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if there's some match
                review = self.corpus[idx]
                review.tfidf_score = similarities[idx]
                results.append(review)

        return results

    def get_matching_terms(self, review: Review, expanded_terms: List[str]) -> List[str]:
        """Get which expanded terms appear in a review (for UI transparency)."""
        review_words = set(review.text.lower().split())
        return [term for term in expanded_terms if term.lower() in review_words]
```

### Model Persistence

```python
import joblib

# Save fitted vectorizer
def save_model(retriever: TFIDFRetriever, path: str = "models/tfidf_vectorizer.pkl"):
    joblib.dump({
        'vectorizer': retriever.vectorizer,
        'tfidf_matrix': retriever.tfidf_matrix,
        'corpus_ids': [r.review_id for r in retriever.corpus]
    }, path)

# Load fitted vectorizer
def load_model(path: str = "models/tfidf_vectorizer.pkl") -> dict:
    return joblib.load(path)
```

**When to retrain:**
- When corpus changes significantly (>10% new reviews)
- When TF-IDF parameters are tuned

### Evaluation Metrics (1-3)

1. **Retrieval Precision@k**: What fraction of top-k results are relevant?
   ```python
   # Manual annotation of 50 results
   precision = relevant_in_top_k / k
   ```

2. **Term Coverage**: Do retrieved reviews contain expanded terms?
   ```python
   coverage = reviews_with_any_term / total_retrieved
   ```

### UI Output Format
```
TF-IDF Retrieval:
  Found 487 candidate reviews
  Top matching terms: "shipping" (312), "delivery" (98), "issues" (77)

  Sample matches:
  [Score: 0.82] "The shipping was incredibly slow..."
  [Score: 0.79] "Had issues with delivery timing..."
```

---

## Development Checklist

### Week 1-2 (Parallel Development)
- [ ] Set up development environment
- [ ] Implement `BeamSearchExpander` class
- [ ] Write unit tests for beam search
- [ ] Implement `TFIDFRetriever` class
- [ ] Write unit tests for TF-IDF
- [ ] Test with mock data

### Week 3 (Integration)
- [ ] Load real Amazon Reviews dataset
- [ ] Fit TF-IDF on full corpus
- [ ] Test beam search + TF-IDF together
- [ ] Tune beam_width and max_depth
- [ ] Tune TF-IDF parameters (max_features, ngrams)

### Week 4 (Polish)
- [ ] Compute evaluation metrics
- [ ] Prepare UI output format for Person 2
- [ ] Document any interface changes
- [ ] Demo preparation

---

## Dependencies

All dependencies are managed in `pyproject.toml`. Install with:
```bash
uv sync
```

Key dependencies for Person 1:
- `nltk>=3.8` (WordNet synonyms)
- `scikit-learn>=1.3` (TF-IDF vectorizer)

---

## Testing

```python
# tests/test_beam_search.py

def test_expand_returns_synonyms():
    expander = BeamSearchExpander()
    result = expander.expand("fast shipping")

    assert "fast" in result.expanded_terms
    assert "shipping" in result.expanded_terms
    # Should have synonyms
    assert len(result.expanded_terms) > 2

def test_beam_paths_tracked():
    expander = BeamSearchExpander(beam_width=2, max_depth=1)
    result = expander.expand("good product")

    assert len(result.beam_paths) > 0
    assert "path" in result.beam_paths[0]
    assert "score" in result.beam_paths[0]


# tests/test_tfidf.py

def test_retrieve_returns_relevant():
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "product quality is poor", 1, "Bad", "B"),
        Review("3", "delivery took forever", 2, "Slow", "C"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping", "delivery"], top_k=2)

    # Should return reviews about shipping/delivery, not quality
    assert len(results) == 2
    assert any("shipping" in r.text.lower() for r in results)
```
