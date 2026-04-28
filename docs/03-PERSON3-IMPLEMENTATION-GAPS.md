# Person 3 Implementation Gaps

**Branch:** `raphael_update`  
**Reviewed:** 2026-04-27  
**Status:** Needs fixes before merge

---

## Gap Summary

| # | Component | Gap | Severity | Effort to Fix |
|---|-----------|-----|----------|---------------|
| 1 | Integration | Trained models not loaded at runtime | **CRITICAL** | Low |
| 2 | Bayesian Network | Not using pgmpy (course requirement) | **CRITICAL** | High |
| 3 | HMM Sentiment | Not using hmmlearn (course requirement) | **CRITICAL** | High |
| 4 | LLM Summarizer | Missing `key_themes` and `representative_quotes` fields | Medium | Low |
| 5 | LLM Summarizer | No review text passed to LLM | Medium | Low |
| 6 | Tests | No probability correctness tests | Medium | Medium |
| 7 | HMM | Transition probabilities don't sum to 1 per row | Low | Low |

---

## Gap 1: Trained Models Not Loaded at Runtime (CRITICAL, Low Effort)

**Problem:** `scripts/train_models.py` saves Bayesian and HMM state, but `app.py` creates fresh instances that ignore the saved models.

**Current behavior:**
```python
# app.py lines 45-47
bayesian_net = BayesianNetwork()  # Fresh instance, empty _fitted_reviews
hmm = HMMSentiment()              # Fresh instance, default transitions
```

**Saved artifacts:**
- `models/bayesian_network.pkl` → `{"fitted_reviews": [...]}`
- `models/hmm_model.pkl` → `{"global_transitions": {...}}`

**Fix required:**
1. Add `load()` classmethod to `BayesianNetwork`
2. Add `load()` classmethod to `HMMSentiment`
3. Update `app.py` to load trained models

---

## Gap 2: Bayesian Network Not Using pgmpy (CRITICAL, High Effort)

**Spec requirement (docs/04-PERSON3-PROBABILISTIC-LLM.md lines 59-182):**
```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

self.model = BayesianNetwork([
    ('topic', 'sentiment'),
    ('sentiment', 'rating_category')
])
self.model.fit(df, estimator=MaximumLikelihoodEstimator)
self.inference = VariableElimination(self.model)
```

**Current implementation:** Simple frequency counting with Laplace smoothing. No DAG, no CPTs, no inference engine.

**Course alignment:** Units 4-5 require demonstrating Bayesian Network concepts (graphical models, conditional independence, variable elimination).

**Options:**
- A) Full rewrite with pgmpy (matches spec exactly)
- B) Add pgmpy wrapper that delegates to existing logic (minimal change, partial compliance)
- C) Keep as-is but rename class and update documentation (honest about deviation)

---

## Gap 3: HMM Not Using hmmlearn (CRITICAL, High Effort)

**Spec requirement (docs/04-PERSON3-PROBABILISTIC-LLM.md lines 272-412):**
```python
from hmmlearn import hmm

self.model = hmm.CategoricalHMM(n_components=3, n_iter=100)
self.model.fit(X, lengths)
_, state_sequence = self.model.decode(obs)  # Viterbi
```

**Current implementation:** VADER labels each sentence directly. States are OBSERVED, not hidden. This is a Markov chain on observed states, not an HMM.

**Course alignment:** Units 4-5 require demonstrating HMM concepts (hidden states, emission matrix, Viterbi decoding).

**Options:**
- A) Full rewrite with hmmlearn (matches spec exactly)
- B) Hybrid: use VADER scores as observations, HMM infers hidden sentiment states
- C) Keep as-is but rename to `SentimentMarkovChain` (honest about what it is)

---

## Gap 4: LLM Missing Structured Output Fields (Medium, Low Effort)

**Spec requirement:**
```python
class SummaryOutput(BaseModel):
    summary: str
    key_themes: list[str]
    representative_quotes: list[str]
```

**Current implementation:**
```python
class SummaryOutput(BaseModel):
    summary: str  # Only this field
```

**Fix:** Add the missing fields to `SummaryOutput`.

---

## Gap 5: No Review Text Passed to LLM (Medium, Low Effort)

**Problem:** The LLM prompt only contains statistics. Without actual review text, it cannot generate representative quotes.

**Current prompt:**
```
Review count: 42
Topic: performance
P(positive|topic): 0.25
...
```

**Fix:** Add `_format_reviews()` method and include top 10 review texts in prompt.

---

## Gap 6: No Probability Correctness Tests (Medium, Medium Effort)

**Problem:** Tests only check `0 <= p <= 1`, not that values are mathematically correct.

**Example weak test:**
```python
assert 0.0 <= insights.p_positive_given_topic <= 1.0  # Passes for random.random()
```

**Fix:** Add tests with known input distributions that verify specific probability values.

---

## Gap 7: HMM Transitions Don't Sum to 1 (Low, Low Effort)

**Problem:** `_compute_transitions()` divides by total count, not per-source-state count.

**Current (buggy):**
```python
total = sum(counts.values())
return {k: counts[k] / total for k in keys}
```

**Should be:** Normalize each row independently so `pos_to_*` sums to 1, `neg_to_*` sums to 1, etc.

---

## Recommended Fix Order

1. **Gap 1** (Integration) - Quick win, makes training useful
2. **Gap 7** (Transition bug) - Small fix, improves correctness
3. **Gap 4 & 5** (LLM fields) - Small changes, improves compliance
4. **Gap 2 & 3** (pgmpy/hmmlearn) - Discuss options with team first
5. **Gap 6** (Tests) - Can be done incrementally

---

## Decision Needed: Gaps 2 & 3

The Bayesian Network and HMM implementations fundamentally deviate from the spec. Before fixing, the team should decide:

1. **Is pgmpy/hmmlearn required for grading?** If the rubric explicitly checks for these libraries, they must be used.

2. **Can we keep existing logic as fallback?** A hybrid approach could use pgmpy/hmmlearn when available, falling back to current implementation.

3. **Is renaming acceptable?** If the current approach is intentional, classes should be renamed to avoid claiming to be something they're not.
