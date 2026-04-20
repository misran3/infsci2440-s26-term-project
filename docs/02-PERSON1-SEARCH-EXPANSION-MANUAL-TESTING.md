# Manual Testing: Search & Expansion Components

Tests that need manual verification on actual Amazon Reviews dataset.

## Quick Setup

```bash
uv run python -c "
from src.search import BeamSearchExpander, TFIDFRetriever
from src.loaders.loader import load_reviews

reviews = load_reviews(sample=True, limit=100)
print(f'Loaded {len(reviews)} reviews')

expander = BeamSearchExpander(beam_width=3, max_depth=2)
retriever = TFIDFRetriever(reviews)
retriever.fit()
"
```

## Test 1: Expansion Improves Recall

**Why manual:** Automated tests use tiny mock corpora. Need to verify expansion helps on real data.

```bash
uv run python -c "
from src.search import BeamSearchExpander, TFIDFRetriever
from src.loaders.loader import load_reviews

reviews = load_reviews(sample=True, limit=100)
expander = BeamSearchExpander(beam_width=3, max_depth=2)
retriever = TFIDFRetriever(reviews)
retriever.fit()

for query in ['good', 'problem', 'easy']:
    expansion = expander.expand(query)
    base = retriever.retrieve([query], top_k=50)
    expanded = retriever.retrieve(expansion.expanded_terms, top_k=50)
    
    print(f'\"{query}\": {len(base)} -> {len(expanded)} results (+{len(expanded)-len(base)})')
"
```

**Expected:** Expansion should increase or maintain recall for most queries.

**Actual results on sample:**
- "good": 12 -> 16 results (+4)
- "problem": 4 -> 6 results (+2)
- "easy": 5 -> 5 results (+0)

## Test 2: Expansion Quality (Semantic Relevance)

**Why manual:** WordNet returns all senses of a word. Need to check if expansions make sense for software reviews.

```bash
uv run python -c "
from src.search import BeamSearchExpander

expander = BeamSearchExpander(beam_width=3, max_depth=2)

for query in ['bug', 'crash', 'good', 'easy']:
    result = expander.expand(query)
    print(f'\"{query}\" expands to: {sorted(result.expanded_terms)[:10]}')
"
```

**Check for:**
- "bug" -> Should include "problem", "issue", "error". May also include insect-related terms (WordNet limitation).
- "crash" -> Should include "freeze", "fail". May include unrelated sound terms.
- "good" -> Should include "great", "excellent", "well".
- "easy" -> Should include "simple", but may include unrelated terms like "aristocratical".

**Known limitation:** WordNet expands based on all word senses, not domain-specific (software reviews). Some irrelevant synonyms are expected.

## Test 3: Retrieved Reviews Are Relevant

**Why manual:** Need human judgment on whether retrieved reviews match query intent.

```bash
uv run python -c "
from src.search import BeamSearchExpander, TFIDFRetriever
from src.loaders.loader import load_reviews

reviews = load_reviews(sample=True, limit=100)
expander = BeamSearchExpander(beam_width=3, max_depth=2)
retriever = TFIDFRetriever(reviews)
retriever.fit()

query = 'problem'
expansion = expander.expand(query)
results = retriever.retrieve(expansion.expanded_terms, top_k=5)

print(f'Query: \"{query}\"')
print(f'Expanded: {expansion.expanded_terms[:8]}...')
print(f'\\nTop 5 results:')
for i, r in enumerate(results, 1):
    print(f'{i}. [{r.tfidf_score:.3f}] {r.text[:100]}...')
"
```

**Check:** Do top results discuss problems/issues? Or are they false positives from unrelated synonym matches?

## Test 4: Score Ordering Makes Sense

**Why manual:** Verify that higher-scored reviews are more relevant than lower-scored ones.

```bash
uv run python -c "
from src.search import TFIDFRetriever
from src.loaders.loader import load_reviews

reviews = load_reviews(sample=True, limit=100)
retriever = TFIDFRetriever(reviews)
retriever.fit()

results = retriever.retrieve(['game', 'fun'], top_k=10)

print('Results for [\"game\", \"fun\"]:')
for i, r in enumerate(results, 1):
    has_game = 'game' in r.text.lower()
    has_fun = 'fun' in r.text.lower()
    print(f'{i}. [{r.tfidf_score:.3f}] game={has_game} fun={has_fun} | {r.text[:60]}...')
"
```

**Check:** Reviews with both terms should score higher than reviews with one term.

## Test 5: Full Pipeline Integration

**Why manual:** End-to-end verification that components work together on real data.

```bash
uv run python -c "
from src.search import BeamSearchExpander, TFIDFRetriever
from src.loaders.loader import load_reviews

# Load real data
reviews = load_reviews(sample=True, limit=100)
print(f'Dataset: {len(reviews)} reviews')

# Initialize components
expander = BeamSearchExpander(beam_width=3, max_depth=2)
retriever = TFIDFRetriever(reviews)
retriever.fit()

# Simulate user query
user_query = 'good quality'

# Step 1: Expand
expansion = expander.expand(user_query)
print(f'\\nQuery: \"{user_query}\"')
print(f'Expanded to {len(expansion.expanded_terms)} terms')

# Step 2: Retrieve
results = retriever.retrieve(expansion.expanded_terms, top_k=5)
print(f'\\nTop 5 results:')
for r in results:
    print(f'  [{r.rating}★] {r.text[:80]}...')
"
```

**Check:**
1. No errors in pipeline
2. Results are relevant to "good quality"
3. Mix of ratings (not all 5★ or all 1★)

## Summary: What Needs Manual Verification

| Test | Automated? | Manual Check |
|------|-----------|--------------|
| Expansion returns terms | Yes | Are terms semantically relevant? |
| Recall improves | Yes | By how much on real data? |
| Results sorted by score | Yes | Does ordering make sense? |
| Pipeline works | Yes | Results relevant to intent? |

**Bottom line:** Automated tests verify correctness of implementation. Manual tests verify quality/usefulness on real data.
