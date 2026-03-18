# Project Overview: Intelligent Survey Response Analysis

## Project Title
**An Intelligent Agent for Probabilistic and Context-Aware Analysis of Free-Text Survey Responses**

## One-Line Summary
A web-based query interface that uses classical AI techniques (Beam Search, Naive Bayes, Bayesian Networks, HMM) to analyze Amazon product reviews, with LLM-powered summarization via Pydantic AI (provider-agnostic).

---

## Architecture Overview

```
USER QUERY + OPTIONAL FILTERS
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  0. PRE-FILTER (if user specifies topic/rating/sentiment)       │
│     Narrow corpus before AI pipeline runs                       │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. BEAM SEARCH QUERY EXPANSION              [Person 1]         │
│     Libraries: nltk (WordNet), custom beam search               │
│     "shipping" → ["shipping", "delivery", "arrived", "package"] │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. TF-IDF RETRIEVAL                         [Person 1]         │
│     Libraries: scikit-learn (TfidfVectorizer)                   │
│     Search corpus with expanded terms                           │
│     Output: ~500-1000 candidate reviews                         │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. NAIVE BAYES TOPIC CLASSIFICATION         [Person 2]         │
│     Libraries: scikit-learn (MultinomialNB)                     │
│     Filter to reviews matching target topic                     │
│     Output: ~100-200 topic-relevant reviews                     │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. BAYESIAN NETWORK REASONING               [Person 3]         │
│     Libraries: pgmpy                                            │
│     Compute P(Sentiment|Topic), P(Rating|Topic,Sentiment)       │
│     Output: Probabilistic insights                              │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. HMM SENTIMENT SEQUENCE ANALYSIS          [Person 3]         │
│     Libraries: hmmlearn or pomegranate                          │
│     Analyze sentiment flow within reviews (sentence-level)      │
│     Output: Sentiment patterns, transition probabilities        │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. LLM SUMMARIZATION                        [Person 3]         │
│     Libraries: anthropic (Claude API) or openai                 │
│     Input: Filtered reviews + probabilistic insights            │
│     Output: Natural language summary for user                   │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. STREAMLIT UI                             [Person 2]         │
│     Display all pipeline outputs to user                        │
│     Show transparency: expansion steps, classifications, etc.   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Course Alignment

| AI Technique | Course Unit | Owner |
|--------------|-------------|-------|
| Beam Search | Unit 2: Agents for Problem Solving | Person 1 |
| Naive Bayes | Unit 7: Statistical Methods | Person 2 |
| Bayesian Network | Units 4-5: Uncertain Knowledge | Person 3 |
| Hidden Markov Model | Units 4-5: Uncertain Knowledge | Person 3 |

---

## Team Task Ownership (2-2-2 Split)

### Person 1: Search & Expansion
- **Beam Search** - Query expansion using WordNet synonyms
- **TF-IDF Retrieval** - Retrieve candidate reviews from corpus

### Person 2: Classification & UI
- **Naive Bayes** - Topic classification of reviews
- **Streamlit UI** - Web interface integrating all components

### Person 3: Probabilistic & LLM
- **Bayesian Network** - Probabilistic reasoning across topics/sentiment/rating
- **HMM** - Sentiment sequence analysis within reviews
- **LLM Integration** - Pydantic AI for final summarization (provider-agnostic)

---

## Scope Definition

### IN SCOPE (Required)

| Component | Description | Libraries |
|-----------|-------------|-----------|
| Beam Search Query Expansion | Expand user queries using synonyms with visible expansion tree | `nltk` (WordNet) |
| TF-IDF Retrieval | Retrieve relevant reviews using term frequency | `scikit-learn` |
| Naive Bayes Classifier | Classify reviews into topics (Shipping, Quality, Service, etc.) | `scikit-learn` |
| Bayesian Network | Model relationships: Topic → Sentiment → Rating | `pgmpy` |
| HMM Sentiment Sequences | Analyze sentiment transitions within reviews | `hmmlearn` |
| LLM Summarization | Generate natural language summary of findings | `pydantic-ai` (provider-agnostic) |
| Streamlit UI | Simple web interface with query input and results display | `streamlit` |
| Basic Evaluation | 1-3 metrics per component demonstrating it works | - |

### OPTIONAL (Nice-to-Have)

| Component | Description | Effort |
|-----------|-------------|--------|
| User Filters | Topic, rating, sentiment dropdowns in UI | S |
| Visualization | Charts showing topic distribution, sentiment flow | M |
| Export Results | Download results as CSV/PDF | S |
| Multi-dataset Support | Allow uploading custom survey CSVs | M |
| Caching | Cache TF-IDF vectors and model predictions | S |

### OUT OF SCOPE (Explicitly Excluded)

| Component | Reason |
|-----------|--------|
| Vector Embeddings | Already done by existing survey analysis solutions |
| LLM-based Query Expansion | Use LLM to generate query variations (if time permits) |
| Cloud Infrastructure | Not needed for course project; local/Streamlit cloud sufficient |
| User Authentication | Overkill for demo; Focus is on pipeline demo; Streamlit handles basic access |
| Production Deployment | Course project scope is demo-quality MVP |
| Multi-turn Session Tracking | Amazon Reviews data doesn't support this (single reviews, not conversations) |

---

## Dataset

**Amazon Reviews 2023 - Software Category**
- Source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Focus: Software product reviews (subset of full dataset)
- Fields used: `text` (review), `rating`, `title`, `parent_asin`

### Why Software Category?
- Manageable size for course project
- Clear topic categories (bugs, features, pricing, support)
- Mix of positive/negative reviews
- Aligns with proposal's stated scope

---

## Training Sequence (MUST FOLLOW)

```
STEP 1: DATA PREP (Blocker for all)
   └── Run scripts: download → preprocess → tokenize → sample → label
   └── Output: amazon_reviews_software.csv, labeled_reviews.csv

STEP 2: PARALLEL TRAINING
   ├── TF-IDF: fit on amazon_reviews_software.csv
   ├── Naive Bayes: fit on labeled_reviews.csv
   └── HMM: fit on amazon_reviews_software.csv (uses sentences)

STEP 3: SEQUENTIAL TRAINING (after Step 2)
   └── Bayesian Network: fit on NB-classified reviews (needs topics from Step 2)
```

### Training Dependencies

| Component | Input Data | Blocked By |
|-----------|------------|------------|
| TF-IDF | `amazon_reviews_software.csv` | Data Prep |
| Naive Bayes | `labeled_reviews.csv` | Data Prep |
| HMM | `amazon_reviews_software.csv` | Data Prep |
| Bayesian Net | Reviews with `topic` field | Naive Bayes |

---

## Development Phases

### Phase 1: Parallel Development (Week 1-2)
- Define shared interfaces (see `01-SHARED-INTERFACES.md`)
- Each person builds their AI component independently
- Use mock data for dependencies
- Unit tests with mocks

### Phase 2: Integration (Week 3)
- Replace mocks with real components
- Integration tests
- Debug interface issues

### Phase 3: UI & Polish (Week 4)
- Streamlit UI built
- End-to-end testing
- Demo preparation
- Documentation

---

## Addressing Instructor Feedback

### "Transparent query expansion is unclear"
**Answer:** "Transparent" means the expansion process is *visible* to users. Unlike LLM-based expansion (black box), Beam Search shows exactly how queries expand:
```
"shipping"
  → beam_1: "shipping delivery" (score: 0.9)
  → beam_2: "shipping package" (score: 0.85)
  → beam_3: "shipping arrived" (score: 0.8)
```
Users can see the expansion tree in the UI.

### "Unclear about tracking user intent with HMM and whether data supports it"
**Answer:** We've redesigned the HMM application. Instead of tracking intent across multi-turn sessions (which the data doesn't support), we use HMM to analyze **sentiment sequences within individual reviews**:
```
Review: "Product arrived fast. [+] But packaging was damaged. [-] Support helped. [+]"
HMM models: P(Negative | Positive), P(Positive | Negative)
```
This works with the Amazon Reviews data we have.

---

## Differentiation from Existing Projects

Some existing projects uses:
- LLM-based query rewriting (LLM generates 10 query variations)
- Vector embeddings for semantic search
- LLM-based analysis (LLM summarizes results)
- Cloud infrastructure

**Our project is different:**
- Classical AI techniques from the course curriculum
- Interpretable/transparent algorithms (not black boxes)
- Educational focus: demonstrate mastery of course material
- Each AI step is visible and explainable

We take **inspiration** from the existing projects:
- Data handling patterns (CSV processing)
- Pipeline architecture (query → filter → analyze → present)
- UI patterns (query input, results display)

---

## Success Criteria

1. **Functional Demo**: User can enter a query and get meaningful results
2. **Transparency**: Each AI step's output is visible in the UI
3. **Course Alignment**: All 4 required AI techniques are implemented
4. **Task Ownership**: Each team member owns their components end-to-end
5. **Basic Metrics**: Each component has 1-3 evaluation metrics showing it works

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Review Mode | SCOPE EXPANSION | Greenfield project, want impressive course project |
| HMM Application | Sentiment sequences within reviews | Original "multi-turn intent" doesn't fit data |
| LLM Usage | Hybrid: Classical AI + LLM summary | Classical AI filters, LLM makes readable |
| LLM Framework | Pydantic AI (provider-agnostic) | Can swap OpenAI/Anthropic/Gemini easily |
| UI Framework | Streamlit | Simple demo UI, fast to build |
| User Filters | Query + optional filters | Users can filter or let AI auto-detect |
| Task Split | 2-2-2 balanced | Each person: 1 AI technique + 1 support |
| Parallel Dev | Interface contracts + mocks | Unblocks parallel work |
| Error Handling | Graceful degradation | Show partial results if components fail |
| Test Coverage | Core unit tests only (~15) | MVP focus, not comprehensive |
| Security | Skip prompt injection sanitization | Not priority for academic project |

---

## Project File Structure

```
survey-analysis-agent/
│
├── .env                          # API keys (never commit)
├── .env.example                  # Template for .env
├── .gitignore
├── pyproject.toml                # Dependencies and project config
├── uv.lock                       # Lock file (commit this)
├── app.py                        # Streamlit web interface
├── README.md
│
├── data/
│   ├── amazon_reviews_software.csv    # Full dataset
│   └── sample_reviews.csv             # Small test dataset (~100 reviews)
│
├── models/                       # Trained model files (.pkl)
│   ├── naive_bayes.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── bayesian_network.pkl
│   └── hmm_model.pkl
│
├── src/
│   ├── __init__.py
│   │
│   ├── loaders/                  # Data loading and structures
│   │   ├── __init__.py
│   │   ├── loader.py             # Load reviews from CSV files
│   │   └── structures.py         # Data structures for the pipeline
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── logger.py             # Centralized logging
│   │   └── errors.py             # Custom exceptions
│   │
│   ├── search/                   # Query expansion & retrieval [Person 1]
│   │   ├── __init__.py
│   │   ├── beam_search.py        # BeamSearchExpander class
│   │   └── tfidf_retriever.py    # TFIDFRetriever class
│   │
│   ├── classification/           # Topic classification & UI [Person 2]
│   │   ├── __init__.py
│   │   └── naive_bayes.py        # TopicClassifier class
│   │
│   ├── reasoning/                # Probabilistic reasoning & LLM [Person 3]
│   │   ├── __init__.py
│   │   ├── bayesian_network.py   # ReviewBayesianNetwork class
│   │   ├── hmm_sentiment.py      # SentimentHMM class
│   │   └── llm_summarizer.py     # LLMSummarizer class (Pydantic AI)
│   │
│   ├── pipeline.py               # SurveyAnalysisPipeline orchestrator
│   └── mocks.py                  # All mock implementations for testing
│
├── scripts/
│   ├── download_data.py          # Download Amazon Reviews dataset
│   ├── preprocess_data.py        # Clean and preprocess reviews
│   ├── tokenize_sentences.py     # Pre-tokenize reviews into sentences
│   ├── create_sample.py          # Create balanced sample dataset
│   ├── label_training_data.py    # Auto-label reviews for training
│   └── train_models.py           # Train and save all models
│
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── test_beam_search.py
│   ├── test_tfidf.py
│   ├── test_naive_bayes.py
│   ├── test_bayesian.py
│   ├── test_hmm.py
│   ├── test_llm.py
│   └── test_pipeline.py          # Integration tests
│
└── docs/                         # Design documentation
    ├── 00-PROJECT-OVERVIEW.md
    ├── 01-SHARED-INTERFACES.md
    ├── 02-PERSON1-SEARCH-EXPANSION.md
    ├── 03-PERSON2-CLASSIFICATION-UI.md
    ├── 04-PERSON3-PROBABILISTIC-LLM.md
    ├── 05-ERROR-HANDLING.md
    ├── 06-TEST-SPECIFICATION.md
    └── 07-DEPLOYMENT-GUIDE.md
```

### Create Structure Command

```bash
mkdir -p src/{loaders,utils,search,classification,reasoning} tests scripts models data docs
touch src/__init__.py src/loaders/__init__.py src/utils/__init__.py
touch src/search/__init__.py src/classification/__init__.py src/reasoning/__init__.py
```

---

## Related Documents

- `01-DATA-PREP.md` - Data acquisition, preprocessing, and labeling (BLOCKER)
- `01-SHARED-INTERFACES.md` - Interface contracts between components
- `02-PERSON1-SEARCH-EXPANSION.md` - Beam Search + TF-IDF design
- `03-PERSON2-CLASSIFICATION-UI.md` - Naive Bayes + Streamlit design
- `04-PERSON3-PROBABILISTIC-LLM.md` - Bayesian Net + HMM + LLM design
- `05-ERROR-HANDLING.md` - Error handling strategy and patterns
- `06-TEST-SPECIFICATION.md` - Test coverage plan and test cases
- `07-DEPLOYMENT-GUIDE.md` - Local setup and Streamlit Cloud deployment
