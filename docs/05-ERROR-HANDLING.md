# Error Handling Strategy

## Design Decision
**Graceful degradation** — each component handles errors gracefully, UI shows partial results if later stages fail.

## Error Registry

| Component | Exception | Rescued | Action | User Sees |
|-----------|-----------|---------|--------|-----------|
| Beam Search | `LookupError` (NLTK) | Yes | Auto-download WordNet | "Downloading..." |
| Beam Search | `ValueError` (empty) | Yes | Return original query | Query used as-is |
| TF-IDF | `MemoryError` | Yes | Log, raise UserError | "Dataset too large" |
| TF-IDF | `RuntimeError` (not fit) | Yes | Auto-fit | Slight delay |
| Naive Bayes | `ValueError` (no data) | Yes | Skip classification | All reviews shown |
| Naive Bayes | Filter to 0 results | Yes | Return all TF-IDF candidates | All candidates shown with note |
| Bayesian Net | `pgmpy` errors | Yes | Return default probs | "Insights unavailable" |
| HMM | `ValueError` (no sequences) | Yes | Skip HMM analysis | "Patterns unavailable" |
| LLM (Pydantic AI) | Provider errors | Yes | Use fallback summary | Basic summary shown |
| LLM (Pydantic AI) | Rate limit | Yes | Retry 3x with backoff | Delay, then result |

## Implementation Pattern

```python
# src/utils/errors.py

class UserError(Exception):
    """Error with user-friendly message."""
    pass

class ComponentError(Exception):
    """Error from a pipeline component."""
    def __init__(self, component: str, message: str, original: Exception = None):
        self.component = component
        self.message = message
        self.original = original
        super().__init__(f"{component}: {message}")
```

```python
# Example usage in Bayesian Network
def fit(self, reviews: List[Review]) -> None:
    try:
        df = self._prepare_data(reviews)
        self.model.fit(df, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)
        self.is_fitted = True
    except Exception as e:
        logger.error(f"Bayesian Network fit failed: {e}")
        # Set fallback mode
        self.is_fitted = False
        self.fallback_mode = True

def query(self, topic: str) -> BayesianInsights:
    if self.fallback_mode:
        # Return uniform/default probabilities
        return BayesianInsights(
            topic=topic,
            p_negative_given_topic=0.5,
            p_positive_given_topic=0.5,
            p_low_rating_given_negative=0.5,
            p_high_rating_given_positive=0.5
        )
    # ... normal inference
```

## Logging Strategy

```python
# src/utils/logger.py
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

    return logger

# Usage in components
logger = setup_logger(__name__)
logger.info("TF-IDF fit complete: %d documents", len(corpus))
logger.warning("No synonyms found for '%s'", word)
logger.error("LLM API failed: %s", str(e))
```

## LLM Summarizer with Pydantic AI

Since we're using Pydantic AI (provider-agnostic), error handling changes:

```python
# src/reasoning/llm_summarizer.py
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetryError

class LLMSummarizer:
    def __init__(self, model: str = "openai:gpt-4o-mini"):
        # Pydantic AI supports: openai, anthropic, gemini, groq, etc.
        self.agent = Agent(model)
        self.fallback_model = Agent("openai:gpt-3.5-turbo")  # Cheaper fallback

    def summarize(self, result: PipelineResult) -> str:
        try:
            response = self.agent.run_sync(
                self._build_prompt(result)
            )
            return response.data
        except ModelRetryError as e:
            logger.warning(f"Primary model failed, trying fallback: {e}")
            try:
                response = self.fallback_model.run_sync(
                    self._build_prompt(result)
                )
                return response.data
            except Exception as e:
                logger.error(f"All LLM models failed: {e}")
                return self._generate_fallback_summary(result)

    def _generate_fallback_summary(self, result: PipelineResult) -> str:
        """Generate basic summary without LLM."""
        bi = result.bayesian_insights
        n = len(result.filtered_reviews)
        return (
            f"Found {n} reviews about {bi.topic}. "
            f"{bi.p_negative_given_topic:.0%} are negative. "
            f"(AI summary unavailable)"
        )
```
