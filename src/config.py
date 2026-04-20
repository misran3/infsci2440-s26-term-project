"""Centralized configuration for data preparation pipeline."""

from pathlib import Path
from dataclasses import dataclass, field

# Project root (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class DataConfig:
    """Data file paths."""

    raw_reviews: Path = DATA_DIR / "amazon_reviews_software_raw.csv"
    clean_reviews: Path = DATA_DIR / "amazon_reviews_software_clean.csv"
    main_corpus: Path = DATA_DIR / "amazon_reviews_software.csv"
    sample_reviews: Path = DATA_DIR / "sample_reviews.csv"
    labeled_reviews: Path = DATA_DIR / "labeled_reviews.csv"
    curated_labels: Path = DATA_DIR / "curated_labels.csv"


@dataclass(frozen=True)
class DatasetConfig:
    """HuggingFace dataset settings."""

    repo_id: str = "McAuley-Lab/Amazon-Reviews-2023"
    filename: str = "raw/review_categories/Software.jsonl"


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters."""

    min_text_length: int = 10
    max_text_length: int = 5000
    min_rating: int = 1
    max_rating: int = 5


@dataclass(frozen=True)
class SampleConfig:
    """Sample dataset settings."""

    reviews_per_rating: int = 20
    random_state: int = 42


@dataclass(frozen=True)
class LabelingConfig:
    """Auto-labeling settings."""

    min_confidence_for_curation: int = 2
    curated_sample_size: int = 300
    topics: tuple = (
        "performance",
        "usability",
        "features",
        "pricing",
        "support",
        "compatibility",
        "other",
    )
    topic_keywords: dict = field(
        default_factory=lambda: {
            "performance": [
                "slow",
                "fast",
                "crash",
                "bug",
                "freeze",
                "lag",
                "speed",
                "memory",
                "cpu",
                "hang",
                "stuck",
                "error",
                "glitch",
            ],
            "usability": [
                "easy",
                "difficult",
                "confusing",
                "intuitive",
                "ui",
                "interface",
                "learn",
                "simple",
                "complicated",
                "user-friendly",
            ],
            "features": [
                "feature",
                "function",
                "capability",
                "missing",
                "wish",
                "need",
                "want",
                "add",
                "include",
                "option",
                "setting",
            ],
            "pricing": [
                "price",
                "cost",
                "expensive",
                "cheap",
                "value",
                "subscription",
                "free",
                "pay",
                "money",
                "worth",
                "affordable",
            ],
            "support": [
                "support",
                "help",
                "customer service",
                "response",
                "documentation",
                "update",
                "team",
                "email",
                "contact",
            ],
            "compatibility": [
                "install",
                "compatible",
                "windows",
                "mac",
                "linux",
                "version",
                "os",
                "update",
                "upgrade",
                "work with",
            ],
        }
    )


@dataclass(frozen=True)
class ModelConfig:
    """Model file paths."""

    tfidf_vectorizer: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
    naive_bayes: Path = MODELS_DIR / "naive_bayes.pkl"
    bayesian_network: Path = MODELS_DIR / "bayesian_network.pkl"
    hmm_model: Path = MODELS_DIR / "hmm_model.pkl"


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings."""

    model_id: str = "us.anthropic.claude-sonnet-4-6"
    region: str = "us-east-1"


# Default instances for easy import
DATA = DataConfig()
DATASET = DatasetConfig()
PREPROCESS = PreprocessConfig()
SAMPLE = SampleConfig()
LABELING = LabelingConfig()
MODELS = ModelConfig()
LLM = LLMConfig()
LLM_HAIKU = LLMConfig(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
