# src/reasoning/hmm_sentiment.py

"""HMM-based sentiment sequence analysis using hmmlearn."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import nltk
from hmmlearn.hmm import CategoricalHMM
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from src.loaders.structures import Review, Sentiment, SentimentSequence

logger = logging.getLogger(__name__)


def _safe_download_nltk() -> None:
    """Ensure required NLTK resources are present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


# Map hidden state indices to Sentiment enum
STATE_MAP = {0: Sentiment.POSITIVE, 1: Sentiment.NEGATIVE, 2: Sentiment.NEUTRAL}
SENTIMENT_TO_STATE = {v: k for k, v in STATE_MAP.items()}


class HMMSentiment:
    """Sentence-level sentiment sequence analyzer using hmmlearn HMM."""

    N_HIDDEN_STATES = 3  # positive, negative, neutral
    N_OBSERVATIONS = 5   # discretized VADER scores

    def __init__(self) -> None:
        _safe_download_nltk()
        self.sia = SentimentIntensityAnalyzer()
        self.model = CategoricalHMM(
            n_components=self.N_HIDDEN_STATES,
            n_features=self.N_OBSERVATIONS,
            n_iter=100,
            random_state=42,
        )
        self._is_fitted = False

    def _discretize_score(self, score: float) -> int:
        """Discretize VADER compound score to observation symbol."""
        if score < -0.6:
            return 0  # very negative
        if score < -0.2:
            return 1  # negative
        if score < 0.2:
            return 2  # neutral
        if score < 0.6:
            return 3  # positive
        return 4  # very positive

    def _sentences_to_observations(self, sentences: list[str]) -> np.ndarray:
        """Convert sentences to observation sequence."""
        obs = []
        for sentence in sentences:
            score = self.sia.polarity_scores(sentence).get("compound", 0.0)
            obs.append(self._discretize_score(score))
        return np.array(obs).reshape(-1, 1)

    def _get_sentences(self, review: Review) -> list[str]:
        """Get sentences from review, tokenizing if needed."""
        if review.sentences:
            return list(review.sentences)
        sentences = [s.strip() for s in sent_tokenize(review.text) if s.strip()]
        return sentences if sentences else [review.text]

    def fit(self, reviews: list[Review]) -> None:
        """Fit HMM using Baum-Welch on observation sequences."""
        all_obs = []
        lengths = []

        for review in reviews:
            sentences = self._get_sentences(review)
            if len(sentences) < 2:
                continue
            obs = self._sentences_to_observations(sentences)
            all_obs.append(obs)
            lengths.append(len(obs))

        if not all_obs:
            return

        X = np.vstack(all_obs)
        self.model.fit(X, lengths)
        self._is_fitted = True

    def _compute_transitions(self, states: list[Sentiment]) -> dict[str, float]:
        """Compute transition probabilities from state sequence."""
        keys = [
            "pos_to_pos", "pos_to_neg", "pos_to_neu",
            "neg_to_pos", "neg_to_neg", "neg_to_neu",
            "neu_to_pos", "neu_to_neg", "neu_to_neu",
        ]
        counts = {k: 0 for k in keys}
        totals = {"pos": 0, "neg": 0, "neu": 0}

        prefix_map = {
            Sentiment.POSITIVE: "pos",
            Sentiment.NEGATIVE: "neg",
            Sentiment.NEUTRAL: "neu",
        }

        for i in range(len(states) - 1):
            curr_prefix = prefix_map[states[i]]
            next_prefix = prefix_map[states[i + 1]]
            key = f"{curr_prefix}_to_{next_prefix}"
            counts[key] += 1
            totals[curr_prefix] += 1

        result = {}
        for key in keys:
            curr_prefix = key.split("_to_")[0]
            total = totals[curr_prefix]
            result[key] = counts[key] / total if total > 0 else 0.0

        return result

    def analyze(self, reviews: list[Review]) -> list[SentimentSequence]:
        """Analyze sentiment sequences using Viterbi decoding."""
        results = []

        for review in reviews:
            sentences = self._get_sentences(review)
            obs = self._sentences_to_observations(sentences)

            if self._is_fitted and len(sentences) >= 2:
                try:
                    _, state_indices = self.model.decode(obs, algorithm="viterbi")
                    states = [STATE_MAP[idx] for idx in state_indices]
                except (ValueError, IndexError, KeyError) as e:
                    logger.warning("Viterbi decoding failed for review '%s': %s", review.review_id, e)
                    states = self._fallback_states(sentences)
            else:
                states = self._fallback_states(sentences)

            transitions = self._compute_transitions(states)

            results.append(SentimentSequence(
                review_id=review.review_id,
                sentences=sentences,
                sentiment_states=states,
                transitions=transitions,
            ))

        return results

    def _fallback_states(self, sentences: list[str]) -> list[Sentiment]:
        """Fallback: use VADER directly when model not fitted."""
        states = []
        for sentence in sentences:
            score = self.sia.polarity_scores(sentence).get("compound", 0.0)
            if score >= 0.2:
                states.append(Sentiment.POSITIVE)
            elif score <= -0.2:
                states.append(Sentiment.NEGATIVE)
            else:
                states.append(Sentiment.NEUTRAL)
        return states

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk.

        Args:
            path: File path to save the model to.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted HMMSentiment")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "HMMSentiment":
        """Load a fitted HMMSentiment from disk.

        Args:
            path: File path to load the model from.

        Returns:
            A fitted HMMSentiment instance.

        Raises:
            ValueError: If the loaded object is not a valid hmmlearn CategoricalHMM.
        """
        loaded = joblib.load(path)
        if not isinstance(loaded, CategoricalHMM):
            raise ValueError(
                f"Invalid model type: expected CategoricalHMM, got {type(loaded).__name__}"
            )

        instance = cls()
        instance.model = loaded
        instance._is_fitted = True
        return instance
