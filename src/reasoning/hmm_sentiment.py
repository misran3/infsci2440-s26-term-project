"""HMM-inspired sentiment sequence analysis for review sentences."""

from __future__ import annotations

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from src.loaders.structures import Review, SentimentSequence
from src.loaders.structures import Sentiment


def _safe_download_nltk() -> None:
    """Ensure required NLTK resources are present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


class HMMSentiment:
    """Sentence-level sentiment sequence analyzer with transition estimates."""

    def __init__(self) -> None:
        _safe_download_nltk()
        self.sia = SentimentIntensityAnalyzer()
        self.global_transitions: dict[str, float] = {
            "pos_to_pos": 0.5,
            "pos_to_neg": 0.2,
            "neg_to_pos": 0.2,
            "neg_to_neg": 0.5,
            "neu_to_pos": 0.2,
            "neu_to_neg": 0.2,
            "neu_to_neu": 0.6,
        }

    def _sentence_sentiment(self, sentence: str) -> Sentiment:
        """Map VADER compound score to project sentiment states."""
        score = self.sia.polarity_scores(sentence).get("compound", 0.0)
        if score >= 0.2:
            return Sentiment.POSITIVE
        if score <= -0.2:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    @staticmethod
    def _transition_key(left: Sentiment, right: Sentiment) -> str:
        """Create transition key with existing naming convention."""
        left_prefix = "pos" if left == Sentiment.POSITIVE else "neg" if left == Sentiment.NEGATIVE else "neu"
        right_prefix = "pos" if right == Sentiment.POSITIVE else "neg" if right == Sentiment.NEGATIVE else "neu"
        return f"{left_prefix}_to_{right_prefix}"

    def _compute_transitions(self, states: list[Sentiment]) -> dict[str, float]:
        """Compute per-sequence transition probabilities."""
        keys = [
            "pos_to_pos",
            "pos_to_neg",
            "neg_to_pos",
            "neg_to_neg",
            "neu_to_pos",
            "neu_to_neg",
            "neu_to_neu",
        ]
        counts = {k: 0 for k in keys}

        if len(states) < 2:
            return dict(self.global_transitions)

        for left, right in zip(states[:-1], states[1:]):
            key = self._transition_key(left, right)
            if key in counts:
                counts[key] += 1

        total = sum(counts.values())
        if total == 0:
            return dict(self.global_transitions)

        return {k: counts[k] / total for k in keys}

    def fit(self, reviews: list[Review]) -> None:
        """Estimate global transition probabilities from review sentences."""
        sequences = self.analyze(reviews)
        if not sequences:
            return

        totals = {k: 0.0 for k in self.global_transitions}
        for sequence in sequences:
            for key, value in sequence.transitions.items():
                if key in totals:
                    totals[key] += value

        n = len(sequences)
        if n > 0:
            self.global_transitions = {k: totals[k] / n for k in totals}

    def analyze(self, reviews: list[Review]) -> list[SentimentSequence]:
        """
        Analyze sentence-level sentiment sequences for each review.

        Args:
            reviews: Reviews to analyze.

        Returns:
            List of SentimentSequence objects.
        """
        results: list[SentimentSequence] = []

        for review in reviews:
            sentences = list(review.sentences or [])
            if not sentences:
                sentences = [s for s in sent_tokenize(review.text) if s.strip()]
            if not sentences:
                sentences = [review.text]

            states = [self._sentence_sentiment(sentence) for sentence in sentences]
            transitions = self._compute_transitions(states)

            results.append(
                SentimentSequence(
                    review_id=review.review_id,
                    sentences=sentences,
                    sentiment_states=states,
                    transitions=transitions,
                )
            )

        return results
