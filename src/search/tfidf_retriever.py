"""TF-IDF based review retrieval."""

from __future__ import annotations

import os
from dataclasses import replace

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.loaders.structures import Review


class TFIDFRetriever:
    """Retrieves reviews using TF-IDF similarity."""

    def __init__(self, corpus: list[Review]) -> None:
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None

    def fit(self) -> None:
        """Fit TF-IDF vectorizer on corpus."""
        if not self.corpus:
            return
        texts = [r.text for r in self.corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, expanded_terms: list[str], top_k: int = 500) -> list[Review]:
        """Retrieve top-k reviews matching expanded terms."""
        if not expanded_terms or self.tfidf_matrix is None:
            return []

        query = " ".join(expanded_terms)
        query_vec = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results: list[Review] = []
        for idx in top_indices:
            if similarities[idx] > 0:
                review = replace(self.corpus[idx], tfidf_score=float(similarities[idx]))
                results.append(review)

        return results

    def get_matching_terms(
        self, review: Review, expanded_terms: list[str]
    ) -> list[str]:
        """Get which expanded terms appear in a review."""
        review_words = set(review.text.lower().split())
        return [term for term in expanded_terms if term.lower() in review_words]


def save_model(retriever: TFIDFRetriever, path: str = "models/tfidf_vectorizer.pkl") -> None:
    """Save fitted TF-IDF model to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(
        {
            "vectorizer": retriever.vectorizer,
            "tfidf_matrix": retriever.tfidf_matrix,
            "corpus_ids": [r.review_id for r in retriever.corpus],
        },
        path,
    )


def load_model(path: str = "models/tfidf_vectorizer.pkl") -> dict:
    """Load fitted TF-IDF model from disk."""
    return joblib.load(path)
