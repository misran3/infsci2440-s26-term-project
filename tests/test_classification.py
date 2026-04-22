"""Tests for Naive Bayes topic classifier."""

import pytest
from src.loaders.structures import Review, Topic
from src.classification.naive_bayes import TopicClassifier


@pytest.fixture
def training_reviews():
    """Small training set for tests."""
    return [
        Review("1", "Software crashes constantly and has many bugs", 1, "Buggy", "P1"),
        Review("2", "Very easy to use, intuitive interface", 5, "Great UI", "P2"),
        Review("3", "Too expensive for what you get", 2, "Overpriced", "P3"),
        Review("4", "App freezes and runs slow", 1, "Slow", "P1"),
        Review("5", "Simple and user-friendly design", 4, "Nice", "P2"),
        Review("6", "Not worth the subscription cost", 2, "Bad value", "P3"),
    ]


@pytest.fixture
def training_labels():
    """Labels for training set."""
    return ["performance", "usability", "pricing", "performance", "usability", "pricing"]


def test_classifier_predicts_correct_topic(training_reviews, training_labels):
    """Train on small set, verify crash/bug text -> performance topic."""
    classifier = TopicClassifier()
    classifier.fit(training_reviews, training_labels)

    test_reviews = [Review("99", "The app freezes all the time", 1, "Bad", "P99")]
    results = classifier.predict(test_reviews)

    assert len(results) == 1
    assert results[0].predicted_topic == Topic.PERFORMANCE
    assert results[0].confidence > 0.3


def test_classifier_handles_empty_input(training_reviews, training_labels):
    """predict([]) returns []."""
    classifier = TopicClassifier()
    classifier.fit(training_reviews, training_labels)

    results = classifier.predict([])

    assert results == []


def test_filter_by_topic_no_matches_returns_distribution(training_reviews, training_labels):
    """Zero matches -> FilterResult with fallback_used=True."""
    classifier = TopicClassifier()
    classifier.fit(training_reviews, training_labels)

    # Filter for "compatibility" which has no matches in training data
    result = classifier.filter_by_topic(training_reviews, "compatibility")

    assert result.fallback_used is True
    assert result.filtered_reviews == []
    assert "performance" in result.topic_distribution
    assert result.topic_distribution["performance"] > 0
