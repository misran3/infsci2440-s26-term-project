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
