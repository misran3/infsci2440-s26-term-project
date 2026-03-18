"""Data structures for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Topic(str, Enum):
    """Review topic categories."""
    PERFORMANCE = "performance"
    USABILITY = "usability"
    FEATURES = "features"
    PRICING = "pricing"
    SUPPORT = "support"
    COMPATIBILITY = "compatibility"
    OTHER = "other"
