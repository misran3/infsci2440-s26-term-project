"""Tests for script idempotency (skip when output exists)."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temp directory structure mimicking data/."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def fake_raw_csv(temp_data_dir):
    """Create a minimal raw CSV file."""
    raw_file = temp_data_dir / "amazon_reviews_software_raw.csv"
    raw_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")
    return raw_file
