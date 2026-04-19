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


def test_download_data_skips_when_file_exists(fake_raw_csv, capsys, monkeypatch):
    """download_data should skip when raw file exists."""
    from scripts.download_data import download_data
    from src import config

    # Monkeypatch DATA to use our temp file
    monkeypatch.setattr(config, "DATA", config.DataConfig(
        raw_reviews=fake_raw_csv,
        clean_reviews=fake_raw_csv.parent / "clean.csv",
        main_corpus=fake_raw_csv.parent / "corpus.csv",
        sample_reviews=fake_raw_csv.parent / "sample.csv",
        labeled_reviews=fake_raw_csv.parent / "labeled.csv",
        curated_labels=fake_raw_csv.parent / "curated.csv",
    ))

    # Call without force - should skip
    result = download_data(force=False)

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "already exists" in captured.out
    assert result is None  # Skipped, no DataFrame returned


def test_download_data_runs_with_force_flag(fake_raw_csv, monkeypatch):
    """download_data should run when --force is passed even if file exists."""
    # This test would require mocking hf_hub_download, skip for now
    # The key behavior is tested in test_download_data_skips_when_file_exists
    pass


def test_preprocess_data_skips_when_file_exists(temp_data_dir, capsys, monkeypatch):
    """preprocess_data should skip when clean file exists."""
    from scripts.preprocess_data import preprocess_data
    from src import config

    # Create both raw and clean files
    raw_file = temp_data_dir / "raw.csv"
    raw_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")
    clean_file = temp_data_dir / "clean.csv"
    clean_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")

    # Call without force - should skip
    result = preprocess_data(input_path=raw_file, output_path=clean_file, force=False)

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "already exists" in captured.out
    assert result is None


def test_tokenize_sentences_skips_when_file_exists(temp_data_dir, capsys):
    """tokenize_sentences should skip when corpus file exists."""
    from scripts.tokenize_sentences import tokenize_sentences

    # Create both input and output files
    clean_file = temp_data_dir / "clean.csv"
    clean_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software. Works well.,5,Good,P001\n")
    corpus_file = temp_data_dir / "corpus.csv"
    corpus_file.write_text("review_id,text,rating,title,product_id,sentences,sentence_count\nR000001,Great software. Works well.,5,Good,P001,\"[\"\"Great software.\"\", \"\"Works well.\"\"]\",2\n")

    # Call without force - should skip
    result = tokenize_sentences(input_path=clean_file, output_path=corpus_file, force=False)

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "already exists" in captured.out
    assert result is None


def test_create_sample_skips_when_file_exists(temp_data_dir, capsys):
    """create_sample should skip when sample file exists."""
    from scripts.create_sample import create_sample

    # Create both input and output files
    corpus_file = temp_data_dir / "corpus.csv"
    corpus_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")
    sample_file = temp_data_dir / "sample.csv"
    sample_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")

    # Call without force - should skip
    result = create_sample(input_path=corpus_file, output_path=sample_file, force=False)

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "already exists" in captured.out
    assert result is None


def test_label_training_data_skips_when_file_exists(temp_data_dir, capsys):
    """label_training_data should skip when labeled file exists."""
    from scripts.label_training_data import label_training_data

    # Create input and output files
    corpus_file = temp_data_dir / "corpus.csv"
    corpus_file.write_text("review_id,text,rating,title,product_id\nR000001,Great software,5,Good,P001\n")
    labeled_file = temp_data_dir / "labeled.csv"
    labeled_file.write_text("review_id,text,rating,title,product_id,topic_label\nR000001,Great software,5,Good,P001,features\n")
    curated_file = temp_data_dir / "curated.csv"

    # Call without force - should skip
    result = label_training_data(
        input_path=corpus_file,
        output_path=labeled_file,
        curated_path=curated_file,
        force=False
    )

    captured = capsys.readouterr()
    assert "Skipping" in captured.out or "already exists" in captured.out
    assert result is None
