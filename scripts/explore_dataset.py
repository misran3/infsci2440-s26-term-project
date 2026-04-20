"""Explore the Amazon reviews dataset to identify good manual test queries."""

import json
from collections import Counter
from pathlib import Path
import re

import pandas as pd
import numpy as np

# Use the project's config
from src.config import DATA
from src.loaders.loader import load_reviews

def analyze_vocabulary(reviews_df: pd.DataFrame, top_n: int = 50):
    """Analyze the most common words in reviews."""
    all_text = ' '.join(reviews_df['text'].astype(str).tolist())
    # Simple tokenization
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    word_counts = Counter(words)

    # Filter out common stop words
    stop_words = {'the', 'and', 'for', 'this', 'that', 'with', 'have',
                  'was', 'but', 'not', 'are', 'you', 'can', 'all', 'your',
                  'from', 'they', 'been', 'has', 'had', 'will', 'more',
                  'when', 'make', 'than', 'into', 'time', 'very', 'about',
                  'out', 'there', 'were', 'get', 'some', 'what', 'only',
                  'other', 'which', 'their', 'also', 'would', 'use', 'used',
                  'using', 'one', 'two', 'any', 'much', 'does', 'did'}

    filtered_counts = {w: c for w, c in word_counts.items() if w not in stop_words}
    return Counter(filtered_counts).most_common(top_n)

def analyze_review_lengths(reviews_df: pd.DataFrame):
    """Analyze review length distribution."""
    lengths = reviews_df['text'].str.len()
    word_counts = reviews_df['text'].str.split().str.len()

    return {
        'char_length_stats': {
            'mean': lengths.mean(),
            'median': lengths.median(),
            'min': lengths.min(),
            'max': lengths.max(),
            'std': lengths.std()
        },
        'word_count_stats': {
            'mean': word_counts.mean(),
            'median': word_counts.median(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'std': word_counts.std()
        }
    }

def find_reviews_with_term(reviews_df: pd.DataFrame, term: str, limit: int = 5):
    """Find sample reviews containing a specific term."""
    mask = reviews_df['text'].str.lower().str.contains(term.lower(), na=False)
    matching = reviews_df[mask].head(limit)
    return matching[['text', 'rating']].to_dict('records')

def analyze_common_phrases(reviews_df: pd.DataFrame, n: int = 2, top_k: int = 30):
    """Extract common n-grams (phrases) from reviews."""
    all_text = ' '.join(reviews_df['text'].astype(str).tolist())
    words = re.findall(r'\b[a-z]+\b', all_text.lower())

    # Create n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)

    ngram_counts = Counter(ngrams)

    # Filter out phrases with only stop words
    stop_words = {'the', 'and', 'for', 'this', 'that', 'with', 'have',
                  'was', 'but', 'not', 'are', 'you', 'can', 'all', 'your',
                  'from', 'they', 'been', 'has', 'had', 'will', 'more',
                  'when', 'make', 'than', 'into', 'time', 'very', 'about'}

    filtered_ngrams = {}
    for ngram, count in ngram_counts.items():
        words_in_ngram = set(ngram.split())
        if not words_in_ngram.issubset(stop_words):
            filtered_ngrams[ngram] = count

    return Counter(filtered_ngrams).most_common(top_k)

def identify_synonym_candidates(reviews_df: pd.DataFrame):
    """Identify terms that likely have meaningful synonyms in the dataset."""
    # Manually curated list of software review terms with common synonyms
    synonym_groups = {
        'fast': ['quick', 'speedy', 'rapid', 'swift', 'responsive'],
        'slow': ['sluggish', 'laggy', 'unresponsive', 'delayed'],
        'easy': ['simple', 'straightforward', 'intuitive', 'user-friendly'],
        'difficult': ['hard', 'complicated', 'complex', 'confusing', 'challenging'],
        'bug': ['error', 'issue', 'problem', 'glitch', 'defect'],
        'expensive': ['pricey', 'costly', 'overpriced'],
        'cheap': ['affordable', 'inexpensive', 'budget-friendly'],
        'good': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
        'bad': ['poor', 'terrible', 'awful', 'horrible'],
        'install': ['setup', 'installation', 'download'],
        'crash': ['freeze', 'hang', 'stuck'],
    }

    results = {}
    for primary_term, synonyms in synonym_groups.items():
        all_terms = [primary_term] + synonyms
        counts = {}
        for term in all_terms:
            mask = reviews_df['text'].str.lower().str.contains(rf'\b{term}\b', na=False, regex=True)
            counts[term] = mask.sum()

        results[primary_term] = counts

    return results

def main():
    """Main analysis function."""
    print("="*80)
    print("AMAZON REVIEWS DATASET EXPLORATION")
    print("="*80)
    print()

    # Load sample reviews for faster exploration
    print("Loading sample reviews...")
    if not DATA.sample_reviews.exists():
        print(f"Sample file not found at {DATA.sample_reviews}")
        print("Checking main corpus...")
        if not DATA.main_corpus.exists():
            print(f"Main corpus not found at {DATA.main_corpus}")
            print("Please run: uv run prepare-data")
            return
        # Use main corpus but limit it
        df = pd.read_csv(DATA.main_corpus, nrows=1000)
        print("Loaded 1000 reviews from main corpus")
    else:
        df = pd.read_csv(DATA.sample_reviews)
        print(f"Loaded {len(df)} sample reviews")

    print()
    print("-"*80)
    print("1. DATASET OVERVIEW")
    print("-"*80)
    print(f"Total reviews: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()

    if 'rating' in df.columns:
        print("Rating distribution:")
        print(df['rating'].value_counts().sort_index())
    print()

    # Review length analysis
    print("-"*80)
    print("2. REVIEW LENGTH DISTRIBUTION")
    print("-"*80)
    length_stats = analyze_review_lengths(df)
    print("Character length:")
    for key, val in length_stats['char_length_stats'].items():
        print(f"  {key}: {val:.1f}")
    print()
    print("Word count:")
    for key, val in length_stats['word_count_stats'].items():
        print(f"  {key}: {val:.1f}")
    print()

    # Vocabulary analysis
    print("-"*80)
    print("3. MOST COMMON TERMS (excluding stop words)")
    print("-"*80)
    top_words = analyze_vocabulary(df, top_n=40)
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:20s} ({count:4d} occurrences)")
    print()

    # Common phrases
    print("-"*80)
    print("4. COMMON 2-WORD PHRASES")
    print("-"*80)
    bigrams = analyze_common_phrases(df, n=2, top_k=25)
    for i, (phrase, count) in enumerate(bigrams, 1):
        print(f"{i:2d}. '{phrase}' ({count} occurrences)")
    print()

    # Synonym analysis
    print("-"*80)
    print("5. SYNONYM/RELATED TERM FREQUENCY ANALYSIS")
    print("-"*80)
    synonym_results = identify_synonym_candidates(df)
    for primary, counts in synonym_results.items():
        total = sum(counts.values())
        if total > 0:  # Only show if terms appear in corpus
            print(f"\n{primary.upper()} group (total: {total}):")
            for term, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  {term:20s}: {count:4d}")
    print()

    # Sample reviews for specific terms
    print("-"*80)
    print("6. SAMPLE REVIEWS FOR CANDIDATE QUERY TERMS")
    print("-"*80)

    candidate_queries = ['easy', 'bug', 'fast', 'crash', 'expensive', 'install']

    for query in candidate_queries:
        print(f"\n>>> Query: '{query}'")
        samples = find_reviews_with_term(df, query, limit=3)
        for i, sample in enumerate(samples, 1):
            text = sample['text']
            # Truncate long reviews
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"\n  Example {i} (rating: {sample['rating']}):")
            print(f"  {text}")
    print()

    # Recommendations
    print("="*80)
    print("RECOMMENDED TEST QUERIES")
    print("="*80)
    print()
    print("Based on the analysis, here are 5 recommended test queries:")
    print()

    recommendations = [
        ("easy", "Has clear synonyms (simple, intuitive, straightforward) and appears in diverse contexts (UI, learning curve, setup)"),
        ("bug", "Can expand to error, issue, problem, glitch - important for software reviews"),
        ("slow", "Performance-related with synonyms (sluggish, laggy, unresponsive)"),
        ("expensive", "Pricing feedback with related terms (costly, pricey, overpriced, worth)"),
        ("install", "Setup-related with variations (installation, download, setup)")
    ]

    for i, (query, reason) in enumerate(recommendations, 1):
        print(f"{i}. '{query}'")
        print(f"   Reason: {reason}")
        print()

if __name__ == "__main__":
    main()
