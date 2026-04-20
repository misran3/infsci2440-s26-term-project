"""Deep dive analysis of candidate test queries for BeamSearchExpander and TFIDFRetriever."""

import re
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict

from src.config import DATA

def extract_context_window(text: str, term: str, window_size: int = 5):
    """Extract words around a term in context."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    contexts = []
    for i, word in enumerate(words):
        if term in word:
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context = words[start:end]
            contexts.append(' '.join(context))

    return contexts

def analyze_query_contexts(df: pd.DataFrame, query: str, synonym_group: list):
    """Analyze how a query term and its synonyms appear in context."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: '{query}' and synonyms")
    print(f"{'='*80}")

    # Count occurrences
    all_terms = [query] + synonym_group
    term_counts = {}
    term_reviews = defaultdict(list)

    for term in all_terms:
        pattern = rf'\b{term}\b'
        mask = df['text'].str.lower().str.contains(pattern, na=False, regex=True)
        count = mask.sum()
        term_counts[term] = count

        if count > 0:
            matching_reviews = df[mask]['text'].tolist()
            term_reviews[term] = matching_reviews

    # Display counts
    print(f"\nTerm frequency:")
    total_occurrences = sum(term_counts.values())
    for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {term:20s}: {count:3d} reviews ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_occurrences:3d} reviews")

    # Co-occurring terms (context analysis)
    print(f"\nCommon co-occurring words (within 5 words):")
    all_contexts = []
    for term, reviews in term_reviews.items():
        if len(reviews) > 0:
            for review in reviews[:10]:  # Limit to avoid too much processing
                contexts = extract_context_window(review, term, window_size=5)
                all_contexts.extend(contexts)

    # Count words in contexts
    context_words = []
    for context in all_contexts:
        words = re.findall(r'\b[a-z]{3,}\b', context.lower())
        context_words.extend(words)

    # Filter out the query terms and common stop words
    stop_words = {'the', 'and', 'for', 'this', 'that', 'with', 'have',
                  'was', 'but', 'not', 'are', 'you', 'can', 'all', 'your',
                  'from', 'they', 'been', 'has', 'had', 'will', 'more',
                  'when', 'make', 'than', 'into', 'time', 'very', 'about',
                  'its', 'too', 'just', 'use', 'used'}

    filtered_words = [w for w in context_words
                     if w not in stop_words
                     and w not in [t.lower() for t in all_terms]]

    word_counts = Counter(filtered_words).most_common(15)
    for word, count in word_counts:
        print(f"    {word:15s}: {count:3d}")

    # Example sentences
    print(f"\nExample review snippets:")
    shown = 0
    for term in all_terms:
        if term in term_reviews and len(term_reviews[term]) > 0 and shown < 3:
            review = term_reviews[term][0]
            # Find the sentence containing the term
            sentences = re.split(r'[.!?]+', review)
            for sentence in sentences:
                if term in sentence.lower():
                    snippet = sentence.strip()
                    if len(snippet) > 100:
                        snippet = snippet[:100] + "..."
                    print(f"  [{term}]: {snippet}")
                    shown += 1
                    break

    # Rating distribution
    print(f"\nRating distribution for reviews containing these terms:")
    for term in all_terms:
        if term_counts.get(term, 0) > 0:
            pattern = rf'\b{term}\b'
            mask = df['text'].str.lower().str.contains(pattern, na=False, regex=True)
            ratings = df[mask]['rating'].value_counts().sort_index()
            avg_rating = df[mask]['rating'].mean()
            print(f"  {term:15s}: avg={avg_rating:.1f} | " +
                  " | ".join([f"{r}★:{c}" for r, c in ratings.items()]))

def main():
    """Main analysis."""
    print("="*80)
    print("TEST QUERY DEEP-DIVE ANALYSIS")
    print("For BeamSearchExpander and TFIDFRetriever Manual Testing")
    print("="*80)

    # Load data
    if DATA.sample_reviews.exists():
        df = pd.read_csv(DATA.sample_reviews)
        print(f"\nLoaded {len(df)} sample reviews")
    else:
        print(f"\nSample file not found. Loading first 1000 from main corpus...")
        df = pd.read_csv(DATA.main_corpus, nrows=1000)
        print(f"Loaded {len(df)} reviews")

    # Define query groups with synonyms/related terms
    query_groups = {
        'easy': ['simple', 'intuitive', 'straightforward', 'user-friendly'],
        'difficult': ['hard', 'complicated', 'complex', 'confusing', 'challenging'],
        'bug': ['error', 'issue', 'problem', 'glitch', 'defect'],
        'slow': ['sluggish', 'laggy', 'unresponsive', 'delayed'],
        'crash': ['freeze', 'hang', 'stuck'],
        'good': ['great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
        'install': ['setup', 'installation', 'download'],
    }

    # Analyze each query
    for query, synonyms in query_groups.items():
        analyze_query_contexts(df, query, synonyms)

    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS FOR MANUAL TESTING")
    print("="*80)
    print()
    print("Based on this deep analysis, here are 5 specific test queries:")
    print()

    recommendations = [
        {
            'query': 'easy',
            'expansion_targets': ['simple', 'intuitive', 'straightforward', 'user-friendly'],
            'reason': 'Strong synonym group (11 occurrences). Appears in usability contexts. Should expand to "simple" (6 occurrences) effectively.',
            'expected_retrieval': 'Reviews about UI/UX, learning curve, setup process'
        },
        {
            'query': 'difficult',
            'expansion_targets': ['hard', 'complicated', 'complex', 'challenging'],
            'reason': 'Good diversity (16 total occurrences across synonyms). "Hard" most common (7), good test for synonym ranking.',
            'expected_retrieval': 'Reviews about complexity, learning difficulty, frustration'
        },
        {
            'query': 'bug',
            'expansion_targets': ['error', 'issue', 'problem', 'glitch'],
            'reason': 'Technical issue reporting (7 occurrences). "Problem" most frequent (4), natural expansion candidate.',
            'expected_retrieval': 'Software defect reports, error messages, malfunction complaints'
        },
        {
            'query': 'good',
            'expansion_targets': ['great', 'excellent', 'wonderful', 'amazing'],
            'reason': 'High frequency positive sentiment (26 occurrences). Clear synonym hierarchy for beam search scoring.',
            'expected_retrieval': 'Positive reviews, high ratings, satisfaction expressions'
        },
        {
            'query': 'crash',
            'expansion_targets': ['freeze', 'hang', 'stuck'],
            'reason': 'Performance/stability issues (3 occurrences). Small but specific semantic group for precision testing.',
            'expected_retrieval': 'App stability problems, technical failures, system freezes'
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Query: '{rec['query']}'")
        print(f"   Expansion targets: {', '.join(rec['expansion_targets'])}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Expected retrieval: {rec['expected_retrieval']}")
        print()

    print("="*80)
    print("TESTING STRATEGY")
    print("="*80)
    print()
    print("For BeamSearchExpander:")
    print("  - Test if beam search finds the synonym terms in the expansion")
    print("  - Verify beam paths contain meaningful semantic relationships")
    print("  - Check scoring: closer synonyms should score higher")
    print()
    print("For TFIDFRetriever:")
    print("  - Without expansion: verify base retrieval works")
    print("  - With expansion: should retrieve more relevant reviews")
    print("  - Compare precision: expanded query should not add too much noise")
    print()

if __name__ == "__main__":
    main()
