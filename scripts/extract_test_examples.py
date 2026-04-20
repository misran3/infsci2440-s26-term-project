"""Extract specific review examples for each test query to use in manual testing."""

import re
import pandas as pd
from src.config import DATA

def get_clean_excerpt(text: str, term: str, max_length: int = 150):
    """Extract a clean excerpt containing the term."""
    # Find sentences containing the term
    sentences = re.split(r'[.!?]+', text)
    for sentence in sentences:
        if term.lower() in sentence.lower():
            clean = sentence.strip()
            # Remove HTML tags
            clean = re.sub(r'<[^>]+>', '', clean)
            if len(clean) > max_length:
                clean = clean[:max_length] + "..."
            return clean
    # Fallback: just return first part
    clean = text[:max_length] + "..."
    return re.sub(r'<[^>]+>', '', clean)

def extract_examples_for_query(df: pd.DataFrame, query: str, related_terms: list, num_examples: int = 3):
    """Extract specific examples for a query and its related terms."""
    print(f"\n{'='*80}")
    print(f"TEST EXAMPLES: '{query}'")
    print(f"{'='*80}\n")

    all_terms = [query] + related_terms

    for term in all_terms:
        pattern = rf'\b{term}\b'
        mask = df['text'].str.lower().str.contains(pattern, na=False, regex=True)
        matches = df[mask]

        if len(matches) > 0:
            print(f"Term: '{term}' ({len(matches)} occurrences)")
            print("-" * 40)

            for idx, (_, row) in enumerate(matches.head(num_examples).iterrows()):
                excerpt = get_clean_excerpt(row['text'], term)
                print(f"  Example {idx+1}:")
                print(f"    Review ID: {row['review_id']}")
                print(f"    Rating: {row['rating']}/5")
                print(f"    Excerpt: \"{excerpt}\"")
                print()

def main():
    """Generate test examples."""
    print("="*80)
    print("MANUAL TEST EXAMPLES")
    print("Concrete examples for BeamSearchExpander & TFIDFRetriever testing")
    print("="*80)

    # Load data
    if DATA.sample_reviews.exists():
        df = pd.read_csv(DATA.sample_reviews)
    else:
        df = pd.read_csv(DATA.main_corpus, nrows=1000)

    print(f"\nDataset: {len(df)} reviews\n")

    # Test queries with related terms
    test_queries = {
        'easy': ['simple', 'intuitive'],
        'difficult': ['hard', 'complicated'],
        'bug': ['error', 'issue', 'problem'],
        'good': ['great', 'excellent'],
        'crash': ['freeze', 'stuck'],
    }

    for query, related in test_queries.items():
        extract_examples_for_query(df, query, related, num_examples=2)

    # Summary
    print("="*80)
    print("HOW TO USE THESE EXAMPLES")
    print("="*80)
    print()
    print("1. BeamSearchExpander Manual Test:")
    print("   - Input: Primary query (e.g., 'easy')")
    print("   - Expected: Expansion should include related terms (e.g., 'simple', 'intuitive')")
    print("   - Verify: Check beam paths and scores make semantic sense")
    print()
    print("2. TFIDFRetriever Manual Test:")
    print("   - Test A: Search with base query only (e.g., 'easy')")
    print("   - Test B: Search with expanded query (e.g., 'easy simple intuitive')")
    print("   - Compare: Test B should retrieve more relevant reviews")
    print("   - Check: Should retrieve the review IDs shown above")
    print()
    print("3. Integration Test:")
    print("   - Run full pipeline: query -> expand -> retrieve")
    print("   - Verify: Results contain reviews with query AND synonym terms")
    print("   - Quality: Check if semantic similarity holds across expansion")
    print()

if __name__ == "__main__":
    main()
