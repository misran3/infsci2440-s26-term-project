# Manual UI Testing Guide

## Prerequisites

1. Train the classifier:
   ```bash
   # For development (fast, lower accuracy)
   uv run python scripts/train_classifier.py --dataset sample
   
   # For production (slower, higher accuracy) 
   uv run python scripts/train_classifier.py --dataset full --evaluate
   ```

2. Export AWS credentials (required for LLM TermFilter):
   ```bash
   export AWS_PROFILE=your-profile-name
   export AWS_REGION=us-east-1
   ```
   
   Or if using explicit credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_REGION=us-east-1
   ```

3. Start the app:
   ```bash
   uv run streamlit run app.py
   ```

4. Open http://localhost:8501 in browser

---

## Understanding the Two Score Types

The UI displays two different scores - don't confuse them:

| Score | What it measures | Where shown |
|-------|------------------|-------------|
| **TF-IDF relevance** | How well review matches search terms | "Show sample reviews" expander |
| **Classification confidence** | Naive Bayes probability for topic | "Show filtered reviews" expander, threshold filter |

A review can have high TF-IDF relevance (matches "crash" keyword) but low classification confidence (classifier unsure if it's about "performance" topic).

---

## Test Cases

### Test 1: Performance Topic (Crashes/Bugs)

**Query:** `app crashes bugs`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | Original: "app crashes bugs"<br>BeamSearch: 15-25 terms<br>LLM Filtered: 5-10 software-relevant terms (crash, bug, glitch, etc.) |
| 2. TF-IDF Retrieval | 300-500 candidate reviews<br>"Show sample reviews" displays top 5 with star ratings and TF-IDF relevance scores |
| 3. Topic Classification | Detected topic: **performance**<br>Bar chart showing topic distribution<br>"Show filtered reviews" displays top 5 with classification confidence |
| 4. Bayesian Network | "Not yet implemented (Person 3)" info box |
| 5. HMM Sentiment | "Not yet implemented (Person 3)" info box |
| 6. LLM Summary | "Summary not available - component pending" |

**Verify:**
- Sample reviews should mention crashes/bugs in text
- Filtered reviews should have confidence scores shown
- Bar chart should show "performance" or "other" as top topics

---

### Test 2: Usability Topic (Easy/Intuitive)

**Query:** `easy to use intuitive`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | LLM should filter to: easy, intuitive, simple, use, user-friendly |
| 2. TF-IDF Retrieval | 50-300 candidate reviews with TF-IDF scores |
| 3. Topic Classification | Detected topic: **usability**<br>Bar chart shows usability prominently<br>Filtered reviews mention ease of use |
| 4-6. | Person 3 stubs |

---

### Test 3: Pricing Topic (Expensive/Value)

**Query:** `too expensive subscription cost`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | LLM should keep: expensive, subscription, cost, price, value |
| 2. TF-IDF Retrieval | 30-200 candidate reviews |
| 3. Topic Classification | Detected topic: **pricing**<br>Filtered reviews discuss cost/value |
| 4-6. | Person 3 stubs |

---

### Test 4: Support Topic (Customer Service/Help)

**Query:** `customer service support help`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | LLM should keep: customer, service, support, help |
| 2. TF-IDF Retrieval | 20-150 candidate reviews |
| 3. Topic Classification | Detected topic: **support**<br>Filtered reviews mention support experiences |
| 4-6. | Person 3 stubs |

---

### Test 5: Compatibility Topic (Install/Windows/Mac)

**Query:** `install windows compatible`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | LLM should keep: install, windows, compatible, mac |
| 2. TF-IDF Retrieval | 30-200 candidate reviews |
| 3. Topic Classification | Detected topic: **compatibility**<br>Filtered reviews mention installation/OS issues |
| 4-6. | Person 3 stubs |

---

### Test 6: Features Topic (Missing Features)

**Query:** `missing feature wish had`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | LLM should keep: missing, feature, wish, want, need |
| 2. TF-IDF Retrieval | 30-200 candidate reviews |
| 3. Topic Classification | Detected topic: **features**<br>Filtered reviews request features |
| 4-6. | Person 3 stubs |

---

### Test 7: Low Confidence Fallback

**Query:** `app crashes` with model trained on sample dataset

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 3. Topic Classification | Warning: "X reviews classified as 'performance' but none met classification confidence threshold (50%)"<br>Note explaining TF-IDF vs classification confidence<br>Suggestion to lower threshold or retrain<br>Bar chart still shows topic distribution |

**Fix:** Either lower Min Confidence slider to 0.3, or retrain on full dataset.

---

### Test 8: Manual Topic Filter Override

**Query:** `software problems`  
**Filter:** Set Topic Filter to "pricing" (override auto-detect)

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 3. Topic Classification | Detected topic: **pricing** (overridden)<br>Filters to pricing-related reviews only |

---

### Test 9: Confidence Threshold Test

**Query:** `app crashes`  
**Filter:** Set Min Confidence to 0.9

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 3. Topic Classification | Very few or zero filtered reviews<br>Only highest-confidence classifications pass |

**Then:** Lower to 0.3 and verify more reviews pass.

---

### Test 10: TermFilter Integration (if AWS configured)

**Query:** `slow performance lag`

**Expected Results:**

| Section | Expected Output |
|---------|-----------------|
| 1. Query Expansion | BeamSearch: 20-30 terms (includes irrelevant synonyms)<br>LLM Filtered: 5-10 terms (software-relevant only)<br>"Show expansion details" shows removed terms like "retard", "dawdle", etc. |

**If AWS not configured:** Shows "(LLM filter skipped - AWS not configured)"

---

## UI Verification Checklist

- [x] Page title shows "Survey Analysis Agent"
- [x] Query input box accepts text
- [x] Search button triggers analysis
- [x] "Analyzing..." spinner appears during processing
- [x] All 6 pipeline sections display
- [x] "Show expansion details" expander works (beam paths, removed terms)
- [x] "Show sample reviews" expander shows TF-IDF relevance scores
- [x] Topic distribution bar chart renders
- [x] "Show filtered reviews" expander shows classification confidence
- [x] Optional Filters expander works (topic dropdown, confidence slider)
- [x] Low confidence warning shows explanation note

---

## Performance Expectations

| Metric | Target |
|--------|--------|
| Cold start (first load) | < 30 seconds |
| Pipeline execution | < 10 seconds |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "Model not found" error | Run `uv run python scripts/train_classifier.py --dataset sample` |
| Low classification confidence | Retrain on full dataset: `--dataset full` |
| No filtered reviews | Lower Min Confidence slider in Optional Filters |
| TermFilter not working | Check AWS_PROFILE and AWS_REGION env vars |
| App crashes on start | Check `uv run python -c "import app"` for import errors |
| Serialization error | Fixed by removing caching - restart app |
