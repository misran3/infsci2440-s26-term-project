# Data Preparation Testing Guide

Manual testing steps for the data preparation pipeline.

## Prerequisites

```bash
uv sync
```

---

## 1. Unit Tests

```bash
uv run pytest tests/test_config.py tests/test_loader.py -v
```

**Expected:** 9 tests pass (4 config + 5 loader)

---

## 2. Individual Scripts (requires internet)

Test each script in order:

### Step 1: Download (~5 min)
```bash
uv run download-data
```
**Expected:** Creates `data/amazon_reviews_software_raw.csv` with ~50K+ rows

### Step 2: Preprocess
```bash
uv run preprocess-data
```
**Expected:** Creates `data/amazon_reviews_software_clean.csv`, prints cleaning stats

### Step 3: Tokenize
```bash
uv run tokenize-sentences
```
**Expected:** Creates `data/amazon_reviews_software.csv` with `sentences` column

### Step 4: Sample
```bash
uv run create-sample
```
**Expected:** Creates `data/sample_reviews.csv` with 100 rows (20 per rating)

### Step 5: Label
```bash
uv run label-data
```
**Expected:**
- Creates `data/labeled_reviews.csv` with `topic_label` column
- Creates `data/curated_labels.csv` with ~300 rows for human curation

### Step 6: Validate
```bash
uv run validate-data
```
**Expected:** All checks pass, ends with `STATUS: READY`

---

## 3. Full Pipeline (after individual testing)

Once individual scripts work, test the consolidated runner:

```bash
uv run prepare-data
```

**Expected:**
- `[1/6]` through `[6/6]` progress messages
- Final `STATUS: READY` with all checks passed
- All data files created in `data/`

**Time:** 5-10 minutes total

---

## Summary of Expected Files

| File | Description |
|------|-------------|
| `amazon_reviews_software_raw.csv` | Raw download |
| `amazon_reviews_software_clean.csv` | After cleaning |
| `amazon_reviews_software.csv` | Main corpus with sentences |
| `sample_reviews.csv` | 100 balanced samples |
| `labeled_reviews.csv` | Auto-labeled corpus |
| `curated_labels.csv` | For human curation |

---

## Idempotency

Scripts skip processing when output files already exist:

```bash
# Second run skips all steps
uv run prepare-data
# Output: "Skipping download: data/amazon_reviews_software_raw.csv already exists"

# Force re-run all steps
uv run prepare-data --force
```

Individual scripts also support `--force`:
```bash
uv run download-data --force
uv run preprocess-data --force
uv run tokenize-sentences --force
uv run create-sample --force
uv run label-data --force
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `uv sync` |
| Download fails | Check internet connection |
| `FileNotFoundError` | Run scripts in order |
| Validation fails | Check which file is missing |
