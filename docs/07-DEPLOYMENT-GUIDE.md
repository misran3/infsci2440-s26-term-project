# Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- Git

### Initial Setup

```bash
# Clone the repository
git clone <repo-url>
cd group-project

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (uv creates venv automatically)
uv sync

# Download NLTK data
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('vader_lexicon')"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...          # For Pydantic AI (if using OpenAI)
ANTHROPIC_API_KEY=sk-ant-...   # For Pydantic AI (if using Anthropic)
```

### Running Locally

```bash
# Train models (first time only)
uv run python scripts/train_models.py

# Run Streamlit app
uv run streamlit run app.py

# App will be available at http://localhost:8501
```

## Project Structure

See `00-PROJECT-OVERVIEW.md` for the complete project file structure.

## Requirements

```toml
# pyproject.toml

[project]
name = "survey-analysis-agent"
version = "0.1.0"
description = "AI-powered survey response analysis using classical AI techniques"
requires-python = ">=3.10"
dependencies = [
    # Core
    "python-dotenv>=1.0.0",
    "pandas>=2.0",
    "numpy>=1.24",
    # NLP
    "nltk>=3.8",
    # Machine Learning
    "scikit-learn>=1.3",
    "pgmpy>=0.1.24",
    "hmmlearn>=0.3",
    # LLM
    "pydantic-ai>=0.0.10",
    # Web UI
    "streamlit>=1.30",
    # Data Loading
    "datasets>=2.16",
    # Model Persistence
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Streamlit Cloud Deployment

### Step 1: Prepare Repository

```bash
# Ensure pyproject.toml has all dependencies (already done)
# Generate requirements.txt for Streamlit Cloud (it doesn't support pyproject.toml yet)
uv pip compile pyproject.toml -o requirements.txt

# Create Streamlit config
mkdir -p .streamlit
```

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

### Step 2: Add Secrets Template

```toml
# .streamlit/secrets.toml.example (commit this)
# Copy to secrets.toml and fill in values (don't commit actual secrets)

OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Step 3: Push to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set main file path: `app.py`
6. Go to "Advanced settings" → "Secrets"
7. Paste your secrets (API keys)
8. Click "Deploy"

### Step 5: Verify Deployment

- Wait for build to complete (~2-3 minutes)
- Click on the app URL
- Test with a sample query

## Troubleshooting

### "No module named 'src'"

Add to the top of `app.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### "NLTK data not found"

Add to `app.py`:
```python
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
```

### "Model file not found"

Ensure models are trained and committed:
```bash
uv run python scripts/train_models.py
git add models/
git commit -m "Add trained models"
git push
```

### Memory Limit on Streamlit Cloud (1GB)

If you hit memory limits:
1. Use a smaller dataset subset
2. Load models lazily (only when needed)
3. Clear cache periodically: `st.cache_data.clear()`

## Demo Preparation Checklist

- [ ] All models trained and saved
- [ ] Test locally with sample queries
- [ ] API keys set in environment
- [ ] README has clear setup instructions
- [ ] Sample queries prepared for demo
- [ ] Error handling tested
- [ ] Backup plan if LLM API fails (show classical AI results only)
