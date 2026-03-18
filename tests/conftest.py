"""Pytest fixtures and configuration."""

import sys
from pathlib import Path

# Add src directory to Python path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
