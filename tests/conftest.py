"""Pytest configuration for AIBrain tests.

This module adds the ai-brain directory to the Python path,
allowing imports to work correctly.
"""
import sys
from pathlib import Path

# Add ai-brain directory to Python path for imports
ai_brain_path = Path(__file__).parent.parent / "ai-brain"
sys.path.insert(0, str(ai_brain_path))
