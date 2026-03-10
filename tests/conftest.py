"""Configure test paths to avoid root __init__.py package import issues."""
import sys
import os

# Add repo root to path so tests can import model, utils directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
