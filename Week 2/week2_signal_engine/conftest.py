"""
pytest conftest — adds the project root to sys.path so tests can import
from src.* without needing an installed package.
"""
import sys
import os

# week2_signal_engine/ is the project root
sys.path.insert(0, os.path.dirname(__file__))
