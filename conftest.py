"""
Pytest configuration — adds the project root to sys.path so that
`from src.xxx import yyy` works in all test files.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
