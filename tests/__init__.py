"""
Test suite for matching_experimental package.

This package contains comprehensive tests for the experimental template matching
system, including unit tests, integration tests, and fixtures for testing.
"""

import sys
from pathlib import Path

# Add project root to path for testing
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "matching_experimental"))

__version__ = "1.0.0"
__author__ = "Template Matching Research Team"