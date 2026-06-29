"""Shared pytest fixtures and configuration."""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary database path for testing."""
    return str(tmp_path / "test_trading.db")
