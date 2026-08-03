import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to sys.path so tests can import the `src` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock Redis and Database globally to prevent real connections during tests."""
    with patch("src.utils.redis_client.get_redis_client", return_value=MagicMock()), \
         patch("src.utils.redis_client.is_redis_available", return_value=True), \
         patch("src.database.get_connection", return_value=MagicMock()):
        yield
