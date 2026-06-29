"""Unit tests for database operations."""
import pytest
import os
import time
import json
from unittest.mock import patch, MagicMock


class TestNormalizeSymbol:
    """Test the _normalize_symbol helper."""

    def test_pair_format(self):
        from src.database import _normalize_symbol
        assert _normalize_symbol("AAPL/USD") == "AAPL"

    def test_base_only(self):
        from src.database import _normalize_symbol
        assert _normalize_symbol("AAPL") == "AAPL"

    def test_btp_isin(self):
        from src.database import _normalize_symbol
        assert _normalize_symbol("IT0001234567") == "IT0001234567"


class TestAdaptSQL:
    """Test the _adapt_sql helper."""

    def test_sqlite_placeholder_conversion(self):
        from src.database import _adapt_sql
        # When backend is sqlite, %s should become ?
        with patch("src.database._backend", "sqlite"):
            result = _adapt_sql("SELECT * FROM trades WHERE id = %s")
            assert "?" in result
            assert "%s" not in result

    def test_postgresql_no_conversion(self):
        from src.database import _adapt_sql
        with patch("src.database._backend", "postgresql"):
            result = _adapt_sql("SELECT * FROM trades WHERE id = %s")
            assert "%s" in result
