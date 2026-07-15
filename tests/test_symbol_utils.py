import pytest
from src.utils.symbol_utils import is_btp_isin


def test_is_btp_isin_valid():
    assert is_btp_isin("IT0001234567") is True


def test_is_btp_isin_valid_with_suffix():
    assert is_btp_isin("IT0001234567/EUR") is True


def test_is_btp_isin_valid_with_quote_suffix():
    assert is_btp_isin("IT0001234567/QUOTE") is True


def test_is_btp_isin_invalid_non_it():
    assert is_btp_isin("US0001234567") is False


def test_is_btp_isin_invalid_short():
    assert is_btp_isin("IT000123456") is False


def test_is_btp_isin_invalid_long():
    assert is_btp_isin("IT00012345678") is False


def test_is_btp_isin_empty():
    assert is_btp_isin("") is False


def test_is_btp_isin_none():
    assert is_btp_isin(None) is False


def test_is_btp_isin_regular_stock():
    assert is_btp_isin("AAPL") is False


def test_is_btp_isin_regular_stock_with_suffix():
    assert is_btp_isin("ENI.MI") is False
