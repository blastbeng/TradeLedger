import pytest
from src.llm.cache import estimate_tokens, compute_market_hash


def test_estimate_tokens():
    assert estimate_tokens("12345678") == 2


def test_compute_market_hash_ignores_volatile_fields():
    data1 = {"price": 100.0, "timestamp": 123456789}
    data2 = {"price": 100.0, "timestamp": 987654321}
    hash1 = compute_market_hash(data1)
    hash2 = compute_market_hash(data2)
    assert hash1 == hash2
