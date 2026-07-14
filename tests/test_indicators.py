import pytest
from src.indicators import compute_atr, compute_rsi, compute_ema


def test_compute_atr():
    candles = [[i, 100 + i, 105 + i, 95 + i, 100 + i, 1000] for i in range(20)]
    result = compute_atr(candles, period=14)
    assert result is not None
    assert isinstance(result, float)


def test_compute_rsi():
    closes = [100 + i for i in range(20)]
    result = compute_rsi(closes, period=14)
    assert result is not None
    assert isinstance(result, float)


def test_compute_ema():
    data = [1.0 * i for i in range(20)]
    result = compute_ema(data, period=10)
    assert isinstance(result, list)
    assert len(result) == 20
    assert result[-1] is not None
