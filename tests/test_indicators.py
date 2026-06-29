"""Unit tests for technical indicator calculations."""
import pytest
import numpy as np
from src.indicators import (
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_macd,
    compute_bollinger_bands,
    compute_stochastic,
    compute_adx,
    compute_obv,
    compute_vwap,
    compute_pivot_points,
    compute_all_indicators,
)


def _make_candles(n=60, base_price=100.0, volatility=0.02):
    """Generate synthetic OHLCV candles for testing."""
    candles = []
    ts = 1609459200000  # 2021-01-01
    price = base_price
    for i in range(n):
        open_p = price
        high = price * (1 + volatility * np.random.random())
        low = price * (1 - volatility * np.random.random())
        close = (high + low) / 2
        volume = 1000000 + np.random.randint(-100000, 100000)
        candles.append([ts, open_p, high, low, close, float(volume)])
        ts += 86400000  # 1 day
        price = close
    return candles


class TestATR:
    def test_atr_returns_float(self):
        candles = _make_candles(30)
        atr = compute_atr(candles, period=14)
        assert atr is not None
        assert isinstance(atr, float)
        assert atr > 0

    def test_atr_insufficient_data(self):
        candles = _make_candles(5)
        atr = compute_atr(candles, period=14)
        assert atr is None


class TestRSI:
    def test_rsi_in_range(self):
        candles = _make_candles(30)
        closes = [c[4] for c in candles]
        rsi = compute_rsi(closes, period=14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        closes = [100.0, 101.0, 102.0]
        rsi = compute_rsi(closes, period=14)
        assert rsi is None


class TestEMA:
    def test_ema_length(self):
        data = [100.0 + i for i in range(30)]
        ema = compute_ema(data, period=10)
        assert len(ema) == 30
        # First values should be None (warmup)
        assert ema[0] is None or isinstance(ema[0], float)

    def test_ema_insufficient_data(self):
        data = [100.0, 101.0]
        ema = compute_ema(data, period=10)
        assert ema == []


class TestMACD:
    def test_macd_returns_tuple(self):
        closes = [100.0 + i * 0.5 for i in range(50)]
        macd, signal, hist = compute_macd(closes)
        assert macd is not None
        assert signal is not None
        assert hist is not None

    def test_macd_insufficient_data(self):
        closes = [100.0, 101.0, 102.0]
        macd, signal, hist = compute_macd(closes)
        assert macd is None


class TestBollingerBands:
    def test_bb_ordering(self):
        closes = [100.0 + np.random.random() for _ in range(30)]
        upper, middle, lower = compute_bollinger_bands(closes)
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert upper >= middle >= lower


class TestStochastic:
    def test_stochastic_in_range(self):
        candles = _make_candles(30)
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        k, d = compute_stochastic(highs, lows, closes)
        assert k is not None
        assert d is not None
        assert 0 <= k <= 100
        assert 0 <= d <= 100


class TestADX:
    def test_adx_returns_values(self):
        candles = _make_candles(30)
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        adx, plus_di, minus_di = compute_adx(highs, lows, closes)
        assert adx is not None
        assert plus_di is not None
        assert minus_di is not None


class TestOBV:
    def test_obv_returns_float(self):
        closes = [100.0, 101.0, 99.0, 102.0]
        volumes = [1000.0, 2000.0, 1500.0, 3000.0]
        obv = compute_obv(closes, volumes)
        assert obv is not None
        assert isinstance(obv, float)


class TestVWAP:
    def test_vwap_calculation(self):
        candles = _make_candles(20)
        vwap = compute_vwap(candles, period=14)
        assert vwap is not None
        assert isinstance(vwap, float)
        assert vwap > 0

    def test_vwap_insufficient_data(self):
        candles = _make_candles(5)
        vwap = compute_vwap(candles, period=14)
        assert vwap is None


class TestPivotPoints:
    def test_pivot_calculation(self):
        pp = compute_pivot_points(105.0, 95.0, 100.0)
        assert pp["pivot"] == pytest.approx(100.0)
        assert pp["r1"] == pytest.approx(105.0)
        assert pp["s1"] == pytest.approx(95.0)
        # r2 = pivot + (high - low) = 100 + 10 = 110
        assert pp["r2"] == pytest.approx(110.0)
        assert pp["s2"] == pytest.approx(90.0)


class TestComputeAllIndicators:
    def test_all_indicators_dict(self):
        candles = _make_candles(60)
        ind = compute_all_indicators(candles)
        assert isinstance(ind, dict)
        assert "atr" in ind
        assert "rsi" in ind
        assert "macd" in ind
        assert "bb_upper" in ind
        assert "ema_9" in ind
        assert "adx" in ind

    def test_all_indicators_empty(self):
        ind = compute_all_indicators([])
        assert ind == {}

    def test_all_indicators_short(self):
        candles = _make_candles(5)
        ind = compute_all_indicators(candles)
        # With only 5 candles, most indicators won't compute
        assert "atr" in ind  # ATR key exists even if value is None
