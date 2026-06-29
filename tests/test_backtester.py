"""Unit tests for the backtesting module."""
import pytest
from src.strategies.backtester import backtest_strategy, format_backtest_summary, walk_forward_backtest


def _make_trending_candles(n=100, start_price=100.0, trend=0.001):
    """Generate candles with a slight upward trend."""
    candles = []
    ts = 1609459200000
    price = start_price
    for i in range(n):
        open_p = price
        close = price * (1 + trend)
        high = close * 1.01
        low = open_p * 0.99
        volume = 1000000.0
        candles.append([ts, open_p, high, low, close, volume])
        ts += 86400000
        price = close
    return candles


def _make_flat_candles(n=100, price=100.0):
    """Generate candles with no trend (flat market)."""
    candles = []
    ts = 1609459200000
    for i in range(n):
        high = price * 1.005
        low = price * 0.995
        candles.append([ts, price, high, low, price, 1000000.0])
        ts += 86400000
    return candles


class TestBacktestBasic:
    def test_empty_candles(self):
        result = backtest_strategy([], stop_loss_pct=0.05, take_profit_pct=0.10)
        assert result["insufficient_data"] is True
        assert result["total_trades"] == 0

    def test_insufficient_candles(self):
        result = backtest_strategy(
            [[0, 100, 101, 99, 100, 1000]],
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        assert result["insufficient_data"] is True

    def test_trending_market_produces_trades(self):
        candles = _make_trending_candles(100)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 10,
        )
        assert result["total_trades"] > 0
        assert result["insufficient_data"] is False
        assert "win_rate" in result
        assert "total_pnl_pct" in result
        assert "max_drawdown_pct" in result
        assert "profit_factor" in result

    def test_flat_market(self):
        candles = _make_flat_candles(100)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 10,
        )
        # In a flat market, most trades should hit stop-loss or max hold
        assert result["total_trades"] > 0


class TestBacktestStats:
    def test_win_rate_calculation(self):
        candles = _make_trending_candles(100, trend=0.005)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.01,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 30,
        )
        assert 0 <= result["win_rate"] <= 1.0
        assert result["wins"] + result["losses"] == result["total_trades"]

    def test_buy_and_hold_pct(self):
        candles = _make_trending_candles(100, start_price=100.0, trend=0.001)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.10,
            max_hold_time_seconds=86400 * 50,
        )
        # Buy and hold should be positive in a trending market
        assert result["buy_and_hold_pct"] > 0

    def test_max_consecutive_losses(self):
        candles = _make_flat_candles(100)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.01,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 5,
        )
        assert result["max_consecutive_losses"] >= 0

    def test_avg_hold_time(self):
        candles = _make_trending_candles(100)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 10,
        )
        assert result["avg_hold_time_seconds"] >= 0


class TestBacktestWithFees:
    def test_intesa_fee_model(self):
        candles = _make_trending_candles(100)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 10,
            fee_model="intesa",
            trade_value=10000.0,
        )
        assert result["total_trades"] > 0
        # Fees should reduce total P&L
        assert result["total_pnl_pct"] is not None


class TestBacktestTrailingStop:
    def test_trailing_stop_enabled(self):
        candles = _make_trending_candles(100, trend=0.003)
        result = backtest_strategy(
            candles,
            stop_loss_pct=0.02,
            take_profit_pct=0.10,
            max_hold_time_seconds=86400 * 30,
            trailing_stop=True,
            trailing_stop_distance_pct=0.02,
        )
        assert result["total_trades"] > 0


class TestFormatBacktestSummary:
    def test_summary_with_data(self):
        stats = {
            "total_trades": 10,
            "wins": 6,
            "losses": 4,
            "win_rate": 0.6,
            "avg_pnl_pct": 0.005,
            "total_pnl_pct": 0.05,
            "max_drawdown_pct": 0.03,
            "profit_factor": 1.5,
            "sharpe_ratio": 0.8,
            "avg_hold_time_seconds": 3600,
            "max_consecutive_losses": 2,
            "partial_tp_count": 0,
            "buy_and_hold_pct": 0.08,
            "insufficient_data": False,
        }
        summary = format_backtest_summary(stats)
        assert "10 trades" in summary
        assert "Win rate" in summary
        assert "60.0%" in summary

    def test_summary_insufficient_data(self):
        stats = {"insufficient_data": True, "total_trades": 0}
        summary = format_backtest_summary(stats)
        assert "Insufficient data" in summary


class TestWalkForward:
    def test_walk_forward_basic(self):
        candles = _make_trending_candles(100)
        result = walk_forward_backtest(
            candles,
            num_windows=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_hold_time_seconds=86400 * 10,
        )
        assert "per_window" in result
        assert "combined_stats" in result
        assert len(result["per_window"]) <= 3

    def test_walk_forward_insufficient_data(self):
        candles = _make_trending_candles(10)
        result = walk_forward_backtest(
            candles,
            num_windows=5,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )
        assert result["insufficient_data"] is True
