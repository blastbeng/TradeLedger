import pytest

from src.strategies.backtester import backtest_strategy, BacktestConfig


def test_backtest_empty_candles():
    config = BacktestConfig()
    result = backtest_strategy(candles=[], config=config)
    assert isinstance(result, dict)
    assert "total_pnl" in result
    assert result["total_pnl"] == 0.0


def test_backtest_basic_run():
    candles = [
        [1700000000, 100.0, 105.0, 99.0, 103.0, 1000.0],
        [1700003600, 103.0, 108.0, 102.0, 107.0, 1200.0],
        [1700007200, 107.0, 110.0, 106.0, 109.0, 800.0],
    ]
    config = BacktestConfig()
    result = backtest_strategy(candles=candles, config=config)
    assert isinstance(result, dict)
    assert "total_pnl" in result
    assert "trades" in result
