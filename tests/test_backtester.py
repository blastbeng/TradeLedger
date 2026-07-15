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


# ---------- Backtest strategy with entry config ----------

def test_backtest_stop_loss_hit():
    """Stop loss should trigger when price drops below the stop level."""
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 100.5, 97.0, 97.5, 1000.0],  # low=97 < stop_loss=98
        [1007200, 97.5, 98.0, 97.0, 97.5, 1000.0],
        [1010800, 97.5, 98.0, 97.0, 97.5, 1000.0],
        [1014400, 97.5, 98.0, 97.0, 97.5, 1000.0],
        [1018000, 97.5, 98.0, 97.0, 97.5, 1000.0],
    ]
    config = BacktestConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        backtest_entry_config={"ema_period": 0},
    )
    result = backtest_strategy(candles=candles, config=config)
    assert result["total_trades"] > 0
    assert result["insufficient_data"] is False


def test_backtest_take_profit_hit():
    """Take profit should trigger when price rises above the take-profit level."""
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 106.0, 100.0, 105.5, 1000.0],  # high=106 > take_profit=105
        [1007200, 105.5, 106.0, 105.0, 105.5, 1000.0],
        [1010800, 105.5, 106.0, 105.0, 105.5, 1000.0],
        [1014400, 105.5, 106.0, 105.0, 105.5, 1000.0],
        [1018000, 105.5, 106.0, 105.0, 105.5, 1000.0],
    ]
    config = BacktestConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        backtest_entry_config={"ema_period": 0},
    )
    result = backtest_strategy(candles=candles, config=config)
    assert result["total_trades"] > 0
    assert result["wins"] >= 1


def test_backtest_max_hold_time():
    """Max hold time should force exit after the configured duration."""
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 101.0, 99.0, 100.5, 1000.0],  # hold_time=3.6s > 3
        [1007200, 100.5, 101.0, 99.0, 100.0, 1000.0],
        [1010800, 100.0, 101.0, 99.0, 100.5, 1000.0],
        [1014400, 100.5, 101.0, 99.0, 100.0, 1000.0],
        [1018000, 100.0, 101.0, 99.0, 100.5, 1000.0],
    ]
    config = BacktestConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        max_hold_time_seconds=3,
        backtest_entry_config={"ema_period": 0},
    )
    result = backtest_strategy(candles=candles, config=config)
    assert result["total_trades"] > 0


def test_backtest_no_entry_config_returns_error():
    """Without backtest_entry_config, the result should contain an error."""
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 101.0, 99.0, 100.5, 1000.0],
        [1007200, 100.5, 101.0, 99.0, 100.0, 1000.0],
        [1010800, 100.0, 101.0, 99.0, 100.5, 1000.0],
        [1014400, 100.5, 101.0, 99.0, 100.0, 1000.0],
    ]
    config = BacktestConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
    )
    result = backtest_strategy(candles=candles, config=config)
    assert result["insufficient_data"] is True
    assert "backtest_entry_config" in result.get("error", "")


# ---------- _compute_intesa_fees ----------

def test_compute_intesa_fees_buy():
    from src.strategies.backtester import _compute_intesa_fees
    # Default: perc=0.0024, min=3.50, fixed=2.50, tobin=0.0012
    # trade_value=10000: commission=max(3.50, 24.0)=24.0, fixed=2.50, tobin=12.0
    # total = 24.0 + 2.50 + 12.0 = 38.5
    result = _compute_intesa_fees(10000.0, "BUY")
    assert result == pytest.approx(38.5, rel=0.01)


def test_compute_intesa_fees_sell():
    from src.strategies.backtester import _compute_intesa_fees
    # SELL: no tobin tax
    # total = 24.0 + 2.50 = 26.5
    result = _compute_intesa_fees(10000.0, "SELL")
    assert result == pytest.approx(26.5, rel=0.01)


def test_compute_intesa_fees_small_trade():
    from src.strategies.backtester import _compute_intesa_fees
    # trade_value=100: commission=max(3.50, 0.24)=3.50, fixed=2.50, tobin=0.12
    # total = 3.50 + 2.50 + 0.12 = 6.12
    result = _compute_intesa_fees(100.0, "BUY")
    assert result == pytest.approx(6.12, rel=0.01)


def test_compute_intesa_fees_btp():
    from src.strategies.backtester import _compute_intesa_fees
    from unittest.mock import patch
    with patch("src.strategies.backtester.BTPPolicy.compute_fees", return_value=5.0):
        result = _compute_intesa_fees(10000.0, "BUY", is_btp=True)
        assert result == 5.0


# ---------- _compute_dynamic_slippage ----------

def test_compute_dynamic_slippage_basic():
    from src.strategies.backtester import _compute_dynamic_slippage
    candles = [[0, 100.0, 101.0, 99.0, 100.0, 1000.0]]
    avg_volumes = [1000.0]
    result = _compute_dynamic_slippage(0, candles, avg_volumes, None, 0.001, 0.01)
    # No volume ratio (current == avg), no ATR → base slippage
    assert result == pytest.approx(0.001, rel=0.01)


def test_compute_dynamic_slippage_with_volume():
    from src.strategies.backtester import _compute_dynamic_slippage
    candles = [[0, 100.0, 101.0, 99.0, 100.0, 500.0]]  # current vol = 500
    avg_volumes = [1000.0]  # avg = 1000, ratio = 2.0
    result = _compute_dynamic_slippage(0, candles, avg_volumes, None, 0.001, 0.01)
    # slippage = 0.001 * min(2.0, 3.0) = 0.002
    assert result == pytest.approx(0.002, rel=0.01)


def test_compute_dynamic_slippage_with_atr():
    from src.strategies.backtester import _compute_dynamic_slippage
    candles = [[0, 100.0, 101.0, 99.0, 100.0, 1000.0]]
    avg_volumes = [1000.0]
    atr_values = [2.0]  # ATR=2.0, close=100.0, atr_pct=0.02
    result = _compute_dynamic_slippage(0, candles, avg_volumes, atr_values, 0.001, 0.01)
    # slippage = 0.001 + 0.02 * 0.05 = 0.002
    assert result == pytest.approx(0.002, rel=0.01)


def test_compute_dynamic_slippage_capped():
    from src.strategies.backtester import _compute_dynamic_slippage
    candles = [[0, 100.0, 101.0, 99.0, 100.0, 100.0]]
    avg_volumes = [10000.0]  # huge ratio
    result = _compute_dynamic_slippage(0, candles, avg_volumes, None, 0.003, 0.005)
    # slippage = 0.003 * 3.0 = 0.009, capped at 0.005
    assert result == pytest.approx(0.005, rel=0.01)


def test_compute_dynamic_slippage_timeframe_scaling():
    from src.strategies.backtester import _compute_dynamic_slippage
    candles = [[0, 100.0, 101.0, 99.0, 100.0, 1000.0]]
    avg_volumes = [1000.0]
    # 1-hour timeframe: time_scale = (3600/60)^0.5 = sqrt(60) ≈ 7.746
    result = _compute_dynamic_slippage(0, candles, avg_volumes, None, 0.001, 0.1,
                                       timeframe_seconds=3600)
    assert result > 0.001  # should be scaled up


# ---------- _detect_gaps ----------

def test_detect_gaps_none():
    from src.strategies.backtester import _detect_gaps
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1007200, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1010800, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1014400, 100.0, 101.0, 99.0, 100.0, 1000.0],
    ]
    result = _detect_gaps(candles)
    assert result is None


def test_detect_gaps_found():
    from src.strategies.backtester import _detect_gaps
    candles = [
        [1000000, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1003600, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1007200, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1012700, 100.0, 101.0, 99.0, 100.0, 1000.0],  # gap: 5500 vs expected 3600
        [1016300, 100.0, 101.0, 99.0, 100.0, 1000.0],
        [1019900, 100.0, 101.0, 99.0, 100.0, 1000.0],
    ]
    result = _detect_gaps(candles)
    assert result is not None
    assert len(result) == 1
    assert result[0]["index"] == 3
    assert result[0]["gap_ratio"] == pytest.approx(1.53, rel=0.05)


def test_detect_gaps_too_few_candles():
    from src.strategies.backtester import _detect_gaps
    result = _detect_gaps([[1000, 100.0, 101.0, 99.0, 100.0, 1000.0]])
    assert result is None


# ---------- format_backtest_summary ----------

def test_format_backtest_summary_error():
    from src.strategies.backtester import format_backtest_summary
    stats = {"error": "Something went wrong"}
    result = format_backtest_summary(stats)
    assert result == "Something went wrong"


def test_format_backtest_summary_insufficient_data():
    from src.strategies.backtester import format_backtest_summary
    stats = {"insufficient_data": True, "total_trades": 0}
    result = format_backtest_summary(stats)
    assert result == "Insufficient data."


def test_format_backtest_summary_basic():
    from src.strategies.backtester import format_backtest_summary
    stats = {
        "total_trades": 10,
        "wins": 6,
        "losses": 4,
        "win_rate": 0.6,
        "avg_pnl_pct": 0.01,
        "total_pnl_pct": 0.10,
        "max_drawdown_pct": 0.05,
        "profit_factor": 1.5,
        "avg_hold_time_seconds": 3600,
        "max_consecutive_losses": 2,
        "sharpe_ratio": 1.2,
        "partial_tp_count": 0,
        "buy_and_hold_pct": 0.08,
        "insufficient_data": False,
        "error": "",
        "total_pnl_currency": 0.0,
        "final_balance": 0.0,
        "total_return_pct": 0.0,
        "gap_warning": "",
        "total_fees_pct": 0.0,
        "total_fees_currency": 0.0,
        "annualized_net_return": 0.0,
        "annualized_gross_return": 0.0,
    }
    result = format_backtest_summary(stats)
    assert "BT(10t)" in result
    assert "WR=60%" in result
    assert "PnL=+10.0%" in result
    assert "B&H=+8.0%" in result


def test_format_backtest_summary_with_gap_warning():
    from src.strategies.backtester import format_backtest_summary
    stats = {
        "total_trades": 5,
        "win_rate": 0.4,
        "total_pnl_pct": -0.02,
        "max_drawdown_pct": 0.03,
        "profit_factor": 0.8,
        "avg_hold_time_seconds": 1800,
        "max_consecutive_losses": 3,
        "sharpe_ratio": 0.5,
        "partial_tp_count": 0,
        "buy_and_hold_pct": 0.01,
        "insufficient_data": False,
        "error": "",
        "gap_warning": "⚠️ DATA GAPS DETECTED",
    }
    result = format_backtest_summary(stats)
    assert "⚠GAPS" in result


# ---------- _empty_result ----------

def test_empty_result():
    from src.strategies.backtester import _empty_result
    result = _empty_result()
    assert result["total_trades"] == 0
    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["win_rate"] == 0.0
    assert result["insufficient_data"] is True
    assert result["error"] == ""
    assert result["gap_warning"] == ""


# ---------- _compute_stats ----------

def test_compute_stats_basic():
    from src.strategies.backtester import _compute_stats
    trades = [
        {"pnl_pct": 0.05, "gross_pnl_pct": 0.06, "hold_time_seconds": 3600,
         "exit_reason": "take_profit"},
        {"pnl_pct": -0.02, "gross_pnl_pct": -0.01, "hold_time_seconds": 1800,
         "exit_reason": "stop_loss"},
    ]
    result = _compute_stats(trades)
    assert result["total_trades"] == 2
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["win_rate"] == 0.5
    assert result["avg_pnl_pct"] == pytest.approx(0.015, rel=0.01)
    assert result["total_pnl_pct"] == pytest.approx(0.03, rel=0.01)
    assert result["max_consecutive_losses"] == 1


def test_compute_stats_all_wins():
    from src.strategies.backtester import _compute_stats
    trades = [
        {"pnl_pct": 0.05, "gross_pnl_pct": 0.05, "hold_time_seconds": 3600,
         "exit_reason": "take_profit"},
        {"pnl_pct": 0.03, "gross_pnl_pct": 0.03, "hold_time_seconds": 1800,
         "exit_reason": "take_profit"},
    ]
    result = _compute_stats(trades)
    assert result["wins"] == 2
    assert result["losses"] == 0
    assert result["win_rate"] == 1.0
    assert result["max_consecutive_losses"] == 0


def test_compute_stats_consecutive_losses():
    from src.strategies.backtester import _compute_stats
    trades = [
        {"pnl_pct": -0.01, "gross_pnl_pct": -0.01, "hold_time_seconds": 100,
         "exit_reason": "stop_loss"},
        {"pnl_pct": -0.02, "gross_pnl_pct": -0.02, "hold_time_seconds": 200,
         "exit_reason": "stop_loss"},
        {"pnl_pct": -0.01, "gross_pnl_pct": -0.01, "hold_time_seconds": 150,
         "exit_reason": "stop_loss"},
    ]
    result = _compute_stats(trades)
    assert result["max_consecutive_losses"] == 3
    assert result["losses"] == 3
    assert result["wins"] == 0
