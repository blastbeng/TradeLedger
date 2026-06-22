"""
Python-based backtesting module for evaluating LLM-proposed strategies.

Instead of asking the LLM to mentally backtest its strategy, we run a concrete
simulation on historical OHLCV data using the proposed stop-loss, take-profit,
max-hold-time, and trailing-stop parameters. The resulting statistics are fed
back to the LLM or used as a hard validation gate.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def backtest_strategy(
    candles: List[List],
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_time_seconds: Optional[int] = None,
    trailing_stop: bool = False,
    trailing_stop_distance_pct: Optional[float] = None,
    trailing_stop_activation_pct: Optional[float] = None,
    fee_rate: float = 0.0,
    max_trades: int = 200,
) -> Dict[str, Any]:
    """
    Backtest a long-only strategy on historical OHLCV candles.

    Simulates entering a long position at the close of each candle, then tracks
    the position forward until stop-loss, take-profit, or max hold time is hit.
    After a trade closes, the next entry starts at the following candle.

    Args:
        candles: List of [timestamp_ms, open, high, low, close, volume]
        stop_loss_pct: Stop-loss as a fraction of entry price (e.g., 0.02 = 2%)
        take_profit_pct: Take-profit as a fraction of entry price (e.g., 0.05 = 5%)
        max_hold_time_seconds: Maximum hold time in seconds (None = no limit)
        trailing_stop: Whether to use a trailing stop
        trailing_stop_distance_pct: Trailing stop distance as a fraction of price
        trailing_stop_activation_pct: Profit % required before trailing stop activates
        fee_rate: Fee rate per trade (e.g., 0.001 = 0.1%)
        max_trades: Maximum number of trades to simulate (safety cap)

    Returns:
        Dict with keys: total_trades, wins, losses, win_rate, avg_pnl_pct,
        total_pnl_pct, max_drawdown_pct, profit_factor, avg_hold_time_seconds,
        max_consecutive_losses, insufficient_data
    """
    if not candles or len(candles) < 5:
        return _empty_result()

    if stop_loss_pct is None or stop_loss_pct <= 0:
        stop_loss_pct = 0.02
    if take_profit_pct is None or take_profit_pct <= 0:
        take_profit_pct = 0.05

    trades = []
    i = 0

    while i < len(candles) - 1 and len(trades) < max_trades:
        entry_candle = candles[i]
        entry_price = entry_candle[4]  # close
        entry_ts = entry_candle[0]

        if entry_price <= 0:
            i += 1
            continue

        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)

        # Trailing stop state
        highest_price = entry_price
        trailing_activated = False
        trailing_stop_price = stop_loss_price

        exit_price = None
        exit_ts = None
        exit_reason = None
        exit_index = len(candles) - 1

        for j in range(i + 1, len(candles)):
            candle = candles[j]
            candle_ts = candle[0]
            candle_high = candle[2]
            candle_low = candle[3]
            candle_close = candle[4]

            hold_time = (candle_ts - entry_ts) / 1000.0

            # Check max hold time
            if max_hold_time_seconds is not None and hold_time >= max_hold_time_seconds:
                exit_price = candle_close
                exit_ts = candle_ts
                exit_reason = "max_hold"
                exit_index = j
                break

            # Update trailing stop
            if trailing_stop and trailing_stop_distance_pct is not None:
                if candle_high > highest_price:
                    highest_price = candle_high

                profit_pct = (highest_price - entry_price) / entry_price

                if trailing_stop_activation_pct is not None:
                    if profit_pct >= trailing_stop_activation_pct:
                        trailing_activated = True

                if trailing_activated or trailing_stop_activation_pct is None:
                    new_ts = highest_price * (1 - trailing_stop_distance_pct)
                    if new_ts > trailing_stop_price:
                        trailing_stop_price = new_ts

                current_stop = trailing_stop_price
            else:
                current_stop = stop_loss_price

            # Check stop-loss (conservative: assume stop hits first if both hit in same candle)
            if candle_low <= current_stop:
                exit_price = current_stop
                exit_ts = candle_ts
                exit_reason = "stop_loss"
                exit_index = j
                break

            # Check take-profit
            if candle_high >= take_profit_price:
                exit_price = take_profit_price
                exit_ts = candle_ts
                exit_reason = "take_profit"
                exit_index = j
                break

        if exit_price is None:
            # Reached end of data without exit
            exit_price = candles[-1][4]
            exit_ts = candles[-1][0]
            exit_reason = "end_of_data"
            exit_index = len(candles) - 1

        # Calculate P&L (including fees)
        entry_fee = entry_price * fee_rate
        exit_fee = exit_price * fee_rate
        gross_pnl_pct = (exit_price - entry_price) / entry_price
        net_pnl_pct = gross_pnl_pct - (entry_fee + exit_fee) / entry_price

        hold_time_seconds = (exit_ts - entry_ts) / 1000.0

        trades.append({
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_pct": net_pnl_pct,
            "hold_time_seconds": hold_time_seconds,
        })

        # Move to the next candle after the exit
        i = exit_index + 1

    if not trades:
        return _empty_result()

    return _compute_stats(trades)


def _empty_result() -> Dict[str, Any]:
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "total_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "avg_hold_time_seconds": 0,
        "max_consecutive_losses": 0,
        "sharpe_ratio": 0.0,
        "insufficient_data": True,
    }


def _compute_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    total_pnl = sum(t["pnl_pct"] for t in trades)
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0

    # Max drawdown from cumulative P&L
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t["pnl_pct"]
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    max_consec_losses = 0
    current_consec = 0
    for t in trades:
        if t["pnl_pct"] <= 0:
            current_consec += 1
            max_consec_losses = max(max_consec_losses, current_consec)
        else:
            current_consec = 0

    avg_hold = sum(t["hold_time_seconds"] for t in trades) / len(trades)

    # Sharpe Ratio (per-trade, annualized approximation not needed for relative comparison)
    pnl_pcts = [t["pnl_pct"] for t in trades]
    mean_pnl = sum(pnl_pcts) / len(pnl_pcts)
    variance = sum((x - mean_pnl) ** 2 for x in pnl_pcts) / len(pnl_pcts)
    std_pnl = variance ** 0.5
    sharpe_ratio = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades), 4),
        "avg_pnl_pct": round(total_pnl / len(trades), 6),
        "total_pnl_pct": round(total_pnl, 6),
        "max_drawdown_pct": round(max_dd, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor < 999.0 else 999.0,
        "avg_hold_time_seconds": round(avg_hold),
        "max_consecutive_losses": max_consec_losses,
        "sharpe_ratio": round(sharpe_ratio, 4),
        "insufficient_data": False,
    }


def format_backtest_summary(stats: Dict[str, Any]) -> str:
    """Format backtest statistics into a human-readable string for LLM prompts and notifications."""
    if stats.get("insufficient_data") or stats.get("total_trades", 0) == 0:
        return "Insufficient data for backtesting."

    return (
        f"Python backtest ({stats['total_trades']} trades): "
        f"Win rate: {stats['win_rate']*100:.1f}%, "
        f"Avg P&L: {stats['avg_pnl_pct']*100:+.2f}%, "
        f"Total P&L: {stats['total_pnl_pct']*100:+.2f}%, "
        f"Max drawdown: {stats['max_drawdown_pct']*100:.2f}%, "
        f"Profit factor: {stats['profit_factor']:.2f}, "
        f"Sharpe ratio: {stats.get('sharpe_ratio', 0.0):.2f}, "
        f"Avg hold: {stats['avg_hold_time_seconds']/3600:.1f}h, "
        f"Max consec. losses: {stats['max_consecutive_losses']}"
    )
