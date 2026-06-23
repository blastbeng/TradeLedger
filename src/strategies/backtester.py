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


def _compute_intesa_fees(trade_value: float, side: str, is_btp: bool = False) -> float:
    """Compute Intesa Sanpaolo Investo fees for a trade."""
    if is_btp:
        commission = max(3.50, trade_value * 0.0024)
        return commission  # no fixed fee, no Tobin tax
    commission = max(3.50, trade_value * 0.0024)
    fixed_fee = 2.50
    tobin_tax = trade_value * 0.0012 if side == "buy" else 0.0
    return commission + fixed_fee + tobin_tax


def backtest_strategy(
    candles: List[List],
    stop_loss_pct: float,
    take_profit_pct: float,
    max_hold_time_seconds: Optional[int] = None,
    trailing_stop: bool = False,
    trailing_stop_distance_pct: Optional[float] = None,
    trailing_stop_activation_pct: Optional[float] = None,
    partial_take_profit_levels: Optional[List[Dict]] = None,
    breakeven_activation_pct: Optional[float] = None,
    trailing_take_profit: bool = False,
    trailing_take_profit_distance_pct: Optional[float] = None,
    trailing_stop_atr_multiple: Optional[float] = None,
    atr_values: Optional[List[Optional[float]]] = None,
    max_unrealized_loss_pct: Optional[float] = None,
    fee_rate: float = 0.0,
    fee_model: str = "flat",
    trade_value: Optional[float] = None,
    is_btp: bool = False,
    max_trades: int = 200,
    cooldown_after_loss_seconds: Optional[int] = None,
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

        # Partial take-profit state
        remaining_fraction = 1.0
        partial_tp_executed: set = set()
        partial_trades: List[Dict[str, Any]] = []

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
            if trailing_stop and (
                trailing_stop_distance_pct is not None
                or (trailing_stop_atr_multiple is not None and atr_values is not None)
            ):
                if candle_high > highest_price:
                    highest_price = candle_high

                profit_pct = (highest_price - entry_price) / entry_price

                if trailing_stop_activation_pct is not None:
                    if profit_pct >= trailing_stop_activation_pct:
                        trailing_activated = True

                if trailing_activated or trailing_stop_activation_pct is None:
                    if trailing_stop_atr_multiple is not None and atr_values is not None:
                        # ATR-based trailing stop (Chandelier Exit)
                        if j < len(atr_values) and atr_values[j] is not None and atr_values[j] > 0:
                            new_ts = highest_price - (atr_values[j] * trailing_stop_atr_multiple)
                            if new_ts > trailing_stop_price:
                                trailing_stop_price = new_ts
                    elif trailing_stop_distance_pct is not None:
                        new_ts = highest_price * (1 - trailing_stop_distance_pct)
                        if new_ts > trailing_stop_price:
                            trailing_stop_price = new_ts

                current_stop = trailing_stop_price
            else:
                current_stop = stop_loss_price

            # --- Breakeven stop ---
            if breakeven_activation_pct is not None and breakeven_activation_pct > 0:
                profit_pct = (candle_high - entry_price) / entry_price
                if profit_pct >= breakeven_activation_pct:
                    breakeven_stop = entry_price
                    if breakeven_stop > current_stop:
                        current_stop = breakeven_stop

            # --- Trailing take-profit ---
            if trailing_take_profit and trailing_take_profit_distance_pct is not None:
                new_tp = candle_high * (1 - trailing_take_profit_distance_pct)
                if new_tp > take_profit_price:
                    take_profit_price = new_tp

            # --- Max unrealized loss (soft stop) ---
            if max_unrealized_loss_pct is not None and max_unrealized_loss_pct > 0:
                unrealized_low = (candle_low - entry_price) / entry_price
                if unrealized_low <= -max_unrealized_loss_pct:
                    target_exit = entry_price * (1 - max_unrealized_loss_pct)
                    # If the candle opens below the soft stop, the fill happens at the open price
                    if candle[1] <= target_exit:
                        exit_price = candle[1]
                    else:
                        exit_price = target_exit
                    exit_ts = candle_ts
                    exit_reason = "max_unrealized_loss"
                    exit_index = j
                    break

            # --- Partial take-profit levels ---
            if partial_take_profit_levels:
                for lvl_idx, level in enumerate(partial_take_profit_levels):
                    if lvl_idx in partial_tp_executed:
                        continue
                    lvl_pct = level.get("take_profit_pct", 0)
                    lvl_frac = level.get("fraction", 0)
                    if lvl_pct <= 0 or lvl_frac <= 0 or lvl_frac >= 1:
                        continue
                    tp_target = entry_price * (1 + lvl_pct)
                    if candle_high >= tp_target:
                        # If the candle opens above the target, the fill happens at the open price (gap up)
                        actual_tp_fill = candle[1] if candle[1] >= tp_target else tp_target
                        partial_gross = (actual_tp_fill - entry_price) / entry_price * lvl_frac
                        if fee_model == "intesa" and trade_value and trade_value > 0:
                            partial_entry_fee_pct = (
                                _compute_intesa_fees(trade_value, "buy", is_btp)
                                / trade_value * lvl_frac
                            )
                            partial_exit_value = trade_value * lvl_frac * (actual_tp_fill / entry_price)
                            partial_exit_fee_pct = (
                                _compute_intesa_fees(partial_exit_value, "sell", is_btp)
                                / trade_value
                            )
                            partial_net = partial_gross - partial_entry_fee_pct - partial_exit_fee_pct
                        else:
                            partial_net = partial_gross - (
                                entry_price * fee_rate + actual_tp_fill * fee_rate
                            ) / entry_price * lvl_frac
                        remaining_fraction *= (1 - lvl_frac)
                        partial_tp_executed.add(lvl_idx)
                        partial_trades.append({
                            "entry_price": entry_price,
                            "exit_price": actual_tp_fill,
                            "exit_reason": f"partial_tp_{lvl_idx}",
                            "pnl_pct": partial_net,
                            "hold_time_seconds": (candle_ts - entry_ts) / 1000.0,
                        })

            # --- If all partial TPs executed, exit the remaining position ---
            if remaining_fraction <= 0:
                exit_price = candle_close
                exit_ts = candle_ts
                exit_reason = "all_partial_tp"
                exit_index = j
                break

            # Check stop-loss (conservative: assume stop hits first if both hit in same candle)
            if candle_low <= current_stop:
                # If the candle opens below the stop, the fill happens at the open price (gap down)
                if candle[1] <= current_stop:
                    exit_price = candle[1]
                else:
                    exit_price = current_stop
                exit_ts = candle_ts
                exit_reason = "stop_loss"
                exit_index = j
                break

            # Check take-profit
            if candle_high >= take_profit_price:
                # If the candle opens above the target, the fill happens at the open price (gap up)
                if candle[1] >= take_profit_price:
                    exit_price = candle[1]
                else:
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

        # Record partial trades first
        for pt in partial_trades:
            trades.append(pt)

        # Calculate final P&L (including fees), scaled by remaining_fraction
        if remaining_fraction > 0:
            if fee_model == "intesa" and trade_value and trade_value > 0:
                entry_fee_pct = (
                    _compute_intesa_fees(trade_value, "buy", is_btp)
                    / trade_value * remaining_fraction
                )
                exit_trade_value = trade_value * remaining_fraction * (exit_price / entry_price)
                exit_fee_pct = (
                    _compute_intesa_fees(exit_trade_value, "sell", is_btp)
                    / trade_value
                )
                gross_pnl_pct = (exit_price - entry_price) / entry_price * remaining_fraction
                net_pnl_pct = gross_pnl_pct - entry_fee_pct - exit_fee_pct
            else:
                entry_fee = entry_price * fee_rate
                exit_fee = exit_price * fee_rate
                gross_pnl_pct = (exit_price - entry_price) / entry_price * remaining_fraction
                net_pnl_pct = gross_pnl_pct - (entry_fee + exit_fee) / entry_price * remaining_fraction

            hold_time_seconds = (exit_ts - entry_ts) / 1000.0

            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": net_pnl_pct,
                "hold_time_seconds": hold_time_seconds,
            })

        # Move to the next candle after the exit
        # Apply cooldown if the trade was a loss
        is_loss = trades[-1]["pnl_pct"] < 0 if trades else False

        if is_loss and cooldown_after_loss_seconds is not None and cooldown_after_loss_seconds > 0:
            cooldown_end_ts = exit_ts + cooldown_after_loss_seconds * 1000
            next_i = exit_index + 1
            while next_i < len(candles) and candles[next_i][0] < cooldown_end_ts:
                next_i += 1
            i = next_i
        else:
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
        "partial_tp_count": 0,
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

    partial_tp_count = sum(
        1 for t in trades if t.get("exit_reason", "").startswith("partial_tp_")
    )

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
        "partial_tp_count": partial_tp_count,
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
        f"Max consec. losses: {stats['max_consecutive_losses']}, "
        f"Partial TPs: {stats.get('partial_tp_count', 0)}"
    )
