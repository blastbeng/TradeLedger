"""
Python-based backtesting module for evaluating LLM-proposed strategies.

Instead of asking the LLM to mentally backtest its strategy, we run a concrete
simulation on historical OHLCV data using the proposed stop-loss, take-profit,
max-hold-time, and trailing-stop parameters. The resulting statistics are fed
back to the LLM or used as a hard validation gate.
"""

import logging
from typing import Dict, Any, Optional, List

from src.indicators import compute_ema
from src.config.settings import settings

logger = logging.getLogger(__name__)


def _compute_intesa_fees(trade_value: float, side: str, is_btp: bool = False) -> float:
    """Compute Intesa Sanpaolo Investo fees for a trade."""
    if is_btp:
        commission = max(settings.BTP_MIN_FEE, trade_value * settings.BTP_FEE_PERC)
        return commission  # no fixed fee, no Tobin tax
    commission = max(settings.STOCK_FEE_MIN, trade_value * settings.STOCK_FEE_PERC)
    fixed_fee = settings.STOCK_FEE_FIXED
    tobin_tax = trade_value * settings.TOBIN_TAX_RATE if side == "buy" else 0.0
    return commission + fixed_fee + tobin_tax


def _compute_dynamic_slippage(
    candle_index: int,
    candles: List[List],
    avg_volumes: List[Optional[float]],
    atr_values: Optional[List[Optional[float]]],
    base_pct: float,
    max_pct: float,
) -> float:
    """Compute dynamic slippage based on relative volume and volatility at a given candle.

    - Low relative volume (current < average) → wider spread → higher slippage (up to 3× base).
    - High ATR% → more price movement between signal and fill → higher slippage.
    """
    slippage = base_pct

    # Volume adjustment
    current_vol = candles[candle_index][5] if candle_index < len(candles) else 0.0
    avg_vol = avg_volumes[candle_index] if candle_index < len(avg_volumes) else None
    if avg_vol is not None and avg_vol > 0 and current_vol > 0:
        vol_ratio = avg_vol / current_vol
        if vol_ratio > 1.0:
            slippage *= min(vol_ratio, 3.0)

    # Volatility adjustment
    atr = atr_values[candle_index] if atr_values and candle_index < len(atr_values) else None
    close = candles[candle_index][4] if candle_index < len(candles) else 0.0
    if atr is not None and atr > 0 and close > 0:
        atr_pct = atr / close
        slippage += atr_pct * 0.05

    return min(slippage, max_pct)


def _detect_gaps(candles: List[List], tolerance_mult: float = 1.5) -> Optional[List[Dict[str, Any]]]:
    """Detect gaps in OHLCV data by comparing consecutive timestamps.

    Returns a list of gap info dicts, or None if no significant gaps are found.
    Each dict contains: index, expected_interval_ms, actual_gap_ms, gap_ratio.
    """
    if len(candles) < 3:
        return None

    # Compute the expected interval from the median of consecutive differences
    intervals = []
    for i in range(1, len(candles)):
        diff = candles[i][0] - candles[i - 1][0]
        if diff > 0:
            intervals.append(diff)

    if not intervals:
        return None

    sorted_intervals = sorted(intervals)
    expected_interval = sorted_intervals[len(sorted_intervals) // 2]

    if expected_interval <= 0:
        return None

    gaps = []
    for i in range(1, len(candles)):
        actual_gap = candles[i][0] - candles[i - 1][0]
        if actual_gap > expected_interval * tolerance_mult:
            gaps.append({
                "index": i,
                "expected_interval_ms": expected_interval,
                "actual_gap_ms": actual_gap,
                "gap_ratio": round(actual_gap / expected_interval, 2),
            })

    return gaps if gaps else None


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
    stop_loss_atr_multiple: Optional[float] = None,
    take_profit_atr_multiple: Optional[float] = None,
    max_unrealized_loss_pct: Optional[float] = None,
    adx_values: Optional[List[Optional[float]]] = None,
    fee_rate: float = 0.0,
    fee_model: str = "flat",
    trade_value: Optional[float] = None,
    is_btp: bool = False,
    max_trades: int = 200,
    cooldown_after_loss_seconds: Optional[int] = None,
    slippage_pct: float = 0.0,
    slippage_model: str = "fixed",
    slippage_base_pct: float = 0.001,
    slippage_max_pct: float = 0.01,
    rsi_values: Optional[List[Optional[float]]] = None,
    max_rsi: float = 100.0,
    macd_hist_values: Optional[List[Optional[float]]] = None,
    backtest_entry_config: Optional[Dict[str, Any]] = None,
    simulate_position_sizing: bool = False,
    initial_balance: float = 10000.0,
    confidence: float = 0.5,
    confidence_sizing_weight: float = 0.0,
    global_risk_multiplier: float = 1.0,
    position_size_multiplier: float = 1.0,
    max_risk_per_trade_pct: Optional[float] = None,
    max_portfolio_risk_pct: Optional[float] = None,
    max_portfolio_exposure_pct: Optional[float] = None,
    max_portfolio_stop_risk_pct: Optional[float] = None,
    position_size_fraction: float = 0.1,
    direction: str = "long",
    gap_tolerance_mult: float = 1.5,
    on_gaps: str = "warn",
    _return_trades: bool = False,
) -> Dict[str, Any]:
    """
    Backtest a long, short, or both-direction strategy on historical OHLCV candles.

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

    if direction not in ("long", "short", "both"):
        direction = "long"

    if not backtest_entry_config:
        result = _empty_result()
        result["error"] = "backtest_entry_config is required — cannot default to entering every candle"
        if _return_trades:
            return [], result
        return result

    # --- Gap detection ---
    gap_warning = ""
    detected_gaps = _detect_gaps(candles, gap_tolerance_mult)
    if detected_gaps:
        gap_count = len(detected_gaps)
        max_gap_ratio = max(g["gap_ratio"] for g in detected_gaps)
        gap_warning = (
            f"⚠️ DATA GAPS DETECTED: {gap_count} gap(s) found in OHLCV data "
            f"(max gap ratio: {max_gap_ratio:.1f}x expected interval). "
            f"Backtest results may be inaccurate — gaps can cause false stop-loss "
            f"triggers or missed take-profit events."
        )
        logger.warning(f"Backtest gap detection: {gap_warning}")
        if on_gaps == "skip":
            result = _empty_result()
            result["error"] = gap_warning
            result["gap_warning"] = gap_warning
            if _return_trades:
                return [], result
            return result

    if direction == "both":
        long_trades, _ = backtest_strategy(
            candles=candles, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
            max_hold_time_seconds=max_hold_time_seconds, trailing_stop=trailing_stop,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            trailing_stop_activation_pct=trailing_stop_activation_pct,
            partial_take_profit_levels=partial_take_profit_levels,
            breakeven_activation_pct=breakeven_activation_pct,
            trailing_take_profit=trailing_take_profit,
            trailing_take_profit_distance_pct=trailing_take_profit_distance_pct,
            trailing_stop_atr_multiple=trailing_stop_atr_multiple,
            atr_values=atr_values, stop_loss_atr_multiple=stop_loss_atr_multiple,
            take_profit_atr_multiple=take_profit_atr_multiple,
            max_unrealized_loss_pct=max_unrealized_loss_pct,
            adx_values=adx_values, fee_rate=fee_rate, fee_model=fee_model,
            trade_value=trade_value, is_btp=is_btp, max_trades=max_trades,
            cooldown_after_loss_seconds=cooldown_after_loss_seconds,
            slippage_pct=slippage_pct,
            slippage_model=slippage_model,
            slippage_base_pct=slippage_base_pct,
            slippage_max_pct=slippage_max_pct,
            rsi_values=rsi_values, max_rsi=max_rsi,
            macd_hist_values=macd_hist_values,
            backtest_entry_config=backtest_entry_config,
            simulate_position_sizing=False,
            direction="long", _return_trades=True,
            gap_tolerance_mult=gap_tolerance_mult,
            on_gaps=on_gaps,
        )
        short_trades, _ = backtest_strategy(
            candles=candles, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
            max_hold_time_seconds=max_hold_time_seconds, trailing_stop=trailing_stop,
            trailing_stop_distance_pct=trailing_stop_distance_pct,
            trailing_stop_activation_pct=trailing_stop_activation_pct,
            partial_take_profit_levels=partial_take_profit_levels,
            breakeven_activation_pct=breakeven_activation_pct,
            trailing_take_profit=trailing_take_profit,
            trailing_take_profit_distance_pct=trailing_take_profit_distance_pct,
            trailing_stop_atr_multiple=trailing_stop_atr_multiple,
            atr_values=atr_values, stop_loss_atr_multiple=stop_loss_atr_multiple,
            take_profit_atr_multiple=take_profit_atr_multiple,
            max_unrealized_loss_pct=max_unrealized_loss_pct,
            adx_values=adx_values, fee_rate=fee_rate, fee_model=fee_model,
            trade_value=trade_value, is_btp=is_btp, max_trades=max_trades,
            cooldown_after_loss_seconds=cooldown_after_loss_seconds,
            slippage_pct=slippage_pct,
            slippage_model=slippage_model,
            slippage_base_pct=slippage_base_pct,
            slippage_max_pct=slippage_max_pct,
            rsi_values=rsi_values, max_rsi=max_rsi,
            macd_hist_values=macd_hist_values,
            backtest_entry_config=backtest_entry_config,
            simulate_position_sizing=False,
            direction="short", _return_trades=True,
            gap_tolerance_mult=gap_tolerance_mult,
            on_gaps=on_gaps,
        )
        all_trades = long_trades + short_trades
        buy_and_hold_pct = 0.0
        if len(candles) >= 2 and candles[0][4] > 0:
            buy_and_hold_pct = (candles[-1][4] - candles[0][4]) / candles[0][4]
        if not all_trades:
            result = _empty_result()
            result["gap_warning"] = gap_warning
            return result
        combined = _compute_stats(all_trades, buy_and_hold_pct=buy_and_hold_pct)
        combined["gap_warning"] = gap_warning
        return combined

    is_short = direction == "short"

    # --- Position sizing simulation state ---
    _psim = simulate_position_sizing and direction != "both"
    cash = initial_balance if _psim else 0.0
    total_pnl_currency = 0.0

    # Parse configurable entry logic
    entry_ema_period = backtest_entry_config.get("ema_period", 0)
    entry_ema_direction = backtest_entry_config.get("ema_direction", "above")
    entry_min_adx = backtest_entry_config.get("min_adx", 0.0)
    entry_max_rsi = backtest_entry_config.get("max_rsi", 100.0)
    entry_min_rsi = backtest_entry_config.get("min_rsi", 0.0)
    entry_macd_filter = backtest_entry_config.get("macd_filter", "none")
    entry_logic = backtest_entry_config.get("logic", "and")

    trades = []
    i = 0

    # Compute EMA for trend filter
    ema_values = []
    if entry_ema_period > 0 and len(candles) >= entry_ema_period:
        closes = [c[4] for c in candles]
        ema_values = compute_ema(closes, entry_ema_period)

    # Pre-compute rolling average volume for dynamic slippage
    avg_volume_series: List[Optional[float]] = []
    if slippage_model == "dynamic":
        vol_period = 20
        volumes = [c[5] for c in candles]
        for idx in range(len(candles)):
            start_idx = max(0, idx - vol_period + 1)
            window = volumes[start_idx:idx + 1]
            avg_volume_series.append(sum(window) / len(window) if window else None)

    while i < len(candles) - 1 and len(trades) < max_trades:
        # --- Configurable entry filters ---
        filter_results = []

        # EMA trend filter
        if entry_ema_period > 0 and ema_values:
            if i < len(ema_values) and ema_values[i] is not None:
                if entry_ema_direction == "above":
                    filter_results.append(candles[i][4] > ema_values[i])
                else:
                    filter_results.append(candles[i][4] < ema_values[i])
            else:
                filter_results.append(False)

        # ADX filter
        if entry_min_adx > 0 and adx_values is not None:
            if i < len(adx_values) and adx_values[i] is not None:
                filter_results.append(adx_values[i] >= entry_min_adx)
            else:
                filter_results.append(False)

        # RSI max filter (overbought)
        if entry_max_rsi > 0 and rsi_values is not None:
            if i < len(rsi_values) and rsi_values[i] is not None:
                filter_results.append(rsi_values[i] <= entry_max_rsi)
            else:
                filter_results.append(False)

        # RSI min filter (oversold)
        if entry_min_rsi > 0 and rsi_values is not None:
            if i < len(rsi_values) and rsi_values[i] is not None:
                filter_results.append(rsi_values[i] >= entry_min_rsi)
            else:
                filter_results.append(False)

        # MACD filter
        if entry_macd_filter != "none" and macd_hist_values is not None:
            if i < len(macd_hist_values) and macd_hist_values[i] is not None:
                if entry_macd_filter == "positive":
                    filter_results.append(macd_hist_values[i] > 0)
                else:
                    filter_results.append(macd_hist_values[i] < 0)
            else:
                filter_results.append(False)

        # Combine filters
        if filter_results:
            if entry_logic == "or":
                enter = any(filter_results)
            else:
                enter = all(filter_results)
            if not enter:
                i += 1
                continue

        entry_candle = candles[i]
        entry_ts = entry_candle[0]

        # Compute effective slippage for entry candle
        if slippage_model == "dynamic" and avg_volume_series:
            entry_slippage = _compute_dynamic_slippage(
                i, candles, avg_volume_series, atr_values,
                slippage_base_pct, slippage_max_pct,
            )
        else:
            entry_slippage = slippage_pct

        if is_short:
            entry_price = entry_candle[4] * (1 - entry_slippage)
        else:
            entry_price = entry_candle[4] * (1 + entry_slippage)

        if entry_price <= 0:
            i += 1
            continue

        # --- Compute position size for this trade ---
        if _psim:
            portfolio_value = cash
            desired_amount = portfolio_value * position_size_fraction
            if confidence_sizing_weight > 0 and confidence < 1.0:
                confidence_mult = 1.0 - confidence_sizing_weight * (1.0 - confidence)
                desired_amount *= confidence_mult
            desired_amount *= global_risk_multiplier
            desired_amount *= position_size_multiplier
            hard_max = float('inf')
            if max_risk_per_trade_pct is not None and stop_loss_pct > 0:
                hard_max = min(hard_max, (portfolio_value * max_risk_per_trade_pct) / stop_loss_pct)
            if max_portfolio_risk_pct is not None and stop_loss_pct > 0:
                hard_max = min(hard_max, (portfolio_value * max_portfolio_risk_pct) / stop_loss_pct)
            if max_portfolio_exposure_pct is not None:
                hard_max = min(hard_max, portfolio_value * max_portfolio_exposure_pct)
            if max_portfolio_stop_risk_pct is not None and stop_loss_pct > 0:
                hard_max = min(hard_max, (portfolio_value * max_portfolio_stop_risk_pct) / stop_loss_pct)
            hard_max = min(hard_max, cash)
            trade_amount = min(desired_amount, hard_max)
            if trade_amount <= 0:
                i += 1
                continue
        else:
            trade_amount = trade_value or 10000.0

        # Dynamic ATR-based stop-loss
        if stop_loss_atr_multiple is not None and atr_values is not None and i < len(atr_values) and atr_values[i] is not None and atr_values[i] > 0:
            if is_short:
                stop_loss_price = entry_price + (atr_values[i] * stop_loss_atr_multiple)
            else:
                stop_loss_price = entry_price - (atr_values[i] * stop_loss_atr_multiple)
        else:
            if is_short:
                stop_loss_price = entry_price * (1 + stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1 - stop_loss_pct)

        # Dynamic ATR-based take-profit
        if take_profit_atr_multiple is not None and atr_values is not None and i < len(atr_values) and atr_values[i] is not None and atr_values[i] > 0:
            if is_short:
                take_profit_price = entry_price - (atr_values[i] * take_profit_atr_multiple)
            else:
                take_profit_price = entry_price + (atr_values[i] * take_profit_atr_multiple)
        else:
            if is_short:
                take_profit_price = entry_price * (1 - take_profit_pct)
            else:
                take_profit_price = entry_price * (1 + take_profit_pct)

        # Trailing stop state
        if is_short:
            lowest_price = entry_price
        else:
            highest_price = entry_price
        trailing_activated = False
        trailing_stop_price = stop_loss_price

        # Partial take-profit state
        remaining_fraction = 1.0
        partial_tp_executed: set = set()
        partial_trades: List[Dict[str, Any]] = []
        entry_fee_charged = False

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

            # Compute effective slippage for this candle
            if slippage_model == "dynamic" and avg_volume_series:
                effective_slippage = _compute_dynamic_slippage(
                    j, candles, avg_volume_series, atr_values,
                    slippage_base_pct, slippage_max_pct,
                )
            else:
                effective_slippage = slippage_pct

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
                if is_short:
                    if candle_low < lowest_price:
                        lowest_price = candle_low
                    profit_pct = (entry_price - lowest_price) / entry_price
                else:
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
                            if is_short:
                                new_ts = lowest_price + (atr_values[j] * trailing_stop_atr_multiple)
                                if new_ts < trailing_stop_price:
                                    trailing_stop_price = new_ts
                            else:
                                new_ts = highest_price - (atr_values[j] * trailing_stop_atr_multiple)
                                if new_ts > trailing_stop_price:
                                    trailing_stop_price = new_ts
                    elif trailing_stop_distance_pct is not None:
                        if is_short:
                            new_ts = lowest_price * (1 + trailing_stop_distance_pct)
                            if new_ts < trailing_stop_price:
                                trailing_stop_price = new_ts
                        else:
                            new_ts = highest_price * (1 - trailing_stop_distance_pct)
                            if new_ts > trailing_stop_price:
                                trailing_stop_price = new_ts

                current_stop = trailing_stop_price
            else:
                current_stop = stop_loss_price

            # --- Breakeven stop ---
            if breakeven_activation_pct is not None and breakeven_activation_pct > 0:
                if is_short:
                    profit_pct = (entry_price - candle_low) / entry_price
                else:
                    profit_pct = (candle_high - entry_price) / entry_price
                if profit_pct >= breakeven_activation_pct:
                    breakeven_stop = entry_price
                    if is_short:
                        if breakeven_stop < current_stop:
                            current_stop = breakeven_stop
                    else:
                        if breakeven_stop > current_stop:
                            current_stop = breakeven_stop

            # --- Trailing take-profit ---
            if trailing_take_profit and trailing_take_profit_distance_pct is not None:
                if is_short:
                    new_tp = candle_low * (1 + trailing_take_profit_distance_pct)
                    if new_tp < take_profit_price:
                        take_profit_price = new_tp
                else:
                    new_tp = candle_high * (1 - trailing_take_profit_distance_pct)
                    if new_tp > take_profit_price:
                        take_profit_price = new_tp

            # --- Max unrealized loss (soft stop) ---
            if max_unrealized_loss_pct is not None and max_unrealized_loss_pct > 0:
                if is_short:
                    unrealized_high = (candle_high - entry_price) / entry_price
                    if unrealized_high >= max_unrealized_loss_pct:
                        target_exit = entry_price * (1 + max_unrealized_loss_pct)
                        if candle[1] >= target_exit:
                            exit_price = candle[1]
                        else:
                            exit_price = target_exit * (1 + effective_slippage)
                        exit_ts = candle_ts
                        exit_reason = "max_unrealized_loss"
                        exit_index = j
                        break
                else:
                    unrealized_low = (candle_low - entry_price) / entry_price
                    if unrealized_low <= -max_unrealized_loss_pct:
                        target_exit = entry_price * (1 - max_unrealized_loss_pct)
                        # If the candle opens below the soft stop, the fill happens at the open price
                        if candle[1] <= target_exit:
                            exit_price = candle[1]
                        else:
                            exit_price = target_exit * (1 - effective_slippage)
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
                    tp_target = entry_price * (1 + lvl_pct) if not is_short else entry_price * (1 - lvl_pct)
                    tp_triggered = (candle_high >= tp_target) if not is_short else (candle_low <= tp_target)
                    if tp_triggered:
                        if is_short:
                            actual_tp_fill = candle[1] if candle[1] <= tp_target else tp_target * (1 + effective_slippage)
                            partial_gross = (entry_price - actual_tp_fill) / entry_price * lvl_frac
                        else:
                            actual_tp_fill = candle[1] if candle[1] >= tp_target else tp_target * (1 - effective_slippage)
                            partial_gross = (actual_tp_fill - entry_price) / entry_price * lvl_frac
                        _fee_base = trade_amount if _psim else (trade_value or 10000.0)
                        if fee_model == "intesa" and _fee_base > 0:
                            entry_fee_pct = 0.0
                            if not entry_fee_charged:
                                entry_fee_pct = _compute_intesa_fees(_fee_base, "buy", is_btp) / _fee_base
                                entry_fee_charged = True
                            partial_exit_value = _fee_base * lvl_frac * (actual_tp_fill / entry_price)
                            partial_exit_fee_pct = _compute_intesa_fees(partial_exit_value, "sell", is_btp) / _fee_base
                            partial_net = partial_gross - entry_fee_pct - partial_exit_fee_pct
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
                            "trade_amount": trade_amount,
                            "pnl_currency": trade_amount * partial_net if _psim else 0.0,
                        })

            # --- If all partial TPs executed, exit the remaining position ---
            if remaining_fraction <= 0:
                exit_price = candle_close
                exit_ts = candle_ts
                exit_reason = "all_partial_tp"
                exit_index = j
                break

            # Check stop-loss (conservative: assume stop hits first if both hit in same candle)
            if is_short:
                stop_triggered = candle_high >= current_stop
            else:
                stop_triggered = candle_low <= current_stop
            if stop_triggered:
                if is_short:
                    if candle[1] >= current_stop:
                        exit_price = candle[1]
                    else:
                        exit_price = current_stop * (1 + effective_slippage)
                else:
                    if candle[1] <= current_stop:
                        exit_price = candle[1]
                    else:
                        exit_price = current_stop * (1 - effective_slippage)
                exit_ts = candle_ts
                exit_reason = "stop_loss"
                exit_index = j
                break

            # Check take-profit
            if is_short:
                tp_triggered = candle_low <= take_profit_price
            else:
                tp_triggered = candle_high >= take_profit_price
            if tp_triggered:
                if is_short:
                    if candle[1] <= take_profit_price:
                        exit_price = candle[1]
                    else:
                        exit_price = take_profit_price * (1 + effective_slippage)
                else:
                    if candle[1] >= take_profit_price:
                        exit_price = candle[1]
                    else:
                        exit_price = take_profit_price * (1 - effective_slippage)
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
            _fee_base = trade_amount if _psim else (trade_value or 10000.0)
            if fee_model == "intesa" and _fee_base > 0:
                entry_fee_pct = 0.0
                if not entry_fee_charged:
                    entry_fee_pct = _compute_intesa_fees(_fee_base, "buy", is_btp) / _fee_base
                    entry_fee_charged = True
                exit_trade_value = _fee_base * remaining_fraction * (exit_price / entry_price)
                exit_fee_pct = _compute_intesa_fees(exit_trade_value, "sell", is_btp) / _fee_base
                if is_short:
                    gross_pnl_pct = (entry_price - exit_price) / entry_price * remaining_fraction
                else:
                    gross_pnl_pct = (exit_price - entry_price) / entry_price * remaining_fraction
                net_pnl_pct = gross_pnl_pct - entry_fee_pct - exit_fee_pct
            else:
                entry_fee = entry_price * fee_rate
                exit_fee = exit_price * fee_rate
                if is_short:
                    gross_pnl_pct = (entry_price - exit_price) / entry_price * remaining_fraction
                else:
                    gross_pnl_pct = (exit_price - entry_price) / entry_price * remaining_fraction
                net_pnl_pct = gross_pnl_pct - (entry_fee + exit_fee) / entry_price * remaining_fraction

            hold_time_seconds = (exit_ts - entry_ts) / 1000.0

            trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": net_pnl_pct,
                "hold_time_seconds": hold_time_seconds,
                "trade_amount": trade_amount,
                "pnl_currency": trade_amount * net_pnl_pct if _psim else 0.0,
            })

        # --- Update portfolio after all trades for this entry ---
        if _psim:
            for pt in partial_trades:
                cash += pt["pnl_currency"]
            if remaining_fraction > 0 and trades:
                cash += trades[-1]["pnl_currency"]
            total_pnl_currency = cash - initial_balance

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

    # Compute buy-and-hold return
    buy_and_hold_pct = 0.0
    if len(candles) >= 2 and candles[0][4] > 0:
        buy_and_hold_pct = (candles[-1][4] - candles[0][4]) / candles[0][4]

    if not trades:
        if _return_trades:
            return [], _empty_result()
        return _empty_result()

    stats = _compute_stats(
        trades,
        buy_and_hold_pct=buy_and_hold_pct,
        initial_balance=initial_balance if _psim else 0.0,
        final_cash=cash if _psim else 0.0,
        total_pnl_currency=total_pnl_currency if _psim else 0.0,
        simulate_position_sizing=_psim,
    )
    stats["gap_warning"] = gap_warning
    if _return_trades:
        return trades, stats
    return stats


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
        "buy_and_hold_pct": 0.0,
        "insufficient_data": True,
        "error": "",
        "total_pnl_currency": 0.0,
        "final_balance": 0.0,
        "total_return_pct": 0.0,
        "gap_warning": "",
    }


def _compute_stats(
    trades: List[Dict[str, Any]],
    buy_and_hold_pct: float = 0.0,
    initial_balance: float = 0.0,
    final_cash: float = 0.0,
    total_pnl_currency: float = 0.0,
    simulate_position_sizing: bool = False,
) -> Dict[str, Any]:
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
        "buy_and_hold_pct": round(buy_and_hold_pct, 4),
        "insufficient_data": False,
        "total_pnl_currency": round(total_pnl_currency, 4) if simulate_position_sizing and initial_balance > 0 else 0.0,
        "final_balance": round(final_cash, 4) if simulate_position_sizing and initial_balance > 0 else 0.0,
        "total_return_pct": round((final_cash - initial_balance) / initial_balance, 4) if simulate_position_sizing and initial_balance > 0 else 0.0,
    }


def format_backtest_summary(stats: Dict[str, Any], entry_config_used: bool = True) -> str:
    """Format backtest statistics into a human-readable string for LLM prompts and notifications."""
    if stats.get("error"):
        return stats["error"]
    if stats.get("insufficient_data") or stats.get("total_trades", 0) == 0:
        return "Insufficient data for backtesting."

    entry_note = "" if entry_config_used else " [NO ENTRY FILTER — enters every candle]"
    portfolio_part = ""
    if stats.get("total_pnl_currency", 0) != 0 and stats.get("final_balance", 0) > 0:
        portfolio_part = (
            f", Portfolio P&L: {stats['total_pnl_currency']:+.2f}"
            f", Final balance: {stats['final_balance']:.2f}"
            f", Total return: {stats.get('total_return_pct', 0)*100:+.2f}%"
        )
    gap_part = ""
    if stats.get("gap_warning"):
        gap_part = f"\n{stats['gap_warning']}"
    return (
        f"Python backtest ({stats['total_trades']} trades){entry_note}: "
        f"Win rate: {stats['win_rate']*100:.1f}%, "
        f"Avg P&L: {stats['avg_pnl_pct']*100:+.2f}%, "
        f"Total P&L: {stats['total_pnl_pct']*100:+.2f}%, "
        f"Buy & Hold: {stats.get('buy_and_hold_pct', 0)*100:+.2f}%, "
        f"Max drawdown: {stats['max_drawdown_pct']*100:.2f}%, "
        f"Profit factor: {stats['profit_factor']:.2f}, "
        f"Sharpe ratio: {stats.get('sharpe_ratio', 0.0):.2f}, "
        f"Avg hold: {stats['avg_hold_time_seconds']/3600:.1f}h, "
        f"Max consec. losses: {stats['max_consecutive_losses']}, "
        f"Partial TPs: {stats.get('partial_tp_count', 0)}"
        f"{portfolio_part}"
        f"{gap_part}"
    )


def walk_forward_backtest(
    candles: List[List],
    num_windows: int = 5,
    **backtest_kwargs,
) -> Dict[str, Any]:
    """Run walk-forward analysis by splitting candles into non-overlapping windows.

    Runs backtest_strategy on each window separately, then combines all trades
    for aggregate out-of-sample stats.
    """
    if not candles or len(candles) < num_windows * 10:
        return {"insufficient_data": True, "per_window": [], "combined_stats": _empty_result()}

    backtest_kwargs.pop("_return_trades", None)

    window_size = len(candles) // num_windows
    per_window_stats = []
    all_trades = []

    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < num_windows - 1 else len(candles)
        window_candles = candles[start:end]
        if len(window_candles) < 5:
            continue

        result = backtest_strategy(
            candles=window_candles,
            _return_trades=True,
            **backtest_kwargs,
        )
        if isinstance(result, tuple):
            window_trades, window_stats = result
            all_trades.extend(window_trades)
        else:
            window_stats = result

        per_window_stats.append({
            "window": i + 1,
            "start_ts": window_candles[0][0],
            "end_ts": window_candles[-1][0],
            "candle_count": len(window_candles),
            **window_stats,
        })

    buy_and_hold_pct = 0.0
    if len(candles) >= 2 and candles[0][4] > 0:
        buy_and_hold_pct = (candles[-1][4] - candles[0][4]) / candles[0][4]
    combined = _compute_stats(all_trades, buy_and_hold_pct=buy_and_hold_pct) if all_trades else _empty_result()

    return {
        "per_window": per_window_stats,
        "combined_stats": combined,
        "num_windows": num_windows,
        "insufficient_data": False,
    }


def format_walk_forward_summary(wf_stats: Dict[str, Any]) -> str:
    if wf_stats.get("insufficient_data"):
        return "Insufficient data for walk-forward analysis."
    per_window = wf_stats.get("per_window", [])
    if not per_window:
        return "No walk-forward windows could be computed."
    # If all windows have errors, surface the first error
    window_errors = [w.get("error") for w in per_window if w.get("error")]
    if window_errors and len(window_errors) == len(per_window):
        return f"Walk-forward failed: {window_errors[0]}"
    lines = [f"Walk-forward ({len(per_window)} windows):"]
    for w in per_window:
        lines.append(
            f"  W{w['window']}: {w.get('total_trades', 0)} trades, "
            f"WR: {w.get('win_rate', 0)*100:.1f}%, "
            f"P&L: {w.get('total_pnl_pct', 0)*100:+.2f}%"
        )
    combined = wf_stats.get("combined_stats", {})
    lines.append(
        f"  Combined: {combined.get('total_trades', 0)} trades, "
        f"WR: {combined.get('win_rate', 0)*100:.1f}%, "
        f"P&L: {combined.get('total_pnl_pct', 0)*100:+.2f}%"
    )
    return "\n".join(lines)
