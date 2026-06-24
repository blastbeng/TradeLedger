from .base import Signal
from typing import Dict, Any, Optional

VALID_STRATEGY_TYPES = {"momentum", "mean_reversion", "breakout", "swing", "position"}


def validate_signal(
    signal: Signal,
    market_data: Optional[Dict[str, Any]] = None,
    atr: Optional[float] = None,
    price: Optional[float] = None,
    spread_pct: Optional[float] = None,
    timeframe_seconds: Optional[int] = None,
    min_stop_atr_mult: float = 1.0,
    min_hold_time_mult: float = 1.0,
    global_min_risk_reward_ratio: Optional[float] = None,
) -> Signal:
    """
    Validate a trading signal.
    - If action is HOLD, return as-is.
    - Validate strategy_type and required risk parameters.
    - Enforce risk/reward and ATR-based stop rules.
    Confidence is NOT used to reject trades; it will be used later for position sizing.
    """
    if signal.action == "HOLD":
        return signal

    # Require risk parameters for BUY/SELL
    if signal.action in ("BUY", "SELL"):
        params = signal.strategy_params or {}
        # Determine stop-loss method (default "fixed")
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method not in ("fixed", "atr_multiple"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_method")

        if stop_method == "atr_multiple":
            # stop_loss_atr_multiple is required
            if "stop_loss_atr_multiple" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing stop_loss_atr_multiple for atr_multiple method")
            atr_mult = params["stop_loss_atr_multiple"]
            if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_atr_multiple")
            # stop_loss_pct is REQUIRED as a fallback when ATR is unavailable at execution time
            if "stop_loss_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing stop_loss_pct (required as fallback for atr_multiple method)")
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
        else:  # "fixed"
            if "stop_loss_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: stop_loss_pct")
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")

        # Enforce minimum fixed stop-loss relative to ATR (if ATR and price are available)
        if stop_method == "fixed" and atr is not None and price is not None and price > 0 and atr > 0:
            atr_pct = atr / price
            min_sl = min_stop_atr_mult * atr_pct
            if sl < min_sl:
                return Signal(
                    action="HOLD",
                    confidence=0.0,
                    reasoning=(
                        f"Fixed stop-loss too tight: must be at least 1.5x ATR "
                        f"(ATR%={atr_pct:.4%}, stop_loss_pct={sl:.4%})"
                    )
                )

        # The rest of the required parameters remain unchanged
        required = ["take_profit_pct", "trailing_stop", "position_size_fraction", "max_hold_time_seconds"]
        for key in required:
            if key not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing required parameter: {key}")
        tp = params.get("take_profit_pct")
        tp_atr = params.get("take_profit_atr_multiple")
        tp_valid = tp is not None and isinstance(tp, (int, float)) and (0 < tp < 10.0)
        tp_atr_valid = tp_atr is not None and isinstance(tp_atr, (int, float)) and tp_atr > 0
        if not tp_valid and not tp_atr_valid:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing take_profit_pct or take_profit_atr_multiple")
        # When using ATR-based take-profit, take_profit_pct must also be valid as a fallback
        if tp_atr_valid and not tp_valid:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct is required as a fallback when using take_profit_atr_multiple")
        trailing = params["trailing_stop"]
        if not isinstance(trailing, bool):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop must be boolean")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            ts_atr = params.get("trailing_stop_atr_multiple")
            tsd_valid = tsd is not None and isinstance(tsd, (int, float)) and (0 < tsd < 1.0)
            ts_atr_valid = ts_atr is not None and isinstance(ts_atr, (int, float)) and ts_atr > 0
            if not tsd_valid and not ts_atr_valid:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing trailing_stop_distance_pct or trailing_stop_atr_multiple")
        psf = params["position_size_fraction"]
        if not isinstance(psf, (int, float)) or not (0 < psf <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")
        mht = params["max_hold_time_seconds"]
        if not isinstance(mht, (int, float)) or mht <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_hold_time_seconds")
        # Enforce a minimum max hold time relative to the candle timeframe
        if timeframe_seconds is not None and mht < min_hold_time_mult * timeframe_seconds:
            return Signal(
                action="HOLD",
                confidence=0.0,
                reasoning=(
                    f"max_hold_time_seconds ({mht}s) is too short for the "
                    f"timeframe ({timeframe_seconds}s candles); "
                    f"minimum is {min_hold_time_mult * timeframe_seconds}s"
                )
            )

        if "cooldown_after_loss_seconds" not in params:
            return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: cooldown_after_loss_seconds")
        cd = params["cooldown_after_loss_seconds"]
        if not isinstance(cd, (int, float)) or cd < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid cooldown_after_loss_seconds")

        # Optional new parameters
        if "trailing_stop_activation_pct" in params:
            tsa = params["trailing_stop_activation_pct"]
            if not isinstance(tsa, (int, float)) or not (0 <= tsa <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trailing_stop_activation_pct")
        if "max_risk_per_trade_pct" in params:
            mrp = params["max_risk_per_trade_pct"]
            if not isinstance(mrp, (int, float)) or not (0 < mrp <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_risk_per_trade_pct")
        if "min_profit_per_trade" in params:
            mpp = params["min_profit_per_trade"]
            if not isinstance(mpp, (int, float)) or mpp < 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_profit_per_trade")
        mrr = params.get("min_risk_reward_ratio")
        if mrr is None and global_min_risk_reward_ratio is not None:
            mrr = global_min_risk_reward_ratio
        if mrr is not None:
            if not isinstance(mrr, (int, float)) or mrr <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_risk_reward_ratio")
            # Enforce the ratio if both sl and tp are available
            if sl is not None and tp is not None:
                if tp / sl < mrr:
                    return Signal(
                        action="HOLD",
                        confidence=0.0,
                        reasoning=f"Risk/reward ratio {tp/sl:.2f} is below minimum {mrr:.2f}"
                    )
        if "min_confidence" in params:
            mc = params["min_confidence"]
            if not isinstance(mc, (int, float)) or not (0.0 <= mc <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_confidence")
        if "news_sentiment_exit_threshold" in params:
            nst = params["news_sentiment_exit_threshold"]
            if not isinstance(nst, (int, float)) or not (-1.0 <= nst <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid news_sentiment_exit_threshold")
        if "strategy_interval_seconds" in params:
            si = params["strategy_interval_seconds"]
            if not isinstance(si, (int, float)) or si <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid strategy_interval_seconds")
        if "backtest_period_days" in params:
            bpd = params["backtest_period_days"]
            if not isinstance(bpd, (int, float)) or bpd < 30:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_period_days (must be >= 30)")

        # Logical consistency checks (no hardcoded values)
        if sl is not None and tp <= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be greater than stop_loss_pct")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is not None and sl is not None and tsd >= sl:
                return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop_distance_pct must be less than stop_loss_pct")

    return signal
