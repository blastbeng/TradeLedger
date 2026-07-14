import logging
from .base import Signal
from typing import Dict, Any, Optional

from src.utils.btp_policy import BTPPolicy

logger = logging.getLogger(__name__)

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
    symbol: Optional[str] = None,
) -> Signal:
    """Wrapper that logs validation rejections before delegating to _validate_signal_impl."""
    result = _validate_signal_impl(
        signal, market_data, atr, price, spread_pct, timeframe_seconds,
        min_stop_atr_mult, min_hold_time_mult, global_min_risk_reward_ratio, symbol,
    )
    if result.action == "HOLD" and signal.action in ("BUY", "SELL") and result.reasoning:
        logger.warning(f"Validator: {signal.action}→HOLD for {symbol}: {result.reasoning}")
    return result


def _validate_backtest_entry_config(params: Dict[str, Any], symbol: Optional[str]) -> Optional[Signal]:
    """Validates backtest_entry_config for BUY signals. Returns a HOLD Signal on failure, None on success."""
    if "backtest_entry_config" not in params:
        # Default to a simple EMA trend filter
        params["backtest_entry_config"] = {
            "ema_period": 21,
            "ema_direction": "above",
            "min_adx": 20,
            "logic": "and",
        }
        logger.info(f"Validator: defaulting backtest_entry_config for {symbol}")
    bec = params["backtest_entry_config"]
    if not isinstance(bec, dict):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config (must be a dict)")

    ema_period = bec.get("ema_period", 0)
    if not isinstance(ema_period, (int, float)) or ema_period < 0:
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: ema_period must be a non-negative integer")
    ema_period = int(ema_period)
    bec["ema_period"] = ema_period

    ema_direction = bec.get("ema_direction", "above")
    if ema_direction not in ("above", "below"):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: ema_direction must be 'above' or 'below'")

    min_adx = bec.get("min_adx", 0.0)
    if not isinstance(min_adx, (int, float)) or min_adx < 0:
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: min_adx must be a non-negative number")

    max_rsi = bec.get("max_rsi", 100.0)
    if not isinstance(max_rsi, (int, float)) or not (0 <= max_rsi <= 100):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: max_rsi must be between 0 and 100")

    min_rsi = bec.get("min_rsi", 0.0)
    if not isinstance(min_rsi, (int, float)) or not (0 <= min_rsi <= 100):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: min_rsi must be between 0 and 100")

    macd_filter = bec.get("macd_filter", "none")
    if macd_filter not in ("none", "positive", "negative"):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: macd_filter must be 'none', 'positive', or 'negative'")

    logic = bec.get("logic", "and")
    if logic not in ("and", "or"):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_entry_config: logic must be 'and' or 'or'")

    return None


def _validate_stop_loss(
    params: Dict[str, Any],
    symbol: Optional[str],
    atr: Optional[float],
    price: Optional[float],
    timeframe_seconds: Optional[int],
    min_stop_atr_mult: float,
    tp: Optional[float],
) -> Optional[Signal]:
    """Validates stop-loss parameters. Returns a HOLD Signal on failure, None on success."""
    stop_method = params.get("stop_loss_method", "fixed")
    if stop_method not in ("fixed", "atr_multiple"):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_method")

    if stop_method == "atr_multiple":
        # stop_loss_atr_multiple is required
        if "stop_loss_atr_multiple" not in params:
            params["stop_loss_atr_multiple"] = 2.0
            logger.info(f"Validator: defaulting stop_loss_atr_multiple to 2.0 for {symbol}")
        atr_mult = params["stop_loss_atr_multiple"]
        if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_atr_multiple")
        # stop_loss_pct is used as a fallback when ATR is unavailable at execution time.
        # If the LLM omitted it, use a sensible default rather than rejecting the signal.
        if "stop_loss_pct" not in params:
            # Calculate from ATR multiplier when ATR is available
            if atr is not None and price is not None and price > 0 and atr > 0:
                sl = (atr_mult * atr) / price
                # For long timeframes, scale ATR to daily equivalent to avoid absurdly wide stops
                if timeframe_seconds is not None and timeframe_seconds > 86400:
                    daily_equiv = atr / price * (86400 / timeframe_seconds) ** 0.5
                    sl = min(sl, daily_equiv * atr_mult * 3)  # cap at 3x daily-equivalent ATR
                sl = max(sl, 0.01)  # floor at 1%
                sl = min(sl, 0.50)  # cap at 50%
            else:
                # ATR truly unavailable — use a conservative default scaled by timeframe
                if timeframe_seconds is not None and timeframe_seconds >= 31_536_000:  # >= 1Y
                    sl = 0.15  # 15% for very long timeframes
                elif timeframe_seconds is not None and timeframe_seconds >= 2_592_000:  # >= 1M
                    sl = 0.10  # 10% for long timeframes
                else:
                    sl = 0.05  # 5% for medium-term
            # Ensure default sl < tp if tp is already known
            if tp is not None and sl >= tp:
                sl = tp * 0.5  # Half the take-profit as a sensible stop
                logger.info(f"Validator: adjusted default stop_loss_pct to {sl:.4f} (must be < take_profit_pct={tp:.4f}) for {symbol}")
            params["stop_loss_pct"] = sl
            logger.info(f"Validator: using default stop_loss_pct={sl} for {symbol} (atr_multiple method)")
        else:
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
    else:  # "fixed"
        if "stop_loss_pct" not in params:
            # Calculate from ATR when available, using min_stop_atr_mult as the floor
            if atr is not None and price is not None and price > 0 and atr > 0:
                atr_pct = atr / price
                # For long timeframes, scale ATR to daily equivalent
                if timeframe_seconds is not None and timeframe_seconds > 86400:
                    atr_pct = atr_pct * (86400 / timeframe_seconds) ** 0.5
                sl = max(min_stop_atr_mult * atr_pct, 2.0 * atr_pct)  # at least 2x ATR, but not below validator minimum
                sl = max(sl, 0.01)  # floor at 1%
                sl = min(sl, 0.50)  # cap at 50%
            else:
                # ATR truly unavailable — scale by timeframe
                if timeframe_seconds is not None and timeframe_seconds >= 31_536_000:
                    sl = 0.15
                elif timeframe_seconds is not None and timeframe_seconds >= 2_592_000:
                    sl = 0.10
                else:
                    sl = 0.05
            # Ensure default sl < tp if tp is already known
            if tp is not None and sl >= tp:
                sl = tp * 0.5
                logger.info(f"Validator: adjusted default stop_loss_pct to {sl:.4f} (must be < take_profit_pct={tp:.4f}) for {symbol}")
            params["stop_loss_pct"] = sl
            logger.info(f"Validator: using default stop_loss_pct={sl} for {symbol}")
        else:
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")

    # Enforce minimum fixed stop-loss relative to ATR (if ATR and price are available)
    if stop_method == "fixed" and atr is not None and price is not None and price > 0 and atr > 0:
        atr_pct = atr / price
        # Scale ATR to a daily equivalent if the timeframe is longer than 1 day
        # to avoid absurdly wide stop-loss requirements for long timeframes (e.g., 5Y)
        if timeframe_seconds is not None and timeframe_seconds > 86400:
            atr_pct = atr_pct * (86400 / timeframe_seconds) ** 0.5
        min_sl = min_stop_atr_mult * atr_pct
        if sl < min_sl:
            # Instead of rejecting, adjust the stop-loss to the minimum required
            sl = min_sl
            params["stop_loss_pct"] = sl
            logger.info(f"Validator: adjusted stop_loss_pct to {sl:.4f} (minimum {min_stop_atr_mult}x ATR) for {symbol}")
            # Re-check consistency with take_profit
            if tp is not None and sl >= tp:
                tp = sl * 1.5
                params["take_profit_pct"] = tp
                logger.info(f"Validator: adjusted take_profit_pct to {tp:.4f} (must be > adjusted stop_loss_pct={sl:.4f}) for {symbol}")

    return None


def _validate_take_profit(
    params: Dict[str, Any],
    symbol: Optional[str],
    atr: Optional[float],
    price: Optional[float],
    timeframe_seconds: Optional[int],
    sl: Optional[float],
) -> Optional[Signal]:
    """Validates take-profit parameters. Returns a HOLD Signal on failure, None on success."""
    tp = params.get("take_profit_pct")
    tp_atr = params.get("take_profit_atr_multiple")
    tp_valid = tp is not None and isinstance(tp, (int, float)) and (0 < tp < 10.0)
    tp_atr_valid = tp_atr is not None and isinstance(tp_atr, (int, float)) and tp_atr > 0
    if not tp_valid and not tp_atr_valid:
        # Calculate from ATR when available, ensuring tp > sl with a 1.5:1 reward:risk floor
        if atr is not None and price is not None and price > 0 and atr > 0:
            atr_pct = atr / price
            if timeframe_seconds is not None and timeframe_seconds > 86400:
                atr_pct = atr_pct * (86400 / timeframe_seconds) ** 0.5
            tp = max(3.0 * atr_pct, sl * 1.5)  # 3x ATR or 1.5x stop-loss, whichever is greater
            tp = min(tp, 5.0)  # cap at 500%
        else:
            # Scale by timeframe when ATR unavailable
            if timeframe_seconds is not None and timeframe_seconds >= 31_536_000:
                tp = 0.30  # 30% for very long timeframes
            elif timeframe_seconds is not None and timeframe_seconds >= 2_592_000:
                tp = 0.20  # 20% for long timeframes
            else:
                tp = 0.10  # 10% for medium-term
            # Ensure tp > sl
            if sl is not None and tp <= sl:
                tp = sl * 1.5
        # Ensure default tp > sl to avoid logical consistency rejection
        if sl is not None and tp <= sl:
            tp = sl * 1.5  # 1.5x the stop-loss as a minimum viable take-profit
            logger.info(f"Validator: adjusted default take_profit_pct to {tp:.4f} (must be > stop_loss_pct={sl:.4f}) for {symbol}")
        params["take_profit_pct"] = tp
        tp_valid = True
        logger.info(f"Validator: using default take_profit_pct={tp} for {symbol}")
    # When using ATR-based take-profit, take_profit_pct is used as a fallback.
    # If the LLM omitted it, compute a default from the ATR multiplier or use a sensible default.
    if tp_atr_valid and not tp_valid:
        if atr is not None and price is not None and price > 0 and atr > 0:
            tp = (tp_atr * atr) / price
        else:
            tp = 0.10  # 10% default for long timeframes
        # Ensure default tp > sl to avoid logical consistency rejection
        if sl is not None and tp <= sl:
            tp = sl * 1.5
            logger.info(f"Validator: adjusted default take_profit_pct to {tp:.4f} (must be > stop_loss_pct={sl:.4f}) for {symbol}")
        params["take_profit_pct"] = tp
        tp_valid = True
        logger.info(f"Validator: using default take_profit_pct={tp} for {symbol} (ATR fallback for atr_multiple take-profit)")
    return None


def _validate_trailing_stop(
    params: Dict[str, Any],
    symbol: Optional[str],
    atr: Optional[float],
    price: Optional[float],
    timeframe_seconds: Optional[int],
    sl: Optional[float],
) -> Optional[Signal]:
    """Validates trailing_stop parameters. Returns a HOLD Signal on failure, None on success."""
    trailing = params["trailing_stop"]
    if not isinstance(trailing, bool):
        return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop must be boolean")
    if trailing:
        if symbol and not BTPPolicy.supports_trailing_stop(symbol):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop is not supported for BTP symbols")
        tsd = params.get("trailing_stop_distance_pct")
        ts_atr = params.get("trailing_stop_atr_multiple")
        tsd_valid = tsd is not None and isinstance(tsd, (int, float)) and (0 < tsd < 1.0)
        ts_atr_valid = ts_atr is not None and isinstance(ts_atr, (int, float)) and ts_atr > 0
        if not tsd_valid and not ts_atr_valid:
            # Calculate from ATR when available
            if atr is not None and price is not None and price > 0 and atr > 0:
                atr_pct = atr / price
                if timeframe_seconds is not None and timeframe_seconds > 86400:
                    atr_pct = atr_pct * (86400 / timeframe_seconds) ** 0.5
                tsd = max(1.5 * atr_pct, sl * 0.5)  # 1.5x ATR or half the stop-loss
                tsd = min(tsd, 0.20)  # cap at 20%
            else:
                tsd = sl * 0.5 if sl is not None else 0.03  # half the stop-loss, or 3% fallback
            # Ensure default tsd < sl to avoid logical consistency rejection
            if sl is not None and tsd >= sl:
                tsd = sl * 0.5  # Half the stop-loss as a sensible trailing distance
                logger.info(f"Validator: adjusted default trailing_stop_distance_pct to {tsd:.4f} (must be < stop_loss_pct={sl:.4f}) for {symbol}")
            params["trailing_stop_distance_pct"] = tsd
            tsd_valid = True
            logger.info(f"Validator: using default trailing_stop_distance_pct={tsd} for {symbol}")
    return None


def _validate_optional_params(
    params: Dict[str, Any],
    symbol: Optional[str],
    sl: Optional[float],
    tp: Optional[float],
    stop_method: str,
    tp_atr_valid: bool,
    atr_mult: Optional[float],
    tp_atr: Optional[float],
    trailing: bool,
    global_min_risk_reward_ratio: Optional[float],
) -> Optional[Signal]:
    """Validates optional strategy parameters. Returns a HOLD Signal on failure, None on success."""
    if "cooldown_after_loss_seconds" not in params:
        params["cooldown_after_loss_seconds"] = 0
        logger.info(f"Validator: defaulting cooldown_after_loss_seconds to 0 for {symbol}")
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
        if not isinstance(mrp, (int, float)) or not (0 <= mrp <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_risk_per_trade_pct")
    if "min_profit_per_trade" in params:
        mpp = params["min_profit_per_trade"]
        if not isinstance(mpp, (int, float)) or mpp < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_profit_per_trade")
    mrr = params.get("min_risk_reward_ratio")
    if mrr is None and global_min_risk_reward_ratio is not None:
        mrr = global_min_risk_reward_ratio
    if mrr is not None:
        if not isinstance(mrr, (int, float)) or mrr < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_risk_reward_ratio")
        # Enforce the ratio if both sl and tp are available
        if sl is not None and tp is not None:
            # If ATR-based stops/TPs are provided, use the ATR multipliers for the ratio
            if tp_atr is not None and stop_method == "atr_multiple":
                actual_ratio = tp_atr / atr_mult
            else:
                actual_ratio = tp / sl
            
            if mrr > 0 and actual_ratio < mrr:
                return Signal(
                    action="HOLD",
                    confidence=0.0,
                    reasoning=f"Risk/reward ratio {actual_ratio:.2f} is below minimum {mrr:.2f}"
                )
    if "min_confidence" in params:
        mc = params["min_confidence"]
        if not isinstance(mc, (int, float)) or not (0.0 <= mc <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_confidence")
    if "news_sentiment_exit_threshold" in params:
        nst = params["news_sentiment_exit_threshold"]
        if not isinstance(nst, (int, float)) or not (-1.0 <= nst <= 0.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid news_sentiment_exit_threshold (must be between -1.0 and 0.0)")
    if "strategy_interval_seconds" in params:
        si = params["strategy_interval_seconds"]
        if not isinstance(si, (int, float)) or si <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid strategy_interval_seconds")
    if "backtest_period_days" in params:
        bpd = params["backtest_period_days"]
        if not isinstance(bpd, (int, float)) or bpd < 30:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid backtest_period_days (must be >= 30)")

    if "partial_take_profit_levels" in params:
        ptpl = params["partial_take_profit_levels"]
        if not isinstance(ptpl, list):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_levels (must be a list)")
        for level in ptpl:
            if not isinstance(level, dict):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_levels (items must be dicts)")
            lvl_pct = level.get("take_profit_pct")
            lvl_frac = level.get("fraction")
            if not isinstance(lvl_pct, (int, float)) or lvl_pct <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_levels: take_profit_pct must be > 0")
            if not isinstance(lvl_frac, (int, float)) or not (0 < lvl_frac < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_levels: fraction must be between 0 and 1")

    if "breakeven_activation_pct" in params:
        bea = params["breakeven_activation_pct"]
        if not isinstance(bea, (int, float)) or not (0 < bea <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid breakeven_activation_pct")

    if "trailing_take_profit" in params:
        ttp = params["trailing_take_profit"]
        if not isinstance(ttp, bool):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_take_profit must be boolean")
        if ttp:
            ttpd = params.get("trailing_take_profit_distance_pct")
            if not isinstance(ttpd, (int, float)) or not (0 < ttpd < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing trailing_take_profit_distance_pct")

    if "max_unrealized_loss_pct" in params:
        mul = params["max_unrealized_loss_pct"]
        if not isinstance(mul, (int, float)) or mul < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_unrealized_loss_pct")

    if "max_portfolio_risk_pct" in params:
        mpr = params["max_portfolio_risk_pct"]
        if not isinstance(mpr, (int, float)) or not (0 <= mpr <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_portfolio_risk_pct")

    if "max_portfolio_exposure_pct" in params:
        mpe = params["max_portfolio_exposure_pct"]
        if not isinstance(mpe, (int, float)) or not (0 <= mpe <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_portfolio_exposure_pct")

    if "max_portfolio_stop_risk_pct" in params:
        mps = params["max_portfolio_stop_risk_pct"]
        if not isinstance(mps, (int, float)) or not (0 <= mps <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_portfolio_stop_risk_pct")

    if "direction" in params:
        d = params["direction"]
        if d not in ("long", "short", "both"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid direction (must be 'long', 'short', or 'both')")

    if "fee_model" in params:
        fm = params["fee_model"]
        if fm not in ("flat", "intesa"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid fee_model (must be 'flat' or 'intesa')")

    if "slippage_model" in params:
        sm = params["slippage_model"]
        if sm not in ("fixed", "dynamic"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid slippage_model (must be 'fixed' or 'dynamic')")

    if "slippage_pct" in params:
        sp = params["slippage_pct"]
        if not isinstance(sp, (int, float)) or sp < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid slippage_pct")

    if "slippage_base_pct" in params:
        sbp = params["slippage_base_pct"]
        if not isinstance(sbp, (int, float)) or sbp <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid slippage_base_pct")

    if "slippage_max_pct" in params:
        smp = params["slippage_max_pct"]
        if not isinstance(smp, (int, float)) or smp <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid slippage_max_pct")

    if "simulate_position_sizing" in params:
        sps = params["simulate_position_sizing"]
        if not isinstance(sps, bool):
            return Signal(action="HOLD", confidence=0.0, reasoning="simulate_position_sizing must be boolean")

    if "global_risk_multiplier" in params:
        grm = params["global_risk_multiplier"]
        if not isinstance(grm, (int, float)) or grm < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid global_risk_multiplier")

    if "position_size_multiplier" in params:
        psm = params["position_size_multiplier"]
        if not isinstance(psm, (int, float)) or psm < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_multiplier")

    if "confidence_sizing_weight" in params:
        csw = params["confidence_sizing_weight"]
        if not isinstance(csw, (int, float)) or csw < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid confidence_sizing_weight")

    if "gap_tolerance_mult" in params:
        gtm = params["gap_tolerance_mult"]
        if not isinstance(gtm, (int, float)) or gtm <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid gap_tolerance_mult")

    if "on_gaps" in params:
        og = params["on_gaps"]
        if og not in ("warn", "skip"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid on_gaps (must be 'warn' or 'skip')")

    if "fee_rate" in params:
        fr = params["fee_rate"]
        if not isinstance(fr, (int, float)) or fr < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid fee_rate")

    if "max_trades" in params:
        mt = params["max_trades"]
        if not isinstance(mt, (int, float)) or mt <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_trades")
        params["max_trades"] = int(mt)

    if "initial_balance" in params:
        ib = params["initial_balance"]
        if not isinstance(ib, (int, float)) or ib <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid initial_balance")

    if "trade_value" in params:
        tv = params["trade_value"]
        if not isinstance(tv, (int, float)) or tv <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trade_value")

    return None


def _apply_required_defaults(params: Dict[str, Any], symbol: Optional[str], timeframe_seconds: Optional[int]) -> None:
    """Applies default values for required strategy parameters if missing."""
    required = ["trailing_stop", "position_size_fraction", "max_hold_time_seconds"]
    for key in required:
        if key not in params:
            if key == "trailing_stop":
                params["trailing_stop"] = False
                logger.info(f"Validator: defaulting trailing_stop to False for {symbol}")
            elif key == "position_size_fraction":
                # Calculate from risk budget if stop_loss_pct and max_risk_per_trade_pct are available
                _sl_for_sizing = params.get("stop_loss_pct")
                _max_risk = params.get("max_risk_per_trade_pct")
                if _sl_for_sizing is not None and _max_risk is not None and _sl_for_sizing > 0:
                    _calculated_psf = min(_max_risk / _sl_for_sizing, 1.0)
                    params["position_size_fraction"] = max(0.01, _calculated_psf)
                    logger.info(f"Validator: calculated position_size_fraction={_calculated_psf:.4f} from max_risk={_max_risk} / sl={_sl_for_sizing} for {symbol}")
                else:
                    # Fallback: scale by timeframe
                    if timeframe_seconds is not None and timeframe_seconds >= 31_536_000:
                        params["position_size_fraction"] = 0.25  # larger positions for long-term (wider stops, smaller risk)
                    elif timeframe_seconds is not None and timeframe_seconds >= 2_592_000:
                        params["position_size_fraction"] = 0.15
                    else:
                        params["position_size_fraction"] = 0.10
                    logger.info(f"Validator: defaulting position_size_fraction to {params['position_size_fraction']} for {symbol}")
            elif key == "max_hold_time_seconds":
                # Default to a reasonable multiple of the timeframe
                if timeframe_seconds is not None:
                    params["max_hold_time_seconds"] = min(int(timeframe_seconds * 10), 157_680_000)
                else:
                    params["max_hold_time_seconds"] = 2_592_000  # 30 days
                logger.info(f"Validator: defaulting max_hold_time_seconds to {params['max_hold_time_seconds']} for {symbol}")


def _validate_required_params(
    params: Dict[str, Any],
    timeframe_seconds: Optional[int],
    min_hold_time_mult: float,
) -> Optional[Signal]:
    """Validates required strategy parameters. Returns a HOLD Signal on failure, None on success."""
    psf = params["position_size_fraction"]
    if not isinstance(psf, (int, float)) or not (0 < psf <= 1.0):
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")
    mht = params["max_hold_time_seconds"]
    if not isinstance(mht, (int, float)) or mht <= 0:
        return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_hold_time_seconds")
    # Enforce a minimum max hold time relative to the candle timeframe
    if timeframe_seconds is not None:
        # Cap the minimum hold time to avoid absurd values for very long timeframes (e.g., 5Y)
        min_hold = min(min_hold_time_mult * timeframe_seconds, 157_680_000)  # cap at ~5 years
        if mht < min_hold:
            return Signal(
                action="HOLD",
                confidence=0.0,
                reasoning=(
                    f"max_hold_time_seconds ({mht}s) is too short for the "
                    f"timeframe ({timeframe_seconds}s candles); "
                    f"minimum is {min_hold}s"
                )
            )
    return None


def _validate_logical_consistency(
    params: Dict[str, Any],
    symbol: Optional[str],
    sl: Optional[float],
    tp: Optional[float],
    stop_method: str,
    tp_atr_valid: bool,
    trailing: bool,
) -> Optional[Signal]:
    """Validates logical consistency between parameters. Returns a HOLD Signal on failure, None on success."""
    # Skip the fixed percentage comparison if both stop and take-profit are ATR-based
    if not (stop_method == "atr_multiple" and tp_atr_valid):
        if sl is not None and tp <= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be greater than stop_loss_pct")
    if trailing:
        if symbol and not BTPPolicy.supports_trailing_stop(symbol):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop is not supported for BTP symbols")
        tsd = params.get("trailing_stop_distance_pct")
        if tsd is not None and sl is not None and tsd >= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop_distance_pct must be less than stop_loss_pct")
    return None


def _validate_signal_impl(
    signal: Signal,
    market_data: Optional[Dict[str, Any]] = None,
    atr: Optional[float] = None,
    price: Optional[float] = None,
    spread_pct: Optional[float] = None,
    timeframe_seconds: Optional[int] = None,
    min_stop_atr_mult: float = 1.0,
    min_hold_time_mult: float = 1.0,
    global_min_risk_reward_ratio: Optional[float] = None,
    symbol: Optional[str] = None,
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

        # Validate backtest_entry_config (only required for BUY signals)
        if signal.action == "BUY":
            bec_error = _validate_backtest_entry_config(params, symbol)
            if bec_error:
                return bec_error

        # Read take-profit value early so default stop-loss consistency checks can use it
        tp = params.get("take_profit_pct")
        sl_error = _validate_stop_loss(params, symbol, atr, price, timeframe_seconds, min_stop_atr_mult, tp)
        if sl_error:
            return sl_error
        
        # Re-read values that may have been updated by _validate_stop_loss
        tp = params.get("take_profit_pct")
        sl = params.get("stop_loss_pct")
        stop_method = params.get("stop_loss_method", "fixed")
        atr_mult = params.get("stop_loss_atr_multiple")

        # take_profit_pct is validated separately below (may use take_profit_atr_multiple instead)
        _apply_required_defaults(params, symbol, timeframe_seconds)
        
        tp_error = _validate_take_profit(params, symbol, atr, price, timeframe_seconds, sl)
        if tp_error:
            return tp_error
        
        # Re-read values that may have been updated by _validate_take_profit
        tp = params.get("take_profit_pct")
        tp_atr = params.get("take_profit_atr_multiple")
        tp_valid = tp is not None and isinstance(tp, (int, float)) and (0 < tp < 10.0)
        tp_atr_valid = tp_atr is not None and isinstance(tp_atr, (int, float)) and tp_atr > 0
        ts_error = _validate_trailing_stop(params, symbol, atr, price, timeframe_seconds, sl)
        if ts_error:
            return ts_error
        
        trailing = params["trailing_stop"]
        
        req_error = _validate_required_params(params, timeframe_seconds, min_hold_time_mult)
        if req_error:
            return req_error

        opt_error = _validate_optional_params(
            params, symbol, sl, tp, stop_method, tp_atr_valid, atr_mult, tp_atr, trailing, global_min_risk_reward_ratio
        )
        if opt_error:
            return opt_error

        cons_error = _validate_logical_consistency(params, symbol, sl, tp, stop_method, tp_atr_valid, trailing)
        if cons_error:
            return cons_error

    return signal
