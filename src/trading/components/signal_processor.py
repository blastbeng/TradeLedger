"""Signal processing component for the TradingEngine.

Handles per-symbol LLM orchestration, backtesting, validation, and execution.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings
from src.database import get_latest_ohlcv_timestamp
from src.strategies.base import Signal

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Handles per-symbol signal processing for the TradingEngine."""

    def __init__(self, engine):
        self.engine = engine

    async def read_position_trigger_flags(
        self, symbol: str, symbol_entry: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Read pre-processing flags and review limits for a symbol.

        Checks max tenure, cooldown after loss, queued orders, and reads
        all position trigger flags (max hold, stop loss, take profit, partial
        TP, dust sweep) and LLM-decided review limits.

        Returns None if the symbol should be skipped (tenure reached,
        cooldown active, or order already queued).
        Otherwise returns a dict with all flags and review limits.
        """
        engine = self.engine
        assigned_tf = symbol_entry["timeframe"]
        tf_seconds = engine._timeframe_to_seconds(assigned_tf)

        # --- Maximum symbol tenure (per-symbol, set by LLM) ---
        max_tenure_hours = symbol_entry.get('max_tenure_hours')
        if max_tenure_hours is not None and max_tenure_hours > 0 and 'entry_time' in symbol_entry:
            tenure_seconds = max_tenure_hours * 3600
            if time.time() - symbol_entry['entry_time'] > tenure_seconds:
                logger.info(f"Max symbol tenure reached for {symbol} ({max_tenure_hours:.1f}h), forcing sell")
                from src.strategies.base import Signal
                signal = Signal(action="SELL", confidence=1.0, reasoning="Max symbol tenure reached")
                await engine._execute_signal(symbol, signal, exit_reason="max_tenure")
                engine._force_eval.pop(symbol, None)
                return None

        # --- Cooldown after a losing trade (LLM-defined) ---
        if symbol not in engine.positions:
            last_loss = engine.last_loss_time.get(symbol)
            if last_loss is not None:
                cooldown = engine.cooldown_durations.get(symbol, 0)
                if cooldown > 0:
                    elapsed = time.time() - last_loss
                    if elapsed < cooldown:
                        remaining = cooldown - elapsed
                        logger.info(
                            f"Skipping {symbol}: cooldown active ({remaining:.0f}s remaining after loss)"
                        )
                        engine._force_eval.pop(symbol, None)
                        return None

        # Skip if there is already a queued order for this symbol
        async with engine._queued_orders_lock:
            has_queued = any(q['symbol'] == symbol for q in engine.queued_orders)
        if has_queued:
            logger.info(f"Skipping {symbol}: order already queued.")
            engine._force_eval.pop(symbol, None)
            return None

        # --- Read position trigger flags ---
        max_hold_expired = False
        max_hold_expired_count = 0
        stop_loss_triggered = False
        stop_loss_review_count = 0
        take_profit_triggered = False
        take_profit_review_count = 0
        partial_tp_triggered = False
        partial_tp_review_count = 0
        partial_tp_triggered_levels = []
        dust_sweep_triggered = False
        dust_sweep_review_count = 0
        if symbol in engine.positions:
            pos = engine.positions[symbol]
            max_hold_expired = pos.get("_max_hold_expired", False)
            max_hold_expired_count = pos.get("_max_hold_expired_count", 1)
            stop_loss_triggered = pos.get("_stop_loss_triggered", False)
            stop_loss_review_count = pos.get("_stop_loss_review_count", 0)
            take_profit_triggered = pos.get("_take_profit_triggered", False)
            take_profit_review_count = pos.get("_take_profit_review_count", 0)
            partial_tp_triggered = pos.get("_partial_tp_triggered", False) or pos.get("_partial_tp_triggered_single", False)
            partial_tp_review_count = pos.get("_partial_tp_review_count", 0) or pos.get("_partial_tp_single_review_count", 0)
            partial_tp_triggered_levels = pos.get("_partial_tp_triggered_levels", [])
            dust_sweep_triggered = pos.get("_dust_sweep_triggered", False)
            dust_sweep_review_count = pos.get("_dust_sweep_review_count", 0)

        # --- Read LLM-decided review limits from Redis ---
        max_sl_reviews_prompt = settings.MAX_STOP_LOSS_REVIEWS
        max_tp_reviews_prompt = settings.MAX_TAKE_PROFIT_REVIEWS
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_stop_loss_reviews")
            if raw:
                max_sl_reviews_prompt = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_take_profit_reviews")
            if raw:
                max_tp_reviews_prompt = int(raw)
        except Exception:
            pass

        max_partial_tp_reviews_prompt = settings.MAX_PARTIAL_TP_REVIEWS
        max_dust_sweep_reviews_prompt = settings.MAX_DUST_SWEEP_REVIEWS
        try:
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_partial_tp_reviews")
            if raw:
                max_partial_tp_reviews_prompt = int(raw)
            raw = await asyncio.to_thread(engine.redis.get, "trading:max_dust_sweep_reviews")
            if raw:
                max_dust_sweep_reviews_prompt = int(raw)
        except Exception:
            pass

        # Scale stop-loss review limit for long-term timeframes
        if tf_seconds >= settings.LONG_TERM_TF_SECONDS:  # >= 1 month
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.LONG_TERM_MAX_STOP_LOSS_REVIEWS)
        elif tf_seconds >= 604_800:  # >= 1 week
            max_sl_reviews_prompt = min(max_sl_reviews_prompt, settings.WEEKLY_MAX_STOP_LOSS_REVIEWS)

        return {
            "max_hold_expired": max_hold_expired,
            "max_hold_expired_count": max_hold_expired_count,
            "stop_loss_triggered": stop_loss_triggered,
            "stop_loss_review_count": stop_loss_review_count,
            "take_profit_triggered": take_profit_triggered,
            "take_profit_review_count": take_profit_review_count,
            "partial_tp_triggered": partial_tp_triggered,
            "partial_tp_review_count": partial_tp_review_count,
            "partial_tp_triggered_levels": partial_tp_triggered_levels,
            "dust_sweep_triggered": dust_sweep_triggered,
            "dust_sweep_review_count": dust_sweep_review_count,
            "max_sl_reviews_prompt": max_sl_reviews_prompt,
            "max_tp_reviews_prompt": max_tp_reviews_prompt,
            "max_partial_tp_reviews_prompt": max_partial_tp_reviews_prompt,
            "max_dust_sweep_reviews_prompt": max_dust_sweep_reviews_prompt,
        }

    async def check_skip_conditions(
        self,
        symbol: str,
        display_symbol: str,
        ticker: Dict[str, Any],
        assigned_tf: str,
        has_position: bool,
        base_balance: float,
    ) -> bool:
        """Check whether a symbol should be skipped before LLM evaluation.

        Returns True if the symbol should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        # --- Staleness guard: skip symbols with stale quotes (unless we have an open position) ---
        if not has_position and await engine._is_quote_too_stale(ticker, assigned_tf):
            logger.info(
                f"Skipping {symbol}: quote data is too stale for timeframe {assigned_tf}."
            )
            stale_notify_key = f"trading:stale_quote_notify:{symbol}"
            should_notify = True
            try:
                last_notify_raw = await asyncio.to_thread(engine.redis.get, stale_notify_key)
                if last_notify_raw:
                    if (time.time() - float(last_notify_raw)) < 3600:
                        should_notify = False
            except Exception:
                pass
            if should_notify and engine.notifier:
                await engine.notifier.send_notification(
                    f"⏸️ Skipping {display_symbol}: quote data is too stale for timeframe {assigned_tf}.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Quote data too stale",
                    }
                )
                try:
                    await asyncio.to_thread(engine.redis.setex, stale_notify_key, 3600, str(time.time()))
                except Exception:
                    pass
            engine._force_eval.pop(symbol, None)
            return True

        # If we have an open position, we must continue evaluating it for SELL signals
        # even when base_balance is 0 (all capital deployed) or effective_max_symbols is 0.
        if not has_position and (base_balance <= 0 or engine.effective_max_symbols == 0):
            logger.warning(
                f"Skipping {symbol}: {engine.base_currency} balance={base_balance:.2f}, "
                f"effective_max_symbols={engine.effective_max_symbols}"
            )
            return True

        return False

    async def check_no_ohlcv(
        self,
        symbol: str,
        display_symbol: str,
        assigned_tf: str,
        ohlcv_data: Dict[str, Any],
    ) -> bool:
        """Check if no OHLCV data is available for the symbol.

        Returns True if the symbol should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine
        no_ohlcv = (
            not ohlcv_data
            or all(len(candles) == 0 for candles in ohlcv_data.values())
        )
        if not no_ohlcv:
            return False

        logger.info(
            f"Skipping {symbol}: no OHLCV data – market data unavailable."
        )
        # Find the most recent OHLCV timestamp across all timeframes
        last_data_ts = None
        last_data_tf = None
        for tf in settings.OHLCV_TIMEFRAMES:
            try:
                ts = await asyncio.to_thread(get_latest_ohlcv_timestamp, symbol, tf)
                if ts is not None and (last_data_ts is None or ts > last_data_ts):
                    last_data_ts = ts
                    last_data_tf = tf
            except Exception:
                pass

        if last_data_ts is not None:
            age_seconds = time.time() - (last_data_ts / 1000.0)
            if age_seconds < 3600:
                age_str = f"{age_seconds/60:.0f} minutes ago"
            elif age_seconds < 86400:
                age_str = f"{age_seconds/3600:.1f} hours ago"
            else:
                age_str = f"{age_seconds/86400:.1f} days ago"
            msg = (
                f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                f"Last data: {last_data_tf} candle from {age_str}. "
                f"Try a manual force-download via the dashboard or Telegram."
            )
        else:
            msg = (
                f"⚠️ Skipping {display_symbol}: no OHLCV data available. "
                f"No historical data found in database. "
                f"Run a force-download via the dashboard or Telegram to populate market data."
            )

        no_ohlcv_notify_key = f"trading:no_ohlcv_notify:{symbol}"
        should_notify = True
        try:
            last_notify_raw = await asyncio.to_thread(engine.redis.get, no_ohlcv_notify_key)
            if last_notify_raw:
                if (time.time() - float(last_notify_raw)) < 3600:
                    should_notify = False
        except Exception:
            pass

        if should_notify and engine.notifier:
            await engine.notifier.send_notification(
                msg,
                summary={
                    "symbol": symbol,
                    "action": "SKIP",
                    "reason": "No OHLCV data",
                    "last_data_timestamp": last_data_ts,
                    "last_data_timeframe": last_data_tf,
                }
            )
            try:
                await asyncio.to_thread(engine.redis.setex, no_ohlcv_notify_key, 3600, str(time.time()))
            except Exception:
                pass

        engine._force_eval.pop(symbol, None)
        return True

    async def handle_triggered_flags(
        self,
        symbol: str,
        display_symbol: str,
        signal: Signal,
        validated: Signal,
        assigned_tf: str,
        current_price: float,
        atr: Optional[float],
        ticker: Dict[str, Any],
        max_hold_expired: bool,
        stop_loss_triggered: bool,
        take_profit_triggered: bool,
        partial_tp_triggered: bool,
        dust_sweep_triggered: bool,
        strategy_model_type: str,
        llm_provider: Optional[str],
        llm_model: Optional[str],
    ) -> bool:
        """Handle triggered position flags (max hold, stop loss, take profit, partial TP, dust sweep).

        Returns True if the caller should return immediately (flag was handled).
        Returns False if the caller should continue with normal execution.
        """
        engine = self.engine
        params = signal.strategy_params or {}

        # --- Handle max‑hold‑expired LLM decision ---
        if max_hold_expired and signal.action == "HOLD":
            new_max_hold = params.get("max_hold_time_seconds") if params else None
            if new_max_hold is not None and new_max_hold > 0:
                logger.info(f"LLM extended max hold time for {symbol} to {new_max_hold}s")
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["max_hold_time_seconds"] = new_max_hold
                        engine.positions[symbol]["timestamp"] = int(time.time() * 1000)
                        engine.positions[symbol].pop("_max_hold_expired", None)
                        engine.positions[symbol].pop("_max_hold_expired_count", None)
                for symbol_entry in engine.current_symbols:
                    if symbol_entry["symbol"] == symbol:
                        symbol_entry["entry_time"] = time.time()
                        break
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ Max hold time for {display_symbol} extended to {new_max_hold}s by LLM.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol,
                            "action": "HOLD",
                            "reason": validated.reasoning,
                            "new_max_hold_seconds": new_max_hold,
                            "model_type": strategy_model_type,
                            "llm_provider": llm_provider,
                            "llm_model": llm_model,
                        }
                    )
                await engine._update_position_params(
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                engine._state_dirty = True
            else:
                logger.warning(
                    f"LLM returned HOLD without new max_hold_time_seconds for {symbol} "
                    f"after max hold expiry – forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⏰ LLM did not extend hold time for {display_symbol} – closing position.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Max hold expired, LLM did not extend",
                            "exit_reason": "max_hold_time_llm_no_extend",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await engine._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Max hold expired, LLM did not extend"),
                    exit_reason="max_hold_time_llm_no_extend"
                )
            return True

        # --- Handle stop-loss-triggered LLM decision ---
        if stop_loss_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_stop_method = new_params.get("stop_loss_method", "fixed")
            new_stop_pct = None
            if new_stop_method == "atr_multiple" and atr is not None and atr > 0:
                atr_mult = new_params.get("stop_loss_atr_multiple")
                if atr_mult is not None:
                    new_stop_pct = (atr_mult * atr) / current_price
            else:
                new_stop_pct = new_params.get("stop_loss_pct")

            if new_stop_pct is not None and new_stop_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after stop-loss trigger, "
                    f"new stop_loss_pct={new_stop_pct:.4%}"
                )
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["stop_loss"] = current_price * (1 - new_stop_pct)
                        engine.positions[symbol].pop("_stop_loss_triggered", None)
                        engine.positions[symbol].pop("_stop_loss_review_count", None)
                    await engine._update_position_params(
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted stop-loss to {new_stop_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_stop_loss_pct": new_stop_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after stop-loss trigger but did not provide "
                    f"a new stop-loss. Forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"⛔ {display_symbol}: LLM did not provide new stop-loss – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Stop-loss triggered, LLM did not provide new stop",
                            "exit_reason": "stop_loss_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await engine._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Stop-loss triggered, LLM did not provide new stop"),
                    exit_reason="stop_loss_llm_no_action"
                )
                return True

        elif stop_loss_triggered and signal.action == "SELL":
            if symbol in engine.positions:
                async with engine._positions_lock:
                    engine.positions[symbol].pop("_stop_loss_triggered", None)
                    engine.positions[symbol].pop("_stop_loss_review_count", None)
            # Continue to normal SELL execution

        # --- Handle take-profit-triggered LLM decision ---
        if take_profit_triggered and signal.action == "HOLD":
            new_params = signal.strategy_params or {}
            new_tp_pct = new_params.get("take_profit_pct")
            if new_tp_pct is not None and new_tp_pct > 0:
                logger.info(
                    f"LLM decided to hold {symbol} after take-profit trigger, "
                    f"new take_profit_pct={new_tp_pct:.4%}"
                )
                if symbol in engine.positions:
                    async with engine._positions_lock:
                        engine.positions[symbol]["take_profit"] = current_price * (1 + new_tp_pct)
                        engine.positions[symbol].pop("_take_profit_triggered", None)
                        engine.positions[symbol].pop("_take_profit_review_count", None)
                    await engine._update_position_params(
                        symbol, new_params, signal.indicator_config, assigned_tf, current_price, atr,
                    )
                    engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted take-profit to {new_tp_pct:.4%} – holding.\n"
                        f"Reasoning: {validated.reasoning}",
                        summary={
                            "symbol": symbol, "action": "HOLD", "reason": validated.reasoning,
                            "new_take_profit_pct": new_tp_pct, "model_type": strategy_model_type,
                            "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                return True
            else:
                logger.warning(
                    f"LLM returned HOLD for {symbol} after take-profit trigger but did not provide "
                    f"a new take-profit. Forcing SELL."
                )
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🎯 {display_symbol}: LLM did not provide new take-profit – selling.",
                        summary={
                            "symbol": symbol, "action": "SELL",
                            "reason": "Take-profit triggered, LLM did not provide new take-profit",
                            "exit_reason": "take_profit_llm_no_action",
                            "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model,
                        }
                    )
                await engine._execute_signal(
                    symbol, Signal(action="SELL", confidence=1.0, reasoning="Take-profit triggered, LLM did not provide new take-profit"),
                    exit_reason="take_profit_llm_no_action"
                )
                return True

        elif take_profit_triggered and signal.action == "SELL":
            if symbol in engine.positions:
                async with engine._positions_lock:
                    engine.positions[symbol].pop("_take_profit_triggered", None)
                    engine.positions[symbol].pop("_take_profit_review_count", None)
            # Continue to normal SELL execution

        # --- Handle partial TP triggered ---
        if partial_tp_triggered and signal.action == "HOLD":
            new_levels = params.get("partial_take_profit_levels") if params else None
            if new_levels is not None:
                async with engine._positions_lock:
                    engine.positions[symbol]["partial_take_profit_levels"] = new_levels
                    engine.positions[symbol].pop("_partial_tp_triggered", None)
                    engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                    engine.positions[symbol].pop("_partial_tp_review_count", None)
                    engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                    engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
                    engine.positions[symbol]["partial_tp_levels_triggered"] = []
                    engine.positions[symbol]["partial_tp_depth_wait_start"] = {}
                logger.info(f"LLM updated partial TP levels for {symbol}")
                await engine._update_position_params(
                    symbol, params, signal.indicator_config, assigned_tf, current_price, atr,
                )
                engine._state_dirty = True
                if engine.notifier:
                    await engine.notifier.send_notification(
                        f"🔄 {display_symbol}: LLM adjusted partial TP levels – holding.",
                        summary={"symbol": symbol, "action": "HOLD", "reason": "Partial TP levels adjusted by LLM", "model_type": strategy_model_type, "llm_provider": llm_provider, "llm_model": llm_model}
                    )
                return True
            else:
                logger.info(f"LLM did not update partial TP levels for {symbol}, executing triggered level(s)")
                if engine.positions[symbol].get("_partial_tp_triggered_single"):
                    await engine._execute_partial_tp_single(symbol, current_price, None, ticker)
                    async with engine._positions_lock:
                        engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                        engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                if engine.positions[symbol].get("_partial_tp_triggered"):
                    for lvl in engine.positions[symbol].get("_partial_tp_triggered_levels", []):
                        await engine._execute_partial_tp_level(symbol, lvl, current_price, None, ticker)
                    async with engine._positions_lock:
                        engine.positions[symbol].pop("_partial_tp_triggered", None)
                        engine.positions[symbol].pop("_partial_tp_review_count", None)
                        engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
                return True

        elif partial_tp_triggered and signal.action == "SELL":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_partial_tp_triggered", None)
                engine.positions[symbol].pop("_partial_tp_triggered_single", None)
                engine.positions[symbol].pop("_partial_tp_review_count", None)
                engine.positions[symbol].pop("_partial_tp_single_review_count", None)
                engine.positions[symbol].pop("_partial_tp_triggered_levels", None)
            # Continue to normal SELL execution

        # --- Handle dust sweep triggered ---
        if dust_sweep_triggered and signal.action == "HOLD":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_dust_sweep_triggered", None)
                if engine.positions[symbol].get("_dust_keep_since") is None:
                    engine.positions[symbol]["_dust_keep_since"] = time.time()
            engine._state_dirty = True
            logger.info(f"LLM decided to hold dust for {symbol}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"🧹 {display_symbol}: LLM decided to keep dust – holding.",
                    summary={"symbol": symbol, "action": "HOLD", "reason": "Dust kept by LLM"}
                )
            return True
        elif dust_sweep_triggered and signal.action == "SELL":
            async with engine._positions_lock:
                engine.positions[symbol].pop("_dust_sweep_triggered", None)
                engine.positions[symbol].pop("_dust_sweep_review_count", None)
            logger.info(f"LLM decided to sell dust for {symbol}")
            await engine._sweep_dust(symbol)
            return True

        return False

    async def check_trade_filters(
        self,
        symbol: str,
        display_symbol: str,
        validated: Signal,
        params: Dict[str, Any],
    ) -> bool:
        """Check trade filters (confidence thresholds, SELL without position).

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        # --- Global confidence rejection threshold (set during stock selection) ---
        if validated.action == "BUY":
            conf_rejection_raw = await asyncio.to_thread(engine.redis.get, "trading:confidence_rejection_threshold")
            if conf_rejection_raw:
                try:
                    conf_threshold = float(conf_rejection_raw)
                    if conf_threshold > 0 and validated.confidence < conf_threshold:
                        logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below global rejection threshold {conf_threshold:.2f}")
                        if engine.notifier:
                            await engine.notifier.send_notification(
                                f"⚠️ Skipping {display_symbol}: confidence {validated.confidence:.2f} below threshold {conf_threshold:.2f}",
                                summary={
                                    "symbol": symbol,
                                    "action": "SKIP",
                                    "reason": "Confidence below rejection threshold",
                                    "confidence": validated.confidence,
                                    "threshold": conf_threshold,
                                }
                            )
                        return True
                except (ValueError, TypeError):
                    pass

        min_conf = params.get("min_confidence")
        if min_conf is not None and validated.confidence < min_conf:
            logger.info(f"Skipping {symbol}: confidence {validated.confidence:.2f} below LLM min {min_conf:.2f}")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping {display_symbol}: confidence too low ({validated.confidence:.2f})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Confidence too low",
                        "confidence": validated.confidence,
                        "min_confidence": min_conf,
                    }
                )
            return True

        # Prevent SELL without an open position (no shorting)
        if validated.action == "SELL" and symbol not in engine.positions:
            logger.info(f"Skipping SELL for {symbol}: no open position.")
            if engine.notifier:
                await engine.notifier.send_notification(
                    f"⚠️ Skipping SELL for {display_symbol}: no open position.",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "No open position",
                    }
                )
            return True

        return False

    async def check_sector_concentration(
        self,
        symbol: str,
        display_symbol: str,
        assigned_tf: str,
    ) -> bool:
        """Check if buying this symbol would exceed the sector concentration limit.

        Returns True if the trade should be skipped (caller should return),
        False if processing should continue.
        """
        engine = self.engine

        current_sector = None
        for entry in engine.current_symbols:
            if entry["symbol"] == symbol:
                current_sector = entry.get("sector")
                break

        if not current_sector:
            return False

        max_positions_per_sector_raw = await asyncio.to_thread(engine.redis.get, "trading:max_positions_per_sector")
        if max_positions_per_sector_raw:
            try:
                max_positions_per_sector = int(max_positions_per_sector_raw)
            except ValueError:
                max_positions_per_sector = None
        else:
            max_positions_per_sector = None

        if max_positions_per_sector is None or max_positions_per_sector <= 0:
            return False

        sector_count = 0
        for pos_sym in engine.positions.keys():
            for entry in engine.current_symbols:
                if entry["symbol"] == pos_sym and entry.get("sector") == current_sector:
                    sector_count += 1
                    break

        if sector_count >= max_positions_per_sector:
            logger.info(
                f"Skipping BUY {symbol}: sector '{current_sector}' already has "
                f"{sector_count} open positions (max {max_positions_per_sector})"
            )
            if engine.notifier:
                stock_name = await engine._get_stock_name(symbol)
                display = engine._format_symbol_display(symbol, stock_name, assigned_tf)
                await engine.notifier.send_notification(
                    f"⚠️ Skipping BUY {display}: sector '{current_sector}' concentration limit reached ({sector_count}/{max_positions_per_sector})",
                    summary={
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Sector concentration limit",
                        "sector": current_sector,
                        "sector_count": sector_count,
                        "max_positions_per_sector": max_positions_per_sector,
                    }
                )
            return True

        return False
