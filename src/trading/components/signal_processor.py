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
