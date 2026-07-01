"""Signal processing component for the TradingEngine.

Handles per-symbol LLM orchestration, backtesting, validation, and execution.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.config.settings import settings

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
