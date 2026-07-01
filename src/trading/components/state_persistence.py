"""State persistence component for the TradingEngine.

Handles saving and loading trading engine state to/from the database.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import logging
from dataclasses import asdict
from typing import Any

from src.database import save_trading_state

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles persistence of trading engine state to the database."""

    def __init__(self, engine):
        self.engine = engine

    async def save_state(self, force: bool = False):
        """Persist current symbols, positions, and trade history to SQLite.

        Uses a lock to serialize concurrent calls and a debounce flag to
        coalesce multiple save requests into fewer DB write batches.

        When *force* is True, the method waits for the lock instead of
        debouncing, guaranteeing the state is flushed even if another
        save is in progress.
        """
        engine = self.engine
        if engine._state_lock.locked():
            if not force:
                engine._state_save_pending = True
                return

        async with engine._state_lock:
            await self._save_state_impl()
            while engine._state_save_pending:
                engine._state_save_pending = False
                await self._save_state_impl()

    async def _save_state_impl(self):
        """Actual state persistence (must be called under _state_lock)."""
        engine = self.engine
        await asyncio.to_thread(save_trading_state, "current_symbols", engine.current_symbols)
        async with engine._positions_lock:
            positions_snapshot = dict(engine.positions)
        await asyncio.to_thread(save_trading_state, "positions", positions_snapshot)
        await asyncio.to_thread(save_trading_state, "queued_orders", engine.queued_orders)
        await asyncio.to_thread(save_trading_state, "recent_signals", engine.recent_signals)
        # Serialize pending entries (convert Signal objects to dicts for JSON storage)
        pending_entries_serializable = {}
        for symbol, entry in engine._pending_entries.items():
            pending_entries_serializable[symbol] = {
                "signal": asdict(entry["signal"]),
                "deadline": entry["deadline"],
                "timeframe": entry["timeframe"],
                "condition": entry["condition"],
            }
        await asyncio.to_thread(save_trading_state, "pending_entries", pending_entries_serializable)
        await asyncio.to_thread(save_trading_state, "symbol_first_seen", engine._symbol_first_seen)
        await asyncio.to_thread(save_trading_state, "entry_signal_state", engine._entry_signal_state)
        await asyncio.to_thread(save_trading_state, "last_eval_snapshot", engine._last_eval_snapshot)
        await asyncio.to_thread(save_trading_state, "force_eval", engine._force_eval)
        await asyncio.to_thread(save_trading_state, "force_eval_time", engine._force_eval_time)
        await asyncio.to_thread(save_trading_state, "strategy_intervals", engine._strategy_intervals)
        await asyncio.to_thread(save_trading_state, "last_decisions", engine._last_decisions)
        await asyncio.to_thread(save_trading_state, "last_loss_time", engine.last_loss_time)
        await asyncio.to_thread(save_trading_state, "cooldown_durations", engine.cooldown_durations)
        await asyncio.to_thread(save_trading_state, "global_risk_multiplier", engine._global_risk_multiplier)
        logger.debug("Saved trading state: %d symbols, %d positions, %d trades",
                     len(engine.current_symbols), len(engine.positions), len(engine.trade_history))
        engine._state_dirty = False
