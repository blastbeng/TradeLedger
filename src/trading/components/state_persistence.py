"""State persistence component for the TradingEngine.

Handles saving and loading trading engine state to/from the database.
Extracted from TradingEngine to reduce class size and improve maintainability.
"""
import asyncio
import atexit
import dataclasses as _dc
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src.config.settings import settings
from src.database import save_trading_state, load_trading_state, get_all_trades, get_latest_close_prices
from src.strategies.base import Signal

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles persistence of trading engine state to the database."""

    def __init__(self, engine, event_bus):
        self.engine = engine
        self.shared_state = engine.shared_state
        self.event_bus = event_bus
        self._persistence_lock = threading.Lock()
        self._periodic_save_task: Optional[asyncio.Task] = None
        self.event_bus.subscribe("save_state", self.save_state)
        self.event_bus.subscribe("get_pause_status", self.get_pause_status)

        # Register handlers to flush state on shutdown/crash
        atexit.register(self._sync_save_state)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            # signal.signal can only be called in the main thread
            logger.warning("Could not register signal handlers (not in main thread). Relying on atexit only.")

    def start_periodic_save(self, interval: int = 60) -> None:
        """Starts a background task that saves state periodically."""
        if self._periodic_save_task is None or self._periodic_save_task.done():
            self._periodic_save_task = asyncio.create_task(self._periodic_save_loop(interval))

    async def _periodic_save_loop(self, interval: int) -> None:
        """Periodically saves state to mitigate data loss on hard crashes."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.save_state(force=True)
            except Exception as e:
                logger.error(f"Periodic state save failed: {type(e).__name__}: {e}", exc_info=True)

    def stop_periodic_save(self) -> None:
        """Cancels the periodic state save background task."""
        if self._periodic_save_task and not self._periodic_save_task.done():
            self._periodic_save_task.cancel()

    def _handle_signal(self, signum, frame):
        """Handle termination signals by flushing state before exiting."""
        logger.info(f"Received signal {signum}, flushing state to database...")
        self._sync_save_state()
        # Use os._exit to bypass atexit handlers (since we already saved)
        # and ensure immediate termination.
        os._exit(0)

    def _sync_save_state(self):
        """Synchronously save state to prevent data loss on crash/shutdown.

        Uses a threading lock to avoid concurrent writes with the async save path.
        If the async save is in progress, waits up to 2 seconds for it to complete.
        """
        engine = self.engine
        if not self._persistence_lock.acquire(blocking=True, timeout=2.0):
            logger.info("Persistence lock held by async save; state is already being persisted.")
            return
        try:
            save_trading_state("current_symbols", self.shared_state.current_symbols)
            save_trading_state("positions", dict(self.shared_state.positions))
            save_trading_state("queued_orders", self.shared_state.queued_orders)
            save_trading_state("recent_signals", self.shared_state.recent_signals)

            pending_entries_serializable = {}
            for symbol, entry in self.shared_state._pending_entries.items():
                pending_entries_serializable[symbol] = {
                    "signal": asdict(entry["signal"]),
                    "deadline": entry["deadline"],
                    "timeframe": entry["timeframe"],
                    "condition": entry["condition"],
                }
            save_trading_state("pending_entries", pending_entries_serializable)
            save_trading_state("symbol_first_seen", self.shared_state._symbol_first_seen)
            save_trading_state("entry_signal_state", self.shared_state._entry_signal_state)
            save_trading_state("last_eval_snapshot", self.shared_state._last_eval_snapshot)
            save_trading_state("force_eval", self.shared_state._force_eval)
            save_trading_state("force_eval_time", self.shared_state._force_eval_time)
            save_trading_state("strategy_intervals", self.shared_state._strategy_intervals)
            save_trading_state("last_decisions", self.shared_state._last_decisions)
            save_trading_state("last_loss_time", self.shared_state.last_loss_time)
            save_trading_state("cooldown_durations", self.shared_state.cooldown_durations)
            save_trading_state("global_risk_multiplier", self.shared_state._global_risk_multiplier)
        except Exception as e:
            logger.critical(f"Failed to save state on exit: {e}", exc_info=True)
        finally:
            self._persistence_lock.release()

    async def save_state(self, force: bool = False):
        """Persist current symbols, positions, and trade history to SQLite.

        Uses a lock to serialize concurrent calls. The debounce flag has been
        removed to ensure state changes are persisted immediately, preventing
        data loss if the process crashes between setting the pending flag and
        the actual save.
        """
        engine = self.engine
        async with self.shared_state._state_lock:
            await self._save_state_impl()

        if force:
            from src.web.app import invalidate_ws_payload_cache
            invalidate_ws_payload_cache()

    async def _save_state_impl(self):
        """Actual state persistence (must be called under _state_lock)."""
        engine = self.engine
        # Acquire threading lock to prevent concurrent writes from _sync_save_state
        await asyncio.to_thread(self._persistence_lock.acquire)
        try:
            await asyncio.to_thread(save_trading_state, "current_symbols", self.shared_state.current_symbols)
            async with self.shared_state._positions_lock:
                positions_snapshot = dict(self.shared_state.positions)
            await asyncio.to_thread(save_trading_state, "positions", positions_snapshot)
            await asyncio.to_thread(save_trading_state, "queued_orders", self.shared_state.queued_orders)
            await asyncio.to_thread(save_trading_state, "recent_signals", self.shared_state.recent_signals)
            # Serialize pending entries (convert Signal objects to dicts for JSON storage)
            async with self.shared_state._pending_entries_lock:
                pending_entries_serializable = {}
                for symbol, entry in self.shared_state._pending_entries.items():
                    pending_entries_serializable[symbol] = {
                        "signal": asdict(entry["signal"]),
                        "deadline": entry["deadline"],
                        "timeframe": entry["timeframe"],
                        "condition": entry["condition"],
                    }
            await asyncio.to_thread(save_trading_state, "pending_entries", pending_entries_serializable)
            await asyncio.to_thread(save_trading_state, "symbol_first_seen", self.shared_state._symbol_first_seen)
            await asyncio.to_thread(save_trading_state, "entry_signal_state", self.shared_state._entry_signal_state)
            await asyncio.to_thread(save_trading_state, "last_eval_snapshot", self.shared_state._last_eval_snapshot)
            await asyncio.to_thread(save_trading_state, "force_eval", self.shared_state._force_eval)
            await asyncio.to_thread(save_trading_state, "force_eval_time", self.shared_state._force_eval_time)
            await asyncio.to_thread(save_trading_state, "strategy_intervals", self.shared_state._strategy_intervals)
            await asyncio.to_thread(save_trading_state, "last_decisions", self.shared_state._last_decisions)
            await asyncio.to_thread(save_trading_state, "last_loss_time", self.shared_state.last_loss_time)
            await asyncio.to_thread(save_trading_state, "cooldown_durations", self.shared_state.cooldown_durations)
            await asyncio.to_thread(save_trading_state, "global_risk_multiplier", self.shared_state._global_risk_multiplier)
        except Exception as e:
            logger.critical(f"Failed to save trading state: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save trading state: {e}")
        finally:
            self._persistence_lock.release()

        # Store open positions count in Redis for _should_use_primary_model() check
        try:
            engine.redis.set("trading:open_positions_count", str(len(self.shared_state.positions)))
        except Exception:
            pass

        logger.debug("Saved trading state: %d symbols, %d positions, %d trades",
                     len(self.shared_state.current_symbols), len(self.shared_state.positions), len(self.shared_state.trade_history))
        self.shared_state._state_dirty = False

    def load_state(self):
        """Load current symbols, positions, trade history, and initial balance from SQLite."""
        engine = self.engine
        state = load_trading_state()

        raw_symbols = state.get("current_symbols", [])
        # Convert old format (list of strings) to new format if needed
        if raw_symbols and isinstance(raw_symbols[0], str):
            default_tf = settings.OHLCV_TIMEFRAMES[0] if settings.OHLCV_TIMEFRAMES else "1h"
            self.shared_state.current_symbols = [{"symbol": s, "timeframe": default_tf} for s in raw_symbols]
        else:
            self.shared_state.current_symbols = raw_symbols
        self.shared_state.positions = state.get("positions", {})
        # Remove any position that lacks LLM-defined risk parameters.
        # Such positions cannot be managed safely.
        for symbol in list(self.shared_state.positions.keys()):
            pos = self.shared_state.positions[symbol]
            if "stop_loss" not in pos or "take_profit" not in pos:
                logger.warning(
                    f"Position for {symbol} is missing stop_loss/take_profit. "
                    f"Will attempt to re-evaluate to obtain LLM risk parameters before force-closing."
                )
                pos["_needs_risk_params"] = True
                pos["_needs_risk_params_attempts"] = 0
                # Force immediate re-evaluation so the LLM can provide risk parameters
                self.shared_state._force_eval[symbol] = True
                self.shared_state._last_strategy_eval.pop(symbol, None)

        # Discard positions with zero amount or zero price (corrupted state)
        for symbol in list(self.shared_state.positions.keys()):
            pos = self.shared_state.positions[symbol]
            amount = pos.get("amount", 0)
            price = pos.get("price", 0)
            if amount <= 0 or price <= 0:
                logger.warning(
                    f"Position for {symbol} has invalid amount={amount} or price={price}. Removing it."
                )
                del self.shared_state.positions[symbol]

        # Initialize trailing stop tracking fields for positions with trailing stops.
        # Fetch the latest close prices to ensure _highest_price is at least the current
        # market price, preventing a too-loose trailing stop immediately after restart.
        trailing_symbols = [sym for sym, pos in self.shared_state.positions.items() if pos.get("trailing_stop")]
        latest_prices = {}
        if trailing_symbols:
            try:
                latest_prices = get_latest_close_prices(trailing_symbols)
            except Exception as e:
                logger.warning(f"Failed to fetch latest close prices for trailing stops: {e}")

        for symbol, pos in self.shared_state.positions.items():
            if pos.get("trailing_stop"):
                if "_highest_price" not in pos:
                    entry_price = pos.get("price", 0.0)
                    latest_close = latest_prices.get(symbol, {}).get("close", 0.0)
                    pos["_highest_price"] = max(entry_price, latest_close)
                if "_last_trailing_check_ts" not in pos:
                    pos["_last_trailing_check_ts"] = time.time()

        all_trades = get_all_trades()
        self.shared_state.trade_history = all_trades[-settings.MAX_TRADES_IN_MEMORY:]
        # Compute the realized P&L offset for trades that were pruned at load time
        self.shared_state._realized_pnl_offset = sum(
            t.get("realized_pnl", 0.0)
            for t in all_trades[:-settings.MAX_TRADES_IN_MEMORY]
            if t.get("side") == "sell"
        )
        self.shared_state.queued_orders = state.get("queued_orders", [])
        for q in self.shared_state.queued_orders:
            q['order_book'] = None
        self.shared_state.recent_signals = state.get("recent_signals", [])
        self.shared_state._symbol_first_seen = state.get("symbol_first_seen", {})
        self.shared_state._entry_signal_state = state.get("entry_signal_state", {})
        self.shared_state._last_eval_snapshot = state.get("last_eval_snapshot", {})
        self.shared_state._force_eval = state.get("force_eval", {})
        self.shared_state._force_eval_time = state.get("force_eval_time", {})
        self.shared_state._strategy_intervals = state.get("strategy_intervals", {})
        self.shared_state._last_decisions = state.get("last_decisions", {})
        self.shared_state.last_loss_time = state.get("last_loss_time", {})
        self.shared_state.cooldown_durations = state.get("cooldown_durations", {})
        self.shared_state._global_risk_multiplier = state.get("global_risk_multiplier")

        # Restore pending entries (reconstruct Signal objects from dicts)
        raw_pending = state.get("pending_entries", {})
        self.shared_state._pending_entries = {}
        valid_signal_keys = {f.name for f in _dc.fields(Signal)}
        for symbol, entry in raw_pending.items():
            try:
                signal_dict = entry["signal"]
                filtered = {k: v for k, v in signal_dict.items() if k in valid_signal_keys}
                if "action" not in filtered:
                    filtered["action"] = "HOLD"
                if "confidence" not in filtered:
                    filtered["confidence"] = 0.0
                if "reasoning" not in filtered:
                    filtered["reasoning"] = ""
                signal = Signal(**filtered)
                self.shared_state._pending_entries[symbol] = {
                    "signal": signal,
                    "deadline": entry["deadline"],
                    "timeframe": entry["timeframe"],
                    "condition": entry["condition"],
                }
            except Exception as e:
                logger.warning(f"Failed to restore pending entry for {symbol}: {type(e).__name__}: {e}")

        # Prune any pending entries whose deadline has already passed
        now = time.time()
        expired = [sym for sym, e in self.shared_state._pending_entries.items() if now >= e["deadline"]]
        for sym in expired:
            logger.info(f"Discarding expired pending entry for {sym} (deadline passed during downtime).")
            del self.shared_state._pending_entries[sym]

        if "initial_balance" in state:
            engine.initial_balance = float(state["initial_balance"])
        else:
            balance = engine.trader.fetch_balance()
            engine.initial_balance = balance.get(engine.base_currency, 0.0)
            save_trading_state("initial_balance", engine.initial_balance)

        logger.info(
            "Loaded trading state: %d symbols, %d positions, %d trades",
            len(self.shared_state.current_symbols),
            len(self.shared_state.positions),
            len(self.shared_state.trade_history),
        )

    async def get_pause_status(self) -> Dict[str, Any]:
        """Return the current trading pause status, reason, remaining duration, and a formatted countdown."""
        engine = self.engine
        paused_raw = await asyncio.to_thread(engine.redis.get, "trading:paused")
        is_paused = paused_raw is not None and paused_raw == "1"

        reason_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_reason")
        reason = reason_raw.decode() if isinstance(reason_raw, bytes) else (reason_raw or "")

        source_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_source")
        source = source_raw.decode() if isinstance(source_raw, bytes) else (source_raw or "")

        remaining_seconds = None
        countdown_str = None

        if is_paused:
            market_time_str = None
            if source == "market_closed":
                # Fetch the current clock to compute a live countdown and current market time
                clock = await engine._market_data_manager.get_clock()

                market_time_str = None
                if clock is not None:
                    market_time_str = clock.timestamp.astimezone(ZoneInfo(settings.MARKET_TIMEZONE)).strftime('%H:%M %d/%m/%Y')
                    if not clock.is_open:
                        now_utc = datetime.now(timezone.utc)
                        next_open = clock.next_open
                        remaining = (next_open - now_utc).total_seconds()
                        if remaining > 0:
                            remaining_seconds = int(remaining)
                            if remaining_seconds > 3600:
                                hours = remaining_seconds // 3600
                                minutes = (remaining_seconds % 3600) // 60
                                countdown_str = f"{hours}h {minutes}m"
                            elif remaining_seconds > 60:
                                minutes = remaining_seconds // 60
                                seconds = remaining_seconds % 60
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{remaining_seconds}s"
                else:
                    # Fallback to the stored next_open if the clock is unavailable
                    next_open_raw = await asyncio.to_thread(engine.redis.get, "trading:market_next_open")
                    if next_open_raw:
                        try:
                            next_open_str = next_open_raw.decode() if isinstance(next_open_raw, bytes) else next_open_raw
                            next_open_dt = datetime.fromisoformat(next_open_str)
                            now_utc = datetime.now(timezone.utc)
                            remaining = (next_open_dt - now_utc).total_seconds()
                            if remaining > 0:
                                remaining_seconds = int(remaining)
                                if remaining_seconds > 3600:
                                    hours = remaining_seconds // 3600
                                    minutes = (remaining_seconds % 3600) // 60
                                    countdown_str = f"{hours}h {minutes}m"
                                elif remaining_seconds > 60:
                                    minutes = remaining_seconds // 60
                                    seconds = remaining_seconds % 60
                                    countdown_str = f"{minutes}m {seconds}s"
                                else:
                                    countdown_str = f"{remaining_seconds}s"
                                reason = "Market closed"
                        except Exception as e:
                            logger.debug(f"get_pause_status: failed to parse next_open: {type(e).__name__}: {e}")
            else:
                # LLM or manual pause with duration
                pause_start_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_start")
                pause_duration_raw = await asyncio.to_thread(engine.redis.get, "trading:pause_duration")
                if pause_start_raw and pause_duration_raw:
                    try:
                        pause_start = float(pause_start_raw)
                        pause_duration = int(pause_duration_raw)
                        elapsed = time.time() - pause_start
                        remaining = pause_duration - elapsed
                        if remaining > 0:
                            remaining_seconds = int(remaining)
                            if remaining_seconds > 3600:
                                hours = remaining_seconds // 3600
                                minutes = (remaining_seconds % 3600) // 60
                                countdown_str = f"{hours}h {minutes}m"
                            elif remaining_seconds > 60:
                                minutes = remaining_seconds // 60
                                seconds = remaining_seconds % 60
                                countdown_str = f"{minutes}m {seconds}s"
                            else:
                                countdown_str = f"{remaining_seconds}s"
                    except (ValueError, TypeError):
                        pass

        return {
            "is_paused": is_paused,
            "reason": reason,
            "remaining_seconds": remaining_seconds,
            "countdown_str": countdown_str,
            "source": source,
            "market_time_str": market_time_str if is_paused and source == "market_closed" else None,
        }
